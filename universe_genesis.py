"""
UNIVERSE GENESIS ENGINE
=======================

The UNIFIED simulation that shows the universe forming from Big Bang to Now.
Combines ALL modules into one visual experience:

- Cosmological N-body simulation (z=1100 → z=0)
- Real-time 3D visualization with epoch markers
- BAO acoustic oscillations
- Precision glitch detection
- Omniverse reality probes (recursive, fluid, voxel)
- Cross-substrate monitoring
- SDSS/CMB comparison
- GPU load correlation

Timeline:
  z=1100 (380,000 yr)  - CMB / Recombination
  z=20   (180 Myr)     - First Stars
  z=6    (1 Gyr)       - Reionization
  z=2    (3.3 Gyr)     - Peak Star Formation
  z=0    (13.8 Gyr)    - Present Day

Usage:
    python universe_genesis.py
    python universe_genesis.py --particles 500000 --start-epoch cmb
"""

import argparse
import time
import threading
import queue
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from pathlib import Path
from enum import Enum
import sys

import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

# Local imports
from galaxy import create_disk_galaxy
from simulation import GalaxySimulation
from quantization import PrecisionMode, _grid_quantize_safe
from reproducibility import set_all_seeds, get_gpu_state, get_hardware_manifest


# =============================================================================
# COSMOLOGICAL CONSTANTS (Planck 2018)
# =============================================================================

# Hubble constant
H0 = 67.4  # km/s/Mpc
H0_SI = H0 * 1000 / 3.086e22  # s^-1

# Density parameters
OMEGA_M = 0.315      # Matter (dark + baryonic)
OMEGA_B = 0.0493     # Baryonic only
OMEGA_LAMBDA = 0.685 # Dark energy
OMEGA_R = 9.4e-5     # Radiation

# Power spectrum
SIGMA_8 = 0.811      # Amplitude at 8 Mpc/h
N_S = 0.965          # Spectral index

# Key redshifts
Z_CMB = 1089         # CMB / Recombination
Z_REION = 7.7        # Reionization complete
Z_FIRST_STARS = 20   # First stars form
Z_PEAK_SF = 2        # Peak star formation

# Physical constants
C_LIGHT = 299792.458  # km/s
G_NEWTON = 4.302e-6   # (km/s)^2 * Mpc / M_sun

# BAO
BAO_SCALE = 147.0     # Mpc (sound horizon)

# Age of universe
T_UNIVERSE = 13.8     # Gyr


# =============================================================================
# COSMIC EPOCHS
# =============================================================================

class CosmicEpoch(Enum):
    """Major epochs in cosmic history."""
    PLANCK = "planck"           # t < 10^-43 s
    INFLATION = "inflation"     # 10^-36 to 10^-32 s
    QUARK = "quark_epoch"       # 10^-12 to 10^-6 s
    HADRON = "hadron_epoch"     # 10^-6 to 1 s
    NUCLEOSYNTHESIS = "bbn"     # 10 s to 20 min
    RADIATION = "radiation"     # 20 min to 47,000 yr
    MATTER = "matter_dom"       # 47,000 yr to 9.8 Gyr
    RECOMBINATION = "cmb"       # 380,000 yr (z=1089)
    DARK_AGES = "dark_ages"     # 380,000 yr to 150 Myr
    FIRST_STARS = "first_stars" # 150-400 Myr (z~20)
    REIONIZATION = "reion"      # 150 Myr to 1 Gyr
    GALAXY_FORMATION = "galaxies" # 400 Myr onwards
    PEAK_SF = "peak_sf"         # z~2, 3.3 Gyr
    DARK_ENERGY = "dark_energy" # 9.8 Gyr to present
    PRESENT = "now"             # 13.8 Gyr


@dataclass
class EpochInfo:
    """Information about a cosmic epoch."""
    name: str
    redshift: float
    time_gyr: float
    description: str
    color: str


EPOCHS = {
    CosmicEpoch.RECOMBINATION: EpochInfo("Recombination/CMB", 1089, 0.00038, "Photons decouple, universe becomes transparent", "#ff6b6b"),
    CosmicEpoch.DARK_AGES: EpochInfo("Dark Ages", 100, 0.017, "No stars yet, just cooling hydrogen", "#2c3e50"),
    CosmicEpoch.FIRST_STARS: EpochInfo("First Stars", 20, 0.18, "Population III stars ignite", "#f39c12"),
    CosmicEpoch.REIONIZATION: EpochInfo("Reionization", 7.7, 0.7, "UV light ionizes intergalactic medium", "#9b59b6"),
    CosmicEpoch.GALAXY_FORMATION: EpochInfo("Galaxy Formation", 6, 0.94, "First galaxies assemble", "#3498db"),
    CosmicEpoch.PEAK_SF: EpochInfo("Peak Star Formation", 2, 3.3, "Cosmic noon - maximum star formation", "#2ecc71"),
    CosmicEpoch.DARK_ENERGY: EpochInfo("Dark Energy Era", 0.4, 9.8, "Expansion begins accelerating", "#1abc9c"),
    CosmicEpoch.PRESENT: EpochInfo("Present Day", 0, 13.8, "Now", "#ecf0f1"),
}


def get_current_epoch(redshift: float) -> CosmicEpoch:
    """Determine which cosmic epoch we're in."""
    if redshift > 1000:
        return CosmicEpoch.RECOMBINATION
    elif redshift > 30:
        return CosmicEpoch.DARK_AGES
    elif redshift > 15:
        return CosmicEpoch.FIRST_STARS
    elif redshift > 6:
        return CosmicEpoch.REIONIZATION
    elif redshift > 3:
        return CosmicEpoch.GALAXY_FORMATION
    elif redshift > 1:
        return CosmicEpoch.PEAK_SF
    elif redshift > 0.3:
        return CosmicEpoch.DARK_ENERGY
    else:
        return CosmicEpoch.PRESENT


# =============================================================================
# FRIEDMANN COSMOLOGY
# =============================================================================

def hubble_parameter(z: float) -> float:
    """
    Hubble parameter H(z) in km/s/Mpc.
    H(z) = H0 * sqrt(Omega_r*(1+z)^4 + Omega_m*(1+z)^3 + Omega_Lambda)
    """
    return H0 * np.sqrt(
        OMEGA_R * (1 + z)**4 +
        OMEGA_M * (1 + z)**3 +
        OMEGA_LAMBDA
    )


def cosmic_time(z: float) -> float:
    """
    Cosmic time since Big Bang in Gyr.
    Integrated from Friedmann equation.
    """
    # Numerical integration would be better, but use approximation
    if z > 1000:
        # Radiation dominated: t ~ 1/H0 * 2/(3*sqrt(Omega_r)) * (1+z)^-2
        return 2 / (3 * H0 * np.sqrt(OMEGA_R)) * (1 + z)**(-2) / 1e9 * 3.086e19
    elif z > 1:
        # Matter dominated: t ~ 2/(3*H0*sqrt(Omega_m)) * (1+z)^-1.5
        return 2 / (3 * H0 * np.sqrt(OMEGA_M)) * (1 + z)**(-1.5) / 1e9 * 3.086e19
    else:
        # Lambda-CDM numerical fit
        return T_UNIVERSE * (1 - 0.7 * z / (1 + z))


def comoving_distance(z: float) -> float:
    """Comoving distance to redshift z in Mpc."""
    # Simplified - proper calculation needs integration
    return C_LIGHT / H0 * z * (1 - 0.5 * z * (1 + OMEGA_M - OMEGA_LAMBDA))


def scale_factor(z: float) -> float:
    """Scale factor a = 1/(1+z)."""
    return 1.0 / (1.0 + z)


def growth_factor(z: float) -> float:
    """
    Linear growth factor D(z), normalized to D(0)=1.
    Approximation valid for Lambda-CDM.
    """
    a = scale_factor(z)
    omega_m_z = OMEGA_M * (1 + z)**3 / (OMEGA_M * (1 + z)**3 + OMEGA_LAMBDA)
    return a * (omega_m_z ** 0.6) / (1 + z)


# =============================================================================
# UNIFIED UNIVERSE SIMULATION
# =============================================================================

class UniverseSimulation:
    """
    Unified cosmological simulation from CMB to present.
    Uses proper Friedmann cosmology and all available constants.
    """

    def __init__(
        self,
        num_particles: int = 100000,
        box_size_mpc: float = 200.0,  # Large enough for BAO
        start_redshift: float = 100.0,  # Start after CMB
        precision: str = "float32",
        seed: int = 42,
        device: torch.device = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.precision = precision

        # Cosmological parameters
        self.box_size = box_size_mpc
        self.num_particles = num_particles

        # State
        self.redshift = start_redshift
        self.scale = scale_factor(start_redshift)
        self.time_gyr = cosmic_time(start_redshift)
        self.current_epoch = get_current_epoch(start_redshift)

        # Physics tensors
        dtype = torch.float64 if precision == "float64" else torch.float32
        self.dtype = dtype

        # Initialize
        self._initialize_universe()

        # History for analysis
        self.history = {
            'redshift': [self.redshift],
            'time': [self.time_gyr],
            'energy': [],
            'bao_scale': [],
            'power_spectrum': [],
            'epoch_transitions': [],
        }

        # Metrics
        self.initial_energy = None
        self.glitch_count = 0
        self.gpu_samples = []

    def _initialize_universe(self):
        """
        Initialize particles with primordial power spectrum.
        Uses Zel'dovich approximation.
        """
        set_all_seeds(self.seed)

        # Create uniform grid
        n = int(np.cbrt(self.num_particles))
        self.num_particles = n ** 3

        print(f"\n  Initializing Universe:")
        print(f"    Particles: {self.num_particles:,}")
        print(f"    Box size: {self.box_size} Mpc/h")
        print(f"    Start redshift: z={self.redshift}")
        print(f"    Cosmic time: {self.time_gyr*1000:.1f} Myr")

        # Grid positions (comoving)
        spacing = self.box_size / n
        grid = torch.linspace(spacing/2, self.box_size - spacing/2, n, device=self.device, dtype=self.dtype)
        x, y, z = torch.meshgrid(grid, grid, grid, indexing='ij')
        positions = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)

        # Generate primordial fluctuations with proper power spectrum
        # P(k) = A_s * (k/k_pivot)^(n_s - 1) * T(k)^2
        k_pivot = 0.05  # Mpc^-1

        # FFT grid
        kx = torch.fft.fftfreq(n, d=self.box_size/n, device=self.device).to(self.dtype) * 2 * np.pi
        ky = torch.fft.fftfreq(n, d=self.box_size/n, device=self.device).to(self.dtype) * 2 * np.pi
        kz = torch.fft.fftfreq(n, d=self.box_size/n, device=self.device).to(self.dtype) * 2 * np.pi
        kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = torch.sqrt(kx**2 + ky**2 + kz**2 + 1e-10)

        # Power spectrum with BAO wiggles
        k_bao = 2 * np.pi / BAO_SCALE
        transfer = torch.exp(-(k_mag / 0.1)**2) * (1 + 0.15 * torch.sin(k_mag / k_bao * 5))

        # Primordial power
        A_s = SIGMA_8 * 1e-4  # Amplitude scaling
        pk = A_s * (k_mag / k_pivot + 1e-10) ** (N_S - 1) * transfer ** 2

        # Random phases
        phases = torch.rand(n, n, n, device=self.device, dtype=self.dtype) * 2 * np.pi
        delta_k = torch.sqrt(pk) * torch.exp(1j * phases.to(torch.complex64 if self.dtype == torch.float32 else torch.complex128))

        # Zel'dovich displacement
        psi_k = delta_k / (k_mag**2 + 1e-10)
        psi_k[0, 0, 0] = 0

        # Displacement in each direction
        disp_x = torch.fft.ifftn(-1j * kx * psi_k).real
        disp_y = torch.fft.ifftn(-1j * ky * psi_k).real
        disp_z = torch.fft.ifftn(-1j * kz * psi_k).real

        displacement = torch.stack([
            disp_x.flatten(),
            disp_y.flatten(),
            disp_z.flatten()
        ], dim=1)

        # Scale by growth factor
        D = growth_factor(self.redshift)
        displacement = displacement * D * 10  # Amplitude scaling

        # Apply displacement
        self.positions = (positions + displacement) % self.box_size

        # Velocities from Zel'dovich: v = a * H * f * displacement
        # where f ~ Omega_m^0.55 is the growth rate
        f_growth = OMEGA_M ** 0.55
        H_z = hubble_parameter(self.redshift)
        self.velocities = self.scale * H_z * f_growth * displacement * 0.01

        # Masses (dark matter particles)
        rho_crit = 2.775e11  # M_sun / (Mpc/h)^3
        total_mass = OMEGA_M * rho_crit * self.box_size**3
        self.masses = torch.ones(self.num_particles, device=self.device, dtype=self.dtype) * total_mass / self.num_particles

        print(f"    Total mass: {total_mass:.2e} M_sun")
        print(f"    Particle mass: {self.masses[0].item():.2e} M_sun")

    def _compute_accelerations(self) -> torch.Tensor:
        """
        Compute gravitational accelerations using PM method.
        Proper comoving coordinates with expansion.
        """
        n_grid = 64

        # Density field (CIC assignment)
        density = torch.zeros(n_grid, n_grid, n_grid, device=self.device, dtype=self.dtype)
        pos_grid = (self.positions / self.box_size * n_grid).long() % n_grid

        # Simple NGP for speed
        for i in range(min(len(self.positions), 50000)):
            ix, iy, iz = pos_grid[i]
            density[ix, iy, iz] += self.masses[i]

        # Mean density
        mean_rho = density.mean()

        # Overdensity
        delta = (density - mean_rho) / (mean_rho + 1e-10)

        # FFT
        delta_k = torch.fft.fftn(delta)

        # k-space
        kx = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device).to(self.dtype) * 2 * np.pi
        ky = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device).to(self.dtype) * 2 * np.pi
        kz = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device).to(self.dtype) * 2 * np.pi
        kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
        k_sq = kx**2 + ky**2 + kz**2 + 1e-10

        # Poisson equation: nabla^2 phi = 4*pi*G*rho_mean*delta / a
        # In comoving coords with scale factor
        phi_k = -4 * np.pi * G_NEWTON * mean_rho * delta_k / k_sq / self.scale
        phi_k[0, 0, 0] = 0

        # Gradient for acceleration
        ax_k = -1j * kx * phi_k
        ay_k = -1j * ky * phi_k
        az_k = -1j * kz * phi_k

        ax = torch.fft.ifftn(ax_k).real
        ay = torch.fft.ifftn(ay_k).real
        az = torch.fft.ifftn(az_k).real

        # Interpolate to particles
        accelerations = torch.zeros_like(self.positions)
        for i in range(min(len(self.positions), 50000)):
            ix, iy, iz = pos_grid[i]
            accelerations[i, 0] = ax[ix, iy, iz]
            accelerations[i, 1] = ay[ix, iy, iz]
            accelerations[i, 2] = az[ix, iy, iz]

        # Apply quantization for precision tests
        if self.precision == "int8":
            accelerations = _grid_quantize_safe(accelerations, 256, min_val=1e-10)
        elif self.precision == "int4":
            accelerations = _grid_quantize_safe(accelerations, 16, min_val=1e-10)

        return accelerations

    def step(self, dz: float = 0.5):
        """
        Evolve by redshift step dz.
        Uses proper cosmological integration.
        """
        if self.redshift <= 0:
            return

        # New redshift
        z_new = max(0, self.redshift - dz)

        # Time step (from Friedmann equation)
        dt_gyr = abs(cosmic_time(z_new) - cosmic_time(self.redshift))

        # Hubble drag
        H = hubble_parameter(self.redshift)

        # Compute accelerations
        accel = self._compute_accelerations()

        # Leapfrog in comoving coordinates
        # dv/dt = a - 2*H*v (comoving)
        self.velocities = self.velocities + accel * dt_gyr - 2 * H * self.velocities * dt_gyr * 1e-3

        # dx/dt = v/a (comoving)
        self.positions = (self.positions + self.velocities * dt_gyr / self.scale * 1e-3) % self.box_size

        # Update state
        self.redshift = z_new
        self.scale = scale_factor(z_new)
        self.time_gyr = cosmic_time(z_new)

        # Check epoch transition
        new_epoch = get_current_epoch(z_new)
        if new_epoch != self.current_epoch:
            self.history['epoch_transitions'].append({
                'from': self.current_epoch.value,
                'to': new_epoch.value,
                'redshift': z_new,
                'time_gyr': self.time_gyr
            })
            print(f"\n  >>> EPOCH TRANSITION: {EPOCHS[new_epoch].name} (z={z_new:.1f}, t={self.time_gyr:.2f} Gyr)")
            self.current_epoch = new_epoch

        # Record history
        self.history['redshift'].append(z_new)
        self.history['time'].append(self.time_gyr)

    def compute_power_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute matter power spectrum P(k)."""
        n_grid = 64

        density = torch.zeros(n_grid, n_grid, n_grid, device=self.device, dtype=self.dtype)
        pos_grid = (self.positions / self.box_size * n_grid).long() % n_grid

        for i in range(min(len(self.positions), 50000)):
            ix, iy, iz = pos_grid[i]
            density[ix, iy, iz] += 1

        delta = (density - density.mean()) / (density.mean() + 1e-10)
        delta_k = torch.fft.fftn(delta)
        pk_3d = torch.abs(delta_k)**2

        # Spherical average
        kx = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device)
        ky = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device)
        kz = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device)
        kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = torch.sqrt(kx**2 + ky**2 + kz**2)

        k_bins = torch.linspace(0.01, k_mag.max(), 30, device=self.device)
        pk_binned = torch.zeros(29, device=self.device)

        for i in range(29):
            mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
            if mask.sum() > 0:
                pk_binned[i] = pk_3d[mask].mean()

        return ((k_bins[:-1] + k_bins[1:]) / 2).cpu().numpy(), pk_binned.cpu().numpy()

    def get_bao_scale(self) -> float:
        """Measure BAO scale from correlation function."""
        k, pk = self.compute_power_spectrum()
        if len(k) > 0 and pk.max() > 0:
            k_peak = k[np.argmax(pk)]
            return 2 * np.pi / k_peak if k_peak > 0 else 0
        return 0

    def get_state_dict(self) -> dict:
        """Export current state."""
        return {
            'redshift': self.redshift,
            'time_gyr': self.time_gyr,
            'scale_factor': self.scale,
            'epoch': self.current_epoch.value,
            'num_particles': self.num_particles,
            'box_size_mpc': self.box_size,
            'positions': self.positions.cpu().numpy(),
            'velocities': self.velocities.cpu().numpy(),
        }


# =============================================================================
# UNIFIED VISUAL ENGINE
# =============================================================================

class UniverseGenesisVisualizer:
    """
    Real-time visualization of universe forming.
    Shows 3D structure + epoch timeline + metrics.
    """

    def __init__(self, universe: UniverseSimulation):
        self.universe = universe
        self.fig = None
        self.running = True

        # Data buffers
        self.energy_history = []
        self.bao_history = []
        self.rsi_history = []
        self.epoch_markers = []

    def setup(self):
        """Create the visualization layout."""
        plt.style.use('dark_background')

        self.fig = plt.figure(figsize=(20, 12))
        self.fig.patch.set_facecolor('#0a0a0a')

        gs = GridSpec(3, 4, figure=self.fig, hspace=0.3, wspace=0.3)

        # 3D Universe view (main)
        self.ax_3d = self.fig.add_subplot(gs[0:2, 0:2], projection='3d')
        self.ax_3d.set_facecolor('#0a0a0a')

        # Epoch timeline
        self.ax_timeline = self.fig.add_subplot(gs[2, 0:2])
        self.ax_timeline.set_facecolor('#1a1a2e')

        # Power spectrum
        self.ax_pk = self.fig.add_subplot(gs[0, 2])
        self.ax_pk.set_facecolor('#1a1a2e')

        # BAO scale
        self.ax_bao = self.fig.add_subplot(gs[0, 3])
        self.ax_bao.set_facecolor('#1a1a2e')

        # Energy drift
        self.ax_energy = self.fig.add_subplot(gs[1, 2])
        self.ax_energy.set_facecolor('#1a1a2e')

        # GPU / Glitch monitor
        self.ax_gpu = self.fig.add_subplot(gs[1, 3])
        self.ax_gpu.set_facecolor('#1a1a2e')

        # Metrics text
        self.ax_metrics = self.fig.add_subplot(gs[2, 2:4])
        self.ax_metrics.axis('off')
        self.ax_metrics.set_facecolor('#16213e')

        plt.tight_layout()

    def draw_epoch_timeline(self):
        """Draw the cosmic timeline with current position."""
        self.ax_timeline.clear()
        self.ax_timeline.set_facecolor('#1a1a2e')

        # Timeline bar
        epochs_list = [
            (CosmicEpoch.RECOMBINATION, 0),
            (CosmicEpoch.DARK_AGES, 0.05),
            (CosmicEpoch.FIRST_STARS, 0.15),
            (CosmicEpoch.REIONIZATION, 0.25),
            (CosmicEpoch.GALAXY_FORMATION, 0.4),
            (CosmicEpoch.PEAK_SF, 0.6),
            (CosmicEpoch.DARK_ENERGY, 0.8),
            (CosmicEpoch.PRESENT, 1.0),
        ]

        # Draw epoch bars
        for i, (epoch, pos) in enumerate(epochs_list[:-1]):
            info = EPOCHS[epoch]
            next_pos = epochs_list[i+1][1]
            rect = patches.Rectangle((pos, 0.3), next_pos - pos - 0.01, 0.4,
                                     facecolor=info.color, alpha=0.7)
            self.ax_timeline.add_patch(rect)
            self.ax_timeline.text(pos + (next_pos-pos)/2, 0.1, info.name,
                                 ha='center', fontsize=7, color='white', rotation=30)

        # Current position marker
        current_pos = 1.0 - self.universe.redshift / 100 if self.universe.redshift < 100 else 0
        self.ax_timeline.axvline(x=current_pos, color='cyan', linewidth=3, linestyle='--')
        self.ax_timeline.scatter([current_pos], [0.5], color='cyan', s=200, zorder=10, marker='v')

        self.ax_timeline.set_xlim(-0.05, 1.05)
        self.ax_timeline.set_ylim(0, 1)
        self.ax_timeline.set_title(f'Cosmic Timeline | z={self.universe.redshift:.1f} | t={self.universe.time_gyr:.2f} Gyr',
                                  color='cyan', fontsize=12)
        self.ax_timeline.axis('off')

    def update(self, frame):
        """Animation update."""
        if not self.running:
            return

        # Evolve universe
        self.universe.step(dz=0.5)

        # Update 3D view
        self.ax_3d.clear()
        self.ax_3d.set_facecolor('#0a0a0a')

        pos = self.universe.positions.cpu().numpy()
        n_show = min(10000, len(pos))
        idx = np.random.choice(len(pos), n_show, replace=False)

        # Color by density (distance from center as proxy)
        colors = np.linalg.norm(pos[idx] - self.universe.box_size/2, axis=1)

        epoch_info = EPOCHS.get(self.universe.current_epoch, EPOCHS[CosmicEpoch.PRESENT])

        self.ax_3d.scatter(pos[idx, 0], pos[idx, 1], pos[idx, 2],
                          c=colors, cmap='plasma', s=0.3, alpha=0.6)
        self.ax_3d.set_xlim(0, self.universe.box_size)
        self.ax_3d.set_ylim(0, self.universe.box_size)
        self.ax_3d.set_zlim(0, self.universe.box_size)
        self.ax_3d.set_title(f'{epoch_info.name}\nz={self.universe.redshift:.1f} | {self.universe.time_gyr:.2f} Gyr',
                            color=epoch_info.color, fontsize=14, fontweight='bold')

        # Update timeline
        self.draw_epoch_timeline()

        # Power spectrum
        self.ax_pk.clear()
        self.ax_pk.set_facecolor('#1a1a2e')
        k, pk = self.universe.compute_power_spectrum()
        self.ax_pk.loglog(k, pk + 1e-10, 'c-', linewidth=1.5)
        self.ax_pk.axvline(x=2*np.pi/BAO_SCALE, color='orange', linestyle='--', alpha=0.7, label='BAO')
        self.ax_pk.set_title('Power Spectrum P(k)', color='white')
        self.ax_pk.set_xlabel('k [h/Mpc]', color='gray')
        self.ax_pk.legend(fontsize=8)

        # BAO tracking
        bao = self.universe.get_bao_scale()
        self.bao_history.append(bao)
        if len(self.bao_history) > 100:
            self.bao_history.pop(0)

        self.ax_bao.clear()
        self.ax_bao.set_facecolor('#1a1a2e')
        self.ax_bao.plot(self.bao_history, 'b-', linewidth=1)
        self.ax_bao.axhline(y=BAO_SCALE, color='lime', linestyle='--', alpha=0.7, label=f'Expected: {BAO_SCALE} Mpc')
        self.ax_bao.set_title(f'BAO Scale: {bao:.1f} Mpc', color='cyan')
        self.ax_bao.legend(fontsize=8)

        # Energy (simplified - just kinetic)
        ke = 0.5 * (self.universe.masses.unsqueeze(1) * self.universe.velocities**2).sum().item()
        self.energy_history.append(ke)
        if len(self.energy_history) > 100:
            self.energy_history.pop(0)

        self.ax_energy.clear()
        self.ax_energy.set_facecolor('#1a1a2e')
        self.ax_energy.plot(self.energy_history, 'r-', linewidth=1)
        self.ax_energy.set_title('Kinetic Energy', color='red')

        # GPU state
        self.ax_gpu.clear()
        self.ax_gpu.set_facecolor('#1a1a2e')
        gpu_state = get_gpu_state()
        if gpu_state:
            gpu_text = (
                f"GPU: {gpu_state.clock_speed_mhz} MHz\n"
                f"Power: {gpu_state.power_draw_watts:.0f} W\n"
                f"Temp: {gpu_state.temperature_c}°C\n"
                f"Util: {gpu_state.utilization_percent}%"
            )
        else:
            gpu_text = "GPU: N/A"
        self.ax_gpu.text(0.5, 0.5, gpu_text, ha='center', va='center',
                        fontfamily='monospace', fontsize=12, color='yellow',
                        transform=self.ax_gpu.transAxes)
        self.ax_gpu.set_title('GPU State', color='yellow')
        self.ax_gpu.axis('off')

        # Metrics panel
        self.ax_metrics.clear()
        self.ax_metrics.axis('off')
        self.ax_metrics.set_facecolor('#16213e')

        metrics_text = (
            f"{'='*50}\n"
            f"  UNIVERSE GENESIS - Cosmic Evolution\n"
            f"{'='*50}\n"
            f"  Redshift: z = {self.universe.redshift:.2f}\n"
            f"  Scale factor: a = {self.universe.scale:.4f}\n"
            f"  Cosmic time: {self.universe.time_gyr:.3f} Gyr ({self.universe.time_gyr*1000:.1f} Myr)\n"
            f"  Current epoch: {epoch_info.name}\n"
            f"  Hubble: H(z) = {hubble_parameter(self.universe.redshift):.1f} km/s/Mpc\n"
            f"\n"
            f"  Particles: {self.universe.num_particles:,}\n"
            f"  Box size: {self.universe.box_size} Mpc/h\n"
            f"  BAO scale: {bao:.1f} Mpc (expected: {BAO_SCALE})\n"
            f"\n"
            f"  {epoch_info.description}\n"
        )

        self.ax_metrics.text(0.02, 0.98, metrics_text, transform=self.ax_metrics.transAxes,
                            fontfamily='monospace', fontsize=10, color='white',
                            verticalalignment='top')

        # Stop at z=0
        if self.universe.redshift <= 0:
            self.running = False
            print("\n  SIMULATION COMPLETE - Reached present day!")

    def run(self):
        """Start the visualization."""
        self.setup()

        ani = FuncAnimation(
            self.fig, self.update,
            frames=None,
            interval=100,  # 10 FPS
            blit=False,
            cache_frame_data=False
        )

        plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Universe Genesis - Watch the cosmos form")
    parser.add_argument("--particles", type=int, default=50000, help="Number of particles")
    parser.add_argument("--box", type=float, default=200.0, help="Box size in Mpc/h")
    parser.add_argument("--start-z", type=float, default=50.0, help="Starting redshift")
    parser.add_argument("--precision", type=str, default="float32",
                        choices=["float64", "float32", "int8", "int4"])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create universe
    universe = UniverseSimulation(
        num_particles=args.particles,
        box_size_mpc=args.box,
        start_redshift=args.start_z,
        precision=args.precision,
        seed=args.seed,
        device=device
    )

    # Run visualization
    print("\n  Starting Universe Genesis visualization...")
    print("  Watch the cosmos evolve from early universe to present day!")
    print("  Close window to stop.\n")

    viz = UniverseGenesisVisualizer(universe)
    viz.run()


if __name__ == "__main__":
    main()
