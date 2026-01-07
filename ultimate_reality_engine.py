"""
ULTIMATE REALITY ENGINE - The Final Test
=========================================

The comprehensive simulation hypothesis test that combines EVERYTHING:
- All precision tests (int4 to float64)
- All reality glitch detectors
- Cosmological simulation (Big Bang to Now)
- Baryonic Acoustic Oscillations
- Cross-Substrate Mirror Test
- Real data comparison (CMB, SDSS patterns)

This is the FINAL BOSS of simulation hypothesis testing.

Run Requirements:
- NVIDIA GPU with 8GB+ VRAM (RTX 5090 recommended)
- 32GB+ system RAM
- ~2 hours for full suite

Usage:
    python ultimate_reality_engine.py --mode full
    python ultimate_reality_engine.py --mode bigbang --particles 10000000
    python ultimate_reality_engine.py --mode bao --duration 1000
    python ultimate_reality_engine.py --mode substrate --export-state
    python ultimate_reality_engine.py --mode compare --other-platform results/mac_state.json
"""

import argparse
import time
import json
import hashlib
import struct
import zlib
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from enum import Enum
import threading
from collections import deque

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Local imports
from galaxy import create_disk_galaxy
from simulation import GalaxySimulation
from quantization import PrecisionMode, _grid_quantize_safe
from reproducibility import (
    set_all_seeds, get_gpu_state, get_hardware_manifest,
    get_software_manifest, print_methodology_notes,
    ExperimentConfig, create_manifest, print_manifest
)
from gpu_profiler import GPUProfiler, measure_instrumentation_overhead

# Try to import other test modules (may not always be available)
try:
    from omniverse_tests import (
        RecursivePhysicsMirror, FluidDynamicsChaos,
        NeuralHardwareBridge, VoxelSpaceTimeGrid, OmniverseReport
    )
    OMNIVERSE_AVAILABLE = True
except ImportError:
    OMNIVERSE_AVAILABLE = False
    print("Note: omniverse_tests module not available")

try:
    from sensitivity_test import run_sensitivity_sweep, SensitivityResult
    SENSITIVITY_AVAILABLE = True
except ImportError:
    SENSITIVITY_AVAILABLE = False
    print("Note: sensitivity_test module not available")

try:
    from orbital_audit import run_full_orbital_audit, OrbitalAuditReport
    ORBITAL_AVAILABLE = True
except ImportError:
    ORBITAL_AVAILABLE = False
    print("Note: orbital_audit module not available")


# =============================================================================
# PHYSICAL CONSTANTS - Cosmological
# =============================================================================

# Planck units
PLANCK_TIME = 5.391247e-44  # seconds
PLANCK_LENGTH = 1.616255e-35  # meters
PLANCK_MASS = 2.176434e-8  # kg
PLANCK_ENERGY = 1.9561e9  # joules

# Cosmological parameters (Planck 2018)
H0 = 67.4  # km/s/Mpc - Hubble constant
OMEGA_M = 0.315  # Matter density
OMEGA_LAMBDA = 0.685  # Dark energy density
OMEGA_B = 0.0493  # Baryonic matter
SIGMA_8 = 0.811  # Power spectrum amplitude
N_S = 0.965  # Spectral index

# CMB parameters
T_CMB = 2.7255  # K - CMB temperature
Z_RECOMBINATION = 1089  # Redshift at recombination
Z_REIONIZATION = 7.7  # Redshift at reionization

# BAO scale
BAO_SCALE_MPC = 147  # Sound horizon in Mpc

# Speed of light
C_LIGHT = 299792458  # m/s


# =============================================================================
# COSMOLOGICAL SIMULATION ENGINE
# =============================================================================

class CosmologicalPrecision(Enum):
    """Precision modes for cosmological simulation."""
    FLOAT64 = "float64"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class CosmologicalState:
    """State of the cosmological simulation."""
    redshift: float
    scale_factor: float  # a = 1/(1+z)
    time_gyr: float  # Time in Gyr since Big Bang
    positions: np.ndarray  # (N, 3) comoving Mpc
    velocities: np.ndarray  # (N, 3) km/s
    masses: np.ndarray  # Solar masses
    num_particles: int
    precision: str
    seed: int
    state_hash: str  # For cross-platform comparison


@dataclass
class PowerSpectrum:
    """Matter power spectrum."""
    k: np.ndarray  # Wavenumber (h/Mpc)
    pk: np.ndarray  # Power P(k)
    k_peak: float  # BAO peak location
    bao_amplitude: float  # BAO oscillation amplitude


@dataclass
class FilamentStructure:
    """Detected cosmic web structure."""
    num_filaments: int
    num_voids: int
    void_positions: List[Tuple[float, float, float]]
    void_radii: List[float]
    filament_density: float
    great_void_match: bool  # Does it match real Great Void?


class CosmologicalSimulation:
    """
    N-body cosmological simulation from Big Bang to Now.

    Implements:
    - Lambda-CDM expansion
    - Primordial power spectrum initialization
    - Particle-Mesh gravity solver
    - Baryonic Acoustic Oscillation tracking
    """

    def __init__(
        self,
        num_particles: int = 1000000,
        box_size_mpc: float = 100.0,
        precision: CosmologicalPrecision = CosmologicalPrecision.FLOAT32,
        seed: int = 42,
        device: torch.device = None
    ):
        self.num_particles = num_particles
        self.box_size = box_size_mpc  # Mpc/h
        self.precision = precision
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set dtype based on precision
        self.dtype_map = {
            CosmologicalPrecision.FLOAT64: torch.float64,
            CosmologicalPrecision.FLOAT32: torch.float32,
            CosmologicalPrecision.FLOAT16: torch.float16,
            CosmologicalPrecision.INT8: torch.float32,  # Quantize later
            CosmologicalPrecision.INT4: torch.float32,
        }
        self.dtype = self.dtype_map[precision]

        # State
        self.positions = None
        self.velocities = None
        self.masses = None
        self.redshift = 100.0  # Start at z=100
        self.scale_factor = 1.0 / (1.0 + self.redshift)
        self.time_gyr = 0.0

        # History
        self.power_spectrum_history = []
        self.bao_peak_history = []
        self.energy_history = []

        # Initialize
        self._initialize_particles()

    def _initialize_particles(self):
        """
        Initialize particles with primordial power spectrum.
        Uses Zel'dovich approximation for initial displacements.
        """
        set_all_seeds(self.seed)

        # Create uniform grid
        n_per_dim = int(np.cbrt(self.num_particles))
        self.num_particles = n_per_dim ** 3  # Adjust to perfect cube

        # Grid positions
        grid = torch.linspace(0, self.box_size, n_per_dim, device=self.device, dtype=self.dtype)
        x, y, z = torch.meshgrid(grid, grid, grid, indexing='ij')
        positions = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)

        # Generate primordial fluctuations (simplified power spectrum)
        # P(k) ~ k^n_s for primordial, modified by transfer function
        k_nyq = np.pi * n_per_dim / self.box_size

        # Random phases
        phases = torch.rand(n_per_dim, n_per_dim, n_per_dim, device=self.device, dtype=self.dtype) * 2 * np.pi

        # Amplitude from power spectrum
        kx = torch.fft.fftfreq(n_per_dim, d=self.box_size/n_per_dim, device=self.device).to(self.dtype)
        ky = torch.fft.fftfreq(n_per_dim, d=self.box_size/n_per_dim, device=self.device).to(self.dtype)
        kz = torch.fft.fftfreq(n_per_dim, d=self.box_size/n_per_dim, device=self.device).to(self.dtype)
        kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = torch.sqrt(kx**2 + ky**2 + kz**2) + 1e-10

        # Primordial power spectrum with BAO imprint
        # P(k) = A * k^ns * T(k)^2, where T(k) includes BAO wiggles
        k_bao = 2 * np.pi / BAO_SCALE_MPC  # BAO scale in k-space
        bao_wiggles = 1 + 0.1 * torch.sin(k_mag / k_bao * 10)  # Simplified BAO
        pk = SIGMA_8 * (k_mag * self.box_size / (2*np.pi)) ** N_S * bao_wiggles
        pk[0, 0, 0] = 0  # Zero DC component

        # Generate displacement field (Zel'dovich approximation)
        amplitude = torch.sqrt(pk) * torch.exp(1j * phases.to(torch.complex64 if self.dtype == torch.float32 else torch.complex128))

        # Displacement in each direction
        displacement_k = amplitude.unsqueeze(-1) * torch.stack([kx, ky, kz], dim=-1) / (k_mag.unsqueeze(-1) + 1e-10)
        displacement = torch.fft.ifftn(displacement_k, dim=(0,1,2)).real

        # Apply displacements (scaled by growth factor at z=100)
        growth_factor = self.scale_factor  # Simplified
        displacement_flat = displacement.reshape(-1, 3) * growth_factor * 0.01  # Scale factor

        self.positions = (positions + displacement_flat) % self.box_size  # Periodic BC

        # Initial velocities from Zel'dovich
        hubble = self._hubble(self.redshift)
        self.velocities = displacement_flat * hubble * self.scale_factor  # v = a * H * displacement

        # Masses (equal mass particles)
        total_mass = OMEGA_M * 2.775e11 * self.box_size**3  # Solar masses in box
        self.masses = torch.ones(self.num_particles, device=self.device, dtype=self.dtype) * total_mass / self.num_particles

        print(f"  Initialized {self.num_particles:,} particles")
        print(f"  Box size: {self.box_size} Mpc/h")
        print(f"  Initial redshift: z={self.redshift}")

    def _hubble(self, z: float) -> float:
        """Hubble parameter at redshift z."""
        a = 1.0 / (1.0 + z)
        return H0 * np.sqrt(OMEGA_M * a**(-3) + OMEGA_LAMBDA)

    def _compute_accelerations_pm(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute gravitational accelerations using Particle-Mesh method.
        Efficient for cosmological simulations with periodic boundaries.
        """
        n_grid = 128  # PM grid resolution

        # Assign particles to grid (Cloud-In-Cell)
        grid_density = torch.zeros(n_grid, n_grid, n_grid, device=self.device, dtype=self.dtype)

        # Particle positions in grid units
        pos_grid = positions / self.box_size * n_grid
        pos_grid = pos_grid % n_grid  # Periodic

        # CIC assignment (simplified - just nearest grid point for speed)
        idx = pos_grid.long() % n_grid

        # This is inefficient but works for demonstration
        for i in range(len(positions)):
            ix, iy, iz = idx[i]
            grid_density[ix, iy, iz] += self.masses[i]

        # Solve Poisson equation in Fourier space
        # nabla^2 phi = 4*pi*G*rho
        density_k = torch.fft.fftn(grid_density)

        kx = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device).to(self.dtype)
        ky = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device).to(self.dtype)
        kz = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device).to(self.dtype)
        kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
        k_sq = kx**2 + ky**2 + kz**2 + 1e-10

        # Green's function
        G = 4.302e-6  # G in (km/s)^2 * Mpc / M_sun
        phi_k = -G * density_k / k_sq
        phi_k[0, 0, 0] = 0

        # Gradient to get acceleration
        ax_k = -1j * kx * phi_k
        ay_k = -1j * ky * phi_k
        az_k = -1j * kz * phi_k

        ax = torch.fft.ifftn(ax_k).real
        ay = torch.fft.ifftn(ay_k).real
        az = torch.fft.ifftn(az_k).real

        # Interpolate acceleration to particle positions
        accelerations = torch.zeros_like(positions)
        for i in range(len(positions)):
            ix, iy, iz = idx[i]
            accelerations[i, 0] = ax[ix, iy, iz]
            accelerations[i, 1] = ay[ix, iy, iz]
            accelerations[i, 2] = az[ix, iy, iz]

        # Apply quantization if needed
        if self.precision == CosmologicalPrecision.INT8:
            accelerations = _grid_quantize_safe(accelerations, 256, min_val=1e-10)
        elif self.precision == CosmologicalPrecision.INT4:
            accelerations = _grid_quantize_safe(accelerations, 16, min_val=1e-10)

        return accelerations

    def step(self, dt_myr: float = 10.0):
        """
        Advance simulation by dt_myr million years.
        Uses leapfrog integration with comoving coordinates.
        """
        dt_gyr = dt_myr / 1000.0

        # Hubble parameter
        H = self._hubble(self.redshift)

        # Compute accelerations
        accel = self._compute_accelerations_pm(self.positions)

        # Leapfrog integration in comoving coordinates
        # v_new = v + a*dt - H*v*dt (Hubble drag)
        self.velocities = self.velocities + accel * dt_gyr - H * self.velocities * dt_gyr * 0.001

        # x_new = x + v*dt/a (comoving)
        self.positions = (self.positions + self.velocities * dt_gyr / self.scale_factor * 0.001) % self.box_size

        # Update time and redshift
        self.time_gyr += dt_gyr
        # Simplified redshift evolution (proper calculation would integrate Friedmann eq)
        self.redshift = max(0, self.redshift - dt_gyr * H * 0.1)
        self.scale_factor = 1.0 / (1.0 + self.redshift)

    def evolve_to_redshift(self, z_target: float, dt_myr: float = 50.0,
                           callback=None, callback_interval: int = 10):
        """
        Evolve simulation from current redshift to z_target.
        """
        print(f"\n  Evolving from z={self.redshift:.1f} to z={z_target:.1f}...")

        step_count = 0
        while self.redshift > z_target:
            self.step(dt_myr)
            step_count += 1

            if callback and step_count % callback_interval == 0:
                callback(self, step_count)

            if step_count % 100 == 0:
                print(f"    z={self.redshift:.2f}, t={self.time_gyr:.2f} Gyr, step {step_count}")

        print(f"  Reached z={self.redshift:.2f} after {step_count} steps")

    def compute_power_spectrum(self, n_bins: int = 50) -> PowerSpectrum:
        """
        Compute matter power spectrum P(k).
        """
        n_grid = 64

        # Density field
        density = torch.zeros(n_grid, n_grid, n_grid, device=self.device, dtype=self.dtype)
        pos_grid = (self.positions / self.box_size * n_grid).long() % n_grid

        for i in range(len(self.positions)):
            ix, iy, iz = pos_grid[i]
            density[ix, iy, iz] += 1

        # Overdensity
        mean_density = density.mean()
        delta = (density - mean_density) / mean_density

        # FFT
        delta_k = torch.fft.fftn(delta)
        pk_3d = torch.abs(delta_k) ** 2

        # Spherical average
        kx = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device)
        ky = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device)
        kz = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device)
        kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = torch.sqrt(kx**2 + ky**2 + kz**2)

        k_bins = torch.linspace(0.01, k_mag.max(), n_bins, device=self.device)
        pk_binned = torch.zeros(n_bins - 1, device=self.device)

        for i in range(n_bins - 1):
            mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
            if mask.sum() > 0:
                pk_binned[i] = pk_3d[mask].mean()

        k_centers = ((k_bins[:-1] + k_bins[1:]) / 2).cpu().numpy()
        pk_values = pk_binned.cpu().numpy()

        # Find BAO peak
        k_bao = 2 * np.pi / BAO_SCALE_MPC
        bao_idx = np.argmin(np.abs(k_centers - k_bao))
        bao_amplitude = pk_values[bao_idx] / np.mean(pk_values) if np.mean(pk_values) > 0 else 0

        return PowerSpectrum(
            k=k_centers,
            pk=pk_values,
            k_peak=k_centers[np.argmax(pk_values)] if len(pk_values) > 0 else 0,
            bao_amplitude=bao_amplitude
        )

    def detect_structures(self) -> FilamentStructure:
        """
        Detect cosmic web structures (filaments, voids).
        """
        # Simple void detection based on density
        n_grid = 32
        density = torch.zeros(n_grid, n_grid, n_grid, device=self.device)
        pos_grid = (self.positions / self.box_size * n_grid).long() % n_grid

        for i in range(len(self.positions)):
            ix, iy, iz = pos_grid[i]
            density[ix, iy, iz] += 1

        mean_density = density.mean()

        # Voids: regions with density < 0.2 * mean
        void_threshold = 0.2 * mean_density
        void_mask = density < void_threshold

        # Find void centers
        void_positions = []
        void_radii = []

        for i in range(n_grid):
            for j in range(n_grid):
                for k in range(n_grid):
                    if void_mask[i, j, k]:
                        pos = (
                            (i + 0.5) * self.box_size / n_grid,
                            (j + 0.5) * self.box_size / n_grid,
                            (k + 0.5) * self.box_size / n_grid
                        )
                        void_positions.append(pos)
                        void_radii.append(self.box_size / n_grid / 2)

        # Filaments: regions with intermediate density
        filament_threshold_low = 0.5 * mean_density
        filament_threshold_high = 2.0 * mean_density
        filament_mask = (density > filament_threshold_low) & (density < filament_threshold_high)
        num_filaments = filament_mask.sum().item()

        # Check for Great Void match (centered around specific coordinates)
        # Real Great Void is at RA~240, Dec~15 - we use box center as proxy
        box_center = self.box_size / 2
        great_void_region = (
            (box_center - 10, box_center + 10),
            (box_center - 10, box_center + 10),
            (box_center - 10, box_center + 10)
        )

        great_void_match = False
        for vp in void_positions:
            if (great_void_region[0][0] < vp[0] < great_void_region[0][1] and
                great_void_region[1][0] < vp[1] < great_void_region[1][1] and
                great_void_region[2][0] < vp[2] < great_void_region[2][1]):
                great_void_match = True
                break

        return FilamentStructure(
            num_filaments=num_filaments,
            num_voids=len(void_positions),
            void_positions=void_positions[:100],  # Top 100
            void_radii=void_radii[:100],
            filament_density=num_filaments / (n_grid ** 3),
            great_void_match=great_void_match
        )

    def get_state(self) -> CosmologicalState:
        """Get current simulation state for cross-platform comparison."""
        pos_bytes = self.positions.cpu().numpy().tobytes()
        state_hash = hashlib.sha256(pos_bytes).hexdigest()

        return CosmologicalState(
            redshift=self.redshift,
            scale_factor=self.scale_factor,
            time_gyr=self.time_gyr,
            positions=self.positions.cpu().numpy(),
            velocities=self.velocities.cpu().numpy(),
            masses=self.masses.cpu().numpy(),
            num_particles=self.num_particles,
            precision=self.precision.value,
            seed=self.seed,
            state_hash=state_hash
        )


# =============================================================================
# BARYONIC ACOUSTIC OSCILLATIONS TEST
# =============================================================================

@dataclass
class BAOResult:
    """Results from BAO analysis."""
    measured_scale_mpc: float
    expected_scale_mpc: float = BAO_SCALE_MPC
    scale_error_percent: float = 0.0
    frequency_hz: float = 0.0  # In simulation "ticks"
    gpu_clock_mhz: int = 0
    planck_frequency_ratio: float = 0.0
    clock_correlation: float = 0.0
    interpretation: str = ""


def run_bao_test(
    sim: CosmologicalSimulation,
    gpu_profiler: GPUProfiler = None
) -> BAOResult:
    """
    Analyze Baryonic Acoustic Oscillations.

    The "Matrix Proof": If BAO frequency is throttled by GPU clock speed,
    and matches Planck frequency, we've found the hardware refresh rate.
    """
    print(f"\n{'='*70}")
    print("  BARYONIC ACOUSTIC OSCILLATIONS TEST")
    print("  Goal: Find correlation between BAO frequency and GPU clock")
    print(f"{'='*70}")

    # Compute power spectrum at multiple times
    pk_history = []
    bao_peaks = []
    gpu_clocks = []

    print(f"\n  Evolving and measuring BAO at multiple epochs...")

    # Get GPU state if profiler available
    if gpu_profiler and gpu_profiler.handle:
        import pynvml
        handle = gpu_profiler.handle

    for epoch in range(5):
        # Evolve a bit
        target_z = max(0, sim.redshift - 20)

        start_time = time.perf_counter()
        sim.evolve_to_redshift(target_z, dt_myr=100, callback_interval=1000)
        elapsed = time.perf_counter() - start_time

        # Compute power spectrum
        pk = sim.compute_power_spectrum()
        pk_history.append(pk)
        bao_peaks.append(pk.k_peak)

        # Get GPU clock
        if gpu_profiler:
            gpu_state = get_gpu_state()
            if gpu_state:
                gpu_clocks.append(gpu_state.clock_speed_mhz)
            else:
                gpu_clocks.append(0)

        # Frequency = evolution rate
        frequency = 1.0 / elapsed if elapsed > 0 else 0

        print(f"    Epoch {epoch+1}: z={sim.redshift:.1f}, BAO k={pk.k_peak:.4f}, freq={frequency:.2f} Hz")

    # Analyze BAO scale
    if bao_peaks:
        measured_k = np.mean(bao_peaks)
        measured_scale = 2 * np.pi / measured_k if measured_k > 0 else 0
    else:
        measured_scale = 0

    scale_error = abs(measured_scale - BAO_SCALE_MPC) / BAO_SCALE_MPC * 100

    # Compute "frequency" correlation with GPU clock
    if gpu_clocks and len(gpu_clocks) > 1:
        mean_clock = np.mean(gpu_clocks)
        clock_variance = np.var(gpu_clocks)

        # Planck frequency ratio
        planck_freq = 1.0 / PLANCK_TIME
        sim_freq = mean_clock * 1e6  # MHz to Hz
        planck_ratio = sim_freq / planck_freq

        # Correlation between clock and BAO variation
        if len(bao_peaks) == len(gpu_clocks):
            correlation = np.corrcoef(bao_peaks, gpu_clocks)[0, 1]
        else:
            correlation = 0
    else:
        mean_clock = 0
        planck_ratio = 0
        correlation = 0

    # Interpretation
    if abs(correlation) > 0.7:
        interpretation = f"CRITICAL: Strong correlation ({correlation:.2f}) between GPU clock and BAO scale! Hardware refresh rate may affect physics."
    elif scale_error < 5:
        interpretation = f"CONSISTENT: Measured BAO scale {measured_scale:.1f} Mpc matches real universe ({scale_error:.1f}% error)."
    else:
        interpretation = f"DEVIATION: BAO scale differs from expected by {scale_error:.1f}%. Simulation parameters may need calibration."

    print(f"\n  RESULTS:")
    print(f"    Measured BAO scale: {measured_scale:.1f} Mpc")
    print(f"    Expected BAO scale: {BAO_SCALE_MPC} Mpc")
    print(f"    Scale error: {scale_error:.1f}%")
    if mean_clock > 0:
        print(f"    Mean GPU clock: {mean_clock:.0f} MHz")
        print(f"    Clock-BAO correlation: {correlation:.3f}")
    print(f"\n  {interpretation}")

    return BAOResult(
        measured_scale_mpc=measured_scale,
        scale_error_percent=scale_error,
        frequency_hz=np.mean(bao_peaks) if bao_peaks else 0,
        gpu_clock_mhz=int(mean_clock) if mean_clock else 0,
        planck_frequency_ratio=planck_ratio,
        clock_correlation=correlation if not np.isnan(correlation) else 0,
        interpretation=interpretation
    )


# =============================================================================
# CROSS-SUBSTRATE MIRROR TEST
# =============================================================================

@dataclass
class SubstrateMirrorResult:
    """Results from cross-substrate comparison."""
    platform_a: str
    platform_b: str
    seed: int
    num_particles: int
    final_redshift: float

    # Position comparison
    position_hash_a: str
    position_hash_b: str
    hashes_match: bool

    # Statistical comparison
    mean_position_diff: float
    max_position_diff: float
    position_correlation: float

    # Velocity comparison
    mean_velocity_diff: float
    velocity_correlation: float

    # Structure comparison
    void_count_a: int
    void_count_b: int
    void_overlap_percent: float

    # The verdict
    deterministic: bool  # Did different hardware produce same result?
    admin_intervention_detected: bool
    interpretation: str


def export_state_for_comparison(sim: CosmologicalSimulation, filepath: str):
    """
    Export simulation state for cross-platform comparison.
    """
    state = sim.get_state()

    export_data = {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "os": os.name,
            "python": sys.version,
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "device": str(sim.device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        },
        "simulation": {
            "seed": state.seed,
            "precision": state.precision,
            "num_particles": state.num_particles,
            "redshift": state.redshift,
            "time_gyr": state.time_gyr,
            "state_hash": state.state_hash
        },
        "positions": state.positions.tolist(),
        "velocities": state.velocities.tolist(),
        "masses": state.masses.tolist()
    }

    with open(filepath, 'w') as f:
        json.dump(export_data, f)

    print(f"  Exported state to: {filepath}")
    print(f"  State hash: {state.state_hash}")

    return state.state_hash


def compare_substrate_states(state_a_path: str, state_b_path: str) -> SubstrateMirrorResult:
    """
    Compare simulation states from different platforms.

    The Matrix Proof: If positions match EXACTLY despite different hardware,
    an external force is enforcing determinism.
    """
    print(f"\n{'='*70}")
    print("  CROSS-SUBSTRATE MIRROR TEST")
    print("  Goal: Detect 'Admin' intervention enforcing outcomes")
    print(f"{'='*70}")

    # Load states
    with open(state_a_path) as f:
        state_a = json.load(f)
    with open(state_b_path) as f:
        state_b = json.load(f)

    platform_a = f"{state_a['platform']['os']}/{state_a['platform']['gpu']}"
    platform_b = f"{state_b['platform']['os']}/{state_b['platform']['gpu']}"

    print(f"\n  Platform A: {platform_a}")
    print(f"  Platform B: {platform_b}")
    print(f"  Seed A: {state_a['simulation']['seed']}")
    print(f"  Seed B: {state_b['simulation']['seed']}")

    # Check seeds match
    if state_a['simulation']['seed'] != state_b['simulation']['seed']:
        print("  WARNING: Seeds don't match! Comparison may not be meaningful.")

    # Compare hashes
    hash_a = state_a['simulation']['state_hash']
    hash_b = state_b['simulation']['state_hash']
    hashes_match = hash_a == hash_b

    print(f"\n  Hash A: {hash_a}")
    print(f"  Hash B: {hash_b}")
    print(f"  Exact match: {'YES!' if hashes_match else 'NO'}")

    # Statistical comparison
    pos_a = np.array(state_a['positions'])
    pos_b = np.array(state_b['positions'])
    vel_a = np.array(state_a['velocities'])
    vel_b = np.array(state_b['velocities'])

    # Position differences
    pos_diff = np.abs(pos_a - pos_b)
    mean_pos_diff = pos_diff.mean()
    max_pos_diff = pos_diff.max()

    # Correlation
    pos_corr = np.corrcoef(pos_a.flatten(), pos_b.flatten())[0, 1]
    vel_corr = np.corrcoef(vel_a.flatten(), vel_b.flatten())[0, 1]

    print(f"\n  Position difference: mean={mean_pos_diff:.6f}, max={max_pos_diff:.6f}")
    print(f"  Position correlation: {pos_corr:.6f}")
    print(f"  Velocity correlation: {vel_corr:.6f}")

    # Scientific expectation: Different FP implementations should diverge
    # If correlation > 0.9999, that's suspiciously deterministic

    deterministic = hashes_match or (pos_corr > 0.9999 and mean_pos_diff < 1e-6)

    # "Admin intervention" = deterministic despite hardware differences
    admin_detected = deterministic and (platform_a != platform_b)

    # Interpretation
    if admin_detected:
        interpretation = "CRITICAL: IDENTICAL results from DIFFERENT hardware! An external force is enforcing this outcome. The 'Admin' is real."
    elif hashes_match:
        interpretation = "EXACT MATCH: Positions identical to bit level. Either same hardware or deterministic substrate."
    elif pos_corr > 0.999:
        interpretation = f"HIGHLY CORRELATED ({pos_corr:.6f}): Results nearly identical. FP differences minimal."
    elif pos_corr > 0.9:
        interpretation = f"CORRELATED ({pos_corr:.4f}): Expected divergence from FP math differences."
    else:
        interpretation = f"DIVERGED ({pos_corr:.4f}): Significant differences as expected from different hardware."

    print(f"\n  VERDICT: {'ADMIN DETECTED!' if admin_detected else 'No admin intervention'}")
    print(f"  {interpretation}")

    return SubstrateMirrorResult(
        platform_a=platform_a,
        platform_b=platform_b,
        seed=state_a['simulation']['seed'],
        num_particles=state_a['simulation']['num_particles'],
        final_redshift=state_a['simulation']['redshift'],
        position_hash_a=hash_a,
        position_hash_b=hash_b,
        hashes_match=hashes_match,
        mean_position_diff=mean_pos_diff,
        max_position_diff=max_pos_diff,
        position_correlation=pos_corr,
        mean_velocity_diff=np.abs(vel_a - vel_b).mean(),
        velocity_correlation=vel_corr,
        void_count_a=0,  # Would need structure detection
        void_count_b=0,
        void_overlap_percent=0,
        deterministic=deterministic,
        admin_intervention_detected=admin_detected,
        interpretation=interpretation
    )


# =============================================================================
# INTEGRATED ULTIMATE TEST
# =============================================================================

@dataclass
class UltimateRealityReport:
    """The final, comprehensive reality test report."""
    timestamp: str
    device: str
    gpu_name: str

    # Cosmological simulation
    cosmo_seed: int
    cosmo_particles: int
    final_redshift: float
    evolution_time_seconds: float

    # BAO results
    bao: BAOResult

    # Structure detection
    structures: FilamentStructure
    great_void_match: bool

    # Cross-substrate (if available)
    substrate_comparison: Optional[SubstrateMirrorResult]

    # Power spectrum
    final_power_spectrum: PowerSpectrum

    # Energy conservation
    energy_drift_percent: float

    # GPU profiling
    mean_power_watts: float
    mean_clock_mhz: float
    throttle_events: int

    # Scores from all subsystems
    precision_score: float
    recursion_score: float
    fluid_score: float
    neural_score: float
    voxel_score: float
    cosmo_score: float

    # Final verdict
    combined_score: float
    simulation_probability: float
    verdict: str


def run_ultimate_reality_test(
    num_particles: int = 100000,
    target_redshift: float = 0.0,
    precision: str = "float32",
    seed: int = 42,
    compare_state_path: str = None,
    export_state_path: str = None,
    output_dir: str = None,
    device: torch.device = None
) -> UltimateRealityReport:
    """
    Run the ULTIMATE reality test combining everything.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*70)
    print("  ULTIMATE REALITY ENGINE - The Final Test")
    print("  Combining all simulation hypothesis probes")
    print("="*70)
    print(f"  Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU: {gpu_name}")
    else:
        gpu_name = "CPU"
    print(f"  Particles: {num_particles:,}")
    print(f"  Precision: {precision}")
    print(f"  Seed: {seed}")
    print(f"  Target redshift: z={target_redshift}")
    print("="*70)

    # Initialize profiler
    profiler = GPUProfiler(sample_interval_ms=100)
    profiler.start("ultimate_reality_test")

    # Create cosmological simulation
    print(f"\n  PHASE 1: Cosmological Simulation")
    print(f"  {'-'*50}")

    prec_map = {
        "float64": CosmologicalPrecision.FLOAT64,
        "float32": CosmologicalPrecision.FLOAT32,
        "float16": CosmologicalPrecision.FLOAT16,
        "int8": CosmologicalPrecision.INT8,
        "int4": CosmologicalPrecision.INT4,
    }

    start_time = time.perf_counter()

    sim = CosmologicalSimulation(
        num_particles=num_particles,
        box_size_mpc=100.0,
        precision=prec_map.get(precision, CosmologicalPrecision.FLOAT32),
        seed=seed,
        device=device
    )

    # Track initial energy
    initial_ke = 0.5 * (sim.masses.unsqueeze(1) * sim.velocities ** 2).sum().item()

    # Evolution tracking
    def evolution_callback(s, step):
        if step % 50 == 0:
            pk = s.compute_power_spectrum()
            s.power_spectrum_history.append(pk)

    # Evolve to target
    sim.evolve_to_redshift(target_redshift, dt_myr=100, callback=evolution_callback, callback_interval=10)

    evolution_time = time.perf_counter() - start_time
    print(f"\n  Evolution completed in {evolution_time:.1f}s")

    # Final energy
    final_ke = 0.5 * (sim.masses.unsqueeze(1) * sim.velocities ** 2).sum().item()
    energy_drift = abs(final_ke - initial_ke) / (abs(initial_ke) + 1e-10) * 100
    print(f"  Energy drift: {energy_drift:.2f}%")

    # Phase 2: BAO Analysis
    print(f"\n  PHASE 2: Baryonic Acoustic Oscillations")
    print(f"  {'-'*50}")

    bao_result = run_bao_test(sim, profiler)

    # Phase 3: Structure Detection
    print(f"\n  PHASE 3: Cosmic Web Structure Detection")
    print(f"  {'-'*50}")

    structures = sim.detect_structures()
    print(f"  Detected {structures.num_filaments:,} filament cells")
    print(f"  Detected {structures.num_voids} void regions")
    print(f"  Great Void match: {'YES!' if structures.great_void_match else 'NO'}")

    # Phase 4: Export/Compare State
    substrate_result = None

    if export_state_path:
        print(f"\n  PHASE 4: Exporting State for Cross-Platform Comparison")
        print(f"  {'-'*50}")
        export_state_for_comparison(sim, export_state_path)

    if compare_state_path:
        print(f"\n  PHASE 4: Cross-Substrate Comparison")
        print(f"  {'-'*50}")
        # Export current state
        current_state_path = "/tmp/current_state.json" if os.name != 'nt' else "current_state.json"
        export_state_for_comparison(sim, current_state_path)
        substrate_result = compare_substrate_states(current_state_path, compare_state_path)

    # Stop profiler
    profile_result = profiler.stop()

    # Final power spectrum
    final_pk = sim.compute_power_spectrum()

    # Compute scores
    print(f"\n  PHASE 5: Computing Scores")
    print(f"  {'-'*50}")

    # BAO score (how well it matches real universe)
    cosmo_score = max(0, 100 - bao_result.scale_error_percent * 10)

    # Structure score
    structure_score = 50
    if structures.great_void_match:
        structure_score += 30
    if structures.num_voids > 10:
        structure_score += 20

    # Energy score (lower drift = better)
    energy_score = max(0, 100 - energy_drift * 10)

    # Substrate score
    substrate_score = 50
    if substrate_result:
        if substrate_result.admin_intervention_detected:
            substrate_score = 100
        elif substrate_result.position_correlation > 0.999:
            substrate_score = 80

    # Placeholder scores for other tests (would integrate with other modules)
    precision_score = max(0, 100 - energy_drift * 5)
    recursion_score = 50  # Placeholder
    fluid_score = 50
    neural_score = 50
    voxel_score = 50

    # Combined score
    scores = [cosmo_score, structure_score, energy_score, precision_score]
    if substrate_result:
        scores.append(substrate_score)

    combined_score = np.mean(scores)

    # Simulation probability estimate (HEURISTIC - not formal probability!)
    sim_probability = min(100, combined_score * 1.2)

    # Verdict
    if combined_score > 80 or (substrate_result and substrate_result.admin_intervention_detected):
        verdict = "CRITICAL: Strong evidence of simulation substrate detected!"
    elif combined_score > 60:
        verdict = "SIGNIFICANT: Multiple anomalies suggest computational nature of reality."
    elif combined_score > 40:
        verdict = "SUGGESTIVE: Some anomalies detected, further investigation needed."
    else:
        verdict = "INCONCLUSIVE: Results consistent with continuous physics."

    # Print final summary
    print(f"\n{'='*70}")
    print("  ULTIMATE REALITY TEST - FINAL REPORT")
    print("="*70)

    print(f"\n  SIMULATION METRICS:")
    print(f"    Particles: {num_particles:,}")
    print(f"    Evolution time: {evolution_time:.1f}s")
    print(f"    Final redshift: z={sim.redshift:.2f}")
    print(f"    Energy drift: {energy_drift:.2f}%")

    print(f"\n  BAO ANALYSIS:")
    print(f"    Measured scale: {bao_result.measured_scale_mpc:.1f} Mpc")
    print(f"    Expected scale: {BAO_SCALE_MPC} Mpc")
    print(f"    Error: {bao_result.scale_error_percent:.1f}%")

    print(f"\n  STRUCTURE DETECTION:")
    print(f"    Voids found: {structures.num_voids}")
    print(f"    Great Void match: {structures.great_void_match}")

    if substrate_result:
        print(f"\n  CROSS-SUBSTRATE:")
        print(f"    Hash match: {substrate_result.hashes_match}")
        print(f"    Correlation: {substrate_result.position_correlation:.6f}")
        print(f"    Admin detected: {substrate_result.admin_intervention_detected}")

    if profile_result:
        print(f"\n  GPU PROFILE:")
        print(f"    Mean power: {profile_result.mean_power_watts:.1f}W")
        print(f"    Mean clock: {profile_result.mean_clock_mhz:.0f} MHz")
        print(f"    Throttle events: {profile_result.throttle_events}")

    print(f"\n  SCORES:")
    print(f"    Cosmological: {cosmo_score:.0f}/100")
    print(f"    Structure: {structure_score:.0f}/100")
    print(f"    Energy: {energy_score:.0f}/100")
    print(f"    Precision: {precision_score:.0f}/100")
    if substrate_result:
        print(f"    Substrate: {substrate_score:.0f}/100")

    print(f"\n  COMBINED SCORE: {combined_score:.1f}/100")
    print(f"  SIMULATION PROBABILITY: {sim_probability:.1f}%")
    print(f"\n  VERDICT: {verdict}")
    print("="*70)

    # Create report
    report = UltimateRealityReport(
        timestamp=datetime.now().isoformat(),
        device=str(device),
        gpu_name=gpu_name,
        cosmo_seed=seed,
        cosmo_particles=num_particles,
        final_redshift=sim.redshift,
        evolution_time_seconds=evolution_time,
        bao=bao_result,
        structures=structures,
        great_void_match=structures.great_void_match,
        substrate_comparison=substrate_result,
        final_power_spectrum=final_pk,
        energy_drift_percent=energy_drift,
        mean_power_watts=profile_result.mean_power_watts if profile_result else 0,
        mean_clock_mhz=profile_result.mean_clock_mhz if profile_result else 0,
        throttle_events=profile_result.throttle_events if profile_result else 0,
        precision_score=precision_score,
        recursion_score=recursion_score,
        fluid_score=fluid_score,
        neural_score=neural_score,
        voxel_score=voxel_score,
        cosmo_score=cosmo_score,
        combined_score=combined_score,
        simulation_probability=sim_probability,
        verdict=verdict
    )

    # Save report
    if output_dir:
        Path(output_dir).mkdir(exist_ok=True)
        report_path = Path(output_dir) / "ultimate_reality_report.json"

        def serialize(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return asdict(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            return str(obj)

        with open(report_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=serialize)
        print(f"\n  Report saved to: {report_path}")

    return report


# =============================================================================
# SDSS/CMB REAL DATA COMPARISON
# =============================================================================

@dataclass
class SDSSComparison:
    """Results from SDSS galaxy survey comparison."""
    # Correlation function
    xi_r_simulated: np.ndarray  # 2-point correlation function from sim
    xi_r_sdss: np.ndarray  # Reference SDSS values
    correlation_match: float  # How well they match (0-100)

    # BAO scale
    bao_scale_simulated: float
    bao_scale_sdss: float  # ~147 Mpc
    bao_error_percent: float

    # Void statistics
    void_size_distribution_match: float

    interpretation: str


@dataclass
class CMBComparison:
    """Results from CMB power spectrum comparison."""
    # Angular power spectrum
    l_values: np.ndarray  # Multipole moments
    cl_simulated: np.ndarray  # Simulated spectrum
    cl_planck: np.ndarray  # Planck reference

    # Key peaks
    first_peak_l_sim: int
    first_peak_l_planck: int  # ~220
    peak_match: bool

    # Spectral index
    ns_simulated: float
    ns_planck: float  # 0.965

    # Overall match
    chi_squared: float
    match_percent: float

    interpretation: str


# Reference cosmological data (from Planck 2018 and SDSS DR16)
SDSS_BAO_SCALE = 147.09  # Mpc (BOSS measurement)
SDSS_BAO_ERROR = 0.26  # Mpc

# SDSS 2-point correlation function reference (approximate)
SDSS_XI_R = {
    1: 40.0, 2: 15.0, 5: 4.0, 10: 1.5, 20: 0.5, 50: 0.1, 100: 0.02
}

# CMB power spectrum peaks (Planck 2018)
CMB_PEAKS = {
    'first': 220,   # First acoustic peak at l~220
    'second': 546,  # Second peak
    'third': 800,   # Third peak
}


def compute_2point_correlation(positions: np.ndarray, box_size: float,
                               r_bins: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the 2-point correlation function xi(r).

    This is the key statistic for comparing to SDSS.
    """
    if r_bins is None:
        r_bins = np.array([1, 2, 5, 10, 20, 50, 100])

    n = len(positions)

    # For large datasets, subsample
    if n > 10000:
        idx = np.random.choice(n, 10000, replace=False)
        positions = positions[idx]
        n = 10000

    # Compute pairwise distances
    xi_values = []

    for r in r_bins:
        # Count pairs within shell [r-dr, r+dr]
        dr = r * 0.2  # Shell thickness

        pair_count = 0
        for i in range(min(n, 1000)):  # Limit for speed
            diff = positions - positions[i]
            # Apply periodic boundary
            diff = np.where(diff > box_size/2, diff - box_size, diff)
            diff = np.where(diff < -box_size/2, diff + box_size, diff)
            dist = np.sqrt((diff**2).sum(axis=1))

            pair_count += np.sum((dist > r-dr) & (dist < r+dr) & (dist > 0))

        # Expected count for random distribution
        shell_volume = 4/3 * np.pi * ((r+dr)**3 - max(0, r-dr)**3)
        expected = 1000 * n / box_size**3 * shell_volume

        # Correlation function
        xi = (pair_count / expected - 1) if expected > 0 else 0
        xi_values.append(xi)

    return r_bins, np.array(xi_values)


def compare_to_sdss(sim: 'CosmologicalSimulation') -> SDSSComparison:
    """
    Compare simulation results to SDSS galaxy survey data.
    """
    print(f"\n{'='*70}")
    print("  SDSS DATA COMPARISON")
    print("  Comparing to Sloan Digital Sky Survey observations")
    print(f"{'='*70}")

    positions = sim.positions.cpu().numpy()

    # 2-point correlation
    print("  Computing 2-point correlation function...")
    r_bins, xi_sim = compute_2point_correlation(positions, sim.box_size)

    # Reference SDSS values
    xi_sdss = np.array([SDSS_XI_R.get(int(r), 1.0) for r in r_bins])

    # Correlation match (normalized comparison)
    xi_sim_norm = xi_sim / (np.max(np.abs(xi_sim)) + 1e-10)
    xi_sdss_norm = xi_sdss / (np.max(np.abs(xi_sdss)) + 1e-10)
    correlation_match = 100 * (1 - np.mean(np.abs(xi_sim_norm - xi_sdss_norm)))

    print(f"  Correlation function match: {correlation_match:.1f}%")

    # BAO comparison
    pk = sim.compute_power_spectrum()
    bao_sim = 2 * np.pi / pk.k_peak if pk.k_peak > 0 else 0
    bao_error = abs(bao_sim - SDSS_BAO_SCALE) / SDSS_BAO_SCALE * 100

    print(f"  BAO scale: {bao_sim:.1f} Mpc (SDSS: {SDSS_BAO_SCALE:.1f} Mpc)")
    print(f"  BAO error: {bao_error:.1f}%")

    # Void statistics
    structures = sim.detect_structures()
    void_match = 50 + (30 if structures.num_voids > 5 else 0) + (20 if structures.great_void_match else 0)

    print(f"  Void statistics match: {void_match:.0f}%")

    # Interpretation
    if correlation_match > 70 and bao_error < 10:
        interpretation = f"EXCELLENT: Simulation closely matches SDSS observations!"
    elif correlation_match > 50:
        interpretation = f"GOOD: Reasonable match to SDSS, some discrepancies expected."
    else:
        interpretation = f"POOR: Significant deviation from SDSS. Check simulation parameters."

    print(f"\n  {interpretation}")

    return SDSSComparison(
        xi_r_simulated=xi_sim,
        xi_r_sdss=xi_sdss,
        correlation_match=correlation_match,
        bao_scale_simulated=bao_sim,
        bao_scale_sdss=SDSS_BAO_SCALE,
        bao_error_percent=bao_error,
        void_size_distribution_match=void_match,
        interpretation=interpretation
    )


def compare_to_cmb(sim: 'CosmologicalSimulation') -> CMBComparison:
    """
    Compare simulation's initial conditions to CMB power spectrum.

    This tests whether the primordial fluctuations match Planck observations.
    """
    print(f"\n{'='*70}")
    print("  CMB POWER SPECTRUM COMPARISON")
    print("  Comparing to Planck 2018 observations")
    print(f"{'='*70}")

    # Compute angular power spectrum (approximation from 3D power spectrum)
    pk = sim.compute_power_spectrum(n_bins=100)

    # Convert k to l (angular multipole) approximately
    # l ~ k * D_A where D_A is angular diameter distance to CMB
    D_A_cmb = 14000  # Mpc (approximate)
    l_values = pk.k * D_A_cmb

    # Simulated C_l (very rough approximation)
    cl_sim = pk.pk / (l_values + 1)**2

    # Planck reference (simplified Sachs-Wolfe plateau + peaks)
    l_ref = np.linspace(2, 2000, 100)
    cl_planck = np.zeros_like(l_ref)

    # First peak at l~220
    for i, l in enumerate(l_ref):
        # Sachs-Wolfe plateau
        cl_planck[i] = 1000 / (l + 1)**0.1

        # Add acoustic peaks
        for peak_l in [220, 546, 800, 1100]:
            width = 50
            cl_planck[i] += 500 * np.exp(-(l - peak_l)**2 / (2 * width**2))

    # Find first peak in simulation
    peak_idx = np.argmax(cl_sim)
    first_peak_sim = int(l_values[peak_idx]) if peak_idx < len(l_values) else 0

    # Compare
    peak_match = abs(first_peak_sim - CMB_PEAKS['first']) < 50

    # Spectral index (from power spectrum slope)
    if len(pk.k) > 5:
        log_k = np.log(pk.k[1:6])
        log_pk = np.log(pk.pk[1:6] + 1e-10)
        ns_sim = np.polyfit(log_k, log_pk, 1)[0] + 1  # P(k) ~ k^(ns-1)
    else:
        ns_sim = 1.0

    # Chi-squared (simplified)
    # Interpolate to common l values
    common_l = np.linspace(10, 1000, 50)
    cl_sim_interp = np.interp(common_l, l_values, cl_sim)
    cl_planck_interp = np.interp(common_l, l_ref, cl_planck)

    # Normalize
    cl_sim_interp = cl_sim_interp / np.max(cl_sim_interp)
    cl_planck_interp = cl_planck_interp / np.max(cl_planck_interp)

    chi_sq = np.sum((cl_sim_interp - cl_planck_interp)**2)
    match_percent = 100 * np.exp(-chi_sq / 10)

    print(f"  First peak location: l={first_peak_sim} (Planck: {CMB_PEAKS['first']})")
    print(f"  Peak match: {'YES' if peak_match else 'NO'}")
    print(f"  Spectral index ns: {ns_sim:.3f} (Planck: {N_S})")
    print(f"  Overall match: {match_percent:.1f}%")

    # Interpretation
    if match_percent > 70 and peak_match:
        interpretation = "EXCELLENT: Power spectrum matches CMB observations!"
    elif match_percent > 40:
        interpretation = "MODERATE: Partial match to CMB, primordial conditions approximate."
    else:
        interpretation = "POOR: Significant deviation from CMB. Initial conditions need work."

    print(f"\n  {interpretation}")

    return CMBComparison(
        l_values=l_values,
        cl_simulated=cl_sim,
        cl_planck=cl_planck,
        first_peak_l_sim=first_peak_sim,
        first_peak_l_planck=CMB_PEAKS['first'],
        peak_match=peak_match,
        ns_simulated=ns_sim,
        ns_planck=N_S,
        chi_squared=chi_sq,
        match_percent=match_percent,
        interpretation=interpretation
    )


# =============================================================================
# COMPREHENSIVE ALL-TESTS INTEGRATION
# =============================================================================

@dataclass
class ComprehensiveTestReport:
    """Report from running ALL available tests."""
    timestamp: str
    device: str
    gpu_name: str
    total_runtime_seconds: float

    # Individual test results
    cosmological: Optional['UltimateRealityReport'] = None
    sdss_comparison: Optional[SDSSComparison] = None
    cmb_comparison: Optional[CMBComparison] = None
    omniverse_results: Optional[Dict[str, Any]] = None
    sensitivity_results: Optional[List[Any]] = None
    orbital_results: Optional[Any] = None

    # Aggregate scores
    cosmological_score: float = 0.0
    precision_score: float = 0.0
    real_data_score: float = 0.0
    omniverse_score: float = 0.0
    orbital_score: float = 0.0

    # Final
    combined_score: float = 0.0
    simulation_probability: float = 0.0
    verdict: str = ""


def run_all_tests(
    num_particles: int = 100000,
    target_redshift: float = 0.0,
    precision: str = "float32",
    seed: int = 42,
    skip_slow: bool = False,
    output_dir: str = None,
    device: torch.device = None
) -> ComprehensiveTestReport:
    """
    Run EVERY available test in the suite.

    This is the ultimate comprehensive test.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    print("\n" + "="*70)
    print("  COMPREHENSIVE REALITY TEST SUITE")
    print("  Running ALL available tests")
    print("="*70)
    print(f"  Device: {device} ({gpu_name})")
    print(f"  Particles: {num_particles:,}")
    print(f"  Precision: {precision}")
    print(f"  Skip slow tests: {skip_slow}")
    print("="*70)

    start_time = time.perf_counter()
    scores = []

    # =========================================================================
    # TEST 1: Cosmological Simulation (Main)
    # =========================================================================
    print(f"\n{'='*70}")
    print("  TEST 1: COSMOLOGICAL SIMULATION")
    print(f"{'='*70}")

    cosmo_report = run_ultimate_reality_test(
        num_particles=num_particles,
        target_redshift=target_redshift,
        precision=precision,
        seed=seed,
        output_dir=output_dir,
        device=device
    )
    cosmo_score = cosmo_report.combined_score
    scores.append(("Cosmological", cosmo_score))

    # =========================================================================
    # TEST 2: SDSS/CMB Real Data Comparison
    # =========================================================================
    print(f"\n{'='*70}")
    print("  TEST 2: REAL COSMOLOGICAL DATA COMPARISON")
    print(f"{'='*70}")

    # Re-create simulation for comparisons
    prec_map = {
        "float64": CosmologicalPrecision.FLOAT64,
        "float32": CosmologicalPrecision.FLOAT32,
        "float16": CosmologicalPrecision.FLOAT16,
        "int8": CosmologicalPrecision.INT8,
        "int4": CosmologicalPrecision.INT4,
    }

    sim = CosmologicalSimulation(
        num_particles=num_particles,
        box_size_mpc=100.0,
        precision=prec_map.get(precision, CosmologicalPrecision.FLOAT32),
        seed=seed,
        device=device
    )
    sim.evolve_to_redshift(target_redshift, dt_myr=100, callback_interval=1000)

    sdss_result = compare_to_sdss(sim)
    cmb_result = compare_to_cmb(sim)

    real_data_score = (sdss_result.correlation_match + cmb_result.match_percent) / 2
    scores.append(("Real Data", real_data_score))

    # =========================================================================
    # TEST 3: Precision Sensitivity Sweep
    # =========================================================================
    precision_score = 50.0
    sensitivity_results = None

    if SENSITIVITY_AVAILABLE and not skip_slow:
        print(f"\n{'='*70}")
        print("  TEST 3: PRECISION SENSITIVITY SWEEP")
        print(f"{'='*70}")

        try:
            # Use fewer stars for speed
            sensitivity_results = run_sensitivity_sweep(
                num_stars=min(1000, num_particles),
                num_ticks=200,
                device=device
            )

            # Score based on monotonicity of energy drift
            if sensitivity_results:
                drifts = [r.energy_drift_pct for r in sensitivity_results]
                # Check if lower precision = more drift (expected)
                monotonic = sum(1 for i in range(len(drifts)-1) if drifts[i] >= drifts[i+1])
                precision_score = 100 * monotonic / (len(drifts) - 1)

            scores.append(("Precision", precision_score))
        except Exception as e:
            print(f"  Sensitivity test failed: {e}")
    else:
        print(f"\n  Skipping sensitivity sweep (skip_slow={skip_slow}, available={SENSITIVITY_AVAILABLE})")

    # =========================================================================
    # TEST 4: Omniverse Tests
    # =========================================================================
    omniverse_results = {}
    omniverse_score = 50.0

    if OMNIVERSE_AVAILABLE and not skip_slow:
        print(f"\n{'='*70}")
        print("  TEST 4: OMNIVERSE REALITY PROBES")
        print(f"{'='*70}")

        try:
            # Create galaxy for omniverse tests
            positions, velocities, masses = create_disk_galaxy(
                num_stars=min(2000, num_particles),
                galaxy_radius=10.0,
                device=device
            )

            # Recursive Physics Mirror
            print("\n  4a. Recursive Physics Mirror...")
            rpm = RecursivePhysicsMirror(device=device)
            rpm_result = rpm.run_test(
                positions.clone(), velocities.clone(), masses.clone(),
                max_depth=4
            )
            omniverse_results['recursive'] = rpm_result

            # Fluid Dynamics Chaos
            print("\n  4b. Fluid Dynamics Chaos...")
            fdc = FluidDynamicsChaos(device=device)
            fdc_result = fdc.run_test(positions.clone(), velocities.clone(), masses.clone())
            omniverse_results['fluid'] = fdc_result

            # Voxel Space-Time Grid
            print("\n  4c. Voxel Space-Time Grid...")
            vst = VoxelSpaceTimeGrid(device=device)
            vst_result = vst.run_test(positions.clone(), velocities.clone(), masses.clone())
            omniverse_results['voxel'] = vst_result

            # Compute aggregate score
            omni_scores = []
            if hasattr(rpm_result, 'recursion_limit_detected'):
                omni_scores.append(80 if rpm_result.recursion_limit_detected else 50)
            if hasattr(fdc_result, 'lod_artifacts_detected'):
                omni_scores.append(80 if fdc_result.lod_artifacts_detected else 50)
            if hasattr(vst_result, 'spatial_quantization_detected'):
                omni_scores.append(80 if vst_result.spatial_quantization_detected else 50)

            if omni_scores:
                omniverse_score = np.mean(omni_scores)

            scores.append(("Omniverse", omniverse_score))

        except Exception as e:
            print(f"  Omniverse tests failed: {e}")
    else:
        print(f"\n  Skipping omniverse tests (skip_slow={skip_slow}, available={OMNIVERSE_AVAILABLE})")

    # =========================================================================
    # TEST 5: Orbital Mechanics (if available)
    # =========================================================================
    orbital_results = None
    orbital_score = 50.0

    if ORBITAL_AVAILABLE and not skip_slow:
        print(f"\n{'='*70}")
        print("  TEST 5: ORBITAL MECHANICS AUDIT")
        print(f"{'='*70}")

        try:
            orbital_results = run_full_orbital_audit(
                satellite_id=25544,  # ISS
                duration_hours=6,
                device=device
            )

            if orbital_results:
                orbital_score = orbital_results.overall_simulation_score
                scores.append(("Orbital", orbital_score))
        except Exception as e:
            print(f"  Orbital audit failed: {e}")
    else:
        print(f"\n  Skipping orbital audit (skip_slow={skip_slow}, available={ORBITAL_AVAILABLE})")

    # =========================================================================
    # FINAL SCORING
    # =========================================================================
    total_time = time.perf_counter() - start_time

    # Combined score
    combined_score = np.mean([s[1] for s in scores])

    # Simulation probability (heuristic!)
    sim_prob = min(100, combined_score * 1.1)

    # Verdict
    if combined_score > 80:
        verdict = "CRITICAL: Strong evidence of computational substrate across ALL tests!"
    elif combined_score > 65:
        verdict = "SIGNIFICANT: Multiple independent anomalies detected."
    elif combined_score > 50:
        verdict = "SUGGESTIVE: Some anomalies warrant further investigation."
    else:
        verdict = "INCONCLUSIVE: Results largely consistent with continuous physics."

    # Print summary
    print(f"\n{'='*70}")
    print("  COMPREHENSIVE TEST SUITE - FINAL SUMMARY")
    print("="*70)

    print(f"\n  Test Results:")
    for name, score in scores:
        bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
        print(f"    {name:15} [{bar}] {score:.1f}/100")

    print(f"\n  Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\n  COMBINED SCORE: {combined_score:.1f}/100")
    print(f"  SIMULATION PROBABILITY: {sim_prob:.1f}%")
    print(f"\n  VERDICT: {verdict}")
    print("="*70)

    report = ComprehensiveTestReport(
        timestamp=datetime.now().isoformat(),
        device=str(device),
        gpu_name=gpu_name,
        total_runtime_seconds=total_time,
        cosmological=cosmo_report,
        sdss_comparison=sdss_result,
        cmb_comparison=cmb_result,
        omniverse_results=omniverse_results,
        sensitivity_results=sensitivity_results,
        orbital_results=orbital_results,
        cosmological_score=cosmo_score,
        precision_score=precision_score,
        real_data_score=real_data_score,
        omniverse_score=omniverse_score,
        orbital_score=orbital_score,
        combined_score=combined_score,
        simulation_probability=sim_prob,
        verdict=verdict
    )

    # Save report
    if output_dir:
        Path(output_dir).mkdir(exist_ok=True)
        report_path = Path(output_dir) / "comprehensive_report.json"

        def serialize(obj):
            if obj is None:
                return None
            if hasattr(obj, '__dataclass_fields__'):
                return {k: serialize(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(v) for v in obj]
            return str(obj)

        try:
            with open(report_path, 'w') as f:
                json.dump(serialize(report), f, indent=2)
            print(f"\n  Report saved to: {report_path}")
        except Exception as e:
            print(f"\n  Warning: Could not save report: {e}")

    return report


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ultimate Reality Engine - The Final Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  full      - Run cosmological simulation with all local tests
  all       - Run EVERY test (cosmological, SDSS/CMB, omniverse, orbital)
  bigbang   - Cosmological simulation only
  bao       - BAO analysis only
  substrate - Export state for cross-platform comparison
  compare   - Compare with another platform's state

Examples:
  # Full cosmological test
  python ultimate_reality_engine.py --mode full --particles 1000000

  # Run EVERYTHING (comprehensive)
  python ultimate_reality_engine.py --mode all --particles 100000

  # Quick comprehensive test (skip slow modules)
  python ultimate_reality_engine.py --mode all --skip-slow

  # Export state for Mac comparison
  python ultimate_reality_engine.py --mode substrate --export results/windows_5090.json

  # Compare with Mac results
  python ultimate_reality_engine.py --mode compare --other-platform results/mac_m2.json

  # Different precision modes
  python ultimate_reality_engine.py --mode full --precision int4
        """
    )

    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "all", "bigbang", "bao", "substrate", "compare"],
                        help="Test mode")
    parser.add_argument("--particles", type=int, default=100000,
                        help="Number of particles")
    parser.add_argument("--redshift", type=float, default=0.0,
                        help="Target redshift (0 = present day)")
    parser.add_argument("--precision", type=str, default="float32",
                        choices=["float64", "float32", "float16", "int8", "int4"],
                        help="Simulation precision")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (use same seed for cross-platform comparison!)")
    parser.add_argument("--export", type=str, default=None,
                        help="Export state to this path")
    parser.add_argument("--other-platform", type=str, default=None,
                        help="Path to other platform's state for comparison")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--skip-slow", action="store_true",
                        help="Skip slow tests (orbital, sensitivity)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Run appropriate mode
    if args.mode == "all":
        # Run EVERYTHING
        report = run_all_tests(
            num_particles=args.particles,
            target_redshift=args.redshift,
            precision=args.precision,
            seed=args.seed,
            skip_slow=args.skip_slow,
            output_dir=args.output,
            device=device
        )
    else:
        # Default: run_ultimate_reality_test handles other modes
        report = run_ultimate_reality_test(
            num_particles=args.particles,
            target_redshift=args.redshift,
            precision=args.precision,
            seed=args.seed,
            compare_state_path=args.other_platform,
            export_state_path=args.export,
            output_dir=args.output,
            device=device
        )

    return report


if __name__ == "__main__":
    main()
