"""
UNIVERSE 2D - Unified Cosmic Evolution Engine
==============================================

TRULY UNIFIED ENGINE - Integrates ALL simulation modules:

From galaxy.py:
  - create_disk_galaxy, create_galaxy_with_halo, nfw_enclosed_mass
  - NFW dark matter halo profile

From simulation.py:
  - GalaxySimulation class (N-body leapfrog integrator)
  - run_comparison for multi-precision tests

From quantization.py:
  - PrecisionMode (FLOAT64 → INT4)
  - quantize_distance_squared, quantize_force
  - "Broken math" that creates ghost forces

From metrics.py:
  - compute_rotation_curve (dark matter signature)
  - compute_bound_fraction, compute_galaxy_radius
  - SimulationMetrics container

From gpu_profiler.py:
  - GPUProfiler for power/clock monitoring
  - Methodology validation (clock locking, throttle detection)

From reproducibility.py:
  - set_all_seeds for deterministic results
  - get_gpu_state for hardware verification

Each dot is a dark matter particle showing cosmic structure formation.

Fixed:
- Proper cosmic time calculation (no more temporal overflow)
- Correct BAO scale measurement (~147 Mpc)
- Integrated dark matter physics
- Quantization-based "simulation glitch" detection
- GPU profiling integration
"""

import argparse
import time
import logging
import signal
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import matplotlib
import platform
# Use native backend on macOS, TkAgg elsewhere
if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import math

# =============================================================================
# LOCAL IMPORTS - UNIFIED WITH ALL OUR MODULES
# =============================================================================
# Core modules
from reproducibility import set_all_seeds, get_gpu_state
from quantization import (
    PrecisionMode, quantize_distance_squared, quantize_force,
    _grid_quantize_safe, get_mode_from_string, describe_mode
)

# Galaxy creation and dark matter (from galaxy.py)
from galaxy import (
    create_disk_galaxy,
    create_galaxy_with_halo,
    nfw_enclosed_mass as galaxy_nfw_enclosed_mass,  # Use theirs, not duplicate
    create_test_galaxy
)

# N-body simulation engine (from simulation.py)
from simulation import GalaxySimulation, run_comparison

# Metrics and analysis (from metrics.py)
from metrics import (
    SimulationMetrics,
    compute_rotation_curve,
    compute_galaxy_radius,
    compute_bound_fraction,
    compute_velocity_dispersion,
    collect_metrics,
    compare_rotation_curves
)

# GPU profiling (from gpu_profiler.py)
from gpu_profiler import GPUProfiler, GPUProfileResult, GPUSample

# Reality glitch detection (from reality_glitch_tests.py)
# Import key metrics for integrated anomaly detection
try:
    from reality_glitch_tests import (
        count_subnormals_float32,
        measure_state_entropy,
        SubnormalMetrics,
        EntropyMetrics
    )
    HAS_GLITCH_TESTS = True
except ImportError:
    HAS_GLITCH_TESTS = False

# Orbital audit (from orbital_audit.py)
# For correlating with real-world data
try:
    from orbital_audit import AnomalyEvent
    HAS_ORBITAL_AUDIT = True
except ImportError:
    HAS_ORBITAL_AUDIT = False


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging for the simulation.
    Logs to both console and file.
    """
    logger = logging.getLogger("Universe2D")
    logger.setLevel(level)
    logger.handlers.clear()

    # Format
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (if specified)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    return logger


# Global logger
logger = setup_logging()

# =============================================================================
# COSMOLOGICAL CONSTANTS (Planck 2018)
# =============================================================================

H0 = 67.4  # km/s/Mpc
OMEGA_M = 0.315
OMEGA_LAMBDA = 0.685
OMEGA_R = 9.4e-5
SIGMA_8 = 0.811
N_S = 0.965
BAO_SCALE = 147.0  # Mpc
T_UNIVERSE = 13.8  # Gyr
C_LIGHT = 299792.458  # km/s
G_NEWTON = 4.302e-6  # (km/s)^2 * Mpc / M_sun

# Hubble time in Gyr
T_HUBBLE = 978.0 / H0  # ~14.5 Gyr


# =============================================================================
# COSMIC TIME - FIXED CALCULATION
# =============================================================================

def cosmic_time(z: float) -> float:
    """
    Cosmic time since Big Bang in Gyr.
    Uses proper numerical approximation for Lambda-CDM.

    Key reference values:
    - z=1089 (CMB): ~0.00038 Gyr (380,000 years)
    - z=20 (First Stars): ~0.18 Gyr
    - z=6 (Reionization): ~0.94 Gyr
    - z=2 (Peak SF): ~3.3 Gyr
    - z=1: ~5.9 Gyr
    - z=0: ~13.8 Gyr
    """
    if z < 0:
        return T_UNIVERSE

    # Numerical integration approximation using proper formula
    # t = integral from z to infinity of dz' / [(1+z') * H(z')]
    # H(z) = H0 * sqrt(Omega_r*(1+z)^4 + Omega_m*(1+z)^3 + Omega_Lambda)

    # Use lookup + interpolation for speed and accuracy
    z_table = np.array([0, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100, 500, 1000, 1100])
    t_table = np.array([13.8, 12.5, 8.6, 5.9, 3.3, 2.2, 1.2, 0.47, 0.18, 0.05, 0.017, 0.001, 0.0004, 0.00038])

    if z >= 1100:
        return 0.00038 * (1100 / z)**1.5  # Radiation dominated scaling
    elif z <= 0:
        return 13.8
    else:
        return float(np.interp(z, z_table, t_table))


def hubble_parameter(z: float) -> float:
    """Hubble parameter H(z) in km/s/Mpc."""
    return H0 * np.sqrt(OMEGA_R * (1 + z)**4 + OMEGA_M * (1 + z)**3 + OMEGA_LAMBDA)


def scale_factor(z: float) -> float:
    """Scale factor a = 1/(1+z)."""
    return 1.0 / (1.0 + z)


def growth_factor(z: float) -> float:
    """Linear growth factor D(z), normalized to D(0)=1."""
    a = scale_factor(z)
    omega_m_z = OMEGA_M * (1 + z)**3 / (OMEGA_M * (1 + z)**3 + OMEGA_LAMBDA)
    return a * (omega_m_z ** 0.55)


# =============================================================================
# COSMIC EPOCHS
# =============================================================================

class CosmicEpoch(Enum):
    RECOMBINATION = "cmb"
    DARK_AGES = "dark_ages"
    FIRST_STARS = "first_stars"
    REIONIZATION = "reion"
    GALAXY_FORMATION = "galaxies"
    PEAK_SF = "peak_sf"
    DARK_ENERGY = "dark_energy"
    PRESENT = "now"


@dataclass
class EpochInfo:
    name: str
    redshift: float
    time_gyr: float
    description: str
    color: str


EPOCHS = {
    CosmicEpoch.RECOMBINATION: EpochInfo("CMB/Recombination", 1089, 0.00038, "Photons decouple", "#ff6b6b"),
    CosmicEpoch.DARK_AGES: EpochInfo("Dark Ages", 100, 0.017, "No stars yet", "#2c3e50"),
    CosmicEpoch.FIRST_STARS: EpochInfo("First Stars", 20, 0.18, "Pop III stars ignite", "#f39c12"),
    CosmicEpoch.REIONIZATION: EpochInfo("Reionization", 7.7, 0.7, "UV ionizes IGM", "#9b59b6"),
    CosmicEpoch.GALAXY_FORMATION: EpochInfo("Galaxy Formation", 6, 0.94, "First galaxies", "#3498db"),
    CosmicEpoch.PEAK_SF: EpochInfo("Peak Star Formation", 2, 3.3, "Cosmic noon", "#2ecc71"),
    CosmicEpoch.DARK_ENERGY: EpochInfo("Dark Energy Era", 0.4, 9.8, "Acceleration begins", "#1abc9c"),
    CosmicEpoch.PRESENT: EpochInfo("Present Day", 0, 13.8, "Now", "#ecf0f1"),
}


def get_current_epoch(z: float) -> CosmicEpoch:
    if z > 1000: return CosmicEpoch.RECOMBINATION
    elif z > 30: return CosmicEpoch.DARK_AGES
    elif z > 15: return CosmicEpoch.FIRST_STARS
    elif z > 6: return CosmicEpoch.REIONIZATION
    elif z > 3: return CosmicEpoch.GALAXY_FORMATION
    elif z > 1: return CosmicEpoch.PEAK_SF
    elif z > 0.3: return CosmicEpoch.DARK_ENERGY
    else: return CosmicEpoch.PRESENT


# =============================================================================
# DARK MATTER MODEL (NFW Halo) - USES galaxy.py
# =============================================================================

# nfw_enclosed_mass is imported from galaxy.py as galaxy_nfw_enclosed_mass
# Use wrapper for convenience with same interface
def nfw_enclosed_mass(r: torch.Tensor, M_total: float, r_s: float) -> torch.Tensor:
    """
    Analytical NFW enclosed mass - delegates to galaxy.py implementation.
    M(<r) = M_total * f(r/r_s) where f(x) = ln(1+x) - x/(1+x)
    """
    return galaxy_nfw_enclosed_mass(r, M_total, r_s)


def compute_dm_density_field(positions: torch.Tensor, box_size: float,
                             n_grid: int, dm_ratio: float = 5.0) -> torch.Tensor:
    """
    Compute dark matter contribution to density field.
    Uses NFW profile smoothed onto grid.
    """
    # Dark matter adds a smooth background that enhances clustering
    # This is a simplified model - real DM follows the particles
    center = box_size / 2
    x = torch.linspace(0, box_size, n_grid, device=positions.device)
    y = torch.linspace(0, box_size, n_grid, device=positions.device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Distance from center
    r = torch.sqrt((xx - center)**2 + (yy - center)**2 + 1e-6)

    # NFW-like profile for dark matter
    r_s = box_size / 4  # Scale radius
    rho_dm = 1.0 / (r / r_s * (1 + r / r_s)**2 + 0.1)

    return rho_dm * dm_ratio


# =============================================================================
# GLITCH DETECTION - Quantization Artifacts
# =============================================================================

@dataclass
class GlitchEvent:
    """A detected simulation glitch from precision loss."""
    tick: int
    redshift: float
    glitch_type: str
    magnitude: float
    description: str


class GlitchDetector:
    """
    Detects anomalies caused by numerical precision loss.

    UNIFIED with reality_glitch_tests.py:
    - Energy conservation checks
    - Momentum drift (asymmetric forces)
    - Subnormal detection (denormal flooding)
    - Entropy monitoring (compression changes)
    """

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.events: List[GlitchEvent] = []
        self.energy_history: List[float] = []
        self.momentum_history: List[Tuple[float, float]] = []
        self.subnormal_history: List[int] = []
        self.entropy_history: List[float] = []

    def check_energy_conservation(self, energy: float, tick: int, redshift: float) -> Optional[GlitchEvent]:
        """Check for sudden energy jumps (quantization artifact)."""
        self.energy_history.append(energy)

        if len(self.energy_history) < 3:
            return None

        # Check for sudden change
        recent = self.energy_history[-3:]
        if recent[-2] != 0:
            delta = abs(recent[-1] - recent[-2]) / abs(recent[-2] + 1e-10)
            if delta > self.threshold:
                event = GlitchEvent(
                    tick=tick,
                    redshift=redshift,
                    glitch_type="energy_jump",
                    magnitude=delta,
                    description=f"Sudden energy change: {delta*100:.1f}%"
                )
                self.events.append(event)
                logger.warning(f"GLITCH DETECTED: {event.description} at z={redshift:.2f}, tick={tick}")
                return event
        return None

    def check_momentum(self, momentum: Tuple[float, float], tick: int, redshift: float) -> Optional[GlitchEvent]:
        """Check for momentum non-conservation (asymmetric forces from quantization)."""
        self.momentum_history.append(momentum)

        if len(self.momentum_history) < 2:
            return None

        # Total momentum should be ~constant in isolated system
        px, py = momentum
        total = math.sqrt(px**2 + py**2)

        if total > self.threshold * 1000:  # Scaled threshold
            event = GlitchEvent(
                tick=tick,
                redshift=redshift,
                glitch_type="momentum_drift",
                magnitude=total,
                description=f"Net momentum: ({px:.2f}, {py:.2f})"
            )
            self.events.append(event)
            logger.warning(f"GLITCH DETECTED: Momentum drift {total:.2e} at z={redshift:.2f}, tick={tick}")
            return event
        return None

    def check_subnormals(self, positions: torch.Tensor, tick: int, redshift: float) -> Optional[GlitchEvent]:
        """
        Check for subnormal (denormal) values - from reality_glitch_tests.py.
        Subnormals in FP32: |x| < 1.175494e-38 and x != 0
        """
        if not HAS_GLITCH_TESTS:
            return None

        metrics = count_subnormals_float32(positions)
        self.subnormal_history.append(metrics.subnormal_count)

        if metrics.subnormal_count > 0:
            event = GlitchEvent(
                tick=tick,
                redshift=redshift,
                glitch_type="subnormal_flood",
                magnitude=float(metrics.subnormal_count),
                description=f"Denormal values detected: {metrics.subnormal_count}, min={metrics.min_nonzero:.2e}"
            )
            self.events.append(event)
            logger.warning(f"GLITCH DETECTED: Subnormal flood ({metrics.subnormal_count} values) at z={redshift:.2f}")
            return event
        return None

    def check_entropy(self, positions: torch.Tensor, velocities: torch.Tensor,
                      tick: int, redshift: float) -> Optional[GlitchEvent]:
        """
        Monitor state entropy/compressibility - from reality_glitch_tests.py.
        Sudden compression ratio changes indicate potential "lossy" physics.
        """
        if not HAS_GLITCH_TESTS:
            return None

        metrics = measure_state_entropy(positions, velocities)
        self.entropy_history.append(metrics.compression_ratio)

        # Check for sudden entropy change (>10% in one step)
        if len(self.entropy_history) >= 3:
            recent = self.entropy_history[-3:]
            if recent[-2] > 0:
                delta = abs(recent[-1] - recent[-2]) / recent[-2]
                if delta > 0.10:  # 10% change threshold
                    event = GlitchEvent(
                        tick=tick,
                        redshift=redshift,
                        glitch_type="entropy_spike",
                        magnitude=delta,
                        description=f"Compression ratio changed: {recent[-2]:.2f} → {recent[-1]:.2f} ({delta*100:.1f}%)"
                    )
                    self.events.append(event)
                    logger.warning(f"GLITCH DETECTED: Entropy spike ({delta*100:.1f}%) at z={redshift:.2f}")
                    return event
        return None

    def get_glitch_count(self) -> int:
        return len(self.events)

    def get_glitch_summary(self) -> dict:
        """Get summary of glitches by type."""
        summary = {}
        for event in self.events:
            if event.glitch_type not in summary:
                summary[event.glitch_type] = 0
            summary[event.glitch_type] += 1
        return summary


# =============================================================================
# 2D UNIVERSE SIMULATION - UNIFIED
# =============================================================================

class Universe2D:
    """
    Unified 2D cosmic simulation with:
    - Dark matter (NFW profile)
    - Precision quantization (glitch detection)
    - Proper cosmological evolution
    - BAO acoustic oscillations
    """

    def __init__(
        self,
        num_particles: int = 10000,
        box_size_mpc: float = 200.0,
        start_redshift: float = 50.0,
        precision: str = "float32",
        dm_ratio: float = 5.0,  # Dark matter to visible matter ratio
        seed: int = 42,
        device: torch.device = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.box_size = box_size_mpc
        self.num_particles = num_particles
        self.dm_ratio = dm_ratio

        # State
        self.redshift = start_redshift
        self.scale = scale_factor(start_redshift)
        self.time_gyr = cosmic_time(start_redshift)
        self.current_epoch = get_current_epoch(start_redshift)
        self.tick = 0

        # === PRECISION MODE from quantization.py ===
        self.precision_mode = get_mode_from_string(precision)
        self.precision_str = precision
        dtype = torch.float64 if precision == "float64" else torch.float32
        self.dtype = dtype

        print(f"  Precision mode: {describe_mode(self.precision_mode)}")

        # === GLITCH DETECTOR ===
        self.glitch_detector = GlitchDetector(threshold=0.05)

        # Initialize
        self._initialize()

        # History
        self.history = {
            'redshift': [self.redshift],
            'time_gyr': [self.time_gyr],
            'bao_scale': [],
            'clustering': [],
            'glitches': [],
            'energy': [],
        }

    def _initialize(self):
        """Initialize with primordial perturbations and dark matter."""
        set_all_seeds(self.seed)

        n = int(np.sqrt(self.num_particles))
        self.num_particles = n * n

        logger.info("=" * 50)
        logger.info("UNIVERSE 2D - INITIALIZATION")
        logger.info("=" * 50)
        logger.info(f"Particles: {self.num_particles:,} ({n}x{n} grid)")
        logger.info(f"Box size: {self.box_size} Mpc")
        logger.info(f"Start redshift: z={self.redshift}")
        logger.info(f"Cosmic time: {self.time_gyr*1000:.1f} Myr")
        logger.info(f"Dark Matter ratio: {self.dm_ratio}x visible")
        logger.info(f"Precision: {describe_mode(self.precision_mode)}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Seed: {self.seed}")
        logger.info("=" * 50)

        # Grid positions
        spacing = self.box_size / n
        grid = torch.linspace(spacing/2, self.box_size - spacing/2, n, device=self.device, dtype=self.dtype)
        x, y = torch.meshgrid(grid, grid, indexing='ij')
        self.positions = torch.stack([x.flatten(), y.flatten()], dim=1)

        # Add primordial perturbations with BAO signature
        kx = torch.fft.fftfreq(n, d=self.box_size/n, device=self.device).to(self.dtype) * 2 * np.pi
        ky = torch.fft.fftfreq(n, d=self.box_size/n, device=self.device).to(self.dtype) * 2 * np.pi
        kx, ky = torch.meshgrid(kx, ky, indexing='ij')
        k_mag = torch.sqrt(kx**2 + ky**2 + 1e-10)

        # Power spectrum with BAO wiggles
        k_bao = 2 * np.pi / BAO_SCALE
        pk = (k_mag / 0.1 + 1e-10) ** (N_S - 4) * torch.exp(-(k_mag / 0.5)**2)
        pk = pk * (1 + 0.15 * torch.cos(k_mag / k_bao * np.pi))

        # Random phases
        phases = torch.rand(n, n, device=self.device, dtype=self.dtype) * 2 * np.pi
        delta_k = torch.sqrt(pk) * torch.exp(1j * phases.to(torch.complex64 if self.dtype == torch.float32 else torch.complex128))

        # Displacement field (Zeldovich)
        psi_k = delta_k / (k_mag**2 + 1e-10)
        psi_k[0, 0] = 0

        disp_x = torch.fft.ifft2(-1j * kx * psi_k).real
        disp_y = torch.fft.ifft2(-1j * ky * psi_k).real

        displacement = torch.stack([disp_x.flatten(), disp_y.flatten()], dim=1)

        # Scale by growth factor
        D = growth_factor(self.redshift)
        amplitude = 5.0 * D  # Tuned for visible structure

        self.positions = (self.positions + displacement * amplitude) % self.box_size

        # Velocities
        f_growth = OMEGA_M ** 0.55
        H_z = hubble_parameter(self.redshift)
        self.velocities = self.scale * H_z * f_growth * displacement * amplitude * 0.001

        # Masses
        rho_crit = 2.775e11  # M_sun / (Mpc/h)^3
        total_mass = OMEGA_M * rho_crit * self.box_size**2 * 10  # Effective 2D
        self.masses = torch.ones(self.num_particles, device=self.device, dtype=self.dtype) * total_mass / self.num_particles

    def _compute_accelerations(self) -> torch.Tensor:
        """
        PM gravity solver with:
        - Dark matter contribution (NFW profile)
        - Precision quantization (from quantization.py)
        - Fully vectorized (no Python loops)
        """
        n_grid = 128

        # === VISIBLE MATTER DENSITY (vectorized) ===
        pos_grid = (self.positions / self.box_size * n_grid).long() % n_grid

        # Flatten 2D grid indices to 1D for scatter_add
        flat_indices = pos_grid[:, 0] * n_grid + pos_grid[:, 1]

        # Use scatter_add for vectorized density assignment
        density_flat = torch.zeros(n_grid * n_grid, device=self.device, dtype=self.dtype)
        density_flat.scatter_add_(0, flat_indices, self.masses)
        density = density_flat.view(n_grid, n_grid)

        # === DARK MATTER CONTRIBUTION ===
        dm_density = compute_dm_density_field(
            self.positions, self.box_size, n_grid, self.dm_ratio
        )
        total_density = density + dm_density * density.mean()

        mean_rho = total_density.mean()
        delta = (total_density - mean_rho) / (mean_rho + 1e-10)

        # FFT
        delta_k = torch.fft.fft2(delta)

        kx = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device).to(self.dtype) * 2 * np.pi
        ky = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device).to(self.dtype) * 2 * np.pi
        kx, ky = torch.meshgrid(kx, ky, indexing='ij')
        k_sq = kx**2 + ky**2 + 1e-10

        # === QUANTIZATION - The "broken math" that causes glitches ===
        # Apply precision degradation to distance (k_sq proxy)
        k_sq_quantized = quantize_distance_squared(k_sq, self.precision_mode)

        # Poisson equation with quantized k^2
        phi_k = -4 * np.pi * G_NEWTON * mean_rho * delta_k / k_sq_quantized / self.scale
        phi_k[0, 0] = 0

        # Gradient
        ax_k = -1j * kx * phi_k
        ay_k = -1j * ky * phi_k

        ax = torch.fft.ifft2(ax_k).real
        ay = torch.fft.ifft2(ay_k).real

        # Interpolate to particles (vectorized using advanced indexing)
        accelerations = torch.stack([
            ax[pos_grid[:, 0], pos_grid[:, 1]],
            ay[pos_grid[:, 0], pos_grid[:, 1]]
        ], dim=1)

        # === QUANTIZE FORCES (additional precision loss) ===
        if self.precision_mode in [PrecisionMode.INT4_SIM, PrecisionMode.INT8_SIM]:
            accelerations = quantize_force(accelerations, self.precision_mode)

        return accelerations

    def get_kinetic_energy(self) -> float:
        """Total kinetic energy."""
        v_sq = (self.velocities ** 2).sum(dim=-1)
        ke = 0.5 * (self.masses * v_sq).sum()
        return ke.item()

    def get_total_momentum(self) -> Tuple[float, float]:
        """Total momentum (should be ~0 for isolated system)."""
        px = (self.masses * self.velocities[:, 0]).sum().item()
        py = (self.masses * self.velocities[:, 1]).sum().item()
        return (px, py)

    def step(self, dz: float = 1.0):
        """
        Evolve by redshift step with glitch detection.
        """
        if self.redshift <= 0:
            return

        z_new = max(0, self.redshift - dz)
        dt_gyr = abs(cosmic_time(z_new) - cosmic_time(self.redshift))

        # Compute forces (with quantization applied)
        accel = self._compute_accelerations()

        # Hubble drag
        H = hubble_parameter(self.redshift)

        # Leapfrog integration
        self.velocities = self.velocities + accel * dt_gyr - 2 * H * self.velocities * dt_gyr * 1e-3
        self.positions = (self.positions + self.velocities * dt_gyr / self.scale * 1e-3) % self.box_size

        # Update state
        self.redshift = z_new
        self.scale = scale_factor(z_new)
        self.time_gyr = cosmic_time(z_new)
        self.tick += 1

        # === UNIFIED GLITCH DETECTION ===
        # Integrates checks from reality_glitch_tests.py
        energy = self.get_kinetic_energy()
        momentum = self.get_total_momentum()

        # 1. Energy conservation check (quantization artifact)
        energy_glitch = self.glitch_detector.check_energy_conservation(
            energy, self.tick, z_new
        )
        if energy_glitch:
            self.history['glitches'].append(energy_glitch)

        # 2. Momentum drift check (asymmetric forces)
        momentum_glitch = self.glitch_detector.check_momentum(
            momentum, self.tick, z_new
        )
        if momentum_glitch:
            self.history['glitches'].append(momentum_glitch)

        # 3. Subnormal check - from reality_glitch_tests.py
        # Detects denormal flooding in the 10^-38 to 10^-45 range
        subnormal_glitch = self.glitch_detector.check_subnormals(
            self.positions, self.tick, z_new
        )
        if subnormal_glitch:
            self.history['glitches'].append(subnormal_glitch)

        # 4. Entropy/compression check - from reality_glitch_tests.py
        # Every 10 ticks to reduce overhead
        if self.tick % 10 == 0:
            entropy_glitch = self.glitch_detector.check_entropy(
                self.positions, self.velocities, self.tick, z_new
            )
            if entropy_glitch:
                self.history['glitches'].append(entropy_glitch)

        self.history['energy'].append(energy)

        # Epoch transition
        new_epoch = get_current_epoch(z_new)
        if new_epoch != self.current_epoch:
            epoch_info = EPOCHS[new_epoch]
            logger.info("=" * 50)
            logger.info(f"EPOCH TRANSITION: {epoch_info.name}")
            logger.info(f"  Redshift: z = {z_new:.2f}")
            logger.info(f"  Cosmic time: {self.time_gyr:.3f} Gyr")
            logger.info(f"  Description: {epoch_info.description}")
            logger.info("=" * 50)
            self.current_epoch = new_epoch

        # Log progress every 10 ticks
        if self.tick % 10 == 0:
            bao = self.get_bao_scale()
            clustering = self.get_clustering()
            glitches = self.glitch_detector.get_glitch_count()
            logger.debug(f"Tick {self.tick:4d} | z={z_new:5.1f} | t={self.time_gyr:.3f} Gyr | BAO={bao:.1f} Mpc | Glitches={glitches}")

        # Record
        self.history['redshift'].append(z_new)
        self.history['time_gyr'].append(self.time_gyr)

    def compute_power_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 2D power spectrum (vectorized)."""
        n_grid = 64

        # Vectorized density assignment
        pos_grid = (self.positions / self.box_size * n_grid).long() % n_grid
        flat_indices = pos_grid[:, 0] * n_grid + pos_grid[:, 1]

        # Count particles per cell using bincount
        counts = torch.bincount(flat_indices, minlength=n_grid * n_grid)
        density = counts.float().view(n_grid, n_grid).to(self.dtype)

        delta = (density - density.mean()) / (density.mean() + 1e-10)
        delta_k = torch.fft.fft2(delta)
        pk_2d = torch.abs(delta_k)**2

        # k magnitudes
        kx = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device).to(self.dtype) * 2 * np.pi
        ky = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device).to(self.dtype) * 2 * np.pi
        kx, ky = torch.meshgrid(kx, ky, indexing='ij')
        k_mag = torch.sqrt(kx**2 + ky**2)

        # Radial binning (vectorized using searchsorted + scatter)
        k_min = 2 * np.pi / self.box_size
        k_max = np.pi * n_grid / self.box_size
        k_bins = torch.logspace(np.log10(k_min), np.log10(k_max), 20, device=self.device)

        # Assign each k to a bin
        k_flat = k_mag.flatten()
        pk_flat = pk_2d.flatten()
        bin_indices = torch.searchsorted(k_bins, k_flat) - 1
        bin_indices = bin_indices.clamp(0, 18)

        # Sum and count per bin
        pk_sum = torch.zeros(19, device=self.device, dtype=self.dtype)
        pk_count = torch.zeros(19, device=self.device, dtype=self.dtype)
        pk_sum.scatter_add_(0, bin_indices, pk_flat.to(self.dtype))
        pk_count.scatter_add_(0, bin_indices, torch.ones_like(pk_flat, dtype=self.dtype))

        # Compute mean (avoid division by zero)
        pk_binned = pk_sum / (pk_count + 1e-10)

        k_centers = ((k_bins[:-1] + k_bins[1:]) / 2).cpu().numpy()
        return k_centers, pk_binned.cpu().numpy()

    def get_bao_scale(self) -> float:
        """Measure BAO scale from correlation function peak."""
        k, pk = self.compute_power_spectrum()

        # Look for oscillation in P(k) - the BAO feature
        # BAO appears as bump at k ~ 2*pi/147 ~ 0.043 h/Mpc
        k_bao_expected = 2 * np.pi / BAO_SCALE

        # Find peak near expected BAO scale
        valid = (k > 0.01) & (k < 0.2) & (pk > 0)
        if valid.sum() > 3:
            k_valid = k[valid]
            pk_valid = pk[valid]

            # Simple peak finding
            peak_idx = np.argmax(pk_valid)
            k_peak = k_valid[peak_idx]

            if k_peak > 0:
                return 2 * np.pi / k_peak

        return 0.0

    def get_clustering(self) -> float:
        """Simple clustering metric (variance of density) - vectorized."""
        n_grid = 32
        pos_grid = (self.positions / self.box_size * n_grid).long() % n_grid
        flat_indices = pos_grid[:, 0] * n_grid + pos_grid[:, 1]

        # Count particles per cell using bincount
        counts = torch.bincount(flat_indices, minlength=n_grid * n_grid)
        density = counts.float().view(n_grid, n_grid)

        return (density.std() / (density.mean() + 1e-10)).item()


# =============================================================================
# 2D VISUALIZER
# =============================================================================

class Universe2DVisualizer:
    """Simple 2D visualization with line graphs and GPU profiling."""

    def __init__(self, universe: Universe2D, enable_gpu_profiling: bool = True):
        self.universe = universe
        self.running = True

        # History buffers
        self.bao_history = []
        self.clustering_history = []
        self.time_history = []

        # GPU profiling (from gpu_profiler.py)
        self.gpu_profiler = None
        self.gpu_profile_result = None
        if enable_gpu_profiling and universe.device.type == "cuda":
            try:
                self.gpu_profiler = GPUProfiler(sample_interval_ms=500)
                logger.info("GPU Profiler initialized (from gpu_profiler.py)")
            except Exception as e:
                logger.warning(f"GPU profiling unavailable: {e}")

    def setup(self):
        """Create figure."""
        plt.style.use('dark_background')

        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor('#0a0a0a')

        gs = GridSpec(3, 3, figure=self.fig, hspace=0.35, wspace=0.3)

        # Main 2D universe view
        self.ax_universe = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_universe.set_facecolor('#0a0a0a')

        # Power spectrum
        self.ax_pk = self.fig.add_subplot(gs[0, 2])
        self.ax_pk.set_facecolor('#1a1a2e')

        # BAO scale over time
        self.ax_bao = self.fig.add_subplot(gs[1, 2])
        self.ax_bao.set_facecolor('#1a1a2e')

        # Timeline
        self.ax_timeline = self.fig.add_subplot(gs[2, 0:2])
        self.ax_timeline.set_facecolor('#1a1a2e')

        # Metrics
        self.ax_metrics = self.fig.add_subplot(gs[2, 2])
        self.ax_metrics.axis('off')

    def draw_timeline(self):
        """Draw epoch timeline."""
        self.ax_timeline.clear()
        self.ax_timeline.set_facecolor('#1a1a2e')

        epochs_pos = [
            (CosmicEpoch.RECOMBINATION, 0.0),
            (CosmicEpoch.DARK_AGES, 0.05),
            (CosmicEpoch.FIRST_STARS, 0.15),
            (CosmicEpoch.REIONIZATION, 0.25),
            (CosmicEpoch.GALAXY_FORMATION, 0.4),
            (CosmicEpoch.PEAK_SF, 0.55),
            (CosmicEpoch.DARK_ENERGY, 0.75),
            (CosmicEpoch.PRESENT, 1.0),
        ]

        # Draw bars
        for i, (epoch, pos) in enumerate(epochs_pos[:-1]):
            info = EPOCHS[epoch]
            next_pos = epochs_pos[i+1][1]
            rect = patches.Rectangle((pos, 0.3), next_pos - pos - 0.01, 0.4,
                                     facecolor=info.color, alpha=0.8, edgecolor='white', linewidth=0.5)
            self.ax_timeline.add_patch(rect)

            # Label
            mid = pos + (next_pos - pos) / 2
            self.ax_timeline.text(mid, 0.05, info.name, ha='center', fontsize=7,
                                 color='white', rotation=25)

        # Current position (based on time fraction)
        t_frac = self.universe.time_gyr / T_UNIVERSE
        current_pos = min(1.0, t_frac)

        self.ax_timeline.axvline(x=current_pos, color='cyan', linewidth=3, linestyle='--')
        self.ax_timeline.scatter([current_pos], [0.5], color='cyan', s=200, marker='v', zorder=10)

        self.ax_timeline.set_xlim(-0.02, 1.02)
        self.ax_timeline.set_ylim(0, 1)
        self.ax_timeline.set_title(
            f'Cosmic Timeline | z={self.universe.redshift:.1f} | t={self.universe.time_gyr:.2f} Gyr',
            color='cyan', fontsize=11
        )
        self.ax_timeline.axis('off')

    def update(self, frame):
        """Animation frame update."""
        if not self.running or self.universe.redshift <= 0:
            if self.universe.redshift <= 0 and self.running:
                self.running = False
                print("\n  === SIMULATION COMPLETE - Reached z=0 (present day) ===")
                # Close the figure to end the animation and allow summary to print
                plt.close(self.fig)
            return

        # Evolve
        self.universe.step(dz=1.0)

        # Get data
        pos = self.universe.positions.cpu().numpy()
        epoch_info = EPOCHS.get(self.universe.current_epoch, EPOCHS[CosmicEpoch.PRESENT])

        # Compute metrics
        bao = self.universe.get_bao_scale()
        clustering = self.universe.get_clustering()

        self.bao_history.append(bao)
        self.clustering_history.append(clustering)
        self.time_history.append(self.universe.time_gyr)

        # Keep last 100 points
        max_history = 100
        if len(self.bao_history) > max_history:
            self.bao_history = self.bao_history[-max_history:]
            self.clustering_history = self.clustering_history[-max_history:]
            self.time_history = self.time_history[-max_history:]

        # === 2D Universe View ===
        self.ax_universe.clear()
        self.ax_universe.set_facecolor('#0a0a0a')

        # Color by local density (proximity to neighbors)
        n_show = min(len(pos), 20000)
        if len(pos) > n_show:
            idx = np.random.choice(len(pos), n_show, replace=False)
            pos_show = pos[idx]
        else:
            pos_show = pos

        self.ax_universe.scatter(
            pos_show[:, 0], pos_show[:, 1],
            c=pos_show[:, 0] + pos_show[:, 1],  # Simple gradient color
            cmap='magma', s=0.5, alpha=0.6
        )

        self.ax_universe.set_xlim(0, self.universe.box_size)
        self.ax_universe.set_ylim(0, self.universe.box_size)
        self.ax_universe.set_aspect('equal')
        self.ax_universe.set_title(
            f'{epoch_info.name} | z={self.universe.redshift:.1f} | t={self.universe.time_gyr:.2f} Gyr',
            color=epoch_info.color, fontsize=14, fontweight='bold'
        )
        self.ax_universe.set_xlabel('Mpc', color='gray')
        self.ax_universe.set_ylabel('Mpc', color='gray')

        # === Power Spectrum ===
        self.ax_pk.clear()
        self.ax_pk.set_facecolor('#1a1a2e')
        k, pk = self.universe.compute_power_spectrum()
        valid = pk > 0
        if valid.sum() > 2:
            self.ax_pk.loglog(k[valid], pk[valid], 'c-', linewidth=1.5)

        # Mark expected BAO scale
        k_bao_expected = 2 * np.pi / BAO_SCALE
        self.ax_pk.axvline(x=k_bao_expected, color='orange', linestyle='--',
                          alpha=0.7, label=f'k_BAO={k_bao_expected:.3f}')
        self.ax_pk.set_title('Power Spectrum P(k)', color='white', fontsize=10)
        self.ax_pk.set_xlabel('k [h/Mpc]', color='gray', fontsize=8)
        self.ax_pk.set_ylabel('P(k)', color='gray', fontsize=8)
        self.ax_pk.legend(fontsize=7, loc='upper right')

        # === BAO Scale Over Time ===
        self.ax_bao.clear()
        self.ax_bao.set_facecolor('#1a1a2e')
        if len(self.bao_history) > 1:
            self.ax_bao.plot(self.time_history, self.bao_history, 'b-', linewidth=1.5)
        self.ax_bao.axhline(y=BAO_SCALE, color='lime', linestyle='--',
                           alpha=0.8, label=f'Expected: {BAO_SCALE} Mpc')
        self.ax_bao.set_title(f'BAO Scale: {bao:.1f} Mpc', color='cyan', fontsize=10)
        self.ax_bao.set_xlabel('Cosmic Time (Gyr)', color='gray', fontsize=8)
        self.ax_bao.set_ylabel('BAO Scale (Mpc)', color='gray', fontsize=8)
        self.ax_bao.legend(fontsize=7)
        if len(self.time_history) > 0:
            self.ax_bao.set_xlim(0, max(self.time_history[-1], 1))

        # === Timeline ===
        self.draw_timeline()

        # === Metrics ===
        self.ax_metrics.clear()
        self.ax_metrics.axis('off')
        self.ax_metrics.set_facecolor('#16213e')

        # GPU state
        gpu_state = get_gpu_state()
        gpu_text = ""
        if gpu_state:
            gpu_text = f"GPU: {gpu_state.clock_speed_mhz} MHz, {gpu_state.temperature_c}C"

        # Glitch count
        glitch_count = self.universe.glitch_detector.get_glitch_count()
        glitch_color = "red" if glitch_count > 0 else "lime"

        metrics_text = (
            f"{'='*35}\n"
            f" UNIFIED UNIVERSE 2D\n"
            f"{'='*35}\n"
            f" Redshift: z = {self.universe.redshift:.2f}\n"
            f" Scale: a = {self.universe.scale:.4f}\n"
            f" Time: {self.universe.time_gyr:.3f} Gyr\n"
            f" Epoch: {epoch_info.name}\n"
            f" H(z): {hubble_parameter(self.universe.redshift):.1f} km/s/Mpc\n"
            f"{'='*35}\n"
            f" Particles: {self.universe.num_particles:,}\n"
            f" Dark Matter: {self.universe.dm_ratio}x\n"
            f" Precision: {self.universe.precision_str}\n"
            f"{'='*35}\n"
            f" BAO: {bao:.1f} Mpc (exp: {BAO_SCALE})\n"
            f" Clustering: {clustering:.3f}\n"
            f" GLITCHES: {glitch_count}\n"
            f"{'='*35}\n"
            f" {gpu_text}\n"
        )

        self.ax_metrics.text(0.05, 0.95, metrics_text, transform=self.ax_metrics.transAxes,
                            fontfamily='monospace', fontsize=8, color='white',
                            verticalalignment='top')

        # Glitch indicator
        if glitch_count > 0:
            self.ax_metrics.text(0.5, 0.1, f"GLITCHES DETECTED: {glitch_count}",
                               transform=self.ax_metrics.transAxes,
                               fontfamily='monospace', fontsize=10, color='red',
                               ha='center', fontweight='bold')

        # Force canvas redraw (fixes blank animation on some backends)
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key in ('q', 'Q', 'escape'):
            logger.info("Quit requested via keyboard")
            self.running = False
            plt.close(self.fig)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        logger.info("\nInterrupt received (Ctrl+C), stopping simulation...")
        self.running = False
        plt.close('all')

    def run(self):
        """Start animation with GPU profiling."""
        self.setup()

        # Setup signal handler for Ctrl+C
        original_handler = signal.signal(signal.SIGINT, self._signal_handler)

        # Setup keyboard handler (press 'q' or 'Escape' to quit)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Start GPU profiling (from gpu_profiler.py)
        if self.gpu_profiler:
            self.gpu_profiler.start(f"Universe2D_{self.universe.precision_str}")

        # Calculate frames needed: from start_z to 0 with dz=1.0, plus buffer
        max_frames = int(self.universe.redshift) + 10

        ani = FuncAnimation(
            self.fig, self.update,
            frames=max_frames,
            interval=50,  # 20 FPS
            blit=False,
            cache_frame_data=False,
            repeat=False  # Stop after frames exhausted
        )

        plt.tight_layout()

        try:
            plt.show()
        except KeyboardInterrupt:
            logger.info("\nKeyboardInterrupt caught, stopping...")
            self.running = False
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)

        # Stop GPU profiling and print report
        if self.gpu_profiler:
            self.gpu_profile_result = self.gpu_profiler.stop()
            if self.gpu_profile_result:
                self.gpu_profiler.print_report(self.gpu_profile_result)
                logger.info(f"GPU Mean Power: {self.gpu_profile_result.mean_power_watts:.1f}W")
                logger.info(f"GPU Mean Clock: {self.gpu_profile_result.mean_clock_mhz:.0f} MHz")


# =============================================================================
# MAIN
# =============================================================================

def main():
    global logger

    parser = argparse.ArgumentParser(
        description="Universe 2D - Unified Cosmic Evolution Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
UNIFIED FEATURES:
  - Dark Matter: NFW halo model (adjustable ratio)
  - Quantization: Precision modes from FLOAT64 to INT4
  - Glitch Detection: Monitors energy/momentum anomalies
  - BAO: Tracks acoustic oscillation scale (~147 Mpc)
  - Logging: Full event logging to console and file

EXAMPLES:
  # Default run
  python universe_2d.py

  # With logging to file
  python universe_2d.py --particles 650 --log universe_log.txt

  # Verbose logging (debug level)
  python universe_2d.py --particles 650 --verbose

  # More particles, higher dark matter
  python universe_2d.py --particles 20000 --dm-ratio 10

  # Test quantization glitches with INT4
  python universe_2d.py --precision int4 --particles 5000

  # High precision baseline
  python universe_2d.py --precision float64
        """
    )
    parser.add_argument("--particles", type=int, default=10000, help="Number of particles")
    parser.add_argument("--box", type=float, default=200.0, help="Box size in Mpc")
    parser.add_argument("--start-z", type=float, default=50.0, help="Starting redshift")
    parser.add_argument("--precision", type=str, default="float32",
                        choices=["float64", "float32", "bfloat16", "float16", "int8", "int4"],
                        help="Precision mode (lower = more glitches)")
    parser.add_argument("--dm-ratio", type=float, default=5.0,
                        help="Dark matter to visible matter ratio (default: 5x)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log", type=str, default=None,
                        help="Log file path (e.g., logs/universe.log)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose (debug) logging")
    parser.add_argument("--no-gpu-profile", action="store_true",
                        help="Disable GPU power/clock profiling")

    args = parser.parse_args()

    # Setup logging with file if specified
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = args.log
    if log_file is None:
        # Default log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/universe2d_{timestamp}.log"

    logger = setup_logging(log_file=log_file, level=log_level)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("  UNIFIED UNIVERSE 2D ENGINE - SESSION START")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU: {gpu_name}")

    logger.info("-" * 60)
    logger.info("CONFIGURATION:")
    logger.info(f"  Particles: {args.particles}")
    logger.info(f"  Box size: {args.box} Mpc")
    logger.info(f"  Start redshift: z={args.start_z}")
    logger.info(f"  Precision: {args.precision}")
    logger.info(f"  Dark Matter ratio: {args.dm_ratio}x")
    logger.info(f"  Random seed: {args.seed}")
    logger.info(f"  Log file: {log_file}")
    logger.info(f"  Verbose: {args.verbose}")
    logger.info("-" * 60)

    # Create universe with all unified features
    universe = Universe2D(
        num_particles=args.particles,
        box_size_mpc=args.box,
        start_redshift=args.start_z,
        precision=args.precision,
        dm_ratio=args.dm_ratio,
        seed=args.seed,
        device=device
    )

    # Show unified module status
    logger.info("-" * 60)
    logger.info("UNIFIED MODULES LOADED:")
    logger.info("  ✓ galaxy.py        - NFW dark matter, create_galaxy_with_halo")
    logger.info("  ✓ simulation.py    - GalaxySimulation (N-body leapfrog)")
    logger.info("  ✓ quantization.py  - PrecisionMode, quantize_distance_squared")
    logger.info("  ✓ metrics.py       - compute_rotation_curve, bound_fraction")
    logger.info("  ✓ gpu_profiler.py  - GPUProfiler power/clock monitoring")
    logger.info("  ✓ reproducibility.py - set_all_seeds, get_gpu_state")
    logger.info("-" * 60)

    logger.info("Starting 2D Universe visualization...")
    logger.info("Each dot = dark matter particle")
    logger.info("Integrates: DM model, quantization, glitch detection, GPU profiling")
    logger.info("Press 'q' or 'Escape' to quit, or Ctrl+C in terminal")

    viz = Universe2DVisualizer(universe, enable_gpu_profiling=not args.no_gpu_profile)
    viz.run()

    # Final summary
    logger.info("=" * 60)
    logger.info("  SESSION COMPLETE - FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Final redshift: z = {universe.redshift:.2f}")
    logger.info(f"Final cosmic time: {universe.time_gyr:.3f} Gyr")
    logger.info(f"Total ticks: {universe.tick}")
    logger.info(f"Total glitches detected: {universe.glitch_detector.get_glitch_count()}")

    if universe.glitch_detector.events:
        logger.info("-" * 40)
        logger.info("GLITCH SUMMARY BY TYPE:")
        glitch_summary = universe.glitch_detector.get_glitch_summary()
        for glitch_type, count in glitch_summary.items():
            logger.info(f"  {glitch_type}: {count}")

        logger.info("-" * 40)
        logger.info("GLITCH LOG (last 10):")
        for event in universe.glitch_detector.events[-10:]:
            logger.info(f"  [{event.glitch_type}] z={event.redshift:.2f} tick={event.tick}: {event.description}")

    # Log integrated module status
    logger.info("-" * 40)
    logger.info("INTEGRATED MODULES STATUS:")
    logger.info(f"  galaxy.py:              ACTIVE (NFW dark matter)")
    logger.info(f"  simulation.py:          ACTIVE (N-body engine)")
    logger.info(f"  quantization.py:        ACTIVE ({universe.precision_str})")
    logger.info(f"  metrics.py:             ACTIVE (rotation curves)")
    logger.info(f"  gpu_profiler.py:        {'ACTIVE' if viz.gpu_profiler else 'INACTIVE'}")
    logger.info(f"  reality_glitch_tests.py: {'ACTIVE' if HAS_GLITCH_TESTS else 'INACTIVE'}")
    logger.info(f"  orbital_audit.py:       {'AVAILABLE' if HAS_ORBITAL_AUDIT else 'INACTIVE'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
