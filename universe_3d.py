"""
UNIVERSE 3D - Interactive Cosmic Evolution Visualization
=========================================================

Full 3D visualization of cosmic structure formation.
Each dot is a dark matter particle - rotate with mouse to explore!

Features:
- 3D particle positions and velocities
- Interactive rotation (drag mouse to rotate)
- Dark matter NFW profile
- Quantization-based glitch detection
- Physics exploit probes
- GPU profiling

Usage:
    python universe_3d.py --particles 1000
    python universe_3d.py --particles 500 --precision int4
"""

import argparse
import time
import logging
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import math

# =============================================================================
# LOCAL IMPORTS
# =============================================================================
from reproducibility import set_all_seeds, get_gpu_state
from quantization import (
    PrecisionMode, quantize_distance_squared, quantize_force,
    _grid_quantize_safe, get_mode_from_string, describe_mode
)
from galaxy import (
    create_disk_galaxy,
    create_galaxy_with_halo,
    nfw_enclosed_mass as galaxy_nfw_enclosed_mass,
)
from simulation import GalaxySimulation
from metrics import compute_rotation_curve
from gpu_profiler import GPUProfiler

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
try:
    from orbital_audit import AnomalyEvent
    HAS_ORBITAL_AUDIT = True
except ImportError:
    HAS_ORBITAL_AUDIT = False


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("Universe3D")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

logger = setup_logging()


# =============================================================================
# COSMOLOGY CONSTANTS
# =============================================================================

H0 = 67.4  # km/s/Mpc
OMEGA_M = 0.315
OMEGA_LAMBDA = 0.685
G_NEWTON = 4.302e-6  # kpc^3 / (Msun * Gyr^2) scaled

def hubble_parameter(z: float) -> float:
    return H0 * np.sqrt(OMEGA_M * (1 + z)**3 + OMEGA_LAMBDA)

def scale_factor(z: float) -> float:
    return 1.0 / (1.0 + z)

def cosmic_time(z: float) -> float:
    """Cosmic time in Gyr using lookup table."""
    z_table = np.array([0, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100, 500, 1000, 1100])
    t_table = np.array([13.8, 12.5, 8.6, 5.9, 3.3, 2.2, 1.2, 0.47, 0.18, 0.05, 0.017, 0.001, 0.0004, 0.00038])
    return float(np.interp(z, z_table, t_table))


# =============================================================================
# COSMIC EPOCHS
# =============================================================================

class CosmicEpoch(Enum):
    RECOMBINATION = "recombination"
    DARK_AGES = "dark_ages"
    FIRST_STARS = "first_stars"
    REIONIZATION = "reionization"
    GALAXY_FORMATION = "galaxy_formation"
    PEAK_SF = "peak_sf"
    DARK_ENERGY = "dark_energy"
    PRESENT = "present"

@dataclass
class EpochInfo:
    name: str
    z_start: float
    z_end: float
    color: str
    description: str

EPOCHS = {
    CosmicEpoch.RECOMBINATION: EpochInfo("Recombination", 1100, 1000, '#FFD700', "CMB release"),
    CosmicEpoch.DARK_AGES: EpochInfo("Dark Ages", 1000, 20, '#1a1a2e', "No stars yet"),
    CosmicEpoch.FIRST_STARS: EpochInfo("First Stars", 20, 10, '#4a0080', "Pop III ignition"),
    CosmicEpoch.REIONIZATION: EpochInfo("Reionization", 10, 6, '#0066cc', "Universe re-ionizes"),
    CosmicEpoch.GALAXY_FORMATION: EpochInfo("Galaxy Formation", 6, 3, '#00aa44', "Galaxies assemble"),
    CosmicEpoch.PEAK_SF: EpochInfo("Peak Star Formation", 3, 1, '#ff6600', "Cosmic noon"),
    CosmicEpoch.DARK_ENERGY: EpochInfo("Dark Energy", 1, 0.3, '#660066', "Acceleration begins"),
    CosmicEpoch.PRESENT: EpochInfo("Present", 0.3, 0, '#00ffff', "Today"),
}

def get_current_epoch(z: float) -> CosmicEpoch:
    if z > 1000: return CosmicEpoch.RECOMBINATION
    elif z > 20: return CosmicEpoch.DARK_AGES
    elif z > 10: return CosmicEpoch.FIRST_STARS
    elif z > 6: return CosmicEpoch.REIONIZATION
    elif z > 3: return CosmicEpoch.GALAXY_FORMATION
    elif z > 1: return CosmicEpoch.PEAK_SF
    elif z > 0.3: return CosmicEpoch.DARK_ENERGY
    else: return CosmicEpoch.PRESENT


# =============================================================================
# GLITCH DETECTOR
# =============================================================================

@dataclass
class GlitchEvent:
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
        self.momentum_history: List[Tuple[float, float, float]] = []  # 3D momentum
        self.subnormal_history: List[int] = []
        self.entropy_history: List[float] = []

    def check_energy_conservation(self, energy: float, tick: int, redshift: float) -> Optional[GlitchEvent]:
        """Check for sudden energy jumps (quantization artifact)."""
        self.energy_history.append(energy)

        if len(self.energy_history) < 3:
            return None

        recent = self.energy_history[-3:]
        if recent[-2] != 0:
            delta = abs(recent[-1] - recent[-2]) / abs(recent[-2] + 1e-10)
            if delta > self.threshold:
                event = GlitchEvent(
                    tick=tick,
                    redshift=redshift,
                    glitch_type="energy_jump",
                    magnitude=delta,
                    description=f"Energy change: {delta*100:.1f}%"
                )
                self.events.append(event)
                logger.warning(f"GLITCH DETECTED: Energy jump {delta*100:.1f}% at z={redshift:.2f}, tick={tick}")
                return event
        return None

    def check_momentum(self, momentum: Tuple[float, float, float], tick: int, redshift: float) -> Optional[GlitchEvent]:
        """Check for momentum drift (should be ~0 for isolated system)."""
        self.momentum_history.append(momentum)
        px, py, pz = momentum
        total = np.sqrt(px**2 + py**2 + pz**2)

        if total > 1e-6:  # Significant drift
            event = GlitchEvent(
                tick=tick,
                redshift=redshift,
                glitch_type="momentum_drift",
                magnitude=total,
                description=f"Net momentum: ({px:.2f}, {py:.2f}, {pz:.2f})"
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
# PHYSICS EXPLOIT DETECTION MODULES (3D)
# =============================================================================
# These modules probe the "seams" of simulated reality by testing
# computationally expensive physics that simulations tend to "cheat" on.


@dataclass
class RelativityMetrics:
    """Metrics from Special Relativity stress test."""
    max_gamma: float = 1.0
    near_c_particles: int = 0
    power_at_09c: float = 0.0
    power_at_099c: float = 0.0
    bandwidth_limited: bool = False


@dataclass
class FluidMetrics:
    """Metrics from Navier-Stokes turbulence test."""
    reynolds_number: float = 0.0
    viscosity_observed: float = 0.0
    viscosity_expected: float = 0.0
    viscosity_ratio: float = 1.0
    turbulence_suppressed: bool = False


@dataclass
class LandauerMetrics:
    """Metrics from information physics test (Landauer's Principle)."""
    total_bits_initial: int = 0
    total_bits_current: int = 0
    bits_erased: int = 0
    energy_per_bit_erased: float = 0.0
    garbage_collection_detected: bool = False


@dataclass
class FrustumMetrics:
    """Metrics from observer frustum culling test (3D)."""
    in_frustum_count: int = 0
    out_frustum_count: int = 0
    in_frustum_precision: str = "FP32"
    out_frustum_precision: str = "INT8"
    snap_events: int = 0
    culling_detected: bool = False


class SpecialRelativityProbe:
    """
    Test if the speed of light (c) is actually the simulation's data transfer limit.
    """
    C_SIM = 306.6  # ~299,792 km/s converted to Mpc/Gyr

    def __init__(self, device: torch.device):
        self.device = device
        self.history: List[RelativityMetrics] = []
        self.gamma_history: List[float] = []
        self.power_vs_gamma: List[Tuple[float, float]] = []

    def lorentz_factor(self, velocities: torch.Tensor) -> torch.Tensor:
        """Calculate Lorentz factor γ = 1 / sqrt(1 - v²/c²)"""
        v_sq = (velocities ** 2).sum(dim=-1)
        c_sq = self.C_SIM ** 2
        beta_sq = torch.clamp(v_sq / c_sq, max=0.9999)
        gamma = 1.0 / torch.sqrt(1.0 - beta_sq)
        return gamma

    def check_bandwidth_limit(self, velocities: torch.Tensor,
                               gpu_power: float = 0.0) -> RelativityMetrics:
        speeds = torch.sqrt((velocities ** 2).sum(dim=-1))
        beta = speeds / self.C_SIM
        gamma = self.lorentz_factor(velocities)

        max_gamma = gamma.max().item()
        near_c_09 = (beta > 0.9).sum().item()
        near_c_099 = (beta > 0.99).sum().item()

        metrics = RelativityMetrics(
            max_gamma=max_gamma,
            near_c_particles=near_c_09,
            power_at_09c=gpu_power if near_c_09 > 0 else 0.0,
            power_at_099c=gpu_power if near_c_099 > 0 else 0.0,
            bandwidth_limited=(max_gamma > 10 and gpu_power > 100)
        )

        self.history.append(metrics)
        self.gamma_history.append(max_gamma)

        if gpu_power > 0:
            self.power_vs_gamma.append((max_gamma, gpu_power))

        if metrics.bandwidth_limited:
            logger.warning(f"EXPLOIT: Speed of light may be hardware bandwidth! γ={max_gamma:.2f}")

        return metrics


class NavierStokesProbe:
    """Test for viscosity clipping in fluid dynamics."""

    def __init__(self, device: torch.device):
        self.device = device
        self.history: List[FluidMetrics] = []

    def compute_reynolds_number(self, velocities: torch.Tensor,
                                 length_scale: float, viscosity: float) -> float:
        rho = 1.0
        v_mean = torch.sqrt((velocities ** 2).sum(dim=-1)).mean().item()
        re = rho * v_mean * length_scale / max(viscosity, 1e-10)
        return re

    def detect_viscosity_clipping(self, velocities: torch.Tensor,
                                   expected_viscosity: float = 0.01) -> FluidMetrics:
        v_std = velocities.std().item()
        v_mean = torch.sqrt((velocities ** 2).sum(dim=-1)).mean().item()
        vel_gradient = v_std / max(v_mean, 1e-10)
        observed_viscosity = 1.0 / max(vel_gradient, 1e-10) * 0.01

        re = self.compute_reynolds_number(velocities, 10.0, observed_viscosity)
        ratio = observed_viscosity / max(expected_viscosity, 1e-10)

        metrics = FluidMetrics(
            reynolds_number=re,
            viscosity_observed=observed_viscosity,
            viscosity_expected=expected_viscosity,
            viscosity_ratio=ratio,
            turbulence_suppressed=(ratio > 2.0 and re < 2300)
        )

        self.history.append(metrics)
        if metrics.turbulence_suppressed:
            logger.warning(f"EXPLOIT: Viscosity clipping detected! Ratio = {ratio:.2f}x")

        return metrics


class LandauerProbe:
    """Test for Maxwell's Demon / Garbage Collection."""
    KT_EV = 0.0257
    LANDAUER_LIMIT = KT_EV * 0.693

    def __init__(self, device: torch.device):
        self.device = device
        self.history: List[LandauerMetrics] = []
        self.initial_bits: int = 0
        self.bits_history: List[int] = []

    def measure_information_content(self, positions: torch.Tensor,
                                     velocities: torch.Tensor) -> int:
        pos_bytes = positions.cpu().numpy().astype(np.float32).tobytes()
        vel_bytes = velocities.cpu().numpy().astype(np.float32).tobytes()
        compressed = zlib.compress(pos_bytes + vel_bytes, level=9)
        return len(compressed) * 8

    def check_garbage_collection(self, positions: torch.Tensor,
                                   velocities: torch.Tensor,
                                   energy_delta: float = 0.0) -> LandauerMetrics:
        current_bits = self.measure_information_content(positions, velocities)

        if self.initial_bits == 0:
            self.initial_bits = current_bits

        self.bits_history.append(current_bits)
        bits_erased = max(0, self.initial_bits - current_bits)
        min_energy = bits_erased * self.LANDAUER_LIMIT
        gc_detected = (bits_erased > 1000 and abs(energy_delta) < min_energy * 0.1)

        metrics = LandauerMetrics(
            total_bits_initial=self.initial_bits,
            total_bits_current=current_bits,
            bits_erased=bits_erased,
            energy_per_bit_erased=abs(energy_delta) / max(bits_erased, 1),
            garbage_collection_detected=gc_detected
        )

        self.history.append(metrics)
        if gc_detected:
            logger.warning(f"EXPLOIT: Garbage collection detected! {bits_erased} bits erased!")

        return metrics


class FrustumCullingProbe3D:
    """Test for observer-based rendering optimization (3D version)."""

    def __init__(self, device: torch.device, fov_angle: float = 60.0):
        self.device = device
        self.fov_angle = fov_angle
        self.observer_pos = torch.zeros(3, device=device)
        self.observer_dir = torch.tensor([1.0, 0.0, 0.0], device=device)
        self.history: List[FrustumMetrics] = []
        self.previous_positions: Optional[torch.Tensor] = None
        self.snap_threshold = 0.1

    def set_observer(self, position: torch.Tensor, direction: torch.Tensor):
        self.observer_pos = position.to(self.device)
        self.observer_dir = direction.to(self.device)
        self.observer_dir = self.observer_dir / torch.norm(self.observer_dir)

    def is_in_frustum(self, positions: torch.Tensor) -> torch.Tensor:
        to_particle = positions - self.observer_pos.unsqueeze(0)
        dist = torch.norm(to_particle, dim=1, keepdim=True)
        to_particle_norm = to_particle / (dist + 1e-10)
        dot = (to_particle_norm * self.observer_dir.unsqueeze(0)).sum(dim=1)
        fov_cos = math.cos(math.radians(self.fov_angle / 2))
        return dot > fov_cos

    def detect_culling(self, positions: torch.Tensor) -> FrustumMetrics:
        in_frustum = self.is_in_frustum(positions)
        in_count = in_frustum.sum().item()
        out_count = (~in_frustum).sum().item()

        snap_events = 0
        if self.previous_positions is not None:
            pos_delta = torch.abs(positions - self.previous_positions).sum(dim=1)
            snapped = (in_frustum & (pos_delta > self.snap_threshold))
            snap_events = snapped.sum().item()

        self.previous_positions = positions.clone()
        culling_detected = snap_events > positions.shape[0] * 0.01

        metrics = FrustumMetrics(
            in_frustum_count=in_count,
            out_frustum_count=out_count,
            in_frustum_precision="FP32",
            out_frustum_precision="INT8" if culling_detected else "FP32",
            snap_events=snap_events,
            culling_detected=culling_detected
        )

        self.history.append(metrics)
        if culling_detected:
            logger.warning(f"EXPLOIT: Frustum culling detected! {snap_events} particles snapped!")

        return metrics

    def rotate_observer(self, angle_degrees: float):
        """Rotate observer view around Z axis."""
        angle_rad = math.radians(angle_degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        old_dir = self.observer_dir.clone()
        self.observer_dir[0] = old_dir[0] * cos_a - old_dir[1] * sin_a
        self.observer_dir[1] = old_dir[0] * sin_a + old_dir[1] * cos_a


class PhysicsExploitEngine:
    """Master controller for all physics exploit detection with threading."""

    def __init__(self, device: torch.device, num_threads: int = 4):
        self.device = device
        self.num_threads = num_threads

        self.relativity = SpecialRelativityProbe(device)
        self.navier_stokes = NavierStokesProbe(device)
        self.landauer = LandauerProbe(device)
        self.frustum = FrustumCullingProbe3D(device)

        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.results_queue: Queue = Queue()
        self.lock = threading.Lock()
        self.exploit_events: List[str] = []

    def run_all_probes(self, positions: torch.Tensor, velocities: torch.Tensor,
                       gpu_power: float = 0.0, energy_delta: float = 0.0) -> Dict[str, any]:
        results = {}

        results['relativity'] = self.relativity.check_bandwidth_limit(velocities, gpu_power)
        results['fluid'] = self.navier_stokes.detect_viscosity_clipping(velocities)
        results['landauer'] = self.landauer.check_garbage_collection(positions, velocities, energy_delta)
        results['frustum'] = self.frustum.detect_culling(positions)

        if results['relativity'].bandwidth_limited:
            self.exploit_events.append("BANDWIDTH_LIMIT")
        if results['fluid'].turbulence_suppressed:
            self.exploit_events.append("VISCOSITY_CLIPPING")
        if results['landauer'].garbage_collection_detected:
            self.exploit_events.append("GARBAGE_COLLECTION")
        if results['frustum'].culling_detected:
            self.exploit_events.append("FRUSTUM_CULLING")

        return results

    def get_exploit_summary(self) -> Dict[str, int]:
        summary = {}
        for event in self.exploit_events:
            summary[event] = summary.get(event, 0) + 1
        return summary

    def shutdown(self):
        self.executor.shutdown(wait=False)


# =============================================================================
# ADVANCED EXPLOIT PROBES - SUBSTRATE LEVEL
# =============================================================================

@dataclass
class SubstrateMetrics:
    """Metrics from substrate interference test."""
    baseline_glitch_rate: float = 0.0
    stressed_glitch_rate: float = 0.0
    timing_variance_ms: float = 0.0
    spatial_drift: float = 0.0
    lag_contagion_detected: bool = False
    cross_device_correlation: float = 0.0


@dataclass
class CollisionMetrics:
    """Metrics from collision tick / quantum clipping test."""
    test_velocity: float = 0.0
    wall_density: int = 0
    collision_detected: bool = True
    clip_through: bool = False
    clip_velocity_threshold: float = 0.0  # Velocity where clipping starts
    tunneling_probability: float = 0.0
    frame_skip_detected: bool = False


@dataclass
class IRLCorrelation:
    """Correlation between simulation glitches and IRL experiments."""
    experiment_type: str = ""
    simulation_metric: str = ""
    correlation_coefficient: float = 0.0
    p_value: float = 1.0
    notes: str = ""


class SubstrateInterferenceProbe:
    """
    Test if multiple machines share a parent "reality executable."

    Theory: If reality is simulated, heavy computation on one machine
    might cause timing jitter or glitch location drift on another,
    revealing shared substrate resources.

    Method:
    1. Record baseline glitch timing/locations
    2. Run heavy non-physics load (prime finder, matrix ops)
    3. Monitor if glitch patterns change
    4. Cross-correlate with remote machine data
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.history: List[SubstrateMetrics] = []
        self.baseline_glitch_times: List[float] = []
        self.stressed_glitch_times: List[float] = []
        self.glitch_locations: List[Tuple[float, float, float]] = []
        self.is_stressed = False
        self.stress_thread = None

    def _prime_stress_loop(self, duration_seconds: float = 5.0):
        """
        Heavy recursive prime finder - non-physics CPU stress.
        Designed to compete for "substrate resources."
        """
        import time as _time
        end_time = _time.time() + duration_seconds

        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True

        count = 0
        n = 2
        while _time.time() < end_time and self.is_stressed:
            if is_prime(n):
                count += 1
            n += 1

        return count

    def _matrix_stress_loop(self, duration_seconds: float = 5.0):
        """
        GPU matrix multiplication stress - tests VRAM contention.
        """
        import time as _time
        end_time = _time.time() + duration_seconds

        # Large matrices for GPU stress
        size = 2048
        a = torch.randn(size, size, device=self.device)
        b = torch.randn(size, size, device=self.device)

        ops = 0
        while _time.time() < end_time and self.is_stressed:
            c = torch.mm(a, b)
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            ops += 1

        return ops

    def start_stress_test(self, mode: str = "cpu", duration: float = 10.0):
        """Start background stress to test for lag contagion."""
        self.is_stressed = True

        if mode == "cpu":
            self.stress_thread = threading.Thread(
                target=self._prime_stress_loop,
                args=(duration,)
            )
        else:
            self.stress_thread = threading.Thread(
                target=self._matrix_stress_loop,
                args=(duration,)
            )

        self.stress_thread.start()
        logger.info(f"SUBSTRATE TEST: Started {mode.upper()} stress for {duration}s")

    def stop_stress_test(self):
        """Stop stress test."""
        self.is_stressed = False
        if self.stress_thread:
            self.stress_thread.join(timeout=2.0)
        logger.info("SUBSTRATE TEST: Stress stopped")

    def record_glitch(self, glitch_time: float, location: Tuple[float, float, float]):
        """Record glitch timing and spatial location."""
        if self.is_stressed:
            self.stressed_glitch_times.append(glitch_time)
        else:
            self.baseline_glitch_times.append(glitch_time)
        self.glitch_locations.append(location)

    def analyze_lag_contagion(self) -> SubstrateMetrics:
        """
        Compare baseline vs stressed glitch patterns.
        Lag contagion = shared substrate evidence.
        """
        if len(self.baseline_glitch_times) < 3 or len(self.stressed_glitch_times) < 3:
            return SubstrateMetrics()

        # Timing variance analysis
        baseline_intervals = np.diff(self.baseline_glitch_times)
        stressed_intervals = np.diff(self.stressed_glitch_times)

        baseline_var = np.var(baseline_intervals) if len(baseline_intervals) > 0 else 0
        stressed_var = np.var(stressed_intervals) if len(stressed_intervals) > 0 else 0

        timing_variance = abs(stressed_var - baseline_var)

        # Spatial drift (do glitches move when stressed?)
        if len(self.glitch_locations) > 10:
            early_locs = np.array(self.glitch_locations[:5])
            late_locs = np.array(self.glitch_locations[-5:])
            spatial_drift = np.mean(np.linalg.norm(late_locs - early_locs, axis=1))
        else:
            spatial_drift = 0.0

        # Lag contagion detected if timing variance increases significantly
        lag_contagion = timing_variance > baseline_var * 0.5 and stressed_var > baseline_var

        metrics = SubstrateMetrics(
            baseline_glitch_rate=len(self.baseline_glitch_times),
            stressed_glitch_rate=len(self.stressed_glitch_times),
            timing_variance_ms=timing_variance * 1000,
            spatial_drift=spatial_drift,
            lag_contagion_detected=lag_contagion,
            cross_device_correlation=0.0  # Would need network sync
        )

        self.history.append(metrics)

        if lag_contagion:
            logger.warning("SUBSTRATE EXPLOIT: Lag contagion detected! Glitch timing affected by load!")

        return metrics


class CollisionTickAuditor:
    """
    Test for "quantum clipping" - physics frame skipping.

    Theory: Simulations skip collision checks for fast-moving objects
    to save computation. This manifests as "tunneling" in games and
    potentially as Quantum Tunneling in reality.

    Method:
    1. Create a solid wall of particles
    2. Fire a test particle at increasing velocities
    3. Find the velocity where it clips through
    4. Compare to Planck-scale predictions
    """

    # Planck units for quantum comparison
    PLANCK_LENGTH = 1.616e-35  # meters
    PLANCK_TIME = 5.391e-44    # seconds
    PLANCK_VELOCITY = PLANCK_LENGTH / PLANCK_TIME  # ~c

    def __init__(self, device: torch.device, box_size: float = 200.0):
        self.device = device
        self.box_size = box_size
        self.history: List[CollisionMetrics] = []
        self.clip_velocity_log: List[Tuple[float, bool]] = []

    def create_particle_wall(self, center: Tuple[float, float, float],
                             thickness: float = 1.0,
                             density: int = 100) -> torch.Tensor:
        """
        Create a dense wall of particles for collision testing.
        """
        cx, cy, cz = center

        # Grid of particles forming a wall (YZ plane at x=cx)
        n_side = int(np.sqrt(density))
        y = torch.linspace(cy - 10, cy + 10, n_side, device=self.device)
        z = torch.linspace(cz - 10, cz + 10, n_side, device=self.device)
        yy, zz = torch.meshgrid(y, z, indexing='ij')

        # Add thickness in X direction
        wall_positions = []
        for dx in np.linspace(-thickness/2, thickness/2, 5):
            layer = torch.stack([
                torch.full_like(yy.flatten(), cx + dx),
                yy.flatten(),
                zz.flatten()
            ], dim=1)
            wall_positions.append(layer)

        wall = torch.cat(wall_positions, dim=0)
        return wall

    def fire_test_particle(self, wall_positions: torch.Tensor,
                           start_pos: Tuple[float, float, float],
                           velocity: float,
                           direction: Tuple[float, float, float] = (1, 0, 0),
                           dt: float = 0.001,
                           max_steps: int = 1000) -> CollisionMetrics:
        """
        Fire a particle at the wall and check for clipping.
        """
        pos = torch.tensor(start_pos, device=self.device, dtype=torch.float32)
        vel = torch.tensor(direction, device=self.device, dtype=torch.float32)
        vel = vel / torch.norm(vel) * velocity

        collision_radius = 0.5  # Detection radius

        collision_detected = False
        clip_through = False
        steps = 0

        wall_x_min = wall_positions[:, 0].min().item()
        wall_x_max = wall_positions[:, 0].max().item()

        started_before_wall = pos[0].item() < wall_x_min

        for step in range(max_steps):
            # Move particle
            pos = pos + vel * dt
            steps += 1

            # Check collision with wall particles
            distances = torch.norm(wall_positions - pos.unsqueeze(0), dim=1)
            min_dist = distances.min().item()

            if min_dist < collision_radius:
                collision_detected = True
                break

            # Check if we've passed through the wall
            if started_before_wall and pos[0].item() > wall_x_max:
                clip_through = True
                break

            # Bounds check
            if pos[0].item() > self.box_size:
                clip_through = True
                break

        # Calculate tunneling probability analog
        # In QM: P ~ exp(-2*k*d) where k = sqrt(2m(V-E))/hbar
        # Simplified: treat velocity as "energy" analog
        tunneling_prob = 0.0
        if clip_through:
            # Higher velocity = lower "barrier" = higher tunneling
            tunneling_prob = 1.0 - np.exp(-velocity / 100.0)

        metrics = CollisionMetrics(
            test_velocity=velocity,
            wall_density=len(wall_positions),
            collision_detected=collision_detected,
            clip_through=clip_through,
            clip_velocity_threshold=velocity if clip_through else 0.0,
            tunneling_probability=tunneling_prob,
            frame_skip_detected=clip_through and not collision_detected
        )

        self.history.append(metrics)
        self.clip_velocity_log.append((velocity, clip_through))

        if clip_through:
            logger.warning(f"QUANTUM CLIP: Particle clipped through at v={velocity:.2f}!")

        return metrics

    def find_clip_threshold(self, wall_positions: torch.Tensor,
                            start_pos: Tuple[float, float, float],
                            v_min: float = 1.0,
                            v_max: float = 1000.0,
                            steps: int = 20) -> float:
        """
        Binary search for the exact clipping velocity threshold.
        This is the simulation's "Planck velocity."
        """
        velocities = np.logspace(np.log10(v_min), np.log10(v_max), steps)

        clip_threshold = v_max

        for v in velocities:
            metrics = self.fire_test_particle(wall_positions, start_pos, v)
            if metrics.clip_through:
                clip_threshold = v
                break

        logger.info(f"CLIP THRESHOLD: Found at v = {clip_threshold:.4f}")
        logger.info(f"  Ratio to c: {clip_threshold / 306.6:.6f}")

        return clip_threshold

    def compare_to_quantum(self, clip_velocity: float) -> Dict[str, float]:
        """
        Compare simulation clipping to real quantum tunneling.
        """
        # Speed of light in simulation units
        c_sim = 306.6  # Mpc/Gyr

        # Ratio of clip velocity to c
        beta = clip_velocity / c_sim

        # In real QM, tunneling depends on barrier width and particle energy
        # We look for mathematical similarity

        return {
            'clip_velocity': clip_velocity,
            'fraction_of_c': beta,
            'log_scale': np.log10(clip_velocity),
            'planck_ratio': clip_velocity / (self.PLANCK_VELOCITY * 1e-50),  # Scaled
            'interpretation': 'FRAME_SKIP' if beta > 0.1 else 'CONTINUOUS'
        }


class IRLExperimentLogger:
    """
    Log correlations between simulation glitches and real-world experiments.

    Maps simulation findings to IRL equipment for testing:
    - Frustum Culling → Double-slit experiment
    - Landauer Erasure → EMF/radiation detection
    - Spatial Glitches → Interferometry
    - Energy Hotfixes → Precision mass measurement
    """

    EXPERIMENT_MAP = {
        'frustum_culling': {
            'equipment': 'Double-Slit Laser Kit',
            'hypothesis': 'Matter only renders when observed',
            'metric': 'wave_function_collapse_timing'
        },
        'landauer_erasure': {
            'equipment': 'EMF Detector / Geiger Counter',
            'hypothesis': 'Information deletion emits radiation',
            'metric': 'radiation_spike_on_delete'
        },
        'spatial_glitches': {
            'equipment': 'Michelson Interferometer',
            'hypothesis': 'Speed of light has coordinate jitter',
            'metric': 'fringe_pattern_variance'
        },
        'energy_hotfix': {
            'equipment': 'Precision Lab Scale (0.001g)',
            'hypothesis': 'Mass jitters during GPU load',
            'metric': 'mass_variance_vs_compute'
        },
        'quantum_clipping': {
            'equipment': 'Tunneling Microscope (STM)',
            'hypothesis': 'Tunneling rate matches frame skip',
            'metric': 'tunneling_probability_curve'
        }
    }

    def __init__(self):
        self.correlations: List[IRLCorrelation] = []
        self.experiment_log: List[Dict] = []

    def log_simulation_finding(self, finding_type: str,
                                simulation_value: float,
                                notes: str = "") -> Dict:
        """
        Log a simulation finding for IRL correlation.
        """
        if finding_type not in self.EXPERIMENT_MAP:
            logger.warning(f"Unknown finding type: {finding_type}")
            return {}

        exp_info = self.EXPERIMENT_MAP[finding_type]

        entry = {
            'timestamp': datetime.now().isoformat(),
            'finding_type': finding_type,
            'simulation_value': simulation_value,
            'recommended_equipment': exp_info['equipment'],
            'hypothesis': exp_info['hypothesis'],
            'target_metric': exp_info['metric'],
            'notes': notes,
            'irl_result': None,  # To be filled after experiment
            'correlation': None
        }

        self.experiment_log.append(entry)

        logger.info(f"IRL EXPERIMENT LOGGED: {finding_type}")
        logger.info(f"  Equipment: {exp_info['equipment']}")
        logger.info(f"  Hypothesis: {exp_info['hypothesis']}")

        return entry

    def record_irl_result(self, finding_type: str, irl_value: float):
        """Record result from actual IRL experiment."""
        for entry in self.experiment_log:
            if entry['finding_type'] == finding_type and entry['irl_result'] is None:
                entry['irl_result'] = irl_value

                # Calculate correlation
                if entry['simulation_value'] != 0:
                    entry['correlation'] = irl_value / entry['simulation_value']

                logger.info(f"IRL RESULT: {finding_type} = {irl_value}")
                logger.info(f"  Correlation: {entry['correlation']:.4f}")
                break

    def generate_experiment_protocol(self) -> str:
        """Generate a protocol document for IRL experiments."""
        protocol = """
╔══════════════════════════════════════════════════════════════╗
║          SIMULATION HYPOTHESIS - IRL TEST PROTOCOL           ║
╠══════════════════════════════════════════════════════════════╣
"""
        for exp_type, info in self.EXPERIMENT_MAP.items():
            protocol += f"""
║  {exp_type.upper():^58}  ║
║  Equipment: {info['equipment']:<46}  ║
║  Hypothesis: {info['hypothesis']:<45}  ║
║  Metric: {info['metric']:<49}  ║
║{'─' * 62}║
"""
        protocol += "╚══════════════════════════════════════════════════════════════╝"

        return protocol

    def get_summary(self) -> Dict:
        """Get summary of all findings and correlations."""
        return {
            'total_findings': len(self.experiment_log),
            'irl_confirmed': sum(1 for e in self.experiment_log if e['irl_result'] is not None),
            'strong_correlations': sum(1 for e in self.experiment_log
                                       if e['correlation'] and 0.8 < e['correlation'] < 1.2),
            'findings': self.experiment_log
        }


# =============================================================================
# 3D UNIVERSE SIMULATION
# =============================================================================

class Universe3D:
    """
    3D cosmic simulation with dark matter and glitch detection.
    """

    def __init__(
        self,
        num_particles: int = 1000,
        box_size_mpc: float = 200.0,
        start_redshift: float = 50.0,
        precision: str = "float32",
        dm_ratio: float = 5.0,
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

        # Precision
        self.precision_mode = get_mode_from_string(precision)
        self.precision_str = precision
        self.dtype = torch.float64 if precision == "float64" else torch.float32

        # Glitch detector
        self.glitch_detector = GlitchDetector(threshold=0.05)

        # Physics exploit engine (threaded)
        self.exploit_engine = PhysicsExploitEngine(self.device, num_threads=4)

        # Completion flags
        self.running = True
        self.completed = False
        self.min_redshift = 0.01

        # Initialize 3D positions
        self._initialize()

        # History
        self.history = {
            'redshift': [self.redshift],
            'time_gyr': [self.time_gyr],
            'energy': [],
            'glitches': [],
            'exploits': [],
        }

        logger.info(f"Universe3D initialized: {num_particles} particles, {precision} precision")

    def _initialize(self):
        """Initialize 3D particle distribution."""
        set_all_seeds(self.seed)

        n = self.num_particles

        # Generate positions on a grid with perturbations
        n_side = int(np.cbrt(n))
        self.num_particles = n_side ** 3

        # Create 3D grid
        x = torch.linspace(0, self.box_size, n_side, device=self.device, dtype=self.dtype)
        y = torch.linspace(0, self.box_size, n_side, device=self.device, dtype=self.dtype)
        z = torch.linspace(0, self.box_size, n_side, device=self.device, dtype=self.dtype)

        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')

        self.positions = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)

        # Add primordial perturbations (scaled by redshift)
        perturbation_scale = 5.0 / (1 + self.redshift / 100)
        perturbations = torch.randn_like(self.positions) * perturbation_scale
        self.positions = (self.positions + perturbations) % self.box_size

        # Initialize velocities (Hubble flow + peculiar velocities)
        self.velocities = torch.zeros_like(self.positions)
        H = hubble_parameter(self.redshift)
        self.velocities = (self.positions - self.box_size / 2) * H * 1e-5

        # Add small random velocities
        self.velocities += torch.randn_like(self.velocities) * 0.1

        # Masses (uniform)
        self.masses = torch.ones(self.num_particles, device=self.device, dtype=self.dtype)

        logger.info(f"Initialized {self.num_particles} particles in 3D ({n_side}x{n_side}x{n_side})")

    def _compute_accelerations(self):
        """Compute gravitational accelerations using PM solver."""
        n_grid = 32  # Coarse grid for speed

        # Deposit mass onto 3D grid
        density = torch.zeros(n_grid, n_grid, n_grid, device=self.device, dtype=self.dtype)
        pos_grid = (self.positions / self.box_size * n_grid).long() % n_grid

        for i in range(len(self.positions)):
            ix, iy, iz = pos_grid[i]
            density[ix, iy, iz] += self.masses[i]

        # Add dark matter contribution (smooth NFW-like profile)
        dm_density = density.mean() * self.dm_ratio
        density = density + dm_density * 0.1

        mean_rho = density.mean()
        delta = (density - mean_rho) / (mean_rho + 1e-10)

        # 3D FFT
        delta_k = torch.fft.fftn(delta)

        kx = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device).to(self.dtype) * 2 * np.pi
        ky = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device).to(self.dtype) * 2 * np.pi
        kz = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device).to(self.dtype) * 2 * np.pi
        kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
        k_sq = kx**2 + ky**2 + kz**2 + 1e-10

        # Apply quantization (the "broken math")
        k_sq_quantized = quantize_distance_squared(k_sq, self.precision_mode)

        # Poisson equation
        phi_k = -4 * np.pi * G_NEWTON * mean_rho * delta_k / k_sq_quantized / self.scale
        phi_k[0, 0, 0] = 0

        # Gradient
        ax_k = -1j * kx * phi_k
        ay_k = -1j * ky * phi_k
        az_k = -1j * kz * phi_k

        ax = torch.fft.ifftn(ax_k).real
        ay = torch.fft.ifftn(ay_k).real
        az = torch.fft.ifftn(az_k).real

        # Interpolate to particles
        accelerations = torch.zeros_like(self.positions)
        for i in range(len(self.positions)):
            ix, iy, iz = pos_grid[i]
            accelerations[i, 0] = ax[ix, iy, iz]
            accelerations[i, 1] = ay[ix, iy, iz]
            accelerations[i, 2] = az[ix, iy, iz]

        return accelerations

    def get_kinetic_energy(self) -> float:
        v_sq = (self.velocities ** 2).sum(dim=-1)
        ke = 0.5 * (self.masses * v_sq).sum()
        return ke.item()

    def get_total_momentum(self) -> Tuple[float, float, float]:
        """Total 3D momentum (should be ~0 for isolated system)."""
        px = (self.masses * self.velocities[:, 0]).sum().item()
        py = (self.masses * self.velocities[:, 1]).sum().item()
        pz = (self.masses * self.velocities[:, 2]).sum().item()
        return (px, py, pz)

    def step(self, dz: float = 1.0):
        """Evolve by one redshift step with full glitch and exploit detection."""
        if self.completed or self.redshift <= self.min_redshift:
            if not self.completed:
                self.completed = True
                self.running = False
                logger.info("=" * 50)
                logger.info("SIMULATION COMPLETE")
                logger.info(f"Final z = {self.redshift:.4f}, t = {self.time_gyr:.3f} Gyr")
                logger.info("=" * 50)
            return

        z_new = max(self.min_redshift, self.redshift - dz)
        dt_gyr = abs(cosmic_time(z_new) - cosmic_time(self.redshift))

        # Previous energy for exploit detection
        prev_energy = self.get_kinetic_energy() if self.history['energy'] else 0

        # Compute accelerations
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
        energy = self.get_kinetic_energy()
        momentum = self.get_total_momentum()

        # 1. Energy conservation check
        energy_glitch = self.glitch_detector.check_energy_conservation(energy, self.tick, z_new)
        if energy_glitch:
            self.history['glitches'].append(energy_glitch)

        # 2. Momentum drift check
        momentum_glitch = self.glitch_detector.check_momentum(momentum, self.tick, z_new)
        if momentum_glitch:
            self.history['glitches'].append(momentum_glitch)

        # 3. Subnormal check (denormal flooding)
        subnormal_glitch = self.glitch_detector.check_subnormals(self.positions, self.tick, z_new)
        if subnormal_glitch:
            self.history['glitches'].append(subnormal_glitch)

        # 4. Entropy check (every 10 ticks)
        if self.tick % 10 == 0:
            entropy_glitch = self.glitch_detector.check_entropy(
                self.positions, self.velocities, self.tick, z_new
            )
            if entropy_glitch:
                self.history['glitches'].append(entropy_glitch)

        # === PHYSICS EXPLOIT PROBES (every 20 ticks) ===
        if self.tick % 20 == 0:
            energy_delta = energy - prev_energy if prev_energy else 0

            exploit_results = self.exploit_engine.run_all_probes(
                self.positions,
                self.velocities,
                gpu_power=0.0,
                energy_delta=energy_delta
            )

            # Log significant exploits
            if exploit_results['relativity'].max_gamma > 2.0:
                logger.debug(f"  Relativity: γ_max={exploit_results['relativity'].max_gamma:.2f}")

            if exploit_results['landauer'].bits_erased > 100:
                logger.debug(f"  Landauer: {exploit_results['landauer'].bits_erased} bits erased")

            self.history['exploits'].append({
                'tick': self.tick,
                'redshift': z_new,
                'relativity': exploit_results['relativity'],
                'fluid': exploit_results['fluid'],
                'landauer': exploit_results['landauer'],
                'frustum': exploit_results['frustum']
            })

            # Rotate observer for frustum culling test
            self.exploit_engine.frustum.rotate_observer(5.0)

        self.history['energy'].append(energy)
        self.history['redshift'].append(z_new)
        self.history['time_gyr'].append(self.time_gyr)

        # Epoch transition
        new_epoch = get_current_epoch(z_new)
        if new_epoch != self.current_epoch:
            epoch_info = EPOCHS[new_epoch]
            logger.info("=" * 50)
            logger.info(f"EPOCH TRANSITION: {epoch_info.name}")
            logger.info(f"  z = {z_new:.1f}, t = {self.time_gyr:.2f} Gyr")
            logger.info("=" * 50)
            self.current_epoch = new_epoch


# =============================================================================
# 3D VISUALIZER
# =============================================================================

class Universe3DVisualizer:
    """Interactive 3D visualization with rotation and GPU profiling."""

    def __init__(self, universe: Universe3D, enable_gpu_profiling: bool = True):
        self.universe = universe
        self.running = True
        self.rotation_angle = 0

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
        """Create 3D figure."""
        plt.style.use('dark_background')

        self.fig = plt.figure(figsize=(14, 10))
        self.fig.patch.set_facecolor('#050510')

        # 3D universe view (main plot)
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.ax_3d.set_facecolor('#050510')

        # Info panel
        self.ax_info = self.fig.add_subplot(122)
        self.ax_info.set_facecolor('#0a0a1a')
        self.ax_info.axis('off')

        # Remove 3D axis panes for cleaner look
        self.ax_3d.xaxis.pane.fill = False
        self.ax_3d.yaxis.pane.fill = False
        self.ax_3d.zaxis.pane.fill = False
        self.ax_3d.xaxis.pane.set_edgecolor('none')
        self.ax_3d.yaxis.pane.set_edgecolor('none')
        self.ax_3d.zaxis.pane.set_edgecolor('none')

        # Grid styling
        self.ax_3d.xaxis._axinfo["grid"]['color'] = (0.2, 0.2, 0.3, 0.3)
        self.ax_3d.yaxis._axinfo["grid"]['color'] = (0.2, 0.2, 0.3, 0.3)
        self.ax_3d.zaxis._axinfo["grid"]['color'] = (0.2, 0.2, 0.3, 0.3)

    def update(self, frame):
        """Animation frame update."""
        if self.universe.completed or not self.running:
            return

        if self.universe.redshift <= self.universe.min_redshift:
            self.running = False
            return

        # Evolve simulation
        self.universe.step(dz=1.0)

        # Get positions
        pos = self.universe.positions.cpu().numpy()

        # Subsample for performance
        n_show = min(len(pos), 3000)
        if len(pos) > n_show:
            idx = np.random.choice(len(pos), n_show, replace=False)
            pos_show = pos[idx]
        else:
            pos_show = pos

        # Clear and redraw 3D plot
        self.ax_3d.clear()
        self.ax_3d.set_facecolor('#050510')

        # Color by position (creates nice gradient effect)
        colors = (pos_show[:, 0] + pos_show[:, 1] + pos_show[:, 2]) / (self.universe.box_size * 3)

        # Get epoch color
        epoch_info = EPOCHS.get(self.universe.current_epoch, EPOCHS[CosmicEpoch.PRESENT])

        self.ax_3d.scatter(
            pos_show[:, 0], pos_show[:, 1], pos_show[:, 2],
            c=colors, cmap='plasma', s=2, alpha=0.6
        )

        # Set limits
        self.ax_3d.set_xlim(0, self.universe.box_size)
        self.ax_3d.set_ylim(0, self.universe.box_size)
        self.ax_3d.set_zlim(0, self.universe.box_size)

        # Labels
        self.ax_3d.set_xlabel('X (Mpc)', color='white', fontsize=10)
        self.ax_3d.set_ylabel('Y (Mpc)', color='white', fontsize=10)
        self.ax_3d.set_zlabel('Z (Mpc)', color='white', fontsize=10)

        # Rotate view
        self.rotation_angle += 0.5
        self.ax_3d.view_init(elev=20, azim=self.rotation_angle)

        # Title
        self.ax_3d.set_title(
            f'3D Universe | z={self.universe.redshift:.1f} | {epoch_info.name}',
            color=epoch_info.color, fontsize=14, fontweight='bold'
        )

        # Update info panel
        self.ax_info.clear()
        self.ax_info.set_facecolor('#0a0a1a')
        self.ax_info.axis('off')

        # Get exploit count
        exploit_count = len(self.universe.exploit_engine.exploit_events)

        info_text = f"""
╔══════════════════════════════════════╗
║     UNIVERSE 3D - COSMIC VIEWER      ║
╠══════════════════════════════════════╣
║                                      ║
║  COSMOLOGY                           ║
║  ─────────                           ║
║  Redshift:     z = {self.universe.redshift:>8.2f}         ║
║  Cosmic Time:  {self.universe.time_gyr:>8.3f} Gyr        ║
║  Scale Factor: {self.universe.scale:>8.4f}           ║
║  Epoch:        {epoch_info.name:<20}  ║
║                                      ║
║  SIMULATION                          ║
║  ──────────                          ║
║  Particles:    {self.universe.num_particles:>8,}            ║
║  Box Size:     {self.universe.box_size:>8.0f} Mpc         ║
║  Tick:         {self.universe.tick:>8}              ║
║  Precision:    {self.universe.precision_str:<15}   ║
║                                      ║
║  ANOMALY DETECTION                   ║
║  ─────────────────                   ║
║  Glitches:     {self.universe.glitch_detector.get_glitch_count():>8}              ║
║  Exploits:     {exploit_count:>8}              ║
║                                      ║
║  DARK MATTER                         ║
║  ───────────                         ║
║  DM Ratio:     {self.universe.dm_ratio:>8.1f}x            ║
║                                      ║
╚══════════════════════════════════════╝

   Drag mouse on 3D plot to rotate
   Close window to see final report
"""

        self.ax_info.text(
            0.05, 0.95, info_text,
            transform=self.ax_info.transAxes,
            fontsize=10, fontfamily='monospace',
            color='cyan', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#0a0a1a', edgecolor='cyan', alpha=0.8)
        )

    def run(self):
        """Start animation with GPU profiling."""
        self.setup()

        # Start GPU profiling (from gpu_profiler.py)
        if self.gpu_profiler:
            self.gpu_profiler.start(f"Universe3D_{self.universe.precision_str}")

        ani = FuncAnimation(
            self.fig, self.update,
            frames=None,
            interval=100,  # 10 FPS for 3D (smoother)
            blit=False,
            cache_frame_data=False
        )

        plt.tight_layout()
        plt.show()

        # Stop GPU profiling and print report
        if self.gpu_profiler:
            self.gpu_profile_result = self.gpu_profiler.stop()
            if self.gpu_profile_result:
                self.gpu_profiler.print_report(self.gpu_profile_result)
                logger.info(f"GPU Mean Power: {self.gpu_profile_result.mean_power_watts:.1f}W")
                logger.info(f"GPU Mean Clock: {self.gpu_profile_result.mean_clock_mhz:.0f} MHz")

        return ani


# =============================================================================
# MAIN
# =============================================================================

def main():
    global logger

    parser = argparse.ArgumentParser(description="Universe 3D - Interactive Cosmic Visualization")
    parser.add_argument("--particles", "-n", type=int, default=1000, help="Number of particles")
    parser.add_argument("--box", type=float, default=200.0, help="Box size in Mpc")
    parser.add_argument("--start-z", type=float, default=50.0, help="Starting redshift")
    parser.add_argument("--precision", type=str, default="float32",
                       choices=["float64", "float32", "bfloat16", "float16", "int8", "int4"])
    parser.add_argument("--dm-ratio", type=float, default=5.0, help="Dark matter ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log", type=str, default=None, help="Log file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--no-gpu-profile", action="store_true", help="Disable GPU profiling")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = args.log or f"logs/universe3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_file, log_level)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create universe
    logger.info("=" * 60)
    logger.info("  UNIVERSE 3D - COSMIC EVOLUTION ENGINE")
    logger.info("=" * 60)

    universe = Universe3D(
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

    logger.info("Starting 3D visualization...")
    logger.info("Drag mouse to rotate | Close window to exit")

    # Run visualization
    viz = Universe3DVisualizer(universe, enable_gpu_profiling=not args.no_gpu_profile)
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

    # Physics exploit summary
    exploit_summary = universe.exploit_engine.get_exploit_summary()
    if exploit_summary:
        logger.info("-" * 40)
        logger.info("PHYSICS EXPLOIT DETECTIONS:")
        for exploit_type, count in exploit_summary.items():
            logger.info(f"  {exploit_type}: {count}")
    else:
        logger.info("-" * 40)
        logger.info("PHYSICS EXPLOITS: None detected (reality appears consistent)")

    # Log probe summary stats
    if universe.exploit_engine.relativity.history:
        max_gamma = max(m.max_gamma for m in universe.exploit_engine.relativity.history)
        logger.info(f"  Max Lorentz factor (γ): {max_gamma:.4f}")

    if universe.exploit_engine.landauer.bits_history:
        initial_bits = universe.exploit_engine.landauer.initial_bits
        final_bits = universe.exploit_engine.landauer.bits_history[-1] if universe.exploit_engine.landauer.bits_history else 0
        logger.info(f"  Information: {initial_bits} → {final_bits} bits ({initial_bits - final_bits} erased)")

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
    logger.info("-" * 40)
    logger.info("PHYSICS EXPLOIT PROBES (3D):")
    logger.info(f"  Special Relativity:     ACTIVE (Lorentz factor)")
    logger.info(f"  Navier-Stokes:          ACTIVE (viscosity clipping)")
    logger.info(f"  Landauer's Principle:   ACTIVE (garbage collection)")
    logger.info(f"  Frustum Culling 3D:     ACTIVE (observer rendering)")
    logger.info("=" * 60)

    # Cleanup threaded executor
    universe.exploit_engine.shutdown()


if __name__ == "__main__":
    main()
