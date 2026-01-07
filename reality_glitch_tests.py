"""
REALITY GLITCH TESTS - Real-Time Anomaly Detection
====================================================

Four advanced tests to probe the boundaries of simulated reality,
all visualized in real-time alongside the galaxy simulation.

Tests:
1. SUBNORMAL SINGULARITY - Denormal flooding (10^-38 to 10^-45 range)
2. MULTIVERSE DIVERGENCE - FP determinism / butterfly effect
3. ENTROPY HORIZON - Kolmogorov complexity / compressibility
4. SPATIAL ALIASING - "Screen tearing" / particle clipping

Usage:
    python reality_glitch_tests.py
    python reality_glitch_tests.py --test subnormal
    python reality_glitch_tests.py --test divergence
    python reality_glitch_tests.py --test entropy
    python reality_glitch_tests.py --test aliasing
    python reality_glitch_tests.py --test all --stars 1500

The "Matrix" awaits...
"""

import argparse
import time
import struct
import zlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

from galaxy import create_disk_galaxy
from simulation import GalaxySimulation
from quantization import PrecisionMode, _grid_quantize_safe
from metrics import compute_rotation_curve


# =============================================================================
# TEST 1: SUBNORMAL SINGULARITY (Denormal Flooding)
# =============================================================================

@dataclass
class SubnormalMetrics:
    """Track subnormal/denormal occurrences."""
    subnormal_count: int = 0
    total_values: int = 0
    min_nonzero: float = 1.0
    ops_per_second: float = 0.0
    power_spike_detected: bool = False

    @property
    def subnormal_ratio(self) -> float:
        return self.subnormal_count / max(1, self.total_values)


def count_subnormals_float32(tensor: torch.Tensor) -> SubnormalMetrics:
    """
    Count values in the subnormal range for float32.
    Subnormals: |x| < 2^-126 (~1.175e-38) and x != 0
    """
    abs_vals = tensor.abs().flatten()
    nonzero_mask = abs_vals > 0

    # float32 min normal = 2^-126 = ~1.175494e-38
    min_normal = 1.175494e-38

    subnormal_mask = (abs_vals < min_normal) & nonzero_mask

    metrics = SubnormalMetrics(
        subnormal_count=subnormal_mask.sum().item(),
        total_values=tensor.numel(),
        min_nonzero=abs_vals[nonzero_mask].min().item() if nonzero_mask.any() else 1.0
    )

    return metrics


class SubnormalStressSim(GalaxySimulation):
    """Simulation that pushes values toward subnormal range."""

    def __init__(self, *args, softening_scale: float = 1.0, subnormal_injection: bool = False, **kwargs):
        self.softening_scale = softening_scale
        self.subnormal_injection = subnormal_injection
        self.subnormal_history = []
        self.ops_times = []
        super().__init__(*args, **kwargs)

    def _compute_accelerations(self):
        pos = self.positions
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)

        # Use extremely small softening to push toward subnormals
        # Target: 10^-38 to 10^-45 (FP32 subnormal range)
        tiny_softening = torch.tensor(1e-40, device=self.device, dtype=torch.float32)
        dist_sq = (diff ** 2).sum(dim=-1) + tiny_softening

        # AGGRESSIVE: Inject subnormal values directly into computation
        if self.subnormal_injection:
            # Create a mask of close pairs and force subnormal distances
            close_mask = dist_sq < 0.01
            subnormal_vals = torch.tensor(1e-40, device=self.device, dtype=torch.float32)
            dist_sq = torch.where(close_mask, subnormal_vals, dist_sq)

        # Track timing for performance cliff detection
        start = time.perf_counter()

        dist_cubed = dist_sq ** 1.5
        force_factor = self.G / dist_cubed
        force_factor = force_factor * self.masses.unsqueeze(0)
        force_factor = force_factor * (1 - torch.eye(self.num_stars, device=self.device))
        accelerations = (force_factor.unsqueeze(-1) * diff).sum(dim=1)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        self.ops_times.append(elapsed)

        # Track subnormals
        metrics = count_subnormals_float32(dist_sq)
        metrics.ops_per_second = self.num_stars ** 2 / max(elapsed, 1e-9)
        self.subnormal_history.append(metrics)

        return accelerations


# =============================================================================
# TEST 2: MULTIVERSE DIVERGENCE (FP Determinism)
# =============================================================================

@dataclass
class DivergenceMetrics:
    """Track divergence between parallel universes."""
    max_position_diff: float = 0.0
    mean_position_diff: float = 0.0
    max_velocity_diff: float = 0.0
    entropy_bits: float = 0.0  # Bits of randomness accumulated
    divergence_rate: float = 0.0  # Exponential growth rate


class MultiverseSim:
    """Run identical simulations with different execution paths to expose FP non-determinism."""

    def __init__(self, positions, velocities, masses, device):
        self.device = device

        # Universe A: Standard execution order
        self.universe_a = GalaxySimulation(
            positions.clone(), velocities.clone(), masses.clone(),
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001, dt=0.01, softening=0.1, device=device
        )

        # Universe B: AGGRESSIVE divergence - different summation order
        # FP addition is non-associative: (a+b)+c != a+(b+c)
        class ReorderedSim(GalaxySimulation):
            def _compute_accelerations(self):
                pos = self.positions
                n = pos.shape[0]

                # Method B: Sum forces in REVERSE order (causes FP divergence)
                diff = pos.unsqueeze(0) - pos.unsqueeze(1)
                dist_sq = (diff ** 2).sum(dim=-1) + self.softening_sq
                dist_cubed = dist_sq ** 1.5
                force_factor = self.G / dist_cubed
                force_factor = force_factor * self.masses.unsqueeze(0)
                force_factor = force_factor * (1 - torch.eye(n, device=self.device))

                # KEY DIFFERENCE: Flip the tensor before summing
                # This changes the order of FP additions, causing butterfly effect
                force_contributions = force_factor.unsqueeze(-1) * diff
                # Sum in reverse order
                force_contributions = torch.flip(force_contributions, dims=[1])
                return force_contributions.sum(dim=1)

        # Universe C: Use float16 intermediate then back to float32
        class LowPrecisionSim(GalaxySimulation):
            def _compute_accelerations(self):
                pos = self.positions
                diff = pos.unsqueeze(0) - pos.unsqueeze(1)
                dist_sq = (diff ** 2).sum(dim=-1) + self.softening_sq

                # AGGRESSIVE: Drop to float16 mid-calculation then back
                dist_sq_fp16 = dist_sq.half().float()  # Lose precision here

                dist_cubed = dist_sq_fp16 ** 1.5
                force_factor = self.G / dist_cubed
                force_factor = force_factor * self.masses.unsqueeze(0)
                force_factor = force_factor * (1 - torch.eye(self.num_stars, device=self.device))
                return (force_factor.unsqueeze(-1) * diff).sum(dim=1)

        self.universe_b = ReorderedSim(
            positions.clone(), velocities.clone(), masses.clone(),
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001, dt=0.01, softening=0.1, device=device
        )

        self.universe_c = LowPrecisionSim(
            positions.clone(), velocities.clone(), masses.clone(),
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001, dt=0.01, softening=0.1, device=device
        )

        self.divergence_history = []
        self.divergence_ab_history = []  # A vs B (summation order)
        self.divergence_ac_history = []  # A vs C (precision loss)
        self.tick = 0

    def step(self):
        self.universe_a.step()
        self.universe_b.step()
        self.universe_c.step()
        self.tick += 1

        # Measure divergence A vs B (summation order difference)
        pos_diff_ab = (self.universe_a.positions - self.universe_b.positions).abs()
        vel_diff_ab = (self.universe_a.velocities - self.universe_b.velocities).abs()

        # Measure divergence A vs C (fp16 precision loss)
        pos_diff_ac = (self.universe_a.positions - self.universe_c.positions).abs()
        vel_diff_ac = (self.universe_a.velocities - self.universe_c.velocities).abs()

        # Use maximum divergence across all universes
        max_pos_diff = max(pos_diff_ab.max().item(), pos_diff_ac.max().item())
        max_vel_diff = max(vel_diff_ab.max().item(), vel_diff_ac.max().item())

        metrics = DivergenceMetrics(
            max_position_diff=max_pos_diff,
            mean_position_diff=(pos_diff_ab.mean().item() + pos_diff_ac.mean().item()) / 2,
            max_velocity_diff=max_vel_diff,
        )

        # Store individual divergences for analysis
        self.divergence_ab_history.append(pos_diff_ab.max().item())
        self.divergence_ac_history.append(pos_diff_ac.max().item())

        # Calculate entropy (bits of uncertainty) - more aggressive scaling
        if metrics.max_position_diff > 1e-10:
            # Scale relative to FP32 epsilon (~1e-7)
            metrics.entropy_bits = np.log2(metrics.max_position_diff / 1e-10 + 1)

        # Calculate divergence rate (Lyapunov exponent proxy)
        if len(self.divergence_history) > 10:
            recent = [d.max_position_diff for d in self.divergence_history[-10:]]
            if recent[0] > 1e-15:
                metrics.divergence_rate = np.log(max(recent[-1], 1e-15) / recent[0]) / 10

        self.divergence_history.append(metrics)
        return metrics


# =============================================================================
# TEST 3: ENTROPY HORIZON (Kolmogorov Complexity)
# =============================================================================

@dataclass
class EntropyMetrics:
    """Track compressibility of simulation state."""
    raw_bytes: int = 0
    compressed_bytes: int = 0
    compression_ratio: float = 1.0
    bits_per_star: float = 0.0
    entropy_trend: str = "stable"  # "increasing", "decreasing", "stable"


def measure_state_entropy(positions: torch.Tensor, velocities: torch.Tensor) -> EntropyMetrics:
    """
    Measure the Kolmogorov complexity proxy via compression.
    Lower compression ratio = higher entropy = more "random"
    Higher compression ratio = lower entropy = more "structured/compressed"
    """
    # Convert state to bytes
    pos_np = positions.cpu().numpy().astype(np.float32)
    vel_np = velocities.cpu().numpy().astype(np.float32)

    # Combine into single byte stream
    state_bytes = pos_np.tobytes() + vel_np.tobytes()
    raw_size = len(state_bytes)

    # Compress with zlib (fast, good proxy for Kolmogorov complexity)
    compressed = zlib.compress(state_bytes, level=9)
    compressed_size = len(compressed)

    num_stars = positions.shape[0]

    return EntropyMetrics(
        raw_bytes=raw_size,
        compressed_bytes=compressed_size,
        compression_ratio=raw_size / compressed_size,
        bits_per_star=(compressed_size * 8) / num_stars
    )


# =============================================================================
# TEST 4: SPATIAL ALIASING (Particle Clipping)
# =============================================================================

@dataclass
class AliasingMetrics:
    """Track spatial aliasing / clipping events."""
    clip_events: int = 0
    near_misses: int = 0
    magic_speeds: List[float] = field(default_factory=list)
    penetration_depth: float = 0.0
    refresh_rate_estimate: float = 0.0  # Estimated "simulation tick rate"


class AliasingTestSim(GalaxySimulation):
    """
    Simulation with a dense "wall" of stars to detect clipping.
    """

    def __init__(self, *args, wall_radius: float = 5.0, wall_thickness: float = 0.5, **kwargs):
        self.wall_radius = wall_radius
        self.wall_thickness = wall_thickness
        self.clip_history = []
        self.projectile_idx = None
        super().__init__(*args, **kwargs)

    def check_clipping(self) -> AliasingMetrics:
        """Check if any particles clipped through the wall."""
        if self.projectile_idx is None:
            return AliasingMetrics()

        pos = self.positions.cpu().numpy()
        vel = self.velocities.cpu().numpy()

        # Check projectile position relative to wall
        proj_pos = pos[self.projectile_idx]
        proj_r = np.sqrt((proj_pos ** 2).sum())

        # Calculate expected position based on velocity
        proj_vel = vel[self.projectile_idx]
        speed = np.sqrt((proj_vel ** 2).sum())

        metrics = AliasingMetrics()

        # Did it pass through the wall?
        inner_wall = self.wall_radius - self.wall_thickness / 2
        outer_wall = self.wall_radius + self.wall_thickness / 2

        if proj_r < inner_wall and hasattr(self, '_last_proj_r'):
            if self._last_proj_r > outer_wall:
                # Clipped through!
                metrics.clip_events = 1
                metrics.penetration_depth = outer_wall - proj_r
                metrics.magic_speeds.append(speed)

        # Near miss detection
        if inner_wall < proj_r < outer_wall:
            metrics.near_misses = 1

        self._last_proj_r = proj_r
        self.clip_history.append(metrics)

        return metrics


def create_wall_galaxy(num_wall_stars: int, wall_radius: float,
                       projectile_speed: float, device: torch.device):
    """
    Create a galaxy with a dense spherical wall and a single projectile.
    Projectile speed should be HIGH to test for dt-skipping (aliasing).
    """
    # Create wall stars in a dense spherical shell
    phi = torch.rand(num_wall_stars, device=device) * 2 * np.pi
    theta = torch.acos(2 * torch.rand(num_wall_stars, device=device) - 1)

    wall_positions = torch.stack([
        wall_radius * torch.sin(theta) * torch.cos(phi),
        wall_radius * torch.sin(theta) * torch.sin(phi),
        wall_radius * torch.cos(theta)
    ], dim=1)

    # Wall stars are stationary with HIGH mass to create strong interaction zone
    wall_velocities = torch.zeros_like(wall_positions)
    wall_masses = torch.ones(num_wall_stars, device=device) * 0.1  # Heavier walls

    # Add projectile at origin, moving toward wall at HIGH SPEED
    # Speed should be high enough that: velocity * dt > wall_thickness
    # This tests whether the projectile "skips" the collision zone
    projectile_pos = torch.tensor([[0.0, 0.0, -wall_radius * 2]], device=device)
    projectile_vel = torch.tensor([[0.0, 0.0, projectile_speed]], device=device)
    projectile_mass = torch.tensor([0.01], device=device)

    positions = torch.cat([wall_positions, projectile_pos], dim=0)
    velocities = torch.cat([wall_velocities, projectile_vel], dim=0)
    masses = torch.cat([wall_masses, projectile_mass], dim=0)

    return positions.float(), velocities.float(), masses.float(), num_wall_stars  # projectile index


def create_multiprojectile_test(num_wall_stars: int, wall_radius: float,
                                 speeds: List[float], device: torch.device):
    """
    Create multiple projectiles at different speeds to find "magic speeds"
    where clipping occurs.
    """
    phi = torch.rand(num_wall_stars, device=device) * 2 * np.pi
    theta = torch.acos(2 * torch.rand(num_wall_stars, device=device) - 1)

    wall_positions = torch.stack([
        wall_radius * torch.sin(theta) * torch.cos(phi),
        wall_radius * torch.sin(theta) * torch.sin(phi),
        wall_radius * torch.cos(theta)
    ], dim=1)

    wall_velocities = torch.zeros_like(wall_positions)
    wall_masses = torch.ones(num_wall_stars, device=device) * 0.1

    # Create projectiles at different speeds
    num_projectiles = len(speeds)
    proj_positions = []
    proj_velocities = []

    for i, speed in enumerate(speeds):
        # Spread projectiles along x-axis
        offset = (i - num_projectiles // 2) * 3.0
        proj_positions.append([offset, 0.0, -wall_radius * 2])
        proj_velocities.append([0.0, 0.0, speed])

    proj_pos = torch.tensor(proj_positions, device=device)
    proj_vel = torch.tensor(proj_velocities, device=device)
    proj_mass = torch.ones(num_projectiles, device=device) * 0.01

    positions = torch.cat([wall_positions, proj_pos], dim=0)
    velocities = torch.cat([wall_velocities, proj_vel], dim=0)
    masses = torch.cat([wall_masses, proj_mass], dim=0)

    return positions.float(), velocities.float(), masses.float(), num_wall_stars


# =============================================================================
# REAL-TIME VISUALIZATION
# =============================================================================

class RealityGlitchVisualizer:
    """
    Real-time visualization of all four reality glitch tests.
    """

    def __init__(
        self,
        num_stars: int = 1000,
        test_mode: str = "all",  # "subnormal", "divergence", "entropy", "aliasing", "all"
        device: torch.device = None,
        log_interval: int = 50  # Print log every N ticks
    ):
        self.num_stars = num_stars
        self.test_mode = test_mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tick = 0
        self.log_interval = log_interval
        self.start_time = time.time()

        # Anomaly tracking for final report
        self.anomalies_detected = {
            "subnormal_floods": 0,
            "divergence_events": 0,
            "compression_spikes": 0,
            "clip_events": 0
        }

        # Initialize based on test mode
        self._setup_simulations()
        self._setup_figure()

        # Print header
        self._print_header()

    def _setup_simulations(self):
        """Initialize the appropriate simulations."""
        positions, velocities, masses = create_disk_galaxy(
            num_stars=self.num_stars,
            galaxy_radius=10.0,
            device=self.device
        )
        positions = positions.float()
        velocities = velocities.float()
        masses = masses.float()

        self.initial_positions = positions.clone()
        self.initial_velocities = velocities.clone()

        # Test 1: Subnormal - AGGRESSIVE MODE
        # Push values into 10^-38 to 10^-45 range to trigger denormal handling
        if self.test_mode in ["subnormal", "all"]:
            self.subnormal_sim = SubnormalStressSim(
                positions.clone(), velocities.clone(), masses.clone(),
                softening_scale=1e-35,  # Extreme: push deep into subnormal range
                subnormal_injection=True,  # Force subnormals into close pairs
                precision_mode=PrecisionMode.FLOAT32,
                G=0.001, dt=0.01, softening=1e-20, device=self.device  # Tiny softening
            )
            self.subnormal_baseline = SubnormalStressSim(
                positions.clone(), velocities.clone(), masses.clone(),
                softening_scale=1.0,
                subnormal_injection=False,
                precision_mode=PrecisionMode.FLOAT32,
                G=0.001, dt=0.01, softening=0.1, device=self.device
            )

        # Test 2: Divergence - Three parallel universes
        if self.test_mode in ["divergence", "all"]:
            self.multiverse = MultiverseSim(
                positions.clone(), velocities.clone(), masses.clone(),
                device=self.device
            )

        # Test 3: Entropy - with quantized physics to accelerate compression
        if self.test_mode in ["entropy", "all"]:
            # Use quantized sim to see faster entropy changes
            class QuantizedEntropySim(GalaxySimulation):
                def _compute_accelerations(self):
                    pos = self.positions
                    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
                    dist_sq = (diff ** 2).sum(dim=-1) + self.softening_sq
                    # Quantize to 64 levels to introduce lossy compression
                    dist_sq = _grid_quantize_safe(dist_sq, 64, min_val=0.01)
                    dist_cubed = dist_sq ** 1.5
                    force_factor = self.G / dist_cubed
                    force_factor = force_factor * self.masses.unsqueeze(0)
                    force_factor = force_factor * (1 - torch.eye(self.num_stars, device=self.device))
                    return (force_factor.unsqueeze(-1) * diff).sum(dim=1)

            self.entropy_sim = QuantizedEntropySim(
                positions.clone(), velocities.clone(), masses.clone(),
                precision_mode=PrecisionMode.FLOAT32,
                G=0.001, dt=0.01, softening=0.1, device=self.device
            )
            self.entropy_history = []

        # Test 4: Aliasing - HIGH SPEED projectile to trigger clipping
        # Key: projectile_speed * dt > wall_thickness causes "screen tearing"
        if self.test_mode in ["aliasing", "all"]:
            # dt=0.1, wall_thickness=1.0, so speed > 10 should clip
            # Using speed=50 to guarantee clipping at some point
            wall_pos, wall_vel, wall_mass, proj_idx = create_wall_galaxy(
                num_wall_stars=800,  # Denser wall
                wall_radius=8.0,
                projectile_speed=50.0,  # VERY FAST - will skip collision zone
                device=self.device
            )
            self.aliasing_sim = AliasingTestSim(
                wall_pos, wall_vel, wall_mass,
                wall_radius=8.0, wall_thickness=1.0,
                precision_mode=PrecisionMode.FLOAT32,
                G=0.0001, dt=0.1, softening=0.1, device=self.device  # Large dt!
            )
            self.aliasing_sim.projectile_idx = proj_idx
            self.aliasing_sim._last_proj_r = 16.0

    def _setup_figure(self):
        """Setup matplotlib figure based on test mode."""
        if self.test_mode == "all":
            self.fig = plt.figure(figsize=(18, 12))
            gs = self.fig.add_gridspec(3, 4, height_ratios=[2, 1, 1])

            # Row 1: Galaxy views for each test
            self.ax_subnormal = self.fig.add_subplot(gs[0, 0])
            self.ax_divergence = self.fig.add_subplot(gs[0, 1])
            self.ax_entropy = self.fig.add_subplot(gs[0, 2])
            self.ax_aliasing = self.fig.add_subplot(gs[0, 3])

            # Row 2: Metrics
            self.ax_subnormal_metric = self.fig.add_subplot(gs[1, 0])
            self.ax_divergence_metric = self.fig.add_subplot(gs[1, 1])
            self.ax_entropy_metric = self.fig.add_subplot(gs[1, 2])
            self.ax_aliasing_metric = self.fig.add_subplot(gs[1, 3])

            # Row 3: Combined analysis
            self.ax_combined = self.fig.add_subplot(gs[2, :2])
            self.ax_summary = self.fig.add_subplot(gs[2, 2:])

        else:
            self.fig = plt.figure(figsize=(14, 10))
            gs = self.fig.add_gridspec(2, 2)

            self.ax_main = self.fig.add_subplot(gs[0, :])
            self.ax_metric1 = self.fig.add_subplot(gs[1, 0])
            self.ax_metric2 = self.fig.add_subplot(gs[1, 1])

        self.fig.patch.set_facecolor('#0a0a0f')
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

    def _print_header(self):
        """Print the startup header to terminal."""
        print("\n" + "=" * 70)
        print("  REALITY GLITCH DETECTOR - Live Anomaly Log")
        print("  [AGGRESSIVE MODE - Pushing into danger zones]")
        print("=" * 70)
        print(f"  Mode: {self.test_mode.upper()}")
        print(f"  Stars: {self.num_stars}")
        print(f"  Device: {self.device}")
        print(f"  Log interval: every {self.log_interval} ticks")
        print()
        print("  TEST PARAMETERS:")
        if self.test_mode in ["subnormal", "all"]:
            print(f"    SUBNORMAL: softening=1e-20, injection=ON (target: 1e-40)")
        if self.test_mode in ["divergence", "all"]:
            print(f"    DIVERGENCE: 3 universes (FP32, reordered, FP16-intermediate)")
        if self.test_mode in ["entropy", "all"]:
            print(f"    ENTROPY: 64-level quantization (lossy physics)")
        if self.test_mode in ["aliasing", "all"]:
            print(f"    ALIASING: speed=50, dt=0.1 (skip_dist=5.0 > wall_thickness)")
        print("=" * 70)
        print()
        print(f"{'TICK':<8} {'TEST':<12} {'STATUS':<40} {'ANOMALY':<10}")
        print("-" * 70)

    def _print_log(self):
        """Print current status to terminal."""
        elapsed = time.time() - self.start_time
        tps = self.tick / elapsed if elapsed > 0 else 0

        # Subnormal status
        if self.test_mode in ["subnormal", "all"] and self.subnormal_sim.subnormal_history:
            sub = self.subnormal_sim.subnormal_history[-1]
            status = f"count={sub.subnormal_count:,} min={sub.min_nonzero:.2e}"
            anomaly = ""
            if sub.subnormal_count > 0:
                anomaly = "DENORMAL!"
                self.anomalies_detected["subnormal_floods"] += 1
            print(f"{self.tick:<8} {'SUBNORMAL':<12} {status:<40} {anomaly:<10}")

        # Divergence status
        if self.test_mode in ["divergence", "all"] and self.multiverse.divergence_history:
            div = self.multiverse.divergence_history[-1]
            status = f"max_diff={div.max_position_diff:.6f} entropy={div.entropy_bits:.1f}bits"
            anomaly = ""
            # FP16 intermediate will diverge fast, threshold at 5 bits
            if div.entropy_bits > 5:
                anomaly = "BUTTERFLY!"
                self.anomalies_detected["divergence_events"] += 1
            elif div.max_position_diff > 1e-5:
                anomaly = "DRIFT"
            print(f"{self.tick:<8} {'DIVERGENCE':<12} {status:<40} {anomaly:<10}")

        # Entropy status
        if self.test_mode in ["entropy", "all"] and self.entropy_history:
            ent = self.entropy_history[-1]
            # Track trend over last 10 measurements
            trend = ""
            if len(self.entropy_history) > 10:
                recent = [e.compression_ratio for e in self.entropy_history[-10:]]
                if recent[-1] > recent[0] * 1.05:
                    trend = "UP"
                elif recent[-1] < recent[0] * 0.95:
                    trend = "DN"
            status = f"ratio={ent.compression_ratio:.2f}x bits/star={ent.bits_per_star:.1f} {trend}"
            anomaly = ""
            # Lower threshold - quantized sim will compress differently
            if ent.compression_ratio > 1.8:
                anomaly = "LOSSY!"
                self.anomalies_detected["compression_spikes"] += 1
            print(f"{self.tick:<8} {'ENTROPY':<12} {status:<40} {anomaly:<10}")

        # Aliasing status
        if self.test_mode in ["aliasing", "all"] and self.aliasing_sim.clip_history:
            total_clips = sum(c.clip_events for c in self.aliasing_sim.clip_history)
            pos = self.aliasing_sim.positions.cpu().numpy()
            vel = self.aliasing_sim.velocities.cpu().numpy()
            proj_r = np.sqrt((pos[-1] ** 2).sum())
            proj_speed = np.sqrt((vel[-1] ** 2).sum())
            # Check if projectile passed wall without expected interaction
            dt = self.aliasing_sim.dt
            skip_dist = proj_speed * dt  # Distance traveled per tick
            status = f"r={proj_r:.1f} spd={proj_speed:.1f} skip={skip_dist:.1f} clips={total_clips}"
            anomaly = ""
            if total_clips > self.anomalies_detected["clip_events"]:
                anomaly = "CLIPPED!"
                self.anomalies_detected["clip_events"] = total_clips
            elif skip_dist > self.aliasing_sim.wall_thickness:
                anomaly = "RISK!"  # Could clip
            print(f"{self.tick:<8} {'ALIASING':<12} {status:<40} {anomaly:<10}")

        # Separator every log interval
        if self.test_mode == "all":
            print(f"{'─'*70}  [{tps:.1f} ticks/s]")

    def _print_final_report(self):
        """Print final summary report."""
        elapsed = time.time() - self.start_time

        print("\n" + "=" * 70)
        print("  FINAL REPORT - Reality Glitch Analysis")
        print("=" * 70)
        print(f"  Total ticks: {self.tick}")
        print(f"  Runtime: {elapsed:.1f}s ({self.tick/elapsed:.1f} ticks/s)")
        print()
        print("  ANOMALIES DETECTED:")
        print(f"    - Denormal floods:    {self.anomalies_detected['subnormal_floods']}")
        print(f"    - Divergence events:  {self.anomalies_detected['divergence_events']}")
        print(f"    - Compression spikes: {self.anomalies_detected['compression_spikes']}")
        print(f"    - Clip events:        {self.anomalies_detected['clip_events']}")
        print()

        # Verdict
        total_anomalies = sum(self.anomalies_detected.values())
        if total_anomalies == 0:
            print("  VERDICT: Reality appears STABLE at tested parameters.")
        elif total_anomalies < 5:
            print("  VERDICT: Minor glitches detected. Reality is MOSTLY STABLE.")
        elif total_anomalies < 20:
            print("  VERDICT: Significant anomalies! Reality may be SIMULATED.")
        else:
            print("  VERDICT: CRITICAL! Multiple reality boundaries breached!")
            print("           Strong evidence of computational substrate.")

        print("=" * 70 + "\n")

    def _update(self, frame):
        """Animation update function."""
        steps_per_frame = 3

        for _ in range(steps_per_frame):
            self._step_all()
            self.tick += 1

            # Print log at intervals
            if self.tick % self.log_interval == 0:
                self._print_log()

        if self.test_mode == "all":
            self._draw_all_tests()
        else:
            self._draw_single_test()

        return []

    def _step_all(self):
        """Step all active simulations."""
        if self.test_mode in ["subnormal", "all"]:
            self.subnormal_sim.step()
            self.subnormal_baseline.step()

        if self.test_mode in ["divergence", "all"]:
            self.multiverse.step()

        if self.test_mode in ["entropy", "all"]:
            self.entropy_sim.step()
            metrics = measure_state_entropy(
                self.entropy_sim.positions,
                self.entropy_sim.velocities
            )
            self.entropy_history.append(metrics)

        if self.test_mode in ["aliasing", "all"]:
            self.aliasing_sim.step()
            self.aliasing_sim.check_clipping()

    def _draw_all_tests(self):
        """Draw all four tests."""
        # === TEST 1: SUBNORMAL ===
        self.ax_subnormal.clear()
        pos = self.subnormal_sim.positions.cpu().numpy()
        self.ax_subnormal.scatter(pos[:, 0], pos[:, 1], s=1, c='cyan', alpha=0.6)
        self.ax_subnormal.set_xlim(-20, 20)
        self.ax_subnormal.set_ylim(-20, 20)
        self.ax_subnormal.set_facecolor('black')
        self.ax_subnormal.set_title("1. SUBNORMAL STRESS", color='cyan', fontsize=11, fontweight='bold')

        # Subnormal metrics
        self.ax_subnormal_metric.clear()
        if self.subnormal_sim.subnormal_history:
            counts = [m.subnormal_count for m in self.subnormal_sim.subnormal_history[-100:]]
            ops = [m.ops_per_second / 1e6 for m in self.subnormal_sim.subnormal_history[-100:]]

            ax2 = self.ax_subnormal_metric.twinx()
            self.ax_subnormal_metric.plot(counts, 'c-', label='Subnormals', linewidth=1.5)
            ax2.plot(ops, 'y-', alpha=0.7, label='MOps/s', linewidth=1)

            self.ax_subnormal_metric.set_ylabel("Subnormal Count", color='cyan', fontsize=9)
            ax2.set_ylabel("MOps/s", color='yellow', fontsize=9)
            ax2.tick_params(colors='yellow', labelsize=8)

            latest = self.subnormal_sim.subnormal_history[-1]
            self.ax_subnormal_metric.set_title(
                f"Min: {latest.min_nonzero:.2e}",
                color='white', fontsize=9
            )

        self.ax_subnormal_metric.set_facecolor('#0a0a1a')
        self.ax_subnormal_metric.tick_params(colors='cyan', labelsize=8)

        # === TEST 2: DIVERGENCE ===
        self.ax_divergence.clear()
        pos_a = self.multiverse.universe_a.positions.cpu().numpy()
        pos_b = self.multiverse.universe_b.positions.cpu().numpy()
        diff = np.sqrt(((pos_a - pos_b) ** 2).sum(axis=1))

        # Color by divergence
        scatter = self.ax_divergence.scatter(
            pos_a[:, 0], pos_a[:, 1], s=1, c=diff,
            cmap='hot', alpha=0.7, vmin=0, vmax=max(0.001, diff.max())
        )
        self.ax_divergence.set_xlim(-20, 20)
        self.ax_divergence.set_ylim(-20, 20)
        self.ax_divergence.set_facecolor('black')
        self.ax_divergence.set_title("2. MULTIVERSE DIVERGENCE", color='orange', fontsize=11, fontweight='bold')

        # Divergence metrics
        self.ax_divergence_metric.clear()
        if self.multiverse.divergence_history:
            divs = [d.max_position_diff for d in self.multiverse.divergence_history[-100:]]
            self.ax_divergence_metric.semilogy(divs, 'orange', linewidth=1.5)

            latest = self.multiverse.divergence_history[-1]
            self.ax_divergence_metric.set_title(
                f"Entropy: {latest.entropy_bits:.1f} bits | Rate: {latest.divergence_rate:.4f}",
                color='white', fontsize=9
            )

            # Mark "Entropy Horizon" if divergence is significant
            if latest.max_position_diff > 0.1:
                self.ax_divergence_metric.axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
                self.ax_divergence_metric.text(
                    len(divs) * 0.7, 0.15, "ENTROPY HORIZON",
                    color='red', fontsize=8
                )

        self.ax_divergence_metric.set_facecolor('#0a0a1a')
        self.ax_divergence_metric.set_ylabel("Max Divergence", color='orange', fontsize=9)
        self.ax_divergence_metric.tick_params(colors='orange', labelsize=8)

        # === TEST 3: ENTROPY ===
        self.ax_entropy.clear()
        pos = self.entropy_sim.positions.cpu().numpy()
        self.ax_entropy.scatter(pos[:, 0], pos[:, 1], s=1, c='lime', alpha=0.6)
        self.ax_entropy.set_xlim(-20, 20)
        self.ax_entropy.set_ylim(-20, 20)
        self.ax_entropy.set_facecolor('black')
        self.ax_entropy.set_title("3. ENTROPY / COMPLEXITY", color='lime', fontsize=11, fontweight='bold')

        # Entropy metrics
        self.ax_entropy_metric.clear()
        if self.entropy_history:
            ratios = [e.compression_ratio for e in self.entropy_history[-100:]]
            self.ax_entropy_metric.plot(ratios, 'lime', linewidth=1.5)
            self.ax_entropy_metric.axhline(y=1.0, color='white', linestyle='--', alpha=0.3)

            latest = self.entropy_history[-1]
            trend = "COMPRESSING" if len(ratios) > 10 and ratios[-1] > ratios[-10] else "EXPANDING"
            color = 'red' if trend == "COMPRESSING" else 'lime'

            self.ax_entropy_metric.set_title(
                f"Ratio: {latest.compression_ratio:.2f}x | {latest.bits_per_star:.1f} bits/star | {trend}",
                color=color, fontsize=9
            )

        self.ax_entropy_metric.set_facecolor('#0a0a1a')
        self.ax_entropy_metric.set_ylabel("Compression Ratio", color='lime', fontsize=9)
        self.ax_entropy_metric.tick_params(colors='lime', labelsize=8)

        # === TEST 4: ALIASING ===
        self.ax_aliasing.clear()
        pos = self.aliasing_sim.positions.cpu().numpy()

        # Draw wall as circle
        wall_circle = Circle((0, 0), self.aliasing_sim.wall_radius,
                             fill=False, color='red', linewidth=2, linestyle='--')
        self.ax_aliasing.add_patch(wall_circle)

        # Wall stars in red, projectile in yellow
        wall_stars = pos[:-1]
        projectile = pos[-1]

        self.ax_aliasing.scatter(wall_stars[:, 0], wall_stars[:, 1], s=1, c='red', alpha=0.5)
        self.ax_aliasing.scatter([projectile[0]], [projectile[1]], s=50, c='yellow', marker='*')

        self.ax_aliasing.set_xlim(-15, 15)
        self.ax_aliasing.set_ylim(-15, 15)
        self.ax_aliasing.set_facecolor('black')
        self.ax_aliasing.set_title("4. SPATIAL ALIASING", color='red', fontsize=11, fontweight='bold')

        # Aliasing metrics
        self.ax_aliasing_metric.clear()
        if self.aliasing_sim.clip_history:
            clips = [c.clip_events for c in self.aliasing_sim.clip_history]
            total_clips = sum(clips)

            # Show projectile trajectory
            proj_r = np.sqrt(projectile[0]**2 + projectile[1]**2)

            self.ax_aliasing_metric.bar(['Clips', 'Near Miss'],
                                        [total_clips, sum(c.near_misses for c in self.aliasing_sim.clip_history)],
                                        color=['red', 'yellow'])

            self.ax_aliasing_metric.set_title(
                f"Projectile R: {proj_r:.2f} | Wall R: {self.aliasing_sim.wall_radius}",
                color='white', fontsize=9
            )

            if total_clips > 0:
                self.ax_aliasing_metric.text(
                    0.5, 0.5, "CLIP DETECTED!", transform=self.ax_aliasing_metric.transAxes,
                    ha='center', va='center', fontsize=14, color='red', fontweight='bold'
                )

        self.ax_aliasing_metric.set_facecolor('#0a0a1a')
        self.ax_aliasing_metric.tick_params(colors='white', labelsize=8)

        # === COMBINED ANALYSIS ===
        self.ax_combined.clear()

        # Plot all metrics normalized
        if (self.multiverse.divergence_history and self.entropy_history and
            self.subnormal_sim.subnormal_history):

            n = min(len(self.multiverse.divergence_history),
                   len(self.entropy_history),
                   len(self.subnormal_sim.subnormal_history))

            if n > 5:
                x = range(n)

                # Normalize each metric
                div = np.array([d.max_position_diff for d in self.multiverse.divergence_history[:n]])
                ent = np.array([e.compression_ratio for e in self.entropy_history[:n]])
                sub = np.array([s.subnormal_count for s in self.subnormal_sim.subnormal_history[:n]])

                div_norm = div / (div.max() + 1e-10)
                ent_norm = (ent - ent.min()) / (ent.max() - ent.min() + 1e-10)
                sub_norm = sub / (sub.max() + 1e-10)

                self.ax_combined.plot(x, div_norm, 'orange', label='Divergence', linewidth=1.5)
                self.ax_combined.plot(x, ent_norm, 'lime', label='Entropy', linewidth=1.5)
                self.ax_combined.plot(x, sub_norm, 'cyan', label='Subnormals', linewidth=1.5)

                self.ax_combined.legend(loc='upper left', fontsize=8)

        self.ax_combined.set_facecolor('#0a0a1a')
        self.ax_combined.set_title("COMBINED ANOMALY TRACKING", color='white', fontsize=11)
        self.ax_combined.tick_params(colors='white', labelsize=8)

        # === SUMMARY ===
        self.ax_summary.clear()
        self.ax_summary.axis('off')

        summary = f"""
        REALITY GLITCH DETECTOR
        ═══════════════════════════════════
        Tick: {self.tick}
        Stars: {self.num_stars}

        TEST STATUS:
        """

        if self.subnormal_sim.subnormal_history:
            sub = self.subnormal_sim.subnormal_history[-1]
            summary += f"\n  1. Subnormals: {sub.subnormal_count:,}"
            if sub.subnormal_count > 0:
                summary += " [DENORMAL FLOOD]"

        if self.multiverse.divergence_history:
            div = self.multiverse.divergence_history[-1]
            summary += f"\n  2. Divergence: {div.max_position_diff:.6f}"
            if div.entropy_bits > 10:
                summary += " [BUTTERFLY EFFECT]"

        if self.entropy_history:
            ent = self.entropy_history[-1]
            summary += f"\n  3. Compression: {ent.compression_ratio:.2f}x"
            if ent.compression_ratio > 2.0:
                summary += " [LOSSY UNIVERSE]"

        if self.aliasing_sim.clip_history:
            clips = sum(c.clip_events for c in self.aliasing_sim.clip_history)
            summary += f"\n  4. Clip Events: {clips}"
            if clips > 0:
                summary += " [REALITY TEAR]"

        self.ax_summary.text(
            0.05, 0.95, summary, transform=self.ax_summary.transAxes,
            fontsize=10, fontfamily='monospace', color='white',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8)
        )

        self.fig.suptitle(
            f"REALITY GLITCH TESTS - Tick {self.tick}",
            fontsize=14, fontweight='bold', color='white'
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

    def _draw_single_test(self):
        """Draw a single test in detail."""
        self.ax_main.clear()
        self.ax_metric1.clear()
        self.ax_metric2.clear()

        if self.test_mode == "subnormal":
            self._draw_subnormal_detail()
        elif self.test_mode == "divergence":
            self._draw_divergence_detail()
        elif self.test_mode == "entropy":
            self._draw_entropy_detail()
        elif self.test_mode == "aliasing":
            self._draw_aliasing_detail()

        self.fig.suptitle(
            f"REALITY GLITCH: {self.test_mode.upper()} - Tick {self.tick}",
            fontsize=14, fontweight='bold', color='white'
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])

    def _draw_subnormal_detail(self):
        """Detailed subnormal test view."""
        # Main: Side-by-side baseline vs stressed
        pos_baseline = self.subnormal_baseline.positions.cpu().numpy()
        pos_stress = self.subnormal_sim.positions.cpu().numpy()

        self.ax_main.scatter(pos_baseline[:, 0] - 12, pos_baseline[:, 1],
                            s=1, c='green', alpha=0.6, label='Normal')
        self.ax_main.scatter(pos_stress[:, 0] + 12, pos_stress[:, 1],
                            s=1, c='cyan', alpha=0.6, label='Subnormal Stress')

        self.ax_main.axvline(x=0, color='white', linestyle='--', alpha=0.3)
        self.ax_main.set_xlim(-30, 30)
        self.ax_main.set_ylim(-20, 20)
        self.ax_main.set_facecolor('black')
        self.ax_main.legend(loc='upper center', ncol=2)
        self.ax_main.set_title("Normal vs Subnormal-Stressed Physics", color='cyan')

        # Metric 1: Subnormal count over time
        if self.subnormal_sim.subnormal_history:
            counts = [m.subnormal_count for m in self.subnormal_sim.subnormal_history]
            self.ax_metric1.plot(counts, 'c-', linewidth=2)
            self.ax_metric1.fill_between(range(len(counts)), counts, alpha=0.3, color='cyan')
            self.ax_metric1.set_title("Subnormal Count (Denormal Flooding)", color='cyan')
            self.ax_metric1.set_xlabel("Tick")
            self.ax_metric1.set_ylabel("Count")
        self.ax_metric1.set_facecolor('#0a0a1a')
        self.ax_metric1.tick_params(colors='white')

        # Metric 2: Operations per second (performance cliff detection)
        if self.subnormal_sim.ops_times:
            times = self.subnormal_sim.ops_times
            baseline_times = self.subnormal_baseline.ops_times if hasattr(self.subnormal_baseline, 'ops_times') else []

            self.ax_metric2.plot(times, 'r-', label='Stressed', linewidth=2)
            if baseline_times:
                self.ax_metric2.plot(baseline_times, 'g-', label='Normal', linewidth=2)

            self.ax_metric2.set_title("Compute Time (Performance Cliff)", color='yellow')
            self.ax_metric2.set_xlabel("Tick")
            self.ax_metric2.set_ylabel("Time (s)")
            self.ax_metric2.legend()
        self.ax_metric2.set_facecolor('#0a0a1a')
        self.ax_metric2.tick_params(colors='white')

    def _draw_divergence_detail(self):
        """Detailed divergence test view."""
        pos_a = self.multiverse.universe_a.positions.cpu().numpy()
        pos_b = self.multiverse.universe_b.positions.cpu().numpy()
        diff = np.sqrt(((pos_a - pos_b) ** 2).sum(axis=1))

        # Main: Overlay both universes
        self.ax_main.scatter(pos_a[:, 0], pos_a[:, 1], s=2, c='blue', alpha=0.5, label='Universe A')
        self.ax_main.scatter(pos_b[:, 0], pos_b[:, 1], s=2, c='red', alpha=0.5, label='Universe B')
        self.ax_main.set_xlim(-20, 20)
        self.ax_main.set_ylim(-20, 20)
        self.ax_main.set_facecolor('black')
        self.ax_main.legend(loc='upper right')
        self.ax_main.set_title("Parallel Universes (Same Initial Conditions)", color='orange')

        # Metric 1: Divergence over time (log scale)
        if self.multiverse.divergence_history:
            divs = [d.max_position_diff for d in self.multiverse.divergence_history]
            self.ax_metric1.semilogy(divs, 'orange', linewidth=2)

            # Fit exponential to show Lyapunov exponent
            if len(divs) > 20:
                x = np.arange(len(divs))
                log_divs = np.log(np.array(divs) + 1e-15)
                coeffs = np.polyfit(x[-50:], log_divs[-50:], 1)
                fit_line = np.exp(coeffs[1] + coeffs[0] * x)
                self.ax_metric1.semilogy(x, fit_line, 'r--', alpha=0.7,
                                         label=f'Lyapunov: {coeffs[0]:.4f}')
                self.ax_metric1.legend()

            self.ax_metric1.set_title("Butterfly Effect (Exponential Divergence)", color='orange')
            self.ax_metric1.set_xlabel("Tick")
            self.ax_metric1.set_ylabel("Max Position Difference (log)")
        self.ax_metric1.set_facecolor('#0a0a1a')
        self.ax_metric1.tick_params(colors='white')

        # Metric 2: Entropy bits accumulated
        if self.multiverse.divergence_history:
            entropy = [d.entropy_bits for d in self.multiverse.divergence_history]
            self.ax_metric2.plot(entropy, 'purple', linewidth=2)
            self.ax_metric2.axhline(y=23, color='red', linestyle='--', alpha=0.5, label='float32 mantissa bits')
            self.ax_metric2.set_title("Entropy Horizon (Bits of Uncertainty)", color='purple')
            self.ax_metric2.set_xlabel("Tick")
            self.ax_metric2.set_ylabel("Entropy Bits")
            self.ax_metric2.legend()
        self.ax_metric2.set_facecolor('#0a0a1a')
        self.ax_metric2.tick_params(colors='white')

    def _draw_entropy_detail(self):
        """Detailed entropy test view."""
        pos = self.entropy_sim.positions.cpu().numpy()

        # Main: Galaxy colored by local density (proxy for structure)
        self.ax_main.scatter(pos[:, 0], pos[:, 1], s=2, c='lime', alpha=0.6)
        self.ax_main.set_xlim(-20, 20)
        self.ax_main.set_ylim(-20, 20)
        self.ax_main.set_facecolor('black')
        self.ax_main.set_title("Galaxy State (Measuring Compressibility)", color='lime')

        # Metric 1: Compression ratio over time
        if self.entropy_history:
            ratios = [e.compression_ratio for e in self.entropy_history]
            self.ax_metric1.plot(ratios, 'lime', linewidth=2)
            self.ax_metric1.axhline(y=1.0, color='white', linestyle='--', alpha=0.3)

            # Trend line
            if len(ratios) > 20:
                x = np.arange(len(ratios))
                coeffs = np.polyfit(x, ratios, 1)
                trend = coeffs[0]
                trend_str = "COMPRESSING" if trend > 0 else "EXPANDING"
                color = 'red' if trend > 0 else 'cyan'
                self.ax_metric1.text(len(ratios) * 0.7, max(ratios) * 0.9,
                                    f"{trend_str}\nSlope: {trend:.6f}",
                                    color=color, fontsize=10)

            self.ax_metric1.set_title("Kolmogorov Complexity Proxy", color='lime')
            self.ax_metric1.set_xlabel("Tick")
            self.ax_metric1.set_ylabel("Compression Ratio")
        self.ax_metric1.set_facecolor('#0a0a1a')
        self.ax_metric1.tick_params(colors='white')

        # Metric 2: Bits per star
        if self.entropy_history:
            bits = [e.bits_per_star for e in self.entropy_history]
            self.ax_metric2.plot(bits, 'yellow', linewidth=2)
            self.ax_metric2.set_title("Information Density (Bits per Star)", color='yellow')
            self.ax_metric2.set_xlabel("Tick")
            self.ax_metric2.set_ylabel("Bits/Star")
        self.ax_metric2.set_facecolor('#0a0a1a')
        self.ax_metric2.tick_params(colors='white')

    def _draw_aliasing_detail(self):
        """Detailed aliasing test view."""
        pos = self.aliasing_sim.positions.cpu().numpy()
        wall_stars = pos[:-1]
        projectile = pos[-1]

        # Main: Wall and projectile with trajectory
        wall_circle = Circle((0, 0), self.aliasing_sim.wall_radius,
                             fill=False, color='red', linewidth=3)
        self.ax_main.add_patch(wall_circle)

        self.ax_main.scatter(wall_stars[:, 0], wall_stars[:, 1], s=3, c='red', alpha=0.5)
        self.ax_main.scatter([projectile[0]], [projectile[1]], s=100, c='yellow', marker='*', zorder=10)

        # Draw trajectory line
        self.ax_main.plot([0, projectile[0]], [-16, projectile[1]], 'y--', alpha=0.5)

        self.ax_main.set_xlim(-18, 18)
        self.ax_main.set_ylim(-18, 18)
        self.ax_main.set_facecolor('black')
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title("Projectile vs Wall (Testing for Clipping)", color='red')

        # Metric 1: Projectile radius over time
        proj_r = np.sqrt(projectile[0]**2 + projectile[1]**2)

        self.ax_metric1.axhline(y=self.aliasing_sim.wall_radius, color='red', linewidth=2, label='Wall')
        self.ax_metric1.axhspan(
            self.aliasing_sim.wall_radius - self.aliasing_sim.wall_thickness/2,
            self.aliasing_sim.wall_radius + self.aliasing_sim.wall_thickness/2,
            color='red', alpha=0.3
        )
        self.ax_metric1.axhline(y=proj_r, color='yellow', linewidth=2, label=f'Projectile R={proj_r:.2f}')
        self.ax_metric1.set_ylim(0, 20)
        self.ax_metric1.legend()
        self.ax_metric1.set_title("Radial Position", color='yellow')
        self.ax_metric1.set_facecolor('#0a0a1a')
        self.ax_metric1.tick_params(colors='white')

        # Metric 2: Clip events
        if self.aliasing_sim.clip_history:
            clips = [c.clip_events for c in self.aliasing_sim.clip_history]
            cumulative = np.cumsum(clips)
            self.ax_metric2.plot(cumulative, 'r-', linewidth=2)
            self.ax_metric2.set_title(f"Cumulative Clips: {cumulative[-1] if len(cumulative) > 0 else 0}", color='red')
            self.ax_metric2.set_xlabel("Tick")
            self.ax_metric2.set_ylabel("Total Clip Events")
        self.ax_metric2.set_facecolor('#0a0a1a')
        self.ax_metric2.tick_params(colors='white')

    def run(self, interval: int = 50, max_frames: int = None):
        """Run the visualization."""
        print("\n  Close the visualization window to see final report.\n")

        self.ani = FuncAnimation(
            self.fig,
            self._update,
            frames=max_frames,
            interval=interval,
            blit=False,
            repeat=False
        )

        plt.show()

        # Print final report when window closes
        self._print_final_report()


def main():
    parser = argparse.ArgumentParser(
        description="Reality Glitch Tests - Real-Time Anomaly Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tests:
  subnormal  - Denormal flooding (10^-38 to 10^-45)
  divergence - FP determinism / butterfly effect
  entropy    - Kolmogorov complexity / compressibility
  aliasing   - Spatial aliasing / particle clipping
  all        - Run all tests simultaneously
        """
    )
    parser.add_argument("--test", type=str, default="all",
                        choices=["subnormal", "divergence", "entropy", "aliasing", "all"],
                        help="Which test to run")
    parser.add_argument("--stars", type=int, default=1000, help="Number of stars")
    parser.add_argument("--interval", type=int, default=50, help="Animation interval (ms)")
    parser.add_argument("--log-interval", type=int, default=50, help="Print log every N ticks")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    viz = RealityGlitchVisualizer(
        num_stars=args.stars,
        test_mode=args.test,
        device=device,
        log_interval=args.log_interval
    )

    viz.run(interval=args.interval)


if __name__ == "__main__":
    main()
