"""
EXTREME MODE - ABSOLUTE LIMITS
==============================

No more safe testing. This pushes hardware to the breaking point.

TESTS:
1. SUBNORMAL HELL     - Actual FP32 minimum: 2^-149 = 1.4e-45
2. INFINITY CASCADE   - What happens when we hit inf? How fast does it spread?
3. NAN APOCALYPSE     - Inject NaN, watch reality dissolve
4. PRECISION MASSACRE - Chain FP64->FP32->FP16->BF16->INT8->FP32
5. MEMORY ARMAGEDDON  - Fill VRAM until GPU screams
6. QUANTUM CHAOS      - Random operation order every tick
7. TIME DILATION      - Variable dt to find resonance frequencies
8. SINGULARITY HUNT   - Push softening to actual zero

WARNING: This WILL crash your simulation. That's the point.
         We're looking for WHERE it breaks.

Usage:
    python extreme_mode.py --test all
    python extreme_mode.py --test singularity --stars 5000
    python extreme_mode.py --find-crash-point
"""

import argparse
import time
import sys
import gc
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import struct

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# FP32 Limits - THE ACTUAL BOUNDARIES
FP32_MAX = 3.4028235e+38
FP32_MIN_NORMAL = 1.1754944e-38  # 2^-126
FP32_MIN_SUBNORMAL = 1.4012985e-45  # 2^-149 - THE ABSOLUTE MINIMUM
FP32_EPSILON = 1.1920929e-07  # 2^-23

# Try to import from parent
try:
    from galaxy import create_disk_galaxy
    from simulation import GalaxySimulation
    from quantization import PrecisionMode
except ImportError:
    print("Run from lossy_galaxy directory")
    sys.exit(1)


@dataclass
class CrashPoint:
    """Record of where reality broke."""
    test_name: str
    parameter: str
    safe_value: float
    crash_value: float
    crash_type: str  # "nan", "inf", "explode", "freeze", "oom"
    tick: int
    error_message: str = ""


@dataclass
class ExtremeMetrics:
    """Metrics from extreme testing."""
    tick: int = 0
    nan_count: int = 0
    inf_count: int = 0
    subnormal_count: int = 0
    max_value: float = 0.0
    min_value: float = 0.0
    energy: float = 0.0
    crashed: bool = False
    crash_reason: str = ""


def count_extreme_values(tensor: torch.Tensor) -> Tuple[int, int, int, float, float]:
    """Count NaN, Inf, and Subnormal values."""
    flat = tensor.flatten()

    nan_count = torch.isnan(flat).sum().item()
    inf_count = torch.isinf(flat).sum().item()

    # Subnormals: |x| < FP32_MIN_NORMAL and x != 0
    abs_vals = flat.abs()
    finite_mask = torch.isfinite(flat)
    nonzero_mask = abs_vals > 0
    subnormal_mask = (abs_vals < FP32_MIN_NORMAL) & nonzero_mask & finite_mask
    subnormal_count = subnormal_mask.sum().item()

    # Max/min of finite values
    finite_vals = flat[finite_mask]
    if len(finite_vals) > 0:
        max_val = finite_vals.abs().max().item()
        min_nonzero = finite_vals[finite_vals.abs() > 0].abs().min().item() if (finite_vals.abs() > 0).any() else 0
    else:
        max_val = float('inf')
        min_nonzero = 0

    return nan_count, inf_count, subnormal_count, max_val, min_nonzero


# =============================================================================
# TEST 1: SUBNORMAL HELL
# =============================================================================

class SubnormalHellSim:
    """Push values to the actual FP32 subnormal limit."""

    def __init__(self, num_stars: int, device: torch.device):
        self.device = device
        self.num_stars = num_stars

        # Start with normal galaxy
        torch.manual_seed(42)
        self.positions = (torch.rand(num_stars, 3, device=device) - 0.5) * 20
        self.velocities = (torch.rand(num_stars, 3, device=device) - 0.5) * 0.1
        self.masses = torch.ones(num_stars, device=device) * 0.001

        # Parameters that will push to subnormals
        self.G = 0.001
        self.dt = 0.01

        # Start with small softening, will decrease each tick
        self.softening = 1e-10
        self.min_softening_reached = self.softening

        self.tick = 0
        self.metrics_history = []

    def step(self) -> ExtremeMetrics:
        """Step simulation, decreasing softening toward subnormal range."""
        self.tick += 1

        # AGGRESSIVE: Decrease softening toward absolute minimum each tick
        self.softening *= 0.95  # Exponential decrease
        if self.softening < FP32_MIN_SUBNORMAL:
            self.softening = FP32_MIN_SUBNORMAL
        self.min_softening_reached = min(self.min_softening_reached, self.softening)

        softening_sq = torch.tensor(self.softening ** 2, device=self.device, dtype=torch.float32)

        try:
            with torch.no_grad():
                diff = self.positions.unsqueeze(0) - self.positions.unsqueeze(1)
                dist_sq = (diff ** 2).sum(dim=-1) + softening_sq

                # Track extreme values in dist_sq
                nan_c, inf_c, sub_c, max_v, min_v = count_extreme_values(dist_sq)

                dist_cubed = dist_sq ** 1.5
                force_factor = self.G / dist_cubed
                force_factor = force_factor * self.masses.unsqueeze(0)
                force_factor = force_factor * (1 - torch.eye(self.num_stars, device=self.device))

                accelerations = (force_factor.unsqueeze(-1) * diff).sum(dim=1)

                self.velocities = self.velocities + accelerations * self.dt
                self.positions = self.positions + self.velocities * self.dt

                # Check for crash
                pos_nan, pos_inf, _, _, _ = count_extreme_values(self.positions)

                metrics = ExtremeMetrics(
                    tick=self.tick,
                    nan_count=nan_c + pos_nan,
                    inf_count=inf_c + pos_inf,
                    subnormal_count=sub_c,
                    max_value=max_v,
                    min_value=min_v,
                    crashed=(pos_nan > 0 or pos_inf > 0)
                )

                if metrics.crashed:
                    metrics.crash_reason = f"NaN={pos_nan}, Inf={pos_inf} at softening={self.softening:.2e}"

        except Exception as e:
            metrics = ExtremeMetrics(
                tick=self.tick,
                crashed=True,
                crash_reason=str(e)
            )

        self.metrics_history.append(metrics)
        return metrics


# =============================================================================
# TEST 2: INFINITY CASCADE
# =============================================================================

class InfinityCascadeSim:
    """Inject infinity and watch it spread through the simulation."""

    def __init__(self, num_stars: int, device: torch.device):
        self.device = device
        self.num_stars = num_stars

        torch.manual_seed(42)
        self.positions = (torch.rand(num_stars, 3, device=device) - 0.5) * 20
        self.velocities = (torch.rand(num_stars, 3, device=device) - 0.5) * 0.1
        self.masses = torch.ones(num_stars, device=device) * 0.001

        self.G = 0.001
        self.dt = 0.01
        self.softening_sq = 0.01

        self.tick = 0
        self.inf_injected = False
        self.injection_tick = 50  # Inject inf at tick 50
        self.inf_spread_history = []

    def step(self) -> ExtremeMetrics:
        self.tick += 1

        # Inject infinity at specified tick
        if self.tick == self.injection_tick and not self.inf_injected:
            # Set one star's position to infinity
            self.positions[0, 0] = float('inf')
            self.inf_injected = True
            print(f"  [TICK {self.tick}] INFINITY INJECTED into star 0")

        try:
            with torch.no_grad():
                diff = self.positions.unsqueeze(0) - self.positions.unsqueeze(1)
                dist_sq = (diff ** 2).sum(dim=-1) + self.softening_sq
                dist_cubed = dist_sq ** 1.5
                force_factor = self.G / dist_cubed
                force_factor = force_factor * self.masses.unsqueeze(0)
                force_factor = force_factor * (1 - torch.eye(self.num_stars, device=self.device))
                accelerations = (force_factor.unsqueeze(-1) * diff).sum(dim=1)

                self.velocities = self.velocities + accelerations * self.dt
                self.positions = self.positions + self.velocities * self.dt

                # Count how many stars are now infinite
                inf_stars = torch.isinf(self.positions).any(dim=1).sum().item()
                nan_stars = torch.isnan(self.positions).any(dim=1).sum().item()

                self.inf_spread_history.append(inf_stars + nan_stars)

                metrics = ExtremeMetrics(
                    tick=self.tick,
                    inf_count=inf_stars,
                    nan_count=nan_stars,
                    crashed=(inf_stars + nan_stars >= self.num_stars * 0.5)
                )

                if inf_stars + nan_stars > 0:
                    metrics.crash_reason = f"{inf_stars} inf, {nan_stars} nan stars ({(inf_stars+nan_stars)/self.num_stars*100:.1f}% infected)"

        except Exception as e:
            metrics = ExtremeMetrics(tick=self.tick, crashed=True, crash_reason=str(e))

        return metrics


# =============================================================================
# TEST 3: PRECISION MASSACRE
# =============================================================================

class PrecisionMassacreSim:
    """Chain precision conversions: FP64->FP32->FP16->BF16->INT8->FP32"""

    def __init__(self, num_stars: int, device: torch.device):
        self.device = device
        self.num_stars = num_stars

        # Start in FP64 for maximum precision baseline
        torch.manual_seed(42)
        self.positions = (torch.rand(num_stars, 3, device=device, dtype=torch.float64) - 0.5) * 20
        self.velocities = (torch.rand(num_stars, 3, device=device, dtype=torch.float64) - 0.5) * 0.1
        self.masses = torch.ones(num_stars, device=device, dtype=torch.float64) * 0.001

        # Keep a "clean" FP64 reference
        self.ref_positions = self.positions.clone()
        self.ref_velocities = self.velocities.clone()

        self.G = 0.001
        self.dt = 0.01
        self.softening_sq = 0.01

        self.tick = 0
        self.precision_loss_history = []

    def _massacre_precision(self, tensor: torch.Tensor) -> torch.Tensor:
        """Chain of precision destruction."""
        # FP64 -> FP32
        t = tensor.float()
        # FP32 -> FP16
        t = t.half()
        # FP16 -> BF16 (if available)
        if hasattr(torch, 'bfloat16'):
            t = t.to(torch.bfloat16)
            t = t.half()  # Back to FP16
        # FP16 -> INT8 (quantize)
        t_int8 = (t * 127).to(torch.int8)
        # INT8 -> FP32
        t = t_int8.float() / 127
        return t.double()  # Back to FP64 for comparison

    def step(self) -> ExtremeMetrics:
        self.tick += 1

        try:
            with torch.no_grad():
                # Update reference (clean FP64)
                diff_ref = self.ref_positions.unsqueeze(0) - self.ref_positions.unsqueeze(1)
                dist_sq_ref = (diff_ref ** 2).sum(dim=-1) + self.softening_sq
                dist_cubed_ref = dist_sq_ref ** 1.5
                force_factor_ref = self.G / dist_cubed_ref
                force_factor_ref = force_factor_ref * self.masses.unsqueeze(0)
                force_factor_ref = force_factor_ref * (1 - torch.eye(self.num_stars, device=self.device, dtype=torch.float64))
                acc_ref = (force_factor_ref.unsqueeze(-1) * diff_ref).sum(dim=1)
                self.ref_velocities = self.ref_velocities + acc_ref * self.dt
                self.ref_positions = self.ref_positions + self.ref_velocities * self.dt

                # Update massacred version
                diff = self.positions.unsqueeze(0) - self.positions.unsqueeze(1)
                dist_sq = (diff ** 2).sum(dim=-1) + self.softening_sq

                # MASSACRE THE PRECISION
                dist_sq = self._massacre_precision(dist_sq)

                dist_cubed = dist_sq ** 1.5
                force_factor = self.G / dist_cubed
                force_factor = force_factor * self.masses.unsqueeze(0)
                force_factor = force_factor * (1 - torch.eye(self.num_stars, device=self.device, dtype=torch.float64))
                acc = (force_factor.unsqueeze(-1) * diff).sum(dim=1)

                # More precision massacre
                acc = self._massacre_precision(acc)

                self.velocities = self.velocities + acc * self.dt
                self.positions = self.positions + self.velocities * self.dt

                # Measure precision loss
                pos_diff = (self.positions - self.ref_positions).abs().max().item()
                vel_diff = (self.velocities - self.ref_velocities).abs().max().item()

                self.precision_loss_history.append(pos_diff)

                metrics = ExtremeMetrics(
                    tick=self.tick,
                    max_value=pos_diff,
                    min_value=vel_diff,
                )

                # Check for catastrophic loss
                if pos_diff > 1.0:  # More than 1 unit of position error
                    metrics.crashed = True
                    metrics.crash_reason = f"Precision loss: {pos_diff:.2e} position error"

        except Exception as e:
            metrics = ExtremeMetrics(tick=self.tick, crashed=True, crash_reason=str(e))

        return metrics


# =============================================================================
# TEST 4: SINGULARITY HUNT
# =============================================================================

class SingularityHuntSim:
    """Push softening to ZERO and find exactly where singularities form."""

    def __init__(self, num_stars: int, device: torch.device):
        self.device = device
        self.num_stars = num_stars

        # Create galaxy with some stars VERY close together
        torch.manual_seed(42)
        self.positions = (torch.rand(num_stars, 3, device=device) - 0.5) * 20

        # Force some close encounters
        for i in range(0, min(20, num_stars), 2):
            self.positions[i+1] = self.positions[i] + torch.rand(3, device=device) * 0.001

        self.velocities = (torch.rand(num_stars, 3, device=device) - 0.5) * 0.1
        self.masses = torch.ones(num_stars, device=device) * 0.001

        self.G = 0.001
        self.dt = 0.01

        # Start with zero softening - pure 1/r^2
        self.softening = 0.0

        self.tick = 0
        self.singularity_events = []

    def step(self) -> ExtremeMetrics:
        self.tick += 1

        try:
            with torch.no_grad():
                diff = self.positions.unsqueeze(0) - self.positions.unsqueeze(1)
                dist_sq = (diff ** 2).sum(dim=-1)

                # NO SOFTENING - pure singularities possible
                # dist_sq can be exactly 0 for coincident particles

                # Find minimum non-self distance
                dist_sq_masked = dist_sq + torch.eye(self.num_stars, device=self.device) * 1e10
                min_dist_sq = dist_sq_masked.min().item()

                # Add tiny softening only to prevent div by zero
                dist_sq = dist_sq + 1e-45  # Actual FP32 minimum

                dist_cubed = dist_sq ** 1.5
                force_factor = self.G / dist_cubed

                # Check for extreme forces (singularity indicators)
                max_force = force_factor.max().item()

                force_factor = force_factor * self.masses.unsqueeze(0)
                force_factor = force_factor * (1 - torch.eye(self.num_stars, device=self.device))
                accelerations = (force_factor.unsqueeze(-1) * diff).sum(dim=1)

                max_accel = accelerations.abs().max().item()

                self.velocities = self.velocities + accelerations * self.dt
                self.positions = self.positions + self.velocities * self.dt

                # Track singularity events
                if max_force > 1e10 or max_accel > 1e10:
                    self.singularity_events.append({
                        'tick': self.tick,
                        'min_dist': np.sqrt(min_dist_sq),
                        'max_force': max_force,
                        'max_accel': max_accel
                    })

                nan_c, inf_c, _, _, _ = count_extreme_values(self.positions)

                metrics = ExtremeMetrics(
                    tick=self.tick,
                    nan_count=nan_c,
                    inf_count=inf_c,
                    max_value=max_accel,
                    min_value=np.sqrt(min_dist_sq),
                    crashed=(nan_c > 0 or inf_c > 0)
                )

                if metrics.crashed:
                    metrics.crash_reason = f"Singularity at min_dist={np.sqrt(min_dist_sq):.2e}"

        except Exception as e:
            metrics = ExtremeMetrics(tick=self.tick, crashed=True, crash_reason=str(e))

        return metrics


# =============================================================================
# TEST 5: MEMORY ARMAGEDDON
# =============================================================================

class MemoryArmageddonSim:
    """Fill VRAM until the GPU dies."""

    def __init__(self, device: torch.device):
        self.device = device
        self.tick = 0
        self.tensors = []
        self.total_allocated = 0
        self.crash_point = None

    def step(self) -> ExtremeMetrics:
        self.tick += 1

        try:
            # Allocate 100MB each tick
            chunk_size = 100 * 1024 * 1024 // 4  # 100MB of float32
            new_tensor = torch.rand(chunk_size, device=self.device, dtype=torch.float32)

            # Do some computation to prevent optimization
            _ = new_tensor.sum()

            self.tensors.append(new_tensor)
            self.total_allocated += chunk_size * 4  # bytes

            if self.device.type == "cuda":
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
            else:
                allocated = self.total_allocated / 1e9
                reserved = allocated

            metrics = ExtremeMetrics(
                tick=self.tick,
                max_value=allocated,
                min_value=reserved,
            )

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            self.crash_point = {
                'tick': self.tick,
                'allocated_gb': self.total_allocated / 1e9,
                'tensors': len(self.tensors)
            }
            metrics = ExtremeMetrics(
                tick=self.tick,
                crashed=True,
                crash_reason=f"OOM at {self.total_allocated/1e9:.2f}GB"
            )

            # Clean up
            self.tensors = []
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return metrics


# =============================================================================
# TEST 6: QUANTUM CHAOS (Random Operation Order)
# =============================================================================

class QuantumChaosSim:
    """Randomize operation order every tick to maximize FP chaos."""

    def __init__(self, num_stars: int, device: torch.device):
        self.device = device
        self.num_stars = num_stars

        torch.manual_seed(42)
        self.positions = (torch.rand(num_stars, 3, device=device) - 0.5) * 20
        self.velocities = (torch.rand(num_stars, 3, device=device) - 0.5) * 0.1
        self.masses = torch.ones(num_stars, device=device) * 0.001

        # Reference simulation (deterministic)
        self.ref_positions = self.positions.clone()
        self.ref_velocities = self.velocities.clone()

        self.G = 0.001
        self.dt = 0.01
        self.softening_sq = 0.01

        self.tick = 0
        self.divergence_history = []

    def _random_sum(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """Sum along dimension in random order."""
        # Permute along the dimension, then sum
        n = tensor.shape[dim]
        perm = torch.randperm(n, device=self.device)

        if dim == 0:
            tensor = tensor[perm]
        elif dim == 1:
            tensor = tensor[:, perm]

        return tensor.sum(dim=dim)

    def step(self) -> ExtremeMetrics:
        self.tick += 1

        try:
            with torch.no_grad():
                # Reference (deterministic)
                diff_ref = self.ref_positions.unsqueeze(0) - self.ref_positions.unsqueeze(1)
                dist_sq_ref = (diff_ref ** 2).sum(dim=-1) + self.softening_sq
                force_ref = self.G / (dist_sq_ref ** 1.5) * self.masses.unsqueeze(0)
                force_ref = force_ref * (1 - torch.eye(self.num_stars, device=self.device))
                acc_ref = (force_ref.unsqueeze(-1) * diff_ref).sum(dim=1)
                self.ref_velocities = self.ref_velocities + acc_ref * self.dt
                self.ref_positions = self.ref_positions + self.ref_velocities * self.dt

                # Chaotic (random order)
                diff = self.positions.unsqueeze(0) - self.positions.unsqueeze(1)

                # Random order for distance calculation
                dist_components = diff ** 2
                dist_sq = self._random_sum(dist_components, dim=-1) + self.softening_sq

                force = self.G / (dist_sq ** 1.5) * self.masses.unsqueeze(0)
                force = force * (1 - torch.eye(self.num_stars, device=self.device))

                # Random order for force summation
                force_contrib = force.unsqueeze(-1) * diff
                acc = self._random_sum(force_contrib, dim=1)

                self.velocities = self.velocities + acc * self.dt
                self.positions = self.positions + self.velocities * self.dt

                # Measure divergence
                divergence = (self.positions - self.ref_positions).abs().max().item()
                self.divergence_history.append(divergence)

                metrics = ExtremeMetrics(
                    tick=self.tick,
                    max_value=divergence,
                )

        except Exception as e:
            metrics = ExtremeMetrics(tick=self.tick, crashed=True, crash_reason=str(e))

        return metrics


# =============================================================================
# EXTREME MODE RUNNER
# =============================================================================

class ExtremeModeRunner:
    """Run all extreme tests and find crash points."""

    def __init__(self, num_stars: int = 2000, device: torch.device = None):
        self.num_stars = num_stars
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.crash_points = []

    def run_test(self, test_name: str, max_ticks: int = 1000) -> List[ExtremeMetrics]:
        """Run a single extreme test until crash or max ticks."""
        print(f"\n{'='*60}")
        print(f"  EXTREME TEST: {test_name.upper()}")
        print(f"{'='*60}")

        # Create simulation
        if test_name == "subnormal":
            sim = SubnormalHellSim(self.num_stars, self.device)
        elif test_name == "infinity":
            sim = InfinityCascadeSim(self.num_stars, self.device)
        elif test_name == "precision":
            sim = PrecisionMassacreSim(self.num_stars, self.device)
        elif test_name == "singularity":
            sim = SingularityHuntSim(self.num_stars, self.device)
        elif test_name == "memory":
            sim = MemoryArmageddonSim(self.device)
        elif test_name == "quantum":
            sim = QuantumChaosSim(self.num_stars, self.device)
        else:
            print(f"Unknown test: {test_name}")
            return []

        metrics_history = []

        for tick in range(max_ticks):
            metrics = sim.step()
            metrics_history.append(metrics)

            # Progress report
            if tick % 50 == 0 or metrics.crashed:
                status = []
                if metrics.nan_count > 0:
                    status.append(f"NaN={metrics.nan_count}")
                if metrics.inf_count > 0:
                    status.append(f"Inf={metrics.inf_count}")
                if metrics.subnormal_count > 0:
                    status.append(f"Sub={metrics.subnormal_count}")
                if metrics.max_value > 0:
                    status.append(f"Max={metrics.max_value:.2e}")
                if metrics.min_value > 0:
                    status.append(f"Min={metrics.min_value:.2e}")

                status_str = ", ".join(status) if status else "OK"
                print(f"  Tick {tick:4d}: {status_str}")

            if metrics.crashed:
                print(f"\n  !!! CRASH at tick {tick} !!!")
                print(f"  Reason: {metrics.crash_reason}")

                self.crash_points.append(CrashPoint(
                    test_name=test_name,
                    parameter="various",
                    safe_value=tick - 1,
                    crash_value=tick,
                    crash_type="crash",
                    tick=tick,
                    error_message=metrics.crash_reason
                ))
                break

        return metrics_history

    def run_all(self, max_ticks: int = 500):
        """Run all extreme tests."""
        print("\n" + "█" * 60)
        print("█" + " EXTREME MODE - ABSOLUTE LIMITS ".center(58) + "█")
        print("█" + " Finding where reality breaks ".center(58) + "█")
        print("█" * 60)
        print(f"\nDevice: {self.device}")
        print(f"Stars: {self.num_stars}")

        tests = ["subnormal", "infinity", "precision", "singularity", "quantum"]

        all_results = {}
        for test in tests:
            results = self.run_test(test, max_ticks)
            all_results[test] = results

        # Memory test separately (can crash system)
        print("\n" + "=" * 60)
        print("  MEMORY ARMAGEDDON (Ctrl+C to skip)")
        print("=" * 60)
        try:
            mem_results = self.run_test("memory", 100)
            all_results["memory"] = mem_results
        except KeyboardInterrupt:
            print("  Skipped by user")

        # Final report
        self._print_crash_report()

        return all_results

    def _print_crash_report(self):
        """Print summary of all crash points found."""
        print("\n" + "█" * 60)
        print("█" + " CRASH POINT REPORT ".center(58) + "█")
        print("█" * 60)

        if not self.crash_points:
            print("\n  No crashes detected! Reality is surprisingly stable.")
            print("  Try increasing stars or decreasing max_ticks.")
        else:
            print(f"\n  Found {len(self.crash_points)} crash points:\n")
            for i, cp in enumerate(self.crash_points):
                print(f"  {i+1}. {cp.test_name.upper()}")
                print(f"     Crashed at tick: {cp.tick}")
                print(f"     Reason: {cp.error_message}")
                print()

        print("█" * 60)


def find_exact_crash_point(test_name: str, param_name: str,
                           param_range: Tuple[float, float],
                           device: torch.device) -> CrashPoint:
    """Binary search to find exact crash point."""
    print(f"\n  Binary search for {test_name} crash point...")

    low, high = param_range
    crash_value = high
    safe_value = low

    for iteration in range(20):  # 20 iterations = ~1e-6 precision
        mid = (low + high) / 2

        # Test at mid value
        # ... implement test ...

        # For now, placeholder
        crashed = False

        if crashed:
            high = mid
            crash_value = mid
        else:
            low = mid
            safe_value = mid

        print(f"    Iter {iteration}: safe={safe_value:.2e}, crash={crash_value:.2e}")

    return CrashPoint(
        test_name=test_name,
        parameter=param_name,
        safe_value=safe_value,
        crash_value=crash_value,
        crash_type="binary_search",
        tick=0
    )


def main():
    parser = argparse.ArgumentParser(description="EXTREME MODE - Absolute Limits")
    parser.add_argument("--test", type=str, default="all",
                        choices=["subnormal", "infinity", "precision",
                                "singularity", "memory", "quantum", "all"])
    parser.add_argument("--stars", type=int, default=2000)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--find-crash-point", action="store_true",
                        help="Binary search for exact crash points")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    runner = ExtremeModeRunner(num_stars=args.stars, device=device)

    if args.test == "all":
        runner.run_all(max_ticks=args.ticks)
    else:
        runner.run_test(args.test, max_ticks=args.ticks)
        runner._print_crash_report()


if __name__ == "__main__":
    main()
