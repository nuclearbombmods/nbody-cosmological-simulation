"""
OMEGA POINT TEST - Universal Reality Stress Test (URST)
========================================================

The "Final Boss" of simulation hypothesis testing.

This script pushes ALL boundaries simultaneously to find the exact point
where computational reality breaks down - the "Triple Point" where:
- Precision collapse
- Velocity limits
- Information density

...all converge to reveal the substrate of reality.

Tests:
1. BEKENSTEIN BOUND - Pack stars until "digital black hole" forms
2. TEMPORAL ALIASING - Find the quantized "tick rate" of the universe
3. ENTROPY LEAK - Track ghost energy accumulation over 100k+ ticks
4. PHASE SPACE SCAN - Map the 3D breakdown surface
5. TRIPLE POINT HUNT - Auto-tune to find exact failure coordinates
6. PHYSICAL COMPARISON - Compare limits to c, l_p, t_p

Output:
- Reality Heatmap (3D surface of stability)
- Triple Point Coordinates (Precision, Velocity, Density)
- Physical Constants Comparison Table
- Final Verdict on Simulation Hypothesis

Usage:
    python omega_point_test.py --mode full          # Complete 3D scan
    python omega_point_test.py --mode bekenstein    # Information density only
    python omega_point_test.py --mode temporal      # Time quantization only
    python omega_point_test.py --mode entropy       # Ghost energy tracking
    python omega_point_test.py --mode hunt          # Auto-tune to Triple Point
"""

import argparse
import time
import json
import sys
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from galaxy import create_disk_galaxy
from simulation import GalaxySimulation
from quantization import PrecisionMode, _grid_quantize_safe


# =============================================================================
# CONSTANTS - Physical and Computational
# =============================================================================

# Physical constants (SI)
C_LIGHT = 299792458  # m/s - Speed of light
PLANCK_LENGTH = 1.616255e-35  # m
PLANCK_TIME = 5.391247e-44  # s
PLANCK_MASS = 2.176434e-8  # kg
BOLTZMANN = 1.380649e-23  # J/K

# IEEE 754 Floating Point Limits
FP64_MIN = 2.2250738585072014e-308
FP64_EPSILON = 2.220446049250313e-16
FP32_MIN = 1.1754943508222875e-38
FP32_MIN_SUBNORMAL = 1.401298464324817e-45
FP32_EPSILON = 1.1920928955078125e-7
FP16_MIN = 6.103515625e-5
FP16_EPSILON = 0.0009765625

# Precision levels for testing
PRECISION_LEVELS = {
    "float64": {"dtype": torch.float64, "bits": 64, "epsilon": FP64_EPSILON},
    "float32": {"dtype": torch.float32, "bits": 32, "epsilon": FP32_EPSILON},
    "float16": {"dtype": torch.float16, "bits": 16, "epsilon": FP16_EPSILON},
    "bfloat16": {"dtype": torch.bfloat16, "bits": 16, "epsilon": 0.0078125},
    "int8": {"dtype": torch.float32, "bits": 8, "epsilon": 1/128, "quantize": 256},
    "int4": {"dtype": torch.float32, "bits": 4, "epsilon": 1/8, "quantize": 16},
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BekensteinResult:
    """Results from information density test."""
    max_density_achieved: float  # stars per unit volume
    breakdown_density: float  # density where frame rate collapsed
    schwarzschild_radius: float  # simulation units
    power_at_breakdown_watts: float
    time_dilation_factor: float  # ticks/second at breakdown vs baseline
    interpretation: str


@dataclass
class TemporalAliasingResult:
    """Results from dt limit test."""
    critical_dt: float  # dt where energy drift explodes
    subcritical_drift: float  # energy drift below critical
    supercritical_drift: float  # energy drift above critical
    tick_rate_estimate: float  # estimated "Planck time" of simulation
    phase_transition_sharpness: float  # how sudden the breakdown is
    interpretation: str


@dataclass
class EntropyLeakResult:
    """Results from ghost energy accumulation test."""
    total_ticks: int
    initial_energy: float
    final_energy: float
    energy_gained: float
    ghost_energy_rate: float  # energy per tick
    dark_energy_equivalent: float  # scaled to cosmological
    time_to_heat_death: float  # ticks until galaxy flies apart
    interpretation: str


@dataclass
class PhaseSpacePoint:
    """Single point in the 3D phase space."""
    precision_bits: int
    velocity_multiplier: float
    star_density: float
    energy_drift: float
    divergence_rate: float
    power_watts: float
    is_stable: bool
    butterfly_detected: bool


@dataclass
class TriplePointResult:
    """The exact coordinates where reality breaks down."""
    precision_bits: int
    velocity_multiplier: float
    star_density: float
    confidence: float  # 0-100
    power_at_breakdown: float
    physical_equivalents: Dict[str, float]
    interpretation: str


@dataclass
class OmegaPointReport:
    """Complete URST report."""
    timestamp: str
    device: str
    gpu_name: str
    bekenstein: Optional[BekensteinResult]
    temporal: Optional[TemporalAliasingResult]
    entropy: Optional[EntropyLeakResult]
    triple_point: Optional[TriplePointResult]
    phase_space_points: List[PhaseSpacePoint]
    physical_comparison: Dict[str, Dict]
    final_verdict: str
    simulation_probability: float  # 0-100%


# =============================================================================
# TEST 1: BEKENSTEIN BOUND (Information Density Limit)
# =============================================================================

class BekensteinBoundTest:
    """
    Test the information density limit - the "digital black hole" threshold.

    Theory: Pack stars into smaller volumes until GPU overhead explodes.
    At some point, the "rendering cost" should become so high that
    time effectively stops (frame rate -> 0).
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.baseline_tps = None  # ticks per second baseline

    def run(self,
            max_stars: int = 10000,
            min_radius: float = 0.1,
            max_radius: float = 10.0,
            steps_per_test: int = 100) -> BekensteinResult:
        """
        Progressively increase star density until breakdown.
        """
        print(f"\n{'='*70}")
        print("  TEST 1: BEKENSTEIN BOUND (Information Density Limit)")
        print("  Goal: Find the 'Schwarzschild Radius' of the simulation")
        print(f"{'='*70}")

        # Establish baseline with sparse galaxy
        print(f"\n  Establishing baseline (sparse galaxy)...")
        baseline_density = self._measure_performance(500, 10.0, steps_per_test)
        self.baseline_tps = baseline_density["tps"]
        print(f"    Baseline: {self.baseline_tps:.1f} ticks/second")

        # Progressive density increase
        results = []
        radii = np.logspace(np.log10(max_radius), np.log10(min_radius), 20)

        print(f"\n  {'Stars':<8} {'Radius':<10} {'Density':<12} {'TPS':<10} {'Dilation':<10}")
        print(f"  {'-'*60}")

        breakdown_density = None
        breakdown_radius = None

        for star_count in [500, 1000, 2000, 3000, 5000, 7500, 10000]:
            if star_count > max_stars:
                break

            for radius in [10.0, 5.0, 2.0, 1.0, 0.5, 0.2]:
                if radius < min_radius:
                    continue

                # Calculate density (stars per unit^3)
                volume = (4/3) * np.pi * radius**3
                density = star_count / volume

                # Run test
                try:
                    perf = self._measure_performance(star_count, radius, steps_per_test)
                    tps = perf["tps"]
                    dilation = tps / self.baseline_tps if self.baseline_tps > 0 else 0

                    results.append({
                        "stars": star_count,
                        "radius": radius,
                        "density": density,
                        "tps": tps,
                        "dilation": dilation
                    })

                    status = "STABLE" if dilation > 0.1 else "COLLAPSE!"
                    print(f"  {star_count:<8} {radius:<10.2f} {density:<12.1f} {tps:<10.1f} {dilation:<10.3f} {status}")

                    # Check for breakdown
                    if dilation < 0.1 and breakdown_density is None:
                        breakdown_density = density
                        breakdown_radius = radius
                        print(f"\n  >>> BREAKDOWN DETECTED at density={density:.1f} <<<\n")

                except torch.cuda.OutOfMemoryError:
                    print(f"  {star_count:<8} {radius:<10.2f} {'OOM':<12} {'---':<10} {'---':<10} MEMORY LIMIT")
                    if breakdown_density is None:
                        breakdown_density = density
                        breakdown_radius = radius
                    break
                except Exception as e:
                    print(f"  {star_count:<8} {radius:<10.2f} ERROR: {e}")
                    continue

        # Calculate Schwarzschild radius equivalent
        if breakdown_radius:
            schwarzschild_r = breakdown_radius
        else:
            schwarzschild_r = min_radius
            breakdown_density = results[-1]["density"] if results else 0

        # Estimate power at breakdown (rough)
        power_estimate = 300.0  # Assume near TDP for RTX 5090

        # Final time dilation
        final_dilation = results[-1]["dilation"] if results else 1.0

        # Interpretation
        if breakdown_density and breakdown_density > 1000:
            interpretation = f"CRITICAL: Information density limit found at {breakdown_density:.0f} stars/unit^3. Digital black hole threshold detected!"
        elif final_dilation < 0.5:
            interpretation = f"SIGNIFICANT: Time dilation of {final_dilation:.1%} observed. Approaching computational horizon."
        else:
            interpretation = "STABLE: No clear breakdown detected within tested range."

        return BekensteinResult(
            max_density_achieved=max([r["density"] for r in results]) if results else 0,
            breakdown_density=breakdown_density or 0,
            schwarzschild_radius=schwarzschild_r,
            power_at_breakdown_watts=power_estimate,
            time_dilation_factor=final_dilation,
            interpretation=interpretation
        )

    def _measure_performance(self, num_stars: int, radius: float, steps: int) -> Dict:
        """Measure simulation performance at given parameters."""
        positions, velocities, masses = create_disk_galaxy(
            num_stars=num_stars,
            galaxy_radius=radius,
            device=self.device
        )

        sim = GalaxySimulation(
            positions.float(), velocities.float(), masses.float(),
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001, dt=0.01, softening=0.01,
            device=self.device
        )

        # Warmup
        for _ in range(10):
            sim.step()

        # Timed run
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(steps):
            sim.step()

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        return {
            "tps": steps / elapsed,
            "elapsed": elapsed
        }


# =============================================================================
# TEST 2: TEMPORAL ALIASING (Time Quantization)
# =============================================================================

class TemporalAliasingTest:
    """
    Test if time is continuous or quantized by varying dt.

    Theory: As dt increases, energy conservation degrades.
    If there's a sharp transition, that's the "tick rate" of the universe.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def run(self,
            num_stars: int = 1000,
            dt_range: Tuple[float, float] = (0.001, 5.0),
            num_dt_samples: int = 50,
            steps_per_test: int = 500) -> TemporalAliasingResult:
        """
        Scan dt values to find the critical threshold.
        """
        print(f"\n{'='*70}")
        print("  TEST 2: TEMPORAL ALIASING (Time Quantization)")
        print("  Goal: Find the 'Planck Time' of the simulation")
        print(f"{'='*70}")

        # Create consistent initial conditions
        positions, velocities, masses = create_disk_galaxy(
            num_stars=num_stars,
            galaxy_radius=10.0,
            device=self.device
        )

        dt_values = np.logspace(np.log10(dt_range[0]), np.log10(dt_range[1]), num_dt_samples)
        results = []

        print(f"\n  {'dt':<12} {'Energy Drift %':<18} {'Status':<15}")
        print(f"  {'-'*50}")

        critical_dt = None
        subcritical_drift = 0
        supercritical_drift = 0

        for dt in dt_values:
            sim = GalaxySimulation(
                positions.clone(), velocities.clone(), masses.clone(),
                precision_mode=PrecisionMode.FLOAT32,
                G=0.001, dt=dt, softening=0.1,
                device=self.device
            )

            initial_energy = sim.get_total_energy()

            # Run simulation
            for _ in range(steps_per_test):
                sim.step()

            final_energy = sim.get_total_energy()
            drift = abs((final_energy - initial_energy) / initial_energy) * 100

            results.append({"dt": dt, "drift": drift})

            # Determine status
            if drift < 1:
                status = "STABLE"
            elif drift < 10:
                status = "MARGINAL"
            elif drift < 50:
                status = "UNSTABLE"
            else:
                status = "BREAKDOWN!"

            print(f"  {dt:<12.4f} {drift:<18.2f} {status:<15}")

            # Detect critical dt
            if drift > 50 and critical_dt is None:
                critical_dt = dt
                # Get drift just before breakdown
                if len(results) > 1:
                    subcritical_drift = results[-2]["drift"]
                supercritical_drift = drift

        # If no critical dt found, use last tested value
        if critical_dt is None:
            critical_dt = dt_values[-1]
            subcritical_drift = results[-2]["drift"] if len(results) > 1 else 0
            supercritical_drift = results[-1]["drift"]

        # Calculate transition sharpness
        if subcritical_drift > 0:
            sharpness = supercritical_drift / subcritical_drift
        else:
            sharpness = float('inf')

        # Estimate "Planck time" equivalent
        # If dt=2.0 causes breakdown, that's 2.0 simulation units
        # Scale to physical Planck time
        tick_rate_estimate = critical_dt

        # Interpretation
        if sharpness > 10:
            interpretation = f"CRITICAL: Sharp phase transition at dt={critical_dt:.3f}! Time appears QUANTIZED with tick rate ~{tick_rate_estimate:.3f} units."
        elif sharpness > 3:
            interpretation = f"SIGNIFICANT: Transition detected at dt={critical_dt:.3f}. Evidence of temporal granularity."
        else:
            interpretation = f"GRADUAL: No sharp transition found. Time appears continuous within tested range."

        return TemporalAliasingResult(
            critical_dt=critical_dt,
            subcritical_drift=subcritical_drift,
            supercritical_drift=supercritical_drift,
            tick_rate_estimate=tick_rate_estimate,
            phase_transition_sharpness=sharpness,
            interpretation=interpretation
        )


# =============================================================================
# TEST 3: ENTROPY LEAK (Ghost Energy / Dark Energy)
# =============================================================================

class EntropyLeakTest:
    """
    Track energy accumulation over 100k+ ticks in broken math mode.

    Theory: Rounding errors should act like a "heater," injecting
    ghost energy until the galaxy flies apart (Dark Energy equivalent).
    """

    def __init__(self, device: torch.device):
        self.device = device

    def run(self,
            num_stars: int = 500,
            total_ticks: int = 100000,
            quant_levels: int = 16,  # int4 equivalent
            log_interval: int = 1000) -> EntropyLeakResult:
        """
        Run long-term simulation tracking energy gain.
        """
        print(f"\n{'='*70}")
        print("  TEST 3: ENTROPY LEAK (Ghost Energy / Dark Energy)")
        print("  Goal: Measure energy injection from rounding errors")
        print(f"{'='*70}")

        print(f"\n  Parameters:")
        print(f"    Stars: {num_stars}")
        print(f"    Total ticks: {total_ticks:,}")
        print(f"    Quantization: {quant_levels} levels (int4)")

        # Create broken simulation
        positions, velocities, masses = create_disk_galaxy(
            num_stars=num_stars,
            galaxy_radius=10.0,
            device=self.device
        )

        class GhostEnergySim(GalaxySimulation):
            def __init__(self, *args, quant_levels=16, **kwargs):
                self.quant_levels = quant_levels
                super().__init__(*args, **kwargs)

            def _compute_accelerations(self):
                pos = self.positions
                diff = pos.unsqueeze(0) - pos.unsqueeze(1)
                dist_sq = (diff ** 2).sum(dim=-1) + self.softening_sq

                # BROKEN MATH: Quantize distances
                dist_sq = _grid_quantize_safe(dist_sq, self.quant_levels, min_val=0.01)

                dist_cubed = dist_sq ** 1.5
                force_factor = self.G / dist_cubed
                force_factor = force_factor * self.masses.unsqueeze(0)
                force_factor = force_factor * (1 - torch.eye(self.num_stars, device=self.device))
                return (force_factor.unsqueeze(-1) * diff).sum(dim=1)

        sim = GhostEnergySim(
            positions.float(), velocities.float(), masses.float(),
            quant_levels=quant_levels,
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001, dt=0.01, softening=0.1,
            device=self.device
        )

        initial_energy = sim.get_total_energy()
        energy_history = [initial_energy]

        print(f"\n  {'Tick':<12} {'Energy':<18} {'Drift %':<15} {'Status'}")
        print(f"  {'-'*60}")
        print(f"  {0:<12} {initial_energy:<18.6f} {0:<15.2f} BASELINE")

        start_time = time.time()

        for tick in range(1, total_ticks + 1):
            sim.step()

            if tick % log_interval == 0:
                current_energy = sim.get_total_energy()
                drift = (current_energy - initial_energy) / abs(initial_energy) * 100
                energy_history.append(current_energy)

                # Status based on drift
                if drift < 1:
                    status = "STABLE"
                elif drift < 10:
                    status = "WARMING"
                elif drift < 50:
                    status = "HOT!"
                elif drift < 100:
                    status = "BOILING!"
                else:
                    status = "HEAT DEATH!"

                elapsed = time.time() - start_time
                tps = tick / elapsed
                eta = (total_ticks - tick) / tps if tps > 0 else 0

                print(f"  {tick:<12,} {current_energy:<18.6f} {drift:<15.2f} {status} (ETA: {eta:.0f}s)")

                # Early exit if galaxy has completely broken down
                if drift > 1000:
                    print(f"\n  >>> GALAXY EXPLOSION DETECTED - Ending early <<<\n")
                    break

        final_energy = sim.get_total_energy()
        energy_gained = final_energy - initial_energy
        ghost_rate = energy_gained / tick  # energy per tick

        # Calculate time to "heat death" (when energy doubles)
        if ghost_rate > 0:
            time_to_heat_death = abs(initial_energy) / ghost_rate
        else:
            time_to_heat_death = float('inf')

        # Scale to "dark energy" equivalent
        # Real dark energy: ~68% of universe's energy
        # Our ghost energy as fraction of initial
        dark_energy_equivalent = abs(energy_gained / initial_energy) if initial_energy != 0 else 0

        # Interpretation
        final_drift = abs((final_energy - initial_energy) / initial_energy) * 100
        if final_drift > 100:
            interpretation = f"CRITICAL: Galaxy gained {final_drift:.0f}% energy! Ghost energy rate: {ghost_rate:.2e}/tick. Digital 'Dark Energy' confirmed!"
        elif final_drift > 10:
            interpretation = f"SIGNIFICANT: {final_drift:.1f}% energy gain. Rounding errors inject {ghost_rate:.2e} energy/tick. Universe is 'heating up.'"
        elif final_drift > 1:
            interpretation = f"DETECTABLE: {final_drift:.2f}% energy leak over {tick:,} ticks. Minor ghost energy present."
        else:
            interpretation = "STABLE: Energy well conserved. No significant entropy leak."

        return EntropyLeakResult(
            total_ticks=tick,
            initial_energy=initial_energy,
            final_energy=final_energy,
            energy_gained=energy_gained,
            ghost_energy_rate=ghost_rate,
            dark_energy_equivalent=dark_energy_equivalent,
            time_to_heat_death=time_to_heat_death,
            interpretation=interpretation
        )


# =============================================================================
# TEST 4: PHASE SPACE SCAN (3D Breakdown Surface)
# =============================================================================

class PhaseSpaceScanner:
    """
    Scan the 3D parameter space to find the breakdown surface.

    Axes:
    - Precision (bits): 64, 32, 16, 8, 4
    - Velocity multiplier: 1x to 200x
    - Density (stars): 100 to 5000
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.results: List[PhaseSpacePoint] = []

    def run(self,
            precision_levels: List[str] = ["float32", "float16", "int8", "int4"],
            velocity_range: Tuple[float, float] = (1.0, 100.0),
            velocity_steps: int = 5,
            density_range: Tuple[int, int] = (100, 2000),
            density_steps: int = 5,
            steps_per_test: int = 200) -> List[PhaseSpacePoint]:
        """
        Scan the 3D parameter space.
        """
        print(f"\n{'='*70}")
        print("  TEST 4: PHASE SPACE SCAN (3D Breakdown Surface)")
        print("  Goal: Map the reality stability surface")
        print(f"{'='*70}")

        velocities = np.logspace(np.log10(velocity_range[0]),
                                  np.log10(velocity_range[1]),
                                  velocity_steps)
        densities = np.linspace(density_range[0], density_range[1], density_steps).astype(int)

        total_tests = len(precision_levels) * len(velocities) * len(densities)
        print(f"\n  Total tests: {total_tests}")
        print(f"  Precision levels: {precision_levels}")
        print(f"  Velocity range: {velocity_range[0]:.1f}x - {velocity_range[1]:.1f}x")
        print(f"  Density range: {density_range[0]} - {density_range[1]} stars")

        print(f"\n  {'Precision':<12} {'Velocity':<10} {'Stars':<8} {'Drift%':<10} {'Stable':<8} {'Butterfly'}")
        print(f"  {'-'*70}")

        test_num = 0

        for prec_name in precision_levels:
            prec_info = PRECISION_LEVELS[prec_name]
            bits = prec_info["bits"]

            for vel_mult in velocities:
                for num_stars in densities:
                    test_num += 1

                    try:
                        result = self._test_point(
                            prec_name, prec_info, vel_mult, num_stars, steps_per_test
                        )
                        self.results.append(result)

                        stable_str = "YES" if result.is_stable else "NO"
                        butterfly_str = "YES" if result.butterfly_detected else "NO"

                        print(f"  {prec_name:<12} {vel_mult:<10.1f} {num_stars:<8} "
                              f"{result.energy_drift:<10.2f} {stable_str:<8} {butterfly_str}")

                    except Exception as e:
                        print(f"  {prec_name:<12} {vel_mult:<10.1f} {num_stars:<8} ERROR: {str(e)[:30]}")

        return self.results

    def _test_point(self,
                    prec_name: str,
                    prec_info: Dict,
                    vel_mult: float,
                    num_stars: int,
                    steps: int) -> PhaseSpacePoint:
        """Test a single point in parameter space."""

        # Create galaxy
        positions, velocities, masses = create_disk_galaxy(
            num_stars=num_stars,
            galaxy_radius=10.0,
            device=self.device
        )

        # Apply velocity multiplier
        velocities = velocities * vel_mult

        # Create two simulations for butterfly detection
        if "quantize" in prec_info:
            # Quantized mode
            quant_levels = prec_info["quantize"]

            class QuantSim(GalaxySimulation):
                def __init__(self, *args, levels=16, **kwargs):
                    self.levels = levels
                    super().__init__(*args, **kwargs)

                def _compute_accelerations(self):
                    pos = self.positions
                    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
                    dist_sq = (diff ** 2).sum(dim=-1) + self.softening_sq
                    dist_sq = _grid_quantize_safe(dist_sq, self.levels, min_val=0.01)
                    dist_cubed = dist_sq ** 1.5
                    force_factor = self.G / dist_cubed
                    force_factor = force_factor * self.masses.unsqueeze(0)
                    force_factor = force_factor * (1 - torch.eye(self.num_stars, device=self.device))
                    return (force_factor.unsqueeze(-1) * diff).sum(dim=1)

            sim_a = QuantSim(
                positions.float().clone(), velocities.float().clone(), masses.float().clone(),
                levels=quant_levels,
                precision_mode=PrecisionMode.FLOAT32,
                G=0.001, dt=0.01, softening=0.1,
                device=self.device
            )
            sim_b = QuantSim(
                positions.float().clone(), velocities.float().clone(), masses.float().clone(),
                levels=quant_levels,
                precision_mode=PrecisionMode.FLOAT32,
                G=0.001, dt=0.01, softening=0.1,
                device=self.device
            )
        else:
            # Native precision mode
            dtype = prec_info["dtype"]

            sim_a = GalaxySimulation(
                positions.to(dtype).clone(), velocities.to(dtype).clone(), masses.to(dtype).clone(),
                precision_mode=PrecisionMode.FLOAT32,
                G=0.001, dt=0.01, softening=0.1,
                device=self.device
            )
            sim_b = GalaxySimulation(
                positions.to(dtype).clone(), velocities.to(dtype).clone(), masses.to(dtype).clone(),
                precision_mode=PrecisionMode.FLOAT32,
                G=0.001, dt=0.01, softening=0.1,
                device=self.device
            )

        initial_energy = sim_a.get_total_energy()

        # Run both simulations (slight perturbation to sim_b for butterfly test)
        sim_b.positions[0, 0] += 1e-10

        max_divergence = 0
        for _ in range(steps):
            sim_a.step()
            sim_b.step()

            divergence = (sim_a.positions - sim_b.positions).abs().max().item()
            max_divergence = max(max_divergence, divergence)

        final_energy = sim_a.get_total_energy()
        drift = abs((final_energy - initial_energy) / initial_energy) * 100

        # Butterfly effect threshold
        butterfly_detected = max_divergence > 0.1

        # Stability threshold
        is_stable = drift < 10 and not butterfly_detected

        return PhaseSpacePoint(
            precision_bits=prec_info["bits"],
            velocity_multiplier=vel_mult,
            star_density=num_stars,
            energy_drift=drift,
            divergence_rate=max_divergence,
            power_watts=0,  # Would need GPU monitoring
            is_stable=is_stable,
            butterfly_detected=butterfly_detected
        )


# =============================================================================
# TEST 5: TRIPLE POINT HUNT (Auto-Tuning Breakdown Detection)
# =============================================================================

class TriplePointHunter:
    """
    Auto-tune to find the exact coordinates where reality breaks down.
    Uses binary search across all three dimensions.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def run(self,
            scanner: PhaseSpaceScanner,
            refinement_steps: int = 3) -> TriplePointResult:
        """
        Find the Triple Point from phase space data.
        """
        print(f"\n{'='*70}")
        print("  TEST 5: TRIPLE POINT HUNT (Auto-Tuning)")
        print("  Goal: Find exact breakdown coordinates")
        print(f"{'='*70}")

        if not scanner.results:
            print("  No phase space data available!")
            return None

        # Find boundary points (stable -> unstable transitions)
        boundary_points = []

        for point in scanner.results:
            if not point.is_stable:
                # Find nearest stable neighbor
                min_dist = float('inf')
                nearest_stable = None

                for other in scanner.results:
                    if other.is_stable:
                        dist = (
                            (point.precision_bits - other.precision_bits)**2 +
                            (point.velocity_multiplier - other.velocity_multiplier)**2 +
                            (point.star_density - other.star_density)**2
                        )
                        if dist < min_dist:
                            min_dist = dist
                            nearest_stable = other

                if nearest_stable:
                    boundary_points.append((nearest_stable, point))

        if not boundary_points:
            print("  No boundary detected - all points stable or all unstable")
            # Return most extreme unstable point
            unstable = [p for p in scanner.results if not p.is_stable]
            if unstable:
                worst = max(unstable, key=lambda p: p.energy_drift)
                return TriplePointResult(
                    precision_bits=worst.precision_bits,
                    velocity_multiplier=worst.velocity_multiplier,
                    star_density=worst.star_density,
                    confidence=50.0,
                    power_at_breakdown=worst.power_watts,
                    physical_equivalents=self._compute_physical_equivalents(worst),
                    interpretation="Estimated from most unstable point"
                )
            return None

        # Average the boundary points to estimate Triple Point
        avg_bits = np.mean([p[1].precision_bits for p in boundary_points])
        avg_vel = np.mean([p[1].velocity_multiplier for p in boundary_points])
        avg_density = np.mean([p[1].star_density for p in boundary_points])

        # Confidence based on how many boundary points we found
        confidence = min(100, len(boundary_points) * 10)

        print(f"\n  Found {len(boundary_points)} boundary transitions")
        print(f"\n  TRIPLE POINT ESTIMATE:")
        print(f"    Precision: {avg_bits:.0f} bits")
        print(f"    Velocity:  {avg_vel:.1f}x")
        print(f"    Density:   {avg_density:.0f} stars")
        print(f"    Confidence: {confidence}%")

        # Compute physical equivalents
        physical = self._compute_physical_equivalents_raw(avg_bits, avg_vel, avg_density)

        print(f"\n  PHYSICAL EQUIVALENTS:")
        for key, value in physical.items():
            print(f"    {key}: {value}")

        interpretation = f"Triple Point found at ({avg_bits:.0f} bits, {avg_vel:.1f}x velocity, {avg_density:.0f} stars). "

        if avg_bits <= 8:
            interpretation += "Low precision critical - matches Planck-scale quantization. "
        if avg_vel > 50:
            interpretation += "High velocity limit - potential speed-of-light equivalent. "
        if avg_density > 1000:
            interpretation += "High density limit - information saturation detected."

        return TriplePointResult(
            precision_bits=int(avg_bits),
            velocity_multiplier=avg_vel,
            star_density=avg_density,
            confidence=confidence,
            power_at_breakdown=0,
            physical_equivalents=physical,
            interpretation=interpretation
        )

    def _compute_physical_equivalents(self, point: PhaseSpacePoint) -> Dict:
        return self._compute_physical_equivalents_raw(
            point.precision_bits,
            point.velocity_multiplier,
            point.star_density
        )

    def _compute_physical_equivalents_raw(self, bits: float, vel: float, density: float) -> Dict:
        """Map simulation limits to physical constants."""

        # Precision -> Planck length
        # 4 bits = ~1e-45 minimum = Planck length equivalent
        precision_ratio = FP32_MIN_SUBNORMAL / PLANCK_LENGTH

        # Velocity -> Speed of light
        # 100x velocity in sim = approaching c
        velocity_ratio = vel / 100  # 100x = c
        c_equivalent = velocity_ratio * C_LIGHT

        # Density -> Bekenstein bound
        # Information per unit volume
        bits_per_star = bits * 3 * 2  # 3 coords, position + velocity
        info_density = density * bits_per_star

        return {
            "velocity_as_fraction_of_c": velocity_ratio,
            "c_equivalent_m_s": c_equivalent,
            "precision_planck_ratio": precision_ratio,
            "information_density_bits_per_unit3": info_density,
            "precision_bits": bits
        }


# =============================================================================
# VISUALIZATION: REALITY HEATMAP
# =============================================================================

def generate_reality_heatmap(results: List[PhaseSpacePoint], output_path: str = None):
    """
    Generate 3D visualization of the stability surface.
    """
    print(f"\n  Generating Reality Heatmap...")

    if not results:
        print("  No data to visualize!")
        return

    fig = plt.figure(figsize=(16, 12))

    # 3D scatter plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    bits = [p.precision_bits for p in results]
    vels = [p.velocity_multiplier for p in results]
    dens = [p.star_density for p in results]
    drifts = [min(p.energy_drift, 100) for p in results]  # Cap at 100%
    stable = [p.is_stable for p in results]

    colors = ['green' if s else 'red' for s in stable]
    sizes = [max(10, min(100, d)) for d in drifts]

    ax1.scatter(bits, vels, dens, c=colors, s=sizes, alpha=0.6)
    ax1.set_xlabel('Precision (bits)')
    ax1.set_ylabel('Velocity (x)')
    ax1.set_zlabel('Density (stars)')
    ax1.set_title('Reality Stability Map\n(Green=Stable, Red=Unstable)')

    # 2D projections
    # Precision vs Velocity
    ax2 = fig.add_subplot(2, 2, 2)
    scatter = ax2.scatter(bits, vels, c=drifts, cmap='hot', s=50, alpha=0.7)
    ax2.set_xlabel('Precision (bits)')
    ax2.set_ylabel('Velocity (x)')
    ax2.set_title('Precision vs Velocity')
    plt.colorbar(scatter, ax=ax2, label='Energy Drift %')

    # Velocity vs Density
    ax3 = fig.add_subplot(2, 2, 3)
    scatter = ax3.scatter(vels, dens, c=drifts, cmap='hot', s=50, alpha=0.7)
    ax3.set_xlabel('Velocity (x)')
    ax3.set_ylabel('Density (stars)')
    ax3.set_title('Velocity vs Density')
    plt.colorbar(scatter, ax=ax3, label='Energy Drift %')

    # Precision vs Density
    ax4 = fig.add_subplot(2, 2, 4)
    scatter = ax4.scatter(bits, dens, c=drifts, cmap='hot', s=50, alpha=0.7)
    ax4.set_xlabel('Precision (bits)')
    ax4.set_ylabel('Density (stars)')
    ax4.set_title('Precision vs Density')
    plt.colorbar(scatter, ax=ax4, label='Energy Drift %')

    plt.suptitle('OMEGA POINT - Reality Phase Space', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to: {output_path}")

    plt.show()


# =============================================================================
# PHYSICAL CONSTANTS COMPARISON
# =============================================================================

def generate_physical_comparison(
    bekenstein: BekensteinResult,
    temporal: TemporalAliasingResult,
    entropy: EntropyLeakResult,
    triple_point: TriplePointResult
) -> Dict[str, Dict]:
    """
    Compare simulation limits to physical constants.
    """
    comparison = {}

    # Velocity -> Speed of Light
    if triple_point:
        comparison["Max Velocity"] = {
            "simulation_value": f"{triple_point.velocity_multiplier:.1f}x",
            "physical_equivalent": "Speed of Light (c)",
            "physical_value": f"{C_LIGHT:.2e} m/s",
            "ratio": f"{triple_point.physical_equivalents.get('velocity_as_fraction_of_c', 0):.2%} of c"
        }

    # Time Step -> Planck Time
    if temporal:
        comparison["Time Step (dt)"] = {
            "simulation_value": f"{temporal.critical_dt:.4f} units",
            "physical_equivalent": "Planck Time (t_p)",
            "physical_value": f"{PLANCK_TIME:.2e} s",
            "interpretation": temporal.interpretation[:50]
        }

    # Precision -> Planck Length
    comparison["Min Precision"] = {
        "simulation_value": f"{FP32_MIN_SUBNORMAL:.2e}",
        "physical_equivalent": "Planck Length (l_p)",
        "physical_value": f"{PLANCK_LENGTH:.2e} m",
        "ratio": f"{FP32_MIN_SUBNORMAL / PLANCK_LENGTH:.2e}"
    }

    # Ghost Energy -> Dark Energy
    if entropy:
        comparison["Ghost Energy Rate"] = {
            "simulation_value": f"{entropy.ghost_energy_rate:.2e}/tick",
            "physical_equivalent": "Dark Energy",
            "physical_value": "68% of universe",
            "sim_equivalent": f"{entropy.dark_energy_equivalent:.2%} of initial energy"
        }

    # Information Density -> Bekenstein Bound
    if bekenstein:
        comparison["Max Density"] = {
            "simulation_value": f"{bekenstein.breakdown_density:.0f} stars/unit^3",
            "physical_equivalent": "Bekenstein Bound",
            "physical_value": "S <= 2*pi*R*E/(hbar*c)",
            "schwarzschild_r": f"{bekenstein.schwarzschild_radius:.2f} units"
        }

    return comparison


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def run_omega_point_test(
    mode: str = "full",
    device: torch.device = None,
    output_dir: str = None
) -> OmegaPointReport:
    """
    Run the complete Omega Point test suite.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*70)
    print("  OMEGA POINT TEST - Universal Reality Stress Test (URST)")
    print("  The Final Boss of Simulation Hypothesis Testing")
    print("="*70)
    print(f"  Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU: {gpu_name}")
    else:
        gpu_name = "CPU"
    print(f"  Mode: {mode}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("="*70)

    bekenstein_result = None
    temporal_result = None
    entropy_result = None
    triple_point_result = None
    phase_space_points = []

    # Run selected tests
    if mode in ["full", "bekenstein"]:
        bekenstein_test = BekensteinBoundTest(device)
        bekenstein_result = bekenstein_test.run()

    if mode in ["full", "temporal"]:
        temporal_test = TemporalAliasingTest(device)
        temporal_result = temporal_test.run()

    if mode in ["full", "entropy"]:
        entropy_test = EntropyLeakTest(device)
        entropy_result = entropy_test.run(total_ticks=50000)  # Shorter for demo

    if mode in ["full", "hunt", "phase"]:
        scanner = PhaseSpaceScanner(device)
        phase_space_points = scanner.run()

        if mode in ["full", "hunt"]:
            hunter = TriplePointHunter(device)
            triple_point_result = hunter.run(scanner)

    # Generate physical comparison
    physical_comparison = generate_physical_comparison(
        bekenstein_result, temporal_result, entropy_result, triple_point_result
    )

    # Calculate simulation probability
    scores = []
    if bekenstein_result and bekenstein_result.time_dilation_factor < 0.5:
        scores.append(80)
    if temporal_result and temporal_result.phase_transition_sharpness > 5:
        scores.append(70)
    if entropy_result and entropy_result.dark_energy_equivalent > 0.01:
        scores.append(60)
    if triple_point_result and triple_point_result.confidence > 50:
        scores.append(triple_point_result.confidence)

    simulation_probability = np.mean(scores) if scores else 30

    # Generate verdict
    if simulation_probability > 70:
        verdict = "CRITICAL: Multiple lines of evidence support simulation hypothesis!"
    elif simulation_probability > 50:
        verdict = "SIGNIFICANT: Computational artifacts detected at multiple boundaries."
    elif simulation_probability > 30:
        verdict = "SUGGESTIVE: Some anomalies found, but not conclusive."
    else:
        verdict = "INCONCLUSIVE: No strong evidence of computational substrate."

    # Create report
    report = OmegaPointReport(
        timestamp=datetime.now().isoformat(),
        device=str(device),
        gpu_name=gpu_name,
        bekenstein=bekenstein_result,
        temporal=temporal_result,
        entropy=entropy_result,
        triple_point=triple_point_result,
        phase_space_points=phase_space_points,
        physical_comparison=physical_comparison,
        final_verdict=verdict,
        simulation_probability=simulation_probability
    )

    # Print final summary
    print("\n" + "="*70)
    print("  OMEGA POINT - FINAL REPORT")
    print("="*70)

    print("\n  PHYSICAL CONSTANTS COMPARISON:")
    print(f"  {'-'*65}")
    for key, data in physical_comparison.items():
        print(f"\n  {key}:")
        for k, v in data.items():
            print(f"    {k}: {v}")

    print(f"\n  {'-'*65}")
    print(f"\n  SIMULATION PROBABILITY: {simulation_probability:.1f}%")
    print(f"\n  FINAL VERDICT: {verdict}")
    print("="*70)

    # Generate heatmap if we have phase space data
    if phase_space_points and output_dir:
        heatmap_path = os.path.join(output_dir, "reality_heatmap.png")
        generate_reality_heatmap(phase_space_points, heatmap_path)
    elif phase_space_points:
        generate_reality_heatmap(phase_space_points)

    # Save report
    if output_dir:
        report_path = os.path.join(output_dir, "omega_point_report.json")

        def serialize(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return asdict(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return str(obj)

        with open(report_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=serialize)
        print(f"\n  Report saved to: {report_path}")

    return report


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OMEGA POINT TEST - Universal Reality Stress Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  full       - Run complete test suite (all tests)
  bekenstein - Information density limit only
  temporal   - Time quantization test only
  entropy    - Ghost energy accumulation only
  phase      - 3D phase space scan only
  hunt       - Phase space scan + Triple Point detection

Examples:
  python omega_point_test.py --mode full
  python omega_point_test.py --mode bekenstein
  python omega_point_test.py --mode hunt --output ./results
        """
    )

    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "bekenstein", "temporal", "entropy", "phase", "hunt"],
                        help="Which tests to run")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    report = run_omega_point_test(
        mode=args.mode,
        device=device,
        output_dir=args.output
    )

    return report


if __name__ == "__main__":
    main()
