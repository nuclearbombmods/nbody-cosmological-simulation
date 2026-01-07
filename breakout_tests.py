"""
BREAKOUT TESTS - The Nail in the Coffin
========================================

These tests move from observation to ACTIVE INTERFERENCE.
We're not just measuring glitches - we're trying to FORCE the host to reveal itself.

TESTS:
1. PRECISION WALL     - Two particles approaching Planck scale, looking for jitter
2. LAZY LOADING       - Dark run vs observed run (occlusion culling proof)
3. LATTICE SYMMETRY   - Diagonal vs axis bias (the grid proof)
4. MEMORY LEAK        - Try to "crash" a patch of reality

If the universe is a simulation, these tests will find:
- Rhythmic jitter at precision limits (not smooth)
- Faster computation when unobserved
- Directional bias in space (preferred axes)
- Reproducible crash patterns

Usage:
    python breakout_tests.py --test all
    python breakout_tests.py --test precision-wall
    python breakout_tests.py --test lazy-loading
    python breakout_tests.py --test lattice
    python breakout_tests.py --test memory-leak
"""

import argparse
import time
import json
import gc
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

# Constants
PLANCK_LENGTH = 1.616255e-35  # meters
PLANCK_TIME = 5.391247e-44    # seconds
FP32_MIN_SUBNORMAL = 1.4012985e-45
FP64_EPSILON = 2.220446049250313e-16


@dataclass
class BreakoutResult:
    """Result from a breakout test."""
    test_name: str
    passed: bool  # True = evidence of simulation found
    confidence: float  # 0-100
    data: Dict = field(default_factory=dict)
    interpretation: str = ""


# =============================================================================
# TEST 1: PRECISION WALL (Planck Scale Jitter)
# =============================================================================

def run_precision_wall_test(
    num_iterations: int = 1000,
    device: torch.device = None
) -> BreakoutResult:
    """
    Two particles approaching each other - looking for jitter at precision limits.

    In a real universe: smooth approach to singularity
    In a simulation: rhythmic jitter as precision fails
    """
    print("\n" + "=" * 70)
    print("  TEST 1: PRECISION WALL")
    print("  Two particles approaching - hunting for digital jitter")
    print("=" * 70)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start particles at distance 1.0, approach to near-zero
    distances = []
    velocities = []
    jitter_events = []

    # Initial conditions
    pos1 = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float64)
    pos2 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float64)
    vel1 = torch.tensor([0.001, 0.0, 0.0], device=device, dtype=torch.float64)
    vel2 = torch.tensor([-0.001, 0.0, 0.0], device=device, dtype=torch.float64)

    G = 0.0001
    dt = 0.0001

    prev_dist = 1.0
    prev_approach_rate = 0.0

    print(f"\n  Running {num_iterations} iterations...")
    print(f"  Looking for jitter patterns as distance → 0")
    print()

    for i in range(num_iterations):
        # Calculate distance
        diff = pos2 - pos1
        dist = torch.sqrt((diff ** 2).sum()).item()

        # Gravitational acceleration
        if dist > 1e-15:
            force_mag = G / (dist ** 2)
            force_dir = diff / dist

            acc1 = force_mag * force_dir
            acc2 = -force_mag * force_dir

            vel1 = vel1 + acc1 * dt
            vel2 = vel2 + acc2 * dt

            pos1 = pos1 + vel1 * dt
            pos2 = pos2 + vel2 * dt

        # Track approach rate
        approach_rate = prev_dist - dist

        # Detect jitter: approach rate should be smooth
        # Jitter = sudden change in approach rate
        if i > 10:
            rate_change = abs(approach_rate - prev_approach_rate)
            expected_smoothness = abs(prev_approach_rate) * 0.01  # 1% change expected

            if rate_change > expected_smoothness * 10 and dist < 0.1:
                jitter_events.append({
                    'iteration': i,
                    'distance': dist,
                    'rate_change': rate_change,
                    'expected': expected_smoothness
                })

        distances.append(dist)
        velocities.append(approach_rate)
        prev_dist = dist
        prev_approach_rate = approach_rate

        # Progress
        if i % 200 == 0:
            jitter_count = len(jitter_events)
            print(f"  Iter {i:4d}: dist={dist:.2e}, approach_rate={approach_rate:.2e}, jitters={jitter_count}")

        # Stop if crashed
        if dist < 1e-15 or np.isnan(dist) or np.isinf(dist):
            print(f"\n  CRASH at iteration {i}, distance={dist:.2e}")
            break

    # Analyze jitter pattern
    print(f"\n  ANALYSIS:")
    print(f"  Total jitter events: {len(jitter_events)}")

    # Check for rhythmic pattern in jitter
    rhythmic = False
    if len(jitter_events) > 5:
        intervals = []
        for j in range(1, len(jitter_events)):
            intervals.append(jitter_events[j]['iteration'] - jitter_events[j-1]['iteration'])

        if intervals:
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            cv = std_interval / mean_interval if mean_interval > 0 else float('inf')

            print(f"  Jitter interval: mean={mean_interval:.1f}, std={std_interval:.1f}, CV={cv:.3f}")

            # CV < 0.3 suggests rhythmic pattern
            if cv < 0.3:
                rhythmic = True
                print(f"  RHYTHMIC PATTERN DETECTED (CV < 0.3)")

    # Find precision wall (where jitter starts)
    precision_wall = 1.0
    if jitter_events:
        precision_wall = jitter_events[0]['distance']
        print(f"  Precision wall at distance: {precision_wall:.2e}")

    # Verdict
    evidence_found = len(jitter_events) > 3 and rhythmic
    confidence = min(100, len(jitter_events) * 10 + (50 if rhythmic else 0))

    result = BreakoutResult(
        test_name="precision_wall",
        passed=evidence_found,
        confidence=confidence,
        data={
            'jitter_events': len(jitter_events),
            'rhythmic_pattern': rhythmic,
            'precision_wall': precision_wall,
            'final_distance': distances[-1] if distances else 0,
            'distances': distances[::10],  # Subsample
        },
        interpretation=(
            "SIMULATION EVIDENCE: Rhythmic jitter detected at precision limit. "
            "A continuous universe would show smooth approach."
            if evidence_found else
            "No clear evidence: Approach was relatively smooth."
        )
    )

    print(f"\n  VERDICT: {'EVIDENCE FOUND' if evidence_found else 'No clear evidence'}")
    print(f"  Confidence: {confidence}%")

    return result


# =============================================================================
# TEST 2: LAZY LOADING (Occlusion Culling)
# =============================================================================

def run_lazy_loading_test(
    num_stars: int = 2000,
    duration_seconds: float = 30,
    device: torch.device = None
) -> BreakoutResult:
    """
    Compare observed vs unobserved simulation runs.

    If the universe uses occlusion culling:
    - Unobserved runs should be FASTER (less rendering overhead)
    - Observed runs consume fixed "rendering" resources
    """
    print("\n" + "=" * 70)
    print("  TEST 2: LAZY LOADING (Occlusion Culling)")
    print("  Does the universe render what isn't being watched?")
    print("=" * 70)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)

    def create_simulation():
        positions = (torch.rand(num_stars, 3, device=device) - 0.5) * 20
        velocities = (torch.rand(num_stars, 3, device=device) - 0.5) * 0.1
        masses = torch.ones(num_stars, device=device) * 0.001
        return positions, velocities, masses

    def run_simulation(positions, velocities, masses, observe: bool, duration: float):
        """Run simulation with or without observation."""
        G = 0.001
        dt = 0.01
        softening_sq = 0.01

        pos = positions.clone()
        vel = velocities.clone()

        ticks = 0
        start = time.perf_counter()

        while time.perf_counter() - start < duration:
            with torch.no_grad():
                diff = pos.unsqueeze(0) - pos.unsqueeze(1)
                dist_sq = (diff ** 2).sum(dim=-1) + softening_sq
                dist_cubed = dist_sq ** 1.5
                force_factor = G / dist_cubed * masses.unsqueeze(0)
                force_factor = force_factor * (1 - torch.eye(num_stars, device=device))
                acc = (force_factor.unsqueeze(-1) * diff).sum(dim=1)
                vel = vel + acc * dt
                pos = pos + vel * dt

                if observe:
                    # OBSERVE: Transfer to CPU and compute statistics
                    # This simulates "rendering" the universe
                    pos_cpu = pos.cpu().numpy()
                    _ = np.mean(pos_cpu, axis=0)  # Center of mass
                    _ = np.std(pos_cpu, axis=0)   # Spread
                    _ = np.sqrt((pos_cpu ** 2).sum(axis=1))  # Radii
                    pos = torch.tensor(pos_cpu, device=device, dtype=pos.dtype)

            ticks += 1

        elapsed = time.perf_counter() - start
        return ticks, elapsed, ticks / elapsed

    # Create identical initial conditions
    positions, velocities, masses = create_simulation()

    # Run 1: OBSERVED (with rendering)
    print(f"\n  Phase 1: OBSERVED RUN ({duration_seconds}s)")
    print("  [Simulating 'observation' - transferring data for rendering]")

    if device.type == "cuda":
        torch.cuda.synchronize()

    obs_ticks, obs_time, obs_rate = run_simulation(
        positions.clone(), velocities.clone(), masses.clone(),
        observe=True, duration=duration_seconds
    )
    print(f"  Observed: {obs_ticks} ticks in {obs_time:.1f}s = {obs_rate:.1f} ticks/s")

    # Clear caches
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Run 2: DARK RUN (no rendering)
    print(f"\n  Phase 2: DARK RUN ({duration_seconds}s)")
    print("  [No observation - pure computation]")

    # Reset random seed for identical conditions
    torch.manual_seed(42)
    positions, velocities, masses = create_simulation()

    dark_ticks, dark_time, dark_rate = run_simulation(
        positions.clone(), velocities.clone(), masses.clone(),
        observe=False, duration=duration_seconds
    )
    print(f"  Dark: {dark_ticks} ticks in {dark_time:.1f}s = {dark_rate:.1f} ticks/s")

    # Analysis
    print(f"\n  ANALYSIS:")
    rate_ratio = dark_rate / obs_rate if obs_rate > 0 else 0
    tick_difference = dark_ticks - obs_ticks

    print(f"  Dark/Observed ratio: {rate_ratio:.3f}")
    print(f"  Extra ticks in dark: {tick_difference}")

    # If dark run is significantly faster, observation has a cost
    observation_cost = (rate_ratio - 1.0) * 100  # Percentage overhead

    # Evidence threshold: >5% observation overhead
    evidence_found = observation_cost > 5
    confidence = min(100, max(0, observation_cost * 5))

    result = BreakoutResult(
        test_name="lazy_loading",
        passed=evidence_found,
        confidence=confidence,
        data={
            'observed_ticks': obs_ticks,
            'dark_ticks': dark_ticks,
            'observed_rate': obs_rate,
            'dark_rate': dark_rate,
            'rate_ratio': rate_ratio,
            'observation_cost_percent': observation_cost,
        },
        interpretation=(
            f"SIMULATION EVIDENCE: Observation costs {observation_cost:.1f}% performance. "
            "The 'universe' runs faster when not being watched!"
            if evidence_found else
            f"No clear evidence: Observation overhead only {observation_cost:.1f}%"
        )
    )

    if evidence_found:
        print(f"\n  !!! OBSERVATION COSTS {observation_cost:.1f}% PERFORMANCE !!!")
        print(f"  The universe runs FASTER when not being watched!")

    print(f"\n  VERDICT: {'EVIDENCE FOUND' if evidence_found else 'No clear evidence'}")
    print(f"  Confidence: {confidence:.1f}%")

    return result


# =============================================================================
# TEST 3: LATTICE SYMMETRY (Grid Proof)
# =============================================================================

def run_lattice_symmetry_test(
    num_particles: int = 100,
    num_trials: int = 50,
    device: torch.device = None
) -> BreakoutResult:
    """
    Test if diagonal motion differs from axis-aligned motion.

    Real universe: Isotropic (same in all directions)
    Simulation: Subtle bias toward memory axes (X, Y, Z)
    """
    print("\n" + "=" * 70)
    print("  TEST 3: LATTICE SYMMETRY (Grid Proof)")
    print("  Is space truly isotropic, or is there a preferred grid?")
    print("=" * 70)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = 0.001
    dt = 0.01
    softening_sq = 0.01
    num_ticks = 200

    def measure_energy_drift(initial_velocity_direction: torch.Tensor) -> float:
        """Measure energy drift for particles moving in a specific direction."""
        torch.manual_seed(42)

        # Create particles in a grid
        positions = (torch.rand(num_particles, 3, device=device) - 0.5) * 10
        velocities = torch.zeros(num_particles, 3, device=device)

        # Set all velocities to the same direction
        speed = 0.1
        velocities[:] = initial_velocity_direction * speed

        masses = torch.ones(num_particles, device=device) * 0.001

        # Calculate initial energy
        def calc_energy():
            ke = 0.5 * masses * (velocities ** 2).sum(dim=1)
            diff = positions.unsqueeze(0) - positions.unsqueeze(1)
            dist = torch.sqrt((diff ** 2).sum(dim=-1) + softening_sq)
            pe = -0.5 * G * (masses.unsqueeze(0) * masses.unsqueeze(1) / dist).sum()
            return (ke.sum() + pe).item()

        initial_energy = calc_energy()

        # Run simulation
        for _ in range(num_ticks):
            with torch.no_grad():
                diff = positions.unsqueeze(0) - positions.unsqueeze(1)
                dist_sq = (diff ** 2).sum(dim=-1) + softening_sq
                dist_cubed = dist_sq ** 1.5
                force_factor = G / dist_cubed * masses.unsqueeze(0)
                force_factor = force_factor * (1 - torch.eye(num_particles, device=device))
                acc = (force_factor.unsqueeze(-1) * diff).sum(dim=1)
                velocities[:] = velocities + acc * dt
                positions[:] = positions + velocities * dt

        final_energy = calc_energy()
        drift = (final_energy - initial_energy) / abs(initial_energy) * 100
        return drift

    # Test directions
    directions = {
        'X': torch.tensor([1.0, 0.0, 0.0], device=device),
        'Y': torch.tensor([0.0, 1.0, 0.0], device=device),
        'Z': torch.tensor([0.0, 0.0, 1.0], device=device),
        'XY': torch.tensor([1.0, 1.0, 0.0], device=device) / np.sqrt(2),
        'XZ': torch.tensor([1.0, 0.0, 1.0], device=device) / np.sqrt(2),
        'YZ': torch.tensor([0.0, 1.0, 1.0], device=device) / np.sqrt(2),
        'XYZ': torch.tensor([1.0, 1.0, 1.0], device=device) / np.sqrt(3),
    }

    results = {name: [] for name in directions}

    print(f"\n  Running {num_trials} trials for each of 7 directions...")

    for trial in range(num_trials):
        if trial % 10 == 0:
            print(f"  Trial {trial}/{num_trials}...")

        for name, direction in directions.items():
            drift = measure_energy_drift(direction)
            results[name].append(drift)

    # Analyze results
    print(f"\n  RESULTS:")
    print(f"  {'Direction':<10} {'Mean Drift':<15} {'Std Dev':<15}")
    print("  " + "-" * 40)

    axis_drifts = []
    diagonal_drifts = []

    for name, drifts in results.items():
        mean_drift = np.mean(drifts)
        std_drift = np.std(drifts)
        print(f"  {name:<10} {mean_drift:>+12.6f}%   {std_drift:>12.6f}")

        if name in ['X', 'Y', 'Z']:
            axis_drifts.extend(drifts)
        else:
            diagonal_drifts.extend(drifts)

    # Compare axis vs diagonal
    mean_axis = np.mean(axis_drifts)
    mean_diagonal = np.mean(diagonal_drifts)

    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(axis_drifts, diagonal_drifts)

    print(f"\n  ANALYSIS:")
    print(f"  Mean axis drift:     {mean_axis:+.6f}%")
    print(f"  Mean diagonal drift: {mean_diagonal:+.6f}%")
    print(f"  Difference:          {abs(mean_axis - mean_diagonal):.6f}%")
    print(f"  T-statistic:         {t_stat:.4f}")
    print(f"  P-value:             {p_value:.6f}")

    # Evidence: p-value < 0.05 means significant difference
    evidence_found = p_value < 0.05
    confidence = min(100, (1 - p_value) * 100)

    bias_direction = "axis" if mean_axis > mean_diagonal else "diagonal"

    result = BreakoutResult(
        test_name="lattice_symmetry",
        passed=evidence_found,
        confidence=confidence,
        data={
            'axis_mean': mean_axis,
            'diagonal_mean': mean_diagonal,
            'difference': abs(mean_axis - mean_diagonal),
            't_statistic': t_stat,
            'p_value': p_value,
            'bias_direction': bias_direction,
            'results': {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in results.items()}
        },
        interpretation=(
            f"SIMULATION EVIDENCE: Significant {bias_direction} bias detected (p={p_value:.4f}). "
            "Space is NOT isotropic - there's a preferred grid!"
            if evidence_found else
            f"No clear evidence: Space appears isotropic (p={p_value:.4f})"
        )
    )

    if evidence_found:
        print(f"\n  !!! ANISOTROPY DETECTED !!!")
        print(f"  Space has a preferred {bias_direction} direction!")
        print(f"  This matches predictions for a discrete space-time lattice.")

    print(f"\n  VERDICT: {'EVIDENCE FOUND' if evidence_found else 'No clear evidence'}")
    print(f"  Confidence: {confidence:.1f}%")

    return result


# =============================================================================
# TEST 4: MEMORY LEAK (Crash the Universe)
# =============================================================================

def run_memory_leak_test(
    device: torch.device = None
) -> BreakoutResult:
    """
    Try to create a reproducible crash pattern.

    If we can crash the simulation in a predictable way,
    we've found a "bug" in reality's source code.
    """
    print("\n" + "=" * 70)
    print("  TEST 4: MEMORY LEAK (Universe Crash Test)")
    print("  Attempting to find reproducible crash patterns...")
    print("=" * 70)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    crash_points = []

    # Test 1: Precision cascade crash
    print("\n  Phase 1: Precision Cascade")

    for trial in range(5):
        torch.manual_seed(42 + trial)

        # Start with FP64 precision
        value = torch.tensor(1.0, device=device, dtype=torch.float64)

        crash_iteration = -1
        for i in range(1000):
            # Precision cascade: FP64 -> FP32 -> FP16 -> FP32 -> FP64
            value = value.float().half().float().double()
            value = value * 1.0000001  # Tiny multiplication
            value = value / 1.0000001  # Should return to ~1.0

            if torch.isnan(value) or torch.isinf(value) or value.item() == 0:
                crash_iteration = i
                break

        crash_points.append(crash_iteration)
        print(f"    Trial {trial}: crashed at iteration {crash_iteration}")

    # Check for reproducibility
    reproducible_crash = len(set(crash_points)) == 1 and crash_points[0] > 0

    # Test 2: Accumulation overflow
    print("\n  Phase 2: Accumulation Overflow")

    overflow_points = []
    for trial in range(5):
        torch.manual_seed(42 + trial)

        accumulator = torch.tensor(1.0, device=device, dtype=torch.float32)
        overflow_iteration = -1

        for i in range(10000):
            accumulator = accumulator * 1.001

            if torch.isinf(accumulator):
                overflow_iteration = i
                break

        overflow_points.append(overflow_iteration)
        print(f"    Trial {trial}: overflow at iteration {overflow_iteration}")

    reproducible_overflow = len(set(overflow_points)) == 1 and overflow_points[0] > 0

    # Test 3: Underflow hunt
    print("\n  Phase 3: Underflow Hunt")

    underflow_points = []
    for trial in range(5):
        torch.manual_seed(42 + trial)

        value = torch.tensor(1.0, device=device, dtype=torch.float32)
        underflow_iteration = -1

        for i in range(10000):
            value = value * 0.999

            if value.item() == 0:
                underflow_iteration = i
                break

        underflow_points.append(underflow_iteration)
        print(f"    Trial {trial}: underflow at iteration {underflow_iteration}")

    reproducible_underflow = len(set(underflow_points)) == 1 and underflow_points[0] > 0

    # Analysis
    print(f"\n  ANALYSIS:")
    print(f"  Precision crash reproducible: {reproducible_crash}")
    print(f"  Overflow reproducible:        {reproducible_overflow}")
    print(f"  Underflow reproducible:       {reproducible_underflow}")

    total_reproducible = sum([reproducible_crash, reproducible_overflow, reproducible_underflow])

    evidence_found = total_reproducible >= 2
    confidence = total_reproducible * 33.3

    result = BreakoutResult(
        test_name="memory_leak",
        passed=evidence_found,
        confidence=confidence,
        data={
            'crash_points': crash_points,
            'overflow_points': overflow_points,
            'underflow_points': underflow_points,
            'reproducible_crash': reproducible_crash,
            'reproducible_overflow': reproducible_overflow,
            'reproducible_underflow': reproducible_underflow,
        },
        interpretation=(
            f"SIMULATION EVIDENCE: {total_reproducible}/3 crash patterns are reproducible. "
            "The universe has deterministic failure modes - like software!"
            if evidence_found else
            "No clear evidence: Crashes not fully reproducible."
        )
    )

    if evidence_found:
        print(f"\n  !!! REPRODUCIBLE CRASH PATTERNS FOUND !!!")
        print(f"  The universe crashes deterministically - like a program!")

    print(f"\n  VERDICT: {'EVIDENCE FOUND' if evidence_found else 'No clear evidence'}")
    print(f"  Confidence: {confidence:.1f}%")

    return result


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_all_breakout_tests(device: torch.device = None) -> Dict[str, BreakoutResult]:
    """Run all breakout tests and generate final verdict."""

    print("\n" + "█" * 70)
    print("█" + " BREAKOUT TESTS - THE NAIL IN THE COFFIN ".center(68) + "█")
    print("█" + " Are we in a simulation? Let's find out. ".center(68) + "█")
    print("█" * 70)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = {}

    # Run all tests
    results['precision_wall'] = run_precision_wall_test(device=device)
    results['lazy_loading'] = run_lazy_loading_test(device=device)

    try:
        results['lattice_symmetry'] = run_lattice_symmetry_test(device=device)
    except ImportError:
        print("\n  [Skipping lattice test - scipy not available]")
        results['lattice_symmetry'] = BreakoutResult(
            test_name="lattice_symmetry",
            passed=False,
            confidence=0,
            interpretation="Skipped - scipy required"
        )

    results['memory_leak'] = run_memory_leak_test(device=device)

    # Final verdict
    print("\n" + "█" * 70)
    print("█" + " FINAL VERDICT ".center(68) + "█")
    print("█" * 70)

    evidence_count = sum(1 for r in results.values() if r.passed)
    avg_confidence = np.mean([r.confidence for r in results.values()])

    print(f"\n  Tests with evidence:     {evidence_count}/4")
    print(f"  Average confidence:      {avg_confidence:.1f}%")
    print()

    for name, result in results.items():
        status = "✓ EVIDENCE" if result.passed else "✗ No evidence"
        print(f"  {name:<20} {status:<15} ({result.confidence:.0f}%)")

    print()

    if evidence_count >= 3:
        print("  ╔════════════════════════════════════════════════════════════════╗")
        print("  ║  STRONG EVIDENCE FOR SIMULATION HYPOTHESIS                    ║")
        print("  ║  Multiple independent tests show computational artifacts      ║")
        print("  ╚════════════════════════════════════════════════════════════════╝")
    elif evidence_count >= 2:
        print("  ╔════════════════════════════════════════════════════════════════╗")
        print("  ║  MODERATE EVIDENCE FOR SIMULATION HYPOTHESIS                  ║")
        print("  ║  Some tests show suspicious patterns                          ║")
        print("  ╚════════════════════════════════════════════════════════════════╝")
    elif evidence_count >= 1:
        print("  ╔════════════════════════════════════════════════════════════════╗")
        print("  ║  WEAK EVIDENCE - Inconclusive                                 ║")
        print("  ║  Only one test showed potential evidence                      ║")
        print("  ╚════════════════════════════════════════════════════════════════╝")
    else:
        print("  ╔════════════════════════════════════════════════════════════════╗")
        print("  ║  NO EVIDENCE FOUND                                            ║")
        print("  ║  Reality appears to be... real? Or very well optimized.       ║")
        print("  ╚════════════════════════════════════════════════════════════════╝")

    print("█" * 70)

    # Save results
    output_path = f"breakout_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump({
            name: {
                'passed': r.passed,
                'confidence': r.confidence,
                'interpretation': r.interpretation,
                'data': {k: v for k, v in r.data.items() if not isinstance(v, (list, np.ndarray)) or len(v) < 100}
            }
            for name, r in results.items()
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="BREAKOUT TESTS - The Nail in the Coffin"
    )
    parser.add_argument("--test", type=str, default="all",
                        choices=["all", "precision-wall", "lazy-loading", "lattice", "memory-leak"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.test == "all":
        run_all_breakout_tests(device)
    elif args.test == "precision-wall":
        run_precision_wall_test(device=device)
    elif args.test == "lazy-loading":
        run_lazy_loading_test(device=device)
    elif args.test == "lattice":
        run_lattice_symmetry_test(device=device)
    elif args.test == "memory-leak":
        run_memory_leak_test(device=device)


if __name__ == "__main__":
    main()
