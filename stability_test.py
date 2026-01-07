"""
Stability Test - Find the Quantization Floor

Tests all precision modes to find where stability breaks down.
Creates a table showing ticks until explosion for each mode.

Usage:
    python stability_test.py
    python stability_test.py --stars 3000 --max-ticks 5000
"""

import argparse
import torch
import time
from dataclasses import dataclass

from galaxy import create_disk_galaxy
from simulation import GalaxySimulation
from quantization import PrecisionMode, describe_mode


@dataclass
class StabilityResult:
    """Results from a stability test run."""
    mode: str
    stable_ticks: int
    final_energy: float
    initial_energy: float
    energy_drift_percent: float
    exploded: bool
    runtime_seconds: float


def detect_explosion(sim: GalaxySimulation, initial_energy: float) -> bool:
    """
    Detect if the simulation has exploded.

    Signs of explosion:
    - Energy increased by more than 1000%
    - Energy became positive (unbound) when it started negative
    - Any NaN or Inf values
    """
    current_energy = sim.get_total_energy()

    # Check for NaN/Inf
    if not torch.isfinite(sim.positions).all():
        return True
    if not torch.isfinite(sim.velocities).all():
        return True

    # Check for massive energy increase
    if abs(initial_energy) > 1e-10:
        drift = abs(current_energy - initial_energy) / abs(initial_energy)
        if drift > 10.0:  # More than 1000% drift
            return True

    # Check if bound system became unbound
    if initial_energy < 0 and current_energy > abs(initial_energy):
        return True

    return False


def test_precision_mode(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    masses: torch.Tensor,
    mode: PrecisionMode,
    max_ticks: int = 2000,
    check_interval: int = 50,
    **sim_kwargs
) -> StabilityResult:
    """
    Test a single precision mode for stability.

    Returns when the simulation explodes or reaches max_ticks.
    """
    print(f"\n  Testing {mode.value}...", end=" ", flush=True)

    start_time = time.time()

    sim = GalaxySimulation(
        positions.clone(),
        velocities.clone(),
        masses.clone(),
        precision_mode=mode,
        **sim_kwargs
    )

    initial_energy = sim.get_total_energy()
    stable_ticks = 0
    exploded = False

    for tick in range(0, max_ticks, check_interval):
        # Run a batch of ticks
        for _ in range(check_interval):
            sim.step()

        stable_ticks = tick + check_interval

        # Check for explosion
        if detect_explosion(sim, initial_energy):
            exploded = True
            print(f"EXPLODED at tick {stable_ticks}")
            break

        # Progress indicator
        if stable_ticks % 500 == 0:
            print(f"{stable_ticks}", end=" ", flush=True)

    runtime = time.time() - start_time
    final_energy = sim.get_total_energy()

    if abs(initial_energy) > 1e-10:
        drift_percent = (final_energy - initial_energy) / abs(initial_energy) * 100
    else:
        drift_percent = 0.0

    if not exploded:
        print(f"STABLE ({max_ticks} ticks, {drift_percent:+.2f}% drift)")

    return StabilityResult(
        mode=mode.value,
        stable_ticks=stable_ticks,
        final_energy=final_energy,
        initial_energy=initial_energy,
        energy_drift_percent=drift_percent,
        exploded=exploded,
        runtime_seconds=runtime
    )


def run_stability_suite(
    num_stars: int = 2000,
    max_ticks: int = 2000,
    device: torch.device = None
) -> list[StabilityResult]:
    """
    Run stability tests for all precision modes.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print("STABILITY TEST SUITE")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Stars: {num_stars}")
    print(f"Max ticks: {max_ticks}")

    # Create initial galaxy (same for all tests)
    print("\nCreating test galaxy...")
    positions, velocities, masses = create_disk_galaxy(
        num_stars=num_stars,
        galaxy_radius=10.0,
        device=device
    )
    positions = positions.float()
    velocities = velocities.float()
    masses = masses.float()

    # Modes to test (from most to least stable expected)
    modes = [
        PrecisionMode.FLOAT64,
        PrecisionMode.FLOAT32,
        PrecisionMode.BFLOAT16,
        PrecisionMode.FLOAT16,
        PrecisionMode.INT8_SIM,
        PrecisionMode.INT4_SIM,
    ]

    print("\nRunning stability tests...")
    results = []

    for mode in modes:
        result = test_precision_mode(
            positions, velocities, masses,
            mode=mode,
            max_ticks=max_ticks,
            device=device,
            G=0.001,
            dt=0.01,
            softening=0.1
        )
        results.append(result)

    return results


def print_results_table(results: list[StabilityResult]):
    """Print results as a formatted table."""
    print(f"\n{'='*80}")
    print("RESULTS: Quantization Stability Floor")
    print(f"{'='*80}")
    print()
    print(f"{'Precision':<15} {'Ticks':<10} {'Status':<12} {'Energy Drift':<15} {'Time (s)':<10}")
    print("-" * 62)

    for r in results:
        status = "EXPLODED" if r.exploded else "STABLE"
        drift_str = f"{r.energy_drift_percent:+.4f}%" if not r.exploded else "∞"

        # Color coding via symbols
        if r.exploded:
            status_symbol = "💥"
        elif abs(r.energy_drift_percent) < 0.1:
            status_symbol = "✅"
        elif abs(r.energy_drift_percent) < 1.0:
            status_symbol = "⚠️"
        else:
            status_symbol = "🔶"

        print(f"{r.mode:<15} {r.stable_ticks:<10} {status_symbol} {status:<10} {drift_str:<15} {r.runtime_seconds:.2f}")

    print()
    print("Legend:")
    print("  ✅ = Excellent stability (< 0.1% drift)")
    print("  ⚠️ = Good stability (< 1% drift)")
    print("  🔶 = Marginal stability (> 1% drift)")
    print("  💥 = Exploded (simulation unstable)")


def main():
    parser = argparse.ArgumentParser(description="Test precision stability thresholds")
    parser.add_argument("--stars", type=int, default=2000, help="Number of stars")
    parser.add_argument("--max-ticks", type=int, default=2000, help="Maximum ticks to test")
    args = parser.parse_args()

    results = run_stability_suite(
        num_stars=args.stars,
        max_ticks=args.max_ticks
    )

    print_results_table(results)

    # Find the stability threshold
    stable_modes = [r for r in results if not r.exploded]
    unstable_modes = [r for r in results if r.exploded]

    if stable_modes and unstable_modes:
        threshold = stable_modes[-1].mode  # Last stable mode
        print(f"\n📊 Stability Threshold: {threshold}")
        print(f"   Modes above this are stable; modes below explode.")
    elif not unstable_modes:
        print(f"\n📊 All modes are stable at {args.max_ticks} ticks!")
        print("   Try increasing --max-ticks or reducing softening.")


if __name__ == "__main__":
    main()
