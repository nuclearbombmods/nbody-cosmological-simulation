"""
Sensitivity Test - Precision vs Dark Matter Effect

Tests whether the "dark matter effect" scales predictably with precision.
If the hypothesis is correct:
  - Lower precision = MORE dark matter effect
  - Higher precision = LESS dark matter effect
  - The relationship should be smooth and monotonic

This is the key scientific test. A predictable scaling relationship
would be strong evidence for the "rounding error" hypothesis.

Usage:
    python sensitivity_test.py
    python sensitivity_test.py --stars 3000 --ticks 1000
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

from galaxy import create_disk_galaxy
from simulation import GalaxySimulation
from quantization import PrecisionMode, _grid_quantize_safe
from metrics import compute_rotation_curve


@dataclass
class SensitivityResult:
    """Results from a single precision level test."""
    bits: float              # Effective bits (log2 of levels)
    levels: int              # Number of quantization levels
    label: str               # Human readable label
    energy_drift_pct: float  # Energy change as percentage
    outer_slope: float       # Rotation curve outer slope
    mean_outer_velocity: float
    final_radius: float      # Galaxy radius at end


def custom_quantize_simulation(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    masses: torch.Tensor,
    levels: int,
    num_ticks: int = 500,
    device: torch.device = None
) -> SensitivityResult:
    """
    Run simulation with a specific number of quantization levels.
    """
    # Create a custom simulation that uses our specific level count
    class CustomQuantSim(GalaxySimulation):
        def __init__(self, *args, quant_levels: int, **kwargs):
            # MUST set quant_levels BEFORE super().__init__ because it calls _compute_accelerations
            self.quant_levels = quant_levels
            super().__init__(*args, **kwargs)

        def _compute_accelerations(self):
            pos = self.positions
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)
            dist_sq = (diff ** 2).sum(dim=-1) + self.softening_sq

            # Apply custom quantization
            if self.quant_levels < 10000:  # Only quantize if not "infinite"
                dist_sq = _grid_quantize_safe(dist_sq, self.quant_levels, min_val=0.01)

            dist_cubed = dist_sq ** 1.5
            force_factor = self.G / dist_cubed
            force_factor = force_factor * self.masses.unsqueeze(0)
            force_factor = force_factor * (1 - torch.eye(self.num_stars, device=self.device))
            accelerations = (force_factor.unsqueeze(-1) * diff).sum(dim=1)

            return accelerations

    sim = CustomQuantSim(
        positions.clone(),
        velocities.clone(),
        masses.clone(),
        quant_levels=levels,
        precision_mode=PrecisionMode.FLOAT32,
        G=0.001,
        dt=0.01,
        softening=0.1,
        device=device
    )

    initial_energy = sim.get_total_energy()

    # Run simulation
    for _ in range(num_ticks):
        sim.step()

    final_energy = sim.get_total_energy()
    energy_drift = (final_energy - initial_energy) / abs(initial_energy) * 100

    # Compute rotation curve
    curve = compute_rotation_curve(sim.positions, sim.velocities, num_bins=12)
    valid = ~np.isnan(curve["velocities"])
    radii = curve["radii"][valid]
    vels = curve["velocities"][valid]

    # Compute outer slope (flatness indicator)
    if len(vels) >= 4:
        mid = len(vels) // 2
        outer_r, outer_v = radii[mid:], vels[mid:]
        if len(outer_r) >= 2:
            slope = np.polyfit(outer_r, outer_v, 1)[0]
            mean_v = outer_v.mean()
        else:
            slope, mean_v = 0, 0
    else:
        slope, mean_v = 0, 0

    # Galaxy radius
    pos_np = sim.positions.cpu().numpy()
    radii_all = np.sqrt((pos_np ** 2).sum(axis=1))
    final_radius = np.percentile(radii_all, 90)

    bits = np.log2(levels) if levels > 1 else 0
    label = f"{levels} levels ({bits:.1f} bits)"

    return SensitivityResult(
        bits=bits,
        levels=levels,
        label=label,
        energy_drift_pct=energy_drift,
        outer_slope=slope,
        mean_outer_velocity=mean_v,
        final_radius=final_radius
    )


def run_sensitivity_sweep(
    num_stars: int = 2000,
    num_ticks: int = 500,
    device: torch.device = None
) -> list[SensitivityResult]:
    """
    Run simulations across a range of precision levels.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Precision levels to test (from extreme to baseline)
    # levels = 2^bits, so: 4=2bit, 8=3bit, 16=4bit, 32=5bit, etc.
    test_levels = [
        4,      # 2-bit (EXTREME - 4 levels only)
        8,      # 3-bit
        16,     # 4-bit (int4)
        32,     # 5-bit
        64,     # 6-bit
        128,    # 7-bit
        256,    # 8-bit (int8)
        512,    # 9-bit
        1024,   # 10-bit
        4096,   # 12-bit
        65536,  # 16-bit equivalent
        100000, # "Infinite" (essentially float)
    ]

    print(f"\nCreating test galaxy with {num_stars} stars...")
    positions, velocities, masses = create_disk_galaxy(
        num_stars=num_stars,
        galaxy_radius=10.0,
        device=device
    )
    positions = positions.float()
    velocities = velocities.float()
    masses = masses.float()

    results = []

    print(f"\nRunning sensitivity sweep ({len(test_levels)} precision levels)...")
    print("-" * 60)

    for i, levels in enumerate(test_levels):
        bits = np.log2(levels) if levels > 1 else 0
        print(f"  [{i+1}/{len(test_levels)}] Testing {levels} levels ({bits:.1f} bits)...", end=" ")

        result = custom_quantize_simulation(
            positions, velocities, masses,
            levels=levels,
            num_ticks=num_ticks,
            device=device
        )
        results.append(result)

        print(f"Energy drift: {result.energy_drift_pct:+.2f}%, Slope: {result.outer_slope:+.4f}")

    return results


def plot_sensitivity(results: list[SensitivityResult], save_path: str = None):
    """
    Plot the relationship between precision and dark matter effect.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    bits = [r.bits for r in results]
    energy_drift = [r.energy_drift_pct for r in results]
    slopes = [r.outer_slope for r in results]
    mean_v = [r.mean_outer_velocity for r in results]
    radii = [r.final_radius for r in results]

    # 1. Energy Drift vs Precision
    ax1 = axes[0, 0]
    ax1.semilogx([r.levels for r in results], energy_drift, 'o-', linewidth=2, markersize=8, color='red')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Quantization Levels (log scale)", fontsize=12)
    ax1.set_ylabel("Energy Drift (%)", fontsize=12)
    ax1.set_title("Energy Injection vs Precision", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Add annotation for key points
    ax1.annotate('int4 (16 levels)', xy=(16, energy_drift[2]), xytext=(30, energy_drift[2]+5),
                 arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10)
    ax1.annotate('int8 (256 levels)', xy=(256, energy_drift[6]), xytext=(400, energy_drift[6]+2),
                 arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10)

    # 2. Rotation Curve Slope vs Precision
    ax2 = axes[0, 1]
    ax2.semilogx([r.levels for r in results], slopes, 'o-', linewidth=2, markersize=8, color='blue')
    ax2.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Flat curve (DM signature)')
    ax2.set_xlabel("Quantization Levels (log scale)", fontsize=12)
    ax2.set_ylabel("Outer Rotation Curve Slope", fontsize=12)
    ax2.set_title("Curve Flatness vs Precision", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Energy vs Bits (linear scale)
    ax3 = axes[1, 0]
    ax3.plot(bits, energy_drift, 'o-', linewidth=2, markersize=8, color='red')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel("Effective Bits", fontsize=12)
    ax3.set_ylabel("Energy Drift (%)", fontsize=12)
    ax3.set_title("Energy Injection vs Bit Depth", fontsize=14)
    ax3.grid(True, alpha=0.3)

    # Fit exponential decay
    valid_idx = [i for i, e in enumerate(energy_drift) if e > 0.01]
    if len(valid_idx) >= 3:
        from numpy.polynomial import polynomial as P
        log_drift = np.log([energy_drift[i] for i in valid_idx])
        bits_valid = [bits[i] for i in valid_idx]
        coeffs = np.polyfit(bits_valid, log_drift, 1)
        fit_bits = np.linspace(min(bits), max(bits), 50)
        fit_drift = np.exp(coeffs[1] + coeffs[0] * fit_bits)
        ax3.plot(fit_bits, fit_drift, '--', color='orange', alpha=0.7,
                 label=f'Exponential fit (slope={coeffs[0]:.2f})')
        ax3.legend()

    # 4. Summary table as text
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create summary text
    summary = "SENSITIVITY ANALYSIS SUMMARY\n"
    summary += "=" * 40 + "\n\n"

    # Check for monotonic relationship
    energy_monotonic = all(energy_drift[i] >= energy_drift[i+1] for i in range(len(energy_drift)-1))

    summary += f"Energy drift monotonic: {'YES' if energy_monotonic else 'NO'}\n\n"

    summary += "Key findings:\n"
    summary += f"  - 2-bit (4 levels):   {energy_drift[0]:+.2f}% drift\n"
    summary += f"  - 4-bit (16 levels):  {energy_drift[2]:+.2f}% drift\n"
    summary += f"  - 8-bit (256 levels): {energy_drift[6]:+.2f}% drift\n"
    summary += f"  - 16-bit equivalent:  {energy_drift[10]:+.2f}% drift\n\n"

    # Interpretation
    if energy_monotonic and energy_drift[0] > 5 and energy_drift[-1] < 1:
        summary += "RESULT: Clear correlation!\n"
        summary += "Lower precision = more 'ghost force'\n"
        summary += "Effect scales predictably with bit depth."
    elif energy_drift[0] > energy_drift[-1]:
        summary += "RESULT: Weak correlation.\n"
        summary += "Trend exists but not perfectly monotonic."
    else:
        summary += "RESULT: No clear correlation.\n"
        summary += "Precision doesn't predict effect strength."

    ax4.text(0.1, 0.9, summary, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle("Sensitivity Analysis: Does Precision Predict Dark Matter Effect?",
                 fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to {save_path}")

    return fig


def print_results_table(results: list[SensitivityResult]):
    """Print detailed results table."""
    print(f"\n{'='*70}")
    print("SENSITIVITY TEST RESULTS")
    print(f"{'='*70}")
    print()
    print(f"{'Levels':<10} {'Bits':<8} {'Energy Drift':<15} {'Outer Slope':<15} {'Radius':<10}")
    print("-" * 70)

    for r in results:
        print(f"{r.levels:<10} {r.bits:<8.1f} {r.energy_drift_pct:+12.4f}%   {r.outer_slope:+12.6f}   {r.final_radius:.2f}")

    print()
    print("Interpretation:")
    print("  - If energy drift DECREASES as levels INCREASE → Precision affects 'ghost force'")
    print("  - If outer slope INCREASES as levels INCREASE → Less 'dark matter' at higher precision")
    print("  - A smooth monotonic trend = strong evidence for the hypothesis")


def main():
    parser = argparse.ArgumentParser(description="Test precision sensitivity")
    parser.add_argument("--stars", type=int, default=2000, help="Number of stars")
    parser.add_argument("--ticks", type=int, default=500, help="Simulation ticks")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = run_sensitivity_sweep(
        num_stars=args.stars,
        num_ticks=args.ticks,
        device=device
    )

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    print_results_table(results)

    plot_sensitivity(results, save_path=str(output_dir / "sensitivity_analysis.png"))

    plt.show()


if __name__ == "__main__":
    main()
