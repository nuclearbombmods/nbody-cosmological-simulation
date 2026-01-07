"""
Density Limit Test - Does the Simulation Lag Before It Crashes?

Test how the simulation behaves as we increase star density.
If we're in a simulation, there should be a "render limit" where
the parent universe starts lagging before things actually break.

We look for:
1. Non-linear scaling - O(N²) should become O(N³) or worse at limits
2. Pre-crash lag - timing anomalies before actual crashes
3. Unexplained overhead that scales with density
4. The exact density where "ghost forces" appear

Usage:
    python density_limit_test.py
    python density_limit_test.py --max-stars 15000
"""

import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

from galaxy import create_disk_galaxy
from simulation import GalaxySimulation
from quantization import PrecisionMode, _grid_quantize_safe
from metrics import compute_rotation_curve

# Try GPU power monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_POWER = True
except:
    HAS_POWER = False
    GPU_HANDLE = None


@dataclass
class DensityResult:
    """Results from a density test."""
    num_stars: int
    time_per_tick: float          # Seconds per simulation tick
    total_time: float             # Total test time
    ticks_completed: int
    energy_drift_pct: float
    avg_gpu_power: float          # Average power draw
    power_per_star: float         # Power normalized by star count
    power_per_interaction: float  # Power normalized by N²
    rotation_curve_slope: float   # Flatness of final curve
    crashed: bool
    crash_reason: str


def get_gpu_power() -> float:
    """Get current GPU power in watts."""
    if HAS_POWER and GPU_HANDLE:
        try:
            return pynvml.nvmlDeviceGetPowerUsage(GPU_HANDLE) / 1000.0
        except:
            pass
    return 0.0


def run_density_test(
    num_stars: int,
    num_ticks: int = 200,
    use_quantization: bool = False,
    device: torch.device = None
) -> DensityResult:
    """
    Run a single density test.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create galaxy
    positions, velocities, masses = create_disk_galaxy(
        num_stars=num_stars,
        galaxy_radius=10.0,
        device=device
    )
    positions = positions.float()
    velocities = velocities.float()
    masses = masses.float()

    # Create simulation (with or without quantization)
    if use_quantization:
        class QuantSim(GalaxySimulation):
            def __init__(self, *args, **kwargs):
                self.quant_levels = 16  # int4 equivalent
                super().__init__(*args, **kwargs)

            def _compute_accelerations(self):
                pos = self.positions
                diff = pos.unsqueeze(0) - pos.unsqueeze(1)
                dist_sq = (diff ** 2).sum(dim=-1) + self.softening_sq
                dist_sq = _grid_quantize_safe(dist_sq, 16, min_val=0.01)
                dist_cubed = dist_sq ** 1.5
                force_factor = self.G / dist_cubed
                force_factor = force_factor * self.masses.unsqueeze(0)
                force_factor = force_factor * (1 - torch.eye(self.num_stars, device=self.device))
                return (force_factor.unsqueeze(-1) * diff).sum(dim=1)

        sim = QuantSim(
            positions, velocities, masses,
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001, dt=0.01, softening=0.1, device=device
        )
    else:
        sim = GalaxySimulation(
            positions, velocities, masses,
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001, dt=0.01, softening=0.1, device=device
        )

    initial_energy = sim.get_total_energy()

    # Run simulation and measure timing
    power_samples = []
    tick_times = []
    crashed = False
    crash_reason = ""

    start_time = time.time()

    for tick in range(num_ticks):
        tick_start = time.time()

        try:
            sim.step()

            # Check for crash conditions
            if torch.isnan(sim.positions).any():
                crashed = True
                crash_reason = "NaN positions"
                break
            if torch.isinf(sim.positions).any():
                crashed = True
                crash_reason = "Infinite positions"
                break

            # Sample power
            power = get_gpu_power()
            if power > 0:
                power_samples.append(power)

        except Exception as e:
            crashed = True
            crash_reason = str(e)
            break

        tick_times.append(time.time() - tick_start)

    total_time = time.time() - start_time
    ticks_completed = len(tick_times)

    # Calculate metrics
    final_energy = sim.get_total_energy() if not crashed else float('nan')
    energy_drift = ((final_energy - initial_energy) / abs(initial_energy) * 100
                    if not crashed and initial_energy != 0 else float('nan'))

    avg_power = np.mean(power_samples) if power_samples else 0.0
    avg_tick_time = np.mean(tick_times) if tick_times else 0.0

    # Power scaling
    power_per_star = avg_power / num_stars if num_stars > 0 else 0
    power_per_interaction = avg_power / (num_stars ** 2) if num_stars > 0 else 0

    # Rotation curve analysis
    if not crashed:
        curve = compute_rotation_curve(sim.positions, sim.velocities, num_bins=12)
        valid = ~np.isnan(curve["velocities"])
        if valid.sum() >= 4:
            radii = curve["radii"][valid]
            vels = curve["velocities"][valid]
            mid = len(vels) // 2
            if len(radii[mid:]) >= 2:
                slope = np.polyfit(radii[mid:], vels[mid:], 1)[0]
            else:
                slope = 0
        else:
            slope = 0
    else:
        slope = float('nan')

    return DensityResult(
        num_stars=num_stars,
        time_per_tick=avg_tick_time,
        total_time=total_time,
        ticks_completed=ticks_completed,
        energy_drift_pct=energy_drift,
        avg_gpu_power=avg_power,
        power_per_star=power_per_star,
        power_per_interaction=power_per_interaction,
        rotation_curve_slope=slope,
        crashed=crashed,
        crash_reason=crash_reason
    )


def run_density_sweep(
    star_counts: list[int],
    num_ticks: int = 200,
    device: torch.device = None
) -> dict:
    """
    Run density tests across multiple star counts.
    Compare clean (float32) vs broken (int4) math.
    """
    print("\n" + "="*60)
    print("DENSITY LIMIT TEST")
    print("Does the simulation lag before it crashes?")
    print("="*60)

    results = {"clean": [], "broken": []}

    for num_stars in star_counts:
        print(f"\n{'='*50}")
        print(f"Testing {num_stars} stars...")
        print(f"{'='*50}")

        # Test clean math (float32)
        print(f"  Running CLEAN (float32)...")
        clean_result = run_density_test(
            num_stars=num_stars,
            num_ticks=num_ticks,
            use_quantization=False,
            device=device
        )
        results["clean"].append(clean_result)

        status = "CRASHED" if clean_result.crashed else "OK"
        print(f"    {status}: {clean_result.time_per_tick*1000:.2f}ms/tick, "
              f"Energy: {clean_result.energy_drift_pct:+.2f}%")
        if clean_result.crashed:
            print(f"    Crash: {clean_result.crash_reason}")

        # Test broken math (int4)
        print(f"  Running BROKEN (int4)...")
        broken_result = run_density_test(
            num_stars=num_stars,
            num_ticks=num_ticks,
            use_quantization=True,
            device=device
        )
        results["broken"].append(broken_result)

        status = "CRASHED" if broken_result.crashed else "OK"
        print(f"    {status}: {broken_result.time_per_tick*1000:.2f}ms/tick, "
              f"Energy: {broken_result.energy_drift_pct:+.2f}%")
        if broken_result.crashed:
            print(f"    Crash: {broken_result.crash_reason}")

        # Compare
        if not clean_result.crashed and not broken_result.crashed:
            time_ratio = broken_result.time_per_tick / clean_result.time_per_tick
            energy_diff = broken_result.energy_drift_pct - clean_result.energy_drift_pct
            print(f"  COMPARISON:")
            print(f"    Time ratio (broken/clean): {time_ratio:.2f}x")
            print(f"    Energy diff (broken-clean): {energy_diff:+.2f}%")

    return results


def analyze_scaling(results: dict) -> dict:
    """
    Analyze how metrics scale with star count.
    Looking for non-linear scaling that indicates hitting limits.
    """
    analysis = {}

    for mode in ["clean", "broken"]:
        mode_results = [r for r in results[mode] if not r.crashed]

        if len(mode_results) < 3:
            continue

        stars = np.array([r.num_stars for r in mode_results])
        times = np.array([r.time_per_tick for r in mode_results])
        energies = np.array([r.energy_drift_pct for r in mode_results])

        # Fit power law: time = a * N^b
        # If b > 2, we're seeing worse-than-expected scaling
        log_stars = np.log(stars)
        log_times = np.log(times + 1e-10)

        if len(log_stars) >= 2:
            coeffs = np.polyfit(log_stars, log_times, 1)
            scaling_exponent = coeffs[0]
        else:
            scaling_exponent = 0

        # Expected is O(N²), so exponent should be ~2
        # If > 2.5, we're hitting limits
        scaling_anomaly = scaling_exponent - 2.0

        # Energy scaling
        if len(stars) >= 2:
            energy_coeffs = np.polyfit(stars, energies, 1)
            energy_per_star = energy_coeffs[0]
        else:
            energy_per_star = 0

        analysis[mode] = {
            "scaling_exponent": scaling_exponent,
            "scaling_anomaly": scaling_anomaly,
            "energy_per_star": energy_per_star,
            "max_stable_stars": stars.max() if len(stars) > 0 else 0
        }

    return analysis


def plot_density_results(results: dict, analysis: dict, save_dir: str):
    """Plot density test results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data
    clean = [r for r in results["clean"] if not r.crashed]
    broken = [r for r in results["broken"] if not r.crashed]

    # 1. Time per tick vs star count
    ax1 = axes[0, 0]
    if clean:
        stars_c = [r.num_stars for r in clean]
        times_c = [r.time_per_tick * 1000 for r in clean]
        ax1.loglog(stars_c, times_c, 'go-', label='Clean (float32)', linewidth=2, markersize=8)

    if broken:
        stars_b = [r.num_stars for r in broken]
        times_b = [r.time_per_tick * 1000 for r in broken]
        ax1.loglog(stars_b, times_b, 'ro-', label='Broken (int4)', linewidth=2, markersize=8)

    # Add O(N²) reference line
    if clean:
        n_ref = np.array(stars_c)
        t_ref = times_c[0] * (n_ref / n_ref[0]) ** 2
        ax1.loglog(n_ref, t_ref, 'k--', alpha=0.5, label='O(N²) expected')

    ax1.set_xlabel("Number of Stars")
    ax1.set_ylabel("Time per Tick (ms)")
    ax1.set_title("Computation Scaling")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Energy drift vs star count
    ax2 = axes[0, 1]
    if clean:
        stars_c = [r.num_stars for r in clean]
        energy_c = [r.energy_drift_pct for r in clean]
        ax2.semilogx(stars_c, energy_c, 'go-', label='Clean (float32)', linewidth=2, markersize=8)

    if broken:
        stars_b = [r.num_stars for r in broken]
        energy_b = [r.energy_drift_pct for r in broken]
        ax2.semilogx(stars_b, energy_b, 'ro-', label='Broken (int4)', linewidth=2, markersize=8)

    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Number of Stars")
    ax2.set_ylabel("Energy Drift (%)")
    ax2.set_title("Energy Injection vs Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Power per interaction
    ax3 = axes[1, 0]
    if clean and any(r.avg_gpu_power > 0 for r in clean):
        stars_c = [r.num_stars for r in clean if r.avg_gpu_power > 0]
        ppi_c = [r.power_per_interaction * 1e6 for r in clean if r.avg_gpu_power > 0]
        ax3.semilogx(stars_c, ppi_c, 'go-', label='Clean', linewidth=2, markersize=8)

    if broken and any(r.avg_gpu_power > 0 for r in broken):
        stars_b = [r.num_stars for r in broken if r.avg_gpu_power > 0]
        ppi_b = [r.power_per_interaction * 1e6 for r in broken if r.avg_gpu_power > 0]
        ax3.semilogx(stars_b, ppi_b, 'ro-', label='Broken', linewidth=2, markersize=8)

    ax3.set_xlabel("Number of Stars")
    ax3.set_ylabel("Power per Interaction (µW)")
    ax3.set_title("Power Efficiency (should be flat if O(N²))")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary = "DENSITY LIMIT ANALYSIS\n"
    summary += "="*40 + "\n\n"

    if analysis:
        for mode, data in analysis.items():
            summary += f"{mode.upper()}:\n"
            summary += f"  Scaling exponent: {data['scaling_exponent']:.2f}\n"
            summary += f"  (Expected: 2.0, Anomaly: {data['scaling_anomaly']:+.2f})\n"
            summary += f"  Energy per star: {data['energy_per_star']:.4f}%\n"
            summary += f"  Max stable stars: {data['max_stable_stars']}\n\n"

        # Interpretation
        if "broken" in analysis and "clean" in analysis:
            broken_anomaly = analysis["broken"]["scaling_anomaly"]
            clean_anomaly = analysis["clean"]["scaling_anomaly"]

            if broken_anomaly > clean_anomaly + 0.2:
                summary += "! BROKEN math scales WORSE than clean\n"
                summary += "  Quantization adds computational overhead\n"
                summary += "  Could indicate 'error correction' work\n"
            elif clean_anomaly > broken_anomaly + 0.2:
                summary += "! CLEAN math scales WORSE than broken\n"
                summary += "  High precision has more overhead\n"
            else:
                summary += "Scaling similar for both modes.\n"

    ax4.text(0.1, 0.9, summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle("Density Limit Test: Where Does the Simulation Lag?",
                 fontsize=16, y=1.02)
    plt.tight_layout()

    save_path = Path(save_dir) / "density_limit.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {save_path}")

    return fig


def print_summary(results: dict, analysis: dict):
    """Print test summary."""
    print("\n" + "="*60)
    print("DENSITY LIMIT TEST SUMMARY")
    print("="*60)

    print("\nRESULTS TABLE:")
    print("-" * 80)
    print(f"{'Stars':<10} {'Mode':<10} {'ms/tick':<12} {'Energy %':<12} {'Status':<15}")
    print("-" * 80)

    for mode in ["clean", "broken"]:
        for r in results[mode]:
            status = f"CRASH: {r.crash_reason[:20]}" if r.crashed else "OK"
            energy = f"{r.energy_drift_pct:+.4f}" if not r.crashed else "N/A"
            time_ms = f"{r.time_per_tick*1000:.2f}" if not r.crashed else "N/A"
            print(f"{r.num_stars:<10} {mode:<10} {time_ms:<12} {energy:<12} {status:<15}")

    print("\n" + "-"*40)
    print("SCALING ANALYSIS:")

    if analysis:
        for mode, data in analysis.items():
            exp = data['scaling_exponent']
            print(f"\n  {mode.upper()}: O(N^{exp:.2f})")
            if exp > 2.3:
                print(f"    WARNING: Scaling is worse than expected O(N²)")
                print(f"    This suggests we're hitting simulation limits!")
            elif exp < 1.8:
                print(f"    INTERESTING: Scaling is better than O(N²)")
                print(f"    GPU parallelization is very efficient.")
            else:
                print(f"    Normal O(N²) scaling.")

    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("""
If the simulation is hitting its limits:
  - Scaling exponent should increase (> 2.0)
  - Energy drift should scale non-linearly with density
  - "Broken" math might show different scaling than "clean"

If dark matter is simulation overhead:
  - The "unexplained power" should scale with density
  - Higher density = more "ghost force" = more dark matter effect
""")


def main():
    parser = argparse.ArgumentParser(description="Density limit test")
    parser.add_argument("--max-stars", type=int, default=8000,
                        help="Maximum star count to test")
    parser.add_argument("--ticks", type=int, default=150,
                        help="Ticks per test")
    parser.add_argument("--output", type=str, default="output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if HAS_POWER:
        print("GPU power monitoring: ENABLED")
    else:
        print("GPU power monitoring: DISABLED (install pynvml)")

    # Star counts to test (exponential spacing)
    star_counts = [100, 250, 500, 1000, 2000, 4000]

    # Add higher counts up to max
    current = 4000
    while current * 2 <= args.max_stars:
        current *= 2
        star_counts.append(current)

    if args.max_stars not in star_counts and args.max_stars > star_counts[-1]:
        star_counts.append(args.max_stars)

    print(f"\nTesting star counts: {star_counts}")

    # Run tests
    results = run_density_sweep(
        star_counts=star_counts,
        num_ticks=args.ticks,
        device=device
    )

    # Analyze
    analysis = analyze_scaling(results)

    # Print summary
    print_summary(results, analysis)

    # Plot
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    plot_density_results(results, analysis, args.output)

    # Cleanup
    if HAS_POWER:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

    plt.show()


if __name__ == "__main__":
    main()
