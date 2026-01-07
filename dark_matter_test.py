"""
Dark Matter Ratio Test

Compare rotation curves with different amounts of dark matter.
This shows what REAL dark matter does to galaxy dynamics.

Usage:
    python dark_matter_test.py
    python dark_matter_test.py --ratios 0,2,5,10
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from galaxy import create_disk_galaxy, create_galaxy_with_halo
from simulation import GalaxySimulation
from quantization import PrecisionMode
from metrics import compute_rotation_curve


def run_dm_comparison(
    dm_ratios: list[float],
    num_stars: int = 3000,
    num_ticks: int = 1000,
    device: torch.device = None
) -> dict:
    """
    Run simulations with different dark matter ratios.

    Uses ANALYTICAL NFW halo (no DM particles = fast N-body).
    The DM affects initial velocities but doesn't add particles.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    for ratio in dm_ratios:
        print(f"\n{'='*50}")
        print(f"Dark Matter Ratio: {ratio}x visible mass")
        print(f"{'='*50}")

        if ratio == 0:
            positions, velocities, masses = create_disk_galaxy(
                num_stars=num_stars,
                galaxy_radius=10.0,
                device=device
            )
        else:
            positions, velocities, masses = create_galaxy_with_halo(
                num_stars=num_stars,
                galaxy_radius=10.0,
                halo_radius=30.0,
                dm_mass_ratio=ratio,
                device=device
            )

        print(f"  Stars: {num_stars}")
        print(f"  DM ratio: {ratio}x (analytical NFW halo)")
        print(f"  Visible mass: {masses.sum().item():.0f}")

        sim = GalaxySimulation(
            positions.float(),
            velocities.float(),
            masses.float(),
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001,
            dt=0.01,
            softening=0.1,
            device=device
        )

        initial_curve = compute_rotation_curve(positions, velocities, num_bins=15)

        print(f"  Running {num_ticks} ticks...")

        for tick in range(0, num_ticks, 100):
            for _ in range(100):
                sim.step()

            if (tick + 100) % 500 == 0:
                print(f"    Tick {tick + 100}")

        final_curve = compute_rotation_curve(sim.positions, sim.velocities, num_bins=15)

        results[ratio] = {
            "initial_curve": initial_curve,
            "final_curve": final_curve,
            "num_stars": num_stars,
            "dm_ratio": ratio,
            "total_mass": masses.sum().item(),
        }

    return results


def plot_dm_comparison(results: dict, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(results)))

    ax1 = axes[0]
    for (ratio, data), color in zip(results.items(), colors):
        curve = data["initial_curve"]
        valid = ~np.isnan(curve["velocities"])
        label = f"DM ratio = {ratio}x" if ratio > 0 else "No Dark Matter"
        ax1.plot(curve["radii"][valid], curve["velocities"][valid],
                 'o-', color=color, label=label, linewidth=2, markersize=4)

    ax1.set_xlabel("Radius", fontsize=12)
    ax1.set_ylabel("Circular Velocity", fontsize=12)
    ax1.set_title("Initial Rotation Curves", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    r_ref = np.linspace(1, 15, 50)
    v_kep = 1.5 / np.sqrt(r_ref)
    ax1.plot(r_ref, v_kep, '--', color='gray', alpha=0.5, label='Keplerian')

    ax2 = axes[1]
    for (ratio, data), color in zip(results.items(), colors):
        curve = data["final_curve"]
        valid = ~np.isnan(curve["velocities"])
        label = f"DM ratio = {ratio}x" if ratio > 0 else "No Dark Matter"
        ax2.plot(curve["radii"][valid], curve["velocities"][valid],
                 'o-', color=color, label=label, linewidth=2, markersize=4)

    ax2.set_xlabel("Radius", fontsize=12)
    ax2.set_ylabel("Circular Velocity", fontsize=12)
    ax2.set_title("Final Rotation Curves (after simulation)", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.plot(r_ref, v_kep, '--', color='gray', alpha=0.5)

    plt.suptitle("Dark Matter Effect on Rotation Curves", fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def print_dm_analysis(results: dict):
    print(f"\n{'='*60}")
    print("DARK MATTER ANALYSIS")
    print(f"{'='*60}")
    print()
    print("The 'Dark Matter Signature' is a FLAT rotation curve at outer radii.")
    print("Without DM, velocity should decline as 1/sqrt(r) (Keplerian).")
    print("With DM, velocity stays roughly constant (flat curve).")
    print()

    for ratio, data in results.items():
        curve = data["final_curve"]
        valid = ~np.isnan(curve["velocities"])
        radii = curve["radii"][valid]
        vels = curve["velocities"][valid]

        if len(vels) < 4:
            continue

        mid_idx = len(vels) // 2
        outer_r = radii[mid_idx:]
        outer_v = vels[mid_idx:]

        if len(outer_r) > 2:
            slope = np.polyfit(outer_r, outer_v, 1)[0]
            mean_v = outer_v.mean()
            dm_label = f"DM = {ratio}x" if ratio > 0 else "No DM"
            print(f"{dm_label:12} | Outer slope: {slope:+.4f} | Mean outer v: {mean_v:.3f}")

    print()
    print("Interpretation:")
    print("  - Negative slope = velocity declining (Keplerian, no DM needed)")
    print("  - Near-zero slope = flat curve (Dark Matter signature!)")
    print("  - Positive slope = velocity increasing (strong DM dominance)")


def main():
    parser = argparse.ArgumentParser(description="Test dark matter effects on rotation curves")
    parser.add_argument("--ratios", "-r", type=str, default="0,2,5,10",
                        help="Comma-separated DM/visible mass ratios")
    parser.add_argument("--stars", type=int, default=2000, help="Number of visible stars")
    parser.add_argument("--ticks", type=int, default=800, help="Simulation ticks")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    ratios = [float(r) for r in args.ratios.split(",")]
    print(f"Testing DM ratios: {ratios}")

    results = run_dm_comparison(
        dm_ratios=ratios,
        num_stars=args.stars,
        num_ticks=args.ticks,
        device=device
    )

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    plot_dm_comparison(results, save_path=str(output_dir / "dark_matter_comparison.png"))
    print_dm_analysis(results)

    plt.show()


if __name__ == "__main__":
    main()
