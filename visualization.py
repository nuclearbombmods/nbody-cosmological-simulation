"""
Visualization module.
Creates comparison plots between different precision modes.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

from metrics import SimulationMetrics


def plot_galaxy_comparison(
    results: dict,
    save_path: str = None,
    title: str = "Galaxy Comparison: Precision Effects"
):
    """
    Create side-by-side galaxy snapshots for different precision modes.

    Args:
        results: Dictionary from run_comparison with mode -> result data
        save_path: Optional path to save figure
        title: Plot title
    """
    modes = list(results.keys())
    n_modes = len(modes)

    fig, axes = plt.subplots(1, n_modes, figsize=(5 * n_modes, 5))
    if n_modes == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        state = results[mode]["final_state"]
        pos = state["positions"].cpu().numpy()

        ax.scatter(pos[:, 0], pos[:, 1], s=1, alpha=0.5, c='white')
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        ax.set_title(f"{mode}", fontsize=12, color='white')
        ax.set_xlabel("X", color='white')
        ax.set_ylabel("Y", color='white')
        ax.tick_params(colors='white')

        # Set consistent limits
        max_extent = max(np.abs(pos).max() * 1.1, 15)
        ax.set_xlim(-max_extent, max_extent)
        ax.set_ylim(-max_extent, max_extent)

    fig.patch.set_facecolor('#1a1a2e')
    plt.suptitle(title, fontsize=14, color='white', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a2e', bbox_inches='tight')
        print(f"Saved galaxy comparison to {save_path}")

    return fig


def plot_rotation_curves(
    metrics_dict: dict[str, SimulationMetrics],
    save_path: str = None,
    title: str = "Rotation Curves: The Dark Matter Signature"
):
    """
    Plot rotation curves for different precision modes.

    Flat curves at outer radii = "dark matter" effect.

    Args:
        metrics_dict: Dictionary mapping mode name -> SimulationMetrics
        save_path: Optional path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(metrics_dict)))

    for (mode, metrics), color in zip(metrics_dict.items(), colors):
        if metrics.rotation_curves:
            # Use final rotation curve
            curve = metrics.rotation_curves[-1]
            radii = curve["radii"]
            velocities = curve["velocities"]

            # Filter NaN
            valid = ~np.isnan(velocities)

            ax.plot(
                radii[valid],
                velocities[valid],
                'o-',
                color=color,
                label=mode,
                markersize=4,
                linewidth=2
            )

    # Add Keplerian reference (v ~ 1/sqrt(r))
    r_ref = np.linspace(1, 15, 50)
    v_keplerian = 1.5 / np.sqrt(r_ref)  # Normalized
    ax.plot(r_ref, v_keplerian, '--', color='red', alpha=0.5,
            label='Keplerian (no dark matter)', linewidth=1.5)

    ax.set_xlabel("Radius", fontsize=12)
    ax.set_ylabel("Circular Velocity", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved rotation curves to {save_path}")

    return fig


def plot_energy_evolution(
    metrics_dict: dict[str, SimulationMetrics],
    save_path: str = None,
    title: str = "Energy Evolution: Rounding Error Injection"
):
    """
    Plot total energy over time for different precision modes.

    Energy increasing = rounding errors "injecting" energy (ghost mass effect).

    Args:
        metrics_dict: Dictionary mapping mode name -> SimulationMetrics
        save_path: Optional path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(metrics_dict)))

    # Left plot: Absolute energy
    ax1 = axes[0]
    for (mode, metrics), color in zip(metrics_dict.items(), colors):
        if metrics.total_energy:
            ax1.plot(
                metrics.ticks,
                metrics.total_energy,
                '-',
                color=color,
                label=mode,
                linewidth=2
            )

    ax1.set_xlabel("Simulation Tick", fontsize=12)
    ax1.set_ylabel("Total Energy", fontsize=12)
    ax1.set_title("Total Energy Over Time", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Energy relative to initial
    ax2 = axes[1]
    for (mode, metrics), color in zip(metrics_dict.items(), colors):
        if metrics.total_energy:
            initial = metrics.total_energy[0]
            if abs(initial) > 1e-10:
                relative = [(e - initial) / abs(initial) * 100 for e in metrics.total_energy]
                ax2.plot(
                    metrics.ticks,
                    relative,
                    '-',
                    color=color,
                    label=mode,
                    linewidth=2
                )

    ax2.set_xlabel("Simulation Tick", fontsize=12)
    ax2.set_ylabel("Energy Change (%)", fontsize=12)
    ax2.set_title("Energy Drift (% of initial)", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved energy evolution to {save_path}")

    return fig


def plot_galaxy_radius_evolution(
    metrics_dict: dict[str, SimulationMetrics],
    save_path: str = None,
    title: str = "Galaxy Radius: Does Quantization Keep Stars Bound?"
):
    """
    Plot galaxy radius (90th percentile) over time.

    If quantized version maintains smaller radius, stars are staying
    bound due to "ghost mass" from rounding.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(metrics_dict)))

    for (mode, metrics), color in zip(metrics_dict.items(), colors):
        if metrics.galaxy_radius_90:
            ax.plot(
                metrics.ticks,
                metrics.galaxy_radius_90,
                '-',
                color=color,
                label=mode,
                linewidth=2
            )

    ax.set_xlabel("Simulation Tick", fontsize=12)
    ax.set_ylabel("Galaxy Radius (90th percentile)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved radius evolution to {save_path}")

    return fig


def plot_full_comparison(
    results: dict,
    metrics_dict: dict[str, SimulationMetrics],
    save_dir: str = "output",
    show: bool = True
):
    """
    Generate all comparison plots and save them.

    Args:
        results: Results from run_comparison
        metrics_dict: Metrics collected during simulation
        save_dir: Directory to save plots
        show: Whether to display plots
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # Generate all plots
    fig1 = plot_galaxy_comparison(
        results,
        save_path=str(save_path / "galaxy_comparison.png")
    )

    fig2 = plot_rotation_curves(
        metrics_dict,
        save_path=str(save_path / "rotation_curves.png")
    )

    fig3 = plot_energy_evolution(
        metrics_dict,
        save_path=str(save_path / "energy_evolution.png")
    )

    fig4 = plot_galaxy_radius_evolution(
        metrics_dict,
        save_path=str(save_path / "radius_evolution.png")
    )

    if show:
        plt.show()

    return [fig1, fig2, fig3, fig4]


def print_summary(metrics_dict: dict[str, SimulationMetrics]):
    """Print text summary of results."""
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 60)

    for mode, metrics in metrics_dict.items():
        print(f"\n{mode}:")
        print("-" * 40)

        if metrics.total_energy:
            initial_e = metrics.total_energy[0]
            final_e = metrics.total_energy[-1]
            drift = (final_e - initial_e) / abs(initial_e) * 100 if abs(initial_e) > 1e-10 else 0
            print(f"  Energy drift: {drift:+.2f}%")

        if metrics.galaxy_radius_90:
            initial_r = metrics.galaxy_radius_90[0]
            final_r = metrics.galaxy_radius_90[-1]
            change = (final_r - initial_r) / initial_r * 100 if initial_r > 0 else 0
            print(f"  Radius change: {change:+.2f}%")
            print(f"  Final radius: {final_r:.2f}")

        if metrics.bound_fraction:
            print(f"  Final bound fraction: {metrics.bound_fraction[-1]:.1%}")

        if metrics.velocity_dispersion:
            initial_d = metrics.velocity_dispersion[0]
            final_d = metrics.velocity_dispersion[-1]
            change = (final_d - initial_d) / initial_d * 100 if initial_d > 0 else 0
            print(f"  Velocity dispersion change: {change:+.2f}%")

    print("\n" + "=" * 60)
