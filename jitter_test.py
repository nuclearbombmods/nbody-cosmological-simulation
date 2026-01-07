"""
Jitter / Frame-Rate Stress Test

Tests the hypothesis that "Dark Matter = Simulation Lag"

The idea: If the universe is a simulation with a finite frame rate,
objects moving near the "tick limit" would experience jitter -
position uncertainty between frames that creates kinetic energy.

We test this by:
1. Creating nested/recursive structures (stress the renderer)
2. Running at extreme tick rates (small dt = high FPS demand)
3. Measuring "jitter" - spurious velocity/energy injection
4. Checking if jitter creates dark-matter-like binding

Usage:
    python jitter_test.py
    python jitter_test.py --nested-levels 5 --max-velocity 0.9
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

from galaxy import create_disk_galaxy
from simulation import GalaxySimulation
from quantization import PrecisionMode
from metrics import compute_rotation_curve


@dataclass
class JitterResult:
    """Results from a jitter test."""
    dt: float
    ticks_per_unit_time: int
    position_jitter: float      # RMS position variance between frames
    velocity_jitter: float      # RMS velocity variance
    energy_injection: float     # Energy gained from jitter
    final_energy_drift: float   # Total energy change


def create_nested_galaxy(
    num_stars: int = 2000,
    nested_levels: int = 3,
    device: torch.device = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a "recursive" galaxy structure - galaxies within galaxies.

    This stresses the simulation by having structure at multiple scales,
    like the Inception/3008 concept of nested realities.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_positions = []
    all_velocities = []
    all_masses = []

    stars_per_level = num_stars // nested_levels

    for level in range(nested_levels):
        # Each level is smaller and denser
        scale = 10.0 / (2 ** level)  # 10, 5, 2.5, 1.25, ...

        pos, vel, mass = create_disk_galaxy(
            num_stars=stars_per_level,
            galaxy_radius=scale,
            device=device
        )

        # Inner levels have higher mass density (more "important")
        mass = mass * (2 ** level)

        all_positions.append(pos)
        all_velocities.append(vel)
        all_masses.append(mass)

    positions = torch.cat(all_positions, dim=0)
    velocities = torch.cat(all_velocities, dim=0)
    masses = torch.cat(all_masses, dim=0)

    return positions, velocities, masses


def create_high_velocity_galaxy(
    num_stars: int = 2000,
    max_velocity_fraction: float = 0.5,
    device: torch.device = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create galaxy with stars moving at high fractions of "c" (our sim's speed limit).

    In a simulation, objects near the tick limit would jitter most.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pos, vel, mass = create_disk_galaxy(
        num_stars=num_stars,
        galaxy_radius=10.0,
        device=device
    )

    # Boost velocities to be a fraction of "c" (max stable velocity)
    # In our sim, we'll define c = 10 units/tick as the "speed limit"
    c_sim = 10.0
    target_speed = c_sim * max_velocity_fraction

    current_speeds = torch.sqrt((vel ** 2).sum(dim=-1, keepdim=True))
    current_speeds = current_speeds.clamp(min=0.01)

    # Scale velocities to target speed (but keep direction)
    vel = vel / current_speeds * target_speed

    return pos, vel, mass


def measure_jitter(
    positions_history: list[torch.Tensor],
    velocities_history: list[torch.Tensor],
    dt: float
) -> dict:
    """
    Measure jitter from position/velocity history.

    Jitter = unexpected high-frequency oscillations that shouldn't exist
    in smooth physics.
    """
    if len(positions_history) < 3:
        return {"position_jitter": 0, "velocity_jitter": 0}

    # Convert to numpy for analysis
    pos_stack = torch.stack(positions_history).cpu().numpy()  # (T, N, 2)
    vel_stack = torch.stack(velocities_history).cpu().numpy()

    # Compute second derivative of position (acceleration jitter)
    # Smooth physics: d²x/dt² should be smooth
    # Jittery physics: d²x/dt² oscillates wildly

    pos_accel = np.diff(pos_stack, n=2, axis=0) / (dt ** 2)

    # Jitter = RMS of acceleration variations
    accel_mean = pos_accel.mean(axis=0)
    accel_var = ((pos_accel - accel_mean) ** 2).mean()
    position_jitter = np.sqrt(accel_var)

    # Velocity jitter: unexpected velocity changes
    vel_diff = np.diff(vel_stack, axis=0) / dt
    vel_var = vel_diff.var()
    velocity_jitter = np.sqrt(vel_var)

    return {
        "position_jitter": float(position_jitter),
        "velocity_jitter": float(velocity_jitter)
    }


def run_framerate_stress_test(
    dt_values: list[float],
    total_time: float = 5.0,
    num_stars: int = 1500,
    nested_levels: int = 3,
    device: torch.device = None
) -> list[JitterResult]:
    """
    Test different "frame rates" (dt values) for jitter.

    Smaller dt = higher frame rate = more stress on the simulation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*60)
    print("FRAME-RATE STRESS TEST")
    print("Testing if high tick rates cause energy-injecting jitter")
    print("="*60)

    # Create the nested galaxy (stresses multi-scale rendering)
    print(f"\nCreating nested galaxy ({nested_levels} levels, {num_stars} stars)...")
    positions, velocities, masses = create_nested_galaxy(
        num_stars=num_stars,
        nested_levels=nested_levels,
        device=device
    )
    positions, velocities, masses = positions.float(), velocities.float(), masses.float()

    results = []

    for dt in dt_values:
        num_ticks = int(total_time / dt)
        sample_interval = max(1, num_ticks // 100)  # Sample 100 points

        print(f"\n  dt={dt:.4f} ({num_ticks} ticks for {total_time}s simulation time)...")

        sim = GalaxySimulation(
            positions.clone(),
            velocities.clone(),
            masses.clone(),
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001,
            dt=dt,
            softening=0.1,
            device=device
        )

        initial_energy = sim.get_total_energy()

        # Track history for jitter analysis
        pos_history = [sim.positions.clone()]
        vel_history = [sim.velocities.clone()]
        energy_history = [initial_energy]

        for tick in range(num_ticks):
            sim.step()

            if tick % sample_interval == 0:
                pos_history.append(sim.positions.clone())
                vel_history.append(sim.velocities.clone())
                energy_history.append(sim.get_total_energy())

        final_energy = sim.get_total_energy()

        # Measure jitter
        jitter = measure_jitter(pos_history, vel_history, dt * sample_interval)

        # Energy injection from jitter
        energy_changes = np.diff(energy_history)
        energy_injection = np.sum(np.maximum(energy_changes, 0))  # Only positive changes

        energy_drift = (final_energy - initial_energy) / abs(initial_energy) * 100

        result = JitterResult(
            dt=dt,
            ticks_per_unit_time=int(1.0 / dt),
            position_jitter=jitter["position_jitter"],
            velocity_jitter=jitter["velocity_jitter"],
            energy_injection=energy_injection,
            final_energy_drift=energy_drift
        )
        results.append(result)

        print(f"    Position jitter: {result.position_jitter:.4f}")
        print(f"    Velocity jitter: {result.velocity_jitter:.4f}")
        print(f"    Energy drift: {result.final_energy_drift:+.4f}%")

    return results


def run_velocity_stress_test(
    velocity_fractions: list[float],
    num_stars: int = 1500,
    num_ticks: int = 500,
    device: torch.device = None
) -> list[dict]:
    """
    Test if high velocities (near "c") cause more jitter.

    Objects near the simulation's speed limit should jitter more
    if there's a frame-rate effect.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*60)
    print("VELOCITY STRESS TEST")
    print("Testing if high velocities cause more jitter")
    print("="*60)

    results = []

    for v_frac in velocity_fractions:
        print(f"\n  Testing velocity = {v_frac*100:.0f}% of 'c'...")

        positions, velocities, masses = create_high_velocity_galaxy(
            num_stars=num_stars,
            max_velocity_fraction=v_frac,
            device=device
        )
        positions, velocities, masses = positions.float(), velocities.float(), masses.float()

        sim = GalaxySimulation(
            positions, velocities, masses,
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001,
            dt=0.01,
            softening=0.1,
            device=device
        )

        initial_energy = sim.get_total_energy()
        initial_ke = 0.5 * (masses * (velocities ** 2).sum(dim=-1)).sum().item()

        pos_history = []
        vel_history = []

        for tick in range(num_ticks):
            sim.step()
            if tick % 10 == 0:
                pos_history.append(sim.positions.clone())
                vel_history.append(sim.velocities.clone())

        final_energy = sim.get_total_energy()
        jitter = measure_jitter(pos_history, vel_history, 0.01 * 10)

        results.append({
            "velocity_fraction": v_frac,
            "initial_ke": initial_ke,
            "position_jitter": jitter["position_jitter"],
            "velocity_jitter": jitter["velocity_jitter"],
            "energy_drift": (final_energy - initial_energy) / abs(initial_energy) * 100
        })

        print(f"    Jitter: {jitter['velocity_jitter']:.4f}")
        print(f"    Energy drift: {results[-1]['energy_drift']:+.4f}%")

    return results


def plot_jitter_results(framerate_results, velocity_results, save_dir):
    """Plot jitter test results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Jitter vs Frame Rate (dt)
    ax1 = axes[0, 0]
    if framerate_results:
        dts = [r.dt for r in framerate_results]
        fps = [1/r.dt for r in framerate_results]
        pos_jitter = [r.position_jitter for r in framerate_results]
        vel_jitter = [r.velocity_jitter for r in framerate_results]

        ax1.loglog(fps, pos_jitter, 'o-', label='Position jitter', linewidth=2)
        ax1.loglog(fps, vel_jitter, 's-', label='Velocity jitter', linewidth=2)
        ax1.set_xlabel("Ticks per unit time (FPS)", fontsize=12)
        ax1.set_ylabel("Jitter (RMS)", fontsize=12)
        ax1.set_title("Jitter vs Frame Rate", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. Energy Drift vs Frame Rate
    ax2 = axes[0, 1]
    if framerate_results:
        dts = [r.dt for r in framerate_results]
        fps = [1/r.dt for r in framerate_results]
        drift = [r.final_energy_drift for r in framerate_results]

        ax2.semilogx(fps, drift, 'o-', color='red', linewidth=2)
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.set_xlabel("Ticks per unit time (FPS)", fontsize=12)
        ax2.set_ylabel("Energy Drift (%)", fontsize=12)
        ax2.set_title("Energy Injection vs Frame Rate", fontsize=14)
        ax2.grid(True, alpha=0.3)

    # 3. Jitter vs Velocity
    ax3 = axes[1, 0]
    if velocity_results:
        v_fracs = [r["velocity_fraction"] for r in velocity_results]
        vel_jitter = [r["velocity_jitter"] for r in velocity_results]

        ax3.plot(v_fracs, vel_jitter, 'o-', color='purple', linewidth=2)
        ax3.set_xlabel("Velocity (fraction of 'c')", fontsize=12)
        ax3.set_ylabel("Velocity Jitter", fontsize=12)
        ax3.set_title("Jitter vs Object Speed", fontsize=14)
        ax3.grid(True, alpha=0.3)

    # 4. Summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary = "JITTER / FRAME-RATE TEST SUMMARY\n"
    summary += "="*40 + "\n\n"

    if framerate_results:
        # Check if jitter increases with frame rate
        jitters = [r.velocity_jitter for r in framerate_results]
        fps_list = [1/r.dt for r in framerate_results]

        # Correlation
        corr = np.corrcoef(fps_list, jitters)[0, 1]

        summary += f"Frame rate correlation: {corr:+.3f}\n"

        if corr > 0.5:
            summary += "Higher FPS = MORE jitter\n"
            summary += "(Supports 'lag causes energy' hypothesis)\n\n"
        elif corr < -0.5:
            summary += "Higher FPS = LESS jitter\n"
            summary += "(Opposes 'lag causes energy' hypothesis)\n\n"
        else:
            summary += "No clear FPS-jitter relationship\n\n"

    if velocity_results:
        # Check if high velocity = more jitter
        v_fracs = [r["velocity_fraction"] for r in velocity_results]
        v_jitters = [r["velocity_jitter"] for r in velocity_results]

        corr = np.corrcoef(v_fracs, v_jitters)[0, 1]
        summary += f"Velocity correlation: {corr:+.3f}\n"

        if corr > 0.5:
            summary += "High velocity = MORE jitter\n"
            summary += "(Near-c objects would jitter most)"
        elif corr < -0.5:
            summary += "High velocity = LESS jitter\n"
            summary += "(Unexpected result)"
        else:
            summary += "No clear velocity-jitter relationship"

    ax4.text(0.1, 0.9, summary, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle("Jitter Hypothesis: Does Simulation Lag Create Dark Matter?",
                 fontsize=16, y=1.02)
    plt.tight_layout()

    save_path = Path(save_dir) / "jitter_test.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {save_path}")

    return fig


def print_analysis(framerate_results, velocity_results):
    """Print analysis of jitter tests."""
    print("\n" + "="*60)
    print("JITTER HYPOTHESIS ANALYSIS")
    print("="*60)

    print("""
The "3008 Frame-Rate" Hypothesis:
- The universe has a finite "tick rate" (Planck time?)
- Objects moving fast relative to this rate would "jitter"
- Jitter = uncertainty in position between frames
- This jitter could inject energy, mimicking dark matter

Test Results:
""")

    if framerate_results:
        print("FRAME RATE TEST:")
        print("-" * 50)
        print(f"{'dt':<12} {'FPS':<10} {'Pos Jitter':<12} {'Vel Jitter':<12} {'Energy %':<10}")
        print("-" * 50)
        for r in framerate_results:
            print(f"{r.dt:<12.4f} {1/r.dt:<10.0f} {r.position_jitter:<12.4f} {r.velocity_jitter:<12.4f} {r.final_energy_drift:<+10.4f}")

    if velocity_results:
        print("\nVELOCITY TEST:")
        print("-" * 50)
        print(f"{'V/c':<10} {'Vel Jitter':<12} {'Energy %':<10}")
        print("-" * 50)
        for r in velocity_results:
            print(f"{r['velocity_fraction']:<10.2f} {r['velocity_jitter']:<12.4f} {r['energy_drift']:<+10.4f}")

    # Verdict
    print("\n" + "-"*40)
    print("VERDICT:")

    if framerate_results:
        # Check for increasing jitter with FPS
        fps_list = [1/r.dt for r in framerate_results]
        jitters = [r.velocity_jitter for r in framerate_results]

        if jitters[-1] > jitters[0] * 1.5:
            print("  + Jitter INCREASES with frame rate")
            print("    This supports the 'simulation lag' hypothesis!")
        else:
            print("  - Jitter does NOT increase with frame rate")
            print("    This weakens the 'simulation lag' hypothesis.")

    if velocity_results:
        v_fracs = [r["velocity_fraction"] for r in velocity_results]
        v_jitters = [r["velocity_jitter"] for r in velocity_results]

        if v_jitters[-1] > v_jitters[0] * 1.5:
            print("  + Jitter INCREASES with velocity")
            print("    Fast objects jitter more (like near light speed)!")
        else:
            print("  - Jitter does NOT increase with velocity")
            print("    Speed doesn't cause more jitter.")


def main():
    parser = argparse.ArgumentParser(description="Jitter / Frame-Rate Stress Test")
    parser.add_argument("--stars", type=int, default=1500)
    parser.add_argument("--nested-levels", type=int, default=3,
                        help="Levels of nested galaxy structure")
    parser.add_argument("--output", type=str, default="output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Test 1: Frame rate stress test
    # Different dt values = different "frame rates"
    dt_values = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

    framerate_results = run_framerate_stress_test(
        dt_values=dt_values,
        total_time=3.0,
        num_stars=args.stars,
        nested_levels=args.nested_levels,
        device=device
    )

    # Test 2: Velocity stress test
    # Objects at different speeds relative to "c"
    velocity_fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    velocity_results = run_velocity_stress_test(
        velocity_fractions=velocity_fractions,
        num_stars=args.stars,
        num_ticks=500,
        device=device
    )

    # Analysis
    print_analysis(framerate_results, velocity_results)

    # Plot
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    plot_jitter_results(framerate_results, velocity_results, args.output)
    plt.show()


if __name__ == "__main__":
    main()
