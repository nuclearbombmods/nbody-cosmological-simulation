"""
Crash Point Finder - Where Does Reality Break?

Find the exact thresholds where the simulation can no longer "render" properly.
Like finding the edge of a video game map where you clip through walls.

We look for:
1. NaN explosions - when math produces undefined results
2. Teleportation - objects jumping impossible distances
3. Energy singularities - infinite energy from nowhere
4. Velocity overflow - objects exceeding "c"

The hypothesis: The universe has similar limits. Dark matter might be
the "error correction" that prevents reality from clipping.

Usage:
    python crash_point_test.py
    python crash_point_test.py --mode velocity
    python crash_point_test.py --mode all
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from galaxy import create_disk_galaxy
from simulation import GalaxySimulation
from quantization import PrecisionMode, _grid_quantize_safe


@dataclass
class CrashReport:
    """Report of a crash/instability event."""
    parameter: str
    value: float
    crash_type: str
    tick: int
    details: str
    severity: float  # 0-1, how bad was it


def detect_crash(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    prev_positions: torch.Tensor,
    prev_velocities: torch.Tensor,
    energy: float,
    prev_energy: float,
    dt: float,
    tick: int
) -> Optional[CrashReport]:
    """
    Detect if the simulation has crashed/glitched.
    """
    # Check for NaN
    if torch.isnan(positions).any() or torch.isnan(velocities).any():
        return CrashReport(
            parameter="nan",
            value=0,
            crash_type="NaN_EXPLOSION",
            tick=tick,
            details="Positions or velocities became NaN",
            severity=1.0
        )

    # Check for Inf
    if torch.isinf(positions).any() or torch.isinf(velocities).any():
        return CrashReport(
            parameter="inf",
            value=0,
            crash_type="INFINITY_OVERFLOW",
            tick=tick,
            details="Values exceeded representable range",
            severity=1.0
        )

    # Check for teleportation (object moved impossibly far in one tick)
    if prev_positions is not None:
        displacement = torch.sqrt(((positions - prev_positions) ** 2).sum(dim=-1))
        max_displacement = displacement.max().item()
        expected_max = velocities.abs().max().item() * dt * 10  # 10x buffer

        if max_displacement > expected_max and max_displacement > 1.0:
            return CrashReport(
                parameter="teleport",
                value=max_displacement,
                crash_type="TELEPORTATION",
                tick=tick,
                details=f"Object moved {max_displacement:.2f} in one tick (expected max {expected_max:.2f})",
                severity=min(1.0, max_displacement / 100)
            )

    # Check for velocity overflow (exceeding simulation "c")
    speeds = torch.sqrt((velocities ** 2).sum(dim=-1))
    max_speed = speeds.max().item()
    c_sim = 100.0  # Our simulation's "speed of light"

    if max_speed > c_sim:
        return CrashReport(
            parameter="velocity",
            value=max_speed,
            crash_type="VELOCITY_OVERFLOW",
            tick=tick,
            details=f"Object exceeds c_sim ({max_speed:.2f} > {c_sim})",
            severity=min(1.0, max_speed / (c_sim * 10))
        )

    # Check for energy singularity
    if prev_energy is not None and prev_energy != 0:
        energy_ratio = abs(energy / prev_energy)
        if energy_ratio > 100 or energy_ratio < 0.01:
            return CrashReport(
                parameter="energy",
                value=energy,
                crash_type="ENERGY_SINGULARITY",
                tick=tick,
                details=f"Energy changed by {energy_ratio:.2f}x in one tick",
                severity=min(1.0, abs(np.log10(energy_ratio)) / 5)
            )

    # Check for galaxy explosion (radius suddenly huge)
    radii = torch.sqrt((positions ** 2).sum(dim=-1))
    max_radius = radii.max().item()

    if max_radius > 1000:  # Way beyond normal
        return CrashReport(
            parameter="radius",
            value=max_radius,
            crash_type="GALAXY_EXPLOSION",
            tick=tick,
            details=f"Galaxy radius exploded to {max_radius:.2f}",
            severity=min(1.0, max_radius / 10000)
        )

    return None


def find_velocity_crash_point(
    num_stars: int = 1000,
    device: torch.device = None
) -> dict:
    """
    Find the velocity threshold where simulation breaks.
    """
    print("\n" + "="*60)
    print("VELOCITY CRASH POINT SEARCH")
    print("How fast can objects go before reality breaks?")
    print("="*60)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    velocity_multipliers = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    results = []

    for mult in velocity_multipliers:
        print(f"\n  Testing velocity multiplier: {mult}x...")

        positions, velocities, masses = create_disk_galaxy(
            num_stars=num_stars,
            galaxy_radius=10.0,
            device=device
        )
        velocities = velocities.float() * mult
        positions = positions.float()
        masses = masses.float()

        initial_speed = torch.sqrt((velocities ** 2).sum(dim=-1)).mean().item()
        print(f"    Mean velocity: {initial_speed:.2f}")

        sim = GalaxySimulation(
            positions, velocities, masses,
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001, dt=0.01, softening=0.1, device=device
        )

        crash = None
        prev_pos = None
        prev_vel = None
        prev_energy = None

        for tick in range(500):
            try:
                prev_pos = sim.positions.clone()
                prev_vel = sim.velocities.clone()
                prev_energy = sim.get_total_energy()

                sim.step()

                crash = detect_crash(
                    sim.positions, sim.velocities,
                    prev_pos, prev_vel,
                    sim.get_total_energy(), prev_energy,
                    sim.dt, tick
                )

                if crash:
                    crash.parameter = "velocity_mult"
                    crash.value = mult
                    break

            except Exception as e:
                crash = CrashReport(
                    parameter="velocity_mult",
                    value=mult,
                    crash_type="EXCEPTION",
                    tick=tick,
                    details=str(e),
                    severity=1.0
                )
                break

        if crash:
            print(f"    CRASH at tick {crash.tick}: {crash.crash_type}")
            print(f"    {crash.details}")
            results.append({"multiplier": mult, "crash": crash, "stable_ticks": crash.tick})
        else:
            print(f"    Stable for 500 ticks")
            results.append({"multiplier": mult, "crash": None, "stable_ticks": 500})

    return {"test": "velocity", "results": results}


def find_dt_crash_point(
    num_stars: int = 1000,
    device: torch.device = None
) -> dict:
    """
    Find the time step threshold where simulation breaks.
    """
    print("\n" + "="*60)
    print("TIME STEP CRASH POINT SEARCH")
    print("How big can dt be before physics breaks?")
    print("="*60)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dt_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    results = []

    positions, velocities, masses = create_disk_galaxy(
        num_stars=num_stars, galaxy_radius=10.0, device=device
    )
    positions, velocities, masses = positions.float(), velocities.float(), masses.float()

    for dt in dt_values:
        print(f"\n  Testing dt = {dt}...")

        sim = GalaxySimulation(
            positions.clone(), velocities.clone(), masses.clone(),
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001, dt=dt, softening=0.1, device=device
        )

        crash = None
        prev_pos = None
        prev_energy = None

        for tick in range(200):
            try:
                prev_pos = sim.positions.clone()
                prev_vel = sim.velocities.clone()
                prev_energy = sim.get_total_energy()

                sim.step()

                crash = detect_crash(
                    sim.positions, sim.velocities,
                    prev_pos, prev_vel,
                    sim.get_total_energy(), prev_energy,
                    dt, tick
                )

                if crash:
                    crash.parameter = "dt"
                    crash.value = dt
                    break

            except Exception as e:
                crash = CrashReport(
                    parameter="dt", value=dt,
                    crash_type="EXCEPTION", tick=tick,
                    details=str(e), severity=1.0
                )
                break

        if crash:
            print(f"    CRASH at tick {crash.tick}: {crash.crash_type}")
            results.append({"dt": dt, "crash": crash, "stable_ticks": crash.tick})
        else:
            print(f"    Stable for 200 ticks")
            results.append({"dt": dt, "crash": None, "stable_ticks": 200})

    return {"test": "dt", "results": results}


def find_quantization_crash_point(
    num_stars: int = 1000,
    device: torch.device = None
) -> dict:
    """
    Find the quantization level where simulation breaks.
    """
    print("\n" + "="*60)
    print("QUANTIZATION CRASH POINT SEARCH")
    print("How low can precision go before reality glitches?")
    print("="*60)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test from high to low precision
    quant_levels = [1000000, 65536, 4096, 1024, 256, 64, 32, 16, 8, 4, 3, 2]
    results = []

    positions, velocities, masses = create_disk_galaxy(
        num_stars=num_stars, galaxy_radius=10.0, device=device
    )
    positions, velocities, masses = positions.float(), velocities.float(), masses.float()

    for levels in quant_levels:
        bits = np.log2(levels)
        print(f"\n  Testing {levels} levels ({bits:.1f} bits)...")

        class QuantSim(GalaxySimulation):
            def __init__(self, *args, quant_levels, **kwargs):
                self.quant_levels = quant_levels
                super().__init__(*args, **kwargs)

            def _compute_accelerations(self):
                pos = self.positions
                diff = pos.unsqueeze(0) - pos.unsqueeze(1)
                dist_sq = (diff ** 2).sum(dim=-1) + self.softening_sq

                if self.quant_levels < 100000:
                    dist_sq = _grid_quantize_safe(dist_sq, self.quant_levels, min_val=0.01)

                dist_cubed = dist_sq ** 1.5
                force_factor = self.G / dist_cubed
                force_factor = force_factor * self.masses.unsqueeze(0)
                force_factor = force_factor * (1 - torch.eye(self.num_stars, device=self.device))
                return (force_factor.unsqueeze(-1) * diff).sum(dim=1)

        sim = QuantSim(
            positions.clone(), velocities.clone(), masses.clone(),
            quant_levels=levels,
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001, dt=0.01, softening=0.1, device=device
        )

        crash = None
        prev_energy = None

        for tick in range(300):
            try:
                prev_pos = sim.positions.clone()
                prev_vel = sim.velocities.clone()
                prev_energy_val = sim.get_total_energy()

                sim.step()

                crash = detect_crash(
                    sim.positions, sim.velocities,
                    prev_pos, prev_vel,
                    sim.get_total_energy(), prev_energy_val,
                    sim.dt, tick
                )

                if crash:
                    crash.parameter = "quant_levels"
                    crash.value = levels
                    break

            except Exception as e:
                crash = CrashReport(
                    parameter="quant_levels", value=levels,
                    crash_type="EXCEPTION", tick=tick,
                    details=str(e), severity=1.0
                )
                break

        if crash:
            print(f"    CRASH at tick {crash.tick}: {crash.crash_type}")
            results.append({"levels": levels, "bits": bits, "crash": crash, "stable_ticks": crash.tick})
        else:
            print(f"    Stable for 300 ticks")
            results.append({"levels": levels, "bits": bits, "crash": None, "stable_ticks": 300})

    return {"test": "quantization", "results": results}


def find_softening_crash_point(
    num_stars: int = 1000,
    device: torch.device = None
) -> dict:
    """
    Find the softening threshold where close encounters explode.
    """
    print("\n" + "="*60)
    print("SOFTENING CRASH POINT SEARCH")
    print("How close can objects get before singularities form?")
    print("="*60)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    softening_values = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0001]
    results = []

    positions, velocities, masses = create_disk_galaxy(
        num_stars=num_stars, galaxy_radius=10.0, device=device
    )
    positions, velocities, masses = positions.float(), velocities.float(), masses.float()

    for soft in softening_values:
        print(f"\n  Testing softening = {soft}...")

        sim = GalaxySimulation(
            positions.clone(), velocities.clone(), masses.clone(),
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001, dt=0.01, softening=soft, device=device
        )

        crash = None

        for tick in range(200):
            try:
                prev_pos = sim.positions.clone()
                prev_vel = sim.velocities.clone()
                prev_energy = sim.get_total_energy()

                sim.step()

                crash = detect_crash(
                    sim.positions, sim.velocities,
                    prev_pos, prev_vel,
                    sim.get_total_energy(), prev_energy,
                    sim.dt, tick
                )

                if crash:
                    crash.parameter = "softening"
                    crash.value = soft
                    break

            except Exception as e:
                crash = CrashReport(
                    parameter="softening", value=soft,
                    crash_type="EXCEPTION", tick=tick,
                    details=str(e), severity=1.0
                )
                break

        if crash:
            print(f"    CRASH at tick {crash.tick}: {crash.crash_type}")
            results.append({"softening": soft, "crash": crash, "stable_ticks": crash.tick})
        else:
            print(f"    Stable for 200 ticks")
            results.append({"softening": soft, "crash": None, "stable_ticks": 200})

    return {"test": "softening", "results": results}


def plot_crash_points(all_results: list[dict], save_dir: str):
    """Plot crash point analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Velocity crash
    ax1 = axes[0, 0]
    vel_data = next((r for r in all_results if r["test"] == "velocity"), None)
    if vel_data:
        mults = [r["multiplier"] for r in vel_data["results"]]
        ticks = [r["stable_ticks"] for r in vel_data["results"]]
        crashed = [r["crash"] is not None for r in vel_data["results"]]

        colors = ['red' if c else 'green' for c in crashed]
        ax1.bar(range(len(mults)), ticks, color=colors, alpha=0.7)
        ax1.set_xticks(range(len(mults)))
        ax1.set_xticklabels([f"{m}x" for m in mults], rotation=45)
        ax1.set_xlabel("Velocity Multiplier")
        ax1.set_ylabel("Stable Ticks")
        ax1.set_title("Velocity Crash Point")
        ax1.axhline(y=500, color='gray', linestyle='--', alpha=0.5)

    # 2. dt crash
    ax2 = axes[0, 1]
    dt_data = next((r for r in all_results if r["test"] == "dt"), None)
    if dt_data:
        dts = [r["dt"] for r in dt_data["results"]]
        ticks = [r["stable_ticks"] for r in dt_data["results"]]
        crashed = [r["crash"] is not None for r in dt_data["results"]]

        colors = ['red' if c else 'green' for c in crashed]
        ax2.bar(range(len(dts)), ticks, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(dts)))
        ax2.set_xticklabels([f"{d}" for d in dts], rotation=45)
        ax2.set_xlabel("Time Step (dt)")
        ax2.set_ylabel("Stable Ticks")
        ax2.set_title("Time Step Crash Point")

    # 3. Quantization crash
    ax3 = axes[1, 0]
    quant_data = next((r for r in all_results if r["test"] == "quantization"), None)
    if quant_data:
        bits = [r["bits"] for r in quant_data["results"]]
        ticks = [r["stable_ticks"] for r in quant_data["results"]]
        crashed = [r["crash"] is not None for r in quant_data["results"]]

        colors = ['red' if c else 'green' for c in crashed]
        ax3.bar(range(len(bits)), ticks, color=colors, alpha=0.7)
        ax3.set_xticks(range(len(bits)))
        ax3.set_xticklabels([f"{b:.1f}" for b in bits], rotation=45)
        ax3.set_xlabel("Precision (bits)")
        ax3.set_ylabel("Stable Ticks")
        ax3.set_title("Quantization Crash Point")

    # 4. Softening crash
    ax4 = axes[1, 1]
    soft_data = next((r for r in all_results if r["test"] == "softening"), None)
    if soft_data:
        softs = [r["softening"] for r in soft_data["results"]]
        ticks = [r["stable_ticks"] for r in soft_data["results"]]
        crashed = [r["crash"] is not None for r in soft_data["results"]]

        colors = ['red' if c else 'green' for c in crashed]
        ax4.bar(range(len(softs)), ticks, color=colors, alpha=0.7)
        ax4.set_xticks(range(len(softs)))
        ax4.set_xticklabels([f"{s}" for s in softs], rotation=45)
        ax4.set_xlabel("Softening Length")
        ax4.set_ylabel("Stable Ticks")
        ax4.set_title("Softening Crash Point")

    plt.suptitle("Crash Points: Where Does Reality Break?", fontsize=16, y=1.02)
    plt.tight_layout()

    save_path = Path(save_dir) / "crash_points.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {save_path}")

    return fig


def print_crash_summary(all_results: list[dict]):
    """Print summary of all crash points."""
    print("\n" + "="*60)
    print("CRASH POINT SUMMARY")
    print("="*60)

    print("""
These are the boundaries of our simulated reality.
Beyond these limits, objects "clip through" physics.
""")

    for result in all_results:
        print(f"\n{result['test'].upper()} BOUNDARY:")
        print("-" * 40)

        # Find first crash
        crashes = [r for r in result["results"] if r.get("crash")]
        stable = [r for r in result["results"] if not r.get("crash")]

        if crashes:
            first_crash = crashes[0]
            if "multiplier" in first_crash:
                print(f"  Crash at: {first_crash['multiplier']}x velocity")
            elif "dt" in first_crash:
                print(f"  Crash at: dt = {first_crash['dt']}")
            elif "levels" in first_crash:
                print(f"  Crash at: {first_crash['levels']} levels ({first_crash['bits']:.1f} bits)")
            elif "softening" in first_crash:
                print(f"  Crash at: softening = {first_crash['softening']}")

            print(f"  Type: {first_crash['crash'].crash_type}")
            print(f"  Tick: {first_crash['crash'].tick}")

        if stable:
            last_stable = stable[-1]
            if "multiplier" in last_stable:
                print(f"  Safe up to: {last_stable['multiplier']}x velocity")
            elif "dt" in last_stable:
                print(f"  Safe up to: dt = {last_stable['dt']}")
            elif "levels" in last_stable:
                print(f"  Safe down to: {last_stable['levels']} levels")
            elif "softening" in last_stable:
                print(f"  Safe down to: softening = {last_stable['softening']}")

    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("""
These crash points define the "render distance" of our physics engine.

If our universe is a simulation:
  - Speed of light = velocity crash point
  - Planck time = dt crash point
  - Planck length = softening crash point
  - Quantum uncertainty = quantization crash point

Dark matter might be the "error correction" that keeps
the universe from hitting these boundaries.
""")


def main():
    parser = argparse.ArgumentParser(description="Find simulation crash points")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["velocity", "dt", "quantization", "softening", "all"])
    parser.add_argument("--stars", type=int, default=1000)
    parser.add_argument("--output", type=str, default="output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = []

    if args.mode in ["velocity", "all"]:
        all_results.append(find_velocity_crash_point(args.stars, device))

    if args.mode in ["dt", "all"]:
        all_results.append(find_dt_crash_point(args.stars, device))

    if args.mode in ["quantization", "all"]:
        all_results.append(find_quantization_crash_point(args.stars, device))

    if args.mode in ["softening", "all"]:
        all_results.append(find_softening_crash_point(args.stars, device))

    # Summary
    print_crash_summary(all_results)

    # Plot
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    plot_crash_points(all_results, args.output)
    plt.show()


if __name__ == "__main__":
    main()
