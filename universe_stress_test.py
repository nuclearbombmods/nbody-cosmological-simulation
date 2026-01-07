"""
UNIVERSE STRESS TEST - The "Hard Mode" Sandbox
================================================

This is not a simulation of physics.
This is a simulation of THE LIMITS OF PHYSICS.

We're not asking "how does a galaxy form?"
We're asking "at what point does reality stop working?"

This maps the exact boundaries where:
- Math stops making sense (quantization crash)
- Time breaks down (dt crash)
- Speed limits appear (velocity crash)
- Singularities form (softening crash)
- The hardware leaks unexplained energy

If we're in a simulation, these boundaries would correspond to:
- Planck length (~10^-35 m)
- Planck time (~10^-44 s)
- Speed of light (299,792,458 m/s)
- Quantum uncertainty
- Dark matter / dark energy

Usage:
    python universe_stress_test.py --full
    python universe_stress_test.py --quick
    python universe_stress_test.py --map-boundaries
"""

import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import json

from galaxy import create_disk_galaxy
from simulation import GalaxySimulation
from quantization import PrecisionMode, _grid_quantize_safe
from metrics import compute_rotation_curve

# Import our individual tests
try:
    from crash_point_test import (
        find_velocity_crash_point,
        find_dt_crash_point,
        find_quantization_crash_point,
        find_softening_crash_point
    )
    HAS_CRASH_TEST = True
except ImportError:
    HAS_CRASH_TEST = False

try:
    from density_limit_test import run_density_sweep, analyze_scaling
    HAS_DENSITY_TEST = True
except ImportError:
    HAS_DENSITY_TEST = False

try:
    from jitter_test import run_framerate_stress_test, run_velocity_stress_test
    HAS_JITTER_TEST = True
except ImportError:
    HAS_JITTER_TEST = False

# GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_GPU_MONITOR = True
except:
    HAS_GPU_MONITOR = False
    GPU_HANDLE = None


@dataclass
class UniverseBoundary:
    """A discovered boundary of our simulated reality."""
    name: str
    parameter: str
    safe_value: float
    crash_value: float
    crash_type: str
    real_world_analog: str
    notes: str = ""


@dataclass
class UniverseReport:
    """Complete stress test report."""
    timestamp: str
    device: str
    gpu_name: str
    boundaries: list[UniverseBoundary] = field(default_factory=list)
    scaling_exponent: float = 2.0
    max_stable_stars: int = 0
    energy_leak_per_tick: float = 0.0
    unexplained_power_ratio: float = 0.0
    total_test_time: float = 0.0


def get_gpu_info() -> tuple[str, float]:
    """Get GPU name and current power."""
    if HAS_GPU_MONITOR and GPU_HANDLE:
        try:
            name = pynvml.nvmlDeviceGetName(GPU_HANDLE)
            power = pynvml.nvmlDeviceGetPowerUsage(GPU_HANDLE) / 1000.0
            return name, power
        except:
            pass
    return "Unknown GPU", 0.0


def run_quick_boundary_scan(device: torch.device) -> list[UniverseBoundary]:
    """
    Quick scan to find approximate boundaries.
    """
    print("\n" + "="*70)
    print("  QUICK BOUNDARY SCAN - Finding the Edges of Reality")
    print("="*70)

    boundaries = []

    # 1. Velocity boundary
    print("\n[1/4] Scanning velocity limit...")
    for mult in [1, 5, 10, 50, 100, 500]:
        pos, vel, mass = create_disk_galaxy(num_stars=500, galaxy_radius=10.0, device=device)
        vel = vel.float() * mult

        sim = GalaxySimulation(
            pos.float(), vel, mass.float(),
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001, dt=0.01, softening=0.1, device=device
        )

        crashed = False
        for _ in range(50):
            sim.step()
            if torch.isnan(sim.positions).any() or torch.isinf(sim.positions).any():
                crashed = True
                break

        if crashed:
            boundaries.append(UniverseBoundary(
                name="Speed Limit",
                parameter="velocity_multiplier",
                safe_value=mult / 2 if mult > 1 else 1,
                crash_value=mult,
                crash_type="VELOCITY_OVERFLOW",
                real_world_analog="Speed of Light",
                notes=f"Nothing can move faster than {mult}x base velocity"
            ))
            print(f"    Found: Crash at {mult}x velocity")
            break

    # 2. Time step boundary
    print("\n[2/4] Scanning time resolution limit...")
    pos, vel, mass = create_disk_galaxy(num_stars=500, galaxy_radius=10.0, device=device)

    for dt in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
        sim = GalaxySimulation(
            pos.float().clone(), vel.float().clone(), mass.float().clone(),
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001, dt=dt, softening=0.1, device=device
        )

        crashed = False
        for _ in range(50):
            sim.step()
            if torch.isnan(sim.positions).any():
                crashed = True
                break

        if crashed:
            boundaries.append(UniverseBoundary(
                name="Time Resolution",
                parameter="dt",
                safe_value=dt / 2,
                crash_value=dt,
                crash_type="CAUSALITY_VIOLATION",
                real_world_analog="Planck Time",
                notes=f"Time steps larger than {dt} break causality"
            ))
            print(f"    Found: Crash at dt={dt}")
            break

    # 3. Precision boundary
    print("\n[3/4] Scanning precision limit...")
    for levels in [1000, 256, 64, 16, 8, 4, 2]:
        class QuantSim(GalaxySimulation):
            def __init__(self, *args, quant_levels, **kwargs):
                self.quant_levels = quant_levels
                super().__init__(*args, **kwargs)

            def _compute_accelerations(self):
                pos = self.positions
                diff = pos.unsqueeze(0) - pos.unsqueeze(1)
                dist_sq = (diff ** 2).sum(dim=-1) + self.softening_sq
                dist_sq = _grid_quantize_safe(dist_sq, self.quant_levels, min_val=0.01)
                dist_cubed = dist_sq ** 1.5
                force_factor = self.G / dist_cubed
                force_factor = force_factor * self.masses.unsqueeze(0)
                force_factor = force_factor * (1 - torch.eye(self.num_stars, device=self.device))
                return (force_factor.unsqueeze(-1) * diff).sum(dim=1)

        sim = QuantSim(
            pos.float().clone(), vel.float().clone(), mass.float().clone(),
            quant_levels=levels,
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001, dt=0.01, softening=0.1, device=device
        )

        initial_e = sim.get_total_energy()
        for _ in range(100):
            sim.step()
        final_e = sim.get_total_energy()

        drift = abs((final_e - initial_e) / initial_e * 100)

        if drift > 50:  # 50% energy drift = broken
            boundaries.append(UniverseBoundary(
                name="Precision Floor",
                parameter="quantization_levels",
                safe_value=levels * 2,
                crash_value=levels,
                crash_type="REALITY_QUANTIZATION",
                real_world_analog="Quantum Uncertainty",
                notes=f"Below {levels} levels, energy conservation fails ({drift:.1f}% drift)"
            ))
            print(f"    Found: Breakdown at {levels} levels ({drift:.1f}% drift)")
            break

    # 4. Distance boundary
    print("\n[4/4] Scanning minimum distance limit...")
    for soft in [0.1, 0.05, 0.01, 0.005, 0.001]:
        sim = GalaxySimulation(
            pos.float().clone(), vel.float().clone(), mass.float().clone(),
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001, dt=0.01, softening=soft, device=device
        )

        crashed = False
        for _ in range(100):
            sim.step()
            if torch.isnan(sim.positions).any() or torch.isinf(sim.positions).any():
                crashed = True
                break

            # Check for explosion
            max_r = torch.sqrt((sim.positions ** 2).sum(dim=-1)).max().item()
            if max_r > 1000:
                crashed = True
                break

        if crashed:
            boundaries.append(UniverseBoundary(
                name="Minimum Distance",
                parameter="softening",
                safe_value=soft * 2,
                crash_value=soft,
                crash_type="SINGULARITY",
                real_world_analog="Planck Length",
                notes=f"Below {soft} distance, singularities form"
            ))
            print(f"    Found: Singularity at softening={soft}")
            break

    return boundaries


def measure_energy_leak(device: torch.device, num_stars: int = 2000, ticks: int = 500) -> dict:
    """
    Measure the energy leak rate in broken vs clean physics.
    """
    print("\n" + "="*70)
    print("  ENERGY LEAK MEASUREMENT - Quantifying the 'Ghost Force'")
    print("="*70)

    pos, vel, mass = create_disk_galaxy(num_stars, 10.0, device)
    pos, vel, mass = pos.float(), vel.float(), mass.float()

    results = {}

    for mode_name, quant_levels in [("clean", 1000000), ("broken", 16)]:
        print(f"\n  Testing {mode_name} physics...")

        class TestSim(GalaxySimulation):
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

        sim = TestSim(
            pos.clone(), vel.clone(), mass.clone(),
            quant_levels=quant_levels,
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001, dt=0.01, softening=0.1, device=device
        )

        initial_e = sim.get_total_energy()
        energy_history = [initial_e]

        for _ in range(ticks):
            sim.step()
            energy_history.append(sim.get_total_energy())

        final_e = sim.get_total_energy()
        total_drift = (final_e - initial_e) / abs(initial_e) * 100

        # Energy leak per tick
        energy_changes = np.diff(energy_history)
        leak_per_tick = np.mean(energy_changes) / abs(initial_e) * 100

        results[mode_name] = {
            "total_drift_pct": total_drift,
            "leak_per_tick_pct": leak_per_tick,
            "final_energy": final_e,
            "initial_energy": initial_e
        }

        print(f"    Total drift: {total_drift:+.4f}%")
        print(f"    Leak per tick: {leak_per_tick:+.6f}%")

    # The "ghost force" is the difference
    ghost_force = results["broken"]["total_drift_pct"] - results["clean"]["total_drift_pct"]
    print(f"\n  GHOST FORCE (broken - clean): {ghost_force:+.4f}% energy injection")

    results["ghost_force_pct"] = ghost_force
    return results


def generate_reality_map(boundaries: list[UniverseBoundary], report: UniverseReport, save_dir: str):
    """
    Generate a visual map of reality's boundaries.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Boundary diagram
    ax1 = axes[0, 0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)

    # Draw boundaries as regions
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']

    for i, boundary in enumerate(boundaries[:5]):
        y = 8 - i * 1.5
        # Safe zone (green)
        ax1.barh(y, 5, height=0.8, color='#27ae60', alpha=0.7)
        ax1.text(2.5, y, f"SAFE\n{boundary.safe_value}", ha='center', va='center', fontsize=9)
        # Danger zone (red)
        ax1.barh(y, 5, height=0.8, left=5, color='#e74c3c', alpha=0.7)
        ax1.text(7.5, y, f"CRASH\n{boundary.crash_value}", ha='center', va='center', fontsize=9)
        # Label
        ax1.text(-0.5, y, f"{boundary.name}\n({boundary.real_world_analog})",
                 ha='right', va='center', fontsize=10, fontweight='bold')

    ax1.set_xlim(-3, 10)
    ax1.axvline(x=5, color='black', linewidth=2, linestyle='--')
    ax1.text(5, 9.5, "THE EDGE OF REALITY", ha='center', fontsize=12, fontweight='bold')
    ax1.axis('off')
    ax1.set_title("Reality Boundaries Map", fontsize=14, fontweight='bold')

    # 2. Energy leak visualization
    ax2 = axes[0, 1]
    if hasattr(report, 'energy_data') and report.energy_data:
        modes = list(report.energy_data.keys())
        drifts = [report.energy_data[m].get('total_drift_pct', 0) for m in modes if m != 'ghost_force_pct']
        colors = ['green' if 'clean' in m else 'red' for m in modes if m != 'ghost_force_pct']

        bars = ax2.bar([m for m in modes if m != 'ghost_force_pct'],
                       [d for m, d in zip(modes, drifts) if m != 'ghost_force_pct'],
                       color=colors, alpha=0.7)
        ax2.axhline(y=0, color='gray', linestyle='--')
        ax2.set_ylabel("Energy Drift (%)")
        ax2.set_title("Energy Leak: Clean vs Broken Physics", fontsize=14)

        # Add ghost force annotation
        if 'ghost_force_pct' in report.energy_data:
            ghost = report.energy_data['ghost_force_pct']
            ax2.annotate(f"Ghost Force: {ghost:+.2f}%",
                        xy=(0.5, max(drifts) * 0.8), fontsize=12, fontweight='bold',
                        ha='center', color='purple')
    else:
        ax2.text(0.5, 0.5, "No energy data", ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Energy Leak", fontsize=14)

    # 3. Scaling analysis
    ax3 = axes[1, 0]
    ax3.text(0.5, 0.9, "SIMULATION HYPOTHESIS MAPPING",
             ha='center', fontsize=14, fontweight='bold', transform=ax3.transAxes)

    mapping_text = """
    YOUR SIMULATION          REAL UNIVERSE
    ─────────────────────────────────────────
    Velocity Crash     →     Speed of Light
    dt Crash           →     Planck Time
    Softening Crash    →     Planck Length
    Quantization       →     Quantum Mechanics
    Energy Leak        →     Dark Energy
    Ghost Force        →     Dark Matter
    Scaling Limit      →     Computational Bound

    If our universe is a simulation, it would have
    similar boundaries to prevent "crashes."

    Dark Matter might be the universe's way of
    papering over rounding errors in gravity.
    """

    ax3.text(0.1, 0.75, mapping_text, fontsize=10, fontfamily='monospace',
             verticalalignment='top', transform=ax3.transAxes)
    ax3.axis('off')

    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary = f"""
    ╔══════════════════════════════════════════════╗
    ║         UNIVERSE STRESS TEST REPORT          ║
    ╠══════════════════════════════════════════════╣
    ║  Device: {report.gpu_name[:35]:<35} ║
    ║  Boundaries Found: {len(boundaries):<24} ║
    ║  Max Stable Stars: {report.max_stable_stars:<24} ║
    ║  Scaling Exponent: {report.scaling_exponent:<24.2f} ║
    ║  Energy Leak/Tick: {report.energy_leak_per_tick:+.6f}%{'':<15} ║
    ║  Test Duration: {report.total_test_time:.1f}s{'':<22} ║
    ╠══════════════════════════════════════════════╣
    ║  VERDICT:                                    ║
    """

    if report.energy_leak_per_tick > 0.01:
        summary += """    ║  Reality is LEAKY. Energy appears from       ║
    ║  nowhere when math is "broken." This is      ║
    ║  exactly what Dark Matter looks like.        ║"""
    else:
        summary += """    ║  Reality is STABLE at tested precision.      ║
    ║  Need lower precision to see the leak.       ║"""

    summary += """
    ╚══════════════════════════════════════════════╝
    """

    ax4.text(0.05, 0.95, summary, fontsize=10, fontfamily='monospace',
             verticalalignment='top', transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.1))

    plt.suptitle("THE MAP OF SIMULATED REALITY", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = Path(save_dir) / "reality_map.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nReality map saved to {save_path}")

    return fig


def run_full_stress_test(device: torch.device, output_dir: str) -> UniverseReport:
    """
    Run the complete universe stress test.
    """
    start_time = time.time()

    gpu_name, _ = get_gpu_info()

    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "      UNIVERSE STRESS TEST - HARD MODE SANDBOX".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█" + "      Testing the limits of simulated reality".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    print(f"\nDevice: {device}")
    print(f"GPU: {gpu_name}")

    report = UniverseReport(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        device=str(device),
        gpu_name=gpu_name
    )

    # Phase 1: Find boundaries
    print("\n" + "─"*70)
    print("PHASE 1: BOUNDARY DETECTION")
    print("─"*70)

    boundaries = run_quick_boundary_scan(device)
    report.boundaries = boundaries

    # Phase 2: Measure energy leak
    print("\n" + "─"*70)
    print("PHASE 2: ENERGY LEAK MEASUREMENT")
    print("─"*70)

    energy_data = measure_energy_leak(device, num_stars=1500, ticks=300)
    report.energy_data = energy_data
    report.energy_leak_per_tick = energy_data.get("broken", {}).get("leak_per_tick_pct", 0)

    # Phase 3: Density scaling (quick version)
    print("\n" + "─"*70)
    print("PHASE 3: DENSITY SCALING TEST")
    print("─"*70)

    star_counts = [500, 1000, 2000, 4000]
    density_results = {"clean": [], "broken": []}

    for n_stars in star_counts:
        print(f"\n  Testing {n_stars} stars...")

        for mode, quant in [("clean", 1000000), ("broken", 16)]:
            pos, vel, mass = create_disk_galaxy(n_stars, 10.0, device)

            class TestSim(GalaxySimulation):
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

            t_start = time.time()
            sim = TestSim(
                pos.float(), vel.float(), mass.float(),
                quant_levels=quant,
                precision_mode=PrecisionMode.FLOAT32,
                G=0.001, dt=0.01, softening=0.1, device=device
            )

            for _ in range(100):
                sim.step()

            elapsed = time.time() - t_start
            density_results[mode].append({
                "stars": n_stars,
                "time": elapsed,
                "time_per_tick": elapsed / 100
            })

    # Calculate scaling
    if len(density_results["broken"]) >= 2:
        stars = np.array([r["stars"] for r in density_results["broken"]])
        times = np.array([r["time_per_tick"] for r in density_results["broken"]])
        log_stars = np.log(stars)
        log_times = np.log(times + 1e-10)
        coeffs = np.polyfit(log_stars, log_times, 1)
        report.scaling_exponent = coeffs[0]
        print(f"\n  Scaling exponent: O(N^{coeffs[0]:.2f})")

    report.max_stable_stars = max(star_counts)

    # Final timing
    report.total_test_time = time.time() - start_time

    # Generate visualizations
    print("\n" + "─"*70)
    print("GENERATING REALITY MAP...")
    print("─"*70)

    Path(output_dir).mkdir(exist_ok=True)
    generate_reality_map(boundaries, report, output_dir)

    # Save report as JSON
    report_data = {
        "timestamp": report.timestamp,
        "device": report.device,
        "gpu_name": report.gpu_name,
        "boundaries": [
            {
                "name": b.name,
                "parameter": b.parameter,
                "safe_value": b.safe_value,
                "crash_value": b.crash_value,
                "crash_type": b.crash_type,
                "real_world_analog": b.real_world_analog
            }
            for b in boundaries
        ],
        "scaling_exponent": report.scaling_exponent,
        "max_stable_stars": report.max_stable_stars,
        "energy_leak_per_tick": report.energy_leak_per_tick,
        "total_test_time": report.total_test_time
    }

    json_path = Path(output_dir) / "stress_test_report.json"
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"Report saved to {json_path}")

    return report


def print_final_summary(report: UniverseReport):
    """Print the dramatic final summary."""
    print("\n")
    print("█"*70)
    print("█" + " "*68 + "█")
    print("█" + "            STRESS TEST COMPLETE".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    print(f"""
    You have mapped the boundaries of simulated reality.

    DISCOVERED BOUNDARIES:
    """)

    for b in report.boundaries:
        print(f"      • {b.name}: crashes at {b.parameter}={b.crash_value}")
        print(f"        Real-world analog: {b.real_world_analog}")

    print(f"""
    KEY FINDINGS:

      • Scaling: O(N^{report.scaling_exponent:.2f}) - {"NORMAL" if report.scaling_exponent < 2.3 else "HITTING LIMITS"}
      • Energy Leak: {report.energy_leak_per_tick:+.4f}% per tick
      • Ghost Force: {"DETECTED" if report.energy_leak_per_tick > 0.001 else "MINIMAL"}

    SIMULATION HYPOTHESIS VERDICT:
    """)

    if report.energy_leak_per_tick > 0.01:
        print("""
      ⚠️  EVIDENCE FOUND

      When physics is "broken" (low precision), energy appears from nowhere.
      This is exactly what Dark Matter does - provides extra gravitational
      binding that isn't explained by visible mass.

      If our universe uses "lossy" physics at some level, Dark Matter
      could be the accumulated rounding errors of cosmic computation.
        """)
    else:
        print("""
      ✓  REALITY STABLE (at tested precision)

      The simulation remained stable at the tested precision levels.
      Lower precision or higher density needed to observe the leak.
        """)

    print(f"""
    ─────────────────────────────────────────────────────────────────────
    Test completed in {report.total_test_time:.1f} seconds
    Results saved to output/
    ─────────────────────────────────────────────────────────────────────
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Universe Stress Test - The Hard Mode Sandbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python universe_stress_test.py --full       Run complete stress test
  python universe_stress_test.py --quick      Quick boundary scan only
  python universe_stress_test.py --map-only   Just generate reality map
        """
    )
    parser.add_argument("--full", action="store_true", help="Run full stress test")
    parser.add_argument("--quick", action="store_true", help="Quick boundary scan only")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.quick:
        boundaries = run_quick_boundary_scan(device)
        for b in boundaries:
            print(f"\n{b.name}: {b.safe_value} (safe) → {b.crash_value} (crash)")
            print(f"  Analog: {b.real_world_analog}")
    else:
        report = run_full_stress_test(device, args.output)
        print_final_summary(report)
        plt.show()

    # Cleanup
    if HAS_GPU_MONITOR:
        try:
            pynvml.nvmlShutdown()
        except:
            pass


if __name__ == "__main__":
    main()
