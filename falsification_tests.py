"""
Falsification Tests - Can the Hypothesis Survive Scientific Scrutiny?

This tests the major objections to "Dark Matter = Rounding Errors":

HOLE 1: Infinite Precision Problem
  - Does the effect vanish at high precision?
  - Convergence test from 2-bit to 64-bit equivalent

HOLE 2: Bullet Cluster Problem
  - Can "ghost mass" separate from visible mass during collision?
  - Real DM separates from gas; can quantization errors do this?

HOLE 4: Softening/Parameter Artifacts
  - Does the effect depend on simulation parameters?
  - Sweep softening length and time step

(Hole 3 - CMB - requires cosmological simulations beyond N-body scope)

Usage:
    python falsification_tests.py --test convergence
    python falsification_tests.py --test bullet
    python falsification_tests.py --test parameters
    python falsification_tests.py --all
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


# =============================================================================
# HOLE 1: CONVERGENCE TEST - Does effect vanish at high precision?
# =============================================================================

def test_convergence(num_stars: int = 2000, num_ticks: int = 500, device=None):
    """
    Test if the "dark matter effect" converges to zero at high precision.

    If the effect vanishes as precision increases, it's a numerical artifact.
    If it persists even at very high precision, something else is happening.
    """
    print("\n" + "="*60)
    print("HOLE 1: CONVERGENCE TEST")
    print("Does the 'ghost force' vanish at high precision?")
    print("="*60)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extended precision range - from 2-bit to effectively infinite
    test_levels = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
                   8192, 16384, 32768, 65536, 131072, 1000000]

    print(f"\nCreating test galaxy with {num_stars} stars...")
    positions, velocities, masses = create_disk_galaxy(
        num_stars=num_stars, galaxy_radius=10.0, device=device
    )
    positions, velocities, masses = positions.float(), velocities.float(), masses.float()

    results = []

    for levels in test_levels:
        # Custom quantized simulation
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

        sim = QuantSim(positions.clone(), velocities.clone(), masses.clone(),
                       quant_levels=levels, precision_mode=PrecisionMode.FLOAT32,
                       G=0.001, dt=0.01, softening=0.1, device=device)

        initial_energy = sim.get_total_energy()
        for _ in range(num_ticks):
            sim.step()
        final_energy = sim.get_total_energy()

        drift = (final_energy - initial_energy) / abs(initial_energy) * 100
        bits = np.log2(levels)
        results.append((levels, bits, drift))
        print(f"  {levels:>7} levels ({bits:>5.1f} bits): {drift:+.6f}% drift")

    # Analysis
    print("\n" + "-"*40)
    print("CONVERGENCE ANALYSIS:")

    # Check if drift approaches zero
    final_drift = abs(results[-1][2])
    initial_drift = abs(results[0][2])

    if final_drift < 0.01 and initial_drift > 1.0:
        print("✓ Effect CONVERGES to zero at high precision")
        print("  This suggests the effect is a numerical artifact.")
        print("  The universe would need 'low resolution' for this to work.")
    elif final_drift > 0.1:
        print("✗ Effect PERSISTS even at very high precision")
        print("  This is unexpected - may indicate a bug or other effect.")
    else:
        print("? Effect decreases but doesn't fully vanish")
        print("  Needs investigation with even higher precision.")

    return results


# =============================================================================
# HOLE 2: BULLET CLUSTER TEST - Can ghost mass separate from visible mass?
# =============================================================================

def test_bullet_cluster(num_stars: int = 1000, num_ticks: int = 800, device=None):
    """
    Simulate two galaxies colliding to test if "ghost mass" can separate.

    The Bullet Cluster shows dark matter separating from visible gas.
    If quantization errors can't produce this separation, the hypothesis fails.
    """
    print("\n" + "="*60)
    print("HOLE 2: BULLET CLUSTER TEST")
    print("Can 'ghost mass' separate from visible mass in a collision?")
    print("="*60)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create two galaxies offset and moving toward each other
    print("\nCreating two colliding galaxies...")

    # Galaxy 1: Left side, moving right
    pos1, vel1, mass1 = create_disk_galaxy(num_stars=num_stars, galaxy_radius=5.0, device=device)
    pos1[:, 0] -= 15  # Offset left
    vel1[:, 0] += 0.5  # Moving right

    # Galaxy 2: Right side, moving left
    pos2, vel2, mass2 = create_disk_galaxy(num_stars=num_stars, galaxy_radius=5.0, device=device)
    pos2[:, 0] += 15  # Offset right
    vel2[:, 0] -= 0.5  # Moving left

    # Combine
    positions = torch.cat([pos1, pos2], dim=0).float()
    velocities = torch.cat([vel1, vel2], dim=0).float()
    masses = torch.cat([mass1, mass2], dim=0).float()

    # Track which stars belong to which galaxy
    galaxy_id = torch.cat([
        torch.zeros(num_stars, device=device),
        torch.ones(num_stars, device=device)
    ])

    results = {"float64": None, "int4": None}

    for mode_name, levels in [("float64", 1000000), ("int4", 16)]:
        print(f"\nRunning collision with {mode_name} precision...")

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

        sim = QuantSim(positions.clone(), velocities.clone(), masses.clone(),
                       quant_levels=levels, precision_mode=PrecisionMode.FLOAT32,
                       G=0.001, dt=0.01, softening=0.2, device=device)

        # Track centers of mass over time
        history = {"com": [], "grav_center": [], "ticks": []}

        for tick in range(num_ticks):
            sim.step()

            if tick % 50 == 0:
                # Center of mass (where stars actually are)
                com = (sim.positions * sim.masses.unsqueeze(-1)).sum(dim=0) / sim.masses.sum()

                # "Gravitational center" - where gravity seems to come from
                # Approximate by weighting by local density
                pos = sim.positions
                diff = pos.unsqueeze(0) - pos.unsqueeze(1)
                dist = torch.sqrt((diff ** 2).sum(dim=-1) + 0.1)

                # Local density approximation
                local_density = (1.0 / dist).sum(dim=1)
                grav_weights = local_density * sim.masses
                grav_center = (pos * grav_weights.unsqueeze(-1)).sum(dim=0) / grav_weights.sum()

                history["com"].append(com.cpu().numpy())
                history["grav_center"].append(grav_center.cpu().numpy())
                history["ticks"].append(tick)

        results[mode_name] = {
            "final_positions": sim.positions.cpu().numpy(),
            "history": history,
            "galaxy_id": galaxy_id.cpu().numpy()
        }

    # Analyze separation
    print("\n" + "-"*40)
    print("SEPARATION ANALYSIS:")

    # Compute max separation between COM and gravitational center
    for mode in ["float64", "int4"]:
        coms = np.array(results[mode]["history"]["com"])
        gravs = np.array(results[mode]["history"]["grav_center"])
        separations = np.sqrt(((coms - gravs) ** 2).sum(axis=1))
        max_sep = separations.max()
        print(f"  {mode}: Max separation = {max_sep:.4f}")

    sep_f64 = np.array(results["float64"]["history"]["com"]) - np.array(results["float64"]["history"]["grav_center"])
    sep_int4 = np.array(results["int4"]["history"]["com"]) - np.array(results["int4"]["history"]["grav_center"])

    max_sep_f64 = np.sqrt((sep_f64 ** 2).sum(axis=1)).max()
    max_sep_int4 = np.sqrt((sep_int4 ** 2).sum(axis=1)).max()

    if max_sep_int4 > max_sep_f64 * 1.5:
        print("\n✓ int4 shows MORE separation than float64")
        print("  This COULD support mass/gravity separation in collisions.")
    else:
        print("\n✗ No significant separation difference")
        print("  Quantization doesn't create Bullet Cluster-like behavior.")

    return results


# =============================================================================
# HOLE 4: PARAMETER ARTIFACTS TEST
# =============================================================================

def test_parameter_sensitivity(num_stars: int = 1500, num_ticks: int = 400, device=None):
    """
    Test if the "dark matter effect" is actually just a parameter artifact.

    Sweep softening length and time step to see if the effect is robust
    or just depends on simulation settings.
    """
    print("\n" + "="*60)
    print("HOLE 4: PARAMETER SENSITIVITY TEST")
    print("Is the effect robust or just a simulation artifact?")
    print("="*60)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create base galaxy
    positions, velocities, masses = create_disk_galaxy(
        num_stars=num_stars, galaxy_radius=10.0, device=device
    )
    positions, velocities, masses = positions.float(), velocities.float(), masses.float()

    # Test parameters
    softening_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    dt_values = [0.001, 0.005, 0.01, 0.02, 0.05]

    print("\n1. SOFTENING LENGTH SWEEP (dt=0.01, int4):")
    print("-" * 40)

    softening_results = []
    for soft in softening_values:
        class QuantSim(GalaxySimulation):
            def __init__(self, *args, quant_levels, **kwargs):
                self.quant_levels = quant_levels
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

        sim = QuantSim(positions.clone(), velocities.clone(), masses.clone(),
                       quant_levels=16, precision_mode=PrecisionMode.FLOAT32,
                       G=0.001, dt=0.01, softening=soft, device=device)

        initial_e = sim.get_total_energy()
        for _ in range(num_ticks):
            sim.step()
        final_e = sim.get_total_energy()

        drift = (final_e - initial_e) / abs(initial_e) * 100
        softening_results.append((soft, drift))
        print(f"  softening={soft:.2f}: {drift:+.4f}% drift")

    print("\n2. TIME STEP SWEEP (softening=0.1, int4):")
    print("-" * 40)

    dt_results = []
    for dt in dt_values:
        # Adjust ticks to keep total time similar
        adjusted_ticks = int(num_ticks * 0.01 / dt)

        class QuantSim(GalaxySimulation):
            def __init__(self, *args, quant_levels, **kwargs):
                self.quant_levels = quant_levels
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

        sim = QuantSim(positions.clone(), velocities.clone(), masses.clone(),
                       quant_levels=16, precision_mode=PrecisionMode.FLOAT32,
                       G=0.001, dt=dt, softening=0.1, device=device)

        initial_e = sim.get_total_energy()
        for _ in range(adjusted_ticks):
            sim.step()
        final_e = sim.get_total_energy()

        drift = (final_e - initial_e) / abs(initial_e) * 100
        dt_results.append((dt, drift))
        print(f"  dt={dt:.3f}: {drift:+.4f}% drift ({adjusted_ticks} ticks)")

    # Analysis
    print("\n" + "-"*40)
    print("PARAMETER SENSITIVITY ANALYSIS:")

    # Check if effect is consistent
    soft_drifts = [r[1] for r in softening_results]
    dt_drifts = [r[1] for r in dt_results]

    soft_variation = (max(soft_drifts) - min(soft_drifts)) / (abs(np.mean(soft_drifts)) + 0.01)
    dt_variation = (max(dt_drifts) - min(dt_drifts)) / (abs(np.mean(dt_drifts)) + 0.01)

    print(f"  Softening variation: {soft_variation:.1%}")
    print(f"  Time step variation: {dt_variation:.1%}")

    if soft_variation > 2.0 or dt_variation > 2.0:
        print("\n✗ Effect varies WILDLY with parameters")
        print("  This suggests it's a numerical artifact, not physics.")
    elif soft_variation < 0.5 and dt_variation < 0.5:
        print("\n✓ Effect is ROBUST to parameter changes")
        print("  This is a good sign - effect may be real.")
    else:
        print("\n? Effect shows moderate parameter sensitivity")
        print("  Further investigation needed.")

    return {"softening": softening_results, "dt": dt_results}


def plot_falsification_results(convergence, bullet, parameters, save_dir):
    """Create summary plots for all tests."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Convergence plot
    ax1 = axes[0, 0]
    if convergence:
        levels = [r[0] for r in convergence]
        drifts = [abs(r[2]) for r in convergence]
        ax1.loglog(levels, drifts, 'o-', linewidth=2, markersize=8, color='red')
        ax1.set_xlabel("Quantization Levels", fontsize=12)
        ax1.set_ylabel("Energy Drift (%)", fontsize=12)
        ax1.set_title("Hole 1: Convergence Test", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.01, color='green', linestyle='--', label='Convergence threshold')
        ax1.legend()

    # 2. Bullet cluster (simplified)
    ax2 = axes[0, 1]
    ax2.text(0.5, 0.5, "Bullet Cluster Test\n(See console output)",
             ha='center', va='center', fontsize=14, transform=ax2.transAxes)
    ax2.set_title("Hole 2: Bullet Cluster Test", fontsize=14)
    ax2.axis('off')

    # 3. Softening sweep
    ax3 = axes[1, 0]
    if parameters and "softening" in parameters:
        soft = [r[0] for r in parameters["softening"]]
        drift = [r[1] for r in parameters["softening"]]
        ax3.plot(soft, drift, 'o-', linewidth=2, markersize=8, color='blue')
        ax3.set_xlabel("Softening Length", fontsize=12)
        ax3.set_ylabel("Energy Drift (%)", fontsize=12)
        ax3.set_title("Hole 4a: Softening Sensitivity", fontsize=14)
        ax3.grid(True, alpha=0.3)

    # 4. Time step sweep
    ax4 = axes[1, 1]
    if parameters and "dt" in parameters:
        dt = [r[0] for r in parameters["dt"]]
        drift = [r[1] for r in parameters["dt"]]
        ax4.plot(dt, drift, 'o-', linewidth=2, markersize=8, color='purple')
        ax4.set_xlabel("Time Step (dt)", fontsize=12)
        ax4.set_ylabel("Energy Drift (%)", fontsize=12)
        ax4.set_title("Hole 4b: Time Step Sensitivity", fontsize=14)
        ax4.grid(True, alpha=0.3)

    plt.suptitle("Falsification Tests: Can the Hypothesis Survive?", fontsize=16, y=1.02)
    plt.tight_layout()

    save_path = Path(save_dir) / "falsification_tests.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved results to {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Falsification tests for the hypothesis")
    parser.add_argument("--test", "-t", type=str, default="all",
                        choices=["convergence", "bullet", "parameters", "all"])
    parser.add_argument("--stars", type=int, default=1500)
    parser.add_argument("--ticks", type=int, default=400)
    parser.add_argument("--output", type=str, default="output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    convergence_results = None
    bullet_results = None
    parameter_results = None

    if args.test in ["convergence", "all"]:
        convergence_results = test_convergence(args.stars, args.ticks, device)

    if args.test in ["bullet", "all"]:
        bullet_results = test_bullet_cluster(args.stars, args.ticks, device)

    if args.test in ["parameters", "all"]:
        parameter_results = test_parameter_sensitivity(args.stars, args.ticks, device)

    # Summary
    print("\n" + "="*60)
    print("FALSIFICATION TEST SUMMARY")
    print("="*60)
    print("""
For the "Dark Matter = Rounding Errors" hypothesis to survive:

1. CONVERGENCE: Effect should persist at some level even at high precision
   (Complete convergence to zero = just a numerical artifact)

2. BULLET CLUSTER: Should show mass/gravity separation in collisions
   (No separation = can't explain real observations)

3. PARAMETERS: Effect should be robust to simulation settings
   (Wild variation = just a simulation bug)

4. CMB: Would need to reproduce CMB power spectrum
   (Requires cosmological simulation - not tested here)
""")

    Path(args.output).mkdir(exist_ok=True)
    plot_falsification_results(convergence_results, bullet_results,
                               parameter_results, args.output)
    plt.show()


if __name__ == "__main__":
    main()
