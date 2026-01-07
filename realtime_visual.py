"""
REAL-TIME LOSSY GALAXY VISUALIZATION
=====================================

Watch the "broken" physics create dark matter in real-time.
See stars gain energy from nowhere as the math fails.

Features:
- Live galaxy rendering
- Side-by-side: Clean vs Broken physics
- Real-time energy meter
- Ghost force indicator
- Rotation curve updating live

Usage:
    python realtime_visual.py
    python realtime_visual.py --stars 2000 --mode compare
    python realtime_visual.py --stars 3000 --mode broken
"""

import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
from pathlib import Path

from galaxy import create_disk_galaxy
from simulation import GalaxySimulation
from quantization import PrecisionMode, _grid_quantize_safe
from metrics import compute_rotation_curve


class RealtimeGalaxyVisualizer:
    """Real-time galaxy simulation with live visualization."""

    def __init__(
        self,
        num_stars: int = 1500,
        mode: str = "compare",  # "compare", "broken", "clean"
        device: torch.device = None
    ):
        self.num_stars = num_stars
        self.mode = mode
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create galaxies
        self._create_galaxies()

        # Tracking
        self.tick = 0
        self.clean_energy_history = []
        self.broken_energy_history = []
        self.ghost_force_history = []

        # Setup visualization
        self._setup_figure()

    def _create_galaxies(self):
        """Create the galaxy simulations."""
        # Initial conditions (same for both)
        positions, velocities, masses = create_disk_galaxy(
            num_stars=self.num_stars,
            galaxy_radius=10.0,
            device=self.device
        )
        self.initial_positions = positions.float().clone()
        self.initial_velocities = velocities.float().clone()
        self.masses = masses.float()

        # Clean simulation (high precision)
        self.clean_sim = GalaxySimulation(
            positions.float().clone(),
            velocities.float().clone(),
            masses.float().clone(),
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001, dt=0.01, softening=0.1,
            device=self.device
        )
        self.clean_initial_energy = self.clean_sim.get_total_energy()

        # Broken simulation (quantized)
        class BrokenSim(GalaxySimulation):
            def __init__(self, *args, **kwargs):
                self.quant_levels = 16  # int4
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

        self.broken_sim = BrokenSim(
            positions.float().clone(),
            velocities.float().clone(),
            masses.float().clone(),
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001, dt=0.01, softening=0.1,
            device=self.device
        )
        self.broken_initial_energy = self.broken_sim.get_total_energy()

    def _setup_figure(self):
        """Setup the matplotlib figure."""
        if self.mode == "compare":
            self.fig = plt.figure(figsize=(16, 10))
            gs = self.fig.add_gridspec(3, 3, height_ratios=[2, 1, 1])

            # Galaxy views
            self.ax_clean = self.fig.add_subplot(gs[0, 0])
            self.ax_broken = self.fig.add_subplot(gs[0, 1])
            self.ax_diff = self.fig.add_subplot(gs[0, 2])

            # Energy plot
            self.ax_energy = self.fig.add_subplot(gs[1, :2])

            # Ghost force meter
            self.ax_ghost = self.fig.add_subplot(gs[1, 2])

            # Rotation curves
            self.ax_rotation = self.fig.add_subplot(gs[2, :])

        else:
            self.fig = plt.figure(figsize=(14, 8))
            gs = self.fig.add_gridspec(2, 2)

            self.ax_main = self.fig.add_subplot(gs[0, :])
            self.ax_energy = self.fig.add_subplot(gs[1, 0])
            self.ax_rotation = self.fig.add_subplot(gs[1, 1])

        self.fig.suptitle("LOSSY GALAXY - Real-Time Simulation", fontsize=16, fontweight='bold')

    def _update_plots(self, frame):
        """Update function for animation."""
        # Step simulations
        steps_per_frame = 5
        for _ in range(steps_per_frame):
            if self.mode in ["compare", "clean"]:
                self.clean_sim.step()
            if self.mode in ["compare", "broken"]:
                self.broken_sim.step()
            self.tick += 1

        # Get current energies
        if self.mode in ["compare", "clean"]:
            clean_energy = self.clean_sim.get_total_energy()
            clean_drift = (clean_energy - self.clean_initial_energy) / abs(self.clean_initial_energy) * 100
            self.clean_energy_history.append(clean_drift)

        if self.mode in ["compare", "broken"]:
            broken_energy = self.broken_sim.get_total_energy()
            broken_drift = (broken_energy - self.broken_initial_energy) / abs(self.broken_initial_energy) * 100
            self.broken_energy_history.append(broken_drift)

        if self.mode == "compare":
            ghost_force = broken_drift - clean_drift
            self.ghost_force_history.append(ghost_force)

        # Clear and redraw
        if self.mode == "compare":
            self._draw_compare_mode()
        else:
            self._draw_single_mode()

        return []

    def _draw_compare_mode(self):
        """Draw comparison view."""
        # Clean galaxy
        self.ax_clean.clear()
        pos_clean = self.clean_sim.positions.cpu().numpy()
        speeds_clean = np.sqrt((self.clean_sim.velocities.cpu().numpy() ** 2).sum(axis=1))
        self.ax_clean.scatter(pos_clean[:, 0], pos_clean[:, 1],
                              c=speeds_clean, cmap='viridis', s=1, alpha=0.7)
        self.ax_clean.set_xlim(-20, 20)
        self.ax_clean.set_ylim(-20, 20)
        self.ax_clean.set_title(f"CLEAN (float32)\nEnergy: {self.clean_energy_history[-1]:+.2f}%",
                                fontsize=12, color='green')
        self.ax_clean.set_aspect('equal')
        self.ax_clean.set_facecolor('black')

        # Broken galaxy
        self.ax_broken.clear()
        pos_broken = self.broken_sim.positions.cpu().numpy()
        speeds_broken = np.sqrt((self.broken_sim.velocities.cpu().numpy() ** 2).sum(axis=1))
        self.ax_broken.scatter(pos_broken[:, 0], pos_broken[:, 1],
                               c=speeds_broken, cmap='hot', s=1, alpha=0.7)
        self.ax_broken.set_xlim(-20, 20)
        self.ax_broken.set_ylim(-20, 20)
        self.ax_broken.set_title(f"BROKEN (int4)\nEnergy: {self.broken_energy_history[-1]:+.2f}%",
                                 fontsize=12, color='red')
        self.ax_broken.set_aspect('equal')
        self.ax_broken.set_facecolor('black')

        # Difference view
        self.ax_diff.clear()
        diff = pos_broken - pos_clean
        diff_mag = np.sqrt((diff ** 2).sum(axis=1))
        self.ax_diff.scatter(pos_clean[:, 0], pos_clean[:, 1],
                             c=diff_mag, cmap='plasma', s=2, alpha=0.8)
        self.ax_diff.set_xlim(-20, 20)
        self.ax_diff.set_ylim(-20, 20)
        self.ax_diff.set_title(f"DIVERGENCE\nMax: {diff_mag.max():.2f}", fontsize=12, color='purple')
        self.ax_diff.set_aspect('equal')
        self.ax_diff.set_facecolor('black')

        # Energy plot
        self.ax_energy.clear()
        ticks = range(len(self.clean_energy_history))
        self.ax_energy.plot(ticks, self.clean_energy_history, 'g-', label='Clean', linewidth=2)
        self.ax_energy.plot(ticks, self.broken_energy_history, 'r-', label='Broken', linewidth=2)
        self.ax_energy.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        self.ax_energy.set_xlabel("Ticks")
        self.ax_energy.set_ylabel("Energy Drift %")
        self.ax_energy.set_title("Energy Conservation", fontsize=12)
        self.ax_energy.legend(loc='upper left')
        self.ax_energy.set_facecolor('#1a1a2e')
        self.ax_energy.tick_params(colors='white')

        # Ghost force meter
        self.ax_ghost.clear()
        if self.ghost_force_history:
            current_ghost = self.ghost_force_history[-1]
            color = 'red' if current_ghost > 0 else 'blue'

            # Bar meter
            self.ax_ghost.barh([0], [current_ghost], color=color, height=0.5)
            self.ax_ghost.set_xlim(-10, 30)
            self.ax_ghost.axvline(x=0, color='white', linewidth=2)
            self.ax_ghost.set_title(f"GHOST FORCE\n{current_ghost:+.2f}%", fontsize=14, fontweight='bold',
                                    color='red' if current_ghost > 1 else 'white')
            self.ax_ghost.set_facecolor('#1a1a2e')
            self.ax_ghost.set_yticks([])

            # Add label
            if current_ghost > 5:
                self.ax_ghost.text(current_ghost/2, 0, "DARK MATTER!", ha='center', va='center',
                                   fontsize=12, fontweight='bold', color='yellow')

        # Rotation curves
        self.ax_rotation.clear()

        # Clean curve
        curve_clean = compute_rotation_curve(self.clean_sim.positions, self.clean_sim.velocities, num_bins=12)
        valid_c = ~np.isnan(curve_clean["velocities"])
        self.ax_rotation.plot(curve_clean["radii"][valid_c], curve_clean["velocities"][valid_c],
                              'go-', label='Clean', linewidth=2, markersize=4)

        # Broken curve
        curve_broken = compute_rotation_curve(self.broken_sim.positions, self.broken_sim.velocities, num_bins=12)
        valid_b = ~np.isnan(curve_broken["velocities"])
        self.ax_rotation.plot(curve_broken["radii"][valid_b], curve_broken["velocities"][valid_b],
                              'ro-', label='Broken', linewidth=2, markersize=4)

        # Keplerian reference
        r_ref = np.linspace(1, 15, 50)
        v_kep = 0.5 / np.sqrt(r_ref)
        self.ax_rotation.plot(r_ref, v_kep, 'w--', alpha=0.5, label='Keplerian (no DM)')

        self.ax_rotation.set_xlabel("Radius")
        self.ax_rotation.set_ylabel("Velocity")
        self.ax_rotation.set_title("Rotation Curves (Flat = Dark Matter Signature)", fontsize=12)
        self.ax_rotation.legend(loc='upper right')
        self.ax_rotation.set_facecolor('#1a1a2e')
        self.ax_rotation.tick_params(colors='white')
        self.ax_rotation.set_xlim(0, 15)
        self.ax_rotation.set_ylim(0, None)

        # Update suptitle with tick count
        self.fig.suptitle(f"LOSSY GALAXY - Tick {self.tick} | Stars: {self.num_stars}",
                         fontsize=16, fontweight='bold')

        plt.tight_layout()

    def _draw_single_mode(self):
        """Draw single galaxy view."""
        sim = self.broken_sim if self.mode == "broken" else self.clean_sim
        history = self.broken_energy_history if self.mode == "broken" else self.clean_energy_history
        color = 'hot' if self.mode == "broken" else 'viridis'
        label = "BROKEN (int4)" if self.mode == "broken" else "CLEAN (float32)"

        # Main galaxy view
        self.ax_main.clear()
        pos = sim.positions.cpu().numpy()
        speeds = np.sqrt((sim.velocities.cpu().numpy() ** 2).sum(axis=1))
        self.ax_main.scatter(pos[:, 0], pos[:, 1], c=speeds, cmap=color, s=2, alpha=0.8)
        self.ax_main.set_xlim(-25, 25)
        self.ax_main.set_ylim(-25, 25)
        self.ax_main.set_title(f"{label} | Tick {self.tick} | Energy: {history[-1]:+.2f}%",
                               fontsize=14, fontweight='bold')
        self.ax_main.set_aspect('equal')
        self.ax_main.set_facecolor('black')

        # Energy plot
        self.ax_energy.clear()
        self.ax_energy.plot(range(len(history)), history, 'r-' if self.mode == "broken" else 'g-',
                            linewidth=2)
        self.ax_energy.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        self.ax_energy.set_xlabel("Ticks")
        self.ax_energy.set_ylabel("Energy Drift %")
        self.ax_energy.set_title("Energy Conservation")

        # Rotation curve
        self.ax_rotation.clear()
        curve = compute_rotation_curve(sim.positions, sim.velocities, num_bins=12)
        valid = ~np.isnan(curve["velocities"])
        self.ax_rotation.plot(curve["radii"][valid], curve["velocities"][valid],
                              'o-', linewidth=2, markersize=6,
                              color='red' if self.mode == "broken" else 'green')

        r_ref = np.linspace(1, 15, 50)
        v_kep = 0.5 / np.sqrt(r_ref)
        self.ax_rotation.plot(r_ref, v_kep, 'k--', alpha=0.5, label='Keplerian')

        self.ax_rotation.set_xlabel("Radius")
        self.ax_rotation.set_ylabel("Velocity")
        self.ax_rotation.set_title("Rotation Curve")
        self.ax_rotation.legend()

        plt.tight_layout()

    def run(self, interval: int = 50, max_frames: int = None):
        """Run the real-time visualization."""
        print(f"\nStarting real-time visualization...")
        print(f"  Mode: {self.mode}")
        print(f"  Stars: {self.num_stars}")
        print(f"  Device: {self.device}")
        print("\nClose the window to stop.")

        # Initialize with first frame
        if self.mode == "compare":
            self.clean_energy_history.append(0)
            self.broken_energy_history.append(0)
            self.ghost_force_history.append(0)
        else:
            if self.mode == "broken":
                self.broken_energy_history.append(0)
            else:
                self.clean_energy_history.append(0)

        self.ani = FuncAnimation(
            self.fig,
            self._update_plots,
            frames=max_frames,
            interval=interval,
            blit=False,
            repeat=False
        )

        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Real-time Lossy Galaxy Visualization")
    parser.add_argument("--stars", type=int, default=1500, help="Number of stars")
    parser.add_argument("--mode", type=str, default="compare",
                        choices=["compare", "broken", "clean"],
                        help="Visualization mode")
    parser.add_argument("--interval", type=int, default=50,
                        help="Animation interval in ms (lower = faster)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    viz = RealtimeGalaxyVisualizer(
        num_stars=args.stars,
        mode=args.mode,
        device=device
    )

    viz.run(interval=args.interval)


if __name__ == "__main__":
    main()
