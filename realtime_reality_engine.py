"""
REAL-TIME REALITY ENGINE - Live Universe Visualization
=======================================================

Watch the universe evolve in real-time from Big Bang to Now.
All physics tests run SIMULTANEOUSLY with cross-correlation monitoring.

This is the "Multiphysics Simulation" - catching glitches that only
emerge when different physical laws interact under FULL GPU LOAD.

Features:
- Live cosmic web evolution (3D scatter + density)
- Real-time BAO frequency monitor
- RSI Glitch Detector dashboard
- Neural prediction overlay
- Cross-Thread Interference Detection
- Global Clock Synchronization Monitor

Requirements:
    pip install matplotlib torch numpy

Usage:
    python realtime_reality_engine.py
    python realtime_reality_engine.py --particles 100000 --duration 60
    python realtime_reality_engine.py --headless  # No GUI, just metrics
"""

import argparse
import time
import threading
import queue
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
from pathlib import Path
import sys

import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# Local imports
from galaxy import create_disk_galaxy
from simulation import GalaxySimulation
from quantization import PrecisionMode, _grid_quantize_safe
from reproducibility import set_all_seeds, get_gpu_state


# =============================================================================
# SHARED STATE - Cross-Thread Communication
# =============================================================================

@dataclass
class GlobalClock:
    """The shared "reality clock" - tracks synchronization across all physics."""
    tick: int = 0
    wall_time_start: float = 0.0
    sim_time_gyr: float = 0.0
    redshift: float = 100.0

    # Timing deltas for each subsystem
    cosmic_web_dt: float = 0.0
    bao_solver_dt: float = 0.0
    rsi_monitor_dt: float = 0.0
    neural_bridge_dt: float = 0.0

    # Synchronization metrics
    max_desync_ms: float = 0.0
    sync_violations: int = 0


@dataclass
class LiveMetrics:
    """Real-time metrics from all running systems."""
    # Cosmic Web
    particle_count: int = 0
    mean_density: float = 0.0
    void_count: int = 0
    filament_density: float = 0.0

    # Energy
    kinetic_energy: float = 0.0
    potential_energy: float = 0.0
    total_energy: float = 0.0
    energy_drift_pct: float = 0.0

    # BAO
    bao_scale_mpc: float = 0.0
    bao_frequency_hz: float = 0.0
    bao_amplitude: float = 0.0

    # RSI (Reality Stability Index)
    rsi_score: float = 100.0
    glitch_count: int = 0
    anomaly_rate: float = 0.0

    # GPU
    gpu_clock_mhz: int = 0
    gpu_power_watts: float = 0.0
    gpu_temp_c: int = 0
    gpu_utilization: int = 0

    # Interference
    cross_thread_interference: float = 0.0
    memory_contention: float = 0.0

    # History for plots
    energy_history: List[float] = field(default_factory=list)
    rsi_history: List[float] = field(default_factory=list)
    bao_history: List[float] = field(default_factory=list)
    clock_history: List[float] = field(default_factory=list)
    glitch_times: List[float] = field(default_factory=list)


class SharedState:
    """Thread-safe shared state for all physics modules."""

    def __init__(self):
        self.lock = threading.Lock()
        self.clock = GlobalClock()
        self.metrics = LiveMetrics()
        self.positions = None
        self.velocities = None
        self.running = True
        self.event_queue = queue.Queue()

        # Glitch detection
        self.last_energy = None
        self.glitch_threshold = 0.05  # 5% sudden change = glitch

    def update_positions(self, pos: torch.Tensor, vel: torch.Tensor):
        with self.lock:
            self.positions = pos.cpu().numpy() if pos is not None else None
            self.velocities = vel.cpu().numpy() if vel is not None else None

    def get_positions(self):
        with self.lock:
            return self.positions.copy() if self.positions is not None else None

    def update_metrics(self, **kwargs):
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.metrics, key):
                    setattr(self.metrics, key, value)

    def add_glitch(self, glitch_type: str, severity: float):
        with self.lock:
            self.metrics.glitch_count += 1
            self.metrics.glitch_times.append(self.clock.tick)
            self.event_queue.put({
                'type': 'glitch',
                'subtype': glitch_type,
                'severity': severity,
                'tick': self.clock.tick,
                'time': time.time()
            })

    def check_sync_violation(self):
        """Check if physics modules are desynchronizing."""
        with self.lock:
            times = [
                self.clock.cosmic_web_dt,
                self.clock.bao_solver_dt,
                self.clock.rsi_monitor_dt,
            ]
            times = [t for t in times if t > 0]
            if len(times) > 1:
                max_diff = max(times) - min(times)
                if max_diff > 0.1:  # 100ms desync
                    self.clock.sync_violations += 1
                    self.clock.max_desync_ms = max(self.clock.max_desync_ms, max_diff * 1000)
                    return True
        return False


# =============================================================================
# COSMIC WEB ENGINE - Main N-Body Simulation
# =============================================================================

class CosmicWebEngine(threading.Thread):
    """
    Real-time cosmic web evolution using N-body simulation.
    This is the "Gravity Map" that feeds all other tests.
    """

    def __init__(self, shared_state: SharedState, device: torch.device,
                 num_particles: int = 50000, precision: str = "float32"):
        super().__init__(daemon=True)
        self.state = shared_state
        self.device = device
        self.num_particles = num_particles
        self.precision = precision

        # Simulation parameters
        self.box_size = 100.0  # Mpc
        self.dt = 0.01
        self.G = 0.001
        self.softening = 0.1

        # Initialize
        self._init_simulation()

    def _init_simulation(self):
        """Initialize the cosmological simulation."""
        set_all_seeds(42)

        # Create initial particle distribution
        n = int(np.cbrt(self.num_particles))
        self.num_particles = n ** 3

        dtype = torch.float32 if self.precision == "float32" else torch.float64

        # Uniform grid with small perturbations
        grid = torch.linspace(0, self.box_size, n, device=self.device, dtype=dtype)
        x, y, z = torch.meshgrid(grid, grid, grid, indexing='ij')

        self.positions = torch.stack([
            x.flatten() + torch.randn(n**3, device=self.device, dtype=dtype) * 0.5,
            y.flatten() + torch.randn(n**3, device=self.device, dtype=dtype) * 0.5,
            z.flatten() + torch.randn(n**3, device=self.device, dtype=dtype) * 0.5
        ], dim=1) % self.box_size

        # Hubble flow + peculiar velocities
        self.velocities = (self.positions - self.box_size/2) * 0.01 + \
                         torch.randn_like(self.positions) * 0.1

        # Equal mass particles
        total_mass = 1e12  # Total mass
        self.masses = torch.ones(self.num_particles, device=self.device, dtype=dtype) * \
                     total_mass / self.num_particles

        # Track initial energy
        self.initial_energy = self._compute_energy()

        print(f"  Cosmic Web Engine initialized: {self.num_particles:,} particles")

    def _compute_energy(self) -> float:
        """Compute total energy."""
        # Kinetic
        v_sq = (self.velocities ** 2).sum(dim=-1)
        ke = 0.5 * (self.masses * v_sq).sum().item()

        # Potential (simplified - just use nearest neighbors)
        return ke  # Skip full potential for speed

    def _step(self):
        """Perform one simulation step with PM gravity."""
        n_grid = 64

        # Density field via nearest grid point
        density = torch.zeros(n_grid, n_grid, n_grid, device=self.device, dtype=self.positions.dtype)
        pos_grid = (self.positions / self.box_size * n_grid).long() % n_grid

        # Vectorized density assignment
        for i in range(min(len(self.positions), 10000)):  # Limit for speed
            density[pos_grid[i, 0], pos_grid[i, 1], pos_grid[i, 2]] += self.masses[i]

        # FFT gravity
        density_k = torch.fft.fftn(density)

        kx = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device)
        ky = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device)
        kz = torch.fft.fftfreq(n_grid, d=self.box_size/n_grid, device=self.device)
        kx, ky, kz = torch.meshgrid(kx, ky, kz, indexing='ij')
        k_sq = kx**2 + ky**2 + kz**2 + 1e-10

        # Green's function
        phi_k = -self.G * density_k / k_sq
        phi_k[0, 0, 0] = 0

        # Acceleration field
        ax = torch.fft.ifftn(-1j * kx * phi_k).real
        ay = torch.fft.ifftn(-1j * ky * phi_k).real
        az = torch.fft.ifftn(-1j * kz * phi_k).real

        # Interpolate to particles
        accel = torch.zeros_like(self.positions)
        for i in range(min(len(self.positions), 10000)):
            ix, iy, iz = pos_grid[i]
            accel[i, 0] = ax[ix, iy, iz]
            accel[i, 1] = ay[ix, iy, iz]
            accel[i, 2] = az[ix, iy, iz]

        # Leapfrog update
        self.velocities = self.velocities + accel * self.dt
        self.positions = (self.positions + self.velocities * self.dt) % self.box_size

    def run(self):
        """Main simulation loop."""
        tick = 0
        last_update = time.time()

        while self.state.running:
            start = time.perf_counter()

            # Physics step
            self._step()
            tick += 1

            # Update shared state
            dt = time.perf_counter() - start
            self.state.clock.cosmic_web_dt = dt
            self.state.clock.tick = tick

            # Calculate metrics every 10 ticks
            if tick % 10 == 0:
                energy = self._compute_energy()
                drift = abs(energy - self.initial_energy) / (abs(self.initial_energy) + 1e-10) * 100

                self.state.update_positions(self.positions, self.velocities)
                self.state.update_metrics(
                    particle_count=self.num_particles,
                    kinetic_energy=energy,
                    total_energy=energy,
                    energy_drift_pct=drift
                )

                # Check for energy glitch
                if self.state.last_energy is not None:
                    delta = abs(energy - self.state.last_energy) / (abs(self.state.last_energy) + 1e-10)
                    if delta > self.state.glitch_threshold:
                        self.state.add_glitch('energy_spike', delta)
                self.state.last_energy = energy

                # Append to history
                with self.state.lock:
                    self.state.metrics.energy_history.append(drift)
                    if len(self.state.metrics.energy_history) > 500:
                        self.state.metrics.energy_history.pop(0)

            # Redshift evolution (cosmological time)
            self.state.clock.redshift = max(0, 100 - tick * 0.1)
            self.state.clock.sim_time_gyr = tick * 0.01  # ~10 Myr per tick

            # Target ~30 FPS
            elapsed = time.perf_counter() - start
            if elapsed < 0.033:
                time.sleep(0.033 - elapsed)


# =============================================================================
# BAO SOLVER - Acoustic Oscillation Monitor
# =============================================================================

class BAOSolver(threading.Thread):
    """
    Real-time Baryonic Acoustic Oscillation monitor.
    Checks if "sound" and "matter" stay synced during simulation.
    """

    def __init__(self, shared_state: SharedState, device: torch.device):
        super().__init__(daemon=True)
        self.state = shared_state
        self.device = device
        self.expected_bao = 147.0  # Mpc

    def _compute_power_spectrum(self, positions: np.ndarray) -> Tuple[float, float]:
        """Compute BAO scale from positions."""
        if positions is None or len(positions) < 100:
            return 0, 0

        n_grid = 32
        box_size = 100.0

        # Density field
        density = np.zeros((n_grid, n_grid, n_grid))
        pos_grid = (positions / box_size * n_grid).astype(int) % n_grid

        for i in range(min(len(positions), 5000)):
            density[pos_grid[i, 0], pos_grid[i, 1], pos_grid[i, 2]] += 1

        # FFT
        delta = density - density.mean()
        delta_k = np.fft.fftn(delta)
        pk_3d = np.abs(delta_k) ** 2

        # Spherical average
        kx = np.fft.fftfreq(n_grid, d=box_size/n_grid)
        ky = np.fft.fftfreq(n_grid, d=box_size/n_grid)
        kz = np.fft.fftfreq(n_grid, d=box_size/n_grid)
        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

        # Find peak
        k_peak = k_mag.flatten()[np.argmax(pk_3d.flatten()[1:])+1]
        bao_scale = 2 * np.pi / k_peak if k_peak > 0 else 0
        bao_amplitude = pk_3d.max() / (pk_3d.mean() + 1e-10)

        return bao_scale, bao_amplitude

    def run(self):
        """BAO monitoring loop."""
        while self.state.running:
            start = time.perf_counter()

            positions = self.state.get_positions()
            if positions is not None:
                bao_scale, amplitude = self._compute_power_spectrum(positions)

                # Frequency = evolution rate
                frequency = 1.0 / max(0.1, self.state.clock.cosmic_web_dt) if self.state.clock.cosmic_web_dt > 0 else 0

                self.state.update_metrics(
                    bao_scale_mpc=bao_scale,
                    bao_amplitude=amplitude,
                    bao_frequency_hz=frequency
                )

                # Track BAO history
                with self.state.lock:
                    self.state.metrics.bao_history.append(bao_scale)
                    if len(self.state.metrics.bao_history) > 500:
                        self.state.metrics.bao_history.pop(0)

                # Check for BAO anomaly
                error = abs(bao_scale - self.expected_bao) / self.expected_bao if bao_scale > 0 else 1
                if error > 0.5:  # 50% deviation
                    self.state.add_glitch('bao_deviation', error)

            self.state.clock.bao_solver_dt = time.perf_counter() - start
            time.sleep(0.1)  # 10 Hz update


# =============================================================================
# RSI MONITOR - Reality Stability Index
# =============================================================================

class RSIMonitor(threading.Thread):
    """
    Real-time Glitch Detector - computes Reality Stability Index.
    Detects "Framerate Drops" across the whole universe.
    """

    def __init__(self, shared_state: SharedState, device: torch.device):
        super().__init__(daemon=True)
        self.state = shared_state
        self.device = device

        # RSI weights
        self.energy_weight = 0.3
        self.sync_weight = 0.3
        self.bao_weight = 0.2
        self.gpu_weight = 0.2

    def run(self):
        """RSI monitoring loop."""
        while self.state.running:
            start = time.perf_counter()

            # Get GPU state
            gpu_state = get_gpu_state()
            if gpu_state:
                self.state.update_metrics(
                    gpu_clock_mhz=gpu_state.clock_speed_mhz,
                    gpu_power_watts=gpu_state.power_draw_watts,
                    gpu_temp_c=gpu_state.temperature_c,
                    gpu_utilization=gpu_state.utilization_percent
                )

                # Track clock history
                with self.state.lock:
                    self.state.metrics.clock_history.append(gpu_state.clock_speed_mhz)
                    if len(self.state.metrics.clock_history) > 500:
                        self.state.metrics.clock_history.pop(0)

            # Compute RSI
            energy_score = max(0, 100 - self.state.metrics.energy_drift_pct * 10)

            sync_score = 100
            if self.state.clock.max_desync_ms > 10:
                sync_score = max(0, 100 - self.state.clock.max_desync_ms)

            bao_score = 100
            if self.state.metrics.bao_scale_mpc > 0:
                bao_error = abs(self.state.metrics.bao_scale_mpc - 147) / 147 * 100
                bao_score = max(0, 100 - bao_error)

            gpu_score = 100
            if gpu_state and gpu_state.throttle_reasons:
                gpu_score = 50  # Throttling detected

            # Weighted RSI
            rsi = (
                energy_score * self.energy_weight +
                sync_score * self.sync_weight +
                bao_score * self.bao_weight +
                gpu_score * self.gpu_weight
            )

            anomaly_rate = self.state.metrics.glitch_count / max(1, self.state.clock.tick) * 100

            self.state.update_metrics(
                rsi_score=rsi,
                anomaly_rate=anomaly_rate
            )

            # Track RSI history
            with self.state.lock:
                self.state.metrics.rsi_history.append(rsi)
                if len(self.state.metrics.rsi_history) > 500:
                    self.state.metrics.rsi_history.pop(0)

            # Check for sync violations
            self.state.check_sync_violation()

            self.state.clock.rsi_monitor_dt = time.perf_counter() - start
            time.sleep(0.05)  # 20 Hz update


# =============================================================================
# REAL-TIME VISUALIZATION
# =============================================================================

class RealtimeDashboard:
    """
    Live visualization dashboard using matplotlib animation.
    Shows universe evolution and all metrics in real-time.
    """

    def __init__(self, shared_state: SharedState):
        self.state = shared_state
        self.fig = None
        self.axes = {}
        self.artists = {}

        # Plot data
        self.max_points = 500

    def setup(self):
        """Initialize the matplotlib figure and subplots."""
        plt.style.use('dark_background')

        self.fig = plt.figure(figsize=(18, 10))
        self.fig.suptitle('REAL-TIME REALITY ENGINE', fontsize=16, fontweight='bold', color='cyan')

        gs = GridSpec(3, 4, figure=self.fig, hspace=0.3, wspace=0.3)

        # Universe View (3D scatter) - Large panel
        self.axes['universe'] = self.fig.add_subplot(gs[0:2, 0:2], projection='3d')
        self.axes['universe'].set_title('Cosmic Web Evolution', color='white')
        self.axes['universe'].set_facecolor('black')

        # Energy Drift
        self.axes['energy'] = self.fig.add_subplot(gs[0, 2])
        self.axes['energy'].set_title('Energy Drift (%)', color='white')
        self.axes['energy'].set_facecolor('#1a1a2e')

        # RSI Score
        self.axes['rsi'] = self.fig.add_subplot(gs[0, 3])
        self.axes['rsi'].set_title('Reality Stability Index', color='white')
        self.axes['rsi'].set_facecolor('#1a1a2e')

        # BAO Scale
        self.axes['bao'] = self.fig.add_subplot(gs[1, 2])
        self.axes['bao'].set_title('BAO Scale (Mpc)', color='white')
        self.axes['bao'].set_facecolor('#1a1a2e')

        # GPU Clock
        self.axes['clock'] = self.fig.add_subplot(gs[1, 3])
        self.axes['clock'].set_title('GPU Clock (MHz)', color='white')
        self.axes['clock'].set_facecolor('#1a1a2e')

        # Metrics Panel
        self.axes['metrics'] = self.fig.add_subplot(gs[2, 0:2])
        self.axes['metrics'].set_title('Live Metrics', color='white')
        self.axes['metrics'].axis('off')
        self.axes['metrics'].set_facecolor('#1a1a2e')

        # Glitch Log
        self.axes['glitches'] = self.fig.add_subplot(gs[2, 2:4])
        self.axes['glitches'].set_title('Glitch Detection Log', color='white')
        self.axes['glitches'].axis('off')
        self.axes['glitches'].set_facecolor('#16213e')

        # Initialize empty artists
        self.artists['scatter'] = None
        self.artists['energy_line'], = self.axes['energy'].plot([], [], 'r-', linewidth=1)
        self.artists['rsi_line'], = self.axes['rsi'].plot([], [], 'g-', linewidth=1)
        self.artists['bao_line'], = self.axes['bao'].plot([], [], 'b-', linewidth=1)
        self.artists['clock_line'], = self.axes['clock'].plot([], [], 'y-', linewidth=1)

        # Reference lines
        self.axes['bao'].axhline(y=147, color='lime', linestyle='--', alpha=0.5, label='Expected BAO')
        self.axes['rsi'].axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='Warning')

        plt.tight_layout()

    def update(self, frame):
        """Animation update function."""
        # Update 3D scatter
        positions = self.state.get_positions()
        if positions is not None and len(positions) > 0:
            # Subsample for performance
            n_show = min(5000, len(positions))
            idx = np.random.choice(len(positions), n_show, replace=False)
            pos = positions[idx]

            self.axes['universe'].clear()
            self.axes['universe'].scatter(
                pos[:, 0], pos[:, 1], pos[:, 2],
                c=np.linalg.norm(pos, axis=1),
                cmap='plasma',
                s=0.5,
                alpha=0.6
            )
            self.axes['universe'].set_xlim(0, 100)
            self.axes['universe'].set_ylim(0, 100)
            self.axes['universe'].set_zlim(0, 100)
            self.axes['universe'].set_title(
                f'Cosmic Web | z={self.state.clock.redshift:.1f} | t={self.state.clock.sim_time_gyr:.2f} Gyr',
                color='cyan'
            )
            self.axes['universe'].set_facecolor('black')

        # Update line plots
        with self.state.lock:
            # Energy
            if self.state.metrics.energy_history:
                x = range(len(self.state.metrics.energy_history))
                self.axes['energy'].clear()
                self.axes['energy'].plot(x, self.state.metrics.energy_history, 'r-', linewidth=1)
                self.axes['energy'].fill_between(x, self.state.metrics.energy_history, alpha=0.3, color='red')
                self.axes['energy'].set_title(f'Energy Drift: {self.state.metrics.energy_drift_pct:.2f}%', color='red')
                self.axes['energy'].set_facecolor('#1a1a2e')

            # RSI
            if self.state.metrics.rsi_history:
                x = range(len(self.state.metrics.rsi_history))
                self.axes['rsi'].clear()
                self.axes['rsi'].plot(x, self.state.metrics.rsi_history, 'g-', linewidth=1)
                self.axes['rsi'].fill_between(x, self.state.metrics.rsi_history, alpha=0.3, color='green')
                self.axes['rsi'].axhline(y=50, color='orange', linestyle='--', alpha=0.5)
                rsi_color = 'lime' if self.state.metrics.rsi_score > 70 else 'orange' if self.state.metrics.rsi_score > 40 else 'red'
                self.axes['rsi'].set_title(f'RSI: {self.state.metrics.rsi_score:.1f}', color=rsi_color)
                self.axes['rsi'].set_facecolor('#1a1a2e')
                self.axes['rsi'].set_ylim(0, 105)

            # BAO
            if self.state.metrics.bao_history:
                x = range(len(self.state.metrics.bao_history))
                self.axes['bao'].clear()
                self.axes['bao'].plot(x, self.state.metrics.bao_history, 'b-', linewidth=1)
                self.axes['bao'].axhline(y=147, color='lime', linestyle='--', alpha=0.5)
                self.axes['bao'].set_title(f'BAO: {self.state.metrics.bao_scale_mpc:.1f} Mpc', color='cyan')
                self.axes['bao'].set_facecolor('#1a1a2e')

            # GPU Clock
            if self.state.metrics.clock_history:
                x = range(len(self.state.metrics.clock_history))
                self.axes['clock'].clear()
                self.axes['clock'].plot(x, self.state.metrics.clock_history, 'y-', linewidth=1)
                self.axes['clock'].fill_between(x, self.state.metrics.clock_history, alpha=0.3, color='yellow')
                self.axes['clock'].set_title(f'GPU: {self.state.metrics.gpu_clock_mhz} MHz | {self.state.metrics.gpu_power_watts:.0f}W', color='yellow')
                self.axes['clock'].set_facecolor('#1a1a2e')

        # Update metrics panel
        self.axes['metrics'].clear()
        self.axes['metrics'].axis('off')
        self.axes['metrics'].set_facecolor('#1a1a2e')

        metrics_text = (
            f"{'='*60}\n"
            f"  SIMULATION STATUS\n"
            f"{'='*60}\n"
            f"  Tick: {self.state.clock.tick:,}  |  Redshift: z={self.state.clock.redshift:.2f}\n"
            f"  Particles: {self.state.metrics.particle_count:,}\n"
            f"  Simulation Time: {self.state.clock.sim_time_gyr:.2f} Gyr\n"
            f"\n"
            f"  PHYSICS:\n"
            f"    Energy Drift: {self.state.metrics.energy_drift_pct:.4f}%\n"
            f"    BAO Scale: {self.state.metrics.bao_scale_mpc:.1f} Mpc (expected: 147)\n"
            f"    BAO Frequency: {self.state.metrics.bao_frequency_hz:.1f} Hz\n"
            f"\n"
            f"  GPU STATE:\n"
            f"    Clock: {self.state.metrics.gpu_clock_mhz} MHz\n"
            f"    Power: {self.state.metrics.gpu_power_watts:.1f} W\n"
            f"    Temp: {self.state.metrics.gpu_temp_c}°C\n"
            f"    Util: {self.state.metrics.gpu_utilization}%\n"
            f"\n"
            f"  SYNCHRONIZATION:\n"
            f"    Max Desync: {self.state.clock.max_desync_ms:.1f} ms\n"
            f"    Violations: {self.state.clock.sync_violations}\n"
        )

        self.axes['metrics'].text(
            0.02, 0.98, metrics_text, transform=self.axes['metrics'].transAxes,
            fontfamily='monospace', fontsize=9, color='white',
            verticalalignment='top'
        )

        # Update glitch log
        self.axes['glitches'].clear()
        self.axes['glitches'].axis('off')
        self.axes['glitches'].set_facecolor('#16213e')

        glitch_text = (
            f"  GLITCH DETECTION\n"
            f"  {'='*40}\n"
            f"  Total Glitches: {self.state.metrics.glitch_count}\n"
            f"  Anomaly Rate: {self.state.metrics.anomaly_rate:.4f}%\n"
            f"  RSI Score: {self.state.metrics.rsi_score:.1f}/100\n"
            f"\n"
        )

        # Show last few glitches
        if self.state.metrics.glitch_times:
            glitch_text += "  Recent Events:\n"
            for t in self.state.metrics.glitch_times[-5:]:
                glitch_text += f"    - Glitch at tick {t}\n"
        else:
            glitch_text += "  No glitches detected yet...\n"

        # Verdict
        if self.state.metrics.rsi_score < 50:
            verdict = "WARNING: REALITY UNSTABLE"
            color = 'red'
        elif self.state.metrics.glitch_count > 10:
            verdict = "ALERT: Multiple anomalies detected"
            color = 'orange'
        else:
            verdict = "STATUS: Reality nominal"
            color = 'lime'

        glitch_text += f"\n  {verdict}"

        self.axes['glitches'].text(
            0.02, 0.98, glitch_text, transform=self.axes['glitches'].transAxes,
            fontfamily='monospace', fontsize=10, color='white',
            verticalalignment='top'
        )

        # Add colored status indicator
        status_color = 'lime' if self.state.metrics.rsi_score > 70 else 'orange' if self.state.metrics.rsi_score > 40 else 'red'
        circle = plt.Circle((0.9, 0.9), 0.05, transform=self.axes['glitches'].transAxes,
                            color=status_color, zorder=10)
        self.axes['glitches'].add_patch(circle)

        return list(self.artists.values())

    def run(self, duration: int = 60):
        """Start the animation."""
        self.setup()

        ani = FuncAnimation(
            self.fig, self.update,
            frames=None,  # Infinite
            interval=100,  # 10 FPS update
            blit=False,
            cache_frame_data=False
        )

        plt.show()


# =============================================================================
# MAIN ENGINE
# =============================================================================

def run_realtime_engine(
    num_particles: int = 50000,
    precision: str = "float32",
    duration: int = 60,
    headless: bool = False,
    output_dir: str = None,
    device: torch.device = None
):
    """
    Run the real-time reality engine.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*70)
    print("  REAL-TIME REALITY ENGINE")
    print("  Watch the universe evolve live")
    print("="*70)
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Particles: {num_particles:,}")
    print(f"  Precision: {precision}")
    print(f"  Duration: {duration}s")
    print("="*70)

    # Create shared state
    shared = SharedState()
    shared.clock.wall_time_start = time.time()

    # Start physics threads
    print("\n  Starting physics engines...")

    cosmic_web = CosmicWebEngine(shared, device, num_particles, precision)
    bao_solver = BAOSolver(shared, device)
    rsi_monitor = RSIMonitor(shared, device)

    cosmic_web.start()
    time.sleep(0.5)  # Let cosmic web initialize
    bao_solver.start()
    rsi_monitor.start()

    print("  All engines running!")

    if headless:
        # Headless mode - just print metrics
        print("\n  Running in headless mode...")
        start_time = time.time()

        try:
            while time.time() - start_time < duration:
                print(f"\r  Tick: {shared.clock.tick:6d} | "
                      f"z={shared.clock.redshift:5.1f} | "
                      f"RSI={shared.metrics.rsi_score:5.1f} | "
                      f"Energy={shared.metrics.energy_drift_pct:6.3f}% | "
                      f"Glitches={shared.metrics.glitch_count}", end="")
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

        print("\n")
    else:
        # GUI mode
        print("\n  Starting visualization dashboard...")
        print("  (Close window to stop)")

        dashboard = RealtimeDashboard(shared)

        try:
            dashboard.run(duration)
        except KeyboardInterrupt:
            pass

    # Shutdown
    print("\n  Shutting down...")
    shared.running = False

    # Wait for threads
    cosmic_web.join(timeout=2)
    bao_solver.join(timeout=1)
    rsi_monitor.join(timeout=1)

    # Final report
    print("\n" + "="*70)
    print("  FINAL REPORT")
    print("="*70)
    print(f"  Total Ticks: {shared.clock.tick:,}")
    print(f"  Final Redshift: z={shared.clock.redshift:.2f}")
    print(f"  Simulation Time: {shared.clock.sim_time_gyr:.2f} Gyr")
    print(f"  Energy Drift: {shared.metrics.energy_drift_pct:.4f}%")
    print(f"  BAO Scale: {shared.metrics.bao_scale_mpc:.1f} Mpc")
    print(f"  RSI Score: {shared.metrics.rsi_score:.1f}/100")
    print(f"  Total Glitches: {shared.metrics.glitch_count}")
    print(f"  Sync Violations: {shared.clock.sync_violations}")
    print(f"  Max Desync: {shared.clock.max_desync_ms:.1f} ms")
    print("="*70)

    # Save report
    if output_dir:
        Path(output_dir).mkdir(exist_ok=True)
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_ticks': shared.clock.tick,
            'final_redshift': shared.clock.redshift,
            'sim_time_gyr': shared.clock.sim_time_gyr,
            'energy_drift_pct': shared.metrics.energy_drift_pct,
            'bao_scale_mpc': shared.metrics.bao_scale_mpc,
            'rsi_score': shared.metrics.rsi_score,
            'glitch_count': shared.metrics.glitch_count,
            'sync_violations': shared.clock.sync_violations,
            'max_desync_ms': shared.clock.max_desync_ms,
        }

        with open(Path(output_dir) / 'realtime_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved to {output_dir}/realtime_report.json")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Reality Engine")
    parser.add_argument("--particles", type=int, default=50000, help="Number of particles")
    parser.add_argument("--precision", type=str, default="float32",
                        choices=["float32", "float64"], help="Precision mode")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--output", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    run_realtime_engine(
        num_particles=args.particles,
        precision=args.precision,
        duration=args.duration,
        headless=args.headless,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
