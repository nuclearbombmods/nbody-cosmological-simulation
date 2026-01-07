"""
Hardware Leak Test - Looking for "Parasitic" Energy

Tests the hypothesis that unexplained energy consumption could indicate
"hidden computation" - like the parent simulation leaking into ours.

We monitor:
1. GPU power draw (via pynvml for NVIDIA)
2. GPU utilization
3. Memory usage
4. Computation complexity

If we find energy spikes that DON'T correlate with GPU utilization,
that's "unexplained work" - potentially the simulation overhead.

The idea: "Broken math" (quantized) might require MORE or LESS
hidden work than "clean math" (float64). The difference could
reveal the simulation's overhead.

Usage:
    python hardware_leak_test.py
    python hardware_leak_test.py --samples 50 --duration 60

Requirements:
    pip install pynvml psutil
"""

import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import threading
import queue

# Try to import monitoring libraries
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False
    print("Warning: pynvml not installed. GPU power monitoring disabled.")
    print("Install with: pip install pynvml")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. CPU monitoring disabled.")
    print("Install with: pip install psutil")

from galaxy import create_disk_galaxy
from simulation import GalaxySimulation
from quantization import PrecisionMode, _grid_quantize_safe


@dataclass
class HardwareSample:
    """A single hardware measurement."""
    timestamp: float
    gpu_power_watts: Optional[float]
    gpu_utilization: Optional[float]
    gpu_memory_used: Optional[float]
    cpu_percent: Optional[float]
    ram_percent: Optional[float]
    computation_type: str
    ops_per_second: float


class HardwareMonitor:
    """Monitor GPU and CPU metrics in a background thread."""

    def __init__(self):
        self.samples = []
        self.running = False
        self.current_label = "idle"
        self.current_ops = 0.0
        self._thread = None
        self._lock = threading.Lock()

        # Initialize NVML
        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.has_gpu = True
                gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
                print(f"GPU monitoring enabled: {gpu_name}")
            except Exception as e:
                print(f"GPU monitoring failed: {e}")
                self.has_gpu = False
        else:
            self.has_gpu = False

    def start(self):
        """Start background monitoring."""
        self.running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def set_label(self, label: str, ops: float = 0.0):
        """Set the current computation label."""
        with self._lock:
            self.current_label = label
            self.current_ops = ops

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            sample = self._take_sample()
            with self._lock:
                self.samples.append(sample)
            time.sleep(0.1)  # 10 Hz sampling

    def _take_sample(self) -> HardwareSample:
        """Take a single hardware measurement."""
        gpu_power = None
        gpu_util = None
        gpu_mem = None
        cpu_pct = None
        ram_pct = None

        # GPU metrics
        if self.has_gpu:
            try:
                power_info = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                gpu_power = power_info / 1000.0  # mW to W

                util_info = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_util = util_info.gpu

                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_mem = mem_info.used / (1024 ** 3)  # GB
            except Exception:
                pass

        # CPU metrics
        if HAS_PSUTIL:
            try:
                cpu_pct = psutil.cpu_percent()
                ram_pct = psutil.virtual_memory().percent
            except Exception:
                pass

        with self._lock:
            label = self.current_label
            ops = self.current_ops

        return HardwareSample(
            timestamp=time.time(),
            gpu_power_watts=gpu_power,
            gpu_utilization=gpu_util,
            gpu_memory_used=gpu_mem,
            cpu_percent=cpu_pct,
            ram_percent=ram_pct,
            computation_type=label,
            ops_per_second=ops
        )

    def get_samples(self) -> list[HardwareSample]:
        """Get all collected samples."""
        with self._lock:
            return list(self.samples)

    def cleanup(self):
        """Cleanup NVML."""
        if HAS_NVML and self.has_gpu:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def run_computation_pattern(
    pattern: str,
    duration: float,
    monitor: HardwareMonitor,
    device: torch.device
) -> dict:
    """
    Run a specific computation pattern and monitor hardware.
    """
    num_stars = 2000

    # Create galaxy
    positions, velocities, masses = create_disk_galaxy(
        num_stars=num_stars,
        galaxy_radius=10.0,
        device=device
    )
    positions = positions.float()
    velocities = velocities.float()
    masses = masses.float()

    start_time = time.time()
    tick_count = 0
    ops_estimate = 0

    if pattern == "idle":
        # Just sit there
        monitor.set_label("idle", 0)
        time.sleep(duration)

    elif pattern == "float64_clean":
        # High precision, clean math
        sim = GalaxySimulation(
            positions.double(), velocities.double(), masses.double(),
            precision_mode=PrecisionMode.FLOAT64,
            G=0.001, dt=0.01, softening=0.1, device=device
        )
        monitor.set_label("float64_clean", num_stars ** 2)

        while time.time() - start_time < duration:
            sim.step()
            tick_count += 1
            ops_estimate = tick_count * num_stars ** 2 / (time.time() - start_time)
            monitor.set_label("float64_clean", ops_estimate)

    elif pattern == "float32_standard":
        # Standard GPU precision
        sim = GalaxySimulation(
            positions, velocities, masses,
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001, dt=0.01, softening=0.1, device=device
        )
        monitor.set_label("float32_standard", num_stars ** 2)

        while time.time() - start_time < duration:
            sim.step()
            tick_count += 1
            ops_estimate = tick_count * num_stars ** 2 / (time.time() - start_time)
            monitor.set_label("float32_standard", ops_estimate)

    elif pattern == "int4_broken":
        # Heavily quantized "broken" math
        class BrokenSim(GalaxySimulation):
            def __init__(self, *args, **kwargs):
                self.quant_levels = 16
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

        sim = BrokenSim(
            positions, velocities, masses,
            precision_mode=PrecisionMode.FLOAT32,
            G=0.001, dt=0.01, softening=0.1, device=device
        )
        monitor.set_label("int4_broken", num_stars ** 2)

        while time.time() - start_time < duration:
            sim.step()
            tick_count += 1
            ops_estimate = tick_count * num_stars ** 2 / (time.time() - start_time)
            monitor.set_label("int4_broken", ops_estimate)

    elif pattern == "recursive_stress":
        # Maximum stress - nested operations
        monitor.set_label("recursive_stress", num_stars ** 2 * 10)

        while time.time() - start_time < duration:
            # Create temporary tensors, do nested ops
            for _ in range(10):
                temp = torch.randn(num_stars, num_stars, device=device)
                temp = torch.matmul(temp, temp.T)
                temp = torch.log(torch.abs(temp) + 1)
                temp = torch.exp(-temp)
                del temp

            tick_count += 1
            ops_estimate = tick_count * num_stars ** 2 * 10 / (time.time() - start_time)
            monitor.set_label("recursive_stress", ops_estimate)

    elif pattern == "memory_thrash":
        # Allocate/deallocate memory rapidly
        monitor.set_label("memory_thrash", 0)

        while time.time() - start_time < duration:
            # Allocate big tensors
            tensors = [torch.randn(1000, 1000, device=device) for _ in range(50)]
            # Do minimal work
            for t in tensors:
                _ = t.sum()
            # Delete
            del tensors
            torch.cuda.empty_cache() if device.type == "cuda" else None
            tick_count += 1

    actual_duration = time.time() - start_time

    return {
        "pattern": pattern,
        "duration": actual_duration,
        "ticks": tick_count,
        "ops_per_second": tick_count * num_stars ** 2 / actual_duration if tick_count > 0 else 0
    }


def analyze_hardware_data(samples: list[HardwareSample]) -> dict:
    """
    Analyze hardware samples for anomalies.

    Look for:
    1. Power draw that doesn't match utilization
    2. "Hidden" energy consumption
    3. Differences between clean and broken math
    """
    if not samples:
        return {}

    # Group by computation type
    by_type = {}
    for s in samples:
        if s.computation_type not in by_type:
            by_type[s.computation_type] = []
        by_type[s.computation_type].append(s)

    analysis = {}

    for comp_type, type_samples in by_type.items():
        powers = [s.gpu_power_watts for s in type_samples if s.gpu_power_watts]
        utils = [s.gpu_utilization for s in type_samples if s.gpu_utilization]
        ops = [s.ops_per_second for s in type_samples if s.ops_per_second > 0]

        if powers and utils:
            avg_power = np.mean(powers)
            avg_util = np.mean(utils)
            avg_ops = np.mean(ops) if ops else 0

            # Power per utilization point
            power_per_util = avg_power / (avg_util + 0.01)

            # Power per operation (efficiency)
            power_per_op = avg_power / (avg_ops + 1) if avg_ops > 0 else 0

            # "Unexplained" power = power that doesn't scale with utilization
            # If utilization is 50% but power is 80% of max, there's unexplained draw
            expected_power_ratio = avg_util / 100.0
            actual_power_ratio = avg_power / max(powers) if max(powers) > 0 else 0
            unexplained_ratio = actual_power_ratio - expected_power_ratio

            analysis[comp_type] = {
                "avg_power": avg_power,
                "avg_utilization": avg_util,
                "avg_ops": avg_ops,
                "power_per_util": power_per_util,
                "power_per_op": power_per_op,
                "unexplained_ratio": unexplained_ratio,
                "samples": len(type_samples)
            }

    return analysis


def plot_hardware_data(samples: list[HardwareSample], analysis: dict, save_dir: str):
    """Plot hardware monitoring results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Convert to arrays
    timestamps = np.array([s.timestamp for s in samples])
    timestamps = timestamps - timestamps[0]  # Relative time

    powers = np.array([s.gpu_power_watts if s.gpu_power_watts else np.nan for s in samples])
    utils = np.array([s.gpu_utilization if s.gpu_utilization else np.nan for s in samples])
    labels = [s.computation_type for s in samples]

    # 1. Power over time
    ax1 = axes[0, 0]
    ax1.plot(timestamps, powers, 'b-', linewidth=1, alpha=0.7)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("GPU Power (W)")
    ax1.set_title("GPU Power Draw Over Time")
    ax1.grid(True, alpha=0.3)

    # Color regions by computation type
    unique_labels = list(dict.fromkeys(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {l: c for l, c in zip(unique_labels, colors)}

    for i, (t, label) in enumerate(zip(timestamps, labels)):
        if i > 0:
            ax1.axvspan(timestamps[i-1], t, alpha=0.2, color=label_to_color[label])

    # 2. Power vs Utilization scatter
    ax2 = axes[0, 1]
    for label in unique_labels:
        mask = np.array([l == label for l in labels])
        ax2.scatter(utils[mask], powers[mask], alpha=0.5, label=label, s=10)

    ax2.set_xlabel("GPU Utilization (%)")
    ax2.set_ylabel("GPU Power (W)")
    ax2.set_title("Power vs Utilization (looking for anomalies)")
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Add trend line
    valid = ~np.isnan(powers) & ~np.isnan(utils)
    if valid.sum() > 10:
        z = np.polyfit(utils[valid], powers[valid], 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 100, 100)
        ax2.plot(x_line, p(x_line), 'r--', alpha=0.5, label='Expected')

    # 3. Power efficiency by type
    ax3 = axes[1, 0]
    if analysis:
        types = list(analysis.keys())
        power_per_util = [analysis[t]["power_per_util"] for t in types]
        unexplained = [analysis[t]["unexplained_ratio"] * 100 for t in types]

        x = np.arange(len(types))
        width = 0.35

        ax3.bar(x - width/2, power_per_util, width, label='W per 1% util', color='blue', alpha=0.7)
        ax3.bar(x + width/2, unexplained, width, label='Unexplained %', color='red', alpha=0.7)

        ax3.set_xlabel("Computation Type")
        ax3.set_ylabel("Value")
        ax3.set_title("Power Efficiency & Unexplained Energy")
        ax3.set_xticks(x)
        ax3.set_xticklabels(types, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

    # 4. Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary = "HARDWARE LEAK ANALYSIS\n"
    summary += "="*40 + "\n\n"

    if analysis:
        # Compare broken vs clean math
        if "int4_broken" in analysis and "float64_clean" in analysis:
            broken = analysis["int4_broken"]
            clean = analysis["float64_clean"]

            power_diff = broken["avg_power"] - clean["avg_power"]
            unexplained_diff = broken["unexplained_ratio"] - clean["unexplained_ratio"]

            summary += f"Broken Math vs Clean Math:\n"
            summary += f"  Power difference: {power_diff:+.1f}W\n"
            summary += f"  Unexplained diff: {unexplained_diff*100:+.1f}%\n\n"

            if abs(unexplained_diff) > 0.1:
                if unexplained_diff > 0:
                    summary += "! Broken math has MORE unexplained power\n"
                    summary += "  Could indicate 'hidden' computation!\n"
                else:
                    summary += "! Clean math has MORE unexplained power\n"
                    summary += "  Unexpected result.\n"
            else:
                summary += "No significant unexplained power difference.\n"

        # Look for any anomalies
        for comp_type, data in analysis.items():
            if data["unexplained_ratio"] > 0.2:
                summary += f"\n! {comp_type}: {data['unexplained_ratio']*100:.1f}% unexplained power"

    else:
        summary += "No GPU monitoring data available.\n"
        summary += "Install pynvml: pip install pynvml"

    ax4.text(0.1, 0.9, summary, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle("Hardware Leak Test: Looking for 'Parasitic' Energy",
                 fontsize=16, y=1.02)
    plt.tight_layout()

    save_path = Path(save_dir) / "hardware_leak_test.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {save_path}")

    return fig


def print_results(analysis: dict):
    """Print analysis results."""
    print("\n" + "="*60)
    print("HARDWARE LEAK TEST RESULTS")
    print("="*60)

    if not analysis:
        print("\nNo data collected. Make sure pynvml is installed.")
        return

    print("\n{:<20} {:>10} {:>10} {:>12} {:>12}".format(
        "Pattern", "Power (W)", "Util %", "W/util", "Unexplained"
    ))
    print("-" * 70)

    for comp_type, data in analysis.items():
        print("{:<20} {:>10.1f} {:>10.1f} {:>12.2f} {:>11.1f}%".format(
            comp_type,
            data["avg_power"],
            data["avg_utilization"],
            data["power_per_util"],
            data["unexplained_ratio"] * 100
        ))

    print("\n" + "-"*40)
    print("INTERPRETATION:")
    print("""
'Unexplained' power = power draw that doesn't correlate with GPU utilization.

If 'broken math' (int4) has MORE unexplained power than 'clean math' (float64),
it could mean:
  - The quantization creates extra hidden work
  - The "simulation" is struggling with inconsistent data
  - There's overhead from error correction or reconciliation

This is speculative - but any consistent pattern is interesting!
""")


def main():
    parser = argparse.ArgumentParser(description="Hardware Leak Test")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Duration per test pattern (seconds)")
    parser.add_argument("--output", type=str, default="output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Initialize monitor
    monitor = HardwareMonitor()

    if not monitor.has_gpu:
        print("\nWarning: No GPU power monitoring available.")
        print("Results will be limited without pynvml.")

    # Test patterns
    patterns = [
        "idle",
        "float64_clean",
        "float32_standard",
        "int4_broken",
        "recursive_stress",
        "memory_thrash"
    ]

    print(f"\nRunning {len(patterns)} test patterns, {args.duration}s each...")
    print("="*60)

    # Start monitoring
    monitor.start()

    # Run each pattern
    for pattern in patterns:
        print(f"\n  Testing: {pattern}...")
        result = run_computation_pattern(
            pattern=pattern,
            duration=args.duration,
            monitor=monitor,
            device=device
        )
        print(f"    Completed: {result['ticks']} ticks, {result['ops_per_second']:.0f} ops/s")

        # Brief pause between patterns
        monitor.set_label("transition", 0)
        time.sleep(1.0)

    # Stop monitoring
    monitor.stop()

    # Get and analyze data
    samples = monitor.get_samples()
    print(f"\nCollected {len(samples)} hardware samples")

    analysis = analyze_hardware_data(samples)

    # Results
    print_results(analysis)

    # Plot
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    plot_hardware_data(samples, analysis, args.output)

    # Cleanup
    monitor.cleanup()

    plt.show()


if __name__ == "__main__":
    main()
