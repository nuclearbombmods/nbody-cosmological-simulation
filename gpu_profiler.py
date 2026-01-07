"""
GPU PROFILER - Proper Power & Performance Measurement
======================================================

Addresses the methodological concern:
"Prove the power result isn't a clocking artifact"

This module:
1. Logs GPU clocks, utilization, perf state throughout experiments
2. Supports clock locking for controlled comparisons
3. Captures Nsight-compatible metrics where possible
4. Detects and reports throttling events
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from collections import deque
import json

import torch

# Try to import NVML
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False
    print("WARNING: pynvml not installed. GPU profiling disabled.")
    print("         Install with: pip install pynvml")


@dataclass
class GPUSample:
    """Single GPU state sample."""
    timestamp: float
    clock_graphics_mhz: int
    clock_memory_mhz: int
    clock_sm_mhz: int
    power_watts: float
    temperature_c: int
    utilization_gpu: int
    utilization_memory: int
    memory_used_mb: int
    performance_state: int  # 0-12
    throttle_reasons: int  # Bitmask
    pcie_tx_mb_s: float
    pcie_rx_mb_s: float


@dataclass
class GPUProfileResult:
    """Complete profiling result."""
    experiment_name: str
    duration_seconds: float
    samples: List[GPUSample]

    # Aggregates
    mean_power_watts: float
    max_power_watts: float
    min_power_watts: float
    std_power_watts: float

    mean_clock_mhz: float
    clock_stability: float  # std/mean - lower is more stable

    mean_utilization: float
    mean_temperature: float

    throttle_events: int
    throttle_breakdown: Dict[str, int]

    # For comparison validation
    clock_locked: bool
    base_clock_mhz: int
    boost_clock_mhz: int


class GPUProfiler:
    """
    Continuous GPU monitoring with proper methodology.

    Usage:
        profiler = GPUProfiler()
        profiler.start("my_experiment")
        # ... run experiment ...
        result = profiler.stop()
        profiler.print_report(result)
    """

    THROTTLE_REASONS = {
        0x0000000000000001: "GPU_IDLE",
        0x0000000000000002: "APP_CLOCKS_SETTING",
        0x0000000000000004: "SW_POWER_CAP",
        0x0000000000000008: "HW_SLOWDOWN",
        0x0000000000000010: "SYNC_BOOST",
        0x0000000000000020: "SW_THERMAL",
        0x0000000000000040: "HW_THERMAL",
        0x0000000000000080: "HW_POWER_BRAKE",
        0x0000000000000100: "DISPLAY_CLOCKS",
    }

    def __init__(self, device_index: int = 0, sample_interval_ms: int = 100):
        self.device_index = device_index
        self.sample_interval = sample_interval_ms / 1000.0
        self.samples: List[GPUSample] = []
        self.running = False
        self.thread = None
        self.experiment_name = ""
        self.start_time = 0

        if HAS_NVML:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

            # Get device limits
            self.base_clock = pynvml.nvmlDeviceGetMaxClockInfo(
                self.handle, pynvml.NVML_CLOCK_GRAPHICS
            )
            try:
                self.boost_clock = pynvml.nvmlDeviceGetMaxClockInfo(
                    self.handle, pynvml.NVML_CLOCK_SM
                )
            except:
                self.boost_clock = self.base_clock

            self.power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self.handle) / 1000

        else:
            self.handle = None
            self.base_clock = 0
            self.boost_clock = 0
            self.power_limit = 0

    def _sample(self) -> Optional[GPUSample]:
        """Capture single GPU state sample."""
        if not HAS_NVML or not self.handle:
            return None

        try:
            clock_graphics = pynvml.nvmlDeviceGetClockInfo(
                self.handle, pynvml.NVML_CLOCK_GRAPHICS
            )
            clock_memory = pynvml.nvmlDeviceGetClockInfo(
                self.handle, pynvml.NVML_CLOCK_MEM
            )
            try:
                clock_sm = pynvml.nvmlDeviceGetClockInfo(
                    self.handle, pynvml.NVML_CLOCK_SM
                )
            except:
                clock_sm = clock_graphics

            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000  # mW to W
            temp = pynvml.nvmlDeviceGetTemperature(
                self.handle, pynvml.NVML_TEMPERATURE_GPU
            )
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            pstate = pynvml.nvmlDeviceGetPerformanceState(self.handle)
            throttle = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle)

            # PCIe throughput
            try:
                pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(
                    self.handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                ) / 1024  # KB/s to MB/s
                pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(
                    self.handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                ) / 1024
            except:
                pcie_tx = pcie_rx = 0

            return GPUSample(
                timestamp=time.perf_counter() - self.start_time,
                clock_graphics_mhz=clock_graphics,
                clock_memory_mhz=clock_memory,
                clock_sm_mhz=clock_sm,
                power_watts=power,
                temperature_c=temp,
                utilization_gpu=util.gpu,
                utilization_memory=util.memory,
                memory_used_mb=mem_info.used // (1024**2),
                performance_state=pstate,
                throttle_reasons=throttle,
                pcie_tx_mb_s=pcie_tx,
                pcie_rx_mb_s=pcie_rx
            )
        except Exception as e:
            print(f"Sampling error: {e}")
            return None

    def _sample_loop(self):
        """Background sampling thread."""
        while self.running:
            sample = self._sample()
            if sample:
                self.samples.append(sample)
            time.sleep(self.sample_interval)

    def start(self, experiment_name: str = "experiment"):
        """Start profiling."""
        if not HAS_NVML:
            print("WARNING: NVML not available, profiling disabled")
            return

        self.experiment_name = experiment_name
        self.samples = []
        self.start_time = time.perf_counter()
        self.running = True
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()
        print(f"  GPU profiling started: {experiment_name}")

    def stop(self) -> Optional[GPUProfileResult]:
        """Stop profiling and return results."""
        if not self.running:
            return None

        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

        duration = time.perf_counter() - self.start_time

        if not self.samples:
            print("  WARNING: No samples collected")
            return None

        return self._analyze()

    def _analyze(self) -> GPUProfileResult:
        """Analyze collected samples."""
        import numpy as np

        powers = [s.power_watts for s in self.samples]
        clocks = [s.clock_graphics_mhz for s in self.samples]
        utils = [s.utilization_gpu for s in self.samples]
        temps = [s.temperature_c for s in self.samples]

        # Count throttle events
        throttle_count = 0
        throttle_breakdown = {}
        for s in self.samples:
            if s.throttle_reasons != 0:
                throttle_count += 1
                for mask, name in self.THROTTLE_REASONS.items():
                    if s.throttle_reasons & mask:
                        throttle_breakdown[name] = throttle_breakdown.get(name, 0) + 1

        # Check if clocks were stable (locked)
        clock_std = np.std(clocks)
        clock_mean = np.mean(clocks)
        clock_stability = clock_std / clock_mean if clock_mean > 0 else float('inf')
        clock_locked = clock_stability < 0.01  # <1% variation = effectively locked

        duration = self.samples[-1].timestamp if self.samples else 0

        return GPUProfileResult(
            experiment_name=self.experiment_name,
            duration_seconds=duration,
            samples=self.samples,
            mean_power_watts=np.mean(powers),
            max_power_watts=np.max(powers),
            min_power_watts=np.min(powers),
            std_power_watts=np.std(powers),
            mean_clock_mhz=clock_mean,
            clock_stability=clock_stability,
            mean_utilization=np.mean(utils),
            mean_temperature=np.mean(temps),
            throttle_events=throttle_count,
            throttle_breakdown=throttle_breakdown,
            clock_locked=clock_locked,
            base_clock_mhz=self.base_clock,
            boost_clock_mhz=self.boost_clock
        )

    def print_report(self, result: GPUProfileResult):
        """Print detailed profiling report."""
        print("\n" + "="*70)
        print(f"  GPU PROFILE REPORT: {result.experiment_name}")
        print("="*70)

        print(f"\n  Duration: {result.duration_seconds:.2f}s")
        print(f"  Samples: {len(result.samples)}")

        print(f"\n  POWER:")
        print(f"    Mean: {result.mean_power_watts:.1f} W")
        print(f"    Min:  {result.min_power_watts:.1f} W")
        print(f"    Max:  {result.max_power_watts:.1f} W")
        print(f"    Std:  {result.std_power_watts:.1f} W")
        print(f"    Limit: {self.power_limit:.0f} W")

        print(f"\n  CLOCKS:")
        print(f"    Mean: {result.mean_clock_mhz:.0f} MHz")
        print(f"    Stability: {result.clock_stability:.4f} (std/mean)")
        print(f"    Locked: {'YES' if result.clock_locked else 'NO'}")
        print(f"    Base/Boost: {result.base_clock_mhz}/{result.boost_clock_mhz} MHz")

        print(f"\n  UTILIZATION:")
        print(f"    GPU: {result.mean_utilization:.1f}%")
        print(f"    Temperature: {result.mean_temperature:.1f}°C")

        print(f"\n  THROTTLING:")
        print(f"    Events: {result.throttle_events} ({100*result.throttle_events/len(result.samples):.1f}% of samples)")
        if result.throttle_breakdown:
            for reason, count in result.throttle_breakdown.items():
                print(f"      {reason}: {count}")
        else:
            print(f"      None detected")

        # Methodology validation
        print(f"\n  METHODOLOGY VALIDATION:")
        if result.clock_locked:
            print(f"    [OK] Clocks stable - power comparison valid")
        else:
            print(f"    [WARN] Clock variance detected - may affect power comparison")
            print(f"           Consider: nvidia-smi -lgc {result.mean_clock_mhz:.0f},{result.mean_clock_mhz:.0f}")

        if result.throttle_events > len(result.samples) * 0.1:
            print(f"    [WARN] Significant throttling - results may be affected")
        else:
            print(f"    [OK] Minimal throttling")

        if result.mean_utilization > 90:
            print(f"    [OK] High utilization - compute-bound workload")
        else:
            print(f"    [WARN] Low utilization ({result.mean_utilization:.0f}%) - may be memory/IO bound")

        print("="*70 + "\n")

    def compare_experiments(self, results: List[GPUProfileResult]):
        """Compare power between experiments with proper controls."""
        print("\n" + "="*70)
        print("  POWER COMPARISON ANALYSIS")
        print("="*70)

        if len(results) < 2:
            print("  Need at least 2 experiments to compare")
            return

        # Check if clocks were consistent
        clocks = [r.mean_clock_mhz for r in results]
        clock_variance = max(clocks) - min(clocks)

        print(f"\n  Clock variance across experiments: {clock_variance:.0f} MHz")
        if clock_variance > 50:
            print("  [WARN] Clocks varied significantly - power comparison may be invalid!")
            print("         Recommendation: Lock clocks with nvidia-smi -lgc before comparing")
        else:
            print("  [OK] Clocks stable across experiments")

        print(f"\n  {'Experiment':<30} {'Power (W)':<15} {'Clock (MHz)':<15} {'Util %'}")
        print(f"  {'-'*75}")

        baseline = results[0]
        for r in results:
            delta = ((r.mean_power_watts - baseline.mean_power_watts) / baseline.mean_power_watts * 100
                     if baseline.mean_power_watts > 0 else 0)
            delta_str = f"({delta:+.1f}%)" if r != baseline else "(baseline)"
            print(f"  {r.experiment_name:<30} {r.mean_power_watts:<10.1f} {delta_str:<5} "
                  f"{r.mean_clock_mhz:<15.0f} {r.mean_utilization:.1f}")

        print(f"\n  INTERPRETATION:")
        print(f"  Power differences are only meaningful if:")
        print(f"    1. Clocks were locked or stable (<50 MHz variance)")
        print(f"    2. Utilization was similar (compute-bound)")
        print(f"    3. No significant throttling occurred")

        print("="*70 + "\n")

    def save_samples(self, filepath: str, result: GPUProfileResult):
        """Save raw samples to JSON for external analysis."""
        data = {
            "experiment": result.experiment_name,
            "duration_seconds": result.duration_seconds,
            "summary": {
                "mean_power_watts": result.mean_power_watts,
                "std_power_watts": result.std_power_watts,
                "mean_clock_mhz": result.mean_clock_mhz,
                "clock_locked": result.clock_locked,
                "throttle_events": result.throttle_events
            },
            "samples": [
                {
                    "t": s.timestamp,
                    "power": s.power_watts,
                    "clock": s.clock_graphics_mhz,
                    "util": s.utilization_gpu,
                    "temp": s.temperature_c,
                    "pstate": s.performance_state,
                    "throttle": s.throttle_reasons
                }
                for s in result.samples
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Samples saved to: {filepath}")

    def __del__(self):
        """Cleanup NVML."""
        if HAS_NVML:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


# =============================================================================
# MEASUREMENT OVERHEAD (formerly "Observer Effect")
# =============================================================================

def measure_instrumentation_overhead(
    workload_fn: Callable,
    num_iterations: int = 100,
    device: torch.device = None
) -> Dict[str, float]:
    """
    Measure the overhead of GPU monitoring itself.

    This replaces the misleading "observer effect" terminology.
    It's just measurement/instrumentation overhead.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n  Measuring instrumentation overhead...")

    # Run without monitoring
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        workload_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
    time_without = time.perf_counter() - start

    # Run with monitoring
    profiler = GPUProfiler(sample_interval_ms=10)  # Aggressive sampling
    profiler.start("overhead_test")

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        workload_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
    time_with = time.perf_counter() - start

    profiler.stop()

    overhead_pct = (time_with - time_without) / time_without * 100

    print(f"\n  INSTRUMENTATION OVERHEAD RESULT:")
    print(f"    Without monitoring: {time_without:.3f}s")
    print(f"    With monitoring:    {time_with:.3f}s")
    print(f"    Overhead:           {overhead_pct:.1f}%")
    print(f"\n    (This is measurement overhead, NOT quantum observer effect)")

    return {
        "time_without_seconds": time_without,
        "time_with_seconds": time_with,
        "overhead_percent": overhead_pct,
        "iterations": num_iterations
    }


if __name__ == "__main__":
    # Demo profiler
    print("GPU Profiler Demo")
    print("="*50)

    if not torch.cuda.is_available():
        print("CUDA not available - demo requires GPU")
        exit(1)

    profiler = GPUProfiler(sample_interval_ms=50)

    # Create dummy workload
    def dummy_workload():
        a = torch.randn(4096, 4096, device="cuda")
        b = torch.randn(4096, 4096, device="cuda")
        c = torch.matmul(a, b)
        torch.cuda.synchronize()

    print("\nRunning test workload for 5 seconds...")
    profiler.start("demo_workload")

    end_time = time.time() + 5
    while time.time() < end_time:
        dummy_workload()

    result = profiler.stop()

    if result:
        profiler.print_report(result)
        profiler.save_samples("gpu_profile_demo.json", result)

    # Measure instrumentation overhead
    measure_instrumentation_overhead(dummy_workload, num_iterations=50)
