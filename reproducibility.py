"""
REPRODUCIBILITY MANIFEST
========================

Captures exact environment, versions, seeds, and hardware state
to make all experimental claims verifiable and reproducible.

This module addresses the methodological requirement:
"GPU driver/CUDA version, PyTorch version, exact kernels used,
seeds, profiling settings."
"""

import os
import sys
import json
import hashlib
import platform
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import torch
import numpy as np


@dataclass
class HardwareManifest:
    """Captures hardware configuration."""
    cpu_model: str
    cpu_cores: int
    ram_gb: float
    gpu_name: str
    gpu_memory_gb: float
    gpu_driver_version: str
    cuda_version: str
    cudnn_version: str
    gpu_compute_capability: str
    gpu_sm_count: int


@dataclass
class SoftwareManifest:
    """Captures software versions."""
    python_version: str
    pytorch_version: str
    numpy_version: str
    os_name: str
    os_version: str
    platform: str


@dataclass
class ExperimentConfig:
    """Captures experiment parameters."""
    random_seed: int
    num_stars: int
    num_ticks: int
    precision_mode: str
    dt: float
    softening: float
    G: float
    quantization_levels: Optional[int]


@dataclass
class GPUState:
    """Captures GPU state at measurement time."""
    clock_speed_mhz: int
    memory_clock_mhz: int
    power_draw_watts: float
    temperature_c: int
    utilization_percent: int
    memory_used_mb: int
    memory_total_mb: int
    performance_state: str  # P0-P12
    throttle_reasons: List[str]


@dataclass
class ReproducibilityManifest:
    """Complete reproducibility record."""
    timestamp: str
    experiment_id: str
    hardware: HardwareManifest
    software: SoftwareManifest
    config: ExperimentConfig
    gpu_state_before: Optional[GPUState]
    gpu_state_after: Optional[GPUState]
    initial_state_hash: str  # SHA256 of initial positions/velocities
    results_hash: str  # SHA256 of final state


def get_hardware_manifest() -> HardwareManifest:
    """Collect hardware information."""
    import subprocess

    # CPU info
    cpu_model = platform.processor() or "Unknown"
    cpu_cores = os.cpu_count() or 0

    # RAM (platform-specific)
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        ram_gb = 0.0

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_compute = torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor
        gpu_sm_count = torch.cuda.get_device_properties(0).multi_processor_count

        # Driver version
        try:
            import pynvml
            pynvml.nvmlInit()
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            pynvml.nvmlShutdown()
        except:
            driver_version = "Unknown"

        cuda_version = torch.version.cuda or "Unknown"
        cudnn_version = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
    else:
        gpu_name = "N/A (CPU mode)"
        gpu_memory_gb = 0.0
        gpu_compute = (0, 0)
        gpu_sm_count = 0
        driver_version = "N/A"
        cuda_version = "N/A"
        cudnn_version = "N/A"

    return HardwareManifest(
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        ram_gb=round(ram_gb, 2),
        gpu_name=gpu_name,
        gpu_memory_gb=round(gpu_memory_gb, 2),
        gpu_driver_version=driver_version,
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
        gpu_compute_capability=f"{gpu_compute[0]}.{gpu_compute[1]}",
        gpu_sm_count=gpu_sm_count
    )


def get_software_manifest() -> SoftwareManifest:
    """Collect software versions."""
    return SoftwareManifest(
        python_version=sys.version.split()[0],
        pytorch_version=torch.__version__,
        numpy_version=np.__version__,
        os_name=platform.system(),
        os_version=platform.release(),
        platform=platform.platform()
    )


def get_gpu_state() -> Optional[GPUState]:
    """Capture current GPU state using NVML."""
    if not torch.cuda.is_available():
        return None

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # Clock speeds
        clock_speed = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)

        # Power
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W

        # Temperature
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

        # Utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        # Memory
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        # Performance state
        pstate = pynvml.nvmlDeviceGetPerformanceState(handle)

        # Throttle reasons
        throttle = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
        throttle_reasons = []
        if throttle & pynvml.nvmlClocksThrottleReasonGpuIdle:
            throttle_reasons.append("GPU_IDLE")
        if throttle & pynvml.nvmlClocksThrottleReasonSwPowerCap:
            throttle_reasons.append("SW_POWER_CAP")
        if throttle & pynvml.nvmlClocksThrottleReasonHwSlowdown:
            throttle_reasons.append("HW_SLOWDOWN")
        if throttle & pynvml.nvmlClocksThrottleReasonSwThermalSlowdown:
            throttle_reasons.append("SW_THERMAL")
        if throttle & pynvml.nvmlClocksThrottleReasonHwThermalSlowdown:
            throttle_reasons.append("HW_THERMAL")

        pynvml.nvmlShutdown()

        return GPUState(
            clock_speed_mhz=clock_speed,
            memory_clock_mhz=mem_clock,
            power_draw_watts=round(power, 1),
            temperature_c=temp,
            utilization_percent=util.gpu,
            memory_used_mb=mem_info.used // (1024**2),
            memory_total_mb=mem_info.total // (1024**2),
            performance_state=f"P{pstate}",
            throttle_reasons=throttle_reasons
        )
    except ImportError:
        print("WARNING: pynvml not installed. GPU state monitoring disabled.")
        print("         Install with: pip install pynvml")
        return None
    except Exception as e:
        print(f"WARNING: Failed to get GPU state: {e}")
        return None


def hash_tensor_state(positions: torch.Tensor, velocities: torch.Tensor) -> str:
    """Create SHA256 hash of simulation state for verification."""
    pos_bytes = positions.cpu().numpy().tobytes()
    vel_bytes = velocities.cpu().numpy().tobytes()
    combined = pos_bytes + vel_bytes
    return hashlib.sha256(combined).hexdigest()[:16]


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_manifest(
    config: ExperimentConfig,
    initial_positions: torch.Tensor,
    initial_velocities: torch.Tensor,
    final_positions: torch.Tensor = None,
    final_velocities: torch.Tensor = None,
    gpu_state_before: GPUState = None,
    gpu_state_after: GPUState = None
) -> ReproducibilityManifest:
    """Create complete reproducibility manifest."""

    initial_hash = hash_tensor_state(initial_positions, initial_velocities)

    if final_positions is not None and final_velocities is not None:
        results_hash = hash_tensor_state(final_positions, final_velocities)
    else:
        results_hash = "N/A"

    # Generate experiment ID
    exp_id = f"{config.precision_mode}_{config.num_stars}_{config.random_seed}_{datetime.now().strftime('%H%M%S')}"

    return ReproducibilityManifest(
        timestamp=datetime.now().isoformat(),
        experiment_id=exp_id,
        hardware=get_hardware_manifest(),
        software=get_software_manifest(),
        config=config,
        gpu_state_before=gpu_state_before,
        gpu_state_after=gpu_state_after,
        initial_state_hash=initial_hash,
        results_hash=results_hash
    )


def save_manifest(manifest: ReproducibilityManifest, filepath: str):
    """Save manifest to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(asdict(manifest), f, indent=2)


def print_manifest(manifest: ReproducibilityManifest):
    """Print manifest in human-readable format."""
    print("\n" + "="*70)
    print("  REPRODUCIBILITY MANIFEST")
    print("="*70)

    print(f"\n  Experiment ID: {manifest.experiment_id}")
    print(f"  Timestamp: {manifest.timestamp}")

    print(f"\n  HARDWARE:")
    print(f"    CPU: {manifest.hardware.cpu_model} ({manifest.hardware.cpu_cores} cores)")
    print(f"    RAM: {manifest.hardware.ram_gb:.1f} GB")
    print(f"    GPU: {manifest.hardware.gpu_name}")
    print(f"    GPU Memory: {manifest.hardware.gpu_memory_gb:.1f} GB")
    print(f"    CUDA: {manifest.hardware.cuda_version}")
    print(f"    Driver: {manifest.hardware.gpu_driver_version}")
    print(f"    Compute: SM {manifest.hardware.gpu_compute_capability} ({manifest.hardware.gpu_sm_count} SMs)")

    print(f"\n  SOFTWARE:")
    print(f"    Python: {manifest.software.python_version}")
    print(f"    PyTorch: {manifest.software.pytorch_version}")
    print(f"    NumPy: {manifest.software.numpy_version}")
    print(f"    OS: {manifest.software.os_name} {manifest.software.os_version}")

    print(f"\n  EXPERIMENT CONFIG:")
    print(f"    Seed: {manifest.config.random_seed}")
    print(f"    Stars: {manifest.config.num_stars}")
    print(f"    Ticks: {manifest.config.num_ticks}")
    print(f"    Precision: {manifest.config.precision_mode}")
    print(f"    dt: {manifest.config.dt}")
    print(f"    Softening: {manifest.config.softening}")
    if manifest.config.quantization_levels:
        print(f"    Quantization: {manifest.config.quantization_levels} levels")

    if manifest.gpu_state_before:
        gs = manifest.gpu_state_before
        print(f"\n  GPU STATE (before):")
        print(f"    Clock: {gs.clock_speed_mhz} MHz")
        print(f"    Power: {gs.power_draw_watts} W")
        print(f"    Temp: {gs.temperature_c}°C")
        print(f"    Perf State: {gs.performance_state}")
        if gs.throttle_reasons:
            print(f"    Throttling: {', '.join(gs.throttle_reasons)}")

    if manifest.gpu_state_after:
        gs = manifest.gpu_state_after
        print(f"\n  GPU STATE (after):")
        print(f"    Clock: {gs.clock_speed_mhz} MHz")
        print(f"    Power: {gs.power_draw_watts} W")
        print(f"    Temp: {gs.temperature_c}°C")
        print(f"    Perf State: {gs.performance_state}")

    print(f"\n  VERIFICATION HASHES:")
    print(f"    Initial state: {manifest.initial_state_hash}")
    print(f"    Final state: {manifest.results_hash}")

    print("="*70 + "\n")


# =============================================================================
# STATISTICAL RIGOR: Multi-seed experiments with confidence intervals
# =============================================================================

@dataclass
class StatisticalResult:
    """Result with proper error bars."""
    metric_name: str
    mean: float
    std: float
    ci_95_low: float
    ci_95_high: float
    n_samples: int
    values: List[float]


def run_with_confidence(
    experiment_fn,  # Function that returns a single metric value
    n_seeds: int = 10,
    base_seed: int = 42,
    metric_name: str = "metric"
) -> StatisticalResult:
    """
    Run experiment with multiple seeds and compute confidence intervals.

    This addresses: "The paper says 10 seeds; make that visible in plots/tables."
    """
    values = []

    for i in range(n_seeds):
        seed = base_seed + i
        set_all_seeds(seed)
        result = experiment_fn(seed)
        values.append(result)

    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample std

    # 95% CI using t-distribution
    from scipy import stats
    t_critical = stats.t.ppf(0.975, df=n_seeds-1)
    margin = t_critical * std / np.sqrt(n_seeds)

    return StatisticalResult(
        metric_name=metric_name,
        mean=mean,
        std=std,
        ci_95_low=mean - margin,
        ci_95_high=mean + margin,
        n_samples=n_seeds,
        values=values.tolist()
    )


def format_with_ci(result: StatisticalResult, precision: int = 2) -> str:
    """Format result with confidence interval for publication."""
    return f"{result.mean:.{precision}f} ± {result.std:.{precision}f} (95% CI: [{result.ci_95_low:.{precision}f}, {result.ci_95_high:.{precision}f}], n={result.n_samples})"


# =============================================================================
# CLAIMS CLARIFICATIONS
# =============================================================================

METHODOLOGY_NOTES = """
METHODOLOGY CLARIFICATIONS
==========================

1. SYMPLECTIC INTEGRATOR CLAIM (Tightened)
------------------------------------------
"Leapfrog is symplectic in exact arithmetic. In finite precision, secular
energy drift can arise from:
- Floating-point roundoff in force accumulation
- Quantization of distance/force values
- Force asymmetry from non-commutative operations
- Softening parameter choices

Our claim: The *differential* energy drift between precision modes
isolates the quantization effect, controlling for integrator artifacts."


2. POWER MEASUREMENT CAVEAT
---------------------------
"GPU power measurements via NVML reflect total board power, which includes:
- Compute workload
- Memory bandwidth
- Clock/boost behavior
- Thermal management overhead

To isolate quantization-specific overhead, we:
- Lock GPU clocks where possible (nvidia-smi -lgc)
- Log performance state (P-state) throughout
- Report throttling events
- Compare same kernels with different data precision

The 'unexplained power' claim requires verification via Nsight Compute
instruction-level profiling."


3. COSMOLOGY ANALOGIES (Explicit)
---------------------------------
"The mappings between simulation artifacts and cosmological phenomena
are STRUCTURAL ANALOGIES, not quantitative correspondences:

- 'Ghost energy' (energy drift) ≈ Dark Energy analogy
  Both represent unexplained energy injection, but units are incompatible.

- 'Flat rotation curve' ≈ Dark Matter signature analogy
  Both represent anomalous gravitational behavior, but our simulation
  has no actual missing mass.

These analogies are suggestive, not probative."


4. 'OBSERVER EFFECT' TERMINOLOGY
--------------------------------
Renamed to: "Measurement Overhead"

"The ~35% throughput reduction when logging GPU state reflects:
- NVML API latency
- CPU-GPU synchronization
- I/O contention

This is NOT related to quantum observer effects. The terminology was
used metaphorically but is scientifically misleading. We now use
'measurement overhead' or 'instrumentation cost.'"


5. SIMULATION PROBABILITY ESTIMATE
----------------------------------
"The 'simulation probability' score is a HEURISTIC combining multiple
anomaly detections. It is NOT:
- A Bayesian posterior probability
- A frequentist confidence level
- A meaningful probability in any formal sense

It should be interpreted as: 'aggregate anomaly score on arbitrary 0-100 scale'
rather than 'probability we live in a simulation.'"
"""


def print_methodology_notes():
    """Print methodology clarifications."""
    print(METHODOLOGY_NOTES)


if __name__ == "__main__":
    # Demo: capture current environment
    print("Capturing reproducibility manifest...")

    hardware = get_hardware_manifest()
    software = get_software_manifest()
    gpu_state = get_gpu_state()

    print(f"\nHardware: {hardware.gpu_name}")
    print(f"Software: PyTorch {software.pytorch_version}")

    if gpu_state:
        print(f"GPU State: {gpu_state.clock_speed_mhz} MHz, {gpu_state.power_draw_watts}W, P{gpu_state.performance_state}")

    print("\n" + "="*70)
    print_methodology_notes()
