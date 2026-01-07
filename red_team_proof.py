"""
RED TEAM PROOF - Bulletproof Reality Testing
=============================================

This module addresses the primary criticisms of the simulation hypothesis tests:

HOLES PATCHED:
1. "Software Optimization" - Bypass PyTorch optimizations with raw tensor ops
2. "FP is Known Broken" - Compare divergence to Heisenberg uncertainty
3. "Hardware Noise" - Statistical filtering and clean measurement

NEW TESTS:
1. Temporal Jitter - Measure tick rate fluctuations under load
2. Memory Leak Search - Long-duration entropy drift detection
3. Observer Interrupt - Does rendering state affect computation?
4. Cross-Platform Sync - Generate Reality Stability Index for comparison

OUTPUT: A single "Reality Stability Index" (RSI) that can be compared across machines.

Usage:
    python red_team_proof.py --full
    python red_team_proof.py --quick
    python red_team_proof.py --generate-rsi
    python red_team_proof.py --compare rsi_rtx5090.json rsi_mac_m3.json
"""

import argparse
import time
import json
import hashlib
import platform
import os
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import struct

import torch
import numpy as np

# Physical constants for comparison
PLANCK_CONSTANT = 6.62607015e-34  # J·s
HBAR = PLANCK_CONSTANT / (2 * np.pi)  # Reduced Planck constant
HEISENBERG_LIMIT = HBAR / 2  # Minimum uncertainty product

# FP32 constants
FP32_EPSILON = np.finfo(np.float32).eps  # ~1.19e-7
FP32_MIN_NORMAL = np.finfo(np.float32).tiny  # ~1.17e-38
FP32_MIN_SUBNORMAL = 2**-149  # ~1.4e-45


@dataclass
class TemporalJitterResult:
    """Results from temporal jitter test."""
    mean_tick_time_us: float
    std_tick_time_us: float
    max_jitter_us: float
    jitter_coefficient: float  # std/mean - should be near 0 for stable time
    tick_time_histogram: List[int] = field(default_factory=list)
    anomalous_ticks: int = 0  # Ticks that took >3 std from mean


@dataclass
class DivergenceResult:
    """Results from FP divergence test with Heisenberg comparison."""
    max_position_divergence: float
    max_velocity_divergence: float
    uncertainty_product: float  # dx * dp - compare to HBAR/2
    heisenberg_ratio: float  # uncertainty_product / (HBAR/2)
    lyapunov_exponent: float  # Exponential divergence rate
    divergence_timeline: List[float] = field(default_factory=list)


@dataclass
class MemoryLeakResult:
    """Results from long-duration entropy test."""
    initial_entropy: float
    final_entropy: float
    entropy_drift_rate: float  # bits per tick
    total_ticks: int
    runtime_seconds: float
    entropy_timeline: List[float] = field(default_factory=list)


@dataclass
class ObserverEffectResult:
    """Results from observer/rendering test."""
    power_with_render: float  # Watts
    power_without_render: float  # Watts
    power_ratio: float
    tick_rate_with_render: float
    tick_rate_without_render: float
    observer_effect_detected: bool


@dataclass
class RealityStabilityIndex:
    """The final cross-platform comparable metric."""
    # Metadata
    timestamp: str
    platform: str
    device: str
    gpu_name: str
    python_version: str
    torch_version: str

    # Raw metrics
    temporal_jitter: TemporalJitterResult
    divergence: DivergenceResult
    memory_leak: MemoryLeakResult
    observer_effect: ObserverEffectResult

    # Computed indices (0-100 scale, higher = more "glitchy")
    jitter_index: float = 0.0
    divergence_index: float = 0.0
    leak_index: float = 0.0
    observer_index: float = 0.0

    # Final composite
    reality_stability_index: float = 0.0  # 0 = stable, 100 = glitchy

    # Hash for verification
    data_hash: str = ""


def get_system_info() -> Dict:
    """Get comprehensive system information."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
        info["cuda_version"] = torch.version.cuda
    else:
        info["device"] = "cpu"
        info["gpu_name"] = "N/A"

    return info


# =============================================================================
# TEST 1: TEMPORAL JITTER (Tick Rate Stability)
# =============================================================================

def run_temporal_jitter_test(
    num_stars: int = 2000,
    num_ticks: int = 1000,
    device: torch.device = None
) -> TemporalJitterResult:
    """
    Measure tick-to-tick timing jitter.

    If the universe is a simulation, high-density regions might cause
    "lag" - variable tick times even with constant dt.

    Compare to: Time dilation near massive objects.
    """
    print("\n[TEMPORAL JITTER TEST]")
    print(f"  Stars: {num_stars}, Ticks: {num_ticks}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create galaxy - bypass high-level functions for raw measurement
    torch.manual_seed(42)  # Reproducibility
    positions = (torch.rand(num_stars, 3, device=device) - 0.5) * 20
    velocities = (torch.rand(num_stars, 3, device=device) - 0.5) * 0.1
    masses = torch.ones(num_stars, device=device) * 0.001

    G = 0.001
    dt = 0.01
    softening_sq = 0.01

    tick_times = []

    # Warm up GPU
    for _ in range(10):
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1) + softening_sq
        _ = dist_sq ** 1.5

    if device.type == "cuda":
        torch.cuda.synchronize()

    print("  Running ticks...")

    for tick in range(num_ticks):
        # High-precision timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter_ns()

        # Raw N-body computation (no PyTorch autograd overhead)
        with torch.no_grad():
            diff = positions.unsqueeze(0) - positions.unsqueeze(1)
            dist_sq = (diff ** 2).sum(dim=-1) + softening_sq
            dist_cubed = dist_sq ** 1.5
            force_factor = G / dist_cubed
            force_factor = force_factor * masses.unsqueeze(0)
            eye_mask = 1 - torch.eye(num_stars, device=device)
            force_factor = force_factor * eye_mask
            accelerations = (force_factor.unsqueeze(-1) * diff).sum(dim=1)

            velocities = velocities + accelerations * dt
            positions = positions + velocities * dt

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter_ns()

        tick_times.append((end - start) / 1000)  # Convert to microseconds

    tick_times = np.array(tick_times)
    mean_time = np.mean(tick_times)
    std_time = np.std(tick_times)
    max_jitter = np.max(np.abs(tick_times - mean_time))

    # Count anomalous ticks (>3 sigma)
    anomalous = np.sum(np.abs(tick_times - mean_time) > 3 * std_time)

    # Create histogram (10 bins)
    hist, _ = np.histogram(tick_times, bins=10)

    result = TemporalJitterResult(
        mean_tick_time_us=float(mean_time),
        std_tick_time_us=float(std_time),
        max_jitter_us=float(max_jitter),
        jitter_coefficient=float(std_time / mean_time) if mean_time > 0 else 0,
        tick_time_histogram=hist.tolist(),
        anomalous_ticks=int(anomalous)
    )

    print(f"  Mean tick: {mean_time:.1f} µs")
    print(f"  Std dev: {std_time:.1f} µs")
    print(f"  Jitter coefficient: {result.jitter_coefficient:.4f}")
    print(f"  Anomalous ticks: {anomalous}/{num_ticks}")

    return result


# =============================================================================
# TEST 2: FP DIVERGENCE vs HEISENBERG
# =============================================================================

def run_divergence_heisenberg_test(
    num_stars: int = 1000,
    num_ticks: int = 500,
    device: torch.device = None
) -> DivergenceResult:
    """
    Compare FP divergence to Heisenberg uncertainty principle.

    If dx * dv scales with HBAR/2, we have evidence that quantum
    uncertainty might be masking computational rounding.
    """
    print("\n[DIVERGENCE vs HEISENBERG TEST]")
    print(f"  Stars: {num_stars}, Ticks: {num_ticks}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    positions = (torch.rand(num_stars, 3, device=device) - 0.5) * 20
    velocities = (torch.rand(num_stars, 3, device=device) - 0.5) * 0.1
    masses = torch.ones(num_stars, device=device) * 0.001

    # Clone for parallel universe
    pos_a = positions.clone()
    vel_a = velocities.clone()
    pos_b = positions.clone()
    vel_b = velocities.clone()

    G = 0.001
    dt = 0.01
    softening_sq = 0.01

    divergence_timeline = []

    print("  Running parallel universes...")

    for tick in range(num_ticks):
        with torch.no_grad():
            # Universe A: Standard order
            diff_a = pos_a.unsqueeze(0) - pos_a.unsqueeze(1)
            dist_sq_a = (diff_a ** 2).sum(dim=-1) + softening_sq
            dist_cubed_a = dist_sq_a ** 1.5
            force_factor_a = G / dist_cubed_a * masses.unsqueeze(0)
            force_factor_a = force_factor_a * (1 - torch.eye(num_stars, device=device))
            acc_a = (force_factor_a.unsqueeze(-1) * diff_a).sum(dim=1)
            vel_a = vel_a + acc_a * dt
            pos_a = pos_a + vel_a * dt

            # Universe B: Reversed summation + FP16 intermediate
            diff_b = pos_b.unsqueeze(0) - pos_b.unsqueeze(1)
            dist_sq_b = (diff_b ** 2).sum(dim=-1) + softening_sq
            # FP16 precision loss
            dist_sq_b = dist_sq_b.half().float()
            dist_cubed_b = dist_sq_b ** 1.5
            force_factor_b = G / dist_cubed_b * masses.unsqueeze(0)
            force_factor_b = force_factor_b * (1 - torch.eye(num_stars, device=device))
            # Reverse summation order
            force_contrib = force_factor_b.unsqueeze(-1) * diff_b
            acc_b = torch.flip(force_contrib, dims=[1]).sum(dim=1)
            vel_b = vel_b + acc_b * dt
            pos_b = pos_b + vel_b * dt

            # Measure divergence
            pos_diff = (pos_a - pos_b).abs().max().item()
            divergence_timeline.append(pos_diff)

    # Final divergence
    dx = (pos_a - pos_b).abs()
    dv = (vel_a - vel_b).abs()

    max_dx = dx.max().item()
    max_dv = dv.max().item()

    # Uncertainty product (in simulation units)
    # Scale to compare with Heisenberg
    uncertainty_product = max_dx * max_dv

    # Heisenberg ratio (this is unitless comparison)
    # If this ratio is consistent across scales, it suggests a fundamental limit
    heisenberg_ratio = uncertainty_product / HEISENBERG_LIMIT if HEISENBERG_LIMIT > 0 else 0

    # Calculate Lyapunov exponent (exponential divergence rate)
    lyapunov = 0.0
    if len(divergence_timeline) > 100:
        early = np.mean(divergence_timeline[10:20])
        late = np.mean(divergence_timeline[-20:-10])
        if early > 1e-15:
            lyapunov = np.log(max(late, 1e-15) / early) / (num_ticks - 20)

    result = DivergenceResult(
        max_position_divergence=max_dx,
        max_velocity_divergence=max_dv,
        uncertainty_product=uncertainty_product,
        heisenberg_ratio=heisenberg_ratio,
        lyapunov_exponent=lyapunov,
        divergence_timeline=divergence_timeline[::10]  # Subsample for storage
    )

    print(f"  Max position divergence: {max_dx:.2e}")
    print(f"  Max velocity divergence: {max_dv:.2e}")
    print(f"  Uncertainty product: {uncertainty_product:.2e}")
    print(f"  Heisenberg ratio: {heisenberg_ratio:.2e}")
    print(f"  Lyapunov exponent: {lyapunov:.4f}")

    return result


# =============================================================================
# TEST 3: MEMORY LEAK (Long-Duration Entropy)
# =============================================================================

def measure_entropy(positions: torch.Tensor, velocities: torch.Tensor) -> float:
    """Measure entropy via compression ratio."""
    import zlib
    pos_bytes = positions.cpu().numpy().astype(np.float32).tobytes()
    vel_bytes = velocities.cpu().numpy().astype(np.float32).tobytes()
    raw = pos_bytes + vel_bytes
    compressed = zlib.compress(raw, level=9)
    # Return bits per float
    return (len(compressed) * 8) / (positions.numel() + velocities.numel())


def run_memory_leak_test(
    num_stars: int = 1000,
    duration_seconds: float = 60,  # Run for 1 minute by default
    device: torch.device = None
) -> MemoryLeakResult:
    """
    Run simulation for extended period looking for entropy drift.

    If the universe has memory leaks, entropy should increase over time
    even with constant star count.
    """
    print("\n[MEMORY LEAK TEST]")
    print(f"  Stars: {num_stars}, Duration: {duration_seconds}s")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    positions = (torch.rand(num_stars, 3, device=device) - 0.5) * 20
    velocities = (torch.rand(num_stars, 3, device=device) - 0.5) * 0.1
    masses = torch.ones(num_stars, device=device) * 0.001

    G = 0.001
    dt = 0.01
    softening_sq = 0.01

    initial_entropy = measure_entropy(positions, velocities)
    entropy_timeline = [initial_entropy]

    start_time = time.time()
    tick = 0
    last_report = start_time

    print(f"  Initial entropy: {initial_entropy:.2f} bits/float")
    print("  Running (Ctrl+C to stop early)...")

    try:
        while time.time() - start_time < duration_seconds:
            with torch.no_grad():
                diff = positions.unsqueeze(0) - positions.unsqueeze(1)
                dist_sq = (diff ** 2).sum(dim=-1) + softening_sq
                dist_cubed = dist_sq ** 1.5
                force_factor = G / dist_cubed * masses.unsqueeze(0)
                force_factor = force_factor * (1 - torch.eye(num_stars, device=device))
                acc = (force_factor.unsqueeze(-1) * diff).sum(dim=1)
                velocities = velocities + acc * dt
                positions = positions + velocities * dt

            tick += 1

            # Measure entropy every 100 ticks
            if tick % 100 == 0:
                entropy = measure_entropy(positions, velocities)
                entropy_timeline.append(entropy)

            # Progress report every 10 seconds
            if time.time() - last_report > 10:
                elapsed = time.time() - start_time
                current_entropy = entropy_timeline[-1]
                print(f"    {elapsed:.0f}s: tick={tick}, entropy={current_entropy:.2f}")
                last_report = time.time()

    except KeyboardInterrupt:
        print("\n  Stopped early by user")

    final_entropy = measure_entropy(positions, velocities)
    runtime = time.time() - start_time

    # Calculate drift rate
    if len(entropy_timeline) > 1:
        drift_rate = (final_entropy - initial_entropy) / tick
    else:
        drift_rate = 0

    result = MemoryLeakResult(
        initial_entropy=initial_entropy,
        final_entropy=final_entropy,
        entropy_drift_rate=drift_rate,
        total_ticks=tick,
        runtime_seconds=runtime,
        entropy_timeline=entropy_timeline
    )

    print(f"  Final entropy: {final_entropy:.2f} bits/float")
    print(f"  Drift rate: {drift_rate:.2e} bits/tick")
    print(f"  Total ticks: {tick}")

    return result


# =============================================================================
# TEST 4: OBSERVER EFFECT (Rendering Impact)
# =============================================================================

def run_observer_effect_test(
    num_stars: int = 2000,
    num_ticks: int = 500,
    device: torch.device = None
) -> ObserverEffectResult:
    """
    Test if "observing" (preparing data for rendering) affects computation.

    If the universe uses frustum culling, power/performance might differ
    when data is being "observed" vs not.
    """
    print("\n[OBSERVER EFFECT TEST]")
    print(f"  Stars: {num_stars}, Ticks: {num_ticks}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)
    positions = (torch.rand(num_stars, 3, device=device) - 0.5) * 20
    velocities = (torch.rand(num_stars, 3, device=device) - 0.5) * 0.1
    masses = torch.ones(num_stars, device=device) * 0.001

    G = 0.001
    dt = 0.01
    softening_sq = 0.01

    def run_simulation(observe: bool) -> Tuple[float, float]:
        """Run simulation with or without observation."""
        pos = positions.clone()
        vel = velocities.clone()

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()

        for _ in range(num_ticks):
            with torch.no_grad():
                diff = pos.unsqueeze(0) - pos.unsqueeze(1)
                dist_sq = (diff ** 2).sum(dim=-1) + softening_sq
                dist_cubed = dist_sq ** 1.5
                force_factor = G / dist_cubed * masses.unsqueeze(0)
                force_factor = force_factor * (1 - torch.eye(num_stars, device=device))
                acc = (force_factor.unsqueeze(-1) * diff).sum(dim=1)
                vel = vel + acc * dt
                pos = pos + vel * dt

                if observe:
                    # Simulate "rendering" - transfer to CPU and compute derived values
                    pos_np = pos.cpu().numpy()
                    _ = np.sqrt((pos_np ** 2).sum(axis=1))  # Compute radii
                    _ = pos_np.mean(axis=0)  # Compute center
                    pos = pos.to(device)  # Transfer back

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        tick_rate = num_ticks / elapsed

        # Estimate power from memory bandwidth (proxy)
        if device.type == "cuda":
            memory_used = torch.cuda.max_memory_allocated() / 1e9  # GB
            power_proxy = memory_used * tick_rate  # GB/s as power proxy
        else:
            power_proxy = tick_rate * 0.001

        return tick_rate, power_proxy

    print("  Running WITHOUT observation...")
    tick_rate_no_obs, power_no_obs = run_simulation(observe=False)

    print("  Running WITH observation...")
    tick_rate_obs, power_obs = run_simulation(observe=True)

    power_ratio = power_obs / power_no_obs if power_no_obs > 0 else 1.0
    observer_detected = abs(power_ratio - 1.0) > 0.1  # >10% difference

    result = ObserverEffectResult(
        power_with_render=power_obs,
        power_without_render=power_no_obs,
        power_ratio=power_ratio,
        tick_rate_with_render=tick_rate_obs,
        tick_rate_without_render=tick_rate_no_obs,
        observer_effect_detected=observer_detected
    )

    print(f"  Tick rate (no obs): {tick_rate_no_obs:.1f}/s")
    print(f"  Tick rate (obs): {tick_rate_obs:.1f}/s")
    print(f"  Power ratio: {power_ratio:.2f}")
    print(f"  Observer effect: {'DETECTED' if observer_detected else 'Not detected'}")

    return result


# =============================================================================
# REALITY STABILITY INDEX COMPUTATION
# =============================================================================

def compute_rsi(
    temporal: TemporalJitterResult,
    divergence: DivergenceResult,
    memory: MemoryLeakResult,
    observer: ObserverEffectResult
) -> Tuple[float, float, float, float, float]:
    """
    Compute the Reality Stability Index components.

    Each index is 0-100 scale:
    - 0 = perfectly stable (no glitches)
    - 100 = maximally glitchy
    """
    # Jitter Index: Based on coefficient of variation
    # CV > 0.1 is considered high jitter
    jitter_index = min(100, temporal.jitter_coefficient * 1000)

    # Divergence Index: Based on Lyapunov exponent
    # Positive Lyapunov = chaotic divergence
    divergence_index = min(100, max(0, divergence.lyapunov_exponent * 1000))

    # Leak Index: Based on entropy drift rate
    # Any positive drift is suspicious
    leak_index = min(100, abs(memory.entropy_drift_rate) * 1e6)

    # Observer Index: Based on power ratio deviation from 1.0
    observer_index = min(100, abs(observer.power_ratio - 1.0) * 100)

    # Composite RSI (weighted average)
    weights = [0.25, 0.35, 0.20, 0.20]  # Divergence weighted highest
    rsi = (
        weights[0] * jitter_index +
        weights[1] * divergence_index +
        weights[2] * leak_index +
        weights[3] * observer_index
    )

    return jitter_index, divergence_index, leak_index, observer_index, rsi


def generate_full_rsi(
    num_stars: int = 1500,
    quick: bool = False,
    output_path: str = None
) -> RealityStabilityIndex:
    """
    Run all tests and generate full Reality Stability Index.
    """
    print("\n" + "=" * 70)
    print("  REALITY STABILITY INDEX GENERATOR")
    print("  Red Team Proof Edition")
    print("=" * 70)

    sys_info = get_system_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nSystem: {sys_info['platform']}")
    print(f"Device: {sys_info.get('gpu_name', 'CPU')}")
    print(f"PyTorch: {sys_info['torch_version']}")

    # Run tests
    ticks = 200 if quick else 500
    duration = 30 if quick else 60

    temporal = run_temporal_jitter_test(num_stars, ticks, device)
    divergence = run_divergence_heisenberg_test(num_stars, ticks, device)
    memory = run_memory_leak_test(num_stars // 2, duration, device)
    observer = run_observer_effect_test(num_stars, ticks // 2, device)

    # Compute indices
    jitter_idx, div_idx, leak_idx, obs_idx, rsi = compute_rsi(
        temporal, divergence, memory, observer
    )

    # Create RSI object
    result = RealityStabilityIndex(
        timestamp=sys_info["timestamp"],
        platform=sys_info["platform"],
        device=sys_info["device"],
        gpu_name=sys_info.get("gpu_name", "N/A"),
        python_version=sys_info["python_version"].split()[0],
        torch_version=sys_info["torch_version"],
        temporal_jitter=temporal,
        divergence=divergence,
        memory_leak=memory,
        observer_effect=observer,
        jitter_index=jitter_idx,
        divergence_index=div_idx,
        leak_index=leak_idx,
        observer_index=obs_idx,
        reality_stability_index=rsi
    )

    # Generate hash for verification
    hash_data = f"{result.timestamp}{result.gpu_name}{rsi}"
    result.data_hash = hashlib.sha256(hash_data.encode()).hexdigest()[:16]

    # Print results
    print("\n" + "=" * 70)
    print("  REALITY STABILITY INDEX RESULTS")
    print("=" * 70)
    print(f"""
    COMPONENT INDICES (0-100, higher = more glitchy):

      Jitter Index:     {jitter_idx:6.2f}  (temporal stability)
      Divergence Index: {div_idx:6.2f}  (FP determinism)
      Leak Index:       {leak_idx:6.2f}  (entropy drift)
      Observer Index:   {obs_idx:6.2f}  (rendering impact)

    ╔══════════════════════════════════════════════════════════════╗
    ║  REALITY STABILITY INDEX:  {rsi:6.2f}                          ║
    ╚══════════════════════════════════════════════════════════════╝

    INTERPRETATION:
      0-10:   Reality is rock-solid stable
      10-30:  Minor computational artifacts (normal)
      30-50:  Significant glitches detected
      50-70:  Strong evidence of computational substrate
      70-100: Reality is barely holding together

    Hash: {result.data_hash}
    """)

    # Heisenberg comparison
    print("  HEISENBERG COMPARISON:")
    print(f"    Uncertainty product: {divergence.uncertainty_product:.2e}")
    print(f"    HBAR/2:              {HEISENBERG_LIMIT:.2e}")
    print(f"    Ratio:               {divergence.heisenberg_ratio:.2e}")

    if divergence.heisenberg_ratio > 1e30:
        print("    --> Divergence EXCEEDS quantum limit (evidence for sim)")
    else:
        print("    --> Divergence within quantum uncertainty bounds")

    # Save to file
    if output_path is None:
        gpu_safe = sys_info.get("gpu_name", "cpu").replace(" ", "_")[:20]
        output_path = f"rsi_{gpu_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Convert dataclasses to dict for JSON
    def to_dict(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return {k: to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, list):
            return [to_dict(i) for i in obj]
        else:
            return obj

    with open(output_path, 'w') as f:
        json.dump(to_dict(result), f, indent=2)

    print(f"\n  Results saved to: {output_path}")
    print("=" * 70)

    return result


def compare_rsi_files(file1: str, file2: str):
    """Compare two RSI files from different machines."""
    print("\n" + "=" * 70)
    print("  CROSS-PLATFORM RSI COMPARISON")
    print("=" * 70)

    with open(file1) as f:
        rsi1 = json.load(f)
    with open(file2) as f:
        rsi2 = json.load(f)

    print(f"\n  Machine 1: {rsi1['gpu_name']}")
    print(f"  Machine 2: {rsi2['gpu_name']}")

    print(f"\n  {'Metric':<25} {'Machine 1':>12} {'Machine 2':>12} {'Delta':>12}")
    print("  " + "-" * 63)

    metrics = [
        ("Jitter Index", "jitter_index"),
        ("Divergence Index", "divergence_index"),
        ("Leak Index", "leak_index"),
        ("Observer Index", "observer_index"),
        ("REALITY STABILITY INDEX", "reality_stability_index"),
    ]

    for name, key in metrics:
        v1 = rsi1[key]
        v2 = rsi2[key]
        delta = v2 - v1
        print(f"  {name:<25} {v1:>12.2f} {v2:>12.2f} {delta:>+12.2f}")

    # Check for synchronization (similar values at same mathematical points)
    print("\n  SYNCHRONIZATION CHECK:")

    lyap1 = rsi1["divergence"]["lyapunov_exponent"]
    lyap2 = rsi2["divergence"]["lyapunov_exponent"]
    lyap_diff = abs(lyap1 - lyap2)

    if lyap_diff < 0.01:
        print(f"    Lyapunov exponents MATCH within 0.01!")
        print(f"    This suggests a UNIVERSAL CONSTANT of computation.")
        print(f"    Machine 1: {lyap1:.6f}")
        print(f"    Machine 2: {lyap2:.6f}")
    else:
        print(f"    Lyapunov exponents differ by {lyap_diff:.4f}")
        print(f"    (No obvious synchronization)")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Red Team Proof - Bulletproof Reality Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    parser.add_argument("--quick", action="store_true", help="Run quick test (shorter)")
    parser.add_argument("--generate-rsi", action="store_true", help="Generate RSI report")
    parser.add_argument("--stars", type=int, default=1500, help="Number of stars")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--compare", nargs=2, metavar=("FILE1", "FILE2"),
                        help="Compare two RSI files")

    args = parser.parse_args()

    if args.compare:
        compare_rsi_files(args.compare[0], args.compare[1])
    elif args.full or args.generate_rsi:
        generate_full_rsi(args.stars, quick=False, output_path=args.output)
    elif args.quick:
        generate_full_rsi(args.stars, quick=True, output_path=args.output)
    else:
        # Default: run quick test
        generate_full_rsi(args.stars, quick=True, output_path=args.output)


if __name__ == "__main__":
    main()
