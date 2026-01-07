"""
Quantization module.
Implements various precision degradation modes to simulate "lossy" physics.
"""

import torch
from enum import Enum


class PrecisionMode(Enum):
    """Available precision modes for the simulation."""
    FLOAT64 = "float64"      # Full precision baseline
    FLOAT32 = "float32"      # Standard GPU precision
    BFLOAT16 = "bfloat16"    # Brain Float - AI precision (same range as f32, less mantissa)
    FLOAT16 = "float16"      # Half precision
    INT8_SIM = "int8_sim"    # Simulated 8-bit quantization
    INT4_SIM = "int4_sim"    # Simulated 4-bit quantization (extreme)
    CUSTOM = "custom"        # User-defined quantization level


def quantize_distance_squared(
    dist_sq: torch.Tensor,
    mode: PrecisionMode,
    custom_levels: int = None,
    min_dist_sq: float = 0.01  # Softening floor - CRITICAL for stability
) -> torch.Tensor:
    """
    Apply precision degradation to distance squared values.
    This is the "broken math" that creates ghost forces.

    IMPORTANT: min_dist_sq prevents quantization from creating
    near-zero distances that cause infinite forces.

    Args:
        dist_sq: Distance squared tensor
        mode: Precision mode to apply
        custom_levels: Number of quantization levels for CUSTOM mode
        min_dist_sq: Minimum allowed distance squared (softening protection)

    Returns:
        Quantized distance squared values
    """
    if mode == PrecisionMode.FLOAT64:
        # No quantization - baseline
        return dist_sq.double()

    elif mode == PrecisionMode.FLOAT32:
        return dist_sq.float()

    elif mode == PrecisionMode.BFLOAT16:
        # Brain Float 16: same exponent range as float32, but 7-bit mantissa
        # This is what AI models use - fast on RTX 5090
        return dist_sq.bfloat16().float()

    elif mode == PrecisionMode.FLOAT16:
        return dist_sq.half().float()

    elif mode == PrecisionMode.INT8_SIM:
        # Simulate 8-bit: 256 discrete levels
        return _grid_quantize_safe(dist_sq, levels=256, min_val=min_dist_sq)

    elif mode == PrecisionMode.INT4_SIM:
        # Simulate 4-bit: 16 discrete levels (most extreme)
        return _grid_quantize_safe(dist_sq, levels=16, min_val=min_dist_sq)

    elif mode == PrecisionMode.CUSTOM:
        levels = custom_levels or 64
        return _grid_quantize_safe(dist_sq, levels=levels, min_val=min_dist_sq)

    else:
        return dist_sq


def _grid_quantize(tensor: torch.Tensor, levels: int) -> torch.Tensor:
    """
    Basic grid quantization (can cause instability with low levels).
    """
    min_val = tensor.min()
    max_val = tensor.max()

    if max_val - min_val < 1e-10:
        return tensor

    normalized = (tensor - min_val) / (max_val - min_val) * (levels - 1)
    quantized = torch.round(normalized)
    result = quantized / (levels - 1) * (max_val - min_val) + min_val

    return result


def _grid_quantize_safe(
    tensor: torch.Tensor,
    levels: int,
    min_val: float = 0.01
) -> torch.Tensor:
    """
    Safe grid quantization with minimum value enforcement.

    This is the key fix: quantization happens ABOVE the softening floor,
    preventing the "infinite slingshot" effect that caused the explosion.

    The quantization now creates subtle "steps" in gravity without
    allowing any distance to become dangerously small.
    """
    # Enforce minimum BEFORE quantizing
    tensor_safe = tensor.clamp(min=min_val)

    # Use logarithmic quantization for better distribution
    # This gives more resolution at small distances (where it matters)
    log_tensor = torch.log(tensor_safe)

    log_min = log_tensor.min()
    log_max = log_tensor.max()

    if log_max - log_min < 1e-10:
        return tensor_safe

    # Quantize in log space
    normalized = (log_tensor - log_min) / (log_max - log_min) * (levels - 1)
    quantized = torch.round(normalized)
    log_result = quantized / (levels - 1) * (log_max - log_min) + log_min

    # Convert back from log space
    result = torch.exp(log_result)

    # Final safety clamp
    return result.clamp(min=min_val)


def quantize_force(
    force: torch.Tensor,
    mode: PrecisionMode,
    custom_levels: int = None
) -> torch.Tensor:
    """
    Alternative: quantize the force values directly.
    Can be used in addition to distance quantization.
    """
    if mode in [PrecisionMode.FLOAT64, PrecisionMode.FLOAT32]:
        return force

    elif mode == PrecisionMode.BFLOAT16:
        return force.bfloat16().float()

    elif mode == PrecisionMode.FLOAT16:
        return force.half().float()

    elif mode == PrecisionMode.INT8_SIM:
        return _grid_quantize(force, levels=256)

    elif mode == PrecisionMode.INT4_SIM:
        return _grid_quantize(force, levels=16)

    elif mode == PrecisionMode.CUSTOM:
        return _grid_quantize(force, levels=custom_levels or 64)

    return force


def get_mode_from_string(mode_str: str) -> PrecisionMode:
    """Convert string to PrecisionMode enum."""
    mode_map = {
        "float64": PrecisionMode.FLOAT64,
        "float32": PrecisionMode.FLOAT32,
        "bfloat16": PrecisionMode.BFLOAT16,
        "bf16": PrecisionMode.BFLOAT16,
        "float16": PrecisionMode.FLOAT16,
        "fp16": PrecisionMode.FLOAT16,
        "int8": PrecisionMode.INT8_SIM,
        "int8_sim": PrecisionMode.INT8_SIM,
        "int4": PrecisionMode.INT4_SIM,
        "int4_sim": PrecisionMode.INT4_SIM,
        "custom": PrecisionMode.CUSTOM,
    }
    return mode_map.get(mode_str.lower(), PrecisionMode.FLOAT64)


def describe_mode(mode: PrecisionMode) -> str:
    """Get human-readable description of precision mode."""
    descriptions = {
        PrecisionMode.FLOAT64: "64-bit float (baseline)",
        PrecisionMode.FLOAT32: "32-bit float (standard GPU)",
        PrecisionMode.BFLOAT16: "Brain Float 16 (AI precision, fast on RTX)",
        PrecisionMode.FLOAT16: "16-bit float (half precision)",
        PrecisionMode.INT8_SIM: "Simulated 8-bit (256 levels)",
        PrecisionMode.INT4_SIM: "Simulated 4-bit (16 levels)",
        PrecisionMode.CUSTOM: "Custom quantization levels",
    }
    return descriptions.get(mode, "Unknown mode")
