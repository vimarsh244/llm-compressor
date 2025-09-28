"""
Utility functions for Power-of-Two quantization.
"""
import torch
from torch import Tensor
from typing import Optional


def quantize_pot(
    tensor: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    num_bits: int = 8,
) -> Tensor:
    """
    Quantize a tensor using Power-of-Two quantization.
    
    True PoT quantization constrains values to exact powers of two:
    0, ±2^0, ±2^1, ±2^2, ..., ±2^(num_bits-2)
    
    :param tensor: tensor to quantize
    :param scale: scale factor (must be a power of two)
    :param zero_point: zero point (should be 0 for symmetric PoT)
    :param num_bits: number of bits for quantization
    :return: quantized tensor (indices into PoT levels)
    """
    # Generate PoT quantization levels: 0, ±1, ±2, ±4, ±8, ...
    max_exp = num_bits - 2  # reserve one bit for sign, one for range
    pot_levels = [0.0]  # include zero
    
    # Add positive and negative powers of two
    for exp in range(max_exp + 1):
        level = 2.0 ** exp
        pot_levels.extend([level, -level])
    
    # Sort levels for easier lookup
    pot_levels = sorted(pot_levels)
    
    # Scale the input tensor
    scaled = (tensor - zero_point) / scale
    
    # Find nearest PoT level for each element
    quantized = torch.zeros_like(scaled, dtype=torch.int32)
    
    # Vectorized approach for efficiency
    scaled_flat = scaled.flatten()
    quantized_flat = quantized.flatten()
    
    for i, val in enumerate(scaled_flat):
        # Find the nearest PoT level
        val_item = val.item()
        nearest_idx = min(range(len(pot_levels)), 
                         key=lambda j: abs(pot_levels[j] - val_item))
        quantized_flat[i] = nearest_idx
    
    return quantized_flat.reshape(tensor.shape)


def dequantize_pot(
    quantized: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    num_bits: int = 8,
) -> Tensor:
    """
    Dequantize a PoT-quantized tensor.
    
    Maps quantized indices back to PoT levels, then applies scale and zero_point.
    
    :param quantized: quantized tensor (indices into PoT levels)
    :param scale: scale factor (must be a power of two)
    :param zero_point: zero point (should be 0 for symmetric PoT)
    :param num_bits: number of bits for quantization
    :return: dequantized tensor
    """
    # Regenerate PoT levels (same as in quantize_pot)
    max_exp = num_bits - 2
    pot_levels = [0.0]
    
    for exp in range(max_exp + 1):
        level = 2.0 ** exp
        pot_levels.extend([level, -level])
    
    pot_levels = sorted(pot_levels)
    
    # Convert to tensor for efficient lookup
    levels_tensor = torch.tensor(pot_levels, dtype=torch.float32, device=quantized.device)
    
    # Look up the PoT levels
    dequantized = levels_tensor[quantized.long()]
    
    # Apply scale and zero point
    dequantized = dequantized * scale + zero_point.to(torch.float32)
    
    return dequantized


def fake_quantize_pot(
    tensor: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    num_bits: int = 8,
) -> Tensor:
    """
    Fake quantize a tensor using PoT quantization (quantize then dequantize).
    
    This constrains values to exact powers of two during training.
    
    :param tensor: tensor to fake quantize
    :param scale: scale factor (must be a power of two)
    :param zero_point: zero point (should be 0 for symmetric PoT)
    :param num_bits: number of bits for quantization
    :return: fake quantized tensor
    """
    quantized = quantize_pot(tensor, scale, zero_point, num_bits)
    dequantized = dequantize_pot(quantized, scale, zero_point, num_bits)
    return dequantized


def compute_pot_scale_shift(scale: Tensor) -> Tensor:
    """
    Compute the bit shift amount for a PoT scale.
    
    Since scale = 2^shift, we can compute shift = log2(scale).
    This is useful for hardware implementation using bit shifts.
    
    :param scale: PoT scale factor
    :return: shift amount
    """
    # compute log2 of the scale
    shift = torch.log2(scale)
    
    # round to nearest integer (should already be integer for true PoT)
    shift = torch.round(shift)
    
    return shift.to(torch.int32)


def pot_multiply_by_shift(
    tensor: Tensor,
    scale: Tensor,
    inverse: bool = False,
) -> Tensor:
    """
    Multiply a tensor by a PoT scale using bit shifts.
    
    :param tensor: input tensor
    :param scale: PoT scale factor
    :param inverse: if True, divide by scale instead of multiply
    :return: scaled tensor
    """
    # compute the shift amount
    shift = compute_pot_scale_shift(scale)
    
    if inverse:
        # for division, shift right (negative shift)
        shift = -shift
    
    # perform the shift operation
    # note: torch doesn't have a direct bit shift op, so we use multiplication
    # in hardware, this would be implemented as actual bit shifts
    result = tensor * torch.pow(2.0, shift.to(torch.float32))
    
    return result