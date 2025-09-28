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
    
    Standard quantization formula: quantized = round((tensor - zero_point) / scale)
    
    :param tensor: tensor to quantize
    :param scale: scale factor (must be a power of two)
    :param zero_point: zero point (should be 0 for symmetric PoT)
    :param num_bits: number of bits for quantization
    :return: quantized tensor
    """
    # apply standard quantization formula: (tensor - zero_point) / scale
    scaled = (tensor - zero_point) / scale
    
    # round to nearest integer
    rounded = torch.round(scaled)
    
    # clamp to the quantization range
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    clamped = torch.clamp(rounded, qmin, qmax)
    
    return clamped.to(torch.int8 if num_bits <= 8 else torch.int32)


def dequantize_pot(
    quantized: Tensor,
    scale: Tensor,
    zero_point: Tensor,
) -> Tensor:
    """
    Dequantize a PoT-quantized tensor.
    
    Standard dequantization formula: dequantized = quantized * scale + zero_point
    
    :param quantized: quantized tensor
    :param scale: scale factor (must be a power of two)
    :param zero_point: zero point (should be 0 for symmetric PoT)
    :return: dequantized tensor
    """
    # apply standard dequantization formula: quantized * scale + zero_point
    dequantized = quantized.to(torch.float32) * scale + zero_point.to(torch.float32)
    
    return dequantized


def fake_quantize_pot(
    tensor: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    num_bits: int = 8,
) -> Tensor:
    """
    Fake quantize a tensor using PoT quantization (quantize then dequantize).
    
    :param tensor: tensor to fake quantize
    :param scale: scale factor (must be a power of two)
    :param zero_point: zero point (should be 0 for symmetric PoT)
    :param num_bits: number of bits for quantization
    :return: fake quantized tensor
    """
    quantized = quantize_pot(tensor, scale, zero_point, num_bits)
    dequantized = dequantize_pot(quantized, scale, zero_point)
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