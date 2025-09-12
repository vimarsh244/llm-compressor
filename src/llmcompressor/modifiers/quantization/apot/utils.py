"""
Utility functions for Additive Power-of-Two quantization.
"""
import torch
from torch import Tensor
from typing import List, Optional, Tuple
import itertools


def generate_apot_levels(num_bits: int, num_terms: int = 2) -> List[float]:
    """
    Generate the quantization levels for APoT quantization.
    
    For k-term APoT, values are represented as sums of k signed powers of two.
    
    :param num_bits: number of bits for quantization
    :param num_terms: number of power-of-two terms
    :return: list of quantization levels
    """
    # generate the base powers of two
    # we use num_bits-1 to account for the sign bit
    max_exp = num_bits - 2
    min_exp = -(num_bits - 2)
    
    powers = [2.0 ** i for i in range(min_exp, max_exp + 1)]
    powers.append(0.0)  # include zero
    
    # generate all possible sums of num_terms powers
    levels = set()
    
    if num_terms == 1:
        # single term: just powers of two
        for p in powers:
            levels.add(p)
            levels.add(-p)
    elif num_terms == 2:
        # two terms: sums and differences of pairs
        for p1, p2 in itertools.combinations_with_replacement(powers, 2):
            if p1 != 0 or p2 != 0:  # exclude (0, 0)
                levels.add(p1 + p2)
                levels.add(p1 - p2)
                levels.add(-p1 - p2)
                levels.add(-p1 + p2)
    else:
        # general case: all combinations of num_terms
        for combo in itertools.combinations_with_replacement(powers, num_terms):
            # generate all sign combinations
            for signs in itertools.product([-1, 1], repeat=num_terms):
                level = sum(s * p for s, p in zip(signs, combo))
                if level != 0 or all(p == 0 for p in combo):
                    levels.add(level)
    
    # convert to sorted list
    levels = sorted(list(levels))
    
    return levels


def find_nearest_apot_level(value: float, levels: List[float]) -> float:
    """
    Find the nearest APoT quantization level for a given value.
    
    :param value: value to quantize
    :param levels: list of APoT quantization levels
    :return: nearest quantization level
    """
    # binary search for efficiency
    left, right = 0, len(levels) - 1
    
    # handle edge cases
    if value <= levels[0]:
        return levels[0]
    if value >= levels[-1]:
        return levels[-1]
    
    # find the two closest levels
    while right - left > 1:
        mid = (left + right) // 2
        if levels[mid] < value:
            left = mid
        else:
            right = mid
    
    # return the closer one
    if abs(value - levels[left]) < abs(value - levels[right]):
        return levels[left]
    else:
        return levels[right]


def quantize_apot(
    tensor: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    num_bits: int = 4,
    num_terms: int = 2,
) -> Tuple[Tensor, List[float]]:
    """
    Quantize a tensor using Additive Power-of-Two quantization.
    
    :param tensor: tensor to quantize
    :param scale: scale factor
    :param zero_point: zero point (should be 0 for symmetric APoT)
    :param num_bits: number of bits for quantization
    :param num_terms: number of power-of-two terms
    :return: tuple of (quantized indices, quantization levels)
    """
    # generate APoT levels
    levels = generate_apot_levels(num_bits, num_terms)
    
    # scale the tensor
    scaled = tensor / scale
    
    # flatten for easier processing
    original_shape = scaled.shape
    scaled_flat = scaled.flatten()
    
    # find nearest APoT level for each value
    quantized_indices = torch.zeros_like(scaled_flat, dtype=torch.int32)
    
    for i, val in enumerate(scaled_flat):
        # find the index of the nearest level
        nearest_level = find_nearest_apot_level(val.item(), levels)
        quantized_indices[i] = levels.index(nearest_level)
    
    # reshape back
    quantized_indices = quantized_indices.reshape(original_shape)
    
    return quantized_indices, levels


def dequantize_apot(
    quantized_indices: Tensor,
    levels: List[float],
    scale: Tensor,
    zero_point: Tensor,
) -> Tensor:
    """
    Dequantize an APoT-quantized tensor.
    
    :param quantized_indices: indices into the quantization levels
    :param levels: list of APoT quantization levels
    :param scale: scale factor
    :param zero_point: zero point (should be 0 for symmetric APoT)
    :return: dequantized tensor
    """
    # convert levels to tensor
    levels_tensor = torch.tensor(levels, dtype=torch.float32, device=quantized_indices.device)
    
    # look up the quantization levels
    dequantized = levels_tensor[quantized_indices.long()]
    
    # multiply by scale
    dequantized = dequantized * scale
    
    return dequantized


def fake_quantize_apot(
    tensor: Tensor,
    scale: Tensor,
    zero_point: Tensor,
    num_bits: int = 4,
    num_terms: int = 2,
) -> Tensor:
    """
    Fake quantize a tensor using APoT quantization (quantize then dequantize).
    
    :param tensor: tensor to fake quantize
    :param scale: scale factor
    :param zero_point: zero point (should be 0 for symmetric APoT)
    :param num_bits: number of bits for quantization
    :param num_terms: number of power-of-two terms
    :return: fake quantized tensor
    """
    quantized_indices, levels = quantize_apot(tensor, scale, zero_point, num_bits, num_terms)
    dequantized = dequantize_apot(quantized_indices, levels, scale, zero_point)
    return dequantized


def decompose_apot_level(level: float, num_terms: int = 2) -> List[Tuple[int, int]]:
    """
    Decompose an APoT level into its power-of-two components.
    
    :param level: the quantization level
    :param num_terms: number of terms expected
    :return: list of (sign, exponent) pairs
    """
    # this is a simplified version - a full implementation would need
    # a more sophisticated algorithm to find the optimal decomposition
    
    if level == 0:
        return [(0, 0)] * num_terms
    
    # for now, use a greedy approach
    components = []
    remaining = abs(level)
    sign = 1 if level > 0 else -1
    
    for _ in range(num_terms):
        if remaining == 0:
            components.append((0, 0))
        else:
            # find the largest power of 2 less than or equal to remaining
            exp = int(torch.log2(torch.tensor(remaining)).floor().item())
            components.append((sign, exp))
            remaining -= 2.0 ** exp
    
    return components


def apot_multiply_by_shift_add(
    tensor: Tensor,
    apot_components: List[Tuple[int, int]],
) -> Tensor:
    """
    Multiply a tensor by an APoT value using shifts and additions.
    
    :param tensor: input tensor
    :param apot_components: list of (sign, exponent) pairs
    :return: scaled tensor
    """
    result = torch.zeros_like(tensor)
    
    for sign, exp in apot_components:
        if sign != 0:
            # shift the tensor by exp positions
            shifted = tensor * (2.0 ** exp)
            # add or subtract based on sign
            result = result + sign * shifted
    
    return result