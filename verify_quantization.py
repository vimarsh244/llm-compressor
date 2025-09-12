#!/usr/bin/env python3
"""
Verify the theoretical correctness of PoT and APoT quantization implementations.
"""

import torch
from llmcompressor.modifiers.quantization.pot.utils import quantize_pot, dequantize_pot
from llmcompressor.modifiers.quantization.apot.utils import quantize_apot, dequantize_apot


def verify_pot_quantization():
    """Verify PoT quantization follows standard formulas."""
    print("Verifying PoT quantization...")
    
    # test case 1: simple values with scale=1, zero_point=0
    tensor = torch.tensor([0.0, 1.0, 2.0, 4.0, -1.0, -2.0, -4.0])
    scale = torch.tensor(1.0)
    zero_point = torch.tensor(0)
    
    print(f"Original tensor: {tensor}")
    print(f"Scale: {scale}, Zero point: {zero_point}")
    
    # quantize
    quantized = quantize_pot(tensor, scale, zero_point, num_bits=8)
    print(f"Quantized: {quantized}")
    
    # dequantize
    dequantized = dequantize_pot(quantized, scale, zero_point)
    print(f"Dequantized: {dequantized}")
    
    # check if roundtrip is exact for integer values
    integer_mask = torch.eq(tensor, torch.round(tensor))
    if torch.any(integer_mask):
        integer_values = tensor[integer_mask]
        dequant_integer_values = dequantized[integer_mask]
        print(f"Integer values recovery: {torch.allclose(integer_values, dequant_integer_values)}")
    
    print()
    
    # test case 2: with non-zero scale
    tensor2 = torch.tensor([0.0, 2.0, 4.0, 8.0])  # multiples of scale
    scale2 = torch.tensor(2.0)  # power of 2
    zero_point2 = torch.tensor(0)
    
    print(f"Test 2 - Original tensor: {tensor2}")
    print(f"Scale: {scale2}, Zero point: {zero_point2}")
    
    quantized2 = quantize_pot(tensor2, scale2, zero_point2, num_bits=8)
    print(f"Quantized: {quantized2}")
    
    dequantized2 = dequantize_pot(quantized2, scale2, zero_point2)
    print(f"Dequantized: {dequantized2}")
    print(f"Exact recovery: {torch.allclose(tensor2, dequantized2)}")
    
    print()


def verify_apot_quantization():
    """Verify APoT quantization logic."""
    print("Verifying APoT quantization...")
    
    tensor = torch.tensor([1.0, 3.0, 5.0, 7.0])  # values that might benefit from APoT
    scale = torch.tensor(1.0)
    zero_point = torch.tensor(0)
    
    print(f"Original tensor: {tensor}")
    print(f"Scale: {scale}, Zero point: {zero_point}")
    
    # quantize with APoT
    quantized_indices, levels = quantize_apot(tensor, scale, zero_point, num_bits=4, num_terms=2)
    print(f"Quantized indices: {quantized_indices}")
    print(f"Available levels: {levels[:10]}...")  # show first 10 levels
    
    # dequantize
    dequantized = dequantize_apot(quantized_indices, levels, scale, zero_point)
    print(f"Dequantized: {dequantized}")
    
    # check approximation quality
    error = torch.abs(tensor - dequantized)
    print(f"Quantization error: {error}")
    print(f"Max error: {torch.max(error)}")
    
    print()


def verify_standard_quantization_formulas():
    """Verify our implementation follows standard quantization formulas."""
    print("Verifying standard quantization formulas...")
    print("Standard formulas:")
    print("  quantized = round((tensor - zero_point) / scale)")
    print("  dequantized = quantized * scale + zero_point")
    print()
    
    # manual verification
    tensor = torch.tensor([3.0])
    scale = torch.tensor(2.0)
    zero_point = torch.tensor(0.0)
    
    # manual calculation
    manual_quantized = torch.round((tensor - zero_point) / scale)
    manual_dequantized = manual_quantized * scale + zero_point
    
    print(f"Manual calculation:")
    print(f"  tensor: {tensor}")
    print(f"  quantized: round(({tensor} - {zero_point}) / {scale}) = {manual_quantized}")
    print(f"  dequantized: {manual_quantized} * {scale} + {zero_point} = {manual_dequantized}")
    
    # our implementation
    our_quantized = quantize_pot(tensor, scale, zero_point, num_bits=8)
    our_dequantized = dequantize_pot(our_quantized, scale, zero_point)
    
    print(f"Our implementation:")
    print(f"  quantized: {our_quantized}")
    print(f"  dequantized: {our_dequantized}")
    
    print(f"Match: {torch.allclose(manual_dequantized, our_dequantized)}")
    print()


if __name__ == "__main__":
    print("="*60)
    print("VERIFYING QUANTIZATION THEORETICAL CORRECTNESS")
    print("="*60)
    
    verify_standard_quantization_formulas()
    verify_pot_quantization()
    verify_apot_quantization()
    
    print("Verification complete!")