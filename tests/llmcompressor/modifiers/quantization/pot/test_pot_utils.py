"""
Unit tests for Power-of-Two quantization utilities.
"""
import pytest
import torch

from llmcompressor.modifiers.quantization.pot.utils import (
    quantize_pot,
    dequantize_pot,
    fake_quantize_pot,
    compute_pot_scale_shift,
    pot_multiply_by_shift,
)


class TestPoTUtils:
    """Test cases for PoT utility functions."""
    
    def test_quantize_pot_basic(self):
        """Test basic PoT quantization."""
        tensor = torch.tensor([1.0, 2.0, 4.0, 8.0])
        scale = torch.tensor(1.0)  # scale of 1 (2^0)
        zero_point = torch.tensor(0)
        
        quantized = quantize_pot(tensor, scale, zero_point, num_bits=8)
        
        # should be quantized to integers
        expected = torch.tensor([1, 2, 4, 8], dtype=torch.int8)
        assert torch.equal(quantized, expected)
    
    def test_dequantize_pot_basic(self):
        """Test basic PoT dequantization."""
        quantized = torch.tensor([1, 2, 4, 8], dtype=torch.int8)
        scale = torch.tensor(1.0)
        zero_point = torch.tensor(0)
        
        dequantized = dequantize_pot(quantized, scale, zero_point)
        
        # should match original values
        expected = torch.tensor([1.0, 2.0, 4.0, 8.0])
        assert torch.allclose(dequantized, expected)
    
    def test_fake_quantize_pot_roundtrip(self):
        """Test that fake quantization preserves values reasonably."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        scale = torch.tensor(1.0)
        zero_point = torch.tensor(0)
        
        fake_quantized = fake_quantize_pot(tensor, scale, zero_point, num_bits=8)
        
        # should be close to original (within quantization error)
        assert torch.allclose(fake_quantized, tensor, atol=1.0)
    
    def test_quantize_with_power_of_two_scale(self):
        """Test quantization with different power-of-two scales."""
        tensor = torch.tensor([1.0, 2.0, 4.0, 8.0])
        
        # test with scale = 2^1 = 2
        scale = torch.tensor(2.0)
        zero_point = torch.tensor(0)
        
        quantized = quantize_pot(tensor, scale, zero_point, num_bits=8)
        dequantized = dequantize_pot(quantized, scale, zero_point)
        
        # should recover original values (scaled by 2)
        assert torch.allclose(dequantized, tensor, atol=1e-6)
    
    def test_clamping_behavior(self):
        """Test that quantization clamps to the valid range."""
        tensor = torch.tensor([-200.0, 200.0])  # values outside 8-bit range
        scale = torch.tensor(1.0)
        zero_point = torch.tensor(0)
        
        quantized = quantize_pot(tensor, scale, zero_point, num_bits=8)
        
        # should be clamped to [-128, 127] for 8-bit
        assert torch.all(quantized >= -128)
        assert torch.all(quantized <= 127)
    
    def test_compute_pot_scale_shift(self):
        """Test computation of bit shift amounts."""
        # test various power-of-two scales
        scales = torch.tensor([1.0, 2.0, 4.0, 8.0, 0.5, 0.25])
        expected_shifts = torch.tensor([0, 1, 2, 3, -1, -2], dtype=torch.int32)
        
        computed_shifts = compute_pot_scale_shift(scales)
        
        assert torch.equal(computed_shifts, expected_shifts)
    
    def test_pot_multiply_by_shift(self):
        """Test multiplication using bit shifts."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # test multiplication by 4 (2^2)
        scale = torch.tensor(4.0)
        result = pot_multiply_by_shift(tensor, scale)
        expected = tensor * 4.0
        
        assert torch.allclose(result, expected)
        
        # test division by 4 (inverse=True)
        result_div = pot_multiply_by_shift(tensor, scale, inverse=True)
        expected_div = tensor / 4.0
        
        assert torch.allclose(result_div, expected_div)
    
    def test_different_bit_widths(self):
        """Test quantization with different bit widths."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        scale = torch.tensor(1.0)
        zero_point = torch.tensor(0)
        
        for num_bits in [2, 4, 8]:
            quantized = quantize_pot(tensor, scale, zero_point, num_bits=num_bits)
            
            # check that values are within the expected range
            qmin = -(2 ** (num_bits - 1))
            qmax = 2 ** (num_bits - 1) - 1
            
            assert torch.all(quantized >= qmin)
            assert torch.all(quantized <= qmax)
    
    def test_negative_values(self):
        """Test quantization with negative values."""
        tensor = torch.tensor([-4.0, -2.0, -1.0, 1.0, 2.0, 4.0])
        scale = torch.tensor(1.0)
        zero_point = torch.tensor(0)
        
        quantized = quantize_pot(tensor, scale, zero_point, num_bits=8)
        dequantized = dequantize_pot(quantized, scale, zero_point)
        
        # should preserve negative values
        assert torch.allclose(dequantized, tensor, atol=1e-6)
    
    def test_zero_point_handling(self):
        """Test that zero point is handled correctly (should be 0 for PoT)."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        scale = torch.tensor(1.0)
        zero_point = torch.tensor(0)  # PoT should always use 0
        
        # test with non-zero zero_point (should still work but not typical for PoT)
        zero_point_nonzero = torch.tensor(1)
        
        quantized = quantize_pot(tensor, scale, zero_point_nonzero, num_bits=8)
        dequantized = dequantize_pot(quantized, scale, zero_point_nonzero)
        
        # should still work (though not typical for PoT)
        assert torch.allclose(dequantized, tensor, atol=1.0)
    
    def test_edge_case_very_small_values(self):
        """Test quantization with very small values."""
        tensor = torch.tensor([1e-6, 2e-6, -1e-6])
        scale = torch.tensor(1e-6)  # very small scale
        zero_point = torch.tensor(0)
        
        fake_quantized = fake_quantize_pot(tensor, scale, zero_point, num_bits=8)
        
        # should handle small values without crashing
        assert not torch.any(torch.isnan(fake_quantized))
        assert not torch.any(torch.isinf(fake_quantized))