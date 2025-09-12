"""
Unit tests for Additive Power-of-Two quantization utilities.
"""
import pytest
import torch

from llmcompressor.modifiers.quantization.apot.utils import (
    generate_apot_levels,
    find_nearest_apot_level,
    quantize_apot,
    dequantize_apot,
    fake_quantize_apot,
    decompose_apot_level,
    apot_multiply_by_shift_add,
)


class TestAPoTUtils:
    """Test cases for APoT utility functions."""
    
    def test_generate_apot_levels_single_term(self):
        """Test generation of APoT levels for single term (equivalent to PoT)."""
        levels = generate_apot_levels(num_bits=3, num_terms=1)
        
        # should include powers of two and their negatives
        assert 0.0 in levels
        assert 1.0 in levels or 2.0 in levels  # at least some power of 2
        assert -1.0 in levels or -2.0 in levels  # at least some negative power of 2
    
    def test_generate_apot_levels_two_terms(self):
        """Test generation of APoT levels for two terms."""
        levels = generate_apot_levels(num_bits=4, num_terms=2)
        
        # should have more levels than single term
        single_term_levels = generate_apot_levels(num_bits=4, num_terms=1)
        assert len(levels) > len(single_term_levels)
        
        # should include zero
        assert 0.0 in levels
        
        # should be symmetric around zero
        positive_levels = [l for l in levels if l > 0]
        negative_levels = [l for l in levels if l < 0]
        assert len(positive_levels) > 0
        assert len(negative_levels) > 0
    
    def test_find_nearest_apot_level(self):
        """Test finding nearest APoT level."""
        levels = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
        
        # test exact matches
        assert find_nearest_apot_level(2.0, levels) == 2.0
        assert find_nearest_apot_level(-1.0, levels) == -1.0
        
        # test approximate matches
        assert find_nearest_apot_level(1.8, levels) == 2.0
        assert find_nearest_apot_level(-1.8, levels) == -2.0
        
        # test edge cases
        assert find_nearest_apot_level(10.0, levels) == 4.0  # clamp to max
        assert find_nearest_apot_level(-10.0, levels) == -4.0  # clamp to min
    
    def test_quantize_apot_basic(self):
        """Test basic APoT quantization."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        scale = torch.tensor(1.0)
        zero_point = torch.tensor(0)
        
        quantized_indices, levels = quantize_apot(
            tensor, scale, zero_point, num_bits=4, num_terms=2
        )
        
        # should return indices and levels
        assert quantized_indices.shape == tensor.shape
        assert len(levels) > 0
        assert isinstance(levels, list)
        
        # indices should be valid
        assert torch.all(quantized_indices >= 0)
        assert torch.all(quantized_indices < len(levels))
    
    def test_dequantize_apot_basic(self):
        """Test basic APoT dequantization."""
        levels = [-2.0, -1.0, 0.0, 1.0, 2.0]
        quantized_indices = torch.tensor([0, 2, 4])  # -2, 0, 2
        scale = torch.tensor(1.0)
        zero_point = torch.tensor(0)
        
        dequantized = dequantize_apot(quantized_indices, levels, scale, zero_point)
        
        expected = torch.tensor([-2.0, 0.0, 2.0])
        assert torch.allclose(dequantized, expected)
    
    def test_fake_quantize_apot_roundtrip(self):
        """Test that fake quantization preserves values reasonably."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        scale = torch.tensor(1.0)
        zero_point = torch.tensor(0)
        
        fake_quantized = fake_quantize_apot(
            tensor, scale, zero_point, num_bits=4, num_terms=2
        )
        
        # should be close to original (within quantization error)
        assert torch.allclose(fake_quantized, tensor, atol=2.0)
        assert fake_quantized.shape == tensor.shape
    
    def test_quantize_with_scaling(self):
        """Test APoT quantization with different scales."""
        tensor = torch.tensor([2.0, 4.0, 6.0, 8.0])
        scale = torch.tensor(2.0)  # scale by 2
        zero_point = torch.tensor(0)
        
        quantized_indices, levels = quantize_apot(
            tensor, scale, zero_point, num_bits=4, num_terms=2
        )
        dequantized = dequantize_apot(quantized_indices, levels, scale, zero_point)
        
        # should approximately recover original values
        assert torch.allclose(dequantized, tensor, atol=2.0)
    
    def test_decompose_apot_level(self):
        """Test decomposition of APoT levels into power-of-two components."""
        # test zero
        components = decompose_apot_level(0.0, num_terms=2)
        assert len(components) == 2
        assert all(sign == 0 for sign, exp in components)
        
        # test positive value
        components = decompose_apot_level(5.0, num_terms=2)  # could be 4+1 = 2^2 + 2^0
        assert len(components) == 2
        
        # test negative value
        components = decompose_apot_level(-3.0, num_terms=2)
        assert len(components) == 2
    
    def test_apot_multiply_by_shift_add(self):
        """Test APoT multiplication using shifts and adds."""
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # test with components representing 2^2 + 2^0 = 4 + 1 = 5
        components = [(1, 2), (1, 0)]  # +2^2 + 2^0
        
        result = apot_multiply_by_shift_add(tensor, components)
        expected = tensor * 5.0  # should multiply by 5
        
        assert torch.allclose(result, expected)
        
        # test with subtraction: 2^3 - 2^1 = 8 - 2 = 6
        components = [(1, 3), (-1, 1)]  # +2^3 - 2^1
        
        result = apot_multiply_by_shift_add(tensor, components)
        expected = tensor * 6.0  # should multiply by 6
        
        assert torch.allclose(result, expected)
    
    def test_different_num_terms(self):
        """Test APoT with different numbers of terms."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        scale = torch.tensor(1.0)
        zero_point = torch.tensor(0)
        
        for num_terms in [1, 2, 3]:
            fake_quantized = fake_quantize_apot(
                tensor, scale, zero_point, num_bits=4, num_terms=num_terms
            )
            
            # should handle different numbers of terms
            assert fake_quantized.shape == tensor.shape
            assert not torch.any(torch.isnan(fake_quantized))
    
    def test_different_bit_widths(self):
        """Test APoT with different bit widths."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        scale = torch.tensor(1.0)
        zero_point = torch.tensor(0)
        
        for num_bits in [3, 4, 5]:
            fake_quantized = fake_quantize_apot(
                tensor, scale, zero_point, num_bits=num_bits, num_terms=2
            )
            
            # should handle different bit widths
            assert fake_quantized.shape == tensor.shape
            assert not torch.any(torch.isnan(fake_quantized))
    
    def test_negative_values(self):
        """Test APoT quantization with negative values."""
        tensor = torch.tensor([-4.0, -2.0, -1.0, 1.0, 2.0, 4.0])
        scale = torch.tensor(1.0)
        zero_point = torch.tensor(0)
        
        fake_quantized = fake_quantize_apot(
            tensor, scale, zero_point, num_bits=4, num_terms=2
        )
        
        # should handle negative values
        assert fake_quantized.shape == tensor.shape
        assert not torch.any(torch.isnan(fake_quantized))
        
        # should preserve sign roughly
        positive_mask = tensor > 0
        negative_mask = tensor < 0
        
        if torch.any(positive_mask):
            assert torch.mean(fake_quantized[positive_mask]) > 0
        if torch.any(negative_mask):
            assert torch.mean(fake_quantized[negative_mask]) < 0
    
    def test_edge_case_small_values(self):
        """Test APoT with very small values."""
        tensor = torch.tensor([1e-3, 2e-3, -1e-3])
        scale = torch.tensor(1e-3)
        zero_point = torch.tensor(0)
        
        fake_quantized = fake_quantize_apot(
            tensor, scale, zero_point, num_bits=4, num_terms=2
        )
        
        # should handle small values without crashing
        assert not torch.any(torch.isnan(fake_quantized))
        assert not torch.any(torch.isinf(fake_quantized))
    
    def test_levels_are_sorted(self):
        """Test that generated APoT levels are sorted."""
        for num_bits in [3, 4, 5]:
            for num_terms in [1, 2, 3]:
                levels = generate_apot_levels(num_bits, num_terms)
                
                # should be sorted in ascending order
                assert levels == sorted(levels)
    
    def test_zero_always_in_levels(self):
        """Test that zero is always included in APoT levels."""
        for num_bits in [3, 4, 5]:
            for num_terms in [1, 2]:
                levels = generate_apot_levels(num_bits, num_terms)
                assert 0.0 in levels