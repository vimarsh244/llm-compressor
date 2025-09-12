"""
Unit tests for Power-of-Two Observer.
"""
import pytest
import torch
from compressed_tensors.quantization.quant_args import QuantizationArgs

from llmcompressor.observers.pot_apot import PoTObserver


class TestPoTObserver:
    """Test cases for PoT Observer."""
    
    def test_pot_observer_creation(self):
        """Test that PoT observer can be created."""
        quant_args = QuantizationArgs(
            num_bits=4,
            symmetric=True,
            type="int",
        )
        observer = PoTObserver(quant_args)
        assert observer.quantization_args.num_bits == 4
        assert observer.quantization_args.symmetric is True
    
    def test_calculate_qparams_basic(self):
        """Test basic quantization parameter calculation."""
        quant_args = QuantizationArgs(
            num_bits=4,
            symmetric=True,
            type="int",
        )
        observer = PoTObserver(quant_args)
        
        # create a simple tensor
        tensor = torch.tensor([1.0, 2.0, 4.0, 8.0])
        
        scale, zero_point = observer.calculate_qparams(tensor)
        
        # check that scale is a power of two
        log2_scale = torch.log2(scale)
        assert torch.allclose(log2_scale, torch.round(log2_scale), atol=1e-6)
        
        # check that zero point is zero for symmetric quantization
        assert torch.allclose(zero_point, torch.zeros_like(zero_point))
    
    def test_calculate_qparams_with_reduce_dims(self):
        """Test quantization parameter calculation with dimension reduction."""
        quant_args = QuantizationArgs(
            num_bits=8,
            symmetric=True,
            type="int",
        )
        observer = PoTObserver(quant_args)
        
        # create a 2D tensor
        tensor = torch.tensor([[1.0, 2.0], [4.0, 8.0]])
        
        # calculate per-channel parameters (reduce dim 0)
        scale, zero_point = observer.calculate_qparams(tensor, reduce_dims=(0,))
        
        assert scale.shape == (2,)  # should have 2 scales
        assert zero_point.shape == (2,)  # should have 2 zero points
        
        # all scales should be powers of two
        log2_scales = torch.log2(scale)
        assert torch.allclose(log2_scales, torch.round(log2_scales), atol=1e-6)
    
    def test_get_qparams_along_dim(self):
        """Test getting quantization parameters along a specific dimension."""
        quant_args = QuantizationArgs(
            num_bits=4,
            symmetric=True,
            type="int",
        )
        observer = PoTObserver(quant_args)
        
        # create a 3D tensor
        tensor = torch.randn(2, 3, 4)
        
        # get parameters along dimension 1
        scale, zero_point = observer.get_qparams_along_dim(tensor, dim=1)
        
        assert scale.shape == (3,)  # should have 3 scales (size of dim 1)
        assert zero_point.shape == (3,)
    
    def test_scale_is_power_of_two(self):
        """Test that calculated scales are always powers of two."""
        quant_args = QuantizationArgs(
            num_bits=4,
            symmetric=True,
            type="int",
        )
        observer = PoTObserver(quant_args)
        
        # test with various tensor values
        test_tensors = [
            torch.tensor([0.1, 0.5, 1.0]),
            torch.tensor([3.7, 7.2, 15.8]),
            torch.tensor([-1.0, -2.0, -4.0, -8.0]),
            torch.randn(10) * 100,
        ]
        
        for tensor in test_tensors:
            scale, zero_point = observer.calculate_qparams(tensor)
            
            # check that scale is a power of two
            log2_scale = torch.log2(scale)
            assert torch.allclose(log2_scale, torch.round(log2_scale), atol=1e-6)
            
            # check that zero point is zero
            assert torch.allclose(zero_point, torch.zeros_like(zero_point))
    
    def test_edge_cases(self):
        """Test edge cases like zero tensors and very small values."""
        quant_args = QuantizationArgs(
            num_bits=4,
            symmetric=True,
            type="int",
        )
        observer = PoTObserver(quant_args)
        
        # test with zero tensor
        zero_tensor = torch.zeros(5)
        scale, zero_point = observer.calculate_qparams(zero_tensor)
        
        # scale should be positive (clamped to minimum)
        assert torch.all(scale > 0)
        assert torch.allclose(zero_point, torch.zeros_like(zero_point))
        
        # test with very small values
        small_tensor = torch.tensor([1e-8, 2e-8, -1e-8])
        scale, zero_point = observer.calculate_qparams(small_tensor)
        
        # scale should still be positive and a power of two
        assert torch.all(scale > 0)
        log2_scale = torch.log2(scale)
        assert torch.allclose(log2_scale, torch.round(log2_scale), atol=1e-6)
    
    def test_different_bit_widths(self):
        """Test observer with different bit widths."""
        bit_widths = [2, 4, 8]
        
        tensor = torch.tensor([1.0, 2.0, 4.0, 8.0])
        
        for num_bits in bit_widths:
            quant_args = QuantizationArgs(
                num_bits=num_bits,
                symmetric=True,
                type="int",
            )
            observer = PoTObserver(quant_args)
            
            scale, zero_point = observer.calculate_qparams(tensor)
            
            # scale should be a power of two regardless of bit width
            log2_scale = torch.log2(scale)
            assert torch.allclose(log2_scale, torch.round(log2_scale), atol=1e-6)
            
            # zero point should be zero for symmetric quantization
            assert torch.allclose(zero_point, torch.zeros_like(zero_point))
    
    def test_global_scale_application(self):
        """Test that global scale is applied correctly."""
        quant_args = QuantizationArgs(
            num_bits=4,
            symmetric=True,
            type="int",
        )
        observer = PoTObserver(quant_args)
        
        tensor = torch.tensor([1.0, 2.0, 4.0])
        global_scale = torch.tensor(2.0)
        
        scale, zero_point = observer.calculate_qparams(tensor, global_scale=global_scale)
        
        # the resulting scale should still be a power of two
        log2_scale = torch.log2(scale)
        assert torch.allclose(log2_scale, torch.round(log2_scale), atol=1e-6)
        
        # and should be affected by the global scale
        scale_no_global, _ = observer.calculate_qparams(tensor)
        assert not torch.allclose(scale, scale_no_global)