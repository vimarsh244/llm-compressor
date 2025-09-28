"""
Power-of-Two (PoT) Observer for quantization.

This observer calculates quantization parameters for PoT quantization where
all values are constrained to powers of two (e.g., ±2^n).
"""
import torch
from compressed_tensors.quantization.quant_args import QuantizationArgs
from torch import FloatTensor, IntTensor, Tensor
from typing import Optional, Tuple

from llmcompressor.observers.base import Observer

__all__ = ["PoTObserver"]


@Observer.register("pot")
class PoTObserver(Observer):
    """
    Observer for Power-of-Two quantization.
    
    PoT quantization constrains all values to be powers of two, enabling
    efficient computation using bit-shift operations instead of multiplication.
    """
    
    def __init__(
        self,
        quantization_args: QuantizationArgs,
        **kwargs,
    ):
        """
        Initialize PoT observer.
        
        :param quantization_args: quantization arguments
        :param kwargs: additional arguments (for registry compatibility)
        """
        super().__init__(quantization_args)
    
    def calculate_qparams(
        self,
        observed: Tensor,
        reduce_dims: Optional[Tuple[int]] = None,
        tensor_id: Optional[str] = None,
        global_scale: Optional[Tensor] = None,
    ) -> Tuple[FloatTensor, IntTensor]:
        """
        Calculate PoT quantization parameters for the observed tensor.
        
        For PoT quantization:
        - Scale is computed to enable proper PoT quantization levels
        - Zero point is always 0 for symmetric quantization
        - Values will be constrained to exact powers of two during quantization
        
        :param observed: tensor to calculate quantization parameters for
        :param reduce_dims: optional dimensions to reduce along
        :param tensor_id: optional id for the tensor being quantized
        :param global_scale: optional global scale to apply
        :return: tuple of scale and zero point tensors
        """
        if reduce_dims is None:
            reduce_dims = tuple(range(observed.ndim))
        
        # calculate the max absolute value
        max_val = observed.abs().amax(dim=reduce_dims, keepdim=True)
        
        # avoid zero max values
        max_val = torch.clamp(max_val, min=1e-8)
        
        num_bits = self.quantization_args.num_bits
        
        # For true PoT quantization, we need to determine the quantization levels
        # PoT levels are: 0, ±2^0, ±2^1, ±2^2, ..., ±2^(num_bits-2)
        max_pot_level = 2 ** (num_bits - 2)  # largest PoT level
        
        # Calculate scale so that max_val maps to max_pot_level
        # This ensures the full range is utilized properly
        scale = max_val / max_pot_level
        
        # Ensure scale is a power of two for efficient hardware implementation
        scale_log2 = torch.log2(scale)
        scale_exp = torch.round(scale_log2)
        scale = torch.pow(2.0, scale_exp)
        
        # for symmetric quantization, zero point is always 0
        zero_point = torch.zeros_like(scale, dtype=torch.int32)
        
        # apply global scale if provided
        if global_scale is not None:
            scale = scale * global_scale
        
        # handle dimension squeezing properly
        if len(reduce_dims) == observed.ndim:
            # reducing all dimensions - return scalar
            scale = scale.squeeze()
            zero_point = zero_point.squeeze()
        else:
            # reducing some dimensions - squeeze only the reduced ones
            # remove the keepdim dimensions that were reduced
            for dim in sorted(reduce_dims, reverse=True):
                scale = scale.squeeze(dim)
                zero_point = zero_point.squeeze(dim)
        
        return scale.to(torch.float32), zero_point.to(torch.int32)
    
    def get_qparams_along_dim(
        self,
        observed: Tensor,
        dim: int,
        tensor_id: Optional[str] = None,
    ) -> Tuple[FloatTensor, IntTensor]:
        """
        Calculate PoT quantization parameters along a specific dimension.
        
        :param observed: tensor to calculate quantization parameters for
        :param dim: dimension to calculate along
        :param tensor_id: optional id for the tensor being quantized
        :return: tuple of scale and zero point tensors
        """
        # determine which dimensions to reduce
        reduce_dims = tuple(i for i in range(observed.ndim) if i != dim)
        
        return self.calculate_qparams(observed, reduce_dims, tensor_id)
    
    def fake_quantize(
        self,
        observed: Tensor,
        scale: Optional[Tensor] = None,
        zero_point: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply PoT fake quantization to the observed tensor.
        
        This method quantizes the tensor using proper PoT logic where
        values are constrained to exact powers of two, then dequantizes
        back to floating point for gradient flow during training.
        
        :param observed: tensor to fake quantize
        :param scale: optional scale override
        :param zero_point: optional zero point override
        :return: fake quantized tensor
        """
        # Calculate qparams if not provided
        if scale is None or zero_point is None:
            calc_scale, calc_zero_point = self.calculate_qparams(observed)
            if scale is None:
                scale = calc_scale
            if zero_point is None:
                zero_point = calc_zero_point
        
        # Import PoT utility functions
        from llmcompressor.modifiers.quantization.pot.utils import fake_quantize_pot
        
        # Apply true PoT fake quantization
        # This will constrain values to actual powers of two
        fake_quantized = fake_quantize_pot(
            observed,
            scale,
            zero_point,
            num_bits=self.quantization_args.num_bits,
        )
        
        return fake_quantized