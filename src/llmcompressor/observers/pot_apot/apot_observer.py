"""
Additive Power-of-Two (APoT) Observer for quantization.

This observer calculates quantization parameters for APoT quantization where
values are represented as sums of signed powers of two.
"""
import torch
from compressed_tensors.quantization.quant_args import QuantizationArgs
from torch import FloatTensor, IntTensor, Tensor
from typing import Optional, Tuple, List

from llmcompressor.observers.base import Observer

__all__ = ["APoTObserver"]


@Observer.register("apot")
class APoTObserver(Observer):
    """
    Observer for Additive Power-of-Two quantization.
    
    APoT quantization represents values as sums of multiple signed powers of two,
    providing denser quantization levels while maintaining hardware efficiency.
    """
    
    def __init__(
        self,
        quantization_args: QuantizationArgs,
        num_terms: int = 2,
        **kwargs,
    ):
        """
        Initialize APoT observer.
        
        :param quantization_args: quantization arguments
        :param num_terms: number of power-of-two terms to use (default: 2)
        :param kwargs: additional arguments (for registry compatibility)
        """
        super().__init__(quantization_args)
        # get num_terms from kwargs (observer_kwargs) if available, otherwise use parameter
        self.num_terms = kwargs.get('num_terms', num_terms)
    
    def calculate_qparams(
        self,
        observed: Tensor,
        reduce_dims: Optional[Tuple[int]] = None,
        tensor_id: Optional[str] = None,
        global_scale: Optional[Tensor] = None,
    ) -> Tuple[FloatTensor, IntTensor]:
        """
        Calculate APoT quantization parameters for the observed tensor.
        
        For APoT quantization:
        - Scale determines the range for APoT quantization levels
        - Zero point is always 0 for symmetric quantization
        - Values will be constrained to sums of powers of two during quantization
        
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
        
        # Import APoT utility functions
        from llmcompressor.modifiers.quantization.apot.utils import generate_apot_levels
        
        # Generate the actual APoT quantization levels
        apot_levels = generate_apot_levels(num_bits, self.num_terms)
        max_apot_level = max(abs(level) for level in apot_levels)
        
        # Calculate scale so that max_val maps to the maximum APoT level
        # This ensures the full APoT range is utilized properly
        scale = max_val / max_apot_level
        
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
    
    def _calculate_num_apot_levels(self, num_bits: int, num_terms: int) -> int:
        """
        Calculate the number of unique quantization levels for APoT.
        
        :param num_bits: number of bits for quantization
        :param num_terms: number of power-of-two terms
        :return: number of unique quantization levels
        """
        # for k-term APoT with n bits:
        # we can represent combinations of k powers from a set of size n-1
        # plus the zero value
        # this is a combinatorial calculation
        
        # simplified approximation for 2-term APoT
        if num_terms == 2:
            # we can have: 0, ±2^i, ±(2^i + 2^j), ±(2^i - 2^j)
            # where i > j
            effective_bits = num_bits - 1
            # approximate number of combinations
            num_levels = 2 + 2 * effective_bits + effective_bits * (effective_bits - 1)
        else:
            # for general k-term APoT, use a more complex calculation
            # this is a placeholder - exact calculation depends on the specific
            # APoT configuration
            num_levels = 2 ** (num_bits - 1)
        
        return num_levels
    
    def get_qparams_along_dim(
        self,
        observed: Tensor,
        dim: int,
        tensor_id: Optional[str] = None,
    ) -> Tuple[FloatTensor, IntTensor]:
        """
        Calculate APoT quantization parameters along a specific dimension.
        
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
        Apply APoT fake quantization to the observed tensor.
        
        This method quantizes the tensor using proper APoT logic where
        values are constrained to sums of powers of two, then dequantizes
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
        
        # Import APoT utility functions
        from llmcompressor.modifiers.quantization.apot.utils import fake_quantize_apot
        
        # Apply true APoT fake quantization
        # This will constrain values to sums of powers of two
        fake_quantized = fake_quantize_apot(
            observed,
            scale,
            zero_point,
            num_bits=self.quantization_args.num_bits,
            num_terms=self.num_terms,
        )
        
        return fake_quantized