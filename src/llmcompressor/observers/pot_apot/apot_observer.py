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
        # get num_terms from quantization_args if available, otherwise use parameter
        self.num_terms = getattr(quantization_args, 'num_terms', num_terms)
    
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
        - Scale determines the base power-of-two range
        - Zero point is always 0 for symmetric quantization
        - The actual quantization levels are determined by the APoT configuration
        
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
        
        # for APoT, we need to determine the range of powers to use
        # we'll use the max value to determine the highest power needed
        log2_max = torch.log2(max_val)
        max_exponent = torch.ceil(log2_max)
        
        # the scale represents the base unit for APoT quantization
        # we scale it based on the number of bits and terms available
        num_bits = self.quantization_args.num_bits
        
        # for APoT with k terms, we can represent more values
        # adjust the scale to account for the additive nature
        effective_bits = num_bits - 1  # reserve one bit for sign
        
        # calculate the range of exponents we'll use
        # this depends on the number of terms and bits available
        exponent_range = effective_bits // self.num_terms
        
        # the scale is based on the maximum exponent we need to represent
        # divided by the number of quantization levels
        num_levels = self._calculate_num_apot_levels(num_bits, self.num_terms)
        scale = torch.pow(2.0, max_exponent) / (num_levels // 2)
        
        # for symmetric quantization, zero point is always 0
        zero_point = torch.zeros_like(scale, dtype=torch.int32)
        
        # apply global scale if provided
        if global_scale is not None:
            scale = scale * global_scale
        
        # ensure scale is a power of two for efficiency
        scale_log2 = torch.log2(scale)
        scale_exp = torch.round(scale_log2)
        scale = torch.pow(2.0, scale_exp)
        
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