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
        - Scale is computed as the nearest power-of-two to the max absolute value
        - Zero point is always 0 for symmetric quantization
        
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
        
        # calculate the nearest power of two for the scale
        # we use log2 and round to get the exponent
        log2_max = torch.log2(max_val)
        
        # for PoT, we want to find the power of 2 that best covers the range
        # we use ceil to ensure we don't lose information
        exponent = torch.ceil(log2_max)
        
        # the scale is 2^exponent / (2^(num_bits-1) - 1)
        # this ensures the quantized range [-2^(n-1)+1, 2^(n-1)-1] maps properly
        num_bits = self.quantization_args.num_bits
        scale = torch.pow(2.0, exponent) / (2 ** (num_bits - 1) - 1)
        
        # for symmetric quantization, zero point is always 0
        zero_point = torch.zeros_like(scale, dtype=torch.int32)
        
        # apply global scale if provided
        if global_scale is not None:
            scale = scale * global_scale
        
        # ensure scale is a power of two
        # round to nearest power of two if needed due to numerical precision
        scale_log2 = torch.log2(scale)
        scale_exp = torch.round(scale_log2)
        scale = torch.pow(2.0, scale_exp)
        
        # squeeze the scale and zero point tensors if reducing all dimensions
        if len(reduce_dims) == observed.ndim:
            scale = scale.squeeze()
            zero_point = zero_point.squeeze()
        
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