"""
Additive Power-of-Two (APoT) Quantization Modifier for LLM Compressor.

This modifier implements APoT quantization which represents values as sums
of signed powers of two, providing better accuracy than simple PoT while
maintaining hardware efficiency.
"""
from typing import Any, Dict, List, Optional, Union

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationType,
)
from pydantic import Field

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers.quantization.quantization.mixin import QuantizationMixin
from llmcompressor.modifiers import Modifier
from llmcompressor.observers.pot_apot import APoTObserver

__all__ = ["APoTQuantizationModifier"]


class APoTQuantizationModifier(Modifier, QuantizationMixin):
    """
    Additive Power-of-Two Quantization Modifier.
    
    Implements APoT quantization which represents values as sums of multiple
    signed powers of two. This provides denser quantization levels than simple
    PoT while still enabling efficient hardware implementation using shifts
    and additions.
    
    Example usage:
        ```python
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization.apot import APoTQuantizationModifier
        
        # Create APoT quantization modifier
        apot_modifier = APoTQuantizationModifier(
            targets=["Linear"],
            num_terms=2,  # use 2-term APoT
            scheme={
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "channel",
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "tensor",
                }
            }
        )
        
        # Apply to model
        oneshot(model=model, recipe=apot_modifier)
        ```
    
    :param apot_bits: number of bits for APoT quantization (default: 4)
    :param num_terms: number of power-of-two terms to use (default: 2)
    :param config_groups: dictionary specifying quantization schemes to apply
    :param targets: list of layer names to quantize
    :param ignore: list of module names to not quantize
    :param scheme: quantization scheme to apply
    """
    
    apot_bits: int = Field(default=4, description="Number of bits for APoT quantization")
    num_terms: int = Field(default=2, description="Number of power-of-two terms for APoT")
    
    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize the APoT quantization modifier.
        
        :param state: current state
        :param kwargs: additional arguments
        :return: True if should run immediately
        """
        # update the quantization schemes to use APoT observer
        self._update_schemes_for_apot()
        
        # initialize quantization using the mixin
        self.initialize_quantization(state.model)
        
        return True
    
    def _update_schemes_for_apot(self):
        """
        Update quantization schemes to use APoT-specific configurations.
        """
        # if using scheme parameter
        if self.scheme is not None:
            if isinstance(self.scheme, dict) and not any(
                key in self.scheme for key in ["weights", "input_activations", "output_activations"]
            ):
                # scheme is a preset name mapping
                for preset_name, targets in self.scheme.items():
                    # update each preset to use APoT
                    self._update_preset_for_apot(preset_name)
            else:
                # scheme is a direct configuration
                self._update_scheme_dict_for_apot(self.scheme)
        
        # if using config_groups
        if self.config_groups is not None:
            for group_name, group_scheme in self.config_groups.items():
                if hasattr(group_scheme, "weights") and group_scheme.weights is not None:
                    self._update_quant_args_for_apot(group_scheme.weights)
                if hasattr(group_scheme, "input_activations") and group_scheme.input_activations is not None:
                    self._update_quant_args_for_apot(group_scheme.input_activations)
                if hasattr(group_scheme, "output_activations") and group_scheme.output_activations is not None:
                    self._update_quant_args_for_apot(group_scheme.output_activations)
    
    def _update_scheme_dict_for_apot(self, scheme_dict: Dict[str, Any]):
        """
        Update a scheme dictionary to use APoT configurations.
        """
        if "weights" in scheme_dict and scheme_dict["weights"] is not None:
            scheme_dict["weights"]["observer"] = "apot"
            scheme_dict["weights"]["symmetric"] = True  # APoT is always symmetric
            scheme_dict["weights"]["num_terms"] = self.num_terms
            if "num_bits" not in scheme_dict["weights"]:
                scheme_dict["weights"]["num_bits"] = self.apot_bits
        
        if "input_activations" in scheme_dict and scheme_dict["input_activations"] is not None:
            scheme_dict["input_activations"]["observer"] = "apot"
            scheme_dict["input_activations"]["symmetric"] = True
            scheme_dict["input_activations"]["num_terms"] = self.num_terms
            if "num_bits" not in scheme_dict["input_activations"]:
                scheme_dict["input_activations"]["num_bits"] = 8
        
        if "output_activations" in scheme_dict and scheme_dict["output_activations"] is not None:
            scheme_dict["output_activations"]["observer"] = "apot"
            scheme_dict["output_activations"]["symmetric"] = True
            scheme_dict["output_activations"]["num_terms"] = self.num_terms
            if "num_bits" not in scheme_dict["output_activations"]:
                scheme_dict["output_activations"]["num_bits"] = 8
    
    def _update_quant_args_for_apot(self, quant_args: QuantizationArgs):
        """
        Update quantization args to use APoT observer.
        """
        # ensure we're using the APoT observer
        quant_args.observer = "apot"
        # APoT quantization is always symmetric
        quant_args.symmetric = True
        # set the quantization type
        quant_args.type = QuantizationType.INT
        # add num_terms as a custom attribute
        quant_args.num_terms = self.num_terms
    
    def _update_preset_for_apot(self, preset_name: str):
        """
        Update a preset scheme to use APoT configurations.
        
        This is a placeholder - in practice, we'd need to handle
        various preset schemes appropriately.
        """
        # preset schemes would need special handling
        # for now, we'll just log a warning
        import logging
        logging.warning(
            f"APoT quantization with preset scheme '{preset_name}' may not be fully supported. "
            "Consider using explicit scheme configuration."
        )
    
    def _initialize_observers(self, module: torch.nn.Module):
        """
        Override observer initialization to use APoT observers.
        """
        # check if this module has a quantization scheme attached
        if hasattr(module, "quantization_scheme"):
            scheme = module.quantization_scheme
            
            # initialize APoT observers for weights
            if scheme.weights is not None:
                # get num_terms from the quant args if available
                num_terms = getattr(scheme.weights, "num_terms", self.num_terms)
                module.weight_observer = APoTObserver(scheme.weights, num_terms=num_terms)
            
            # initialize APoT observers for activations
            if scheme.input_activations is not None:
                num_terms = getattr(scheme.input_activations, "num_terms", self.num_terms)
                module.input_observer = APoTObserver(scheme.input_activations, num_terms=num_terms)
            
            if scheme.output_activations is not None:
                num_terms = getattr(scheme.output_activations, "num_terms", self.num_terms)
                module.output_observer = APoTObserver(scheme.output_activations, num_terms=num_terms)
        
        # call the parent class method
        super()._initialize_observers(module)
    
    def on_start(self, state: State, event: Event, **kwargs):
        """
        Start the APoT quantization calibration process.
        
        :param state: current state
        :param event: start event
        :param kwargs: additional arguments
        """
        # start calibration using the mixin
        self.start_calibration(state.model)
    
    def on_end(self, state: State, event: Event, **kwargs):
        """
        End the APoT quantization calibration process.
        
        :param state: current state
        :param event: end event
        :param kwargs: additional arguments
        """
        # end calibration using the mixin
        self.end_calibration(state.model)
    
    def on_finalize(self, state: State, **kwargs):
        """
        Finalize the APoT quantization.
        
        :param state: current state
        :param kwargs: additional arguments
        """
        # finalize quantization
        self.finalize_quantization(state.model)