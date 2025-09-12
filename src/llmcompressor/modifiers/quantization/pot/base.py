"""
Power-of-Two (PoT) Quantization Modifier for LLM Compressor.

This modifier implements PoT quantization which constrains weights and activations
to powers of two, enabling efficient inference using bit-shift operations.
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
from llmcompressor.observers.pot_apot import PoTObserver

__all__ = ["PoTQuantizationModifier"]


class PoTQuantizationModifier(Modifier, QuantizationMixin):
    """
    Power-of-Two Quantization Modifier.
    
    Implements PoT quantization which constrains all quantized values to powers of two.
    This enables efficient inference on hardware using bit-shift operations instead
    of multiplications.
    
    Example usage:
        ```python
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization.pot import PoTQuantizationModifier
        
        # Create PoT quantization modifier
        pot_modifier = PoTQuantizationModifier(
            targets=["Linear"],
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
        oneshot(model=model, recipe=pot_modifier)
        ```
    
    :param pot_bits: number of bits for PoT quantization (default: 4)
    :param config_groups: dictionary specifying quantization schemes to apply
    :param targets: list of layer names to quantize
    :param ignore: list of module names to not quantize
    :param scheme: quantization scheme to apply
    """
    
    pot_bits: int = Field(default=4, description="Number of bits for PoT quantization")
    
    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize the PoT quantization modifier.
        
        :param state: current state
        :param kwargs: additional arguments
        :return: True if should run immediately
        """
        # update the quantization schemes to use PoT observer
        self._update_schemes_for_pot()
        
        # initialize quantization using the mixin
        self.initialize_quantization(state.model)
        
        return True
    
    def _update_schemes_for_pot(self):
        """
        Update quantization schemes to use PoT-specific configurations.
        """
        # if using scheme parameter
        if self.scheme is not None:
            if isinstance(self.scheme, dict) and not any(
                key in self.scheme for key in ["weights", "input_activations", "output_activations"]
            ):
                # scheme is a preset name mapping
                for preset_name, targets in self.scheme.items():
                    # update each preset to use PoT
                    self._update_preset_for_pot(preset_name)
            else:
                # scheme is a direct configuration
                self._update_scheme_dict_for_pot(self.scheme)
        
        # if using config_groups
        if self.config_groups is not None:
            for group_name, group_scheme in self.config_groups.items():
                if hasattr(group_scheme, "weights") and group_scheme.weights is not None:
                    self._update_quant_args_for_pot(group_scheme.weights)
                if hasattr(group_scheme, "input_activations") and group_scheme.input_activations is not None:
                    self._update_quant_args_for_pot(group_scheme.input_activations)
                if hasattr(group_scheme, "output_activations") and group_scheme.output_activations is not None:
                    self._update_quant_args_for_pot(group_scheme.output_activations)
    
    def _update_scheme_dict_for_pot(self, scheme_dict: Dict[str, Any]):
        """
        Update a scheme dictionary to use PoT configurations.
        """
        if "weights" in scheme_dict and scheme_dict["weights"] is not None:
            scheme_dict["weights"]["observer"] = "pot"
            scheme_dict["weights"]["symmetric"] = True  # PoT is always symmetric
            if "num_bits" not in scheme_dict["weights"]:
                scheme_dict["weights"]["num_bits"] = self.pot_bits
        
        if "input_activations" in scheme_dict and scheme_dict["input_activations"] is not None:
            scheme_dict["input_activations"]["observer"] = "pot"
            scheme_dict["input_activations"]["symmetric"] = True
            if "num_bits" not in scheme_dict["input_activations"]:
                scheme_dict["input_activations"]["num_bits"] = 8
        
        if "output_activations" in scheme_dict and scheme_dict["output_activations"] is not None:
            scheme_dict["output_activations"]["observer"] = "pot"
            scheme_dict["output_activations"]["symmetric"] = True
            if "num_bits" not in scheme_dict["output_activations"]:
                scheme_dict["output_activations"]["num_bits"] = 8
    
    def _update_quant_args_for_pot(self, quant_args: QuantizationArgs):
        """
        Update quantization args to use PoT observer.
        """
        # ensure we're using the PoT observer
        quant_args.observer = "pot"
        # PoT quantization is always symmetric
        quant_args.symmetric = True
        # set the quantization type
        quant_args.type = QuantizationType.INT
    
    def _update_preset_for_pot(self, preset_name: str):
        """
        Update a preset scheme to use PoT configurations.
        
        This is a placeholder - in practice, we'd need to handle
        various preset schemes appropriately.
        """
        # preset schemes would need special handling
        # for now, we'll just log a warning
        import logging
        logging.warning(
            f"PoT quantization with preset scheme '{preset_name}' may not be fully supported. "
            "Consider using explicit scheme configuration."
        )
    
    # removed _initialize_observers override - let the base system handle observer creation
    # through the registry using the observer name in quantization_args
    
    def on_start(self, state: State, event: Event, **kwargs):
        """
        Start the PoT quantization calibration process.
        
        :param state: current state
        :param event: start event
        :param kwargs: additional arguments
        """
        # start calibration using the mixin
        self.start_calibration(state.model)
    
    def on_end(self, state: State, event: Event, **kwargs):
        """
        End the PoT quantization calibration process.
        
        :param state: current state
        :param event: end event
        :param kwargs: additional arguments
        """
        # end calibration using the mixin
        self.end_calibration(state.model)
    
    def on_finalize(self, state: State, **kwargs):
        """
        Finalize the PoT quantization.
        
        :param state: current state
        :param kwargs: additional arguments
        """
        # finalize quantization
        self.finalize_quantization(state.model)