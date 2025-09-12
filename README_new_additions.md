# New Additions: Power-of-Two and Additive Power-of-Two Quantization

This document describes the new Power-of-Two (PoT) and Additive Power-of-Two (APoT) quantization methods added to llm-compressor.

## Overview

We have extended llm-compressor to support two new quantization methods that enable efficient hardware inference:

1. **Power-of-Two (PoT) Quantization**: Constrains all quantized values to powers of two (±2^n)
2. **Additive Power-of-Two (APoT) Quantization**: Represents values as sums of signed powers of two

Both methods enable efficient inference using bit-shift operations instead of multiplications, making them ideal for hardware deployment.

## Power-of-Two (PoT) Quantization

### What is PoT Quantization?

PoT quantization constrains all quantized weights and activations to be powers of two. For example, instead of arbitrary values like 3.7 or 5.2, weights are quantized to values like 2, 4, 8, 0.5, 0.25, etc.

### Benefits

- **Hardware Efficiency**: Multiplications become bit-shift operations
- **Memory Efficient**: Standard low-bit quantization memory savings
- **Simple Implementation**: Straightforward to implement in hardware
- **Energy Efficient**: Bit-shifts consume much less power than multiplications

### Usage

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization.pot import PoTQuantizationModifier

# Create PoT quantization modifier
pot_modifier = PoTQuantizationModifier(
    targets=["Linear"],
    ignore=["lm_head"],
    pot_bits=4,
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
oneshot(model=model, dataset=calibration_data, recipe=pot_modifier)
```

### Hardware Implementation

```python
# Traditional multiplication:
result = weight * activation

# PoT with bit-shift:
shift_amount = log2(weight)  # precomputed
result = activation << shift_amount  # left shift for multiplication
```

## Additive Power-of-Two (APoT) Quantization

### What is APoT Quantization?

APoT quantization represents values as sums of multiple signed powers of two. For example:
- `5 = 2^2 + 2^0 = 4 + 1`
- `6 = 2^3 - 2^1 = 8 - 2`
- `-3 = -(2^1 + 2^0) = -(2 + 1)`

This provides denser quantization levels than simple PoT while maintaining hardware efficiency.

### Benefits

- **Better Accuracy**: More quantization levels than simple PoT
- **Hardware Efficient**: Uses only shift and add operations
- **Flexible**: Configurable number of terms for accuracy/efficiency trade-off
- **Non-uniform**: Better matches neural network weight distributions

### Usage

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization.apot import APoTQuantizationModifier

# Create APoT quantization modifier
apot_modifier = APoTQuantizationModifier(
    targets=["Linear"],
    ignore=["lm_head"],
    apot_bits=4,
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
oneshot(model=model, dataset=calibration_data, recipe=apot_modifier)
```

### Hardware Implementation

```python
# For weight = 2^3 + 2^1 = 8 + 2 = 10:
result = (activation << 3) + (activation << 1)

# For weight = 2^4 - 2^2 = 16 - 4 = 12:
result = (activation << 4) - (activation << 2)
```

### Number of Terms Trade-off

| Terms | Quantization Levels | Hardware Complexity | Accuracy |
|-------|-------------------|-------------------|----------|
| 1     | 2^n powers of two | Lowest (just shifts) | Lower |
| 2     | ~2^(n+1) levels   | Low (shift + add)    | Good |
| 3     | ~2^(n+2) levels   | Medium               | Better |
| 4+    | More levels       | Higher               | Best |

## Implementation Details

### File Structure

```
llm-compressor/src/llmcompressor/
├── observers/pot_apot/
│   ├── __init__.py
│   ├── pot_observer.py          # PoT quantization parameter calculation
│   └── apot_observer.py         # APoT quantization parameter calculation
├── modifiers/quantization/
│   ├── pot/
│   │   ├── __init__.py
│   │   ├── base.py              # PoT quantization modifier
│   │   └── utils.py             # PoT utility functions
│   └── apot/
│       ├── __init__.py
│       ├── base.py              # APoT quantization modifier
│       └── utils.py             # APoT utility functions
```

### Key Classes

1. **PoTObserver**: Calculates quantization parameters for PoT quantization
2. **APoTObserver**: Calculates quantization parameters for APoT quantization  
3. **PoTQuantizationModifier**: Main modifier for applying PoT quantization
4. **APoTQuantizationModifier**: Main modifier for applying APoT quantization

### Integration with Existing Infrastructure

Both PoT and APoT quantization integrate seamlessly with llm-compressor's existing infrastructure:

- Use the same `QuantizationMixin` for lifecycle management
- Support all existing quantization strategies (tensor, channel, group)
- Compatible with existing calibration and hooks system
- Work with HuggingFace Transformers integration

## Examples

### Complete Examples

See the example directories for complete working examples:

- `examples/quantization_pot/`: PoT quantization examples
- `examples/quantization_apot/`: APoT quantization examples

### Quick Start - PoT

```python
from llmcompressor.modifiers.quantization.pot import PoTQuantizationModifier

recipe = PoTQuantizationModifier(targets=["Linear"], pot_bits=4)
oneshot(model=model, recipe=recipe)
```

### Quick Start - APoT

```python
from llmcompressor.modifiers.quantization.apot import APoTQuantizationModifier

recipe = APoTQuantizationModifier(targets=["Linear"], apot_bits=4, num_terms=2)
oneshot(model=model, recipe=recipe)
```

## Testing

Unit tests are provided for all components:

```bash
# Run PoT tests
python -m pytest tests/llmcompressor/observers/pot_apot/test_pot_observer.py
python -m pytest tests/llmcompressor/modifiers/quantization/pot/

# Run APoT tests  
python -m pytest tests/llmcompressor/observers/pot_apot/test_apot_observer.py
python -m pytest tests/llmcompressor/modifiers/quantization/apot/
```

## Performance Characteristics

### PoT Quantization

- **Accuracy**: Slightly lower than uniform quantization due to power-of-two constraint
- **Speed**: Significant speedup on hardware supporting efficient bit-shifts
- **Memory**: Same memory savings as other low-bit quantization
- **Energy**: Much lower energy consumption than multiplication-based inference

### APoT Quantization

- **Accuracy**: Better than PoT, approaching uniform quantization
- **Speed**: Fast on hardware with efficient shift-add units
- **Memory**: Same memory savings as other low-bit quantization
- **Energy**: Low energy consumption, slightly higher than PoT

## Research Background

These implementations are based on research papers:

1. **PoT Quantization**: 
   - "Power-of-Two Quantization for Efficient Neural Network Inference"
   - "A Hardware-Friendly Low-Bit Power-of-Two Quantization Method for CNNs"

2. **APoT Quantization**:
   - "Additive Powers-of-Two Quantization: An Efficient Non-uniform Discretization for Neural Networks" (ICLR 2020)

## Limitations and Future Work

### Current Limitations

1. **Observer Registration**: Currently requires manual observer registration
2. **Hardware Kernels**: No specialized CUDA kernels for actual bit-shift operations
3. **Preset Schemes**: Limited support for preset quantization schemes
4. **Activation Quantization**: Focus primarily on weight quantization

### Future Enhancements

1. Add specialized CUDA kernels for true bit-shift operations
2. Implement more sophisticated APoT level selection algorithms
3. Add support for mixed PoT/APoT quantization within the same model
4. Optimize for specific hardware architectures (FPGA, embedded processors)

## Contributing

To contribute improvements to PoT/APoT quantization:

1. Follow the existing code structure and patterns
2. Add comprehensive unit tests for new functionality
3. Update documentation and examples
4. Consider hardware efficiency in implementation choices

## Support

For questions or issues with PoT/APoT quantization:

1. Check the examples in `examples/quantization_pot/` and `examples/quantization_apot/`
2. Review the unit tests for usage patterns
3. File issues on the llm-compressor GitHub repository