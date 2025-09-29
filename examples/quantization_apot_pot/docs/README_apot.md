# Additive Power-of-Two (APoT) Quantization Examples

This directory contains examples demonstrating how to use Additive Power-of-Two quantization with llm-compressor.

## What is APoT Quantization?

Additive Power-of-Two (APoT) quantization represents values as sums of multiple signed powers of two. For example, a value might be represented as `2^3 + 2^1` or `2^4 - 2^2`. This provides denser quantization levels than simple PoT while maintaining hardware efficiency through shift-and-add operations.

## Key Benefits

- **Better Accuracy**: Denser quantization levels compared to simple PoT
- **Hardware Efficient**: Uses only shift and add operations
- **Flexible**: Configurable number of terms for accuracy/efficiency trade-off
- **Non-uniform**: Better matches the distribution of neural network weights

## Examples

### Script With Calibration

```bash
python llama_apot_example.py
```

This example now downloads the Open-Platypus dataset, runs calibration through the
sequential pipeline, saves the quantized checkpoint, and verifies the output with vLLM.

### Weight-Only Quantization

```bash
python quantize_weights_only.py \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  TinyLlama-1.1B-Chat-v1.0-apot-w4-weight-only \
  --kind apot --weight-bits 4 --apot-terms 2

```

Because APoT uses activation observers, the script explicitly requests the
`datafree` pipeline when calling `oneshot`. This prevents the sequential pipeline
from attempting to iterate over missing calibration data.
```

The weight-only utility skips calibration entirely and still produces a compressed
checkpoint ready for vLLM inference.

## Configuration Options

### APoTQuantizationModifier Parameters

- `apot_bits`: Number of bits for APoT quantization (default: 4)
- `num_terms`: Number of power-of-two terms to use (default: 2)
- `targets`: List of layer types to quantize (e.g., ["Linear"])
- `ignore`: List of layers to skip (e.g., ["lm_head"])
- `scheme`: Detailed quantization configuration

### Scheme Configuration

```python
scheme = {
    "weights": {
        "num_bits": 4,
        "type": "int",
        "symmetric": True,  # APoT is always symmetric
        "strategy": "channel",  # per-channel or per-tensor
    },
    "input_activations": {
        "num_bits": 8,
        "type": "int",
        "symmetric": True,
        "strategy": "tensor",
    }
}
```

## APoT Representation

APoT represents values as:
```
value = ±2^n1 ± 2^n2 ± ... ± 2^nk
```

For 2-term APoT, examples include:
- `2^3 + 2^1 = 8 + 2 = 10`
- `2^4 - 2^2 = 16 - 4 = 12`
- `-(2^2 + 2^0) = -(4 + 1) = -5`

## Hardware Implementation

APoT quantization enables efficient hardware implementation using only shifts and adds:

```python
# For a 2-term APoT value: 2^3 + 2^1
result = (activation << 3) + (activation << 1)

# For a 2-term APoT value: 2^4 - 2^2  
result = (activation << 4) - (activation << 2)
```

## Number of Terms Trade-off

| Terms | Quantization Levels | Hardware Complexity | Accuracy |
|-------|-------------------|-------------------|----------|
| 1     | 2^n powers of two | Lowest (just shifts) | Lower |
| 2     | ~2^(n+1) levels   | Low (shift + add)    | Good |
| 3     | ~2^(n+2) levels   | Medium               | Better |
| 4+    | More levels       | Higher               | Best |

## Performance Considerations

- **Accuracy**: Better accuracy than PoT, approaching uniform quantization
- **Efficiency**: Slightly more complex than PoT but much simpler than multiplication
- **Memory**: Same memory savings as other low-bit quantization methods
- **Calibration**: Benefits from good calibration data for optimal level selection

## Supported Models

APoT quantization works with any model supported by llm-compressor, including:
- LLaMA/LLaMA-2
- Mistral
- CodeLlama
- And other transformer architectures

## Tips for Best Results

1. Start with 2-term APoT for a good accuracy/efficiency balance
2. Use per-channel quantization for weights when possible
3. Keep the output layer (lm_head) at full precision
4. Use sufficient calibration data (256+ samples)
5. Consider 3-term APoT if accuracy is critical and hardware can handle the complexity

## Research References

- "Additive Powers-of-Two Quantization: An Efficient Non-uniform Discretization for Neural Networks" (ICLR 2020)
- "Power-of-Two Quantization for Efficient Neural Network Inference"