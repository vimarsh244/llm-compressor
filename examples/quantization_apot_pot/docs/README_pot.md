# Power-of-Two (PoT) Quantization Examples

This directory contains examples demonstrating how to use Power-of-Two quantization with llm-compressor.

## What is PoT Quantization?

Power-of-Two (PoT) quantization constrains all quantized values to powers of two (e.g., ±2^n). This enables efficient inference on hardware using bit-shift operations instead of multiplications, significantly reducing computational complexity and power consumption.

## Key Benefits

- **Hardware Efficiency**: Multiplication operations are replaced with bit-shift operations
- **Memory Efficient**: Reduced precision reduces memory footprint
- **Fast Inference**: Bit-shift operations are much faster than multiplications
- **Energy Efficient**: Lower power consumption compared to full-precision models

## Examples

### Script With Calibration

```bash
python llama_pot_example.py
```

The example now fetches calibration samples from the Pile validation split, runs PoT
quantization with APoT-compatible hooks, and saves a checkpoint that contains the
compressed-tensors metadata required by vLLM.

### Weight-Only Quantization

```bash
python quantize_weights_only.py \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  TinyLlama-1.1B-Chat-v1.0-pot-w4-weight-only \
  --kind pot --weight-bits 4
```

Use the weight-only helper when you want to skip calibration yet still produce a
compressed checkpoint.

## Configuration Options

### PoTQuantizationModifier Parameters

- `pot_bits`: Number of bits for PoT quantization (default: 4)
- `targets`: List of layer types to quantize (e.g., ["Linear"])
- `ignore`: List of layers to skip (e.g., ["lm_head"])
- `scheme`: Detailed quantization configuration

### Scheme Configuration

```python
scheme = {
    "weights": {
        "num_bits": 4,
        "type": "int",
        "symmetric": True,  # PoT is always symmetric
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

## Hardware Implementation

PoT quantization enables efficient hardware implementation:

```python
# Instead of: result = weight * activation
# Use bit shifts: result = activation << shift_amount
```

Where `shift_amount = log2(weight)` for power-of-two weights.

## Performance Considerations

- **Accuracy**: PoT quantization may have slightly lower accuracy than uniform quantization
- **Efficiency**: Significant speedup and energy savings on appropriate hardware
- **Memory**: Reduced memory bandwidth requirements
- **Calibration**: Requires calibration data for optimal quantization parameters

## Supported Models

PoT quantization works with any model supported by llm-compressor, including:
- LLaMA/LLaMA-2
- Mistral
- CodeLlama
- And other transformer architectures

## Tips for Best Results

1. Use per-channel quantization for weights when possible
2. Keep the output layer (lm_head) at full precision
3. Use sufficient calibration data (256+ samples)
4. Consider the trade-off between efficiency and accuracy for your use case