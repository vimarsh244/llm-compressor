# PoT and APoT Quantization Issues

## Problem Description

The Power-of-Two (PoT) and Additive Power-of-Two (APoT) quantization implementations in llm-compressor are currently incomplete and produce models that output zeros during inference.

## Root Cause Analysis

### 1. Missing Import (Fixed)
- The PoT observer had a missing import statement which has been fixed
- File: `llm-compressor/src/llmcompressor/observers/pot_apot/pot_observer.py`

### 2. Incomplete Integration with compressed_tensors

The main issue is that PoT and APoT quantization are not fully integrated with the compressed_tensors library:

- **Observers Only**: The current implementation only provides custom observers that compute power-of-two scales
- **No Custom Quantization**: The actual quantization still uses standard formulas: `quantized = round((tensor - zero_point) / scale)`
- **No Custom Dequantization**: During inference, compressed_tensors uses standard dequantization: `dequantized = quantized * scale + zero_point`
- **Missing Metadata**: The quantization type (PoT/APoT) and parameters (like num_terms) are not stored in the model

### 3. Why It Outputs Zeros

When using `save_compressed=True`:
1. The PoT/APoT observers compute power-of-two scales
2. Standard quantization is applied using these scales
3. During inference, standard dequantization is used
4. The mismatch between PoT/APoT scales and standard quantization/dequantization produces incorrect values (often zeros)

## Temporary Workaround

Use the `quantize_weights_only_fixed.py` script which:
1. Performs the quantization as before
2. Saves the model without compression (`save_compressed=False`)
3. This preserves the quantized weights in FP16/FP32 format but doesn't achieve compression

## Proper Solution Requirements

To properly implement PoT/APoT quantization, the following changes are needed:

### 1. Custom Quantization Functions
- Implement proper PoT/APoT quantization that uses the power-of-two constraints
- Register these functions with compressed_tensors

### 2. Custom Dequantization Functions  
- Implement corresponding dequantization functions
- Ensure they're used during model loading and inference

### 3. Metadata Storage
- Store quantization type ("pot" or "apot") in the quantization configuration
- Store APoT-specific parameters (num_terms) when applicable

### 4. compressed_tensors Integration
- Extend compressed_tensors to recognize and handle PoT/APoT quantization types
- Ensure the correct quantization/dequantization functions are called based on the type

## Example Usage (Current Workaround)

```bash
# Quantize with PoT (will save uncompressed)
python quantize_weights_only_fixed.py meta-llama/Llama-2-7b-hf ./pot_model --kind pot --weight-bits 4

# Quantize with APoT (will save uncompressed)  
python quantize_weights_only_fixed.py meta-llama/Llama-2-7b-hf ./apot_model --kind apot --weight-bits 4 --apot-terms 2
```

## Testing the Fixed Import

To verify the import fix works:
```python
from llmcompressor.observers.pot_apot import PoTObserver, APoTObserver
# Should import successfully now
```

## Next Steps

1. Implement proper PoT/APoT quantization functions in `pot/utils.py` and `apot/utils.py`
2. Register these functions with compressed_tensors
3. Update the modifiers to use these custom functions
4. Add tests to ensure correct quantization/dequantization behavior
5. Update documentation to reflect the implementation status
