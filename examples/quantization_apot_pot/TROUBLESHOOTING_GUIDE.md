# Troubleshooting Guide for APoT/PoT Quantized Models

This guide helps you resolve common issues when testing quantized models, especially CUDA-related errors.

## 🚨 Common Issues and Solutions

### 1. CUDA Assertion Errors

**Error Message:**
```
CUDA error: device-side assert triggered
_assert_async_cuda_kernel: Assertion `probability tensor contains either `inf`, `nan` or element < 0` failed.
```

**Root Cause:** Quantized models can have numerical instabilities that create invalid probability values during sampling.

**Solutions (in order of preference):**

#### A. Use Stable Test Script (Recommended)
```bash
python test_quantized_models_stable.py
```
This script uses ultra-conservative generation settings designed for quantized models.

#### B. Use Greedy Decoding
```bash
python test_quantized_models.py --use-sampling false
```
Greedy decoding (do_sample=False) is more stable than sampling for quantized models.

#### C. Enable CUDA Debugging
```bash
python test_quantized_models.py --cuda-debug
```
This sets `CUDA_LAUNCH_BLOCKING=1` for better error reporting.

### 2. Attention Mask Warnings

**Warning Message:**
```
The attention mask is not set and cannot be inferred from input because pad token is same as eos token.
```

**Solution:** This is handled automatically in the updated test scripts. The scripts now properly create and use attention masks.

### 3. Model Loading Issues

**Error:** Model fails to load or crashes during loading.

**Solutions:**

#### Diagnose the Issue
```bash
python debug_quantized_models.py
```
This script will systematically check your quantized models and identify specific problems.

#### Use CPU Inference
If GPU inference fails, try CPU inference by modifying the model loading:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="cpu",  # Force CPU usage
    trust_remote_code=True,
)
```

### 4. Memory Issues

**Error:** CUDA out of memory errors.

**Solutions:**
- Reduce `max_new_tokens` in generation
- Clear CUDA cache between generations
- Test models individually instead of together
- Use smaller batch sizes

## 🔧 Using the Improved Test Scripts

### Ultra-Stable Testing
For maximum stability with quantized models:
```bash
python test_quantized_models_stable.py
```

Features:
- ✅ Ultra-conservative generation parameters
- ✅ Progressive fallback strategies 
- ✅ Comprehensive error handling
- ✅ Automatic CUDA debugging enabled
- ✅ Memory management between tests

### Flexible Testing
For customizable testing:
```bash
# Greedy decoding (stable)
python test_quantized_models.py

# Sampling (more creative but less stable)
python test_quantized_models.py --use-sampling

# Test only one model type
python test_quantized_models.py --model-type apot
python test_quantized_models.py --model-type pot

# Debug mode
python test_quantized_models.py --cuda-debug
```

### Diagnostic Testing
To diagnose model issues:
```bash
python debug_quantized_models.py
```

This will:
- ✅ Check model files and structure
- ✅ Test model loading on CPU and GPU
- ✅ Test minimal generation
- ✅ Provide specific error analysis
- ✅ Give recommendations for fixes

## 🎯 Generation Parameter Guidelines

### For Maximum Stability (Recommended)
```python
generation_kwargs = {
    "max_new_tokens": 50,        # Short generations
    "do_sample": False,          # Greedy decoding
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}
```

### For Balanced Performance
```python
generation_kwargs = {
    "max_new_tokens": 100,
    "do_sample": True,
    "temperature": 0.8,          # Conservative temperature
    "top_p": 0.95,              # Conservative top_p
    "top_k": 50,                # Add top_k filtering
    "repetition_penalty": 1.1,   # Prevent repetition
}
```

### For Maximum Creativity (Least Stable)
```python
generation_kwargs = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
}
```

## 🛠️ Environment Setup for Debugging

Set these environment variables for better error reporting:

```bash
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
```

Or in Python:
```python
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
```

## 📊 Interpreting Test Results

### Successful Output
```
Test 1/5 - APOT
Prompt: Hello, my name is
Generated: John and I am a software engineer.
----------------------------------------
```

### Failed Generation
```
Test 1/5 - APOT  
Prompt: Hello, my name is
Error during generation: CUDA error: device-side assert triggered
----------------------------------------
```

### Fallback Success
```
Test 1/5 - APOT
Prompt: Hello, my name is
CUDA error encountered, trying with reduced parameters...
Generated: John.
----------------------------------------
```

## 🔍 When to Use Each Script

| Script | Use Case | Stability | Features |
|--------|----------|-----------|----------|
| `test_quantized_models_stable.py` | First try, maximum stability | ⭐⭐⭐⭐⭐ | Progressive fallback, automatic error handling |
| `test_quantized_models.py` | Flexible testing with options | ⭐⭐⭐⭐ | CLI options, sampling/greedy modes |
| `test_models_fixed.py` | General testing, auto-detect models | ⭐⭐⭐ | Automatic model detection |
| `debug_quantized_models.py` | Diagnosing problems | ⭐⭐⭐⭐⭐ | Systematic diagnosis, detailed error analysis |

## 🚑 Emergency Fixes

### If All Tests Fail
1. **Check model creation**: Ensure quantization completed successfully
   ```bash
   ls -la TinyLlama-1.1B-Chat-v1.0-*pot-*/
   ```

2. **Try CPU inference**: Add `device_map="cpu"` to model loading

3. **Re-run quantization**: Sometimes the quantization process has issues
   ```bash
   python llama_apot_example.py
   python llama_pot_example.py
   ```

4. **Check GPU memory**: Ensure sufficient VRAM available
   ```bash
   nvidia-smi
   ```

### If Only Specific Prompts Fail
- Use shorter, simpler prompts
- Reduce `max_new_tokens`
- Switch to greedy decoding
- Clear CUDA cache between prompts

## 📞 Getting Help

If you're still experiencing issues:

1. **Run the diagnostic script**: `python debug_quantized_models.py`
2. **Check the model files**: Ensure all required files exist
3. **Try CPU inference**: Rule out GPU-specific issues
4. **Use minimal generation**: Test with very short outputs first

The diagnostic script will provide specific recommendations based on the errors encountered.
