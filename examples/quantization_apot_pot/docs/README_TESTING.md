# Testing APoT and PoT Quantized Models

This directory contains scripts for testing models quantized using APoT (Additive Power-of-Two) and PoT (Power-of-Two) quantization methods.

## Files Overview

### Quantization Examples
- `llama_apot_example.py` - Creates APoT quantized models
- `llama_pot_example.py` - Creates PoT quantized models

### Testing Scripts
- `test_apot_model.py` - Test APoT quantized models specifically
- `test_pot_model.py` - Test PoT quantized models specifically  
- `test_quantized_models.py` - Combined test script for both model types
- `test_models_fixed.py` - Fixed version of original test_models.py (no vLLM dependency)
- `benchmark_quantized_models.py` - Performance comparison between APoT and PoT models

### Legacy
- `test_models.py` - Original test script using vLLM (doesn't work with APoT/PoT)

## Quick Start

### 1. Create Quantized Models

First, generate the quantized models:

```bash
# Create APoT quantized model
python llama_apot_example.py

# Create PoT quantized model  
python llama_pot_example.py
```

This will create directories:
- `TinyLlama-1.1B-Chat-v1.0-apot-w4a8-t2/` (APoT model)
- `TinyLlama-1.1B-Chat-v1.0-pot-w4a8/` (PoT model)

### 2. Test Individual Models

```bash
# Test APoT model only
python test_apot_model.py

# Test PoT model only
python test_pot_model.py
```

### 3. Test Both Models

```bash
# Test both models with default paths
python test_quantized_models.py

# Test only APoT model
python test_quantized_models.py --model-type apot

# Test only PoT model
python test_quantized_models.py --model-type pot

# Test with custom paths
python test_quantized_models.py --apot-path /path/to/apot/model --pot-path /path/to/pot/model
```

### 4. Benchmark and Compare

```bash
# Run performance comparison
python benchmark_quantized_models.py
```

### 5. Test Any Available Models

```bash
# Automatically detect and test available models
python test_models_fixed.py
```

## Key Differences from Original test_models.py

The original `test_models.py` uses vLLM for model loading and inference:

```python
from vllm import LLM, SamplingParams
model = LLM("model_path")
```

However, **vLLM doesn't support APoT and PoT quantization formats**. The new test scripts use the proper loading approach:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.utils.dev import dispatch_for_generation

model = AutoModelForCausalLM.from_pretrained(model_path, ...)
dispatch_for_generation(model)  # Important for quantized models!
```

## Test Prompts

All test scripts use the same set of diverse prompts to evaluate model performance:

1. "Hello, my name is"
2. "The capital of France is"
3. "Write a short story about a robot learning to paint:"
4. "Explain quantum computing in simple terms:"
5. "What are the benefits of renewable energy?"
6. "Describe the process of photosynthesis:"
7. "Tell me a joke about programming:"
8. "What is the meaning of life?"
9. "How do neural networks work?"
10. "Write a recipe for chocolate cake:"

## Important Notes

### Memory Management
- The scripts include proper GPU memory cleanup between model tests
- Use `torch.cuda.empty_cache()` to free GPU memory
- Models are deleted after testing to prevent memory leaks

### Generation Parameters
- `max_new_tokens=128` (or 256 for compatibility tests)
- `temperature=0.7`
- `top_p=0.9`
- `do_sample=True` (except in benchmarks where deterministic generation is used)

### Error Handling
- All scripts include try-catch blocks for robust error handling
- Check for model existence before attempting to load
- Graceful handling of generation errors

## Expected Output

Each test will show:
1. Model loading confirmation with device and dtype
2. For each prompt:
   - The input prompt
   - Generated text (with input prompt removed)
   - Separator line
3. Completion message

Benchmark script additionally shows:
- Loading time
- Average inference time
- Tokens per second
- Model size
- Success rate
- Performance comparison between models

## Troubleshooting

### Model Not Found
```
Model not found at: TinyLlama-1.1B-Chat-v1.0-apot-w4a8-t2
Please run llama_apot_example.py first to create the quantized model.
```
**Solution**: Run the corresponding quantization example script first.

### CUDA Out of Memory
**Solution**: 
- Reduce batch size or sequence length
- Test models individually instead of together
- Use smaller models or reduce `max_new_tokens`

### Import Errors
**Solution**: Ensure all dependencies are installed:
```bash
pip install transformers torch datasets compressed-tensors
```

## Advanced Usage

### Custom Model Paths
You can test models from custom locations by modifying the model paths in the scripts or using command-line arguments where available.

### Different Generation Parameters
Modify the generation parameters in the scripts to test different behaviors:
- Change `temperature` for more/less randomness
- Adjust `max_new_tokens` for longer/shorter outputs
- Use `do_sample=False` for deterministic generation

### Adding New Test Prompts
Add prompts to the `TEST_PROMPTS` list in any script to test additional scenarios.
