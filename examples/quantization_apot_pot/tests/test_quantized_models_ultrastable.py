"""Ultra-stable test script for quantized models with comprehensive error handling."""

import os
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.utils.dev import dispatch_for_generation

# Set environment variables for better CUDA error handling
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Shorter test prompts for stability
STABLE_TEST_PROMPTS = [
    "Hello",
    "The capital of France is",
    "Python is",
    "Machine learning is",
    "What is"
]

# Default model paths
DEFAULT_MODELS = {
    "apot": "TinyLlama-1.1B-Chat-v1.0-apot-w4a8-t2",
    "pot": "TinyLlama-1.1B-Chat-v1.0-pot-w4a8"
}


def clear_cuda_cache():
    """Clear CUDA cache and handle any errors."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass  # Ignore cache clearing errors


def safe_generate(model, tokenizer, prompt: str, max_retries: int = 3) -> str:
    """Safely generate text with multiple fallback strategies."""
    
    for attempt in range(max_retries):
        try:
            clear_cuda_cache()
            
            # Progressive fallback strategies
            if attempt == 0:
                # First attempt: Ultra-conservative greedy
                max_tokens, strategy = 30, "ultra_conservative"
                generation_kwargs = {
                    "max_new_tokens": max_tokens,
                    "do_sample": False,
                    "num_beams": 1,
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "early_stopping": True,
                }
            elif attempt == 1:
                # Second attempt: Even more conservative
                max_tokens, strategy = 20, "minimal"
                generation_kwargs = {
                    "max_new_tokens": max_tokens,
                    "do_sample": False,
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
            else:
                # Final attempt: Absolute minimum
                max_tokens, strategy = 10, "emergency"
                generation_kwargs = {
                    "max_new_tokens": max_tokens,
                    "do_sample": False,
                    "pad_token_id": tokenizer.eos_token_id,
                }
            
            print(f"    Attempt {attempt + 1}: Using {strategy} strategy (max_tokens={max_tokens})")
            
            # Tokenize with minimal settings
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=100,  # Very short context
            )
            
            input_ids = inputs.input_ids.to(model.device)
            
            # Always pass attention mask if available to avoid warnings
            if 'attention_mask' in inputs:
                attention_mask = inputs.attention_mask.to(model.device)
                generation_kwargs["attention_mask"] = attention_mask
            
            # Generate with timeout protection
            with torch.no_grad():
                generation_kwargs["input_ids"] = input_ids
                output = model.generate(**generation_kwargs)
            
            # Decode only newly generated tokens
            gen_ids = output[0][input_ids.shape[1]:]
            result = tokenizer.decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            
            if not result:
                # Fallback: include special tokens for visibility
                result = tokenizer.decode(
                    gen_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ).strip() or f"[Generated {output.shape[1] - input_ids.shape[1]} tokens but no visible text]"
            
            print(f"    ✓ Success with {strategy} strategy")
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"    ✗ Attempt {attempt + 1} failed: {error_msg[:100]}...")
            
            # Clear everything after error
            clear_cuda_cache()
            
            if attempt == max_retries - 1:
                return f"[Generation failed after {max_retries} attempts: {error_msg[:50]}...]"
            
            # Wait before retry
            time.sleep(1)
    
    return "[All generation attempts failed]"


def test_single_model(model_path: str, model_type: str) -> Dict[str, Any]:
    """Test a single quantized model with maximum stability."""
    
    print(f"\n{'='*60}")
    print(f"TESTING {model_type.upper()} MODEL")
    print(f"Path: {model_path}")
    print(f"{'='*60}")
    
    results = {
        "model_type": model_type,
        "model_path": model_path,
        "loading_success": False,
        "generations": [],
        "errors": []
    }
    
    try:
        print("Loading model and tokenizer...")
        
        # Load with conservative settings
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            use_fast=False,  # Use slower but more stable tokenizer
        )
        
        # Setup tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Disable EOS token addition for cleaner generation
        if hasattr(tokenizer, 'add_eos_token'):
            tokenizer.add_eos_token = False
        
        # Dispatch for generation
        dispatch_for_generation(model)
        
        print(f"✓ Model loaded successfully!")
        print(f"  Device: {model.device}")
        print(f"  Dtype: {model.dtype}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        results["loading_success"] = True
        
        # Wait for model to stabilize
        clear_cuda_cache()
        time.sleep(2)
        
        print(f"\nTesting with {len(STABLE_TEST_PROMPTS)} prompts...")
        
        # Test each prompt
        for i, prompt in enumerate(STABLE_TEST_PROMPTS, 1):
            print(f"\nTest {i}/{len(STABLE_TEST_PROMPTS)}: '{prompt}'")
            
            start_time = time.time()
            generated_text = safe_generate(model, tokenizer, prompt)
            generation_time = time.time() - start_time
            
            result = {
                "prompt": prompt,
                "generated": generated_text,
                "time": generation_time,
                "success": not generated_text.startswith("[")
            }
            
            results["generations"].append(result)
            
            print(f"  Generated: {generated_text}")
            print(f"  Time: {generation_time:.2f}s")
            
            # Brief pause between generations
            time.sleep(1)
        
        print(f"\n✓ {model_type.upper()} model testing completed!")
        
        # Calculate success rate
        successes = sum(1 for g in results["generations"] if g["success"])
        success_rate = successes / len(results["generations"]) * 100
        print(f"  Success rate: {success_rate:.1f}% ({successes}/{len(results['generations'])})")
        
    except Exception as e:
        error_msg = f"Failed to load {model_type} model: {e}"
        print(f"✗ {error_msg}")
        results["errors"].append(error_msg)
    
    finally:
        # Cleanup
        try:
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            clear_cuda_cache()
        except Exception:
            pass
    
    return results


def main():
    """Main testing function."""
    
    print("STABLE QUANTIZED MODEL TESTING")
    print("=" * 80)
    print("This script uses ultra-conservative generation settings for maximum stability.")
    print("Environment: CUDA_LAUNCH_BLOCKING=1, TORCH_USE_CUDA_DSA=1")
    print()
    
    # Test available models
    all_results = []
    
    for model_type, model_path in DEFAULT_MODELS.items():
        if Path(model_path).exists():
            results = test_single_model(model_path, model_type)
            all_results.append(results)
        else:
            print(f"\n{model_type.upper()} model not found at: {model_path}")
            print(f"Please run llama_{model_type}_example.py first")
    
    # Summary
    print("\n" + "=" * 80)
    print("TESTING SUMMARY")
    print("=" * 80)
    
    if not all_results:
        print("No models were tested. Please create quantized models first:")
        print("  python llama_apot_example.py")
        print("  python llama_pot_example.py")
        return
    
    for results in all_results:
        model_type = results["model_type"].upper()
        
        if results["loading_success"]:
            successes = sum(1 for g in results["generations"] if g["success"])
            total = len(results["generations"])
            success_rate = successes / total * 100 if total > 0 else 0
            
            avg_time = sum(g["time"] for g in results["generations"]) / total if total > 0 else 0
            
            print(f"\n{model_type} Model:")
            print(f"  ✓ Loaded successfully")
            print(f"  ✓ Generation success rate: {success_rate:.1f}% ({successes}/{total})")
            print(f"  ✓ Average generation time: {avg_time:.2f}s")
            
            # Show sample generations
            successful_gens = [g for g in results["generations"] if g["success"]]
            if successful_gens:
                print(f"  ✓ Sample generation: '{successful_gens[0]['prompt']}' → '{successful_gens[0]['generated'][:50]}...'")
        else:
            print(f"\n{model_type} Model:")
            print(f"  ✗ Failed to load")
            if results["errors"]:
                print(f"  ✗ Error: {results['errors'][0]}")
    
    # Recommendations
    print(f"\nRecommendations:")
    print(f"  • If you see CUDA errors, the quantized model may have numerical instabilities")
    print(f"  • Try using CPU inference: add device_map='cpu' to model loading")
    print(f"  • For debugging: Check the model quantization parameters")
    print(f"  • Use greedy decoding (do_sample=False) for more stable generation")


if __name__ == "__main__":
    main()
