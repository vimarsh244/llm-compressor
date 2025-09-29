"""Benchmark script to compare APoT and PoT quantized models."""

from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.utils.dev import dispatch_for_generation

# Benchmark prompts
BENCHMARK_PROMPTS = [
    "The capital of France is",
    "Explain artificial intelligence:",
    "Write a Python function to calculate factorial:",
    "What are the main benefits of machine learning?",
    "Describe the process of neural network training:"
]


def benchmark_model(model_path: str, model_type: str):
    """Benchmark a quantized model and return performance metrics."""
    
    print(f"Benchmarking {model_type.upper()} model: {model_path}")
    
    try:
        # Load model and tokenizer
        start_load = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        dispatch_for_generation(model)
        load_time = time.time() - start_load
        
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Benchmark generation
        total_inference_time = 0
        total_tokens_generated = 0
        successful_generations = 0
        
        for i, prompt in enumerate(BENCHMARK_PROMPTS):
            print(f"  Benchmark {i+1}/{len(BENCHMARK_PROMPTS)}: {prompt[:30]}...")
            
            try:
                input_ids = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).input_ids.to(model.device)
                
                # Measure inference time
                start_inference = time.time()
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_new_tokens=100,
                        do_sample=False,  # Deterministic for fair comparison
                        temperature=1.0,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                
                inference_time = time.time() - start_inference
                total_inference_time += inference_time
                
                # Count tokens generated
                tokens_generated = output.shape[1] - input_ids.shape[1]
                total_tokens_generated += tokens_generated
                successful_generations += 1
                
                # Print the generation for quality assessment
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                if generated_text.startswith(prompt):
                    generated_part = generated_text[len(prompt):].strip()
                else:
                    generated_part = generated_text
                
                print(f"    Generated ({tokens_generated} tokens in {inference_time:.2f}s): {generated_part[:100]}...")
                
            except Exception as e:
                print(f"    Error: {e}")
        
        # Calculate metrics
        avg_inference_time = total_inference_time / successful_generations if successful_generations > 0 else 0
        tokens_per_second = total_tokens_generated / total_inference_time if total_inference_time > 0 else 0
        
        # Get model size estimation
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        results = {
            "model_type": model_type,
            "load_time": load_time,
            "avg_inference_time": avg_inference_time,
            "total_tokens_generated": total_tokens_generated,
            "tokens_per_second": tokens_per_second,
            "successful_generations": successful_generations,
            "model_size_mb": model_size_mb
        }
        
        # Clean up
        del model
        del tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return results
        
    except Exception as e:
        print(f"Error benchmarking {model_type} model: {e}")
        return None


def main():
    # Model paths
    models_to_test = [
        ("APoT", "TinyLlama-1.1B-Chat-v1.0-apot-w4a8-t2"),
        ("PoT", "TinyLlama-1.1B-Chat-v1.0-pot-w4a8")
    ]
    
    results = []
    
    print("=" * 80)
    print("QUANTIZED MODEL BENCHMARK")
    print("=" * 80)
    
    for model_type, model_path in models_to_test:
        if Path(model_path).exists():
            result = benchmark_model(model_path, model_type)
            if result:
                results.append(result)
            print("-" * 60)
        else:
            print(f"{model_type} model not found at: {model_path}")
            print(f"Please run llama_{model_type.lower()}_example.py first")
            print("-" * 60)
    
    # Print comparison results
    if len(results) >= 2:
        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)
        
        for result in results:
            print(f"\n{result['model_type']} Model:")
            print(f"  Load time: {result['load_time']:.2f}s")
            print(f"  Avg inference time: {result['avg_inference_time']:.3f}s")
            print(f"  Tokens per second: {result['tokens_per_second']:.1f}")
            print(f"  Model size: {result['model_size_mb']:.1f} MB")
            print(f"  Successful generations: {result['successful_generations']}/{len(BENCHMARK_PROMPTS)}")
        
        # Compare performance
        if len(results) == 2:
            apot_result = next(r for r in results if r['model_type'] == 'APoT')
            pot_result = next(r for r in results if r['model_type'] == 'PoT')
            
            print(f"\nPerformance Comparison:")
            
            speed_diff = (apot_result['tokens_per_second'] / pot_result['tokens_per_second'] - 1) * 100
            if speed_diff > 0:
                print(f"  APoT is {speed_diff:.1f}% faster than PoT")
            else:
                print(f"  PoT is {-speed_diff:.1f}% faster than APoT")
            
            size_diff = (apot_result['model_size_mb'] / pot_result['model_size_mb'] - 1) * 100
            if size_diff > 0:
                print(f"  APoT model is {size_diff:.1f}% larger than PoT")
            else:
                print(f"  PoT model is {-size_diff:.1f}% larger than APoT")
    
    elif len(results) == 1:
        print(f"\nOnly {results[0]['model_type']} model was tested.")
        print("Run the other quantization example to compare both models.")
    
    else:
        print("\nNo models were successfully benchmarked.")
        print("Please run the quantization examples first:")
        print("  python llama_apot_example.py")
        print("  python llama_pot_example.py")


if __name__ == "__main__":
    main()
