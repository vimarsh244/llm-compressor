"""Combined test script for both APoT and PoT quantized models."""

from pathlib import Path
import time
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.utils.dev import dispatch_for_generation

# Test prompts - varied to test different aspects
TEST_PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "Write a short story about a robot learning to paint:",
    "Explain quantum computing in simple terms:",
    "What are the benefits of renewable energy?",
    "Describe the process of photosynthesis:",
    "Tell me a joke about programming:",
    "What is the meaning of life?",
    "How do neural networks work?",
    "Write a recipe for chocolate cake:"
]

# Default model paths
DEFAULT_MODELS = {
    "apot": "TinyLlama-1.1B-Chat-v1.0-apot-w4a8-t2",
    "pot": "TinyLlama-1.1B-Chat-v1.0-pot-w4a8"
}


def test_quantized_model(model_path: str, model_type: str, use_greedy: bool = True):
    """Test a quantized model with various prompts."""
    
    print(f"Loading {model_type.upper()} quantized model from: {model_path}")
    
    try:
        # Load the quantized model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Set up tokenizer properly for quantized models
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # For llama models, we need to ensure proper token setup
        if hasattr(tokenizer, 'add_eos_token'):
            tokenizer.add_eos_token = False
        
        # Dispatch for generation (important for quantized models)
        dispatch_for_generation(model)
        
        print(f"Model loaded successfully! Device: {model.device}")
        print(f"Model dtype: {model.dtype}")
        print(f"Generation mode: {'Greedy' if use_greedy else 'Sampling'}")
        print("=" * 60)
        
        # Test with various prompts
        for i, prompt in enumerate(TEST_PROMPTS, 1):
            print(f"Test {i}/{len(TEST_PROMPTS)} - {model_type.upper()}")
            print(f"Prompt: {prompt}")
            
            try:
                # Tokenize input with proper attention mask
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,  # Don't pad single inputs
                    truncation=True,
                    max_length=512,
                )
                
                input_ids = inputs.input_ids.to(model.device)
                attention_mask = inputs.attention_mask.to(model.device) if 'attention_mask' in inputs else None
                
                # Use more conservative generation parameters for quantized models
                generation_kwargs = {
                    "input_ids": input_ids,
                    "max_new_tokens": 100,  # Reduced from 128
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                
                # Always pass attention mask to avoid inference warnings
                if attention_mask is not None:
                    generation_kwargs["attention_mask"] = attention_mask
                
                if use_greedy:
                    # Greedy decoding - more stable for quantized models
                    generation_kwargs.update({
                        "do_sample": False,
                        "num_beams": 1,
                    })
                else:
                    # Conservative sampling parameters
                    generation_kwargs.update({
                        "do_sample": True,
                        "temperature": 0.8,  # Slightly higher for stability
                        "top_p": 0.95,      # More conservative
                        "top_k": 50,        # Add top_k filtering
                        "repetition_penalty": 1.1,  # Prevent repetition
                    })
                
                # Generate response with error handling for CUDA issues
                with torch.no_grad():
                    torch.cuda.empty_cache()  # Clear cache before generation
                    
                    try:
                        output = model.generate(**generation_kwargs)
                    except RuntimeError as cuda_error:
                        if "CUDA" in str(cuda_error):
                            print(f"CUDA error encountered, trying with reduced parameters...")
                            # Fallback to very conservative settings
                            fallback_kwargs = {
                                "input_ids": input_ids,
                                "max_new_tokens": 50,
                                "do_sample": False,
                                "pad_token_id": tokenizer.eos_token_id,
                                "eos_token_id": tokenizer.eos_token_id,
                            }
                            if attention_mask is not None:
                                fallback_kwargs["attention_mask"] = attention_mask
                            
                            output = model.generate(**fallback_kwargs)
                        else:
                            raise cuda_error
                
                # Decode only the newly generated tokens
                generated_ids = output[0][input_ids.shape[1]:]
                generated_part = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                
                # If still empty, try decoding without skipping special tokens for visibility
                if not generated_part or generated_part.strip() == "":
                    alt_decoded = tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    generated_part = alt_decoded if alt_decoded else "[No text generated]"
                
                print(f"Generated: {generated_part}")
                
            except Exception as e:
                print(f"Error during generation: {e}")
                print("Trying to continue with next prompt...")
                
                # Clear CUDA cache after error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print("-" * 40)
            time.sleep(1)  # Brief pause between generations
        
        print(f"{model_type.upper()} model testing completed!")
        
        # Clean up memory
        del model
        del tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"Error loading {model_type.upper()} model: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test quantized models (APoT and PoT)")
    parser.add_argument(
        "--model-type", 
        choices=["apot", "pot", "both"], 
        default="both",
        help="Type of model to test (default: both)"
    )
    parser.add_argument(
        "--apot-path", 
        type=str, 
        default=DEFAULT_MODELS["apot"],
        help=f"Path to APoT model (default: {DEFAULT_MODELS['apot']})"
    )
    parser.add_argument(
        "--pot-path", 
        type=str, 
        default=DEFAULT_MODELS["pot"],
        help=f"Path to PoT model (default: {DEFAULT_MODELS['pot']})"
    )
    parser.add_argument(
        "--use-sampling",
        action="store_true",
        help="Use sampling instead of greedy decoding (less stable but more creative)"
    )
    parser.add_argument(
        "--cuda-debug",
        action="store_true", 
        help="Set CUDA_LAUNCH_BLOCKING=1 for better error debugging"
    )
    
    args = parser.parse_args()
    
    # Set CUDA debugging if requested
    if args.cuda_debug:
        import os
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        print("CUDA_LAUNCH_BLOCKING=1 set for debugging")
    
    use_greedy = not args.use_sampling
    
    # Test APoT model
    if args.model_type in ["apot", "both"]:
        if Path(args.apot_path).exists():
            test_quantized_model(args.apot_path, "apot", use_greedy)
        else:
            print(f"APoT model not found at: {args.apot_path}")
            print("Please run llama_apot_example.py first to create the quantized model.")
        
        if args.model_type == "both":
            print("\n" + "=" * 80 + "\n")
    
    # Test PoT model
    if args.model_type in ["pot", "both"]:
        if Path(args.pot_path).exists():
            test_quantized_model(args.pot_path, "pot", use_greedy)
        else:
            print(f"PoT model not found at: {args.pot_path}")
            print("Please run llama_pot_example.py first to create the quantized model.")


if __name__ == "__main__":
    main()