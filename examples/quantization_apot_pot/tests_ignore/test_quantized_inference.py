"""Fixed test script using proper loading for quantized models instead of vLLM."""

from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.utils.dev import dispatch_for_generation
from llmcompressor.transformers.compression.quantization_format import (
    infer_and_set_per_module_quantization_format,
)

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


def load_and_test_model(model_path: str):
    """Load and test a quantized model."""
    
    print(f"Loading model from: {model_path}")
    
    try:
        # Load the model and tokenizer
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
        
        # For llama models, disable automatic EOS token addition
        if hasattr(tokenizer, 'add_eos_token'):
            tokenizer.add_eos_token = False
        
        # Ensure generation config has proper pad/eos ids
        try:
            if getattr(model.config, "pad_token_id", None) is None:
                model.config.pad_token_id = tokenizer.eos_token_id
            if getattr(model.config, "eos_token_id", None) is None:
                model.config.eos_token_id = tokenizer.eos_token_id
            model.generation_config.pad_token_id = tokenizer.eos_token_id
            model.generation_config.eos_token_id = tokenizer.eos_token_id
        except Exception:
            pass

        # Important: set per-module quantization format for compressed-tensors
        try:
            infer_and_set_per_module_quantization_format(model, save_compressed=False)
        except Exception:
            pass

        # Dispatch for generation (important for quantized models)
        dispatch_for_generation(model)
        
        print(f"Model loaded successfully!")
        print(f"Device: {model.device}")
        print(f"Model dtype: {model.dtype}")
        print("=" * 60)
        
        # Test with all prompts
        for i, prompt in enumerate(TEST_PROMPTS, 1):
            print(f"Test {i}/{len(TEST_PROMPTS)}")
            print(f"Prompt: {prompt}")
            
            try:
                # Tokenize input with proper attention mask handling
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,  # Avoid padding for single inputs
                    truncation=True,
                    max_length=512,
                )
                
                input_ids = inputs.input_ids.to(model.device)
                attention_mask = inputs.attention_mask.to(model.device) if 'attention_mask' in inputs else None
                
                # Use the same generation scheme as example scripts
                generation_kwargs = {
                    "input_ids": input_ids,
                    "max_new_tokens": 128,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "use_cache": True,
                }
                
                # Add attention mask if available and meaningful
                if attention_mask is not None:
                    generation_kwargs["attention_mask"] = attention_mask

                # Do not add additional constraints; mirror example behavior closely
                
                # Generate response with error handling
                with torch.no_grad():
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    try:
                        output = model.generate(**generation_kwargs)
                    except RuntimeError as cuda_error:
                        if "CUDA" in str(cuda_error):
                            print(f"    CUDA error, trying fallback generation...")
                            # Fallback with minimal settings
                            fallback_kwargs = {
                                "input_ids": input_ids,
                                "max_new_tokens": 50,
                                "do_sample": False,
                                "pad_token_id": tokenizer.eos_token_id,
                            }
                            output = model.generate(**fallback_kwargs)
                        else:
                            raise cuda_error
                
                # Decode only the newly generated tokens
                gen_ids = output[0][input_ids.shape[1]:]
                generated_part = tokenizer.decode(
                    gen_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                
                if not generated_part or generated_part.strip() == "":
                    alt = tokenizer.decode(
                        gen_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    generated_part = alt if alt else "[No text generated]"

                # If generation is mostly <unk>, do not force suppression; keep original behavior
                
                print(f"Generated: {generated_part}")
                
            except Exception as e:
                print(f"Error during generation: {e}")
            
            print("-" * 40)
            time.sleep(2)  # Keep the same timing as original
        
        print("Model testing completed!")
        
    except Exception as e:
        print(f"Error loading model: {e}")


def main():
    # Test available quantized models
    available_models = []
    
    # Check for APoT model
    apot_path = "TinyLlama-1.1B-Chat-v1.0-apot-w4a8-t2"
    if Path(apot_path).exists():
        available_models.append(("APoT", apot_path))
    
    # Check for PoT model
    pot_path = "TinyLlama-1.1B-Chat-v1.0-pot-w4a8"
    if Path(pot_path).exists():
        available_models.append(("PoT", pot_path))
    
    # You can also test with specific model paths
    # Uncomment and modify these lines to test specific models:
    # available_models.append(("Custom APoT", "path/to/your/apot/model"))
    # available_models.append(("Custom PoT", "path/to/your/pot/model"))
    
    if not available_models:
        print("No quantized models found!")
        print("Available model paths to check:")
        print(f"  APoT: {apot_path}")
        print(f"  PoT: {pot_path}")
        print("\nPlease run the quantization examples first:")
        print("  python llama_apot_example.py")
        print("  python llama_pot_example.py")
        return
    
    # Test each available model
    for model_type, model_path in available_models:
        print(f"\n{'='*80}")
        print(f"TESTING {model_type} MODEL")
        print(f"{'='*80}")
        
        load_and_test_model(model_path)
        
        # Clean up memory between models
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
