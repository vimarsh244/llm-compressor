"""Fixed test script using proper loading for quantized models instead of vLLM."""

from pathlib import Path
import time

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
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
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
                # Tokenize input
                input_ids = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).input_ids.to(model.device)
                
                # Generate response with settings similar to original vLLM params
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        max_new_tokens=256,  # Similar to max_tokens in original
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                
                # Decode generated text
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Extract only the generated part (remove the input prompt)
                if generated_text.startswith(prompt):
                    generated_part = generated_text[len(prompt):].strip()
                else:
                    generated_part = generated_text
                
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
