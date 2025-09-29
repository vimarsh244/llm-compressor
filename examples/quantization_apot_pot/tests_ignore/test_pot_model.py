"""Test script for PoT quantized model."""

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


def test_pot_model(model_path: str):
    """Test the PoT quantized model with various prompts."""
    
    print(f"Loading PoT quantized model from: {model_path}")
    
    # Load the quantized model and tokenizer
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
    
    print(f"Model loaded successfully! Device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    print("=" * 60)
    
    # Test with various prompts
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
            
            # Generate response
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=128,
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
        time.sleep(1)  # Brief pause between generations
    
    print("Testing completed!")


def main():
    # Default model path (created by llama_pot_example.py)
    model_path = "TinyLlama-1.1B-Chat-v1.0-pot-w4a8"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Model not found at: {model_path}")
        print("Please run llama_pot_example.py first to create the quantized model.")
        return
    
    test_pot_model(model_path)


if __name__ == "__main__":
    main()
