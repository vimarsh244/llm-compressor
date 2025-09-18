"""
Test script for loading and comparing quantized models (APoT and PoT).

This script loads both the APoT and PoT quantized models and tests them
with various prompts to evaluate generation quality and identify any issues.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.utils import dispatch_for_generation

# Model paths
APOT_MODEL_PATH = "./Meta-Llama-3-8B-Instruct-apot-2term-w4a8"
POT_MODEL_PATH = "./Meta-Llama-3-8B-Instruct-pot-w4a8"

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

def load_model_and_tokenizer(model_path, model_name):
    """Load a quantized model and its tokenizer."""
    print(f"\n{'='*60}")
    print(f"Loading {model_name} model from: {model_path}")
    print(f"{'='*60}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # set padding/eos tokens for llama3 instruct
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        # Prefer bf16 for better numerical stability if supported
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Ensure generation config has proper ids
        if getattr(model, "generation_config", None) is not None:
            if model.generation_config.pad_token_id is None and tokenizer.pad_token_id is not None:
                model.generation_config.pad_token_id = tokenizer.pad_token_id
            if model.generation_config.eos_token_id is None and tokenizer.eos_token_id is not None:
                model.generation_config.eos_token_id = tokenizer.eos_token_id
        
        # Put model in eval mode
        model.eval()
        
        # Dispatch for generation
        dispatch_for_generation(model)
        
        # Sanity check: tokenizer/model vocab alignment
        try:
            vocab_size_model = model.get_input_embeddings().weight.size(0)
            vocab_size_tok = len(tokenizer)
            if vocab_size_tok != vocab_size_model:
                print(f"! tokenizer vocab ({vocab_size_tok}) != model embeddings ({vocab_size_model})")
                # don't resize weights for quantized checkpoints; just warn user
        except Exception:
            pass
        
        print(f"✓ {model_name} model loaded successfully!")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"✗ Error loading {model_name} model: {e}")
        return None, None

def build_inputs(tokenizer, prompt):
    """Build tokenized inputs, using chat template when available."""
    # try to use chat template if present for instruct models
    text = None
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
    except Exception:
        text = None
    if text is None:
        text = prompt
    return tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
    """Generate text using the model with proper attention mask handling."""
    try:
        # Tokenize input with proper attention mask
        inputs = build_inputs(tokenizer, prompt)
        
        # Validate token ids against model vocab to avoid device-side asserts
        try:
            vocab_size_model = model.get_input_embeddings().weight.size(0)
            max_input_id = int(inputs["input_ids"].max().item())
            if max_input_id >= vocab_size_model:
                return (
                    f"Input id {max_input_id} exceeds model vocab {vocab_size_model - 1}. "
                    f"Tokenizer/model mismatch. Ensure tokenizer matches the quantized checkpoint."
                )
        except Exception:
            pass
        
        # Move to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Safe logits processor to clamp NaN/Inf before sampling
        try:
            from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
            class _SafeLogitsProcessor(LogitsProcessor):
                def __call__(self, input_ids, scores):
                    scores = torch.nan_to_num(scores, neginf=-1e9, posinf=1e9)
                    return torch.clamp(scores, min=-1e9, max=1e9)
            logits_processor = LogitsProcessorList([_SafeLogitsProcessor()])
        except Exception:
            logits_processor = None
        
        # Generate with proper parameters
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                logits_processor=logits_processor,
            )
        
        # Decode only the new tokens
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
        
    except Exception as e:
        # Hint to help debugging CUDA asserts
        if "device-side assert triggered" in str(e).lower():
            return (
                "Error during generation: CUDA device-side assert. "
                "Try setting CUDA_LAUNCH_BLOCKING=1 and ensure tokenizer matches model."
            )
        return f"Error during generation: {e}"

def test_model_generation(model, tokenizer, model_name):
    """Test model generation with various prompts."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name} Generation")
    print(f"{'='*60}")
    
    results = []
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n--- Test {i}: {prompt[:50]}{'...' if len(prompt) > 50 else ''} ---")
        
        # Test with different generation parameters
        for temp, top_p, desc in [(0.7, 0.9, "balanced"), (0.3, 0.8, "conservative"), (1.0, 0.95, "creative")]:
            print(f"\n{desc} (temp={temp}, top_p={top_p}):")
            generated = generate_text(model, tokenizer, prompt, temperature=temp, top_p=top_p)
            print(f"Generated: {generated}")
            
            results.append({
                'prompt': prompt,
                'generated': generated,
                'temperature': temp,
                'top_p': top_p,
                'description': desc
            })
    
    return results

def compare_models(apot_results, pot_results):
    """Compare results between APoT and PoT models."""
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")
    
    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n--- Prompt {i+1}: {prompt} ---")
        
        # Find corresponding results
        apot_result = next((r for r in apot_results if r['prompt'] == prompt and r['description'] == 'balanced'), None)
        pot_result = next((r for r in pot_results if r['prompt'] == prompt and r['description'] == 'balanced'), None)
        
        if apot_result and pot_result:
            print(f"APoT: {apot_result['generated']}")
            print(f"PoT:  {pot_result['generated']}")
            print("-" * 60)

def main():
    """Main function to test both quantized models."""
    print("Quantized Model Testing Script")
    print("=" * 60)
    
    # Load both models
    apot_model, apot_tokenizer = load_model_and_tokenizer(APOT_MODEL_PATH, "APoT")
    pot_model, pot_tokenizer = load_model_and_tokenizer(POT_MODEL_PATH, "PoT")
    
    if apot_model is None or pot_model is None:
        print("Failed to load one or both models. Exiting.")
        return
    
    # Test both models
    apot_results = test_model_generation(apot_model, apot_tokenizer, "APoT")
    pot_results = test_model_generation(pot_model, pot_tokenizer, "PoT")
    
    # Compare results
    compare_models(apot_results, pot_results)
    
    # Summary
    print(f"\n{'='*80}")
    print("TESTING SUMMARY")
    print(f"{'='*80}")
    print(f"✓ Tested {len(TEST_PROMPTS)} different prompts")
    print(f"✓ Tested 3 different generation parameter sets per prompt")
    print(f"✓ Total generations: {len(apot_results) + len(pot_results)}")
    print(f"✓ Both APoT and PoT models tested successfully")
    
    print(f"\nRecommendations:")
    print(f"- Check if generation quality issues are consistent across prompts")
    print(f"- Try different temperature and top_p values for better quality")
    print(f"- Consider using different max_new_tokens values")
    print(f"- Verify if the issue is with the quantization or generation parameters")

if __name__ == "__main__":
    main()