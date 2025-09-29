"""Debug script for diagnosing quantized model issues."""

import os
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.utils.dev import dispatch_for_generation

# Enable full CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

def diagnose_model(model_path: str, model_type: str):
    """Diagnose issues with a quantized model."""
    
    print(f"DIAGNOSING {model_type.upper()} MODEL")
    print(f"Path: {model_path}")
    print("=" * 60)
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model directory does not exist: {model_path}")
        return False
    
    print(f"✅ Model directory exists")
    
    # Check required files
    required_files = ["config.json", "model.safetensors"]
    for file in required_files:
        file_path = Path(model_path) / file
        if file_path.exists():
            print(f"✅ Found {file}")
        else:
            print(f"❌ Missing {file}")
    
    # Try loading model
    try:
        print("\n🔄 Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="cpu",  # Use CPU first for safety
            trust_remote_code=True,
        )
        print(f"✅ Model loaded on CPU")
        print(f"   Model type: {type(model)}")
        print(f"   Model dtype: {model.dtype}")
        print(f"   Parameter count: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Try loading tokenizer
    try:
        print("\n🔄 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"✅ Tokenizer loaded")
        print(f"   Tokenizer type: {type(tokenizer)}")
        print(f"   Vocab size: {len(tokenizer)}")
        print(f"   Pad token: {tokenizer.pad_token}")
        print(f"   EOS token: {tokenizer.eos_token}")
        
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return False
    
    # Setup tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Set pad token to eos token")
    
    # Try moving to GPU
    if torch.cuda.is_available():
        try:
            print(f"\n🔄 Moving model to GPU...")
            model = model.cuda()
            print(f"✅ Model moved to GPU: {model.device}")
        except Exception as e:
            print(f"❌ Failed to move to GPU: {e}")
            print(f"ℹ️  Continuing with CPU inference")
    else:
        print(f"ℹ️  CUDA not available, using CPU")
    
    # Try dispatch for generation
    try:
        print(f"\n🔄 Setting up generation...")
        dispatch_for_generation(model)
        print(f"✅ Generation setup completed")
    except Exception as e:
        print(f"❌ Failed to setup generation: {e}")
        return False
    
    # Test simple tokenization
    try:
        print(f"\n🔄 Testing tokenization...")
        test_text = "Hello"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ Tokenization successful")
        print(f"   Input text: '{test_text}'")
        print(f"   Token IDs: {tokens.input_ids.tolist()}")
        print(f"   Token count: {tokens.input_ids.shape[1]}")
        
        # Decode back
        decoded = tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)
        print(f"   Decoded: '{decoded}'")
        
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        return False
    
    # Test minimal generation
    print(f"\n🔄 Testing minimal generation...")
    
    # Test 1: Absolute minimum generation
    try:
        input_ids = tokenizer("Hi", return_tensors="pt").input_ids.to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"✅ Minimal generation successful: '{generated}'")
        
    except Exception as e:
        print(f"❌ Minimal generation failed: {e}")
        print(f"   This indicates a fundamental issue with the quantized model")
        
        # Try to get more specific error info
        print(f"\n🔍 Detailed error analysis:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        
        if "CUDA" in str(e):
            print(f"   🚨 CUDA-related error detected")
            print(f"   💡 This suggests numerical instability in quantization")
            print(f"   💡 Try using CPU inference or check quantization parameters")
        
        if "assert" in str(e).lower():
            print(f"   🚨 Assertion error detected")
            print(f"   💡 This usually indicates invalid probability values")
            print(f"   💡 Check if the quantization introduced NaN or inf values")
        
        return False
    
    # Test 2: Slightly longer generation
    try:
        print(f"\n🔄 Testing longer generation...")
        input_ids = tokenizer("The", return_tensors="pt").input_ids.to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"✅ Longer generation successful: '{generated}'")
        
    except Exception as e:
        print(f"❌ Longer generation failed: {e}")
        print(f"   Model works for single tokens but fails for longer sequences")
    
    # Cleanup
    try:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    
    print(f"\n✅ Diagnosis completed for {model_type.upper()} model")
    return True


def main():
    """Main diagnostic function."""
    
    print("QUANTIZED MODEL DIAGNOSTICS")
    print("=" * 80)
    print("This script diagnoses issues with APoT and PoT quantized models.")
    print("CUDA debugging is enabled for detailed error reporting.")
    print()
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name()}")
    else:
        print(f"❌ CUDA not available")
    
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Models to diagnose
    models = {
        "apot": "TinyLlama-1.1B-Chat-v1.0-apot-w4a8-t2",
        "pot": "TinyLlama-1.1B-Chat-v1.0-pot-w4a8"
    }
    
    results = {}
    
    for model_type, model_path in models.items():
        try:
            success = diagnose_model(model_path, model_type)
            results[model_type] = success
        except Exception as e:
            print(f"❌ Unexpected error diagnosing {model_type}: {e}")
            results[model_type] = False
        
        print("\n" + "-" * 80 + "\n")
    
    # Summary
    print("DIAGNOSTIC SUMMARY")
    print("=" * 40)
    
    for model_type, success in results.items():
        status = "✅ WORKING" if success else "❌ ISSUES FOUND"
        print(f"{model_type.upper()} model: {status}")
    
    if not any(results.values()):
        print(f"\n❌ All models have issues. Recommendations:")
        print(f"   1. Check if quantization completed successfully")
        print(f"   2. Try re-running the quantization examples")
        print(f"   3. Use CPU inference instead of GPU")
        print(f"   4. Check available GPU memory")
    elif all(results.values()):
        print(f"\n✅ All models appear to be working correctly!")
        print(f"   You can now use the regular test scripts.")
    else:
        print(f"\n⚠️  Some models work, others don't.")
        print(f"   Focus on debugging the failing models.")


if __name__ == "__main__":
    main()
