#!/usr/bin/env python3
"""
Integration test for PoT and APoT quantization on a small model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization.pot import PoTQuantizationModifier
from llmcompressor.modifiers.quantization.apot import APoTQuantizationModifier


def create_dummy_dataset():
    """Create a simple dummy dataset for calibration."""
    class DummyDataset:
        def __init__(self):
            self.data = [
                {"text": "Hello world, this is a test."},
                {"text": "The quick brown fox jumps over the lazy dog."},
                {"text": "Machine learning is fascinating."},
                {"text": "Power-of-two quantization enables efficient inference."},
            ]
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    return DummyDataset()


def test_pot_quantization():
    """Test PoT quantization on a small model."""
    print("Testing PoT quantization...")
    
    try:
        # use a very small model for testing
        MODEL_ID = "microsoft/DialoGPT-small"
        
        print(f"Loading model: {MODEL_ID}")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Model loaded successfully")
        
        # create PoT quantization modifier
        pot_modifier = PoTQuantizationModifier(
            targets=["Linear"],
            ignore=["lm_head"],
            pot_bits=4,
            scheme={
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "tensor",  # use tensor strategy for simplicity
                    "observer": "pot",
                }
            }
        )
        
        print("PoT modifier created")
        
        # create dummy calibration data
        dummy_dataset = create_dummy_dataset()
        
        print("Applying PoT quantization...")
        
        # apply quantization
        oneshot(
            model=model,
            dataset=dummy_dataset,
            recipe=pot_modifier,
            max_seq_length=128,
            num_calibration_samples=4,
        )
        
        print("✅ PoT quantization completed successfully!")
        
        # test inference
        input_text = "Hello"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ PoT quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_apot_quantization():
    """Test APoT quantization on a small model."""
    print("\nTesting APoT quantization...")
    
    try:
        # use a very small model for testing
        MODEL_ID = "microsoft/DialoGPT-small"
        
        print(f"Loading model: {MODEL_ID}")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Model loaded successfully")
        
        # create APoT quantization modifier
        apot_modifier = APoTQuantizationModifier(
            targets=["Linear"],
            ignore=["lm_head"],
            apot_bits=4,
            num_terms=2,
            scheme={
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "tensor",  # use tensor strategy for simplicity
                    "observer": "apot",
                }
            }
        )
        
        print("APoT modifier created")
        
        # create dummy calibration data
        dummy_dataset = create_dummy_dataset()
        
        print("Applying APoT quantization...")
        
        # apply quantization
        oneshot(
            model=model,
            dataset=dummy_dataset,
            recipe=apot_modifier,
            max_seq_length=128,
            num_calibration_samples=4,
        )
        
        print("✅ APoT quantization completed successfully!")
        
        # test inference
        input_text = "Hello"
        inputs = tokenizer(input_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
        
        return True
        
    except Exception as e:
        print(f"❌ APoT quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("INTEGRATION TEST: PoT/APoT QUANTIZATION")
    print("="*60)
    
    pot_success = test_pot_quantization()
    apot_success = test_apot_quantization()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"PoT quantization: {'✅ PASSED' if pot_success else '❌ FAILED'}")
    print(f"APoT quantization: {'✅ PASSED' if apot_success else '❌ FAILED'}")
    
    if pot_success and apot_success:
        print("\n🎉 All integration tests passed!")
        print("The PoT/APoT quantization implementation is ready for use.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")