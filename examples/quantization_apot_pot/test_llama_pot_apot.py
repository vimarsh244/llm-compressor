#!/usr/bin/env python3
"""
Test PoT and APoT quantization on a full LLaMA model.
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization.pot import PoTQuantizationModifier
from llmcompressor.modifiers.quantization.apot import APoTQuantizationModifier
from llmcompressor.utils import dispatch_for_generation


def test_llama_pot_quantization():
    """Test PoT quantization on LLaMA model."""
    print("Testing PoT quantization on LLaMA...")
    
    # use a small LLaMA model for testing
    MODEL_ID = "microsoft/DialoGPT-medium"  # or use "meta-llama/Llama-2-7b-hf" if available
    
    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # load calibration dataset
    print("Loading calibration dataset...")
    DATASET_ID = "mit-han-lab/pile-val-backup"
    ds = load_dataset(DATASET_ID, split="validation[:32]")  # small subset for testing
    ds = ds.shuffle(seed=42)
    
    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                [{"role": "user", "content": example["text"]}],
                tokenize=False,
            )
        }
    
    ds = ds.map(preprocess)
    
    # create PoT quantization modifier
    pot_modifier = PoTQuantizationModifier(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    type="int",
                    symmetric=True,
                    strategy="channel",  # per-channel for better accuracy
                    observer="pot",
                )
            )
        },
        ignore=["lm_head"],
        pot_bits=4,
    )
    
    print("Applying PoT quantization...")
    
    # apply quantization
    oneshot(
        model=model,
        dataset=ds,
        recipe=pot_modifier,
        max_seq_length=512,
        num_calibration_samples=32,
        output_dir="./llama-pot-quantized",
    )
    
    print("✅ PoT quantization completed!")
    
    # test generation
    print("Testing generation...")
    dispatch_for_generation(model)
    input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=50, do_sample=False)
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")


def test_llama_apot_quantization():
    """Test APoT quantization on LLaMA model."""
    print("\nTesting APoT quantization on LLaMA...")
    
    # use a small LLaMA model for testing
    MODEL_ID = "microsoft/DialoGPT-medium"
    
    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # load calibration dataset
    print("Loading calibration dataset...")
    DATASET_ID = "mit-han-lab/pile-val-backup"
    ds = load_dataset(DATASET_ID, split="validation[:32]")
    ds = ds.shuffle(seed=42)
    
    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                [{"role": "user", "content": example["text"]}],
                tokenize=False,
            )
        }
    
    ds = ds.map(preprocess)
    
    # create APoT quantization modifier
    apot_modifier = APoTQuantizationModifier(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=4,
                    type="int",
                    symmetric=True,
                    strategy="channel",  # per-channel for better accuracy
                    observer="apot",
                    observer_kwargs={"num_terms": 2},
                )
            )
        },
        ignore=["lm_head"],
        apot_bits=4,
        num_terms=2,
    )
    
    print("Applying APoT quantization...")
    
    # apply quantization
    oneshot(
        model=model,
        dataset=ds,
        recipe=apot_modifier,
        max_seq_length=512,
        num_calibration_samples=32,
        output_dir="./llama-apot-quantized",
    )
    
    print("✅ APoT quantization completed!")
    
    # test generation
    print("Testing generation...")
    dispatch_for_generation(model)
    input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=50, do_sample=False)
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")


if __name__ == "__main__":
    import torch
    
    print("="*60)
    print("LLAMA PoT/APoT QUANTIZATION TEST")
    print("="*60)
    
    try:
        test_llama_pot_quantization()
    except Exception as e:
        print(f"❌ LLaMA PoT test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_llama_apot_quantization()
    except Exception as e:
        print(f"❌ LLaMA APoT test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 LLaMA quantization tests completed!")
    print("\nTo use with your own models:")
    print("1. For PoT: Use PoTQuantizationModifier with config_groups")
    print("2. For APoT: Use APoTQuantizationModifier with observer_kwargs={'num_terms': 2}")
    print("3. Set output_dir to save the quantized model")
    print("4. Use appropriate calibration data for your domain")