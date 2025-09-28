"""End-to-end PoT quantization example with calibration data."""

from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization.pot import PoTQuantizationModifier
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map
from llmcompressor.transformers.compression.quantization_format import (
    infer_and_set_per_module_quantization_format,
)
from llmcompressor.utils.dev import dispatch_for_generation
from llmcompressor.transformers.finetune.data.open_platypus import OpenPlatypusDataset


def _build_calibration_dataset(dataset_id, dataset_split, num_samples):
    raw = load_dataset(dataset_id, split=f"{dataset_split}[:{num_samples}]")
    raw = raw.shuffle(seed=42)

    template = OpenPlatypusDataset.ALPACA_TEMPLATE

    def to_text(example):
        instruction = example.get("instruction", example.get("text", ""))
        if example.get("input"):
            prompt = template["prompt_input"].format(
                instruction=instruction,
                input=example.get("input", ""),
            )
        else:
            prompt = template["prompt_no_input"].format(instruction=instruction)

        text = prompt
        if example.get("output"):
            text += example["output"]
        return {"text": text}

    processed = raw.map(to_text, remove_columns=raw.column_names)
    return processed


def build_pot_modifier(weight_bits: int, activation_bits: int) -> PoTQuantizationModifier:
    return PoTQuantizationModifier(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=weight_bits,
                    type="int",
                    symmetric=True,
                    strategy="channel",
                ),
                input_activations=QuantizationArgs(
                    num_bits=activation_bits,
                    type="int",
                    symmetric=True,
                    strategy="tensor",
                ),
                output_activations=QuantizationArgs(
                    num_bits=activation_bits,
                    type="int",
                    symmetric=True,
                    strategy="tensor",
                ),
            )
        },
        ignore=["lm_head"],
        pot_bits=weight_bits,
    )


def main():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dataset_id = "mit-han-lab/pile-val-backup"
    dataset_split = "validation"
    num_calibration_samples = 2048
    max_seq_length = 2048
    save_dir = Path(model_id.split("/")[-1] + "-pot-w4a8")

    print(f"Calculating device map for {model_id}")
    device_map = calculate_offload_device_map(
        model_id,
        reserve_for_hessians=True,
        num_gpus=torch.cuda.device_count(),
        trust_remote_code=True,
    )

    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Preparing calibration dataset from {dataset_id}")
    calibration_dataset = _build_calibration_dataset(
        dataset_id, dataset_split, num_calibration_samples
    )

    weight_bits = 4
    activation_bits = 8
    pot_modifier = build_pot_modifier(weight_bits, activation_bits)

    print("Running oneshot calibration with PoT modifier")
    quantized_model = oneshot(
        model=model,
        dataset=calibration_dataset,
        recipe=pot_modifier,
        output_dir=None,
        max_seq_length=max_seq_length,
        num_calibration_samples=min(len(calibration_dataset), num_calibration_samples),
        pad_to_max_length=False,
    )

    print("Inferring quantization formats for compressed-tensors compatibility")
    infer_and_set_per_module_quantization_format(
        quantized_model, save_compressed=True
    )

    print("\n" + "=" * 50)
    print("TESTING QUANTIZED MODEL GENERATION")
    print("=" * 50)
    dispatch_for_generation(quantized_model)

    input_ids = tokenizer(
        "The capital of France is ",
        return_tensors="pt",
    ).input_ids.to(quantized_model.device)

    output = quantized_model.generate(
        input_ids,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")
    print("=" * 50 + "\n")

    print(f"Saving quantized model to: {save_dir}")
    quantized_model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)

    print("Model saved successfully!")
    print("\nTo load the quantized model later:")
    print(f"model = AutoModelForCausalLM.from_pretrained('{save_dir}')")
    print(f"tokenizer = AutoTokenizer.from_pretrained('{save_dir}')")


if __name__ == "__main__":
    main()