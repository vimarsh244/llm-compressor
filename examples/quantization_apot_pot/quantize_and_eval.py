#!/usr/bin/env python3
"""Unified PoT/APoT quantization example with `lm_eval` perplexity reporting."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization.apot import APoTQuantizationModifier
from llmcompressor.modifiers.quantization.pot import PoTQuantizationModifier
from llmcompressor.transformers.compression.quantization_format import (
    infer_and_set_per_module_quantization_format,
)

from lm_eval_utils import (
    compute_perplexity_manual,
    dump_json,
    load_dataset_iterator,
)


@dataclass
class QuantizationConfig:
    weight_bits: int
    activation_bits: int
    num_terms: Optional[int] = None


def _build_pot_modifier(config: QuantizationConfig) -> PoTQuantizationModifier:
    return PoTQuantizationModifier(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=config.weight_bits,
                    type="int",
                    symmetric=True,
                    strategy="channel",
                ),
                input_activations=QuantizationArgs(
                    num_bits=config.activation_bits,
                    type="int",
                    symmetric=True,
                    strategy="tensor",
                ),
                output_activations=QuantizationArgs(
                    num_bits=config.activation_bits,
                    type="int",
                    symmetric=True,
                    strategy="tensor",
                ),
            )
        },
        ignore=["lm_head"],
        pot_bits=config.weight_bits,
    )


def _build_apot_modifier(config: QuantizationConfig) -> APoTQuantizationModifier:
    if config.num_terms is None:
        raise ValueError("APoT requires `num_terms` to be specified")
    return APoTQuantizationModifier(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=config.weight_bits,
                    type="int",
                    symmetric=True,
                    strategy="channel",
                    observer_kwargs={"num_terms": config.num_terms},
                ),
                input_activations=QuantizationArgs(
                    num_bits=config.activation_bits,
                    type="int",
                    symmetric=True,
                    strategy="tensor",
                    observer_kwargs={"num_terms": config.num_terms},
                ),
                output_activations=QuantizationArgs(
                    num_bits=config.activation_bits,
                    type="int",
                    symmetric=True,
                    strategy="tensor",
                    observer_kwargs={"num_terms": config.num_terms},
                ),
            )
        },
        ignore=["lm_head"],
        apot_bits=config.weight_bits,
        num_terms=config.num_terms,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize LLaMA models and evaluate perplexity manually"
    )
    parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset", default="open_platypus")
    parser.add_argument("--max_calibration_samples", type=int, default=512)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--quantization", choices=["pot", "apot"], default="pot")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--activation_bits", type=int, default=8)
    parser.add_argument("--weight_bits", type=int, default=4)
    parser.add_argument("--apot_terms", type=int, default=2)
    parser.add_argument("--save_results", default=None)
    parser.add_argument("--skip_quantization", action="store_true")
    parser.add_argument(
        "--eval_dataset",
        default="lambada_openai",
        help="Dataset used for perplexity evaluation when none is specified",
    )
    parser.add_argument("--eval_split", default="validation")
    parser.add_argument("--eval_text_column", default=None)
    parser.add_argument("--eval_max_samples", type=int, default=200)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_seq_length", type=int, default=1024)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _resolve_device(user_device: Optional[str]) -> torch.device:
    if user_device:
        return torch.device(user_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_model_and_tokenizer(
    model_id: str,
    *,
    trust_remote_code: bool,
    device: torch.device,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map=None,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def _run_quantization(
    *,
    quantization_kind: str,
    model_id: str,
    modifier,
    dataset: str,
    max_samples: int,
    max_seq_length: int,
    trust_remote_code: bool,
    output_dir: Optional[str],
):
    resolved_dir = output_dir or os.path.join(
        os.getcwd(), f"{model_id.split('/')[-1]}-{quantization_kind}"
    )

    quantized_model = oneshot(
        model=model_id,
        recipe=modifier,
        dataset=dataset,
        max_seq_length=max_seq_length,
        num_calibration_samples=max_samples,
        output_dir=resolved_dir,
        trust_remote_code_model=trust_remote_code,
    )
    if quantized_model is None:
        raise RuntimeError("`oneshot` did not return a quantized model instance")

    infer_and_set_per_module_quantization_format(
        quantized_model, save_compressed=True
    )
    quantized_model.save_pretrained(resolved_dir, save_compressed=True)


def main():
    args = _parse_args()

    quant_config = QuantizationConfig(
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        num_terms=args.apot_terms if args.quantization == "apot" else None,
    )

    quant_modifier = (
        _build_apot_modifier(quant_config)
        if args.quantization == "apot"
        else _build_pot_modifier(quant_config)
    )

    eval_dataset = args.eval_dataset or args.dataset
    device = _resolve_device(args.device)

    print(f"Running baseline perplexity for {args.model_id} on {eval_dataset}")
    baseline_model, baseline_tokenizer = _load_model_and_tokenizer(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        device=device,
    )
    baseline_iter = load_dataset_iterator(
        eval_dataset,
        max_samples=args.eval_max_samples,
        split=args.eval_split,
        text_column=args.eval_text_column,
    )
    baseline_ppl = compute_perplexity_manual(
        baseline_model,
        baseline_tokenizer,
        baseline_iter,
        max_seq_length=args.eval_seq_length,
        batch_size=args.eval_batch_size,
        device=device,
    )
    print(f"Baseline perplexity: {baseline_ppl:.4f}")

    if not args.skip_quantization:
        quant_dir = args.output_dir or os.path.join(os.getcwd(), f"{args.model_id.split('/')[-1]}-{args.quantization}-w{args.weight_bits}a{args.activation_bits}")
        print(f"Quantizing model with {args.quantization.upper()} to {quant_dir}")

        _run_quantization(
            quantization_kind=args.quantization,
            model_id=args.model_id,
            modifier=quant_modifier,
            dataset=args.dataset,
            max_samples=args.max_calibration_samples,
            max_seq_length=args.max_seq_length,
            trust_remote_code=args.trust_remote_code,
            output_dir=quant_dir,
        )

        quantized_model_path = quant_dir
    else:
        quantized_model_path = args.output_dir or args.model_id

    print(f"Running perplexity for quantized model at {quantized_model_path}")
    quant_model, quant_tokenizer = _load_model_and_tokenizer(
        quantized_model_path,
        trust_remote_code=args.trust_remote_code,
        device=device,
    )
    quant_iter = load_dataset_iterator(
        eval_dataset,
        max_samples=args.eval_max_samples,
        split=args.eval_split,
        text_column=args.eval_text_column,
    )
    quant_ppl = compute_perplexity_manual(
        quant_model,
        quant_tokenizer,
        quant_iter,
        max_seq_length=args.eval_seq_length,
        batch_size=args.eval_batch_size,
        device=device,
    )
    print(f"Quantized perplexity: {quant_ppl:.4f}")
    delta = quant_ppl - baseline_ppl
    print(f"Perplexity delta (quant - baseline): {delta:.4f}")

    if args.save_results:
        payload = {
            "baseline_perplexity": baseline_ppl,
            "quantized_perplexity": quant_ppl,
            "delta": delta,
        }
        dump_json(args.save_results, payload)


if __name__ == "__main__":
    main()


