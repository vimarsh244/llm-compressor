#!/usr/bin/env python3
"""Unified PoT/APoT quantization example with `lm_eval` perplexity reporting."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization.apot import APoTQuantizationModifier
from llmcompressor.modifiers.quantization.pot import PoTQuantizationModifier
from llmcompressor.transformers.compression.quantization_format import (
    infer_and_set_per_module_quantization_format,
)

from lm_eval_utils import (
    collect_perplexity_metrics,
    compute_perplexity_delta,
    dump_json,
    run_lm_eval,
    summarize_perplexity,
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
    parser = argparse.ArgumentParser(description="Quantize LLaMA models and evaluate perplexity with lm_eval")
    parser.add_argument("--model_id", default="TinyLlama/TinyLlama_v1.1")
    parser.add_argument("--dataset", default="open_platypus")
    parser.add_argument("--max_calibration_samples", type=int, default=512)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--quantization", choices=["pot", "apot"], default="pot")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--tasks", default="lambada_openai")
    parser.add_argument("--batch_size", default="2")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", default="200")
    parser.add_argument("--device", default=None)
    parser.add_argument("--use_accelerate", action="store_true")
    parser.add_argument(
        "--extra_model_args",
        default=None,
        help="Comma separated key=value items passed through to lm_eval",
    )
    parser.add_argument(
        "--lm_eval_run_compressed",
        action="store_true",
        help="If set, run lm_eval against the compressed model without decompressing",
    )
    parser.add_argument("--activation_bits", type=int, default=8)
    parser.add_argument("--weight_bits", type=int, default=4)
    parser.add_argument("--apot_terms", type=int, default=2)
    parser.add_argument("--save_results", default=None)
    parser.add_argument("--skip_quantization", action="store_true")
    return parser.parse_args()


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

    print(f"Running baseline perplexity for {args.model_id} on tasks {args.tasks}")
    baseline_results = run_lm_eval(
        pretrained=args.model_id,
        tasks=args.tasks,
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        device=args.device,
        use_accelerate=args.use_accelerate,
        extra_model_args=args.extra_model_args,
    )
    baseline_metrics = collect_perplexity_metrics(baseline_results)

    print(summarize_perplexity("Baseline perplexity", baseline_metrics))

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
    extra_model_args = args.extra_model_args
    if not args.lm_eval_run_compressed:
        token = "quantization_config.run_compressed=False"
        if extra_model_args:
            if isinstance(extra_model_args, str):
                extra_model_args = f"{extra_model_args},{token}"
            else:
                extra_model_args = f"{extra_model_args},{token}"
        else:
            extra_model_args = token
    try:
        quant_results = run_lm_eval(
            pretrained=quantized_model_path,
            tasks=args.tasks,
            batch_size=args.batch_size,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            device=args.device,
            use_accelerate=args.use_accelerate,
            extra_model_args=extra_model_args,
        )
    except Exception as exc:
        print(f"Quantized model evaluation failed: {exc}")
        raise
    quant_metrics = collect_perplexity_metrics(quant_results)

    delta = compute_perplexity_delta(baseline_metrics, quant_metrics)

    print(summarize_perplexity("Quantized perplexity", quant_metrics))
    print(summarize_perplexity("Delta (quant - base)", delta))

    if args.save_results:
        payload = {
            "baseline": baseline_results,
            "quantized": quant_results,
            "perplexity_summary": {
                "baseline": baseline_metrics,
                "quantized": quant_metrics,
                "delta": delta,
            },
        }
        dump_json(args.save_results, payload)


if __name__ == "__main__":
    main()


