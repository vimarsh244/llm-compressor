"""Utility to perform weight-only quantization without calibration data."""

import argparse
from pathlib import Path

from transformers import AutoTokenizer
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization.apot import APoTQuantizationModifier
from llmcompressor.modifiers.quantization.pot import PoTQuantizationModifier
from llmcompressor.transformers.compression.quantization_format import (
    infer_and_set_per_module_quantization_format,
)


def build_weight_only_modifier(kind: str, weight_bits: int, num_terms: int | None):
    weight_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=weight_bits,
            type="int",
            symmetric=True,
            strategy="channel",
        ),
    )

    if kind == "apot":
        return APoTQuantizationModifier(
            config_groups={"group_0": weight_scheme},
            apot_bits=weight_bits,
            num_terms=num_terms or 2,
            ignore=["lm_head"],
        )
    if kind == "pot":
        return PoTQuantizationModifier(
            config_groups={"group_0": weight_scheme},
            pot_bits=weight_bits,
            ignore=["lm_head"],
        )
    raise ValueError(f"Unsupported quantization kind: {kind}")


def quantize_weight_only(
    model_id: str,
    *,
    output_dir: str,
    kind: str = "pot",
    weight_bits: int = 4,
    apot_terms: int | None = None,
):
    modifier = build_weight_only_modifier(kind, weight_bits, apot_terms)
    quantized_model = oneshot(
        model=model_id,
        recipe=modifier,
        dataset=None,
        pipeline="datafree",
        output_dir=output_dir,
        clear_sparse_session=True,
        quantization_aware_calibration=False,
    )

    infer_and_set_per_module_quantization_format(
        quantized_model, save_compressed=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    quantized_model.save_pretrained(output_dir, save_compressed=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Weight-only quantization helper")
    parser.add_argument("model", help="Model id or path")
    parser.add_argument("output", help="Directory to store the quantized model")
    parser.add_argument("--kind", choices=["pot", "apot"], default="pot")
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--apot-terms", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    quantize_weight_only(
        args.model,
        output_dir=args.output,
        kind=args.kind,
        weight_bits=args.weight_bits,
        apot_terms=args.apot_terms if args.kind == "apot" else None,
    )


if __name__ == "__main__":
    main()

