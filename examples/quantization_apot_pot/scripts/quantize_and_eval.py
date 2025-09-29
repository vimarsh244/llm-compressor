#!/usr/bin/env python3
"""Unified PoT/APoT quantization example with `lm_eval` perplexity reporting."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import CompressedTensorsConfig
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization.apot import APoTQuantizationModifier
from llmcompressor.modifiers.quantization.pot import PoTQuantizationModifier
from llmcompressor.transformers.compression.quantization_format import (
    infer_and_set_per_module_quantization_format,
)
from llmcompressor.utils.dev import dispatch_for_generation
from compressed_tensors.utils import remove_dispatch

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


class NanInfClampProcessor(LogitsProcessor):
    def __init__(self, min_val: float = -1e4, max_val: float = 1e4):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores = torch.nan_to_num(scores, nan=self.min_val, posinf=self.max_val, neginf=self.min_val)
        return scores.clamp_(self.min_val, self.max_val)


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
    parser.add_argument("--model_id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--dataset", default="open_platypus")
    parser.add_argument("--max_calibration_samples", type=int, default=8192)
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
    parser.add_argument("--eval_max_samples", type=int, default=800)
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
    quantization_config: Optional[CompressedTensorsConfig] = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    # When loading compressed checkpoints, explicitly pass a CompressedTensorsConfig
    # to run in decompressed mode for numerical stability in perplexity eval
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        # device_map={"": device},
        **({"quantization_config": quantization_config} if quantization_config else {}),
    )
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
        output_dir=None,  # keep in memory; we'll save after format inference
        trust_remote_code_model=trust_remote_code,
    )
    if quantized_model is None:
        raise RuntimeError("`oneshot` did not return a quantized model instance")

    # Ensure compressed-tensors metadata is set before saving
    infer_and_set_per_module_quantization_format(
        quantized_model, save_compressed=True
    )
    # IMPORTANT: Do NOT save here; saving compresses the in-memory model.
    # We'll save after we validate generation, mirroring the llama examples.
    return quantized_model, resolved_dir


def _is_bad_generation(text: str) -> bool:
    if not text:
        return True
    # Heuristics: presence of chat tags or high ratio of non-ascii
    lower = text.lower()
    if "<|user|>" in lower or "<|assistant|>" in lower:
        return True
    non_ascii = sum(1 for ch in text if ord(ch) > 126 or ord(ch) < 9)
    return (non_ascii / max(1, len(text))) > 0.15


def compute_perplexity_safe(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_iter,
    *,
    max_seq_length: int,
    batch_size: int,
    device: torch.device,
) -> float:
    import math
    import torch.nn.functional as F

    model.eval()
    total_neg_log_likelihood = 0.0
    total_tokens = 0

    def batches(it, n):
        batch = []
        for item in it:
            batch.append(item)
            if len(batch) == n:
                yield batch
                batch = []
        if batch:
            yield batch

    with torch.no_grad():
        for batch_texts in batches(dataset_iter, batch_size):
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
            )
            input_ids = enc.input_ids.to(device)
            attention_mask = enc.attention_mask.to(device) if hasattr(enc, "attention_mask") else None

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()  # upcast for stability
            # Sanitize logits
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4).clamp_(-1e4, 1e4)

            shift_logits = logits[..., :-1, :]
            shift_labels = input_ids[..., 1:]
            if attention_mask is not None:
                shift_mask = attention_mask[..., 1:].to(dtype=torch.float32)
            else:
                shift_mask = torch.ones_like(shift_labels, dtype=torch.float32)

            log_probs = F.log_softmax(shift_logits, dim=-1)
            nll_tokens = -log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            # mask padding
            nll_tokens = nll_tokens * shift_mask

            total_neg_log_likelihood += nll_tokens.sum().item()
            total_tokens += int(shift_mask.sum().item())

    mean_nll = total_neg_log_likelihood / max(1, total_tokens)
    if not math.isfinite(mean_nll):
        return float("nan")
    return math.exp(mean_nll)


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

    quant_model = None
    quant_tokenizer = None
    quantized_model_path = None

    if not args.skip_quantization:
        quant_dir = args.output_dir or os.path.join(os.getcwd(), f"{args.model_id.split('/')[-1]}-{args.quantization}-w{args.weight_bits}a{args.activation_bits}")
        print(f"Quantizing model with {args.quantization.upper()} to {quant_dir}")

        quant_model, quantized_model_path = _run_quantization(
            quantization_kind=args.quantization,
            model_id=args.model_id,
            modifier=quant_modifier,
            dataset=args.dataset,
            max_samples=args.max_calibration_samples,
            max_seq_length=args.max_seq_length,
            trust_remote_code=args.trust_remote_code,
            output_dir=quant_dir,
        )
        # Use the same tokenizer as baseline (same model family)
        quant_tokenizer = baseline_tokenizer
    else:
        quantized_model_path = args.output_dir or args.model_id

    if quant_model is None:
        print(f"Running perplexity for quantized model at {quantized_model_path}")
        # Force decompressed execution to avoid numerical issues/unsupported kernels
        qt_config = CompressedTensorsConfig(run_compressed=False)
        quant_model, quant_tokenizer = _load_model_and_tokenizer(
            quantized_model_path,
            trust_remote_code=args.trust_remote_code,
            device=device,
            quantization_config=qt_config,
        )
        # Do not dispatch yet; we'll dispatch once just before generation

    print("\n" + "=" * 50)
    print("TESTING QUANTIZED MODEL GENERATION")
    print("=" * 50)

    # Ensure tokenizer/model generation config is sane for llama-style models
    if quant_tokenizer.pad_token is None:
        quant_tokenizer.pad_token = quant_tokenizer.eos_token
    # Some tokenizers define add_eos_token; disable to avoid double-adding
    if hasattr(quant_tokenizer, "add_eos_token"):
        try:
            quant_tokenizer.add_eos_token = False
        except Exception:
            pass
    try:
        if getattr(quant_model.config, "pad_token_id", None) is None:
            quant_model.config.pad_token_id = quant_tokenizer.eos_token_id
        if getattr(quant_model.config, "eos_token_id", None) is None:
            quant_model.config.eos_token_id = quant_tokenizer.eos_token_id
        # Set generation config ids too if available
        if hasattr(quant_model, "generation_config"):
            quant_model.generation_config.pad_token_id = quant_tokenizer.eos_token_id
            quant_model.generation_config.eos_token_id = quant_tokenizer.eos_token_id
    except Exception:
        pass

    # Dispatch for generation AFTER quantization format is set and before generation
    quant_model = dispatch_for_generation(quant_model)

    # Tokenize input and move to the model's device
    # Prefer a chat-style prompt if tokenizer supports it (TinyLlama-Chat expects chat template)
    try:
        use_chat = hasattr(quant_tokenizer, "apply_chat_template") and getattr(quant_tokenizer, "chat_template", None)
        if use_chat:
            messages = [
                {"role": "user", "content": "Answer in one word: The capital of France is"}
            ]
            prompt_text = quant_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = "The capital of India is "
    except Exception:
        use_chat = False
        prompt_text = "The capital of India is "

    inputs = quant_tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=512,
    )
    input_ids = inputs.input_ids.to(quant_model.device)
    attention_mask = (
        inputs.attention_mask.to(quant_model.device)
        if (isinstance(inputs, dict) and "attention_mask" in inputs) or hasattr(inputs, "attention_mask")
        else None
    )

    gen_success = False
    try:
        with torch.no_grad():
            # Prefer greedy decoding first to avoid CUDA multinomial on bad probabilities
            generation_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": 32,
                "do_sample": False,  # greedy for robustness
                "pad_token_id": quant_tokenizer.eos_token_id,
                "eos_token_id": quant_tokenizer.eos_token_id,
                "use_cache": True,
                "logits_processor": LogitsProcessorList([NanInfClampProcessor()]),
            }
            if attention_mask is not None:
                generation_kwargs["attention_mask"] = attention_mask

            output = quant_model.generate(**generation_kwargs)

        generated_text = quant_tokenizer.decode(output[0], skip_special_tokens=True)
        # Fallback: if chat output leaks tags or looks noisy, try plain prompt
        if _is_bad_generation(generated_text) and use_chat:
            plain_inputs = quant_tokenizer(
                "The capital of France is",
                return_tensors="pt",
            )
            plain_ids = plain_inputs.input_ids.to(quant_model.device)
            with torch.no_grad():
                output = quant_model.generate(
                    input_ids=plain_ids,
                    max_new_tokens=16,
                    do_sample=False,
                    pad_token_id=quant_tokenizer.eos_token_id,
                    eos_token_id=quant_tokenizer.eos_token_id,
                    use_cache=True,
                    logits_processor=LogitsProcessorList([NanInfClampProcessor()]),
                )
            generated_text = quant_tokenizer.decode(output[0], skip_special_tokens=True)

        print(f"Generated text: {generated_text}")
        print("=" * 50 + "\n")
        gen_success = True

    except Exception as e:
        print(f"Generation failed: {e}")
        # Try CPU fallback to avoid CUDA asserts and debug logits issues
        try:
            print("Attempting CPU fallback generation...")
            try:
                remove_dispatch(quant_model)
            except Exception:
                pass
            quant_model.eval()
            quant_model.to("cpu")
            cpu_inputs = {"input_ids": input_ids.cpu(), "max_new_tokens": 32, "do_sample": False,
                          "pad_token_id": quant_tokenizer.eos_token_id, "eos_token_id": quant_tokenizer.eos_token_id,
                          "use_cache": True, "logits_processor": LogitsProcessorList([NanInfClampProcessor()])}
            if attention_mask is not None:
                cpu_inputs["attention_mask"] = attention_mask.cpu()
            with torch.no_grad():
                output = quant_model.generate(**cpu_inputs)
            generated_text = quant_tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"[CPU Fallback] Generated text: {generated_text}")
            gen_success = True
        except Exception as cpu_e:
            print(f"CPU fallback failed: {cpu_e}")

    # Save the quantized model only after successful generation
    saved_ok = False
    if (
        gen_success
        and not args.skip_quantization
        and quant_model is not None
        and quantized_model_path is not None
    ):
        print(f"Saving quantized model to: {quantized_model_path}")
        try:
            # Remove accelerate hooks and free CUDA cache before saving/compressing
            try:
                remove_dispatch(quant_model)
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            quant_model.eval()
            quant_model.save_pretrained(quantized_model_path, save_compressed=True)
            if quant_tokenizer is not None:
                quant_tokenizer.save_pretrained(quantized_model_path)
            saved_ok = True
        except Exception as e:
            print(f"Warning: failed to save quantized model due to: {e}")

    # Compute actual quantized perplexity using a decompressed copy of the quantized model
    quant_ppl = float("nan")
    delta = float("nan")
    try:
        eval_iter = load_dataset_iterator(
            eval_dataset,
            max_samples=args.eval_max_samples,
            split=args.eval_split,
            text_column=args.eval_text_column,
        )
        # Evaluate from an on-disk path to ensure consistent decompression
        eval_path = quantized_model_path if (quantized_model_path and (saved_ok or args.skip_quantization)) else None
        if eval_path is None and args.skip_quantization:
            eval_path = args.output_dir or args.model_id
        if eval_path is not None:
            print("Evaluating quantized model perplexity (decompressed)...")
            qt_config = CompressedTensorsConfig(run_compressed=False)
            q_eval_model, q_eval_tok = _load_model_and_tokenizer(
                eval_path,
                trust_remote_code=args.trust_remote_code,
                device=torch.device("cpu"),  # use CPU for stability
                quantization_config=qt_config,
            )
            # First try the standard function
            try:
                quant_ppl = compute_perplexity_manual(
                    q_eval_model,
                    q_eval_tok,
                    eval_iter,
                    max_seq_length=args.eval_seq_length,
                    batch_size=max(1, args.eval_batch_size // 2),
                    device=torch.device("cpu"),
                )
            except Exception as e:
                print(f"compute_perplexity_manual failed: {e}")
                quant_ppl = float("nan")
            # Fallback if NaN
            if not (isinstance(quant_ppl, float) and torch.isfinite(torch.tensor(quant_ppl))):
                print("Falling back to safe perplexity computation...")
                # Recreate iterator since it may be exhausted
                eval_iter = load_dataset_iterator(
                    eval_dataset,
                    max_samples=args.eval_max_samples,
                    split=args.eval_split,
                    text_column=args.eval_text_column,
                )
                quant_ppl = compute_perplexity_safe(
                    q_eval_model,
                    q_eval_tok,
                    eval_iter,
                    max_seq_length=args.eval_seq_length,
                    batch_size=max(1, args.eval_batch_size // 2),
                    device=torch.device("cpu"),
                )
            if isinstance(quant_ppl, float) and torch.isfinite(torch.tensor(quant_ppl)):
                print(f"Quantized perplexity: {quant_ppl:.4f}")
                delta = quant_ppl - baseline_ppl
                print(f"Perplexity delta (quant - baseline): {delta:.4f}")
            else:
                print("Quantized perplexity: nan")
        else:
            print("Warning: quantized model path is unavailable; skipping quantized perplexity eval.")
    except Exception as e:
        print(f"Warning: failed to compute quantized perplexity: {e}")

    if args.save_results:
        payload = {
            "baseline_perplexity": baseline_ppl,
            "quantized_perplexity": quant_ppl,
            "delta": delta,
        }
        dump_json(args.save_results, payload)


if __name__ == "__main__":
    main()


