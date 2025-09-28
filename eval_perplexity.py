#!/usr/bin/env python3
import argparse
import math
import os
from typing import Iterable, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Compute perplexity for a CausalLM model on common datasets")
    parser.add_argument("--model_path", type=str, required=True, help="Path or HF repo id of the model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="lambada_openai",
        help="Dataset identifier: lambada_openai | pile_val | <hf_dataset_id>",
    )
    parser.add_argument("--split", type=str, default=None, help="Dataset split to use (auto if not provided)")
    parser.add_argument("--text_column", type=str, default=None, help="Text column name (auto-detect if None)")
    parser.add_argument("--max_samples", type=int, default=5000, help="Limit examples for quick evaluation (0 = all)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--seq_length", type=int, default=1024, help="Sequence length for tokenization")
    parser.add_argument("--stride", type=int, default=0, help="Optional sliding window stride (0 = no sliding)")
    parser.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code to HF loaders")
    parser.add_argument("--bf16", action="store_true", help="Prefer bfloat16 if supported")
    parser.add_argument("--device", type=str, default=None, help="Device like cuda:0 or cpu (default auto)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (unused for simplicity)")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming for HF datasets")
    return parser.parse_args()


def _resolve_device(user_device: Optional[str]) -> torch.device:
    if user_device:
        return torch.device(user_device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _load_model_and_tokenizer(model_path: str, trust_remote_code: bool, prefer_bf16: bool, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = (
        torch.bfloat16
        if prefer_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16 if torch.cuda.is_available() else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map=None,  # force single device to avoid device mismatch during loss computation
    )
    model.to(device)
    model.eval()

    # generation config ids (not strictly required for loss eval, but useful)
    if getattr(model, "generation_config", None) is not None:
        if model.generation_config.pad_token_id is None and tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        if model.generation_config.eos_token_id is None and tokenizer.eos_token_id is not None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id

    # sanity check vocab/embeddings alignment
    try:
        vocab_model = model.get_input_embeddings().weight.size(0)
        vocab_tok = len(tokenizer)
        if vocab_model != vocab_tok:
            print(f"! tokenizer vocab ({vocab_tok}) != model embeddings ({vocab_model}). Ensure matching artifacts.")
    except Exception:
        pass

    return model, tokenizer


def _iter_texts(args) -> Iterable[str]:
    name = args.dataset.lower()
    if name in ("lambada", "lambada_openai", "lambada_plain"):
        # Prefer the OPENAI split if available, otherwise fall back to plain_text config
        split = args.split or "validation"
        ds = None
        try:
            ds = load_dataset("lambada", "openai", split=split, streaming=args.streaming)
        except Exception:
            pass
        if ds is None:
            # try same split without config (plain_text)
            try:
                ds = load_dataset("lambada", split=split, streaming=args.streaming)
            except Exception:
                # final fallback to test split
                ds = load_dataset("lambada", split="test", streaming=args.streaming)
        text_col = args.text_column or "text"
        for i, ex in enumerate(ds):
            if args.max_samples and i >= args.max_samples:
                break
            yield ex[text_col]
    elif name in ("pile_val", "the_pile_val", "pile-validation"):
        # Stream The Pile validation directly from the-eye
        # Ref: HF LLM course shows streaming Pile splits
        # https://huggingface.co/learn/llm-course/en/chapter5/4
        url = "https://the-eye.eu/public/AI/pile/val.jsonl.zst"
        ds = load_dataset("json", data_files=url, split="train", streaming=True)
        text_col = args.text_column or "text"
        for i, ex in enumerate(ds):
            if args.max_samples and i >= args.max_samples:
                break
            yield ex[text_col]
    else:
        # Generic HF dataset id
        # Use provided split or default to validation/test/train
        split = args.split
        if split is None:
            for cand in ("validation", "test", "train"):
                try:
                    ds = load_dataset(name, split=cand, streaming=args.streaming)
                    split = cand
                    break
                except Exception:
                    continue
            if split is None:
                raise ValueError("Could not infer split; please provide --split explicitly")
        else:
            ds = load_dataset(name, split=split, streaming=args.streaming)
        # heuristically pick text column
        if args.text_column:
            text_col = args.text_column
        else:
            cols = getattr(ds, "column_names", None) or []
            text_col = "text" if "text" in cols else cols[0]
        for i, ex in enumerate(ds):
            if args.max_samples and i >= args.max_samples:
                break
            yield ex[text_col]


def _chunk_tokens(input_ids: torch.Tensor, max_length: int, stride: int) -> Iterable[torch.Tensor]:
    # Optionally split long sequences into windows
    if input_ids.size(1) <= max_length or max_length <= 0:
        yield input_ids
        return
    if stride and stride > 0:
        start = 0
        while start < input_ids.size(1):
            end = min(start + max_length, input_ids.size(1))
            yield input_ids[:, start:end]
            if end == input_ids.size(1):
                break
            start = end - stride
            if start < 0:
                start = 0
    else:
        # simple head truncation
        yield input_ids[:, : max_length]


def compute_perplexity(args) -> float:
    device = _resolve_device(args.device)
    model, tokenizer = _load_model_and_tokenizer(args.model_path, args.trust_remote_code, args.bf16, device)

    # debug: test model with a simple forward pass first
    print("Testing model with simple input...")
    try:
        test_input = tokenizer("Hello world", return_tensors="pt").to(device)
        with torch.no_grad():
            test_output = model(**test_input, labels=test_input["input_ids"])
            test_loss = test_output.loss
            print(f"Test forward pass loss: {test_loss.item()}")
            if torch.isnan(test_loss) or torch.isinf(test_loss):
                print("! Model produces NaN/Inf on simple input - quantization may be broken")
                return float('nan')
    except Exception as e:
        print(f"! Model test failed: {e}")
        return float('nan')

    total_neg_log_likelihood = 0.0
    total_tokens = 0
    valid_batches = 0
    nan_batches = 0

    texts_iter = _iter_texts(args)

    batch = []
    for text in texts_iter:
        batch.append(text)
        if len(batch) < args.batch_size:
            continue

        with torch.no_grad():
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=args.seq_length)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # labels equal to input_ids; mask out pads with -100
            labels = input_ids.clone()
            if attention_mask is not None:
                labels[attention_mask == 0] = -100

            if args.stride and args.stride > 0:
                # windowed evaluation
                for i in range(input_ids.size(0)):
                    for window in _chunk_tokens(input_ids[i : i + 1], args.seq_length, args.stride):
                        lab = window.clone()
                        # no pad in windows by construction; still safe
                        outputs = model(input_ids=window, labels=lab)
                        loss = outputs.loss  # mean over tokens
                        # approximate token count as window length - 1 (next token prediction)
                        n_tok = int(window.numel()) - 1
                        total_neg_log_likelihood += float(loss.item()) * n_tok
                        total_tokens += n_tok
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss  # mean over non-ignored tokens
                
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_batches += 1
                    print(f"! NaN/Inf loss in batch, skipping...")
                    batch = []
                    continue
                    
                # estimate number of predicted tokens
                num_pred_tokens = int((labels != -100).sum().item())
                total_neg_log_likelihood += float(loss.item()) * num_pred_tokens
                total_tokens += num_pred_tokens
                valid_batches += 1

        batch = []

    # flush remainder
    if batch:
        with torch.no_grad():
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=args.seq_length)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = input_ids.clone()
            if attention_mask is not None:
                labels[attention_mask == 0] = -100
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                num_pred_tokens = int((labels != -100).sum().item())
                total_neg_log_likelihood += float(loss.item()) * num_pred_tokens
                total_tokens += num_pred_tokens
                valid_batches += 1
            else:
                nan_batches += 1

    print(f"Evaluation summary: {valid_batches} valid batches, {nan_batches} NaN/Inf batches")
    
    if total_tokens == 0:
        print("No valid tokens were evaluated; all batches had NaN/Inf losses")
        return float('nan')

    avg_nll = total_neg_log_likelihood / total_tokens
    ppl = math.exp(avg_nll)
    return ppl


if __name__ == "__main__":
    args = parse_args()
    ppl = compute_perplexity(args)
    print(f"Perplexity on {args.dataset}: {ppl:.4f}")