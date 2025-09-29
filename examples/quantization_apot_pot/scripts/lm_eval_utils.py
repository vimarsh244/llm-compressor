#!/usr/bin/env python3
"""Shared helpers for running manual perplexity evaluation on quantized models."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def tokenize_batch(
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    *,
    device: torch.device,
    max_seq_length: int,
) -> Dict[str, torch.Tensor]:
    batch = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    labels = input_ids.clone()
    if attention_mask is not None:
        labels[attention_mask == 0] = -100
    batch["labels"] = labels
    return batch


def accumulate_nll(
    model: AutoModelForCausalLM,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    # Validate vocabulary alignment to fail fast on tokenizer/model mismatch
    labels = batch.get("labels")
    if labels is not None:
        valid = labels[labels >= 0]
        if valid.numel() > 0:
            max_label = int(valid.max().item())
            vocab_size = model.get_output_embeddings().weight.size(0)
            if max_label >= vocab_size:
                raise ValueError(
                    f"Found token id {max_label} >= vocab size {vocab_size}. "
                    "Ensure tokenizer and model vocabularies are aligned."
                )

    with torch.no_grad():
        # Avoid relying on model.loss; do a forward pass and compute CE manually
        model_inputs = {k: v for k, v in batch.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits  # (B, T, V)

    if logits is None:
        return {"nll": float("nan"), "tokens": 0}

    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")

    # Shift to compute next-token prediction: p(x_t | x_{<t})
    # Use float32 for stability
    logits = logits[:, :-1, :].float()  # (B, T-1, V)
    targets = input_ids[:, 1:]  # (B, T-1)

    if attention_mask is not None:
        token_mask = attention_mask[:, 1:].to(torch.bool)  # (B, T-1)
    else:
        token_mask = torch.ones_like(targets, dtype=torch.bool, device=targets.device)

    # Compute log-probs and gather the ones for targets
    log_probs = F.log_softmax(logits, dim=-1)
    # Flatten for efficient gather
    B, Tm1, V = log_probs.shape
    log_probs_flat = log_probs.view(B * Tm1, V)
    targets_flat = targets.contiguous().view(B * Tm1)

    # Mask out positions corresponding to padding
    token_mask_flat = token_mask.view(B * Tm1)
    if token_mask_flat.sum() == 0:
        return {"nll": 0.0, "tokens": 0}

    selected_log_probs = log_probs_flat[token_mask_flat].gather(
        dim=-1, index=targets_flat[token_mask_flat].unsqueeze(-1)
    ).squeeze(-1)  # (N_valid,)

    if torch.isnan(selected_log_probs).any() or torch.isinf(selected_log_probs).any():
        # Skip this batch if numerically unstable
        return {"nll": float("nan"), "tokens": 0}

    nll = -selected_log_probs.sum().item()
    tokens = int(token_mask_flat.sum().item())
    return {"nll": float(nll), "tokens": tokens}


def compute_perplexity_manual(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_iter: Iterable[str],
    *,
    max_seq_length: int,
    batch_size: int,
    device: torch.device,
) -> float:
    # Don't move model to device - it may already be dispatched with accelerate hooks
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    pending: List[str] = []

    def _flush() -> None:
        nonlocal total_nll, total_tokens, pending
        if not pending:
            return
        batch = tokenize_batch(
            tokenizer,
            pending,
            device=next(model.parameters()).device,
            max_seq_length=max_seq_length,
        )
        stats = accumulate_nll(model, batch)
        if math.isnan(stats["nll"]):
            pending = []
            return
        total_nll += stats["nll"]
        total_tokens += stats["tokens"]
        pending = []

    for text in dataset_iter:
        pending.append(text)
        if len(pending) >= batch_size:
            _flush()

    _flush()

    if total_tokens == 0:
        return float("nan")
    avg_nll = total_nll / total_tokens
    return float(math.exp(avg_nll))


def load_dataset_iterator(
    dataset: str,
    *,
    max_samples: int,
    split: Optional[str] = None,
    text_column: Optional[str] = None,
) -> Iterable[str]:
    dataset_lower = dataset.lower()
    resolved_split = split or "validation"

    if dataset_lower in ("lambada", "lambada_openai", "lambada_plain"):
        try:
            ds = load_dataset("lambada", "openai", split=resolved_split)
        except Exception:
            ds = load_dataset("lambada", split=resolved_split)
        column = text_column or "text"
    elif dataset_lower in ("pile_val", "the_pile_val", "pile-validation"):
        ds = load_dataset("json", data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst", split="train")
        column = text_column or "text"
    else:
        ds = load_dataset(dataset, split=resolved_split)
        if text_column:
            column = text_column
        else:
            cols = getattr(ds, "column_names", None) or []
            column = "text" if "text" in cols else cols[0]

    for idx, row in enumerate(ds):
        if max_samples and max_samples > 0 and idx >= max_samples:
            break
        yield row[column]


def dump_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


