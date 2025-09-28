#!/usr/bin/env python3
"""Shared helpers for running manual perplexity evaluation on quantized models."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.utils.dev import dispatch_for_generation


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
    with torch.no_grad():
        outputs = model(**batch)
        loss = outputs.loss
    if torch.isnan(loss) or torch.isinf(loss):
        return {"nll": float("nan"), "tokens": 0}
    tokens = int((batch["labels"] != -100).sum().item())
    return {"nll": float(loss.item()) * tokens, "tokens": tokens}


def compute_perplexity_manual(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_iter: Iterable[str],
    *,
    max_seq_length: int,
    batch_size: int,
    device: torch.device,
) -> float:
    dispatch_for_generation(model)
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
            device=device,
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


