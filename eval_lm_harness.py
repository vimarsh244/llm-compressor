#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import List, Optional


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a HF model with EleutherAI LM Evaluation Harness")
    parser.add_argument("--pretrained", type=str, required=True, help="HF repo id or local path to model")
    parser.add_argument("--tasks", type=str, default="lambada_openai", help="Comma-separated task list (e.g., lambada_openai,wikitext)")
    parser.add_argument("--batch_size", type=str, default="2", help="Batch size per the harness API (string or int)")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--limit", type=str, default=None, help="Limit number of examples per task (e.g., 100, or null)")
    parser.add_argument("--device", type=str, default=None, help="Device string for model_args (e.g., cuda:0)")
    parser.add_argument("--use_accelerate", action="store_true", help="Pass use_accelerate=True to model_args")
    parser.add_argument("--output_path", type=str, default=None, help="Where to write results JSON")
    parser.add_argument("--model_kind", type=str, default="hf-causal-experimental", help="Harness model adapter (hf-causal or hf-causal-experimental)")
    parser.add_argument("--extra_model_args", type=str, default=None, help="Additional model_args key=value pairs,comma-separated")
    return parser.parse_args()


def build_model_args(pretrained: str, device: Optional[str], use_accelerate: bool, extra: Optional[str]) -> str:
    parts: List[str] = [f"pretrained={pretrained}"]
    if device:
        parts.append(f"device={device}")
    if use_accelerate:
        parts.append("use_accelerate=True")
    if extra:
        # expect comma-separated key=value
        for kv in extra.split(","):
            kv = kv.strip()
            if kv:
                parts.append(kv)
    return ",".join(parts)


def main():
    args = parse_args()

    try:
        from lm_eval import evaluator
    except Exception as e:
        print(
            "lm-eval is not installed. Install with: pip install -U lm-eval",
            file=sys.stderr,
        )
        raise

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    model_args = build_model_args(args.pretrained, args.device, args.use_accelerate, args.extra_model_args)

    print(f"Running lm-eval: model={args.model_kind} tasks={tasks} model_args={model_args}")

    results = evaluator.simple_evaluate(
        model=args.model_kind,
        model_args=model_args,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    # pretty print summary
    print(json.dumps(results, indent=2, sort_keys=True))

    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    main()