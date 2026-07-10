#!/usr/bin/env python3
"""Profile exact chat-prompt and reference-output token lengths per subtask."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from molbench.core.io import atomic_write_json  # noqa: E402
from molbench.core.registry import get_benchmark  # noqa: E402
import molbench.benchmarks  # noqa: E402,F401


DEFAULT_TOKENIZER = "/home/haoqian/Data/SAERAG/Open-Scopes/ChemDFM-Scope/ChemDFM-v2.0-14B"


def percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return round(ordered[lower] * (upper - position) + ordered[upper] * (position - lower))


def summary(values: list[int]) -> dict:
    return {
        "n": len(values),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "p99_9": percentile(values, 0.999),
        "max": max(values, default=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="chemcotbench-v2")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--task", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, default=Path("results/chemcotbench_token_profile.json"))
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    benchmark = get_benchmark(args.benchmark)
    tasks = benchmark.tasks()
    selected = args.task or list(tasks)
    result = {
        "benchmark": args.benchmark,
        "tokenizer": args.tokenizer,
        "tasks": {},
    }
    for task_name in selected:
        task = tasks[task_name]
        examples = benchmark.load_task(task_name, limit=args.limit)
        prompt_tokens = []
        output_tokens = []
        for example in examples:
            messages = [
                {
                    "role": "system",
                    "content": task.build_system_prompt(example)
                    or "You are a helpful assistant.",
                },
                {"role": "user", "content": task.build_prompt(example)},
            ]
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_tokens.append(
                len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            )
            reference = example.get("_process_reference", {}).get("raw_output")
            if reference:
                output_tokens.append(
                    len(tokenizer(reference, add_special_tokens=False)["input_ids"])
                )
        output_summary = summary(output_tokens)
        recommended = 0
        if output_tokens:
            recommended = 128 * math.ceil(1.25 * output_summary["p99_9"] / 128)
        result["tasks"][task_name] = {
            "prompt_tokens": summary(prompt_tokens),
            "reference_output_tokens": output_summary,
            "recommended_max_new_tokens": recommended,
        }
        print(
            f"[profile] {task_name}: prompt_max={max(prompt_tokens, default=0)} "
            f"output_p99.9={output_summary['p99_9']} recommended={recommended}"
        )
    atomic_write_json(str(args.out), result)
    print(f"[profile] wrote {args.out}")


if __name__ == "__main__":
    main()
