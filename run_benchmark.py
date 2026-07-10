#!/usr/bin/env python
"""CLI entry point for the Molecule-LLM ChEBI-20 benchmark.

Examples
--------
Smoke test (5 examples, both tasks):
    python run_benchmark.py --task both --limit 5

Full ChemDFM evaluation on the test split:
    python run_benchmark.py --task both --split test \
        --batch-size 8 --out-dir results/chemdfm

Only molecule captioning:
    python run_benchmark.py --task captioning --split test
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

from benchmark.model import DEFAULT_MODEL_PATH, ChemDFMModel
from benchmark.tasks import run_caption2smiles, run_captioning


def _print_metrics(name: str, metrics: dict) -> None:
    print(f"\n===== {name} =====")
    for k, v in metrics.items():
        print(f"  {k:14s}: {v:.4f}" if isinstance(v, float) else f"  {k:14s}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Molecule-LLM ChEBI-20 benchmark")
    ap.add_argument(
        "--task",
        choices=["captioning", "caption2smiles", "both"],
        default="both",
    )
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--limit", type=int, default=None, help="cap #examples (debug)")
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--do-sample",
        action="store_true",
        help="sample instead of greedy (greedy is the reproducible default)",
    )
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = ChemDFMModel(model_path=args.model_path)

    gen_kwargs = dict(do_sample=args.do_sample)
    common = dict(
        split=args.split,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
    )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_path": args.model_path,
        "split": args.split,
        "limit": args.limit,
        "do_sample": args.do_sample,
        "max_new_tokens": args.max_new_tokens,
        "results": {},
    }

    if args.task in ("captioning", "both"):
        m = run_captioning(model, **common, **gen_kwargs)
        _print_metrics("Molecule Captioning (SMILES -> text)", m)
        summary["results"]["captioning"] = m

    if args.task in ("caption2smiles", "both"):
        m = run_caption2smiles(model, **common, **gen_kwargs)
        _print_metrics("Caption2SMILES (text -> SMILES)", m)
        summary["results"]["caption2smiles"] = m

    summary_path = os.path.join(args.out_dir, f"summary_{args.split}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[done] summary -> {summary_path}")


if __name__ == "__main__":
    main()
