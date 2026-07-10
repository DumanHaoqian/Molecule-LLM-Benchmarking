#!/usr/bin/env python
"""Stage 1 CLI — generate predictions with a ChemDFM model (chemdfm venv).

Run in the generation venv:
    source /home/haoqian/Data/SAERAG/venvs/chemdfm/bin/activate

Examples
--------
Smoke test, 10 examples, both tasks, one model:
    python generate.py --model chemdfm-v2 --task both --limit 10 --out-dir results/smoke

Full test split for the reasoning model:
    python generate.py --model chemdfm-r --task both --split test --out-dir results/full
"""
from __future__ import annotations

import argparse

from benchmark.generation import TASKS, generate_task
from benchmark.model import ChemDFMModel
from benchmark.registry import MODEL_REGISTRY, get_model_cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="ChEBI-20 generation stage")
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument(
        "--task", choices=["captioning", "caption2smiles", "both"], default="both"
    )
    ap.add_argument(
        "--split", default="test", choices=["train", "validation", "test"]
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=None, help="override")
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    cfg = get_model_cfg(args.model)
    print(f"[generate] model={args.model} ({cfg['display_name']}) path={cfg['path']}")
    model = ChemDFMModel(model_path=cfg["path"])

    tasks = TASKS if args.task == "both" else (args.task,)
    for task in tasks:
        generate_task(
            model,
            model_key=args.model,
            task=task,
            split=args.split,
            limit=args.limit,
            batch_size=args.batch_size,
            out_dir=args.out_dir,
            do_sample=args.do_sample,
            max_new_tokens=args.max_new_tokens,
        )
    print("[generate] done")


if __name__ == "__main__":
    main()
