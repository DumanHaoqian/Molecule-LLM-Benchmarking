#!/usr/bin/env python
"""Stage 2 CLI — score predictions and print the two ChEBI-20 tables.

Run in the evaluation venv:
    source /home/haoqian/Data/SAERAG/venvs/ChEBI-20-Eva/bin/activate

Example:
    python evaluate.py --results-dir results/smoke --models chemdfm-v2 chemdfm-r \
        --split test
"""
from __future__ import annotations

import argparse
import json
import os

from benchmark.evaluation import build_tables
from benchmark.registry import MODEL_REGISTRY


def main() -> None:
    ap = argparse.ArgumentParser(description="ChEBI-20 evaluation stage")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument(
        "--models", nargs="+", default=list(MODEL_REGISTRY),
        help="model keys to include (must have prediction files present)",
    )
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--device", default="cpu",
        help="device for FCD / Text2Mol (cpu keeps the eval venv build simple)",
    )
    ap.add_argument("--out", default=None, help="output .md path (default in results-dir)")
    args = ap.parse_args()

    res = build_tables(args.results_dir, args.models, args.split, args.device)

    gen_title = "### Table 1 — Text-based molecule generation (caption2SMILES)"
    cap_title = "### Table 2 — Molecule captioning"
    md = (
        f"{gen_title}\n\n{res['gen_table']}\n\n"
        f"{cap_title}\n\n{res['cap_table']}\n"
    )
    print("\n" + md)

    out_md = args.out or os.path.join(args.results_dir, f"tables_{args.split}.md")
    with open(out_md, "w") as f:
        f.write(md)
    out_json = os.path.join(args.results_dir, f"metrics_{args.split}.json")
    with open(out_json, "w") as f:
        json.dump(
            {"gen_rows": res["gen_rows"], "cap_rows": res["cap_rows"]},
            f, indent=2, ensure_ascii=False,
        )
    print(f"\n[eval] tables -> {out_md}\n[eval] metrics -> {out_json}")


if __name__ == "__main__":
    main()
