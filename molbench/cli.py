"""molbench CLI — `generate` and `evaluate` subcommands.

    # stage 1 (chemdfm venv)
    python -m molbench generate --benchmark chebi20 --model chemdfm-v2 --limit 10
    # stage 2 (ChEBI-20-Eva venv)
    python -m molbench evaluate --benchmark chebi20 --models chemdfm-v2 chemdfm-r

Importing molbench.models / molbench.benchmarks registers everything, so the
CLI is fully generic — new models/benchmarks need no CLI changes.
"""
from __future__ import annotations

import argparse
import json
import os

import molbench.benchmarks  # noqa: F401  (registers benchmarks)
import molbench.models  # noqa: F401  (registers models)
from molbench.core.registry import get_benchmark, list_benchmarks, list_models
from molbench.core.runner import run_evaluation, run_generation


def _resolve_tasks(benchmark: str, task_arg: str):
    if task_arg in (None, "all"):
        return None
    return [task_arg]


def cmd_generate(args) -> None:
    run_generation(
        benchmark_name=args.benchmark,
        model_key=args.model,
        task_names=_resolve_tasks(args.benchmark, args.task),
        split=args.split,
        limit=args.limit,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        do_sample=args.do_sample,
    )
    print("[generate] done")


def cmd_evaluate(args) -> None:
    res = run_evaluation(
        benchmark_name=args.benchmark,
        model_keys=args.models,
        task_names=_resolve_tasks(args.benchmark, args.task),
        split=args.split,
        out_dir=args.out_dir,
        device=args.device,
    )
    print("\n" + res["markdown"])
    os.makedirs(args.out_dir, exist_ok=True)
    md = args.out or os.path.join(args.out_dir, f"tables_{args.benchmark}_{args.split}.md")
    with open(md, "w") as f:
        f.write(res["markdown"])
    js = os.path.join(args.out_dir, f"metrics_{args.benchmark}_{args.split}.json")
    with open(js, "w") as f:
        json.dump(res["tasks"], f, indent=2, ensure_ascii=False)
    print(f"\n[evaluate] tables -> {md}\n[evaluate] metrics -> {js}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="molbench", description="Molecule-LLM benchmarking")
    sub = ap.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="run a model over a benchmark")
    g.add_argument("--benchmark", required=True, choices=list_benchmarks())
    g.add_argument("--model", required=True, choices=list_models())
    g.add_argument("--task", default="all", help="task name or 'all'")
    g.add_argument("--split", default="test")
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--do-sample", action="store_true")
    g.add_argument("--out-dir", default="results")
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("evaluate", help="score predictions + build tables")
    e.add_argument("--benchmark", required=True, choices=list_benchmarks())
    e.add_argument("--models", nargs="+", required=True, choices=list_models())
    e.add_argument("--task", default="all", help="task name or 'all'")
    e.add_argument("--split", default="test")
    e.add_argument("--device", default="cpu")
    e.add_argument("--out-dir", default="results")
    e.add_argument("--out", default=None)
    e.set_defaults(func=cmd_evaluate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
