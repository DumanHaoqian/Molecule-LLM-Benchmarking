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
import os

import molbench.benchmarks  # noqa: F401  (registers benchmarks)
import molbench.models  # noqa: F401  (registers models)
from molbench.core.registry import get_benchmark, list_benchmarks, list_models
from molbench.core.io import atomic_write_json, atomic_write_text
from molbench.core.runner import GracefulStop, run_evaluation, run_generation


def _resolve_tasks(benchmark: str, task_arg: str):
    if task_arg in (None, "all"):
        return None
    return [task_arg]


def _parse_slice(value: str | None):
    if value is None:
        return None
    try:
        start, stop = value.split(":", 1)
        return int(start), int(stop)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("slice must be START:STOP") from exc


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
        resume=args.resume,
        restart=args.restart,
        batching=args.batching,
        length_batch_policy=args.length_batch_policy,
        token_budget=args.token_budget,
        heartbeat_seconds=args.heartbeat_seconds,
        index_slice=args.index_slice,
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
        resume=args.resume,
        restart=args.restart,
        chunk_size=args.chunk_size,
    )
    print("\n" + res["markdown"])
    os.makedirs(args.out_dir, exist_ok=True)
    md = args.out or os.path.join(args.out_dir, f"tables_{args.benchmark}_{args.split}.md")
    atomic_write_text(md, res["markdown"])
    js = os.path.join(args.out_dir, f"metrics_{args.benchmark}_{args.split}.json")
    atomic_write_json(js, res["tasks"])
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
    g.add_argument("--batch-size", "--max-batch-size", dest="batch_size", type=int, default=8)
    g.add_argument("--do-sample", action="store_true")
    g.add_argument("--out-dir", default="results")
    g.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--restart", action="store_true")
    g.add_argument("--batching", choices=("fixed", "length-aware"), default="length-aware")
    g.add_argument(
        "--length-batch-policy",
        default="128:16,256:8,384:4,512:2,inf:1",
        help="comma-separated max_chars:batch_size bands",
    )
    g.add_argument("--token-budget", type=int, default=16384)
    g.add_argument("--heartbeat-seconds", type=int, default=30)
    g.add_argument("--slice", dest="index_slice", type=_parse_slice, default=None)
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("evaluate", help="score predictions + build tables")
    e.add_argument("--benchmark", required=True, choices=list_benchmarks())
    e.add_argument("--models", nargs="+", required=True, choices=list_models())
    e.add_argument("--task", default="all", help="task name or 'all'")
    e.add_argument("--split", default="test")
    e.add_argument("--device", default="cpu")
    e.add_argument("--out-dir", default="results")
    e.add_argument("--out", default=None)
    e.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    e.add_argument("--restart", action="store_true")
    e.add_argument("--chunk-size", type=int, default=32)
    e.set_defaults(func=cmd_evaluate)

    args = ap.parse_args()
    try:
        args.func(args)
    except GracefulStop as exc:
        print(f"[stopped] {exc}")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
