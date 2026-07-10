"""Batch runner for multiple evaluation tasks/subtasks."""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import import_module
from pathlib import Path

from evaluation.core.config import get_task_spec
from evaluation.core.runner import RunnerConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_CLASS_MAP = {
    "mol_edit": ("evaluation.mol_edit.run", "MolEditRunner"),
    "rxn_pred": ("evaluation.rxn_pred.run", "RxnPredRunner"),
    "mol_und": ("evaluation.mol_und.run", "MolUndRunner"),
    "mol_opt": ("evaluation.mol_opt.run", "MolOptRunner"),
}


def _load_runner_class(task: str):
    module_name, class_name = RUNNER_CLASS_MAP[task]
    module = import_module(module_name)
    return getattr(module, class_name)


def _run_one(payload: dict) -> dict:
    task = payload["task"]
    runner_cls = _load_runner_class(task)
    cfg = RunnerConfig(**payload["cfg"])
    summary = runner_cls(cfg).run()
    return {
        "task": task,
        "subtask": payload["cfg"]["subtask"],
        "summary": summary,
    }


def _aggregate_entries(entries: list[dict]) -> list[dict]:
    rows = []
    for entry in entries:
        summary = entry["summary"]
        row = {
            "task": entry["task"],
            "subtask": entry["subtask"],
        }
        layer1 = summary.get("layer1", {})
        layer2 = summary.get("layer2", {})
        layer3 = summary.get("layer3", {})

        for key in (
            "top1_acc",
            "avg_fts",
            "exact_match_acc",
            "mean_mae",
            "avg_tanimoto",
            "sr_pct",
            "dual_sr_pct",
        ):
            if key in layer1:
                row[f"layer1_{key}"] = layer1[key]
        for key in ("primary_metric_name", "primary_metric_value", "primary_metric_direction"):
            if key in layer1:
                row[f"layer1_{key}"] = layer1[key]
        for key in ("state_score", "avg_state_score"):
            if key in layer2:
                row["layer2_state_score"] = layer2[key]
        if "avg_step_score" in layer3:
            row["layer3_step_score"] = layer3["avg_step_score"]
        if "type1" in layer3 and "all_pass_rate" in layer3["type1"]:
            row["layer3_type1_all_pass_rate"] = layer3["type1"]["all_pass_rate"]
        if "type2" in layer3 and "all_fields_match_rate" in layer3["type2"]:
            row["layer3_type2_all_match_rate"] = layer3["type2"]["all_fields_match_rate"]
        rows.append(row)
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run multiple evaluation subtasks")
    parser.add_argument("--tasks", required=True, help="Comma-separated tasks, e.g. mol_edit,rxn_pred")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--base-url", default="https://api.ppio.com/openai")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-workers", type=int, default=4, help="Process-level parallelism")
    parser.add_argument(
        "--sample-workers",
        type=int,
        default=1,
        help="Per-subtask sampling concurrency passed into each runner",
    )
    parser.add_argument("--reasoning-effort", type=str, default=None, help="e.g. low, medium, high")
    parser.add_argument("--timeout", type=float, default=800.0, help="API call timeout in seconds")
    args = parser.parse_args()

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    payloads = []
    for task in tasks:
        spec = get_task_spec(task)
        for subtask in sorted(spec.subtasks):
            payloads.append(
                {
                    "task": task,
                    "cfg": {
                        "task": task,
                        "subtask": subtask,
                        "model_name": args.model,
                        "base_url": args.base_url or "",
                        "api_key": (
                            args.api_key
                            if args.api_key is not None
                            else ("dummy" if args.eval_only else None)
                        ),
                        "max_tokens": args.max_tokens,
                        "temperature": args.temperature,
                        "delay": args.delay,
                        "n_samples": args.n_samples,
                        "seed": args.seed,
                        "eval_only": args.eval_only,
                        "rerun_failed": args.rerun_failed,
                        "max_workers": args.sample_workers,
                        "reasoning_effort": args.reasoning_effort,
                        "timeout": args.timeout,
                    },
                }
            )

    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(_run_one, payload): payload for payload in payloads}
        for future in as_completed(futures):
            payload = futures[future]
            task = payload["task"]
            subtask = payload["cfg"]["subtask"]
            try:
                result = future.result()
            except Exception as exc:
                print(f"[failed] {task}/{subtask}: {exc}")
                failures.append(
                    {
                        "task": task,
                        "subtask": subtask,
                        "error": str(exc),
                    }
                )
                continue
            print(f"[done] {task}/{subtask}")
            results.append(result)

    results.sort(key=lambda item: (item["task"], item["subtask"]))
    rows = _aggregate_entries(results)

    safe_model = args.model.replace("/", "_")
    out_dir = PROJECT_ROOT / "results" / "evaluation" / "run_all" / safe_model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    with open(out_path, "w") as handle:
        json.dump(
            {
                "config": {
                    "tasks": tasks,
                    "model": args.model,
                    "n_samples": args.n_samples,
                    "eval_only": args.eval_only,
                },
                "rows": rows,
                "results": results,
                "failures": failures,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("\nBatch summary")
    print("=" * 70)
    for row in rows:
        print(f"{row['task']}/{row['subtask']}: {json.dumps(row, ensure_ascii=False)}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure['task']}/{failure['subtask']}: {failure['error']}")
    print("=" * 70)
    print(f"Saved aggregated summary to {out_path}")


if __name__ == "__main__":
    main()
