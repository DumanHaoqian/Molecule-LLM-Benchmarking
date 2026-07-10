"""Generic two-stage runner: generation and evaluation.

Works for any registered benchmark/model — no benchmark-specific logic here.
Generation writes prediction jsonl; evaluation reads it back and renders one
markdown table per task (rows = models, columns = the task's declared metrics).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import json
import os

from .io import (
    pred_paths,
    pred_stem,
    read_records,
    write_meta,
    write_records,
)
from .registry import get_benchmark, get_model_spec
from .task import EvalRecord


# ---- stage 1: generation ------------------------------------------------

def run_generation(
    benchmark_name: str,
    model_key: str,
    task_names: Optional[List[str]],
    split: str = "test",
    limit: Optional[int] = None,
    out_dir: str = "results",
    batch_size: int = 8,
    do_sample: bool = False,
) -> None:
    bench = get_benchmark(benchmark_name)
    spec = get_model_spec(model_key)
    tasks = bench.tasks()
    task_names = task_names or list(tasks)

    examples = bench.load(split=split, limit=limit)
    print(f"[gen] {benchmark_name} :: {model_key} :: {len(examples)} examples")
    model = spec.build()

    for tname in task_names:
        task = tasks[tname]
        prompts = [task.build_prompt(e) for e in examples]
        budget = model.answer_budget(task.max_new_tokens)
        raw = model.generate(
            prompts, max_new_tokens=budget, batch_size=batch_size, do_sample=do_sample
        )
        records = [
            EvalRecord(example=e, prompt=p, raw_output=r, prediction=task.postprocess(r))
            for e, p, r in zip(examples, prompts, raw)
        ]
        jsonl, meta = pred_paths(out_dir, benchmark_name, model_key, tname, split)
        write_records(jsonl, records)
        write_meta(
            meta,
            {
                "benchmark": benchmark_name,
                "model_key": model_key,
                "display_name": spec.display_name,
                "params": spec.params,
                "task": tname,
                "split": split,
                "limit": limit,
                "n": len(examples),
                "do_sample": do_sample,
                "max_new_tokens": budget,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        )
        print(f"[gen] wrote {len(records)} -> {jsonl}")


# ---- stage 2: evaluation ------------------------------------------------

def _fmt(v: Any, decimals: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _render_table(task, rows: List[dict]) -> str:
    header = ["Method", "#Params."] + [h for h, _ in task.columns]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        cells = [r["display_name"], r["params"]]
        for _, key in task.columns:
            dec = 1 if key == "levenshtein" else 3
            cells.append(_fmt(r["metrics"].get(key), dec))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_evaluation(
    benchmark_name: str,
    model_keys: List[str],
    task_names: Optional[List[str]],
    split: str = "test",
    out_dir: str = "results",
    device: str = "cpu",
) -> Dict[str, Any]:
    bench = get_benchmark(benchmark_name)
    tasks = bench.tasks()
    task_names = task_names or list(tasks)

    result: Dict[str, Any] = {"benchmark": benchmark_name, "split": split, "tasks": {}}
    md_sections: List[str] = []

    for tname in task_names:
        task = tasks[tname]
        rows = []
        for mkey in model_keys:
            spec = get_model_spec(mkey)
            jsonl, _ = pred_paths(out_dir, benchmark_name, mkey, tname, split)
            print(f"[eval] {benchmark_name} :: {mkey} :: {tname}")
            records = read_records(jsonl)
            metrics = task.evaluate(records, device=device)
            rows.append(
                {"display_name": spec.display_name, "params": spec.params, "metrics": metrics}
            )
            # per-example scores (structured, for bad-case analysis)
            scored = task.score_examples(records, device=device)
            if scored is not None:
                path = pred_stem(out_dir, benchmark_name, mkey, tname, split) + "__scored.jsonl"
                with open(path, "w") as f:
                    for rec, sc in zip(records, scored):
                        f.write(
                            json.dumps(
                                {
                                    "example": rec.example,
                                    "prompt": rec.prompt,
                                    "raw_output": rec.raw_output,
                                    "prediction": rec.prediction,
                                    "scores": sc,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                print(f"[eval] per-example scores -> {path}")
        table = _render_table(task, rows)
        md_sections.append(f"### {benchmark_name} — {tname}\n\n{table}\n")
        result["tasks"][tname] = rows

    result["markdown"] = "\n".join(md_sections)
    return result
