"""Prediction/metadata artifact I/O and naming conventions.

Dependency-free (no torch / datasets) so the evaluation venv can import it.
Filenames include the benchmark so multiple benchmarks coexist in one dir:
    <out_dir>/<benchmark>__<model>__<task>__<split>.jsonl (+ .meta.json)
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from .task import EvalRecord


def pred_stem(out_dir: str, benchmark: str, model: str, task: str, split: str) -> str:
    return os.path.join(out_dir, f"{benchmark}__{model}__{task}__{split}")


def pred_paths(out_dir: str, benchmark: str, model: str, task: str, split: str):
    stem = pred_stem(out_dir, benchmark, model, task, split)
    return stem + ".jsonl", stem + ".meta.json"


def write_records(path: str, records: List[EvalRecord]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(
                json.dumps(
                    {
                        "example": r.example,
                        "prompt": r.prompt,
                        "raw_output": r.raw_output,
                        "prediction": r.prediction,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def read_records(path: str) -> List[EvalRecord]:
    records = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            records.append(
                EvalRecord(
                    example=d["example"],
                    prompt=d.get("prompt", ""),
                    raw_output=d.get("raw_output", ""),
                    prediction=d.get("prediction"),
                )
            )
    return records


def write_meta(path: str, meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
