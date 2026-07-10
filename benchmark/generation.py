"""Stage 1 — generation. Runs a model over ChEBI-20 and writes raw predictions.

Deliberately metric-free so it can run in the lightweight ``chemdfm`` venv.
All scoring happens later in ``evaluate.py`` (the ``ChEBI-20-Eva`` venv).

Output per (model, task, split):
  <out_dir>/<model>__<task>__<split>.jsonl   one prediction per line
  <out_dir>/<model>__<task>__<split>.meta.json   run metadata
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

from .data import SMILES_COL, TEXT_COL, load_chebi20
from .paths import pred_paths
from .prompts import (
    CAPTION2SMILES_PROMPT,
    CAPTIONING_PROMPT,
    caption2smiles_instruction,
    captioning_instruction,
)
from .registry import get_model_cfg
from .smiles_utils import parse_answer

TASKS = ("captioning", "caption2smiles")


def _task_fields(task: str):
    """Return (input_col, target_col, prompt_fn, prompt_template)."""
    if task == "captioning":
        return SMILES_COL, TEXT_COL, captioning_instruction, CAPTIONING_PROMPT
    if task == "caption2smiles":
        return TEXT_COL, SMILES_COL, caption2smiles_instruction, CAPTION2SMILES_PROMPT
    raise ValueError(f"unknown task '{task}'")


def generate_task(
    model,
    model_key: str,
    task: str,
    split: str = "test",
    limit: Optional[int] = None,
    batch_size: int = 8,
    out_dir: str = "results",
    do_sample: bool = False,
    max_new_tokens: Optional[int] = None,
) -> str:
    """Generate predictions for one (model, task); returns the jsonl path."""
    cfg = get_model_cfg(model_key)
    in_col, tgt_col, prompt_fn, template = _task_fields(task)
    if max_new_tokens is None:
        max_new_tokens = cfg["max_new_tokens"][task]

    ds = load_chebi20(split=split, limit=limit)
    inputs = ds[in_col]
    targets = ds[tgt_col]
    instructions = [prompt_fn(x) for x in inputs]

    raw = model.generate(
        instructions,
        system=cfg["system"],
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        do_sample=do_sample,
    )
    predictions = [parse_answer(r, cfg["reasoning"]) for r in raw]

    jsonl_path, meta_path = pred_paths(out_dir, model_key, task, split)
    os.makedirs(out_dir, exist_ok=True)
    with open(jsonl_path, "w") as f:
        for i, (x, tgt, rw, pred) in enumerate(
            zip(inputs, targets, raw, predictions)
        ):
            f.write(
                json.dumps(
                    {
                        "idx": i,
                        "input": x,
                        "target": tgt,
                        "raw_output": rw,
                        "prediction": pred,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    meta = {
        "model_key": model_key,
        "display_name": cfg["display_name"],
        "params": cfg["params"],
        "reasoning": cfg["reasoning"],
        "task": task,
        "split": split,
        "limit": limit,
        "n": len(inputs),
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "system": cfg["system"],
        "prompt_template": template,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[gen] wrote {len(inputs)} preds -> {jsonl_path}")
    return jsonl_path
