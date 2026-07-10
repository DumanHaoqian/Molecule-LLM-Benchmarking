"""Task orchestration: run a model over ChEBI-20 and score it."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .data import SMILES_COL, TEXT_COL, load_chebi20
from .metrics_caption import caption_metrics
from .metrics_molgen import molgen_metrics
from .prompts import caption2smiles_instruction, captioning_instruction
from .smiles_utils import extract_smiles


def _dump_jsonl(rows: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_captioning(
    model,
    split: str = "test",
    limit: Optional[int] = None,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    out_dir: str = "results",
    **gen_kwargs,
) -> Dict[str, float]:
    """SMILES -> description. Returns metric dict and writes predictions."""
    ds = load_chebi20(split=split, limit=limit)
    smiles = ds[SMILES_COL]
    refs = ds[TEXT_COL]
    instructions = [captioning_instruction(s) for s in smiles]

    preds = model.generate(
        instructions,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        **gen_kwargs,
    )
    metrics = caption_metrics(preds, refs)

    rows = [
        {"smiles": s, "reference": r, "prediction": p}
        for s, r, p in zip(smiles, refs, preds)
    ]
    _dump_jsonl(rows, os.path.join(out_dir, f"captioning_{split}_preds.jsonl"))
    return metrics


def run_caption2smiles(
    model,
    split: str = "test",
    limit: Optional[int] = None,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    out_dir: str = "results",
    **gen_kwargs,
) -> Dict[str, float]:
    """Description -> SMILES. Returns metric dict and writes predictions."""
    ds = load_chebi20(split=split, limit=limit)
    descriptions = ds[TEXT_COL]
    golds = ds[SMILES_COL]
    instructions = [caption2smiles_instruction(d) for d in descriptions]

    raw = model.generate(
        instructions,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        **gen_kwargs,
    )
    preds = [extract_smiles(r) for r in raw]
    metrics = molgen_metrics(preds, golds)

    rows = [
        {"description": d, "gold_smiles": g, "raw_output": rw, "pred_smiles": p}
        for d, g, rw, p in zip(descriptions, golds, raw, preds)
    ]
    _dump_jsonl(
        rows, os.path.join(out_dir, f"caption2smiles_{split}_preds.jsonl")
    )
    return metrics
