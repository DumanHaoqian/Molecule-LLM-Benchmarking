"""Stage 2 — evaluation. Scores saved predictions and builds the two tables.

Runs in the ``ChEBI-20-Eva`` venv (has rouge_score, fcd_torch, and optionally
the Text2Mol stack). Reads the jsonl files written by ``generate.py``.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .metrics_caption import caption_metrics
from .metrics_fcd import fcd_metric
from .metrics_molgen import molgen_metrics
from .metrics_text2mol import text2mol_metric
from .paths import pred_paths
from .registry import get_model_cfg
from .smiles_utils import extract_smiles


def _read_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _read_meta(out_dir: str, model_key: str, task: str, split: str) -> dict:
    _, meta_path = pred_paths(out_dir, model_key, task, split)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    cfg = get_model_cfg(model_key)
    return {"display_name": cfg["display_name"], "params": cfg["params"]}


def eval_captioning(
    out_dir: str, model_key: str, split: str, device: str = "cuda"
) -> Dict:
    jsonl, _ = pred_paths(out_dir, model_key, "captioning", split)
    rows = _read_jsonl(jsonl)
    preds = [r["prediction"] for r in rows]
    refs = [r["target"] for r in rows]
    gold_smiles = [r["input"] for r in rows]

    m = caption_metrics(preds, refs)
    # Text2Mol: generated caption vs gold molecule
    m["text2mol"] = text2mol_metric(gold_smiles, preds, device=device)
    return m


def eval_caption2smiles(
    out_dir: str, model_key: str, split: str, device: str = "cuda"
) -> Dict:
    jsonl, _ = pred_paths(out_dir, model_key, "caption2smiles", split)
    rows = _read_jsonl(jsonl)
    pred_smiles = [extract_smiles(r["prediction"]) for r in rows]
    gold_smiles = [r["target"] for r in rows]
    descriptions = [r["input"] for r in rows]

    m = molgen_metrics(pred_smiles, gold_smiles)
    m["fcd"] = fcd_metric(pred_smiles, gold_smiles, device=device)
    # Text2Mol: generated molecule vs gold caption
    m["text2mol"] = text2mol_metric(pred_smiles, descriptions, device=device)
    return m


# ---- table rendering ----------------------------------------------------

def _fmt(v, decimals=3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


GEN_COLS = [
    ("BLEU↑", "bleu"),
    ("EM↑", "exact_match"),
    ("Levenshtein↓", "levenshtein"),
    ("MACCS FTS↑", "maccs_fts"),
    ("RDK FTS↑", "rdk_fts"),
    ("Morgan FTS↑", "morgan_fts"),
    ("FCD↓", "fcd"),
    ("Text2Mol↑", "text2mol"),
    ("Validity↑", "validity"),
]

CAP_COLS = [
    ("BLEU-2↑", "bleu2"),
    ("BLEU-4↑", "bleu4"),
    ("ROUGE-1↑", "rouge1"),
    ("ROUGE-2↑", "rouge2"),
    ("ROUGE-L↑", "rougeL"),
    ("METEOR↑", "meteor"),
    ("Text2Mol↑", "text2mol"),
]


def _render_table(title_cols, header0, rows: List[dict]) -> str:
    header = ["Method", "#Params."] + [c[0] for c in title_cols]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows:
        cells = [r["display_name"], r["params"]]
        for _, key in title_cols:
            dec = 1 if key == "levenshtein" else 3
            cells.append(_fmt(r["metrics"].get(key), dec))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_tables(
    out_dir: str, model_keys: List[str], split: str, device: str = "cuda"
) -> Dict:
    """Evaluate each model on both tasks and return rows + rendered tables."""
    gen_rows, cap_rows = [], []
    for key in model_keys:
        meta_c = _read_meta(out_dir, key, "captioning", split)
        meta_g = _read_meta(out_dir, key, "caption2smiles", split)
        print(f"[eval] {key} :: captioning")
        cap_m = eval_captioning(out_dir, key, split, device)
        print(f"[eval] {key} :: caption2smiles")
        gen_m = eval_caption2smiles(out_dir, key, split, device)
        cap_rows.append(
            {"display_name": meta_c["display_name"], "params": meta_c["params"],
             "metrics": cap_m}
        )
        gen_rows.append(
            {"display_name": meta_g["display_name"], "params": meta_g["params"],
             "metrics": gen_m}
        )

    gen_table = _render_table(GEN_COLS, "gen", gen_rows)
    cap_table = _render_table(CAP_COLS, "cap", cap_rows)
    return {
        "gen_rows": gen_rows,
        "cap_rows": cap_rows,
        "gen_table": gen_table,
        "cap_table": cap_table,
    }
