"""S2-TOMG-Bench (phenixace/S2-TOMG-Bench) — text-based open molecule generation.

9 subtasks in 3 groups: MolCustom {AtomNum, BondNum, FunctionalGroup},
MolEdit {AddComponent, DelComponent, SubComponent}, MolOpt {LogP, MR, QED}.

Modelled as ONE molbench task whose evaluate() returns SR + WSR per group plus
the average — i.e. exactly the paper's leaderboard table (models as rows).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from ...core.benchmark import Benchmark
from ...core.registry import register_benchmark
from ...core.task import EvalRecord, Task
from ...utils.chem import canonicalize, extract_smiles

CSV_URL = "https://huggingface.co/datasets/{repo}/resolve/main/{cfg}.csv"
SCALE_REPO = {"full": "phenixace/S2-TOMG-Bench", "mini": "phenixace/S2-TOMG-Bench-mini"}

# (task_group, subtask) — config name is "{group}_{subtask}"
SUBTASK_CONFIGS = [
    ("MolCustom", "AtomNum"), ("MolCustom", "BondNum"), ("MolCustom", "FunctionalGroup"),
    ("MolEdit", "AddComponent"), ("MolEdit", "DelComponent"), ("MolEdit", "SubComponent"),
    ("MolOpt", "LogP"), ("MolOpt", "MR"), ("MolOpt", "QED"),
]

FORMAT_SUFFIX = (
    "\n\nYou may reason step by step, but your final response must be a single "
    "valid SMILES string in the form: Molecule: <SMILES>."
)

COLUMNS = [
    ("MolCustom SR↑", "molcustom_sr"), ("MolCustom WSR↑", "molcustom_wsr"),
    ("MolEdit SR↑", "moledit_sr"), ("MolEdit WSR↑", "moledit_wsr"),
    ("MolOpt SR↑", "molopt_sr"), ("MolOpt WSR↑", "molopt_wsr"),
    ("Avg SR↑", "avg_sr"), ("Avg WSR↑", "avg_wsr"),
]


def _correct_text(text: str) -> str:
    """Extract a SMILES from raw model output (JSON / 'Molecule:' / => / ->)."""
    if not isinstance(text, str):
        return ""
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            s = json.loads(m.group().replace('""', '"')).get("molecule", "")
        except Exception:
            s = m.group()
    else:
        s = text.replace("\n", " ").strip()
    # strip a trailing "Molecule:" / "SMILES:" label and arrows
    for label in ("molecule:", "smiles:"):
        idx = s.lower().rfind(label)
        if idx != -1:
            s = s[idx + len(label):]
    for sep in ("=>", "->"):
        if sep in s:
            s = s.split(sep)[-1]
    s = s.strip().strip('"').strip("'").strip()
    if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
        s = s[1:-1]
    return s.split()[0] if s.split() else s


class OpenGenerationTask(Task):
    name = "open_generation"
    max_new_tokens = 512
    columns = COLUMNS

    def build_prompt(self, example: Dict[str, Any]) -> str:
        return str(example["Instruction"]) + FORMAT_SUFFIX

    def postprocess(self, answer: str) -> str:
        s = _correct_text(answer)
        if canonicalize(s) is not None:
            return s
        alt = extract_smiles(answer)  # fallback: first RDKit-valid token
        return alt or s

    def evaluate(self, records: List[EvalRecord], device: str = "cpu") -> Dict[str, Any]:
        from ...metrics.tomg import SUBTASKS, TASK_GROUPS, score_subtask

        by_sub: Dict[str, List[EvalRecord]] = {}
        for r in records:
            by_sub.setdefault(r.example["subtask"], []).append(r)

        per_sub = {}
        for sub, recs in by_sub.items():
            per_sub[sub] = score_subtask(
                sub, [r.example for r in recs], [r.prediction for r in recs]
            )

        out: Dict[str, Any] = {}
        for group in TASK_GROUPS:
            subs = [s for s in per_sub if SUBTASKS[s][0] == group]
            key = group.lower()
            if subs:
                out[f"{key}_sr"] = sum(per_sub[s]["sr"] for s in subs) / len(subs)
                out[f"{key}_wsr"] = sum(per_sub[s]["wsr"] for s in subs) / len(subs)
            else:
                out[f"{key}_sr"] = out[f"{key}_wsr"] = None
        if per_sub:
            out["avg_sr"] = sum(v["sr"] for v in per_sub.values()) / len(per_sub)
            out["avg_wsr"] = sum(v["wsr"] for v in per_sub.values()) / len(per_sub)
        out["_per_subtask"] = per_sub
        return out

    def score_examples(self, records, device="cpu"):
        from ...metrics.tomg import SUBTASKS, score_examples_subtask

        by_sub: Dict[str, List[int]] = {}
        for idx, r in enumerate(records):
            by_sub.setdefault(r.example["subtask"], []).append(idx)
        out: List[Any] = [None] * len(records)
        for sub, idxs in by_sub.items():
            rows = [records[i].example for i in idxs]
            preds = [records[i].prediction for i in idxs]
            sc = score_examples_subtask(sub, rows, preds)
            for i, s in zip(idxs, sc):
                s["subtask"] = sub
                s["task_group"] = SUBTASKS[sub][0]
                s["wsr_contrib"] = (s["quality"] or 0.0) * s["success"]
                out[i] = s
        return out


class TOMGBenchmark(Benchmark):
    """`scale`: 'full' (5000/subtask) or 'mini' (500/subtask)."""

    def __init__(self, scale: str = "full"):
        self.scale = scale
        self.name = "tomg" if scale == "full" else "tomg-mini"

    def load(self, split: str = "test", limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """`limit` caps examples PER SUBTASK (so all 9 subtasks are covered)."""
        from datasets import load_dataset

        repo = SCALE_REPO[self.scale]
        examples: List[Dict[str, Any]] = []
        for group, subtask in SUBTASK_CONFIGS:
            cfg = f"{group}_{subtask}"
            url = CSV_URL.format(repo=repo, cfg=cfg)
            ds = load_dataset("csv", data_files={"test": url}, split="test")
            if limit is not None:
                ds = ds.select(range(min(limit, len(ds))))
            for row in ds:
                d = dict(row)
                d["task_group"] = group
                d["subtask"] = subtask
                examples.append(d)
        return examples

    def tasks(self) -> Dict[str, Task]:
        return {"open_generation": OpenGenerationTask()}


register_benchmark("tomg", lambda: TOMGBenchmark("full"))
register_benchmark("tomg-mini", lambda: TOMGBenchmark("mini"))
