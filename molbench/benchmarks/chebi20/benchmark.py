"""ChEBI-20 benchmark (duongttr/chebi-20): captioning + caption2SMILES.

Columns: CAN_SMILES (canonical SMILES), DESCRIPTION (natural-language caption).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...core.benchmark import Benchmark
from ...core.registry import register_benchmark
from ...core.task import EvalRecord, Task
from ...utils.chem import extract_smiles

# NOTE: metric modules (rouge_score / fcd_torch / gensim) are imported lazily
# inside evaluate() so the lightweight *generation* venv — which only needs
# build_prompt / postprocess — can import this benchmark without them.

REPO = "duongttr/chebi-20"
SMILES = "CAN_SMILES"
TEXT = "DESCRIPTION"

CAPTIONING_PROMPT = (
    "Please give me some details about the molecule with the following "
    "SMILES representation.\n{smiles}"
)
CAPTION2SMILES_PROMPT = (
    "Please give me a molecule (represented in SMILES) that fits the "
    "following description.\n{description}"
)

CAPTION_COLUMNS = [
    ("BLEU-2↑", "bleu2"), ("BLEU-4↑", "bleu4"),
    ("ROUGE-1↑", "rouge1"), ("ROUGE-2↑", "rouge2"), ("ROUGE-L↑", "rougeL"),
    ("METEOR↑", "meteor"), ("Text2Mol↑", "text2mol"),
]
GEN_COLUMNS = [
    ("BLEU↑", "bleu"), ("EM↑", "exact_match"), ("Levenshtein↓", "levenshtein"),
    ("MACCS FTS↑", "maccs_fts"), ("RDK FTS↑", "rdk_fts"), ("Morgan FTS↑", "morgan_fts"),
    ("FCD↓", "fcd"), ("Text2Mol↑", "text2mol"), ("Validity↑", "validity"),
]


class CaptioningTask(Task):
    name = "captioning"
    max_new_tokens = 512
    columns = CAPTION_COLUMNS

    def build_prompt(self, example: Dict[str, Any]) -> str:
        return CAPTIONING_PROMPT.format(smiles=example[SMILES])

    def postprocess(self, answer: str) -> str:
        return answer  # the caption text as-is

    def batch_length(self, example: Dict[str, Any], prompt: str) -> int:
        return len(str(example[SMILES]))

    def score_chunk(self, records, device="cpu"):
        from ...metrics.text import caption_metrics_per_example
        from ...metrics.text2mol.metric import text2mol_scores

        preds = [r.prediction for r in records]
        refs = [r.example[TEXT] for r in records]
        gold_smiles = [r.example[SMILES] for r in records]
        per = caption_metrics_per_example(preds, refs)
        t2m = text2mol_scores(gold_smiles, preds, device=device)
        for i, d in enumerate(per):
            d["text2mol"] = t2m[i] if t2m is not None else None
        return per

    def aggregate(self, records, scores, device="cpu"):
        from ...metrics.text import caption_metrics

        if scores is None:
            raise ValueError("captioning aggregate requires per-example scores")
        preds = [r.prediction for r in records]
        refs = [r.example[TEXT] for r in records]
        metrics = caption_metrics(preds, refs, per_example=scores)
        valid = [row["text2mol"] for row in scores if row.get("text2mol") is not None]
        metrics["text2mol"] = sum(valid) / len(valid) if valid else None
        return metrics


class Caption2SMILESTask(Task):
    name = "caption2smiles"
    max_new_tokens = 256
    columns = GEN_COLUMNS

    def build_prompt(self, example: Dict[str, Any]) -> str:
        return CAPTION2SMILES_PROMPT.format(description=example[TEXT])

    def postprocess(self, answer: str) -> str:
        return extract_smiles(answer)

    def evaluate(self, records: List[EvalRecord], device: str = "cpu") -> Dict[str, Any]:
        from ...metrics.fcd import fcd_metric
        from ...metrics.molecule import molgen_metrics
        from ...metrics.text2mol.metric import text2mol_score

        preds = [r.prediction for r in records]
        golds = [r.example[SMILES] for r in records]
        texts = [r.example[TEXT] for r in records]
        m = molgen_metrics(preds, golds)
        m["fcd"] = fcd_metric(preds, golds, device=device)
        # generation: generated molecule vs gold caption
        m["text2mol"] = text2mol_score(preds, texts, device=device)
        return m


class ChEBI20Benchmark(Benchmark):
    name = "chebi20"

    def load(self, split: str = "test", limit: Optional[int] = None) -> List[Dict[str, Any]]:
        from datasets import load_dataset

        ds = load_dataset(REPO, split=split)
        drop = [c for c in ("IMAGE",) if c in ds.column_names]
        if drop:
            ds = ds.remove_columns(drop)
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        return [dict(row) for row in ds]

    def tasks(self) -> Dict[str, Task]:
        return {"captioning": CaptioningTask(), "caption2smiles": Caption2SMILESTask()}


register_benchmark("chebi20", ChEBI20Benchmark)
