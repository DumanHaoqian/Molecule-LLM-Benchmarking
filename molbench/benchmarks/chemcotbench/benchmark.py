"""ChemCoTBench V1 task-scoped dataset and official-metric adapter."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from ...core.benchmark import Benchmark
from ...core.registry import register_benchmark
from ...core.task import Task
from ...utils.chem import extract_smiles


DATASET_REPO = "IDEA-AI4S/ChemCoTBench"
DATASET_REVISION = "4cfab96c6f511a504519e2cc003521b6afbac338"
EVALUATOR_COMMIT = "ac04f91468cb2ec67edb5d597a834e7daddc228a"

SPECS = [
    ("mol_und", "fg_count", "chemcotbench/mol_und/fg_count.json", 100, "count"),
    ("mol_und", "ring_count", "chemcotbench/mol_und/ring_count.json", 20, "count"),
    ("mol_und", "murcko_scaffold", "chemcotbench/mol_und/Murcko_scaffold.json", 40, "scaffold"),
    ("mol_und", "ring_system_scaffold", "chemcotbench/mol_und/ring_system_scaffold.json", 60, "boolean"),
    ("mol_und", "equivalence", "chemcotbench/mol_und/equivalence.json", 100, "boolean"),
    ("mol_edit", "add", "chemcotbench/mol_edit/add.json", 20, "edit"),
    ("mol_edit", "delete", "chemcotbench/mol_edit/delete.json", 20, "edit"),
    ("mol_edit", "sub", "chemcotbench/mol_edit/sub.json", 60, "edit"),
    ("mol_opt", "drd", "chemcotbench/mol_opt/drd.json", 100, "optimization"),
    ("mol_opt", "gsk", "chemcotbench/mol_opt/gsk.json", 100, "optimization"),
    ("mol_opt", "jnk", "chemcotbench/mol_opt/jnk.json", 100, "optimization"),
    ("mol_opt", "logp", "chemcotbench/mol_opt/logp.json", 100, "optimization"),
    ("mol_opt", "qed", "chemcotbench/mol_opt/qed.json", 100, "optimization"),
    ("mol_opt", "solubility", "chemcotbench/mol_opt/solubility.json", 100, "optimization"),
    ("reaction", "fs", "chemcotbench/reaction/fs.json", 100, "molecule"),
    ("reaction", "retro", "chemcotbench/reaction/retro.json", 100, "molecule"),
    ("reaction", "rcr", "chemcotbench/reaction/rcr.json", 90, "molecule"),
    ("reaction", "nepp", "chemcotbench/reaction/nepp.json", 85, "molecule"),
    ("reaction", "mechsel", "chemcotbench/reaction/mechsel.json", 100, "choice"),
]


def task_key(family: str, subtask: str) -> str:
    return f"{family}__{subtask}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    cleaned = cleaned.replace("```json", "").replace("```", "")
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _field(text: str, names: List[str]) -> Any:
    value = _json_object(text)
    lowered = {str(key).lower(): item for key, item in value.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _parse_bool(text: str) -> str | None:
    value = _field(text, ["output", "answer", "equivalent"])
    candidate = str(value if value is not None else text).strip().lower()
    yes = re.search(r"\b(yes|same|true)\b", candidate)
    no = re.search(r"\b(no|different|false)\b", candidate)
    if yes and not no:
        return "yes"
    if no and not yes:
        return "no"
    return None


def _parse_int(text: str) -> int | None:
    value = _field(text, ["count", "answer", "output"])
    candidate = str(value if value is not None else text)
    match = re.search(r"-?\d+", candidate)
    return int(match.group()) if match else None


def _parse_choice(text: str) -> str | None:
    value = _field(text, ["answer", "output", "choice", "selected route"])
    candidate = str(value if value is not None else text).upper()
    match = re.search(r"\b([A-Z])\b", candidate)
    return match.group(1) if match else None


def _parse_smiles(text: str, kind: str) -> str:
    names = [
        "output",
        "output scaffold",
        "answer",
        "smiles",
        "predicted smiles",
        "pred_smi",
    ]
    if kind == "optimization":
        names.insert(0, "final target molecule")
    if kind == "molecule":
        names = ["major product", "reactants", "product", *names]
    value = _field(text, names)
    if value is not None:
        if isinstance(value, list):
            return ".".join(str(item).strip() for item in value)
        return str(value).strip()
    return extract_smiles(text)


def _meta(record: Dict[str, Any]) -> Dict[str, Any]:
    value = record.get("meta", {})
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = {}
    return value if isinstance(value, dict) else {}


def _query_source_smiles(query: str, family: str) -> str:
    if family == "mol_edit":
        match = re.search(
            r"Input Molecule:\s*(.+?),\s*Functional Group(?:s|\s+to)",
            query,
            flags=re.IGNORECASE,
        )
    else:
        match = re.search(r"Source Molecule:\s*([^\r\n]+)", query, flags=re.IGNORECASE)
    return match.group(1).strip().rstrip(".") if match else ""


def _reference_smiles(record: Dict[str, Any], subtask: str) -> str:
    for key in ("reference", "gt_smiles", "answer_smiles", "target"):
        if record.get(key):
            return str(record[key])
    if subtask == "murcko_scaffold" and record.get("largest_scaffold"):
        return str(record["largest_scaffold"])
    if subtask == "retro" and record.get("reactants"):
        value = record["reactants"]
        return ".".join(value) if isinstance(value, list) else str(value)
    raw_gt = record.get("gt")
    if raw_gt:
        if isinstance(raw_gt, str):
            try:
                raw_gt = json.loads(raw_gt)
            except json.JSONDecodeError:
                return raw_gt
        if isinstance(raw_gt, dict):
            for key in ("Major Product", "Reactants", "Product", "output"):
                if raw_gt.get(key):
                    value = raw_gt[key]
                    return ".".join(value) if isinstance(value, list) else str(value)
    meta = _meta(record)
    if subtask == "fs" and meta.get("products"):
        return ".".join(meta["products"])
    return ""


def _reference_bool(record: Dict[str, Any], subtask: str) -> str | None:
    for key in ("gt", "answer", "output", "label"):
        if record.get(key) is not None:
            parsed = _parse_bool(str(record[key]))
            if parsed:
                return parsed
    marker = str(record.get("task", record.get("subtask", ""))).lower()
    if subtask == "equivalence":
        return "no" if "mutat" in marker else "yes" if "permut" in marker else None
    return None


def _reference_count(record: Dict[str, Any], subtask: str) -> int | None:
    keys = ("fg_num", "count", "gt_count", "answer", "gt") if subtask == "fg_count" else (
        "count",
        "gt_count",
        "answer",
        "gt",
    )
    for key in keys:
        if record.get(key) is not None:
            try:
                return int(record[key])
            except (TypeError, ValueError):
                continue
    return None


def _prompt(record: Dict[str, Any], family: str, subtask: str, kind: str) -> str:
    if record.get("query"):
        return str(record["query"])
    if kind == "edit":
        instruction = record.get("Instruction") or (
            f"Modify {record.get('molecule', '')} for the requested {subtask} operation."
        )
        return f'{instruction}\nReturn only JSON: {{"output": "<modified SMILES>"}}'
    if family == "mol_opt":
        return (
            f"Optimize the molecule {record.get('src_smiles', '')} to increase "
            f"{subtask}. Preserve its main scaffold.\n"
            'Return only JSON: {"Final Target Molecule": "<SMILES>"}'
        )
    if subtask == "fg_count":
        return (
            f"Count {record.get('fg_name', record.get('fg_label', 'the requested functional group'))} "
            f"in molecule {record.get('smiles', '')}.\n"
            'Return only JSON: {"count": <integer>}'
        )
    if subtask == "ring_count":
        return (
            f"Count occurrences of ring structure {record.get('ring', '')} in "
            f"molecule {record.get('smiles', '')}.\n"
            'Return only JSON: {"count": <integer>}'
        )
    if subtask == "murcko_scaffold":
        return (
            f"Extract the Bemis-Murcko scaffold of {record.get('smiles', '')}.\n"
            'Return only JSON: {"output": "<scaffold SMILES>"}'
        )
    if subtask == "ring_system_scaffold":
        return (
            f"Determine whether ring-system scaffold {record.get('ring_system_scaffold', '')} "
            f"is contained in molecule {record.get('smiles', '')}.\n"
            'Return only JSON: {"output": "Yes or No"}'
        )
    if subtask == "equivalence":
        second = record.get("mutated", record.get("permutated", record.get("smiles_b", "")))
        return (
            f"Determine whether these SMILES represent the same molecule: "
            f"{record.get('smiles', record.get('smiles_a', ''))} and {second}.\n"
            'Return only JSON: {"output": "Yes or No"}'
        )
    meta = _meta(record)
    if subtask == "fs":
        return (
            f"Predict the major product for reactants {meta.get('reactants', record.get('reactants', ''))} "
            f"with reagents {meta.get('reagents', record.get('reagents', ''))}.\n"
            'Return only JSON: {"Major Product": "<SMILES>"}'
        )
    if subtask == "retro":
        return (
            f"Predict reactants for product {record.get('products', meta.get('products', ''))} "
            f"with reagents {record.get('reagents', meta.get('reagents', ''))}.\n"
            'Return only JSON: {"Reactants": "<dot-separated SMILES>"}'
        )
    return (
        "Solve the following chemistry task and obey its requested output schema.\n"
        + json.dumps(record, ensure_ascii=False, sort_keys=True)
    )


class ChemCoTV1Task(Task):
    def __init__(
        self, family: str, subtask: str, kind: str, data_root: Path
    ):
        self.family = family
        self.subtask = subtask
        self.name = task_key(family, subtask)
        self.kind = kind
        self.data_root = data_root
        self.max_new_tokens = 1024 if family == "reaction" else 512
        if kind == "count":
            self.columns = [("Parse↑", "parse_rate"), ("Accuracy↑", "accuracy"), ("MAE↓", "mae")]
        elif kind in {"boolean", "choice"}:
            self.columns = [("Parse↑", "parse_rate"), ("Accuracy↑", "accuracy")]
        elif kind == "scaffold":
            self.columns = [
                ("Validity↑", "validity"),
                ("Scaffold Hard↑", "scaffold_hard"),
                ("Scaffold Soft↑", "scaffold_soft"),
            ]
        elif kind == "edit":
            self.columns = [("Validity↑", "validity"), ("SR↑", "success_rate"), ("EM↑", "exact_match"), ("Morgan↑", "morgan_sims")]
        elif kind == "optimization":
            self.columns = [("Validity↑", "validity"), ("SR↑", "success_rate"), ("Best↑", "best_rate"), ("Mean Δ↑", "mean_improvement"), ("Scaffold↑", "scaffold_hard")]
        else:
            self.columns = [("Validity↑", "validity"), ("EM↑", "exact_match"), ("BLEU↑", "bleu"), ("Morgan↑", "morgan_sims")]

    def build_prompt(self, example: Dict[str, Any]) -> str:
        return _prompt(example, self.family, self.subtask, self.kind)

    def postprocess(self, answer: str) -> Any:
        if self.kind == "count":
            return _parse_int(answer)
        if self.kind == "boolean":
            return _parse_bool(answer)
        if self.kind == "choice":
            return _parse_choice(answer)
        return _parse_smiles(answer, self.kind)

    def artifact_identity(self) -> Dict[str, Any]:
        return {
            **super().artifact_identity(),
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "official_evaluator_commit": EVALUATOR_COMMIT,
            "metric_kind": self.kind,
        }

    def score_chunk(self, records, device="cpu"):
        from ...metrics.chemcotbench_v1.metrics import (
            edit_score,
            molecule_pair,
            optimization_score,
            scaffold_pair,
        )

        scores = []
        for record in records:
            example = record.example
            if self.kind == "count":
                reference = _reference_count(example, self.subtask)
                prediction = record.prediction
                scores.append(
                    {
                        "parsed": prediction is not None,
                        "correct": float(prediction == reference) if prediction is not None and reference is not None else 0.0,
                        "absolute_error": abs(prediction - reference) if prediction is not None and reference is not None else 0.0,
                    }
                )
            elif self.kind == "boolean":
                reference = _reference_bool(example, self.subtask)
                scores.append(
                    {
                        "parsed": record.prediction is not None,
                        "correct": float(record.prediction == reference) if reference is not None else 0.0,
                    }
                )
            elif self.kind == "choice":
                reference = str(example.get("gt", example.get("answer", ""))).strip().upper()
                scores.append(
                    {
                        "parsed": record.prediction is not None,
                        "correct": float(record.prediction == reference),
                    }
                )
            elif self.kind == "edit":
                score = edit_score(
                    str(example.get("molecule", "")),
                    str(record.prediction or ""),
                    self.subtask,
                    example.get("added_group"),
                    example.get("removed_group"),
                )
                score.update(
                    molecule_pair(
                        str(record.prediction or ""), _reference_smiles(example, self.subtask)
                    )
                )
                scores.append(score)
            elif self.kind == "optimization":
                scores.append(
                    optimization_score(
                        str(example.get("src_smiles", "")),
                        str(record.prediction or ""),
                        self.subtask,
                    )
                )
            elif self.kind == "scaffold":
                scores.append(
                    scaffold_pair(
                        str(record.prediction or ""),
                        _reference_smiles(example, self.subtask),
                    )
                )
            else:
                scores.append(
                    molecule_pair(
                        str(record.prediction or ""), _reference_smiles(example, self.subtask)
                    )
                )
        return scores

    def aggregate(self, records, scores, device="cpu"):
        from ...metrics.chemcotbench_v1.metrics import (
            aggregate_optimization,
            aggregate_molecule_pairs,
            mean_scores,
        )

        if scores is None:
            raise ValueError("ChemCoTBench V1 requires per-example scores")
        if self.kind == "count":
            parsed = [score for score in scores if score.get("parsed")]
            means = mean_scores(parsed, ["correct", "absolute_error"])
            return {"n": len(scores), "parse_rate": len(parsed) / len(scores) if scores else 0.0, "accuracy": means["correct"], "mae": means["absolute_error"]}
        if self.kind in {"boolean", "choice"}:
            parsed = [score for score in scores if score.get("parsed")]
            means = mean_scores(parsed, ["correct"])
            return {"n": len(scores), "parse_rate": len(parsed) / len(scores) if scores else 0.0, "accuracy": means["correct"]}
        if self.kind == "edit":
            means = mean_scores(scores, ["valid", "success", "exact_match", "morgan_sims"])
            return {"n": len(scores), "validity": means["valid"], "success_rate": means["success"], "exact_match": means["exact_match"], "morgan_sims": means["morgan_sims"]}
        if self.kind == "optimization":
            return aggregate_optimization(scores, self.subtask)
        if self.kind == "scaffold":
            means = mean_scores(scores, ["valid", "scaffold_hard", "scaffold_soft"])
            return {
                "n": len(scores),
                "validity": means["valid"],
                "scaffold_hard": means["scaffold_hard"],
                "scaffold_soft": means["scaffold_soft"],
            }
        predictions = [str(record.prediction or "") for record in records]
        references = [_reference_smiles(record.example, self.subtask) for record in records]
        return aggregate_molecule_pairs(predictions, references, scores)


class ChemCoTBenchV1(Benchmark):
    name = "chemcotbench"

    def __init__(self, data_root: str | Path | None = None):
        configured = data_root or os.environ.get("CHEMCOTBENCH_V1_DATA_DIR")
        if configured is None:
            configured = Path(__file__).resolve().parents[3] / "resources" / "chemcotbench" / "v1"
        self.data_root = Path(configured).expanduser().resolve()
        self._loaded_identity: Dict[str, Dict[str, Any]] = {}
        self._specs = {task_key(family, subtask): (family, subtask, path, count, kind) for family, subtask, path, count, kind in SPECS}

    def tasks(self) -> Dict[str, Task]:
        return {
            key: ChemCoTV1Task(family, subtask, kind, self.data_root)
            for key, (family, subtask, _, _, kind) in self._specs.items()
        }

    def load(self, split: str = "test", limit: Optional[int] = None) -> List[Dict[str, Any]]:
        rows = []
        for key in self._specs:
            rows.extend(self.load_task(key, split, limit))
        return rows

    def load_task(
        self, task_name: str, split: str = "test", limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if split != "test":
            raise ValueError("ChemCoTBench V1 only provides the test split")
        if task_name not in self._specs:
            raise KeyError(f"unknown ChemCoTBench V1 task: {task_name}")
        family, subtask, relative, expected, _ = self._specs[task_name]
        path = self.data_root / relative
        if not path.exists():
            raise FileNotFoundError(
                f"ChemCoTBench V1 data not found at {path}. The dataset is gated; "
                "accept its Hugging Face terms, set HF_TOKEN, then run "
                "python scripts/fetch_chemcotbench.py --version v1"
            )
        with path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list) or len(rows) != expected:
            raise ValueError(f"unexpected row count for {task_name}: {len(rows)} != {expected}")
        examples = []
        seen = set()
        for index, row in enumerate(rows):
            example = dict(row)
            metadata = _meta(example)
            query = str(example.get("query", ""))
            if family == "mol_edit":
                example["molecule"] = _query_source_smiles(query, family)
                for key in ("reference", "added_group", "removed_group"):
                    if metadata.get(key) is not None:
                        example[key] = metadata[key]
            elif family == "mol_opt":
                example["src_smiles"] = _query_source_smiles(query, family)
            native_id = str(example.get("id") or f"{family}.{subtask}.{index:04d}")
            if native_id in seen:
                raise ValueError(f"duplicate native ID in {path}: {native_id}")
            seen.add(native_id)
            example["dataset_example_id"] = native_id
            example["task_family"] = family
            example["subtask"] = subtask
            examples.append(example)
        if limit is not None:
            examples = examples[: min(limit, len(examples))]
        self._loaded_identity[task_name] = {"data_sha256": _sha256(path), "n": len(examples)}
        return examples

    def artifact_identity(self, task_name: str) -> Dict[str, Any]:
        receipt = self.data_root / ".molbench-source.json"
        return {
            **super().artifact_identity(task_name),
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "official_evaluator_commit": EVALUATOR_COMMIT,
            "snapshot_receipt_sha256": _sha256(receipt) if receipt.exists() else None,
            **self._loaded_identity.get(task_name, {}),
        }


register_benchmark("chemcotbench", ChemCoTBenchV1)
