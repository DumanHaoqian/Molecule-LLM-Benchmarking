"""ChemCoTBench-V2: 31 formal-trace subtasks and three-layer evaluation."""
from __future__ import annotations

from functools import cached_property
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.benchmark import Benchmark
from ...core.registry import register_benchmark
from ...core.task import EvalRecord, Task
from .official import (
    UPSTREAM_COMMIT,
    OfficialV2Evaluator,
    build_prompts,
    load_released_records,
)


DATASET_REPO = "fresnellll/ChemCoTBench-V2"
DATASET_REVISION = "f0bb2fb00c97cb3257294a639e28f960f2da157e"
EXPECTED_SAMPLES = 5620
EXPECTED_SUBTASKS = 31

MODEL_INPUT_ALIASES = {
    "src_smiles",
    "indexed_smiles",
    "instruction",
    "smiles",
    "fg_name",
    "ring_name",
    "largest_scaffold",
    "ring_system_scaffold",
    "mol_smiles",
    "scaffold_smiles",
    "mutated",
    "permutated",
    "smiles_a",
    "smiles_b",
    "source_subtask",
    "src",
    "src_mol",
    "src_logp",
    "src_qed",
    "src_solubility",
    "src_drd",
    "src_jnk",
    "src_gsk",
}

TASKS = [
    ("mol_edit", "add_v2", "MolEdit/Add"),
    ("mol_edit", "delete_v2", "MolEdit/Delete"),
    ("mol_edit", "substitute_v2", "MolEdit/Substitute"),
    ("mol_und", "fg_detect", "MolUnd/Functional Group"),
    ("mol_und", "ring_count", "MolUnd/Ring Count"),
    ("mol_und", "murcko_scaffold", "MolUnd/Murcko Scaffold"),
    ("mol_und", "ring_sys_scaffold", "MolUnd/Ring-System Scaffold"),
    ("mol_und", "smiles_equivalent", "MolUnd/SMILES Equivalence"),
    ("rxn_pred", "forward", "RxnPred/Product-Level Prediction"),
    ("rxn_pred", "byproduct", "RxnPred/Product-Level Prediction"),
    ("rxn_pred", "nepp", "RxnPred/Product-Level Prediction"),
    ("rxn_pred", "retro", "RxnPred/Retrosynthesis"),
    ("rxn_pred", "rxn_template", "RxnPred/Template/Mechanism Reasoning"),
    ("rxn_pred", "mech_sel", "RxnPred/Template/Mechanism Reasoning"),
    ("rxn_pred", "rcr_catalyst", "RxnPred/Component Recommendation"),
    ("rxn_pred", "rcr_reagent", "RxnPred/Component Recommendation"),
    ("rxn_pred", "rcr_solvent", "RxnPred/Component Recommendation"),
    ("rxn_pred", "condition_ranking", "RxnPred/Condition Ranking"),
    ("rxn_pred", "yield_pred", "RxnPred/Yield Prediction"),
    ("mol_opt", "logp", "MolOpt/PhysChem-Single"),
    ("mol_opt", "qed", "MolOpt/PhysChem-Single"),
    ("mol_opt", "solubility", "MolOpt/PhysChem-Single"),
    ("mol_opt", "drd", "MolOpt/BioTarget-Single"),
    ("mol_opt", "jnk", "MolOpt/BioTarget-Single"),
    ("mol_opt", "gsk", "MolOpt/BioTarget-Single"),
    ("mol_opt", "logp_qed", "MolOpt/PhysChem-Dual"),
    ("mol_opt", "logp_solubility", "MolOpt/PhysChem-Dual"),
    ("mol_opt", "qed_solubility", "MolOpt/PhysChem-Dual"),
    ("mol_opt", "drd_logp", "MolOpt/BioTarget-Dual"),
    ("mol_opt", "drd_solubility", "MolOpt/BioTarget-Dual"),
    ("mol_opt", "gsk_logp", "MolOpt/BioTarget-Dual"),
]

# ChemDFM-v2 tokenizer profile over all 5,620 released reference traces.
# Each value is ceil_to_128(P99.9 reference tokens * 1.25).
MAX_NEW_TOKENS = {
    ("mol_edit", "add_v2"): 1280,
    ("mol_edit", "delete_v2"): 1536,
    ("mol_edit", "substitute_v2"): 1536,
    ("mol_und", "fg_detect"): 1152,
    ("mol_und", "ring_count"): 1280,
    ("mol_und", "murcko_scaffold"): 1664,
    ("mol_und", "ring_sys_scaffold"): 1280,
    ("mol_und", "smiles_equivalent"): 3072,
    ("rxn_pred", "forward"): 2176,
    ("rxn_pred", "byproduct"): 2560,
    ("rxn_pred", "nepp"): 3328,
    ("rxn_pred", "retro"): 2816,
    ("rxn_pred", "rxn_template"): 2048,
    ("rxn_pred", "mech_sel"): 1792,
    ("rxn_pred", "rcr_catalyst"): 1280,
    ("rxn_pred", "rcr_reagent"): 1280,
    ("rxn_pred", "rcr_solvent"): 1280,
    ("rxn_pred", "condition_ranking"): 1152,
    ("rxn_pred", "yield_pred"): 1024,
    ("mol_opt", "logp"): 1536,
    ("mol_opt", "qed"): 1920,
    ("mol_opt", "solubility"): 1152,
    ("mol_opt", "drd"): 1536,
    ("mol_opt", "jnk"): 1408,
    ("mol_opt", "gsk"): 1408,
    ("mol_opt", "logp_qed"): 1152,
    ("mol_opt", "logp_solubility"): 1536,
    ("mol_opt", "qed_solubility"): 1536,
    ("mol_opt", "drd_logp"): 1408,
    ("mol_opt", "drd_solubility"): 1408,
    ("mol_opt", "gsk_logp"): 1408,
}

COLUMNS = [
    ("Parse Rate↑", "parse_rate"),
    ("Layer 1", "primary_layer1"),
    ("Layer 2↑", "layer2_score"),
    ("Layer 3-I↑", "layer3_type1"),
    ("Layer 3-II↑", "layer3_type2"),
]


def task_key(family: str, subtask: str) -> str:
    return f"{family}__{subtask}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ChemCoTV2Task(Task):
    columns = COLUMNS
    # The profiled budget covers the formal trace. Reasoning models also need
    # room for their model-native thinking wrapper before that trace.
    uses_model_reasoning_budget = True

    def __init__(
        self,
        family: str,
        subtask: str,
        reporting_task: str,
        data_root: Path,
    ):
        self.family = family
        self.subtask = subtask
        self.reporting_task = reporting_task
        self.name = task_key(family, subtask)
        self.data_root = data_root
        self.max_new_tokens = MAX_NEW_TOKENS[(family, subtask)]
        self._evaluator: OfficialV2Evaluator | None = None

    def _prompts(self, example: Dict[str, Any]) -> tuple[str, str]:
        model_example = {k: v for k, v in example.items() if k != "_process_reference"}
        return build_prompts(self.family, self.subtask, model_example, self.data_root)

    def build_prompt(self, example: Dict[str, Any]) -> str:
        return self._prompts(example)[1]

    def build_system_prompt(self, example: Dict[str, Any]) -> str:
        return self._prompts(example)[0]

    def postprocess(self, answer: str) -> str:
        return answer.strip()

    def batch_length(self, example: Dict[str, Any], prompt: str) -> int:
        return len(prompt)

    def artifact_identity(self) -> Dict[str, Any]:
        return {
            **super().artifact_identity(),
            "official_evaluator_commit": UPSTREAM_COMMIT,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "output_budget_basis": "ChemDFM-v2 P99.9 reference tokens * 1.25, rounded to 128",
        }

    @property
    def evaluator(self) -> OfficialV2Evaluator:
        if self._evaluator is None:
            self._evaluator = OfficialV2Evaluator(
                self.family, self.subtask, self.data_root
            )
        return self._evaluator

    def score_chunk(self, records, device="cpu"):
        return self.evaluator.score(records)

    def aggregate(self, records, scores, device="cpu"):
        if scores is None:
            raise ValueError("ChemCoTBench-V2 requires per-example scores")
        return self.evaluator.summarize(records, scores)


class ChemCoTBenchV2(Benchmark):
    name = "chemcotbench-v2"

    def __init__(self, data_root: str | Path | None = None):
        configured = data_root or os.environ.get("CHEMCOTBENCH_V2_DATA_DIR")
        if configured is None:
            configured = Path(__file__).resolve().parents[3] / "resources" / "chemcotbench" / "v2"
        self.data_root = Path(configured).expanduser().resolve()
        self._loaded_identity: Dict[str, Dict[str, Any]] = {}

    @cached_property
    def manifest(self) -> Dict[str, Any]:
        path = self.data_root / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(
                f"ChemCoTBench-V2 data not found at {self.data_root}. "
                "Run: python scripts/fetch_chemcotbench.py --version v2"
            )
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("total_samples") != EXPECTED_SAMPLES:
            raise ValueError("unexpected ChemCoTBench-V2 sample count in manifest")
        if len(manifest.get("files", [])) != EXPECTED_SUBTASKS:
            raise ValueError("unexpected ChemCoTBench-V2 subtask count in manifest")
        return manifest

    def _entry(self, family: str, subtask: str) -> Dict[str, Any]:
        for entry in self.manifest["files"]:
            if entry["family"] == family and entry["subtask"] == subtask:
                return entry
        raise KeyError(f"missing manifest entry for {family}/{subtask}")

    def tasks(self) -> Dict[str, Task]:
        return {
            task_key(family, subtask): ChemCoTV2Task(
                family, subtask, reporting, self.data_root
            )
            for family, subtask, reporting in TASKS
        }

    def load(self, split: str = "test", limit: Optional[int] = None) -> List[Dict[str, Any]]:
        examples = []
        for family, subtask, _ in TASKS:
            examples.extend(self.load_task(task_key(family, subtask), split, limit))
        return examples

    def load_task(
        self, task_name: str, split: str = "test", limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if split != "test":
            raise ValueError("ChemCoTBench-V2 only provides the test split")
        task = self.tasks().get(task_name)
        if task is None:
            raise KeyError(f"unknown ChemCoTBench-V2 task: {task_name}")
        entry = self._entry(task.family, task.subtask)
        raw_path = self.data_root / entry["raw_file"]
        process_path = self.data_root / entry["process_file"]
        for path in (raw_path, process_path):
            if not path.exists():
                raise FileNotFoundError(f"manifest references missing file: {path}")
        with raw_path.open(encoding="utf-8") as handle:
            raw_records = json.load(handle)
        with process_path.open(encoding="utf-8") as handle:
            process_records = json.load(handle)
        expected_n = int(entry["n_samples"])
        if len(raw_records) != expected_n or len(process_records) != expected_n:
            raise ValueError(
                f"count mismatch for {task.family}/{task.subtask}: "
                f"raw={len(raw_records)} process={len(process_records)} expected={expected_n}"
            )
        raw_ids = [record.get("anonymous_sample_id") for record in raw_records]
        process_ids = [record.get("anonymous_sample_id") for record in process_records]
        if raw_ids != process_ids or len(set(raw_ids)) != expected_n:
            raise ValueError(f"raw/process ID alignment failure for {task_name}")

        released = load_released_records(task.family, task.subtask, self.data_root)
        released_by_id = {record["anonymous_sample_id"]: record for record in released}
        examples = []
        for raw, process in zip(raw_records, process_records):
            sample_id = raw["anonymous_sample_id"]
            merged = released_by_id[sample_id]
            model_record = dict(merged)
            parsed_reference = process.get("parsed_reference_state") or {}
            for key in parsed_reference:
                if key not in raw and key not in MODEL_INPUT_ALIASES:
                    model_record.pop(key, None)
            for key in (
                "formal_cot_trace",
                "parsed_reference_state",
                "verifier_checks",
                "raw_output",
                "raw_output_steps",
            ):
                model_record.pop(key, None)
            model_record.update(raw)
            if task.subtask == "smiles_equivalent" and not model_record.get("smiles"):
                parsed = process.get("parsed_reference_state") or {}
                smiles_a = parsed.get("step1_canonical_a")
                smiles_b = parsed.get("step2_canonical_b")
                if not smiles_a or not smiles_b:
                    raise ValueError(
                        f"cannot reconstruct missing model inputs for {sample_id}"
                    )
                source_subtask = model_record.get("source_subtask", "permutated")
                model_record["smiles"] = smiles_a
                model_record[source_subtask] = smiles_b
                model_record["input_repair"] = {
                    "reason": "upstream raw record omitted both model-facing SMILES",
                    "source": "parsed_reference_state canonical_a/canonical_b",
                    "dataset_revision": DATASET_REVISION,
                }
            model_record["dataset_example_id"] = sample_id
            model_record["_process_reference"] = merged
            examples.append(model_record)
        if limit is not None:
            examples = examples[: min(limit, len(examples))]
        self._loaded_identity[task_name] = {
            "manifest_sha256": _sha256(self.data_root / "manifest.json"),
            "raw_sha256": _sha256(raw_path),
            "process_sha256": _sha256(process_path),
            "n": len(examples),
        }
        return examples

    def artifact_identity(self, task_name: str) -> Dict[str, Any]:
        receipt = self.data_root / ".molbench-source.json"
        return {
            **super().artifact_identity(task_name),
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "official_evaluator_commit": UPSTREAM_COMMIT,
            "snapshot_receipt_sha256": _sha256(receipt) if receipt.exists() else None,
            **self._loaded_identity.get(task_name, {}),
        }

    def aggregate_task_results(self, task_results: Dict[str, Any]) -> Dict[str, Any]:
        from .reporting import aggregate_reporting

        return aggregate_reporting(task_results)


register_benchmark("chemcotbench-v2", ChemCoTBenchV2)
