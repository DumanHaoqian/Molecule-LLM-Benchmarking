"""Helpers for loading the released ChemCoTBench-V2 benchmark package."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


SOFTWARE_ROOT = Path(__file__).resolve().parents[2]
STEP_SMILES_RE = re.compile(r'SMILES\("([^"]+)"\)')


def _candidate_data_roots() -> list[Path]:
    env_path = os.environ.get("CHEMCOT_DATA_DIR") or os.environ.get("CHEMCOTBENCH_DATA_DIR")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            SOFTWARE_ROOT / "data",
            SOFTWARE_ROOT / "anonymous_data",
            SOFTWARE_ROOT.parent / "anonymous_data",
            SOFTWARE_ROOT.parent / "data",
        ]
    )
    return candidates


def resolve_data_root(required: bool = True) -> Path | None:
    """Return the released data root."""
    for candidate in _candidate_data_roots():
        if (candidate / "manifest.json").exists():
            return candidate
    if required:
        searched = "\n".join(f"  - {path}" for path in _candidate_data_roots())
        raise FileNotFoundError(
            "Could not locate the ChemCoTBench-V2 data package. Set "
            "CHEMCOT_DATA_DIR to the directory containing manifest.json, or "
            "place the released data at ../anonymous_data or ./data.\n"
            f"Searched:\n{searched}"
        )
    return None


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest(data_root: Path | None = None) -> dict[str, Any]:
    root = data_root or resolve_data_root()
    assert root is not None
    return load_json(root / "manifest.json")


def manifest_entry(task: str, subtask: str, data_root: Path | None = None) -> dict[str, Any]:
    manifest = load_manifest(data_root)
    for entry in manifest["files"]:
        if entry["family"] == task and entry["subtask"] == subtask:
            return entry
    raise KeyError(f"No released-data manifest entry for {task}/{subtask}")


def trace_to_raw_output(record: dict[str, Any]) -> str:
    """Reconstruct unified-step text from a released formal_cot_trace."""
    steps = [
        step.get("step_text", "")
        for step in record.get("formal_cot_trace", [])
        if isinstance(step, dict) and step.get("step_text")
    ]
    answer = (
        record.get("answer_smiles")
        or record.get("answer_smi")
        or record.get("answer_letter")
        or record.get("answer_yield")
        or record.get("answer")
    )
    if answer is not None:
        steps.append(f"Answer: {answer}")
    return "\n\n".join(steps)


def _merge_raw_process(raw_records: list[dict[str, Any]], process_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_by_id = {
        record.get("anonymous_sample_id"): record
        for record in raw_records
        if record.get("anonymous_sample_id")
    }
    merged = []
    for process in process_records:
        base = dict(raw_by_id.get(process.get("anonymous_sample_id"), {}))
        base.update(process)
        merged.append(base)
    return merged


def _infer_source_smiles_from_indexed(record: dict[str, Any]) -> None:
    indexed = record.get("indexed_smiles")
    if not indexed or record.get("src_smiles"):
        return
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(indexed)
        if mol is None:
            return
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        record["src_smiles"] = Chem.MolToSmiles(mol)
    except Exception:
        return


def _infer_aliases_from_trace(record: dict[str, Any]) -> None:
    raw_output = record.get("raw_output", "")
    smiles_values = STEP_SMILES_RE.findall(raw_output)
    task = record.get("task_family")
    subtask = record.get("subtask")

    if task == "mol_edit":
        _infer_source_smiles_from_indexed(record)

    if task == "mol_und":
        if subtask in {"fg_detect", "ring_count", "murcko_scaffold"} and smiles_values:
            record.setdefault("smiles", smiles_values[0])
        elif subtask == "ring_sys_scaffold":
            if smiles_values:
                record.setdefault("smiles", smiles_values[0])
                record.setdefault("mol_smiles", smiles_values[0])
            if len(smiles_values) >= 2:
                record.setdefault("scaffold_smiles", smiles_values[1])
                record.setdefault("ring_system_scaffold", smiles_values[1])
        elif subtask == "smiles_equivalent" and len(smiles_values) >= 2:
            record.setdefault("smiles", smiles_values[0])
            record.setdefault("smiles_a", smiles_values[0])
            record.setdefault("smiles_b", smiles_values[1])
            source = record.get("source_subtask")
            if source == "permutated":
                record.setdefault("permutated", smiles_values[1])
            else:
                record.setdefault("mutated", smiles_values[1])

    if task == "mol_opt" and smiles_values:
        record.setdefault("src", smiles_values[0])
        record.setdefault("src_mol", smiles_values[0])
        record.setdefault("tgt", record.get("answer_smiles", ""))
        record.setdefault("tgt_mol", record.get("answer_smiles", ""))


def load_released_records(task: str, subtask: str, data_root: Path | None = None) -> list[dict[str, Any]]:
    """Load model-facing raw records merged with process references."""
    root = data_root or resolve_data_root()
    assert root is not None
    entry = manifest_entry(task, subtask, root)
    raw_records = load_json(root / entry["raw_file"])
    process_records = load_json(root / entry["process_file"])
    records = _merge_raw_process(raw_records, process_records)
    for record in records:
        trace = record.get("formal_cot_trace") or []
        record["raw_output"] = record.get("raw_output") or trace_to_raw_output(record)
        record["raw_output_steps"] = [
            step.get("step_text", "")
            for step in trace
            if isinstance(step, dict) and step.get("step_text")
        ]
        record.setdefault("api_success", True)
        parsed_state = record.get("parsed_reference_state") or {}
        if isinstance(parsed_state, dict):
            for key, value in parsed_state.items():
                record.setdefault(key, value)
        _infer_aliases_from_trace(record)
    return records
