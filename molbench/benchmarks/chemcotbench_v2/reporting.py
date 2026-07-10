"""Paper-facing 18-task aggregation for ChemCoTBench-V2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

from .benchmark import task_key


@dataclass(frozen=True)
class ReportingSpec:
    family: str
    name: str
    subtasks: tuple[tuple[str, str], ...]
    layer1_label: str
    layer1_kind: str


REPORTING_TASKS = [
    ReportingSpec("MolEdit", "Add", (("mol_edit", "add_v2"),), "Exact Acc ↑", "exact_acc"),
    ReportingSpec("MolEdit", "Delete", (("mol_edit", "delete_v2"),), "Exact Acc ↑", "exact_acc"),
    ReportingSpec("MolEdit", "Substitute", (("mol_edit", "substitute_v2"),), "Exact Acc ↑", "exact_acc"),
    ReportingSpec("MolUnd", "Functional Group", (("mol_und", "fg_detect"),), "MAE ↓", "mae"),
    ReportingSpec("MolUnd", "Ring Count", (("mol_und", "ring_count"),), "MAE ↓", "mae"),
    ReportingSpec("MolUnd", "Murcko Scaffold", (("mol_und", "murcko_scaffold"),), "Tanimoto ↑", "tanimoto"),
    ReportingSpec("MolUnd", "Ring-System Scaffold", (("mol_und", "ring_sys_scaffold"),), "Exact Acc ↑", "exact_acc"),
    ReportingSpec("MolUnd", "SMILES Equivalence", (("mol_und", "smiles_equivalent"),), "Exact Acc ↑", "exact_acc"),
    ReportingSpec("RxnPred", "Product-Level Prediction", (("rxn_pred", "forward"), ("rxn_pred", "byproduct"), ("rxn_pred", "nepp")), "Top-1 Acc ↑", "top1"),
    ReportingSpec("RxnPred", "Retrosynthesis", (("rxn_pred", "retro"),), "Top-1 Acc ↑", "top1"),
    ReportingSpec("RxnPred", "Template/Mechanism Reasoning", (("rxn_pred", "rxn_template"), ("rxn_pred", "mech_sel")), "Top-1 Acc ↑", "top1"),
    ReportingSpec("RxnPred", "Component Recommendation", (("rxn_pred", "rcr_catalyst"), ("rxn_pred", "rcr_reagent"), ("rxn_pred", "rcr_solvent")), "Top-1 Acc ↑", "top1"),
    ReportingSpec("RxnPred", "Condition Ranking", (("rxn_pred", "condition_ranking"),), "Top-1 Acc ↑", "top1"),
    ReportingSpec("RxnPred", "Yield Prediction", (("rxn_pred", "yield_pred"),), "MAE ↓", "yield_mae"),
    ReportingSpec("MolOpt", "PhysChem-Single", (("mol_opt", "logp"), ("mol_opt", "qed"), ("mol_opt", "solubility")), "SR ↑", "sr"),
    ReportingSpec("MolOpt", "BioTarget-Single", (("mol_opt", "drd"), ("mol_opt", "jnk"), ("mol_opt", "gsk")), "SR ↑", "sr"),
    ReportingSpec("MolOpt", "PhysChem-Dual", (("mol_opt", "logp_qed"), ("mol_opt", "logp_solubility"), ("mol_opt", "qed_solubility")), "Dual-SR ↑", "dual_sr"),
    ReportingSpec("MolOpt", "BioTarget-Dual", (("mol_opt", "drd_logp"), ("mol_opt", "drd_solubility"), ("mol_opt", "gsk_logp")), "Dual-SR ↑", "dual_sr"),
]


def _weighted(values: Iterable[tuple[float | None, int]]) -> float | None:
    usable = [(float(value), int(n)) for value, n in values if value is not None and n > 0]
    if not usable:
        return None
    return sum(value * n for value, n in usable) / sum(n for _, n in usable)


def _layer1(summary: dict, kind: str) -> float | None:
    key = {
        "exact_acc": "exact_match_acc",
        "mae": "mean_mae",
        "yield_mae": "mae",
        "tanimoto": "avg_tanimoto",
        "top1": "top1_acc",
        "sr": "sr_pct",
        "dual_sr": "dual_sr_pct",
    }[kind]
    return summary.get("layer1", {}).get(key)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def aggregate_reporting(task_results: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    incomplete = []
    for spec in REPORTING_TASKS:
        keys = [task_key(family, subtask) for family, subtask in spec.subtasks]
        if any(key not in task_results for key in keys):
            incomplete.append(spec.name)
            continue
        by_key_model = {
            key: {row["model_key"]: row for row in task_results[key]} for key in keys
        }
        model_keys = set.intersection(*(set(rows_by_model) for rows_by_model in by_key_model.values()))
        for model_key in sorted(model_keys):
            entries = [by_key_model[key][model_key] for key in keys]
            summaries = [entry["metrics"]["official_summary"] for entry in entries]
            counts = [int(entry["metrics"]["n"]) for entry in entries]
            layer1 = _weighted(
                (_layer1(summary, spec.layer1_kind), n)
                for summary, n in zip(summaries, counts)
            )
            layer2 = _weighted(
                (
                    summary.get("layer2", {}).get(
                        "state_score", summary.get("layer2", {}).get("avg_state_score")
                    ),
                    n,
                )
                for summary, n in zip(summaries, counts)
            )
            if spec.family == "MolOpt":
                layer3 = {
                    "kind": "avg_step_score",
                    "value": _weighted(
                        (summary.get("layer3", {}).get("avg_step_score"), n)
                        for summary, n in zip(summaries, counts)
                    ),
                }
            else:
                layer3 = {
                    "kind": "type1_type2",
                    "type1": _weighted(
                        (summary.get("layer3", {}).get("type1", {}).get("all_pass_rate"), n)
                        for summary, n in zip(summaries, counts)
                    ),
                    "type2": _weighted(
                        (summary.get("layer3", {}).get("type2", {}).get("all_fields_match_rate"), n)
                        for summary, n in zip(summaries, counts)
                    ),
                }
            rows.append(
                {
                    "family": spec.family,
                    "reporting_task": spec.name,
                    "subtasks": keys,
                    "n": sum(counts),
                    "model_key": model_key,
                    "display_name": entries[0]["display_name"],
                    "layer1_label": spec.layer1_label,
                    "layer1": layer1,
                    "layer2_state_score": layer2,
                    "layer3": layer3,
                }
            )

    markdown_rows = []
    for row in rows:
        layer3 = row["layer3"]
        layer3_text = (
            _fmt(layer3.get("value"))
            if layer3["kind"] == "avg_step_score"
            else f"{_fmt(layer3.get('type1'))} / {_fmt(layer3.get('type2'))}"
        )
        markdown_rows.append(
            "| "
            + " | ".join(
                [
                    row["display_name"],
                    row["family"],
                    row["reporting_task"],
                    str(row["n"]),
                    f"{row['layer1_label']}: {_fmt(row['layer1'])}",
                    _fmt(row["layer2_state_score"]),
                    layer3_text,
                ]
            )
            + " |"
        )
    markdown = "\n".join(
        [
            "### chemcotbench-v2 — 18 reporting tasks",
            "",
            "| Method | Family | Reporting task | n | Layer 1 | Layer 2 | Layer 3 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            *markdown_rows,
        ]
    )
    return {
        "n_reporting_tasks": len(rows),
        "incomplete_reporting_tasks": incomplete,
        "rows": rows,
        "markdown": markdown,
    }
