"""Adapter around the pinned upstream ChemCoTBench-V2 evaluator."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import importlib
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List

from ...core.task import EvalRecord


UPSTREAM_COMMIT = "dcd35470de4096a1b10ee9ed6f072bcee983a9cc"
UPSTREAM_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "chemcotbench_v2"
MULTI_MOLOPT = {
    "logp_qed",
    "logp_solubility",
    "qed_solubility",
    "drd_logp",
    "drd_solubility",
    "gsk_logp",
}


def activate_upstream(data_root: Path) -> None:
    """Expose the pinned upstream packages under their original import names."""
    root = str(UPSTREAM_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    os.environ["CHEMCOT_DATA_DIR"] = str(data_root)
    loaded = sys.modules.get("evaluation")
    if loaded is not None:
        module_file = str(getattr(loaded, "__file__", ""))
        if module_file and not module_file.startswith(root):
            raise RuntimeError(
                "a different top-level 'evaluation' package is already imported: "
                f"{module_file}"
            )


@lru_cache(maxsize=None)
def _prompt_builder(family: str, subtask: str, data_root: str):
    activate_upstream(Path(data_root))
    if family == "mol_opt":
        prompt = importlib.import_module("evaluation.mol_opt.prompt")
        is_multi = subtask in MULTI_MOLOPT

        class MolOptPromptBuilder:
            system_prompt = prompt.build_system_prompt(subtask, is_multi)

            @staticmethod
            def build_user_prompt(record: dict) -> str:
                return prompt.build_user_prompt(subtask, is_multi, record)

        return MolOptPromptBuilder()
    module = importlib.import_module(f"evaluation.{family}.prompt_builder")
    return module.PromptBuilder(subtask)


def build_prompts(
    family: str, subtask: str, record: Dict[str, Any], data_root: Path
) -> tuple[str, str]:
    builder = _prompt_builder(family, subtask, str(data_root))
    system = builder.system_prompt
    if callable(system):
        system = system(record)
    return str(system), str(builder.build_user_prompt(record))


def load_released_records(
    family: str, subtask: str, data_root: Path
) -> List[Dict[str, Any]]:
    """Load the official raw/process pairing with upstream field aliases."""
    activate_upstream(data_root)
    released = importlib.import_module("evaluation.core.released_data")
    records = released.load_released_records(family, subtask, data_root)
    return [dict(record) for record in records]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _state_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    delta = {}
    for key, value in after.items():
        if key == "raw_output":
            continue
        safe_value = _json_safe(value)
        if key not in before or _json_safe(before[key]) != safe_value:
            delta[key] = safe_value
    return delta


def _first_failed_step(state: Dict[str, Any]) -> str | None:
    failed = []
    for key, value in state.items():
        if value is False and re.match(r"^[sS]\d+_", key):
            failed.append(key)
    return sorted(failed)[0] if failed else None


def _layer_views(state: Dict[str, Any]) -> tuple[dict, dict, dict]:
    layer1 = {
        key.removeprefix("layer1_"): value
        for key, value in state.items()
        if key.startswith("layer1_")
    }
    layer2 = {
        key.removeprefix("layer2_"): value
        for key, value in state.items()
        if key.startswith("layer2_")
    }
    layer3 = {
        "type1": {
            key: value
            for key, value in state.items()
            if key in {"all_pass", "layer3_type1_outcome"}
            or re.match(r"^[sS]\d+_", key)
        },
        "type2": {
            key: value for key, value in state.items() if key.startswith("gt_match_")
        },
    }
    for key, value in state.items():
        if key.startswith("layer3_") and key != "layer3_type1_outcome":
            layer3[key.removeprefix("layer3_")] = value
    return _json_safe(layer1), _json_safe(layer2), _json_safe(layer3)


class OfficialV2Evaluator:
    """Run the pinned parser and all three official evaluation layers."""

    def __init__(self, family: str, subtask: str, data_root: Path):
        self.family = family
        self.subtask = subtask
        self.data_root = data_root
        activate_upstream(data_root)
        if family == "mol_opt":
            from ...utils.tdc import install_rdkit_six_compat, oracle

            install_rdkit_six_compat()
            self.parser = importlib.import_module("evaluation.mol_opt.parser")
            self.layer1 = importlib.import_module("evaluation.mol_opt.evaluator_layer1")
            self.layer2 = importlib.import_module("evaluation.mol_opt.layer2_evaluator")
            self.layer3 = importlib.import_module("evaluation.mol_opt.evaluator_layer3")
            oracle_utils = importlib.import_module("evaluation.mol_opt.utils")
            if not hasattr(oracle_utils, "_molbench_uncached_get_oracle"):
                oracle_utils._molbench_uncached_get_oracle = oracle_utils.get_oracle

                @lru_cache(maxsize=None)
                def cached_get_oracle(prop: str):
                    oracle_name = oracle_utils.TDC_ORACLE_NAME.get(prop)
                    if oracle_name is not None:
                        return oracle(oracle_name)
                    return oracle_utils._molbench_uncached_get_oracle(prop)

                oracle_utils.get_oracle = cached_get_oracle
            # evaluator_layer1 imported get_oracle by value, so update that
            # binding as well. Each expensive TDC model is now loaded once.
            self.layer1.get_oracle = oracle_utils.get_oracle
            self.is_multi = subtask in MULTI_MOLOPT
        else:
            parser_module = importlib.import_module("evaluation.core.parser_adapter")
            self.parser = parser_module.ParserAdapter(family, subtask)
            layer1_name = f"evaluation.{family}.layer1_evaluator"
            layer2_name = f"evaluation.{family}.layer2_evaluator"
            layer3_name = f"evaluation.{family}.layer3_evaluator"
            self.layer1 = importlib.import_module(layer1_name)
            self.layer2 = importlib.import_module(layer2_name)
            self.layer3 = importlib.import_module(layer3_name).Layer3Evaluator(subtask)
            self.metrics = importlib.import_module(f"evaluation.{family}.metrics")

    def _parse(self, records: List[dict]) -> List[dict]:
        if self.family == "mol_opt":
            return self.parser.parse_batch(records)
        return self.parser.parse_batch(records)

    def _evaluate_layer3(self, records: List[dict], references: List[dict]) -> List[dict]:
        if self.family == "mol_opt":
            return self.layer3.evaluate_batch_layer3(records, references)
        return self.layer3.evaluate_batch(records)

    def _evaluate_layer1(self, records: List[dict]) -> List[dict]:
        for record in records:
            if self.family == "mol_edit":
                edit_type = self.subtask.replace("_v2", "")
                record.setdefault("edit_type", edit_type)
                record.setdefault("source_smiles", record.get("src_smiles", ""))
                record.update(self.layer1.evaluate_layer1(record, edit_type))
            elif self.family in {"mol_und", "rxn_pred"}:
                record.update(self.layer1.evaluate_layer1(record, self.subtask))
            elif self.is_multi:
                record.update(self.layer1.evaluate_multi_layer1(record, self.subtask))
            else:
                record.update(self.layer1.evaluate_single_layer1(record, self.subtask))
        return records

    def _evaluate_layer2(self, records: List[dict]) -> List[dict]:
        for record in records:
            if self.family == "mol_opt":
                record.update(self.layer2.evaluate_layer2(record))
            else:
                result = self.layer2.evaluate_record(record, self.subtask)
                for key, value in result.items():
                    record[f"layer2_{key}"] = value
        return records

    def score(self, eval_records: Iterable[EvalRecord]) -> List[Dict[str, Any]]:
        eval_records = list(eval_records)
        source_records = []
        references = []
        before_states = []
        for item in eval_records:
            source = deepcopy(item.example)
            reference = source.pop("_process_reference", {})
            source["raw_output"] = item.raw_output
            source["api_success"] = True
            # The public release intentionally removes difficulty labels, while a
            # few upstream verifier summaries still index this legacy field.
            source.setdefault("difficulty", "unknown")
            before_states.append(deepcopy(source))
            source_records.append(source)
            references.append(reference or deepcopy(source))
        parsed = self._parse(source_records)
        parsed = self._evaluate_layer3(parsed, references)
        parsed = self._evaluate_layer1(parsed)
        parsed = self._evaluate_layer2(parsed)

        scores = []
        for item, before, state in zip(eval_records, before_states, parsed):
            layer1, layer2, layer3 = _layer_views(state)
            finish_reason = item.generation_metadata.get("finish_reason")
            parse_ok = bool(state.get("parse_ok"))
            parse_status = "ok" if parse_ok else "error"
            if finish_reason == "length":
                parse_status = "incomplete_length"
            scores.append(
                {
                    "parse_status": parse_status,
                    "parse_ok": parse_ok,
                    "layer1": layer1,
                    "layer2": layer2,
                    "layer3": layer3,
                    "first_failed_step": _first_failed_step(state),
                    "_official_state": _state_delta(before, state),
                }
            )
        return scores

    def summarize(
        self, records: List[EvalRecord], scores: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        official_records = []
        for record, score in zip(records, scores):
            state = deepcopy(record.example)
            state.pop("_process_reference", None)
            state["raw_output"] = record.raw_output
            state.update(score.get("_official_state", {}))
            official_records.append(state)

        if self.family == "mol_edit":
            summary = self.metrics.compute_summary(official_records)
        elif self.family in {"mol_und", "rxn_pred"}:
            summary = self.metrics.compute_summary(official_records, self.subtask)
        elif self.is_multi:
            summary = {
                "n_total": len(official_records),
                "layer1": self.layer1.summarize_multi_layer1(
                    official_records, self.subtask
                ),
                "layer2": self.layer2.summarize_layer2(official_records),
                "layer3": self.layer3.summarize_layer3(official_records),
            }
        else:
            summary = {
                "n_total": len(official_records),
                "layer1": self.layer1.summarize_single_layer1(
                    official_records, self.subtask
                ),
                "layer2": self.layer2.summarize_layer2(official_records),
                "layer3": self.layer3.summarize_layer3(official_records),
            }
        return normalize_summary(_json_safe(summary), self.family, self.subtask, scores)


def normalize_summary(
    summary: Dict[str, Any], family: str, subtask: str, scores: List[Dict[str, Any]]
) -> Dict[str, Any]:
    layer1 = summary.get("layer1", {})
    if family == "mol_opt":
        primary_name = "dual_sr_pct" if subtask in MULTI_MOLOPT else "sr_pct"
    elif family == "rxn_pred":
        primary_name = "mae" if subtask == "yield_pred" else "top1_acc"
    else:
        primary_name = layer1.get("primary_metric_name", "exact_match_acc")
    primary_value = layer1.get("primary_metric_value", layer1.get(primary_name))
    layer2 = summary.get("layer2", {})
    layer2_score = layer2.get("state_score", layer2.get("avg_state_score"))
    layer3 = summary.get("layer3", {})
    if family == "mol_opt":
        layer3_type1 = layer3.get("avg_step_score")
        layer3_type2 = None
    else:
        layer3_type1 = layer3.get("type1", {}).get("all_pass_rate")
        layer3_type2 = layer3.get("type2", {}).get("all_fields_match_rate")
    n = len(scores)
    return {
        "n": n,
        "parse_rate": sum(bool(score.get("parse_ok")) for score in scores) / n if n else 0.0,
        "length_capped": sum(score.get("parse_status") == "incomplete_length" for score in scores),
        "primary_metric_name": primary_name,
        "primary_layer1": primary_value,
        "layer2_score": layer2_score,
        "layer3_type1": layer3_type1,
        "layer3_type2": layer3_type2,
        "official_summary": summary,
    }
