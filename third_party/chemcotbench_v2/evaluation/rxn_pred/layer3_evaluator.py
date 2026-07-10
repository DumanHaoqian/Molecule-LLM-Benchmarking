"""Layer 3 evaluator: Type I (RDKit verifier) + Type II (PRM GT field comparison)."""
import json
from pathlib import Path
from importlib import import_module

from baselines.cot_eval.mol_edit.mol_edit_structured.utils import (
    canonical_smiles,
    smiles_match_main_frag,
)
from evaluation.core.config import resolve_dataset_name, resolve_gt_dataset_path, resolve_module_name
from evaluation.core.released_data import load_released_records
from evaluation.core.utils import find_gt_record


class Layer3Evaluator:
    def __init__(self, subtask: str):
        self.subtask = subtask
        self.module_name = resolve_module_name("rxn_pred", subtask)
        self.verifier = import_module(f"formal_cot.rxn_pred.{self.module_name}.verifier")
        self.gt_records = self._load_gt()
        self.schema = self._build_schema()

    def _load_gt(self) -> list[dict]:
        gt_path = resolve_gt_dataset_path("rxn_pred", self.subtask)
        if gt_path.exists():
            with open(gt_path) as f:
                return json.load(f)
        return load_released_records("rxn_pred", self.subtask)

    def _build_schema(self) -> list[tuple[str, str, str]]:
        """Build Type II comparison schema: (pred_field, gt_field, comparator)."""
        s = self.subtask
        if s in ("forward", "nepp"):
            return [
                ("step2_rxn_type", "coarse_rxn_cls", "ci_str"),
                ("step4_predicted_smi", "gt_product_smiles", "main_frag"),
                ("answer_smiles", "gt_product_smiles", "main_frag"),
            ]
        elif s == "retro":
            return [
                ("step2_rxn_type", "coarse_rxn_cls", "ci_str"),
                ("step4_reactant_smi", "gt_reactants", "main_frag"),
                ("answer_smi", "gt_reactants", "main_frag"),
            ]
        elif s == "byproduct":
            return [
                ("step2_rxn_type", "coarse_rxn_cls", "ci_str"),
                ("step6_byproduct_smiles", "gt_smiles", "main_frag"),
                ("answer_smiles", "gt_smiles", "main_frag"),
            ]
        elif s == "condition_ranking":
            return [
                ("step1_rxn_class", "coarse_rxn_cls", "ci_str"),
                ("step5_ranking", "gt_ranking", "list_eq"),
                ("answer", "gt_ranking", "list_eq"),
            ]
        elif s == "condition_temperature":
            return [
                ("step1_rxn_class", "coarse_rxn_cls", "ci_str"),
                ("step5_predicted_temp", "gt_temp", "exact_float"),
                ("answer_temp", "gt_temp", "exact_float"),
            ]
        elif s == "yield_pred":
            return [
                ("step1_rxn_class", "coarse_rxn_cls", "ci_str"),
                ("step5_predicted_yield", "gt_float", "exact_float"),
                ("answer_yield", "gt_float", "exact_float"),
            ]
        elif s == "rcr_catalyst":
            return [
                ("step1_rxn_cls", "coarse_rxn_cls", "ci_str"),
                ("step5_predicted_smi", "gt_catalyst_smiles", "main_frag"),
                ("answer_smiles", "gt_catalyst_smiles", "main_frag"),
            ]
        elif s == "rcr_reagent":
            return [
                ("step1_rxn_cls", "coarse_rxn_cls", "ci_str"),
                ("step5_predicted_smi", "gt_reagent_smiles", "main_frag"),
                ("answer_smiles", "gt_reagent_smiles", "main_frag"),
            ]
        elif s == "rcr_solvent":
            return [
                ("step1_rxn_cls", "coarse_rxn_cls", "ci_str"),
                ("step5_predicted_smi", "gt_solvent_smiles", "main_frag"),
                ("answer_smiles", "gt_solvent_smiles", "main_frag"),
            ]
        elif s == "rxn_template":
            return [
                ("step2_rxn_type", "coarse_rxn_cls", "ci_str"),
                ("step6_selected_option", "gt_letter", "ci_str"),
                ("answer_letter", "gt_letter", "ci_str"),
            ]
        elif s == "mech_sel":
            return [
                ("step2_rxn_type", "coarse_rxn_cls", "ci_str"),
                ("step5_selected", "gt_letter", "ci_str"),
                ("answer_letter", "gt_letter", "ci_str"),
            ]
        return []

    def evaluate_batch(self, records: list[dict]) -> list[dict]:
        # Type I: run existing verifier
        if hasattr(self.verifier, "verify_batch"):
            records, _ = self.verifier.verify_batch(records)
        elif hasattr(self.verifier, "verify_all"):
            records, _ = self.verifier.verify_all(records)
        else:
            for i, rec in enumerate(records):
                if hasattr(self.verifier, "verify_one"):
                    records[i] = self.verifier.verify_one(rec)
                elif hasattr(self.verifier, "verify_record"):
                    records[i] = self.verifier.verify_record(rec)

        # Normalize outcome / all_pass fields for downstream metrics
        for r in records:
            # Ensure layer3_type1_outcome exists
            if "outcome" in r:
                r["layer3_type1_outcome"] = r["outcome"]
            # Ensure all_pass exists
            if "all_pass" not in r:
                r["all_pass"] = False

        # Type II: field-by-field GT comparison
        for r in records:
            gt = self._get_gt(r)
            if gt:
                r.update(self._compare(r, gt))
            else:
                for pred_field, _, _ in self.schema:
                    r[f"gt_match_{pred_field}"] = None
                r["gt_match_all_fields"] = None
                r["gt_match_count"] = None
                r["gt_match_total"] = None
        return records

    def _get_gt(self, record: dict) -> dict | None:
        """Get GT record by matching ID field."""
        return find_gt_record(record, self.gt_records, ["pool_id", "id", "src_id", "sample_id"])

    def _compare(self, pred: dict, gt: dict) -> dict:
        results: dict[str, bool] = {}
        all_ok = True
        for pred_field, gt_field, comp in self.schema:
            p_val, g_val = pred.get(pred_field), gt.get(gt_field)
            match = self._cmp(p_val, g_val, comp)
            results[f"gt_match_{pred_field}"] = match
            if not match:
                all_ok = False
        results["gt_match_all_fields"] = all_ok
        results["gt_match_count"] = sum(
            1 for pred_field, _, _ in self.schema if results[f"gt_match_{pred_field}"]
        )
        results["gt_match_total"] = len(self.schema)
        return results

    def _cmp(self, p, g, comp: str) -> bool:
        if p is None or g is None:
            return False
        if comp == "exact_int":
            try:
                return int(p) == int(g)
            except (ValueError, TypeError):
                return False
        if comp == "exact_float":
            try:
                return abs(float(p) - float(g)) < 0.5
            except (ValueError, TypeError):
                return False
        if comp == "ci_str":
            return str(p).strip().upper() == str(g).strip().upper()
        if comp == "canon_smi":
            c_p = canonical_smiles(str(p))
            c_g = canonical_smiles(str(g))
            return (c_p == c_g) if (c_p and c_g) else (str(p).strip() == str(g).strip())
        if comp == "main_frag":
            return smiles_match_main_frag(str(p), str(g))
        if comp == "list_eq":
            # Handle string-encoded lists
            p_list = self._to_list(p)
            g_list = self._to_list(g)
            return p_list == g_list
        return False

    @staticmethod
    def _to_list(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            import ast
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                pass
        return [val] if val is not None else []
