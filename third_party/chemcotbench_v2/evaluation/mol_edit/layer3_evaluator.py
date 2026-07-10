"""Layer 3 evaluator: Type I (RDKit verifier) + Type II (PRM GT field comparison)."""
import json
from pathlib import Path
from importlib import import_module

from baselines.cot_eval.mol_edit.mol_edit_structured.utils import (
    canonical_smiles,
    smiles_match_main_frag,
)
from evaluation.core.config import resolve_gt_dataset_path
from evaluation.core.released_data import load_released_records
from evaluation.core.utils import find_gt_record


class Layer3Evaluator:
    def __init__(self, subtask: str):
        self.subtask = subtask
        self.edit_type = subtask.replace("_v2", "")
        self.verifier = import_module(f"formal_cot.mol_edit.{subtask}.verifier")
        self.gt_records = self._load_gt()
        self.schema = self._build_schema()

    def _load_gt(self) -> list[dict]:
        gt_path = resolve_gt_dataset_path("mol_edit", self.subtask)
        if gt_path.exists():
            with open(gt_path) as f:
                return json.load(f)
        return load_released_records("mol_edit", self.subtask)

    def _build_schema(self) -> list[tuple[str, str]]:
        if self.edit_type == "add":
            return [
                ("step1_anchor_idx", "exact_int"),
                ("step1_anchor_element", "ci_str"),
                ("step1_leaving_smiles", "leaving"),
                ("step2_frag_smiles", "canon_smi"),
                ("step2_heavy_atoms", "exact_int"),
                ("step3_product_smiles", "main_frag"),
                ("step4_n_heavy_src", "exact_int"),
                ("step4_n_heavy_prod", "exact_int"),
                ("step4_heavy_delta", "exact_int"),
                ("step5_n_rings_src", "exact_int"),
                ("step5_n_rings_prod", "exact_int"),
                ("step5_ring_delta", "exact_int"),
                ("answer_smiles", "main_frag"),
            ]
        elif self.edit_type == "delete":
            return [
                ("step1_anchor_idx", "exact_int"),
                ("step1_anchor_element", "ci_str"),
                ("step1_remove_group", "canon_smi"),
                ("step2_remove_smiles", "canon_smi"),
                ("step2_heavy_atoms", "exact_int"),
                ("step3_product_smiles", "main_frag"),
                ("step4_n_heavy_src", "exact_int"),
                ("step4_n_heavy_prod", "exact_int"),
                ("step4_heavy_delta", "exact_int"),
                ("step5_n_rings_src", "exact_int"),
                ("step5_n_rings_prod", "exact_int"),
                ("step5_ring_delta", "exact_int"),
                ("answer_smiles", "main_frag"),
            ]
        else:  # substitute
            return [
                ("step1_anchor_idx", "exact_int"),
                ("step1_anchor_element", "ci_str"),
                ("step1_remove_group_smiles", "canon_smi"),
                ("step1_add_fragment_smiles", "canon_smi"),
                ("step2_remove_heavy", "exact_int"),
                ("step3_add_heavy", "exact_int"),
                ("step4_product_smiles", "main_frag"),
                ("step5_n_heavy_src", "exact_int"),
                ("step5_n_heavy_prod", "exact_int"),
                ("step5_heavy_delta", "exact_int"),
                ("step6_n_rings_src", "exact_int"),
                ("step6_n_rings_prod", "exact_int"),
                ("step6_ring_delta", "exact_int"),
                ("answer_smiles", "main_frag"),
            ]

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

        for r in records:
            r["layer3_type1_outcome"] = r.pop("outcome", None)

        # Type II: field-by-field GT comparison
        for r in records:
            gt = self._get_gt(r)
            if gt:
                r.update(self._compare(r, gt))
            else:
                for field, _ in self.schema:
                    r[f"gt_match_{field}"] = None
                r["gt_match_all_fields"] = None
                r["gt_match_count"] = None
                r["gt_match_total"] = None
        return records

    def _get_gt(self, record: dict) -> dict | None:
        """Get GT record by orig_id."""
        return find_gt_record(record, self.gt_records, ["orig_id"])

    def _compare(self, pred: dict, gt: dict) -> dict:
        results: dict[str, bool] = {}
        all_ok = True
        for field, comp in self.schema:
            p_val, g_val = pred.get(field), gt.get(field)
            match = self._cmp(p_val, g_val, comp)
            results[f"gt_match_{field}"] = match
            if not match:
                all_ok = False
        results["gt_match_all_fields"] = all_ok
        results["gt_match_count"] = sum(
            1 for f, _ in self.schema if results[f"gt_match_{f}"]
        )
        results["gt_match_total"] = len(self.schema)
        return results

    def _cmp(self, p, g, comp: str) -> bool:
        if p is None or g is None:
            return False
        if comp == "exact_int":
            return int(p) == int(g)
        if comp == "ci_str":
            return str(p).strip().upper() == str(g).strip().upper()
        if comp == "leaving":
            p_norm = str(p).strip().lower().strip('"\'')
            g_norm = str(g).strip().lower().strip('"\'')
            return p_norm == g_norm or (p_norm in ("none", "null", "") and g_norm in ("none", "null", ""))
        if comp == "canon_smi":
            c_p = canonical_smiles(str(p))
            c_g = canonical_smiles(str(g))
            return (c_p == c_g) if (c_p and c_g) else (str(p).strip() == str(g).strip())
        if comp == "main_frag":
            return smiles_match_main_frag(str(p), str(g))
        return False
