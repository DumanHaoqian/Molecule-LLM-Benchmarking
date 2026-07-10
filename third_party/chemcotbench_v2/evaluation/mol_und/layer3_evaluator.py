"""Layer 3 evaluator: Type I (RDKit verifier) + Type II (PRM GT field comparison)."""
import json
from pathlib import Path
from importlib import import_module

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from baselines.cot_eval.mol_und.fg_detect_structured.utils import (
    smarts_semantically_matches_gt,
)
from baselines.cot_eval.mol_und.murcko_scaffold_structured.utils import (
    canonical_smiles,
    smiles_are_equal,
)
from evaluation.core.config import resolve_gt_dataset_path
from evaluation.core.released_data import load_released_records


def _tanimoto(smiles_a: str, smiles_b: str) -> float:
    """Morgan fingerprint Tanimoto similarity between two SMILES."""
    try:
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        if mol_a is None or mol_b is None:
            return 0.0
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius=2, nBits=2048)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius=2, nBits=2048)
        return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
    except Exception:
        return 0.0


class Layer3Evaluator:
    def __init__(self, subtask: str):
        self.subtask = subtask
        self.verifier = import_module(f"formal_cot.mol_und.{subtask}.verifier")
        self.gt_records = self._load_gt()
        self.schema = self._build_schema()

    def _load_gt(self) -> list[dict]:
        gt_path = resolve_gt_dataset_path("mol_und", self.subtask)
        if gt_path.exists():
            with open(gt_path) as f:
                return json.load(f)
        return load_released_records("mol_und", self.subtask)

    def _build_schema(self) -> list[tuple[str, str]]:
        """Build comparison schema: list of (field_name, comparison_type)."""
        return self._schema_for_subtask(self.subtask)

    def _schema_for_subtask(self, subtask: str) -> list[tuple[str, str]]:
        if subtask == "fg_detect":
            return [
                ("step1_smarts", "smarts_semantic"),
                ("step2_n", "exact_int"),
                ("step3_count", "exact_int"),
                ("answer", "exact_int"),
            ]
        elif subtask == "ring_count":
            return [
                ("step1_smarts", "smarts_semantic"),
                ("step2_total", "exact_int"),
                ("step3_n", "exact_int"),
                ("step4_count", "exact_int"),
                ("step5_rejected", "exact_int"),
                ("answer", "exact_int"),
            ]
        elif subtask == "murcko_scaffold":
            return [
                ("step1_n_mol_rings", "exact_int"),
                ("step2_scaffold", "tanimoto"),
                ("step3_n_scaf_rings", "exact_int"),
                ("step4_ring_match", "ci_str"),
                ("step5_substructure", "ci_str"),
                ("answer", "tanimoto"),
            ]
        elif subtask == "ring_sys_scaffold":
            return [
                ("step1_n_mol", "exact_int"),
                ("step2_n_scaf", "exact_int"),
                ("step3_non_ring", "ci_str"),
                ("step4_predict", "ci_str"),
                ("answer", "ci_str"),
            ]
        elif subtask == "mutated":
            return [
                ("step1_formula_a", "ci_str"),
                ("step2_formula_b", "ci_str"),
                ("step3_formula_match", "ci_str"),
                ("step4_canonical_equal", "ci_str"),
                ("step5_predict", "ci_str"),
                ("answer", "ci_str"),
            ]
        elif subtask == "permutated":
            return [
                ("step1_canonical_a", "canon_smi"),
                ("step2_canonical_b", "canon_smi"),
                ("step3_identical", "ci_str"),
                ("step4_predict", "ci_str"),
                ("answer", "ci_str"),
            ]
        return []

    def evaluate_batch(self, records: list[dict]) -> list[dict]:
        # Type I: run existing verifier
        if hasattr(self.verifier, "verify_all"):
            records, _ = self.verifier.verify_all(records)
        elif hasattr(self.verifier, "verify_batch"):
            records, _ = self.verifier.verify_batch(records)
        else:
            # Fallback: per-record verification
            for r in records:
                if hasattr(self.verifier, "verify_record"):
                    r.update(self.verifier.verify_record(r))
                elif hasattr(self.verifier, "verify_one"):
                    r.update(self.verifier.verify_one(r))

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
        """Get GT record by orig_idx."""
        orig_idx = record.get("orig_idx")
        if orig_idx is None:
            return None
        # GT records are ordered by orig_idx 0-799
        if 0 <= orig_idx < len(self.gt_records):
            gt = self.gt_records[orig_idx]
            if gt.get("orig_idx") == orig_idx:
                return gt
            # Fallback: search
            for g in self.gt_records:
                if g.get("orig_idx") == orig_idx:
                    return g
        return None

    def _compare(self, pred: dict, gt: dict) -> dict:
        results: dict[str, bool | float] = {}
        all_ok = True
        schema = self._schema_for_subtask(pred.get("source_subtask", self.subtask))
        for field, comp in schema:
            p_val, g_val = pred.get(field), gt.get(field)
            match = self._cmp(p_val, g_val, comp)
            results[f"gt_match_{field}"] = match
            if not match:
                all_ok = False
        results["gt_match_all_fields"] = all_ok
        results["gt_match_count"] = sum(
            1 for f, _ in schema if results[f"gt_match_{f}"]
        )
        results["gt_match_total"] = len(schema)
        return results

    def _cmp(self, p, g, comp: str) -> bool:
        if p is None or g is None:
            return False
        if comp == "exact_int":
            try:
                return int(p) == int(g)
            except (ValueError, TypeError):
                return False
        if comp == "ci_str":
            return str(p).strip().upper() == str(g).strip().upper()
        if comp == "smarts_semantic":
            return smarts_semantically_matches_gt(str(p), str(g))
        if comp == "canon_smi":
            c_p = canonical_smiles(str(p))
            c_g = canonical_smiles(str(g))
            return (c_p == c_g) if (c_p and c_g) else (str(p).strip() == str(g).strip())
        if comp == "tanimoto":
            # For murcko_scaffold: Tanimoto >= 0.9 is considered match
            tan = _tanimoto(str(p), str(g))
            return tan >= 0.9
        return False
