"""Reverse-extract Gold Template fields from parsed A→B output for Layer 2 V-point evaluation."""
from baselines.cot_eval.mol_und.murcko_scaffold_structured.utils import get_ring_count


class Layer2Mapper:
    def __init__(self, subtask: str):
        self.subtask = subtask

    def map_record(self, record: dict) -> dict:
        fn = getattr(self, f"_map_{self.subtask}", lambda r: {})
        mapped = fn(record)
        mapped["subtask"] = self.subtask
        return mapped

    # ------------------------------------------------------------------ #
    # fg_detect
    # ------------------------------------------------------------------ #
    def _map_fg_detect(self, record: dict) -> dict:
        return {
            "target_smarts": record.get("step1_smarts"),
            "count_field": record.get("answer"),
            "answer": record.get("answer"),
        }

    # ------------------------------------------------------------------ #
    # ring_count
    # ------------------------------------------------------------------ #
    def _map_ring_count(self, record: dict) -> dict:
        return {
            "target_smarts": record.get("step1_smarts"),
            "total_rings": record.get("step2_total"),
            "accepted_count": record.get("step4_count"),
            "rejected_count": record.get("step5_rejected"),
            "count_field": record.get("answer"),
            "answer": record.get("answer"),
        }

    # ------------------------------------------------------------------ #
    # murcko_scaffold
    # ------------------------------------------------------------------ #
    def _map_murcko_scaffold(self, record: dict) -> dict:
        return {
            "mol_ring_count": record.get("step1_n_mol_rings"),
            "scaffold_smiles_field": record.get("step2_scaffold"),
            "answer": record.get("answer"),
        }

    # ------------------------------------------------------------------ #
    # ring_sys_scaffold
    # ------------------------------------------------------------------ #
    def _map_ring_sys_scaffold(self, record: dict) -> dict:
        non_ring = record.get("step3_non_ring")
        if non_ring is not None:
            non_ring = non_ring.strip().capitalize()
        return {
            "mol_ring_count": record.get("step1_n_mol"),
            "scaffold_ring_count": record.get("step2_n_scaf"),
            "non_ring_in_scaffold": non_ring,
            "answer": record.get("answer"),
        }

    # ------------------------------------------------------------------ #
    # mutated
    # ------------------------------------------------------------------ #
    def _map_mutated(self, record: dict) -> dict:
        # Construct synthetic key_difference from formula_match + canonical_equal
        fm = record.get("step3_formula_match", "")
        ce = record.get("step4_canonical_equal", "")
        key_diff = None
        if fm == "different":
            key_diff = "Different molecular formulas"
        elif fm == "same":
            if ce == "no":
                key_diff = "Different canonical SMILES"
            elif ce == "yes":
                key_diff = "Same molecule"
        return {
            "formula_a_field": record.get("step1_formula_a"),
            "formula_b_field": record.get("step2_formula_b"),
            "key_difference": key_diff,
            "answer": record.get("answer"),
        }

    # ------------------------------------------------------------------ #
    # permutated
    # ------------------------------------------------------------------ #
    def _map_permutated(self, record: dict) -> dict:
        # Ring count is not in unified format; compute from RDKit to avoid unfair penalty
        smiles = record.get("smiles", "")
        ring_cnt = get_ring_count(smiles)
        # Map smiles_identical from step3_identical
        si = record.get("step3_identical")
        if si is not None:
            si = si.lower()
        return {
            "canonical_a_text": record.get("step1_canonical_a"),
            "canonical_b_text": record.get("step2_canonical_b"),
            "ring_cnt": ring_cnt,
            "smiles_identical": si,
            "answer": record.get("answer"),
        }
