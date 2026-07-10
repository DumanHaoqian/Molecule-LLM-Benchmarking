"""Reverse-extract Gold Template fields from parsed A→B output for Layer 2 V-point evaluation."""


class Layer2Mapper:
    def __init__(self, subtask: str):
        self.edit_type = subtask.replace("_v2", "")

    def map_record(self, record: dict) -> dict:
        fn = getattr(self, f"_map_{self.edit_type}", lambda r: {})
        mapped = fn(record)
        mapped["edit_type"] = self.edit_type
        return mapped

    # ------------------------------------------------------------------ #
    # add_v2
    # ------------------------------------------------------------------ #
    def _map_add(self, record: dict) -> dict:
        ring_check = self._build_ring_check(
            record.get("step5_n_rings_src"),
            record.get("step5_n_rings_prod"),
            record.get("step5_ring_delta"),
        )
        return {
            "target_group": record.get("step1_anchor_element"),
            "attachment_point": str(record["step1_anchor_idx"])
            if record.get("step1_anchor_idx") is not None
            else None,
            "replacement": record.get("step2_frag_smiles"),
            "target_atom_ids": None,
            "product_smiles_field": record.get("step3_product_smiles"),
            "answer": record.get("answer_smiles"),
            "ring_check": ring_check,
        }

    # ------------------------------------------------------------------ #
    # delete_v2
    # ------------------------------------------------------------------ #
    def _map_delete(self, record: dict) -> dict:
        ring_check = self._build_ring_check(
            record.get("step5_n_rings_src"),
            record.get("step5_n_rings_prod"),
            record.get("step5_ring_delta"),
        )
        return {
            "target_group": record.get("step1_remove_group")
            or record.get("step2_remove_smiles"),
            "attachment_point": str(record["step1_anchor_idx"])
            if record.get("step1_anchor_idx") is not None
            else None,
            "replacement": None,
            "target_atom_ids": None,
            "product_smiles_field": record.get("step3_product_smiles"),
            "answer": record.get("answer_smiles"),
            "ring_check": ring_check,
        }

    # ------------------------------------------------------------------ #
    # substitute_v2
    # ------------------------------------------------------------------ #
    def _map_substitute(self, record: dict) -> dict:
        ring_check = self._build_ring_check(
            record.get("step6_n_rings_src"),
            record.get("step6_n_rings_prod"),
            record.get("step6_ring_delta"),
        )
        return {
            "target_group": record.get("step1_remove_group_smiles"),
            "attachment_point": str(record["step1_anchor_idx"])
            if record.get("step1_anchor_idx") is not None
            else None,
            "replacement": record.get("step3_add_fragment_smiles"),
            "target_atom_ids": None,
            "product_smiles_field": record.get("step4_product_smiles"),
            "answer": record.get("answer_smiles"),
            "ring_check": ring_check,
        }

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_ring_check(src_rings, prod_rings, delta) -> str | None:
        if src_rings is None or prod_rings is None or delta is None:
            return None
        return f"Ring count: {src_rings} -> {prod_rings} (delta {int(delta):+d})"
