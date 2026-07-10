"""Reverse-extract Gold Template fields from parsed A→B output for Layer 2 V-point evaluation."""


class Layer2Mapper:
    def __init__(self, subtask: str):
        self.subtask = subtask

    def map_record(self, record: dict) -> dict:
        fn = getattr(self, f"_map_{self.subtask}", lambda r: {})
        mapped = fn(record)
        # Add GT fields so baseline evaluators can work without GT_LOOKUP
        mapped.update(self._extract_gt_fields(record))
        return mapped

    def _extract_gt_fields(self, record: dict) -> dict:
        """Copy GT fields from the record for baseline evaluators."""
        gt = {}
        # Common GT fields
        for k in ["coarse_rxn_cls", "rxn_cls", "gt_rxn_cls"]:
            if k in record:
                gt["rxn_cls"] = record[k]
                gt["reaction_class"] = record[k]
                break
        for k in ["gt_product_smiles", "gt_smiles", "gt_reactants", "gt"]:
            if k in record:
                gt["gt"] = record[k]
                gt["gt_smiles"] = record[k]
                break
        for k in ["gt_catalyst_smiles", "gt_reagent_smiles", "gt_solvent_smiles"]:
            if k in record:
                gt["gt_smiles"] = record[k]
                break
        for k in ["gt_temp"]:
            if k in record:
                gt["gt_temp"] = record[k]
        for k in ["gt_float", "gt_yield_class"]:
            if k in record:
                gt["gt_float"] = record[k]
        for k in ["gt_letter"]:
            if k in record:
                gt["gt_letter"] = record[k]
        for k in ["gt_ranking"]:
            if k in record:
                gt["gt_rank"] = record[k]
        # Ensure id field exists for baseline evaluators
        for k in ["pool_id", "id", "src_id", "sample_id"]:
            if k in record:
                gt["id"] = record[k]
                break
        return gt

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def _map_forward(self, record: dict) -> dict:
        fg_list = record.get("step1_fg_list", [])
        fg_text = "; ".join(fg_list) if isinstance(fg_list, list) else str(fg_list)
        return {
            "reaction_type_field": record.get("step2_rxn_type", ""),
            "functional_groups_text": fg_text,
            "reactive_site_text": fg_text,  # fallback: forward A→B has no explicit reactive site
            "mechanism_text": record.get("step3_mechanism", ""),
            "bond_formed_text": record.get("step6_bond_formed", ""),
            "answer": record.get("answer_smiles", "") or record.get("step4_predicted_smi", ""),
            "reactants": record.get("reactants_list", []),
        }

    # ------------------------------------------------------------------ #
    # retro
    # ------------------------------------------------------------------ #
    def _map_retro(self, record: dict) -> dict:
        return {
            "reaction_type_field": record.get("step2_rxn_type", ""),
            "target_molecule_field": record.get("product_smiles", ""),
            "disconnection_text": record.get("step6_bond_broken", ""),
            "reactant_smiles_field": record.get("step4_reactant_smi", ""),
            "answer": record.get("answer_smi", "") or record.get("step4_reactant_smi", ""),
        }

    # ------------------------------------------------------------------ #
    # nepp
    # ------------------------------------------------------------------ #
    def _map_nepp(self, record: dict) -> dict:
        return {
            "reaction_type_field": record.get("step2_elem_mech", ""),
            "proton_transfer_text": record.get("step2_elem_mech", ""),
            "answer": record.get("answer_smiles", "") or record.get("step4_predicted_smi", ""),
            "query_text": record.get("current_reactants", ""),
        }

    # ------------------------------------------------------------------ #
    # byproduct
    # ------------------------------------------------------------------ #
    def _map_byproduct(self, record: dict) -> dict:
        return {
            "reaction_type_field": record.get("step2_rxn_type", ""),
            "leaving_group_text": record.get("step4_lf_smiles", ""),
            "primary_product_smiles": record.get("step5_product_smiles", ""),
            "leaving_fragment_smiles": record.get("step4_lf_smiles", ""),
            "byproduct_source_text": record.get("step6_byproduct_smiles", ""),
            "answer": record.get("answer_smiles", "") or record.get("step6_byproduct_smiles", ""),
            "reactants": record.get("reactants_list", []),
        }

    # ------------------------------------------------------------------ #
    # condition_ranking
    # ------------------------------------------------------------------ #
    def _map_condition_ranking(self, record: dict) -> dict:
        # Extract key differentiator text from top2_support
        t2 = record.get("step6_top2_support", {})
        if isinstance(t2, dict):
            key_diff = t2.get("field", "")
        else:
            key_diff = ""
        ranking = record.get("step5_ranking", [])
        answer = record.get("answer", [])
        return {
            "reaction_type_field": record.get("step1_rxn_class", ""),
            "condition_scores": {},  # A→B has no explicit condition scores
            "key_differentiator_text": key_diff,
            "ranking_field": ranking if isinstance(ranking, list) else [],
            "answer_ranking": answer if isinstance(answer, list) else [],
            "pred_ranking": answer if isinstance(answer, list) else (ranking if isinstance(ranking, list) else []),
            "gt_rank": record.get("gt_ranking", []),
        }

    # ------------------------------------------------------------------ #
    # condition_temperature
    # ------------------------------------------------------------------ #
    def _map_condition_temperature(self, record: dict) -> dict:
        return {
            "reaction_type_field": record.get("step1_rxn_class", ""),
            "temperature_field": record.get("step5_predicted_temp", ""),
            "answer_temp": record.get("answer_temp", ""),
        }

    # ------------------------------------------------------------------ #
    # rcr_catalyst
    # ------------------------------------------------------------------ #
    def _map_rcr_catalyst(self, record: dict) -> dict:
        return {
            "smiles_field": record.get("step5_predicted_smi", "") or record.get("answer_smiles", ""),
            "reaction_type_field": record.get("step1_rxn_cls", ""),
            "reaction_class_field": record.get("step1_rxn_cls", ""),
            "core_transformation": record.get("step2_core_transform", ""),
            "catalytic_center_text": record.get("step3_catalyst_role", ""),
            "catalyst_metal_text": record.get("step3_catalyst_role", ""),
        }

    # ------------------------------------------------------------------ #
    # rcr_reagent
    # ------------------------------------------------------------------ #
    def _map_rcr_reagent(self, record: dict) -> dict:
        return {
            "smiles_field": record.get("step5_predicted_smi", "") or record.get("answer_smiles", ""),
            "reaction_type_field": record.get("step1_rxn_cls", ""),
            "reaction_class_field": record.get("step1_rxn_cls", ""),
            "core_transformation": record.get("step2_core_transform", ""),
            "reagent_role_text": record.get("step3_reagent_role", ""),
        }

    # ------------------------------------------------------------------ #
    # rcr_solvent
    # ------------------------------------------------------------------ #
    def _map_rcr_solvent(self, record: dict) -> dict:
        return {
            "smiles_field": record.get("step5_predicted_smi", "") or record.get("answer_smiles", ""),
            "reaction_type_field": record.get("step1_rxn_cls", ""),
            "reaction_class_field": record.get("step1_rxn_cls", ""),
            "core_transformation": record.get("step2_core_transform", ""),
            "solvent_role_text": record.get("step3_solvent_role", ""),
        }

    # ------------------------------------------------------------------ #
    # rxn_template
    # ------------------------------------------------------------------ #
    def _map_rxn_template(self, record: dict) -> dict:
        return {
            "reaction_type_field": record.get("step2_rxn_type", ""),
            "smarts_field": record.get("step4_proposed_smarts", ""),
            "selected_option": record.get("step6_selected_option", ""),
            "answer_letter": record.get("answer_letter", ""),
        }

    # ------------------------------------------------------------------ #
    # mech_sel
    # ------------------------------------------------------------------ #
    def _map_mech_sel(self, record: dict) -> dict:
        return {
            "reaction_type_field": record.get("step2_rxn_type", ""),
            "selected_option": record.get("step5_selected", ""),
            "answer_letter": record.get("answer_letter", ""),
        }

    # ------------------------------------------------------------------ #
    # yield_pred
    # ------------------------------------------------------------------ #
    def _map_yield_pred(self, record: dict) -> dict:
        return {
            "reaction_type_field": record.get("step1_rxn_class", ""),
            "yield_field": record.get("step5_predicted_yield", ""),
            "answer_yield": record.get("answer_yield", ""),
        }
