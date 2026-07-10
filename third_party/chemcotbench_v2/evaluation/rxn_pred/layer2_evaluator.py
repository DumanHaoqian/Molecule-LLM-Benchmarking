"""Layer 2 evaluator for RxnPred: template structure compliance + internal self-consistency.

Zero RDKit, zero GT comparison.
"""

import json

# 9 coarse-grained reaction classes
VALID_RXN_CLASSES = {
    "c-c coupling", "heteroatom alkylation and arylation", "acylation",
    "functional group interconversion", "deprotection", "reduction",
    "oxidation", "aromatic heterocycle formation", "protection",
}

VALID_DECISION_FACTORS = {"catalyst", "ligand", "base", "reagent", "additive", "solvent"}


def _norm(val):
    return str(val).strip().lower() if val is not None else ""


def _is_non_empty(val) -> bool:
    return val is not None and str(val).strip() != ""


def _is_floatable(val) -> bool:
    if val is None:
        return False
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def _rxn_class_valid(val) -> bool:
    return _norm(val) in VALID_RXN_CLASSES


def _normalize_cmp_text(val) -> str:
    """Normalise lightweight representation differences before exact compare."""
    if val is None:
        return ""
    if not isinstance(val, str):
        return str(val).strip()

    text = val.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            for key in ("pred_smi", "answer_smiles", "smiles"):
                inner = parsed.get(key)
                if _is_non_empty(inner):
                    text = str(inner).strip()
                    break

    return text.replace("\\\\", "\\")


# ---------------------------------------------------------------------------
# Group A: forward, retro, nepp, byproduct (4 V-points)
# ---------------------------------------------------------------------------

_GROUP_A_STEP_FIELDS = {
    "forward": [
        "step1_fg_list", "step2_rxn_type", "step3_mechanism",
        "step4_predicted_smi", "step5_mol_valid", "step6_bond_formed",
    ],
    "retro": [
        "step1_fg_list", "step2_rxn_type", "step3_mechanism",
        "step4_reactant_smi", "step5_all_valid", "step6_bond_broken",
    ],
    "nepp": [
        "step1_reactants_smi", "step2_elem_mech", "step3_bond_change",
        "step4_predicted_smi", "step5_product_charge", "step6_charge_balanced", "step7_atom_conserved",
    ],
    "byproduct": [
        "step1_fg_list", "step2_rxn_type", "step3_atomic_delta",
        "step4_lf_smiles", "step5_fragment_in_reactant", "step6_byproduct_smiles",
    ],
}

_GROUP_A_PRED_KEYS = {
    "forward": ("step4_predicted_smi", "answer_smiles"),
    "retro": ("step4_reactant_smi", "answer_smi"),
    "nepp": ("step4_predicted_smi", "answer_smiles"),
    "byproduct": ("step6_byproduct_smiles", "answer_smiles"),
}


def _eval_group_a(record: dict, subtask: str) -> dict:
    step_fields = _GROUP_A_STEP_FIELDS[subtask]
    pred_key, ans_key = _GROUP_A_PRED_KEYS[subtask]

    v1 = all(_is_non_empty(record.get(f)) for f in step_fields)
    # V2: rxn_type in valid classes (forward/retro/byproduct); nepp uses elem_mech
    if subtask == "nepp":
        v2 = _is_non_empty(record.get("step2_elem_mech"))
    else:
        v2 = _rxn_class_valid(record.get("step2_rxn_type"))
    v3 = _is_non_empty(record.get(pred_key))

    pred = record.get(pred_key)
    ans = record.get(ans_key)
    v4 = False
    if _is_non_empty(pred) and _is_non_empty(ans):
        v4 = _normalize_cmp_text(pred) == _normalize_cmp_text(ans)

    state_score = (int(v1) + int(v2) + int(v3) + int(v4)) / 4.0
    return {
        "V1": int(v1), "V2": int(v2), "V3": int(v3), "V4": int(v4),
        "state_score": round(state_score, 4),
    }


# ---------------------------------------------------------------------------
# Group B: condition_ranking (6 V-points)
# ---------------------------------------------------------------------------

def _eval_condition_ranking(record: dict) -> dict:
    s1 = record.get("step1_rxn_class")
    s2 = record.get("step2_decision_factor")
    s3 = record.get("step3_pair_diffs")
    s4 = record.get("step4_pairwise_prefs")
    s5 = record.get("step5_ranking")
    s6 = record.get("step6_top2_support")

    v1 = all(x is not None for x in (s1, s2, s3, s4, s5, s6))
    v2 = _norm(s2) in VALID_DECISION_FACTORS

    # V3: pair_diffs covers 3 pairs
    v3 = False
    if s3 is not None:
        if isinstance(s3, dict):
            v3 = len(s3) >= 3
        elif isinstance(s3, str):
            v3 = all(p in s3 for p in ("1/2", "1/3", "2/3"))
        elif isinstance(s3, list):
            v3 = len(s3) >= 3

    # V4: pairwise_prefs has 3 comparisons
    v4 = False
    if s4 is not None:
        if isinstance(s4, list):
            v4 = len(s4) >= 3
        elif isinstance(s4, str):
            v4 = s4.count(">") >= 3

    # V5: ranking is a valid permutation of ["1","2","3"]
    v5 = False
    if s5 is not None:
        if isinstance(s5, list):
            v5 = sorted(s5) == ["1", "2", "3"]
        elif isinstance(s5, str):
            try:
                parsed = [x.strip().strip('"\'') for x in s5.strip("[]").split(",")]
                v5 = sorted(parsed) == ["1", "2", "3"]
            except Exception:
                pass

    # V6: top2_support winner == ranking[0]
    v6 = False
    if s6 is not None and s5 is not None:
        winner = None
        if isinstance(s6, dict):
            winner = str(s6.get("winner", "")).strip()
        elif isinstance(s6, str):
            m = __import__("re").search(r'WINNER\s*=\s*"?([^",\]]+)"?', s6)
            if m:
                winner = m.group(1).strip().strip('"')
        ranking_first = None
        if isinstance(s5, list) and len(s5) > 0:
            ranking_first = str(s5[0]).strip()
        elif isinstance(s5, str):
            try:
                parsed = [x.strip().strip('"\'') for x in s5.strip("[]").split(",")]
                if parsed:
                    ranking_first = parsed[0]
            except Exception:
                pass
        if winner is not None and ranking_first is not None:
            v6 = winner == ranking_first

    state_score = (int(v1) + int(v2) + int(v3) + int(v4) + int(v5) + int(v6)) / 6.0
    return {
        "V1": int(v1), "V2": int(v2), "V3": int(v3),
        "V4": int(v4), "V5": int(v5), "V6": int(v6),
        "state_score": round(state_score, 4),
    }


# ---------------------------------------------------------------------------
# Group C: condition_temperature, yield_pred (3 V-points)
# ---------------------------------------------------------------------------

_GROUP_C_STEP_FIELDS = {
    "condition_temperature": [
        "step1_rxn_class", "step2_substrate", "step2_reactivity",
        "step3_conditions", "step3_efficiency", "step5_predicted_temp", "self_consistent",
    ],
    "yield_pred": [
        "step1_rxn_class", "step2_halide_type", "step3_nucleophile_type",
        "step3_nucleophile_form", "step4_ligand_class", "step5_predicted_yield",
        "step6_self_consistent",
    ],
}

_GROUP_C_PRED_ANS = {
    "condition_temperature": ("step5_predicted_temp", "answer_temp"),
    "yield_pred": ("step5_predicted_yield", "answer_yield"),
}


def _eval_group_c(record: dict, subtask: str) -> dict:
    step_fields = _GROUP_C_STEP_FIELDS[subtask]
    pred_key, ans_key = _GROUP_C_PRED_ANS[subtask]

    v1 = all(_is_non_empty(record.get(f)) for f in step_fields)
    v2 = _is_non_empty(record.get(pred_key)) and _is_floatable(record.get(pred_key))

    pred = record.get(pred_key)
    ans = record.get(ans_key)
    v3 = False
    if _is_non_empty(pred) and _is_non_empty(ans):
        v3 = str(pred).strip() == str(ans).strip()

    state_score = (int(v1) + int(v2) + int(v3)) / 3.0
    return {
        "V1": int(v1), "V2": int(v2), "V3": int(v3),
        "state_score": round(state_score, 4),
    }


# ---------------------------------------------------------------------------
# Group D: rcr_catalyst, rcr_reagent, rcr_solvent (3 V-points)
# ---------------------------------------------------------------------------

_GROUP_D_STEP_FIELDS = {
    "rcr_catalyst": [
        "step1_rxn_cls", "step2_core_transform", "step3_catalyst_role",
        "step4_catalyst_class", "step5_predicted_smi", "step6_self_consistent",
    ],
    "rcr_reagent": [
        "step1_rxn_class", "step2_core_transformation", "step3_reagent_slot",
        "step4_reagent_strategy", "step5_component_mode", "step6_reagent_class",
        "step7_predicted_reagent_smiles", "step8_self_consistent",
    ],
    "rcr_solvent": [
        "step1_rxn_class", "step2_core_transformation", "step3_proticity",
        "step4_polarity", "step5_predicted_solvent_smiles", "step6_self_consistent",
    ],
}

_GROUP_D_PRED_ANS = {
    "rcr_catalyst": ("step5_predicted_smi", "answer_smiles"),
    "rcr_reagent": ("step7_predicted_reagent_smiles", "answer_smiles"),
    "rcr_solvent": ("step5_predicted_solvent_smiles", "answer_smiles"),
}

_GROUP_D_PRED_ALIASES = {
    "rcr_catalyst": ("step5_predicted_catalyst_smiles",),
}


def _eval_group_d(record: dict, subtask: str) -> dict:
    step_fields = _GROUP_D_STEP_FIELDS[subtask]
    pred_key, ans_key = _GROUP_D_PRED_ANS[subtask]
    pred_aliases = _GROUP_D_PRED_ALIASES.get(subtask, ())

    v1 = all(_is_non_empty(record.get(f)) for f in step_fields)
    pred = None
    use_aliases = not bool(record.get("patch_outcome_fixed"))
    if use_aliases:
        for key in pred_aliases:
            if _is_non_empty(record.get(key)):
                pred = record.get(key)
                break
    if pred is None:
        pred = record.get(pred_key)

    v2 = _is_non_empty(pred)

    ans = record.get(ans_key)
    v3 = False
    if _is_non_empty(pred) and _is_non_empty(ans):
        v3 = _normalize_cmp_text(pred) == _normalize_cmp_text(ans)

    state_score = (int(v1) + int(v2) + int(v3)) / 3.0
    return {
        "V1": int(v1), "V2": int(v2), "V3": int(v3),
        "state_score": round(state_score, 4),
    }


# ---------------------------------------------------------------------------
# Group E: rxn_template, mech_sel (3 V-points)
# ---------------------------------------------------------------------------

_GROUP_E_STEP_FIELDS = {
    "rxn_template": [
        "step1_bond_changes", "step2_rxn_type", "step3_mechanism",
        "step4_proposed_smarts", "step5_smarts_parseable", "step6_selected_option",
    ],
    "mech_sel": [
        "step1_smarts_list", "step1_reagent_types", "step2_rxn_type",
        "step3_eliminated", "step4_remaining", "step5_selected",
    ],
}

_GROUP_E_PRED_ANS = {
    "rxn_template": ("step6_selected_option", "answer_letter"),
    "mech_sel": ("step5_selected", "answer_letter"),
}


def _eval_group_e(record: dict, subtask: str) -> dict:
    step_fields = _GROUP_E_STEP_FIELDS[subtask]
    pred_key, ans_key = _GROUP_E_PRED_ANS[subtask]

    v1 = all(_is_non_empty(record.get(f)) for f in step_fields)
    v2 = _is_non_empty(record.get(pred_key))

    pred = record.get(pred_key)
    ans = record.get(ans_key)
    v3 = False
    if _is_non_empty(pred) and _is_non_empty(ans):
        v3 = str(pred).strip() == str(ans).strip()

    state_score = (int(v1) + int(v2) + int(v3)) / 3.0
    return {
        "V1": int(v1), "V2": int(v2), "V3": int(v3),
        "state_score": round(state_score, 4),
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_EVALUATORS = {
    "forward": lambda r: _eval_group_a(r, "forward"),
    "retro": lambda r: _eval_group_a(r, "retro"),
    "nepp": lambda r: _eval_group_a(r, "nepp"),
    "byproduct": lambda r: _eval_group_a(r, "byproduct"),
    "condition_ranking": _eval_condition_ranking,
    "condition_temperature": lambda r: _eval_group_c(r, "condition_temperature"),
    "yield_pred": lambda r: _eval_group_c(r, "yield_pred"),
    "rcr_catalyst": lambda r: _eval_group_d(r, "rcr_catalyst"),
    "rcr_reagent": lambda r: _eval_group_d(r, "rcr_reagent"),
    "rcr_solvent": lambda r: _eval_group_d(r, "rcr_solvent"),
    "rxn_template": lambda r: _eval_group_e(r, "rxn_template"),
    "mech_sel": lambda r: _eval_group_e(r, "mech_sel"),
}


def evaluate_record(record: dict, subtask: str) -> dict:
    """Dispatch to subtask-specific evaluator."""
    fn = _EVALUATORS.get(subtask)
    if fn is None:
        return {"V1": 0, "state_score": 0.0}
    return fn(record)
