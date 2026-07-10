"""Layer 2 evaluator for MolUnd: template structure compliance + internal self-consistency.

Zero RDKit, zero GT comparison.
"""

# ---------------------------------------------------------------------------
# Per-subtask V-point definitions
# ---------------------------------------------------------------------------


def _eval_fg_detect(record: dict) -> dict:
    step1 = record.get("step1_smarts")
    step2 = record.get("step2_n")
    step3 = record.get("step3_count")
    answer = record.get("answer")

    v1 = all(x is not None for x in (step1, step2, step3))
    v2 = bool(step1 and str(step1).strip())
    v3 = bool(answer is not None and str(answer).strip() != "")
    v4 = False
    if step3 is not None and answer is not None:
        try:
            v4 = int(step3) == int(answer)
        except (ValueError, TypeError):
            v4 = str(step3).strip() == str(answer).strip()

    state_score = (int(v1) + int(v2) + int(v3) + int(v4)) / 4.0
    return {
        "V1": int(v1), "V2": int(v2), "V3": int(v3), "V4": int(v4),
        "state_score": round(state_score, 4),
    }


def _eval_ring_count(record: dict) -> dict:
    step1 = record.get("step1_smarts")
    step2 = record.get("step2_total")
    step3 = record.get("step3_n")
    step4 = record.get("step4_count")
    step5 = record.get("step5_rejected")
    answer = record.get("answer")

    v1 = all(x is not None for x in (step1, step2, step3, step4, step5))
    v2 = bool(step1 and str(step1).strip())
    v3 = bool(answer is not None and str(answer).strip() != "")
    v4 = False
    if step4 is not None and answer is not None:
        try:
            v4 = int(step4) == int(answer)
        except (ValueError, TypeError):
            v4 = str(step4).strip() == str(answer).strip()

    state_score = (int(v1) + int(v2) + int(v3) + int(v4)) / 4.0
    return {
        "V1": int(v1), "V2": int(v2), "V3": int(v3), "V4": int(v4),
        "state_score": round(state_score, 4),
    }


def _eval_murcko_scaffold(record: dict) -> dict:
    step1 = record.get("step1_n_mol_rings")
    step2 = record.get("step2_scaffold")
    step3 = record.get("step3_n_scaf_rings")
    step4 = record.get("step4_ring_match")
    step5 = record.get("step5_substructure")
    answer = record.get("answer")

    v1 = all(x is not None for x in (step1, step2, step3, step4, step5))
    v2 = bool(step2 and str(step2).strip()) and bool(answer and str(answer).strip())
    v3 = False
    if step2 is not None and answer is not None:
        v3 = str(step2).strip() == str(answer).strip()

    state_score = (int(v1) + int(v2) + int(v3)) / 3.0
    return {
        "V1": int(v1), "V2": int(v2), "V3": int(v3),
        "state_score": round(state_score, 4),
    }


def _eval_ring_sys_scaffold(record: dict) -> dict:
    step1 = record.get("step1_n_mol")
    step2 = record.get("step2_n_scaf")
    step3 = record.get("step3_non_ring")
    step4 = record.get("step4_predict")
    answer = record.get("answer")

    v1 = all(x is not None for x in (step1, step2, step3, step4))
    v2 = bool(answer is not None and str(answer).strip().capitalize() in ("Yes", "No"))
    v3 = bool(step3 is not None and str(step3).strip().lower() in ("yes", "no"))
    v4 = False
    if step4 is not None and answer is not None:
        v4 = str(step4).strip().capitalize() == str(answer).strip().capitalize()

    state_score = (int(v1) + int(v2) + int(v3) + int(v4)) / 4.0
    return {
        "V1": int(v1), "V2": int(v2), "V3": int(v3), "V4": int(v4),
        "state_score": round(state_score, 4),
    }


def _eval_mutated(record: dict) -> dict:
    step1 = record.get("step1_formula_a")
    step2 = record.get("step2_formula_b")
    step3 = record.get("step3_formula_match")
    step4 = record.get("step4_canonical_equal")
    step5 = record.get("step5_predict")
    answer = record.get("answer")

    v1 = all(x is not None for x in (step1, step2, step3, step4, step5))
    v2 = bool(answer is not None and str(answer).strip() in ("Same", "Different"))
    v3 = bool(step3 is not None and str(step3).strip().lower() in ("same", "different"))

    v4 = False
    if step5 is not None and answer is not None:
        v4 = str(step5).strip().capitalize() == str(answer).strip().capitalize()

    state_score = (int(v1) + int(v2) + int(v3) + int(v4)) / 4.0
    return {
        "V1": int(v1), "V2": int(v2), "V3": int(v3), "V4": int(v4),
        "state_score": round(state_score, 4),
    }


def _eval_permutated(record: dict) -> dict:
    step1 = record.get("step1_canonical_a")
    step2 = record.get("step2_canonical_b")
    step3 = record.get("step3_identical")
    step4 = record.get("step4_predict")
    answer = record.get("answer")

    v1 = all(x is not None for x in (step1, step2, step3, step4))
    v2 = bool(answer is not None and str(answer).strip() in ("Same", "Different"))
    v3 = bool(step3 is not None and str(step3).strip().lower() in ("yes", "no"))
    v4 = False
    if step4 is not None and answer is not None:
        v4 = str(step4).strip().capitalize() == str(answer).strip().capitalize()

    state_score = (int(v1) + int(v2) + int(v3) + int(v4)) / 4.0
    return {
        "V1": int(v1), "V2": int(v2), "V3": int(v3), "V4": int(v4),
        "state_score": round(state_score, 4),
    }


_EVALUATORS = {
    "fg_detect": _eval_fg_detect,
    "ring_count": _eval_ring_count,
    "murcko_scaffold": _eval_murcko_scaffold,
    "ring_sys_scaffold": _eval_ring_sys_scaffold,
    "mutated": _eval_mutated,
    "permutated": _eval_permutated,
}


def evaluate_record(record: dict, subtask: str) -> dict:
    """Dispatch to subtask-specific evaluator."""
    if subtask == "smiles_equivalent":
        subtask = record.get("source_subtask", subtask)
    fn = _EVALUATORS.get(subtask)
    if fn is None:
        return {"V1": 0, "state_score": 0.0}
    return fn(record)
