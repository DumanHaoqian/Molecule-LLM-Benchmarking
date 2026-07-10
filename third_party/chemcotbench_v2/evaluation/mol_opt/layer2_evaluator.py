"""Layer 2 evaluator for MolOpt: template structure compliance + internal self-consistency.

Zero RDKit, zero GT comparison.
"""


# ── helpers ────────────────────────────────────────────────────────────────

def _is_non_empty(val) -> bool:
    return val is not None and str(val).strip() != ""


def _in_set(val, allowed: set[str]) -> bool:
    return str(val).strip().lower() in allowed


# ── per-record evaluation ──────────────────────────────────────────────────

def evaluate_layer2(record: dict) -> dict:
    """Evaluate a single MolOpt record for Layer 2 V-points."""
    # 5 core fields (Part A primary, Part B fallback)
    scaffold = record.get("scaffold_smiles") or record.get("step1_scaffold_smiles")
    fg_removed = record.get("fg_removed") or record.get("step2_fg_removed")
    fg_added = record.get("fg_added") or record.get("step2_fg_added")
    predicted = record.get("predicted_smiles") or record.get("step3_predicted_smiles")
    scaf_claim = record.get("scaffold_preserved") or record.get("step4_scaffold_claimed")
    fg_claim = record.get("fg_consistent") or record.get("step5_fg_consistent_claimed")
    answer = record.get("answer_smiles")

    # V1: all 5 core fields present
    v1 = all(
        x is not None
        for x in (scaffold, fg_removed, fg_added, predicted, scaf_claim, fg_claim, answer)
    )

    # V2: predicted_smiles non-empty
    v2 = _is_non_empty(predicted)

    # V3: scaffold_preserved claim in {yes, no}
    v3 = _in_set(scaf_claim, {"yes", "no"})

    # V4: fg_consistent claim in {yes, no}
    v4 = _in_set(fg_claim, {"yes", "no"})

    # V5: predicted_smiles == answer_smiles (internal self-consistency)
    v5 = False
    if _is_non_empty(predicted) and _is_non_empty(answer):
        v5 = str(predicted).strip() == str(answer).strip()

    state_score = (int(v1) + int(v2) + int(v3) + int(v4) + int(v5)) / 5.0

    return {
        "layer2_V1": int(v1),
        "layer2_V2": int(v2),
        "layer2_V3": int(v3),
        "layer2_V4": int(v4),
        "layer2_V5": int(v5),
        "layer2_state_score": round(state_score, 4),
    }


# ── aggregation ────────────────────────────────────────────────────────────

def summarize_layer2(records: list[dict]) -> dict:
    """Aggregate Layer 2 metrics."""
    n = len(records)
    if n == 0:
        return {}

    v1 = sum(r["layer2_V1"] for r in records) / n
    v2 = sum(r["layer2_V2"] for r in records) / n
    v3 = sum(r["layer2_V3"] for r in records) / n
    v4 = sum(r["layer2_V4"] for r in records) / n
    v5 = sum(r["layer2_V5"] for r in records) / n
    ss = sum(r["layer2_state_score"] for r in records) / n

    by_diff = {"easy": [], "medium": [], "hard": []}
    for r in records:
        d = r.get("difficulty", "medium")
        if d in by_diff:
            by_diff[d].append(r)

    diff_stats = {}
    for d, items in by_diff.items():
        if not items:
            continue
        ni = len(items)
        diff_stats[d] = {
            "n": ni,
            "state_score": round(sum(x["layer2_state_score"] for x in items) / ni, 4),
            "V1": round(sum(x["layer2_V1"] for x in items) / ni, 4),
            "V2": round(sum(x["layer2_V2"] for x in items) / ni, 4),
            "V3": round(sum(x["layer2_V3"] for x in items) / ni, 4),
            "V4": round(sum(x["layer2_V4"] for x in items) / ni, 4),
            "V5": round(sum(x["layer2_V5"] for x in items) / ni, 4),
        }

    return {
        "n": n,
        "V1": round(v1, 4),
        "V2": round(v2, 4),
        "V3": round(v3, 4),
        "V4": round(v4, 4),
        "V5": round(v5, 4),
        "avg_state_score": round(ss, 4),
        # Backward-compat aliases for run.py print_summary
        "V3_product_construction": round(v2, 4),
        "V4_scaffold_preservation": round(v3, 4),
        "V5_fg_verification": round(v4, 4),
        "by_difficulty": diff_stats,
    }
