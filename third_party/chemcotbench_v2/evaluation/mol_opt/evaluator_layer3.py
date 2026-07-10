"""Layer 3 evaluator: Step-wise reasoning correctness (vs PRM GT dataset)."""
import json

from .utils import (
    scaffold_similarity,
    molecule_similarity,
    step2_match,
)


def evaluate_layer3(record: dict, gt_record: dict) -> dict:
    """
    Compare a parsed record against its PRM GT record step by step.
    Returns dict with match results for each step and overall step_score.
    """
    src = record.get("src", "")
    pred = record.get("predicted_smiles", "") or record.get("step3_predicted_smiles", "")

    # ── Step 1: SCAFFOLD_IDENTIFICATION ──
    # Continuous Tanimoto similarity between predicted and GT Murcko scaffolds
    gt_scaffold = gt_record.get("step1_scaffold_smiles", "")
    pred_scaffold = record.get("step1_scaffold_smiles", "")
    step1_score = 0.0
    if gt_scaffold and pred_scaffold:
        step1_score = scaffold_similarity(gt_scaffold, pred_scaffold)

    # ── Step 2: EDIT_PLAN ──
    # Semantic direction match (binary)
    gt_removed = gt_record.get("step2_fg_removed", "")
    gt_added = gt_record.get("step2_fg_added", "")
    pred_removed = record.get("step2_fg_removed", "") or record.get("fg_removed", "")
    pred_added = record.get("step2_fg_added", "") or record.get("fg_added", "")
    step2_match_result = step2_match(
        gt_removed, gt_added, pred_removed, pred_added,
        src, pred
    )

    # ── Step 3: PRODUCT_CONSTRUCTION ──
    # Continuous Tanimoto similarity between predicted and GT products
    gt_pred = gt_record.get("step3_predicted_smiles", "")
    step3_score = 0.0
    if gt_pred and pred:
        step3_score = molecule_similarity(pred, gt_pred)

    # ── Step 4: SCAFFOLD_PRESERVATION ──
    # Exact string match of yes/no claim (binary)
    gt_preserved = gt_record.get("step4_scaffold_claimed", "")
    pred_preserved = record.get("step4_scaffold_claimed", "") or record.get("scaffold_preserved", "")
    step4_match = False
    if gt_preserved and pred_preserved:
        step4_match = gt_preserved.lower() == pred_preserved.lower()

    # ── Step 5: FG_CHANGE_VERIFICATION ──
    # Exact string match of yes/no claim (binary)
    gt_fg = gt_record.get("step5_fg_consistent_claimed", "")
    pred_fg = record.get("step5_fg_consistent_claimed", "") or record.get("fg_consistent", "")
    step5_match = False
    if gt_fg and pred_fg:
        step5_match = gt_fg.lower() == pred_fg.lower()

    step_score = (step1_score + int(step2_match_result) + step3_score
                  + int(step4_match) + int(step5_match)) / 5

    return {
        "layer3_step1_scaffold_match": round(step1_score, 4),
        "layer3_step2_editplan_match": step2_match_result,
        "layer3_step3_product_match": round(step3_score, 4),
        "layer3_step4_preserved_match": step4_match,
        "layer3_step5_fg_match": step5_match,
        "layer3_step_score": round(step_score, 4),
        # GT values for inspection
        "layer3_gt_step1": gt_scaffold,
        "layer3_gt_step2_removed": gt_removed,
        "layer3_gt_step2_added": gt_added,
        "layer3_gt_step3": gt_pred,
        "layer3_gt_step4": gt_preserved,
        "layer3_gt_step5": gt_fg,
    }


def evaluate_batch_layer3(records: list[dict], gt_records: list[dict]) -> list[dict]:
    """
    Evaluate a batch of records against GT records.
    Assumes records and gt_records are aligned by index (both from same dataset).
    """
    results = []
    for i, rec in enumerate(records):
        gt = None
        idx = rec.get("idx")
        if idx is not None:
            for candidate in gt_records:
                if candidate.get("idx") == idx:
                    gt = candidate
                    break
        if gt is None and i < len(gt_records):
            gt = gt_records[i]
        if gt is not None:
            l3 = evaluate_layer3(rec, gt)
            merged = dict(rec)
            merged.update(l3)
            results.append(merged)
        else:
            merged = dict(rec)
            merged.update({
                "layer3_step1_scaffold_match": 0.0,
                "layer3_step2_editplan_match": False,
                "layer3_step3_product_match": 0.0,
                "layer3_step4_preserved_match": False,
                "layer3_step5_fg_match": False,
                "layer3_step_score": 0.0,
                "layer3_error": "no matching GT record",
            })
            results.append(merged)

    # Print summary
    n = len(results)
    if n > 0:
        continuous_steps = {1, 3}
        for step in range(1, 6):
            key = f"layer3_step{step}_scaffold_match" if step == 1 else \
                  f"layer3_step{step}_editplan_match" if step == 2 else \
                  f"layer3_step{step}_product_match" if step == 3 else \
                  f"layer3_step{step}_preserved_match" if step == 4 else \
                  f"layer3_step{step}_fg_match"
            if step in continuous_steps:
                avg_val = sum(r.get(key, 0.0) for r in results) / n
                print(f"  Step {step}: avg={avg_val:.3f}")
            else:
                cnt = sum(1 for r in results if r.get(key))
                print(f"  Step {step}: {cnt}/{n} ({100*cnt/n:.1f}%)")
        avg_score = sum(r["layer3_step_score"] for r in results) / n
        print(f"  Avg Step Score: {avg_score:.3f}")

    return results


def summarize_layer3(records: list[dict]) -> dict:
    """Aggregate Layer 3 metrics."""
    n = len(records)
    if n == 0:
        return {}

    # Step 1 & 3 are continuous scores [0, 1]; Step 2, 4, 5 are binary
    continuous_keys = [
        "layer3_step1_scaffold_match",
        "layer3_step3_product_match",
    ]
    binary_keys = [
        "layer3_step2_editplan_match",
        "layer3_step4_preserved_match",
        "layer3_step5_fg_match",
    ]

    rates = {}
    for k in continuous_keys:
        rates[k] = round(sum(r.get(k, 0.0) for r in records) / n, 4)
    for k in binary_keys:
        rates[k] = round(sum(1 for r in records if r.get(k)) / n, 4)

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
            "step_score": round(sum(x["layer3_step_score"] for x in items) / ni, 4),
        }
        for k in continuous_keys:
            diff_stats[d][k] = round(sum(x.get(k, 0.0) for x in items) / ni, 4)
        for k in binary_keys:
            diff_stats[d][k] = round(sum(1 for x in items if x.get(k)) / ni, 4)

    return {
        "n": n,
        **rates,
        "avg_step_score": round(sum(r["layer3_step_score"] for r in records) / n, 4),
        "by_difficulty": diff_stats,
    }
