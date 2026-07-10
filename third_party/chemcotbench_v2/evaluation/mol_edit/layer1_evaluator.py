"""Layer 1 evaluator: final answer correctness (Top-1 Acc / FTS)."""
from baselines.cot_eval.mol_edit.mol_edit_structured.utils import (
    compute_edit_site,
    edit_site_overlap,
    fts as _fts,
    smiles_match_exact,
    smiles_match_main_frag,
)


def evaluate_layer1(record: dict, edit_type: str) -> dict:
    """Compute Layer 1 metrics for a single record."""
    gt = record.get("gt_smiles", "")
    pred = (
        record.get("step3_product_smiles")
        or record.get("step4_product_smiles")
        or record.get("answer_smiles", "")
    )
    ok = bool(pred and gt and smiles_match_main_frag(pred, gt))

    # V_consistency: product SMILES == Answer line
    answer = record.get("answer_smiles", "")
    v_consistency = int(
        pred is not None
        and bool(answer)
        and smiles_match_exact(pred, answer)
    )

    src = record.get("source_smiles") or record.get("src_smiles", "")
    if src and pred and edit_type:
        gt_site = compute_edit_site(src, gt, edit_type)
        pred_site = compute_edit_site(src, pred, edit_type)
        eso = edit_site_overlap(gt_site, pred_site)
    else:
        eso = 0.0

    return {
        "layer1_top1_acc": ok,
        "layer1_fts": round(_fts(pred or "", gt), 4),
        "layer1_top1_acc_strict": int(
            bool(pred and gt and smiles_match_exact(pred, gt))
        ),
        "layer1_v_consistency": v_consistency,
        "layer1_edit_site_overlap": round(eso, 4),
    }


def summarize_layer1(records: list[dict]) -> dict:
    """Aggregate Layer 1 metrics."""
    n = len(records)
    if n == 0:
        return {}

    acc = sum(r["layer1_top1_acc"] for r in records) / n
    fts_val = sum(r["layer1_fts"] for r in records) / n
    acc_strict = sum(r["layer1_top1_acc_strict"] for r in records) / n
    v_cons = sum(r["layer1_v_consistency"] for r in records) / n
    eso = sum(r["layer1_edit_site_overlap"] for r in records) / n

    summary = {
        "n": n,
        "top1_acc": round(acc, 4),
        "avg_fts": round(fts_val, 4),
        "top1_acc_strict": round(acc_strict, 4),
        "v_consistency": round(v_cons, 4),
        "avg_edit_site_overlap": round(eso, 4),
    }

    # By difficulty
    by_diff: dict[str, list] = {}
    for r in records:
        by_diff.setdefault(r.get("difficulty", "unknown"), []).append(r)

    diff_stats = {}
    for d, recs in by_diff.items():
        nd = len(recs)
        diff_stats[d] = {
            "n": nd,
            "top1_acc": round(sum(r["layer1_top1_acc"] for r in recs) / nd, 4),
            "avg_fts": round(sum(r["layer1_fts"] for r in recs) / nd, 4),
            "top1_acc_strict": round(sum(r["layer1_top1_acc_strict"] for r in recs) / nd, 4),
            "v_consistency": round(sum(r["layer1_v_consistency"] for r in recs) / nd, 4),
            "avg_edit_site_overlap": round(sum(r["layer1_edit_site_overlap"] for r in recs) / nd, 4),
        }

    summary["by_difficulty"] = diff_stats
    return summary
