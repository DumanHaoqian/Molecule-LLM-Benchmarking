"""Layer 1 evaluator: final answer correctness (Top-1 Acc / FTS / MAE / NDCG etc.)."""
import math
import statistics

from baselines.cot_eval.rxn_predict.forward_structured.utils import (
    all_frags_valid,
    fts as _fts,
    smiles_match_set,
)
from baselines.cot_eval.rxn_predict.condition_ranking_structured.utils import (
    ndcg_score,
    mrr_score,
)


def evaluate_layer1(record: dict, subtask: str) -> dict:
    """Compute Layer 1 metrics for a single record."""
    fn = globals().get(f"_eval_{subtask.replace('_', '_')}")
    if fn is None:
        # Generic fallback: try to find a smiles/answer field and compare with GT
        return _eval_generic(record)
    return fn(record)


def _get_pred_and_gt(record: dict, pred_keys: list[str], gt_key: str):
    """Helper: extract pred and gt values."""
    pred = None
    for k in pred_keys:
        if record.get(k):
            pred = record[k]
            break
    gt = record.get(gt_key)
    return pred, gt


def _eval_forward(record: dict) -> dict:
    pred, gt = _get_pred_and_gt(
        record,
        ["step4_predicted_smi", "answer_smiles", "answer"],
        "gt_product_smiles",
    )
    ok = bool(pred and gt and smiles_match_set(pred, gt))
    return {
        "layer1_top1_acc": ok,
        "layer1_fts": round(_fts(pred or "", gt or ""), 4),
        "layer1_top1_acc_strict": int(
            bool(pred and gt and all_frags_valid(pred) and smiles_match_set(pred, gt))
        ),
    }


def _eval_retro(record: dict) -> dict:
    pred, gt = _get_pred_and_gt(
        record,
        ["step4_reactant_smi", "answer_smi", "answer_smiles", "answer"],
        "gt_reactants",
    )
    ok = bool(pred and gt and smiles_match_set(pred, gt))
    return {
        "layer1_top1_acc": ok,
        "layer1_fts": round(_fts(pred or "", gt or ""), 4),
        "layer1_top1_acc_strict": int(
            bool(pred and gt and all_frags_valid(pred) and smiles_match_set(pred, gt))
        ),
    }


def _eval_nepp(record: dict) -> dict:
    # Same as forward
    return _eval_forward(record)


def _eval_byproduct(record: dict) -> dict:
    pred, gt = _get_pred_and_gt(
        record,
        ["step6_byproduct_smiles", "answer_smiles", "answer"],
        "gt_smiles",
    )
    ok = bool(pred and gt and smiles_match_set(pred, gt))
    return {
        "layer1_top1_acc": ok,
        "layer1_fts": round(_fts(pred or "", gt or ""), 4),
        "layer1_top1_acc_strict": int(
            bool(pred and gt and all_frags_valid(pred) and smiles_match_set(pred, gt))
        ),
    }


def _eval_condition_ranking(record: dict) -> dict:
    gt_rank = record.get("gt_ranking", [])
    pred_rank = record.get("answer") or record.get("step5_ranking") or []
    # Normalize to list
    pred_rank = _to_list(pred_rank)
    gt_rank = _to_list(gt_rank)

    top1 = bool(pred_rank and gt_rank and pred_rank[0] == gt_rank[0])
    ndcg = ndcg_score(pred_rank, gt_rank) if pred_rank and gt_rank else 0.0
    mrr = mrr_score(pred_rank, gt_rank[0]) if pred_rank and gt_rank else 0.0

    return {
        "layer1_top1_acc": top1,
        "layer1_ndcg": round(ndcg, 4),
        "layer1_mrr": round(mrr, 4),
        "layer1_fts": 1.0 if top1 else 0.0,
        "layer1_top1_acc_strict": int(top1),
    }


def _eval_condition_temperature(record: dict) -> dict:
    pred = record.get("answer_temp") or record.get("step5_predicted_temp")
    gt = record.get("gt_temp")
    if pred is None or gt is None:
        return {
            "layer1_top1_acc": False,
            "layer1_mae": None,
            "layer1_rmse": None,
            "layer1_within_5": 0.0,
            "layer1_fts": 0.0,
        }
    try:
        pv = float(pred)
        gv = float(gt)
        err = abs(pv - gv)
        return {
            "layer1_top1_acc": err < 0.5,
            "layer1_mae": round(err, 4),
            "layer1_rmse": round(err, 4),
            "layer1_within_5": 1.0 if err <= 5.0 else 0.0,
            "layer1_fts": max(0.0, 1.0 - err / 100.0),
        }
    except (ValueError, TypeError):
        return {
            "layer1_top1_acc": False,
            "layer1_mae": None,
            "layer1_rmse": None,
            "layer1_within_5": 0.0,
            "layer1_fts": 0.0,
        }


def _eval_yield_pred(record: dict) -> dict:
    pred = record.get("answer_yield") or record.get("step5_predicted_yield")
    gt = record.get("gt_float")
    if pred is None or gt is None:
        return {
            "layer1_top1_acc": False,
            "layer1_mae": None,
            "layer1_rmse": None,
            "layer1_within_5": 0.0,
            "layer1_fts": 0.0,
        }
    try:
        pv = float(pred)
        gv = float(gt)
        err = abs(pv - gv)
        return {
            "layer1_top1_acc": err < 0.5,
            "layer1_mae": round(err, 4),
            "layer1_rmse": round(err, 4),
            "layer1_within_5": 1.0 if err <= 5.0 else 0.0,
            "layer1_fts": max(0.0, 1.0 - err / 100.0),
        }
    except (ValueError, TypeError):
        return {
            "layer1_top1_acc": False,
            "layer1_mae": None,
            "layer1_rmse": None,
            "layer1_within_5": 0.0,
            "layer1_fts": 0.0,
        }


def _eval_rcr_catalyst(record: dict) -> dict:
    pred, gt = _get_pred_and_gt(
        record,
        ["step5_predicted_smi", "answer_smiles", "answer"],
        "gt_catalyst_smiles",
    )
    ok = bool(pred and gt and smiles_match_set(pred, gt))
    return {
        "layer1_top1_acc": ok,
        "layer1_fts": round(_fts(pred or "", gt or ""), 4),
        "layer1_top1_acc_strict": int(ok),
    }


def _eval_rcr_reagent(record: dict) -> dict:
    pred, gt = _get_pred_and_gt(
        record,
        ["step5_predicted_smi", "answer_smiles", "answer"],
        "gt_reagent_smiles",
    )
    ok = bool(pred and gt and smiles_match_set(pred, gt))
    return {
        "layer1_top1_acc": ok,
        "layer1_fts": round(_fts(pred or "", gt or ""), 4),
        "layer1_top1_acc_strict": int(ok),
    }


def _eval_rcr_solvent(record: dict) -> dict:
    pred, gt = _get_pred_and_gt(
        record,
        ["step5_predicted_smi", "answer_smiles", "answer"],
        "gt_solvent_smiles",
    )
    ok = bool(pred and gt and smiles_match_set(pred, gt))
    return {
        "layer1_top1_acc": ok,
        "layer1_fts": round(_fts(pred or "", gt or ""), 4),
        "layer1_top1_acc_strict": int(ok),
    }


def _eval_rxn_template(record: dict) -> dict:
    pred = (
        record.get("step6_selected_option")
        or record.get("answer_letter")
        or record.get("answer")
        or ""
    )
    gt = record.get("gt_letter", "")
    ok = bool(pred and gt and str(pred).strip().upper() == str(gt).strip().upper())
    return {
        "layer1_top1_acc": ok,
        "layer1_fts": 1.0 if ok else 0.0,
        "layer1_top1_acc_strict": int(ok),
    }


def _eval_mech_sel(record: dict) -> dict:
    pred = (
        record.get("step5_selected")
        or record.get("answer_letter")
        or record.get("answer")
        or ""
    )
    gt = record.get("gt_letter", "")
    ok = bool(pred and gt and str(pred).strip().upper() == str(gt).strip().upper())
    return {
        "layer1_top1_acc": ok,
        "layer1_fts": 1.0 if ok else 0.0,
        "layer1_top1_acc_strict": int(ok),
    }


def _eval_generic(record: dict) -> dict:
    """Fallback for unknown subtasks."""
    pred = record.get("answer") or record.get("answer_smiles") or ""
    gt = record.get("gt") or record.get("gt_product_smiles") or record.get("gt_smiles") or ""
    ok = bool(pred and gt and smiles_match_set(pred, gt))
    return {
        "layer1_top1_acc": ok,
        "layer1_fts": round(_fts(pred or "", gt or ""), 4),
        "layer1_top1_acc_strict": int(ok),
    }


def _to_list(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        import ast
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return [val] if val is not None else []


def summarize_layer1(records: list[dict], subtask: str) -> dict:
    """Aggregate Layer 1 metrics."""
    n = len(records)
    if n == 0:
        return {}

    result = {"n": n}

    # Top-1 Acc
    if "layer1_top1_acc" in records[0]:
        result["top1_acc"] = round(
            sum(r["layer1_top1_acc"] for r in records) / n, 4
        )
    if "layer1_top1_acc_strict" in records[0]:
        result["top1_acc_strict"] = round(
            sum(r["layer1_top1_acc_strict"] for r in records) / n, 4
        )

    # FTS
    if "layer1_fts" in records[0]:
        fts_vals = [r["layer1_fts"] for r in records if r.get("layer1_fts") is not None]
        result["avg_fts"] = round(statistics.mean(fts_vals), 4) if fts_vals else 0.0

    # NDCG / MRR (condition_ranking)
    if "layer1_ndcg" in records[0]:
        ndcg_vals = [r["layer1_ndcg"] for r in records if r.get("layer1_ndcg") is not None]
        result["avg_ndcg"] = round(statistics.mean(ndcg_vals), 4) if ndcg_vals else 0.0
    if "layer1_mrr" in records[0]:
        mrr_vals = [r["layer1_mrr"] for r in records if r.get("layer1_mrr") is not None]
        result["avg_mrr"] = round(statistics.mean(mrr_vals), 4) if mrr_vals else 0.0

    # MAE / RMSE / within_5 (regression tasks)
    if "layer1_mae" in records[0]:
        mae_vals = [r["layer1_mae"] for r in records if r.get("layer1_mae") is not None]
        if mae_vals:
            result["mae"] = round(statistics.mean(mae_vals), 4)
            result["rmse"] = round(
                math.sqrt(sum(v ** 2 for v in mae_vals) / len(mae_vals)), 4
            )
        within5 = sum(1 for r in records if r.get("layer1_within_5") == 1.0) / n
        result["within_5"] = round(within5, 4)

    return result
