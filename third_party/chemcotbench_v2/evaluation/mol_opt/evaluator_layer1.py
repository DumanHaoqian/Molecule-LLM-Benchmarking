"""Layer 1 evaluator: Outcome metrics (Acc, SR%, delta, FTS, scaffold)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.Chem import AllChem, DataStructs, rdFMCS

from .utils import (
    get_oracle, check_improvement, compute_delta, tanimoto_fts,
    THRESHOLD, calculate_solubility, check_dual_improvement,
    MULTI_SUBTASK_TO_PROPS,
)
from baselines.cot_eval.mol_opt.mol_opt_multi_structured.utils import FIXED_THRESHOLDS as MULTI_THRESHOLDS


# ─── Single-target Layer 1 ────────────────────────────────────────────────────

def evaluate_single_layer1(record: dict, prop: str) -> dict:
    """Compute Layer 1 metrics for a single-target record."""
    src = record.get("src", "")
    tgt = record.get("tgt", "")
    pred = record.get("predicted_smiles", "") or record.get("step3_predicted_smiles", "")

    oracle_fn = get_oracle(prop)
    delta = compute_delta(oracle_fn, src, pred)
    outcome = check_improvement(oracle_fn, src, pred)
    thresh = THRESHOLD.get(prop, 0.3)
    best = delta >= thresh

    fts = tanimoto_fts(pred, tgt) if pred and tgt else None

    return {
        "layer1_src": src,
        "layer1_tgt": tgt,
        "layer1_pred": pred,
        "layer1_delta": round(delta, 6),
        "layer1_outcome": outcome,
        "layer1_best_rate": best,
        "layer1_fts": fts,
        "layer1_property": prop,
    }


def summarize_single_layer1(records: list[dict], prop: str) -> dict:
    """Aggregate Layer 1 metrics over a batch."""
    n = len(records)
    if n == 0:
        return {}

    outcomes = [r["layer1_outcome"] for r in records]
    deltas = [r["layer1_delta"] for r in records]
    fts_vals = [r["layer1_fts"] for r in records if r["layer1_fts"] is not None]
    thresh = THRESHOLD.get(prop, 0.3)

    # Scaffold consistency
    scf = scaffold_consistency(
        [r["layer1_src"] for r in records],
        [r["layer1_pred"] for r in records],
    )

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
        d_deltas = [x["layer1_delta"] for x in items]
        diff_stats[d] = {
            "n": ni,
            "sr": round(sum(x["layer1_outcome"] for x in items) / ni, 4),
            "best_rate": round(sum(1 for dd in d_deltas if dd >= thresh) / ni, 4),
            "mean_delta": round(sum(d_deltas) / ni, 4),
            "avg_fts": sum(x["layer1_fts"] for x in items if x["layer1_fts"] is not None)
                       / max(1, sum(1 for x in items if x["layer1_fts"] is not None)),
        }

    # V1-aligned: invalid_rate = fraction of predictions that are invalid SMILES
    invalid_rate = sum(1 for pred in [r["layer1_pred"] for r in records]
                       if not pred or Chem.MolFromSmiles(pred) is None) / n

    return {
        "n": n,
        "sr_pct": round(sum(outcomes) / n * 100, 2),
        "best_rate": round(sum(1 for d in deltas if d >= thresh) / n, 4),
        "mean_delta": round(sum(deltas) / n, 4),
        "invalid_rate": round(invalid_rate, 4),
        "avg_fts": round(sum(fts_vals) / len(fts_vals), 4) if fts_vals else None,
        "scaffold_hard": scf["hard"],
        "scaffold_soft": scf["soft"],
        "by_difficulty": diff_stats,
    }


# ─── Multi-target Layer 1 ─────────────────────────────────────────────────────

def _norm(delta: float | None, threshold: float) -> float | None:
    """V1-aligned normalized improvement: clamp(delta/threshold, -2, 5)."""
    if delta is None or threshold <= 0:
        return None
    return float(max(-2.0, min(5.0, delta / threshold)))


def evaluate_multi_layer1(record: dict, subtask: str) -> dict:
    """Compute Layer 1 metrics for a multi-target record."""
    import math

    src = record.get("src", "")
    tgt = record.get("tgt", "")
    pred = record.get("predicted_smiles", "") or record.get("step3_predicted_smiles", "")

    props = MULTI_SUBTASK_TO_PROPS[subtask]
    oracle_fns = {p: get_oracle(p) for p in props}
    thresholds = {p: MULTI_THRESHOLDS[p] for p in props}

    outcome, deltas = check_dual_improvement(oracle_fns, src, pred, thresholds)

    prop_results = {}
    norms = {}
    for p in props:
        d = deltas.get(p)
        prop_results[f"delta_{p}"] = d
        prop_results[f"outcome_{p}"] = d is not None and d > 0
        prop_results[f"best_{p}"] = d is not None and d >= thresholds[p]
        norms[p] = _norm(d, thresholds[p])

    norm_min = None
    norm_geo = None
    vals = [v for v in norms.values() if v is not None]
    if len(vals) == len(props):
        norm_min = min(vals)
        norm_geo = math.sqrt(max(0.0, vals[0]) * max(0.0, vals[1])) if len(vals) == 2 else None

    fts = tanimoto_fts(pred, tgt) if pred and tgt else None

    return {
        "layer1_src": src,
        "layer1_tgt": tgt,
        "layer1_pred": pred,
        "layer1_outcome": outcome,
        "layer1_fts": fts,
        "layer1_subtask": subtask,
        **{f"norm_{p}": norms[p] for p in props},
        "norm_min": norm_min,
        "norm_geo": norm_geo,
        **prop_results,
    }


def summarize_multi_layer1(records: list[dict], subtask: str) -> dict:
    """Aggregate Layer 1 metrics for multi-target."""
    n = len(records)
    if n == 0:
        return {}

    props = MULTI_SUBTASK_TO_PROPS[subtask]
    outcomes = [r["layer1_outcome"] for r in records]
    fts_vals = [r["layer1_fts"] for r in records if r["layer1_fts"] is not None]

    scf = scaffold_consistency(
        [r["layer1_src"] for r in records],
        [r["layer1_pred"] for r in records],
    )

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
            "dual_sr": round(sum(x["layer1_outcome"] for x in items) / ni, 4),
            "avg_fts": sum(x["layer1_fts"] for x in items if x["layer1_fts"] is not None)
                       / max(1, sum(1 for x in items if x["layer1_fts"] is not None)),
        }
        for p in props:
            diff_stats[d][f"sr_{p}"] = round(
                sum(x[f"outcome_{p}"] for x in items) / ni, 4
            )
            diff_stats[d][f"best_{p}"] = round(
                sum(x[f"best_{p}"] for x in items) / ni, 4
            )
            deltas = [x[f"delta_{p}"] for x in items if x[f"delta_{p}"] is not None]
            diff_stats[d][f"mean_delta_{p}"] = round(sum(deltas) / len(deltas), 4) if deltas else None
            norm_vals = [x[f"norm_{p}"] for x in items if x.get(f"norm_{p}") is not None]
            diff_stats[d][f"mean_norm_{p}"] = round(sum(norm_vals) / len(norm_vals), 4) if norm_vals else None
        nmins = [x.get("norm_min") for x in items if x.get("norm_min") is not None]
        diff_stats[d]["mean_norm_min"] = round(sum(nmins) / len(nmins), 4) if nmins else None
        ngeos = [x.get("norm_geo") for x in items if x.get("norm_geo") is not None]
        diff_stats[d]["mean_norm_geo"] = round(sum(ngeos) / len(ngeos), 4) if ngeos else None

    summary = {
        "n": n,
        "dual_sr_pct": round(sum(outcomes) / n * 100, 2),
        "avg_fts": round(sum(fts_vals) / len(fts_vals), 4) if fts_vals else None,
        "scaffold_hard": scf["hard"],
        "scaffold_soft": scf["soft"],
        "by_difficulty": diff_stats,
    }
    for p in props:
        summary[f"sr_{p}_pct"] = round(
            sum(r[f"outcome_{p}"] for r in records) / n * 100, 2
        )
        summary[f"best_{p}"] = round(
            sum(r[f"best_{p}"] for r in records) / n, 4
        )
        norm_vals = [r[f"norm_{p}"] for r in records if r.get(f"norm_{p}") is not None]
        summary[f"mean_norm_{p}"] = round(sum(norm_vals) / len(norm_vals), 4) if norm_vals else None
    nmins = [r.get("norm_min") for r in records if r.get("norm_min") is not None]
    summary["mean_norm_min"] = round(sum(nmins) / len(nmins), 4) if nmins else None
    ngeos = [r.get("norm_geo") for r in records if r.get("norm_geo") is not None]
    summary["mean_norm_geo"] = round(sum(ngeos) / len(ngeos), 4) if ngeos else None
    return summary


# ─── Shared scaffold consistency ──────────────────────────────────────────────

def scaffold_consistency(src_list: list, pred_list: list) -> dict:
    """V1-aligned scaffold consistency."""
    assert len(src_list) == len(pred_list)
    total = len(src_list)
    if total == 0:
        return {"hard": 0.0, "soft": 0.0}

    count_same = 0
    scores = []

    for src_smi, pred_smi in zip(src_list, pred_list):
        m_src = Chem.MolFromSmiles(src_smi) if src_smi else None
        m_pred = Chem.MolFromSmiles(pred_smi) if pred_smi else None

        if m_src is None or m_pred is None:
            scores.append(0.0)
            continue

        try:
            sc_src = MurckoScaffoldSmiles(src_smi)
            sc_pred = MurckoScaffoldSmiles(pred_smi)
        except Exception:
            scores.append(0.0)
            continue

        if sc_src == sc_pred:
            count_same += 1
            scores.append(1.0)
        else:
            msc_src = Chem.MolFromSmiles(sc_src) if sc_src else None
            msc_pred = Chem.MolFromSmiles(sc_pred) if sc_pred else None
            if msc_src is None or msc_pred is None:
                scores.append(0.0)
                continue
            mcs = rdFMCS.FindMCS([msc_src, msc_pred])
            if mcs.numAtoms > 0:
                fp1 = AllChem.GetMorganFingerprintAsBitVect(msc_src, 2, 1024)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(msc_pred, 2, 1024)
                scores.append(float(DataStructs.TanimotoSimilarity(fp1, fp2)))
            else:
                scores.append(0.0)

    return {
        "hard": round(count_same / total, 4),
        "soft": round(sum(scores) / total, 4),
    }
