"""Layer 1 evaluator: final answer correctness (Acc / MAE / Tanimoto)."""
import statistics

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from baselines.cot_eval.mol_und.murcko_scaffold_structured.utils import (
    canonical_smiles,
    smiles_are_equal,
)


def _tanimoto(smiles_a: str, smiles_b: str) -> float:
    try:
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        if mol_a is None or mol_b is None:
            return 0.0
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius=2, nBits=2048)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius=2, nBits=2048)
        return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
    except Exception:
        return 0.0


def evaluate_layer1(record: dict, subtask: str) -> dict:
    """Compute Layer 1 metrics for a single record."""
    if subtask == "smiles_equivalent":
        subtask = record.get("source_subtask", subtask)
    if subtask == "fg_detect":
        return _eval_fg_detect(record)
    elif subtask == "ring_count":
        return _eval_ring_count(record)
    elif subtask == "murcko_scaffold":
        return _eval_murcko_scaffold(record)
    elif subtask == "ring_sys_scaffold":
        return _eval_ring_sys_scaffold(record)
    elif subtask == "mutated":
        return _eval_mutated(record)
    elif subtask == "permutated":
        return _eval_permutated(record)
    return {}


def _eval_fg_detect(record: dict) -> dict:
    gt_count = record.get("gt_count")
    answer = record.get("answer")
    pred_int = int(answer) if answer is not None else None
    gt_int = int(gt_count) if gt_count is not None else None
    exact = pred_int == gt_int if pred_int is not None else False
    mae = abs(pred_int - gt_int) if (pred_int is not None and gt_int is not None) else (gt_int or 0)
    return {
        "layer1_exact_match": exact,
        "layer1_mae": mae,
    }


def _eval_ring_count(record: dict) -> dict:
    gt_count = record.get("gt_count")
    answer = record.get("answer")
    pred_int = int(answer) if answer is not None else None
    gt_int = int(gt_count) if gt_count is not None else None
    exact = pred_int == gt_int if pred_int is not None else False
    mae = abs(pred_int - gt_int) if (pred_int is not None and gt_int is not None) else (gt_int or 0)
    return {
        "layer1_exact_match": exact,
        "layer1_mae": mae,
    }


def _eval_murcko_scaffold(record: dict) -> dict:
    gt_scaffold = record.get("gt_scaffold", "")
    answer = record.get("answer", "")
    exact = smiles_are_equal(answer, gt_scaffold) if answer and gt_scaffold else False
    tan = _tanimoto(answer, gt_scaffold) if answer and gt_scaffold else 0.0
    return {
        "layer1_exact_match": exact,
        "layer1_tanimoto": round(tan, 4),
    }


def _eval_ring_sys_scaffold(record: dict) -> dict:
    gt_label = record.get("gt_label", "")
    answer = record.get("answer", "")
    exact = bool(answer and str(answer).strip() == str(gt_label).strip())
    return {
        "layer1_exact_match": exact,
    }


def _eval_mutated(record: dict) -> dict:
    answer = record.get("answer", "")
    exact = str(answer).strip() == "Different"
    return {
        "layer1_exact_match": exact,
    }


def _eval_permutated(record: dict) -> dict:
    answer = record.get("answer", "")
    exact = str(answer).strip() == "Same"
    return {
        "layer1_exact_match": exact,
    }


def summarize_layer1(records: list[dict], subtask: str) -> dict:
    """Aggregate Layer 1 metrics."""
    n = len(records)
    if n == 0:
        return {}

    exact_matches = [r.get("layer1_exact_match", False) for r in records]
    acc = sum(exact_matches) / n

    summary = {
        "n": n,
        "exact_match_acc": round(acc, 4),
    }

    if subtask in ("fg_detect", "ring_count"):
        maes = [r.get("layer1_mae", 0) for r in records]
        summary["mean_mae"] = round(statistics.mean(maes), 4)

    if subtask == "murcko_scaffold":
        tans = [r.get("layer1_tanimoto", 0.0) for r in records]
        summary["avg_tanimoto"] = round(statistics.mean(tans), 4)

    # By difficulty
    by_diff: dict[str, list] = {}
    for r in records:
        by_diff.setdefault(r.get("difficulty", "unknown"), []).append(r)

    diff_stats = {}
    for d, recs in by_diff.items():
        if not recs:
            continue
        nd = len(recs)
        diff_stats[d] = {
            "n": nd,
            "exact_match_acc": round(sum(x.get("layer1_exact_match", False) for x in recs) / nd, 4),
        }
        if subtask in ("fg_detect", "ring_count"):
            diff_stats[d]["mean_mae"] = round(statistics.mean(x.get("layer1_mae", 0) for x in recs), 4)
        if subtask == "murcko_scaffold":
            diff_stats[d]["avg_tanimoto"] = round(statistics.mean(x.get("layer1_tanimoto", 0.0) for x in recs), 4)

    summary["by_difficulty"] = diff_stats
    return summary
