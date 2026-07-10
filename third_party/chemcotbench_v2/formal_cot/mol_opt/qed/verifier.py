"""
Verifier for mol_opt/qed formal A→B CoT output (Unified Step Format, 5 steps).

all_pass = S1 AND S2 AND S3 AND S4 AND S5
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from baselines.cot_eval.mol_opt.mol_opt_single_structured.utils import (
    check_fg_v6_smiles,
    check_improvement,
    compute_delta,
    get_oracle,
    scaffold_preserved_per_sample,
)

_oracle = None

def _get_oracle():
    global _oracle
    if _oracle is None:
        _oracle = get_oracle("qed")
    return _oracle


def _canonical(smiles: str) -> str:
    if not smiles:
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def _scaffold_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


_NONE_VARIANTS = {"none", "n/a", "na", "-", ""}

def _is_none(v: str) -> bool:
    return v.strip().lower() in _NONE_VARIANTS


def check_s1(record: dict) -> bool:
    src = record.get("src_mol", "")
    claimed = record.get("step1_scaffold_smiles", "")
    if not claimed:
        return False
    expected = _scaffold_smiles(src)
    return bool(_canonical(expected) and _canonical(claimed) and _canonical(expected) == _canonical(claimed))


def check_s2(record: dict) -> bool:
    removed = record.get("step2_fg_removed", "")
    added = record.get("step2_fg_added", "")
    return bool(removed is not None and added is not None
                and (not _is_none(removed) or not _is_none(added)))


def check_s3(record: dict) -> bool:
    pred = record.get("step3_predicted_smiles", "")
    return bool(pred and Chem.MolFromSmiles(pred) is not None)


def check_s4(record: dict) -> bool:
    src = record.get("src_mol", "")
    pred = record.get("step3_predicted_smiles", "")
    claimed = record.get("step4_scaffold_claimed", "")
    if not claimed:
        return False
    expected = scaffold_preserved_per_sample(src, pred)
    return claimed.lower() == ("yes" if expected else "no")


def check_s5(record: dict) -> bool:
    fg_removed = record.get("step2_fg_removed", "")
    fg_added = record.get("step2_fg_added", "")
    src = record.get("src_mol", "")
    pred = record.get("step3_predicted_smiles", "")
    mol_src = Chem.MolFromSmiles(src) if src else None
    mol_pred = Chem.MolFromSmiles(pred) if pred else None
    return check_fg_v6_smiles(fg_removed, fg_added, mol_src, mol_pred)


def check_outcome(record: dict) -> bool:
    oracle = _get_oracle()
    src = record.get("src_mol", "")
    pred = record.get("step3_predicted_smiles", "")
    return check_improvement(oracle, src, pred)


def get_oracle_delta(record: dict) -> float:
    oracle = _get_oracle()
    src = record.get("src_mol", "")
    pred = record.get("step3_predicted_smiles", "")
    return compute_delta(oracle, src, pred)


def verify_one(record: dict) -> dict:
    if not record.get("api_success", False):
        updated = dict(record)
        for k in ("S1_scaffold", "S2_edit_plan", "S3_product", "S4_scaffold", "S5_fg", "outcome", "all_pass"):
            updated[k] = False
        updated["oracle_delta"] = 0.0
        updated["verify_skip_reason"] = "api_failure"
        return updated

    if not record.get("parse_ok", False):
        updated = dict(record)
        for k in ("S1_scaffold", "S2_edit_plan", "S3_product", "S4_scaffold", "S5_fg", "outcome", "all_pass"):
            updated[k] = False
        updated["oracle_delta"] = 0.0
        updated["verify_skip_reason"] = "parse_failed"
        return updated

    s1 = check_s1(record)
    s2 = check_s2(record)
    s3 = check_s3(record)
    s4 = check_s4(record)
    s5 = check_s5(record)
    oc = check_outcome(record)
    delta = get_oracle_delta(record)
    all_pass = s1 and s2 and s3 and s4 and s5

    updated = dict(record)
    updated.update({
        "S1_scaffold": s1,
        "S2_edit_plan": s2,
        "S3_product": s3,
        "S4_scaffold": s4,
        "S5_fg": s5,
        "outcome": oc,
        "oracle_delta": delta,
        "all_pass": all_pass,
        "verify_skip_reason": "",
    })
    return updated


def verify_all(records: list[dict]) -> tuple[list[dict], dict]:
    for i, rec in enumerate(records):
        records[i] = verify_one(rec)

    n = len(records)
    if n == 0:
        return records, {}

    def rate(key: str) -> float:
        return round(sum(1 for r in records if r.get(key)) / n, 4)

    checkpoint_rates = {
        "S1_scaffold_rate": rate("S1_scaffold"),
        "S2_edit_plan_rate": rate("S2_edit_plan"),
        "S3_product_rate": rate("S3_product"),
        "S4_scaffold_rate": rate("S4_scaffold"),
        "S5_fg_rate": rate("S5_fg"),
        "outcome_rate": rate("outcome"),
    }

    vpoint_correspondence = {
        "V1 [SCAFFOLD_IDENTIFICATION]  ≡ S1_scaffold": checkpoint_rates["S1_scaffold_rate"],
        "V2 [EDIT_PLAN]                ≡ S2_edit_plan": checkpoint_rates["S2_edit_plan_rate"],
        "V3 [PRODUCT_CONSTRUCTION]     ≡ S3_product": checkpoint_rates["S3_product_rate"],
        "V4 [SCAFFOLD_PRESERVATION]    ≡ S4_scaffold": checkpoint_rates["S4_scaffold_rate"],
        "V5 [FG_CHANGE_VERIFICATION]   ≡ S5_fg": checkpoint_rates["S5_fg_rate"],
        "Outcome (SR%)                 ≡ outcome": checkpoint_rates["outcome_rate"],
    }

    all_pass_rate = rate("all_pass")
    outcome_acc = rate("outcome")

    by_diff: dict[str, list] = {}
    for r in records:
        by_diff.setdefault(r["difficulty"], []).append(r)
    diff_stats = {
        d: {
            "n": len(recs),
            "all_pass_rate": round(sum(r.get("all_pass", False) for r in recs) / len(recs), 4),
            "outcome_acc": round(sum(r.get("outcome", False) for r in recs) / len(recs), 4),
            "S1_rate": round(sum(r.get("S1_scaffold", False) for r in recs) / len(recs), 4),
            "S2_rate": round(sum(r.get("S2_edit_plan", False) for r in recs) / len(recs), 4),
            "S3_rate": round(sum(r.get("S3_product", False) for r in recs) / len(recs), 4),
            "S4_rate": round(sum(r.get("S4_scaffold", False) for r in recs) / len(recs), 4),
            "S5_rate": round(sum(r.get("S5_fg", False) for r in recs) / len(recs), 4),
            "avg_delta": round(sum(r.get("oracle_delta", 0.0) for r in recs) / len(recs), 4),
        }
        for d, recs in sorted(by_diff.items())
    }

    s1_fails = [r for r in records if not r.get("S1_scaffold") and r.get("api_success")]
    s2_fails = [r for r in records if not r.get("S2_edit_plan") and r.get("api_success")]
    s3_fails = [r for r in records if not r.get("S3_product") and r.get("api_success")]
    s4_fails = [r for r in records if not r.get("S4_scaffold") and r.get("api_success")]
    s5_fails = [r for r in records if not r.get("S5_fg") and r.get("api_success")]

    summary = {
        "n_total": n,
        "n_parsed_ok": sum(1 for r in records if r.get("parse_ok", False)),
        "n_all_pass": sum(1 for r in records if r.get("all_pass", False)),
        "all_pass_rate": all_pass_rate,
        "outcome_acc": outcome_acc,
        "avg_oracle_delta": round(sum(r.get("oracle_delta", 0.0) for r in records) / n, 4),
        **checkpoint_rates,
        "vpoint_correspondence": vpoint_correspondence,
        "by_difficulty": diff_stats,
        "n_s1_fails": len(s1_fails),
        "n_s2_fails": len(s2_fails),
        "n_s3_fails": len(s3_fails),
        "n_s4_fails": len(s4_fails),
        "n_s5_fails": len(s5_fails),
        "token_totals": {
            "reasoning": sum(r.get("reasoning_tokens", 0) for r in records),
            "total_completion": sum(r.get("total_completion_tokens", 0) for r in records),
            "prompt": sum(r.get("prompt_tokens", 0) for r in records),
        },
    }

    print("\nVerification summary:")
    keys = ("S1_scaffold", "S2_edit_plan", "S3_product", "S4_scaffold", "S5_fg", "outcome", "all_pass")
    for k in keys:
        cnt = sum(1 for r in records if r.get(k))
        print(f"  {k:20s}: {cnt}/{n} ({100*cnt/n:.1f}%)")

    return records, summary
