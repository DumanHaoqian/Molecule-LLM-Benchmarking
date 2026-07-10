"""
Verifier for rxn_pred/yield_pred formal A→B CoT (regression).

Checkpoints:
  S1_rxn_class          — Type I: coarse RXN_CLASS exact match with GT
  S2_halide_type        — Type I: HALIDE_TYPE matches GT halide
  S3_nucleophile_fmt    — Type I: NUCLEOPHILE_FORM in valid set
  S4_ligand_class_fmt   — Type I: LIGAND_CLASS in {high-performance, standard, poor}
  S5_yield_numeric      — Type I: PREDICTED_YIELD is numeric
  S6_yield_range        — Type I: 0 <= PREDICTED_YIELD <= 100
  S7_answer_consistent  — Type I: Answer == PREDICTED_YIELD
  outcome               — Type II: exact match (|err| < 0.5) + abs_err (INFO)

  all_pass = S1 ∧ S2 ∧ S3 ∧ S4 ∧ S5 ∧ S6 ∧ S7 ∧ outcome

Regression metrics: MAE, RMSE, within_5, within_10
"""

import json
import math
import re
from collections import defaultdict

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from formal_cot.rxn_pred._coarse_mapping import map_to_coarse


# ── RDKit SMARTS for aryl halides ─────────────────────────────────────────────
try:
    from rdkit import Chem
    _ARYL_CL = Chem.MolFromSmarts('[c][Cl]')
    _ARYL_BR = Chem.MolFromSmarts('[c][Br]')
    _ARYL_I  = Chem.MolFromSmarts('[c][I]')
    _RDKIT_OK = True
except ImportError:
    _RDKIT_OK = False
    _ARYL_CL = _ARYL_BR = _ARYL_I = None


def _halide_from_smiles(smiles_str: str) -> str | None:
    if not _RDKIT_OK or not smiles_str:
        return None
    fragments = smiles_str.split(".")
    for frag in fragments:
        frag = frag.strip()
        if not frag:
            continue
        try:
            mol = Chem.MolFromSmiles(frag)
            if mol is None:
                continue
            if mol.HasSubstructMatch(_ARYL_CL):
                return "aryl chloride"
            if mol.HasSubstructMatch(_ARYL_BR):
                return "aryl bromide"
            if mol.HasSubstructMatch(_ARYL_I):
                return "aryl iodide"
        except Exception:
            continue
    return None


def _halide_from_label(label_str: str) -> str | None:
    if not label_str:
        return None
    t = label_str.upper()
    if re.search(r'-CL[-,\s]|[^A-Z]CL[^A-Z]|^CL$|\bCHLORIDE\b|\bCL\b', t):
        return "aryl chloride"
    if re.search(r'-BR[-,\s]|[^A-Z]BR[^A-Z]|^BR$|\bBROMIDE\b', t):
        return "aryl bromide"
    if re.search(r'-OTF[-,\s]|\bOTF\b|TRIFLATE', t):
        return "aryl triflate"
    if re.search(r'-I[-,\s]|[^A-Z]I[^A-Z]|^I$|\bIODIDE\b', t):
        return "aryl iodide"
    if "BPIN" in t or "BORONIC" in t or "BORONATE" in t:
        return None
    return None


def _extract_gt_halide(record: dict) -> str | None:
    reactants = record.get("meta_reactants", [])
    if isinstance(reactants, str):
        reactants = [reactants]
    for r in reactants:
        r_str = str(r).strip()
        if not r_str:
            continue
        if any(c in r_str for c in ("(", "=", "[", "%", "#")):
            h = _halide_from_smiles(r_str)
            if h:
                return h
        else:
            h = _halide_from_label(r_str)
            if h:
                return h
    reactants_str = record.get("reactants", "")
    if reactants_str:
        h = _halide_from_smiles(reactants_str)
        if h:
            return h
    return None


# ── Type I checks ─────────────────────────────────────────────────────────────

def _check_s1(record: dict) -> bool:
    gt_fine = record.get("rxn_cls", "")
    gt_coarse = map_to_coarse(gt_fine)
    pred = (record.get("step1_rxn_class") or "").strip()
    return pred == gt_coarse


def _check_s2(record: dict) -> bool:
    gt_halide = _extract_gt_halide(record)
    model_halide = record.get("step2_halide_type", "")
    if gt_halide is None:
        return True
    return model_halide == gt_halide


def _check_s3(record: dict) -> bool:
    val = record.get("step3_nucleophile_form", "")
    return val in {"free boronic acid", "boronate ester", "trifluoroborate", "amine", "not provided"}


def _check_s4(record: dict) -> bool:
    val = record.get("step4_ligand_class", "")
    return val in {"high-performance", "standard", "poor"}


def _check_s5(record: dict) -> bool:
    val = record.get("step5_predicted_yield")
    if val is None:
        return False
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def _check_s6(record: dict) -> bool:
    val = record.get("step5_predicted_yield")
    if val is None:
        return False
    try:
        v = float(val)
        return 0 <= v <= 100
    except (ValueError, TypeError):
        return False


def _check_s7(record: dict) -> bool:
    pred = record.get("step5_predicted_yield")
    ans = record.get("answer_yield")
    if pred is None or ans is None:
        return False
    try:
        return float(pred) == float(ans)
    except (ValueError, TypeError):
        return False


def _check_outcome(record: dict) -> tuple[bool, float]:
    pred = record.get("step5_predicted_yield")
    gt = record.get("gt_float")
    if pred is None or gt is None:
        return False, float("inf")
    try:
        pv = float(pred)
        gv = float(gt)
        err = abs(pv - gv)
        exact = err < 0.5
        return exact, err
    except (ValueError, TypeError):
        return False, float("inf")


# ── Regression metrics ────────────────────────────────────────────────────────

def _regression_metrics(records: list[dict]) -> dict:
    rows = []
    for r in records:
        if not r.get("parse_ok"):
            continue
        pred = r.get("step5_predicted_yield")
        gt = r.get("gt_float")
        if pred is None or gt is None:
            continue
        try:
            pv = float(pred)
            gv = float(gt)
            rows.append(abs(pv - gv))
        except (ValueError, TypeError):
            continue
    if not rows:
        return {"MAE": 0.0, "RMSE": 0.0, "within_5": 0.0, "within_10": 0.0, "n": 0}
    n = len(rows)
    mae = sum(rows) / n
    rmse = math.sqrt(sum(e ** 2 for e in rows) / n)
    within_5 = sum(1 for e in rows if e <= 5.0) / n
    within_10 = sum(1 for e in rows if e <= 10.0) / n
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "within_5": round(within_5, 4), "within_10": round(within_10, 4),
            "n": n}


# ── Main verify function ───────────────────────────────────────────────────────

def verify_all(
    records: list[dict],
    api_key: str | None = None,
) -> tuple[list[dict], dict]:
    """Verify all records. Returns (annotated_records, summary).
    api_key accepted for signature compatibility but is NOT used.
    """
    del api_key

    for r in records:
        if not r.get("api_success"):
            r.update({
                "S1_rxn_class": False, "S2_halide_type": False,
                "S3_nucleophile_fmt": False, "S4_ligand_class_fmt": False,
                "S5_yield_numeric": False, "S6_yield_range": False,
                "S7_answer_consistent": False, "outcome": False,
                "abs_err": None, "all_pass": False,
                "verify_skip_reason": "api_failure",
            })
            continue
        if not r.get("parse_ok"):
            r.update({
                "S1_rxn_class": False, "S2_halide_type": False,
                "S3_nucleophile_fmt": False, "S4_ligand_class_fmt": False,
                "S5_yield_numeric": False, "S6_yield_range": False,
                "S7_answer_consistent": False, "outcome": False,
                "abs_err": None, "all_pass": False,
                "verify_skip_reason": "parse_failed",
            })
            continue

        s1 = _check_s1(r)
        s2 = _check_s2(r)
        s3 = _check_s3(r)
        s4 = _check_s4(r)
        s5 = _check_s5(r)
        s6 = _check_s6(r)
        s7 = _check_s7(r)
        oc, err = _check_outcome(r)

        all_pass = s1 and s2 and s3 and s4 and s5 and s6 and s7 and oc

        r.update({
            "S1_rxn_class": s1, "S2_halide_type": s2,
            "S3_nucleophile_fmt": s3, "S4_ligand_class_fmt": s4,
            "S5_yield_numeric": s5, "S6_yield_range": s6,
            "S7_answer_consistent": s7,
            "outcome": oc, "abs_err": round(err, 4) if err != float("inf") else None,
            "all_pass": all_pass,
        })

    n = len(records)
    n_parsed = sum(1 for r in records if r.get("parse_ok"))

    def rate(key: str) -> float:
        return round(sum(1 for r in records if r.get(key)) / n, 4) if n > 0 else 0.0

    def rate_by_key(recs: list[dict], key: str) -> float:
        nd = len(recs)
        return round(sum(1 for r in recs if r.get(key)) / nd, 4) if nd > 0 else 0.0

    by_diff: dict[str, list] = defaultdict(list)
    for r in records:
        by_diff[r.get("difficulty", "unknown")].append(r)

    diff_stats = {}
    for diff, recs in sorted(by_diff.items()):
        nd = len(recs)
        diff_metrics = _regression_metrics(recs)
        diff_stats[diff] = {
            "n": nd,
            "all_pass_rate": round(sum(1 for r in recs if r.get("all_pass")) / nd, 4) if nd > 0 else 0.0,
            "outcome_acc": round(sum(1 for r in recs if r.get("outcome")) / nd, 4) if nd > 0 else 0.0,
            "S1_rate": rate_by_key(recs, "S1_rxn_class"),
            "S5_rate": rate_by_key(recs, "S5_yield_numeric"),
            "S7_rate": rate_by_key(recs, "S7_answer_consistent"),
            "MAE": diff_metrics["MAE"],
            "RMSE": diff_metrics["RMSE"],
            "within_5": diff_metrics["within_5"],
            "within_10": diff_metrics["within_10"],
        }

    overall_metrics = _regression_metrics(records)
    valid_records = [r for r in records if r.get("api_success") and r.get("parse_ok")]

    summary = {
        "n_total": n,
        "n_parsed_ok": n_parsed,
        "n_all_pass": sum(1 for r in records if r.get("all_pass")),
        "all_pass_rate": rate("all_pass"),
        "outcome_acc": rate("outcome"),
        "all_pass_formula": "S1 ∧ S2 ∧ S3 ∧ S4 ∧ S5 ∧ S6 ∧ S7 ∧ outcome",
        "S1_rxn_class_rate": rate("S1_rxn_class"),
        "S2_halide_type_rate": rate("S2_halide_type"),
        "S3_nucleophile_fmt_rate": rate("S3_nucleophile_fmt"),
        "S4_ligand_class_fmt_rate": rate("S4_ligand_class_fmt"),
        "S5_yield_numeric_rate": rate("S5_yield_numeric"),
        "S6_yield_range_rate": rate("S6_yield_range"),
        "S7_answer_consistent_rate": rate("S7_answer_consistent"),
        "outcome_rate": rate("outcome"),
        "regression_metrics": overall_metrics,
        "by_difficulty": diff_stats,
        "token_totals": {
            "reasoning": sum(r.get("reasoning_tokens", 0) for r in records),
            "total_completion": sum(r.get("total_completion_tokens", 0) for r in records),
            "prompt": sum(r.get("prompt_tokens", 0) for r in records),
        },
    }

    return records, summary
