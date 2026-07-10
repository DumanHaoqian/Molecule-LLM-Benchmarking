"""
Verifier for rxn_pred/forward formal A→B CoT output.

Checkpoint design
─────────────────
  S1_fg_grounding    (Type I, GATES)  — step1 FG_LIST: ≥1 FG found in reactants
                                         via RDKit SMARTS; ≥50% match rate
  S2_rxn_type_exact  (Type I, GATES)  — step2 RXN_TYPE exactly matches GT
                                         coarse_rxn_cls (one of 9 categories)
  S3_mechanism       (info only)      — MECHANISM_KWORD in 92-term dict
  S4_mol_valid       (Type I, GATES)  — PREDICTED_SMILES is RDKit-parseable
  S5_bond_formed     (info only)      — BOND_FORMED type is a new bond in product
                                         vs reactants; informational because oxidant
                                         bonds cause systematic false-negatives
  outcome            (Type I, GATES)  — canonical(PREDICTED_SMILES) == canonical(GT)

  all_pass = S1 AND S2 AND S4 AND outcome
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.cot_eval.rxn_predict.forward_structured.utils import (
    fg_grounding_score,
    mechanism_keyword_score,
    bond_formed_score,
    smiles_match_set,
    all_frags_valid,
    canonical_smiles,
)


# ── Rule-based checkpoint functions ──────────────────────────────────────────

def check_s1(record: dict) -> bool:
    """S1 (GATES): ≥1 FG in step1 FG_LIST verified via RDKit SMARTS on reactants."""
    fg_list: list[str] = record.get("step1_fg_list", [])
    if not fg_list:
        return False
    fg_text = " ".join(fg_list)
    reactants_list: list[str] = record.get("reactants_list", [])
    if not reactants_list:
        reactants_smi = record.get("reactants_smiles", "")
        reactants_list = [s for s in reactants_smi.split(".") if s.strip()]
    return bool(fg_grounding_score(fg_text, reactants_list))


def check_s2_coarse(record: dict) -> bool:
    """S2 (GATES): predicted coarse reaction type exactly matches GT coarse category."""
    predicted = (record.get("step2_rxn_type", "") or "").strip()
    gt = (record.get("coarse_rxn_cls", "") or "").strip()
    return bool(predicted) and bool(gt) and predicted.lower() == gt.lower()


def check_s3(record: dict) -> bool:
    """S3 (INFO ONLY): MECHANISM_KWORD is in the 92-term MECHANISM_KEYWORDS dict."""
    return bool(mechanism_keyword_score(record.get("step3_mechanism", "")))


def check_s4(record: dict) -> bool:
    """S4 (GATES): PREDICTED_SMILES is a valid RDKit-parseable molecule."""
    pred = record.get("step4_predicted_smi", "")
    return bool(pred and all_frags_valid(pred))


def check_s5(record: dict) -> bool:
    """S5 (INFO ONLY): BOND_FORMED type appears as a new bond in predicted product.

    Informational only: oxidation reactions (Dess-Martin, Oxone, etc.) show
    systematic false-negatives because the oxidant itself carries the same bond
    type, inflating the reactant bond count above the product.
    """
    bond_text      = record.get("step6_bond_formed", "")
    reactants_list = record.get("reactants_list", [])
    pred_smiles    = record.get("step4_predicted_smi", "")

    if not reactants_list:
        reactants_smi  = record.get("reactants_smiles", "")
        reactants_list = [s for s in reactants_smi.split(".") if s.strip()]

    if not bond_text or not reactants_list or not pred_smiles:
        return False
    return bool(bond_formed_score(bond_text, reactants_list, pred_smiles))


def check_outcome(record: dict) -> bool:
    """outcome (GATES): canonical(predicted SMILES) == canonical(GT product SMILES)."""
    pred = record.get("step4_predicted_smi", "")
    gt   = record.get("gt_product_smiles", "")
    if not pred or not gt:
        return False
    return smiles_match_set(pred, gt)


# ── Single-record verifier ────────────────────────────────────────────────────

def verify_one(record: dict) -> dict:
    """Run all checkpoints for one parsed record.

    all_pass = S1 AND S2 AND S4 AND outcome
    """
    updated = dict(record)

    # Short-circuit on API/parse failure
    if not record.get("api_success", False):
        for k in ("S1_fg_grounding", "S2_rxn_type_exact", "S3_mechanism",
                  "S4_mol_valid", "S5_bond_formed", "outcome", "all_pass"):
            updated[k] = False
        updated["verify_skip_reason"] = "api_failure"
        return updated

    if not record.get("parse_ok", False):
        for k in ("S1_fg_grounding", "S2_rxn_type_exact", "S3_mechanism",
                  "S4_mol_valid", "S5_bond_formed", "outcome", "all_pass"):
            updated[k] = False
        updated["verify_skip_reason"] = "parse_failed"
        return updated

    # Rule-based checks
    s1 = check_s1(record)
    s2 = check_s2_coarse(record)  # GATES
    s3 = check_s3(record)         # informational
    s4 = check_s4(record)
    s5 = check_s5(record)         # informational
    oc = check_outcome(record)

    all_pass = s1 and s2 and s4 and oc

    pred_can = canonical_smiles(record.get("step4_predicted_smi", "")) or ""
    gt_can   = canonical_smiles(record.get("gt_product_smiles",   "")) or ""

    updated.update({
        "S1_fg_grounding":    s1,
        "S2_rxn_type_exact":  s2,
        "S3_mechanism":       s3,    # informational
        "S4_mol_valid":       s4,
        "S5_bond_formed":     s5,    # informational
        "outcome":            oc,
        "all_pass":           all_pass,
        "predicted_can":      pred_can,
        "gt_can":             gt_can,
        "verify_skip_reason": "",
    })
    return updated


# ── Batch verifier ────────────────────────────────────────────────────────────

def verify_all(
    records: list[dict],
) -> tuple[list[dict], dict]:
    """Verify all records. Returns (updated_records, summary_dict)."""
    print("\n--- Running rule verification (coarse rxn type exact match) ---")

    for i, rec in enumerate(records):
        records[i] = verify_one(rec)

    n = len(records)
    if n == 0:
        return records, {}

    def rate(key: str) -> float:
        vals = [r for r in records if r.get(key) is not None]
        return round(sum(1 for r in vals if r.get(key)) / n, 4)

    checkpoint_rates = {
        "S1_fg_grounding_rate":   rate("S1_fg_grounding"),
        "S2_rxn_type_exact_rate": rate("S2_rxn_type_exact"),
        "S3_mechanism_rate":      rate("S3_mechanism"),
        "S4_mol_valid_rate":      rate("S4_mol_valid"),
        "S5_bond_formed_rate":    rate("S5_bond_formed"),
        "outcome_rate":           rate("outcome"),
    }

    # Per-difficulty breakdown
    by_diff: dict[str, list] = {}
    for r in records:
        by_diff.setdefault(r.get("difficulty", "unknown"), []).append(r)

    diff_stats = {}
    for d, recs in sorted(by_diff.items()):
        entry = {
            "n":             len(recs),
            "all_pass_rate": round(sum(r.get("all_pass",        False) for r in recs) / len(recs), 4),
            "outcome_acc":   round(sum(r.get("outcome",         False) for r in recs) / len(recs), 4),
            "S1_rate":       round(sum(r.get("S1_fg_grounding", False) for r in recs) / len(recs), 4),
            "S2_rate":       round(sum(r.get("S2_rxn_type_exact",False) for r in recs) / len(recs), 4),
            "S4_rate":       round(sum(r.get("S4_mol_valid",    False) for r in recs) / len(recs), 4),
            "S5_rate":       round(sum(r.get("S5_bond_formed",  False) for r in recs) / len(recs), 4),
        }
        diff_stats[d] = entry

    api_ok    = [r for r in records if r.get("api_success")]
    s1_fails  = [r for r in api_ok if not r.get("S1_fg_grounding")]
    s2_fails  = [r for r in api_ok if not r.get("S2_rxn_type_exact")]
    s4_fails  = [r for r in api_ok if not r.get("S4_mol_valid")]
    oc_fails  = [r for r in api_ok if not r.get("outcome")]

    summary = {
        "n_total":           n,
        "n_parsed_ok":       sum(1 for r in records if r.get("parse_ok",  False)),
        "n_all_pass":        sum(1 for r in records if r.get("all_pass",  False)),
        "all_pass_rate":     rate("all_pass"),
        "outcome_acc":       rate("outcome"),
        "all_pass_formula":  "S1 AND S2 AND S4 AND outcome",
        **checkpoint_rates,
        "by_difficulty":     diff_stats,
        "n_s1_fails":        len(s1_fails),
        "n_s2_fails":        len(s2_fails),
        "n_s4_fails":        len(s4_fails),
        "n_oc_fails":        len(oc_fails),
        "token_totals": {
            "reasoning":        sum(r.get("reasoning_tokens",        0) for r in records),
            "total_completion": sum(r.get("total_completion_tokens", 0) for r in records),
            "prompt":           sum(r.get("prompt_tokens",           0) for r in records),
        },
    }

    # Console summary
    print("\nVerification summary:")
    info_keys = {"S3_mechanism", "S5_bond_formed"}
    for k, r_val in checkpoint_rates.items():
        cnt   = sum(1 for rec in records if rec.get(k.replace("_rate", "")))
        label = k.replace("_rate", "")
        tag   = " (info)" if label in info_keys else ""
        print(f"  {label:22s}: {cnt}/{n} ({100*r_val:.1f}%){tag}")
    ap = sum(1 for r in records if r.get("all_pass"))
    print(f"  {'all_pass':22s}: {ap}/{n} ({100*ap/n:.1f}%)  [S1∧S2∧S4∧outcome]")

    return records, summary
