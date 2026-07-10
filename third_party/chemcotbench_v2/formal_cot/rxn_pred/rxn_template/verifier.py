"""
Verifier for rxn_pred/rxn_template formal A→B CoT output.

Checkpoint design
─────────────────
  S1_bond_changes_valid    (Type I, GATES)   — BOND_CHANGES string is non-empty
                                               and contains both "broken" and
                                               "formed" keywords (case-insensitive).
  S2_rxn_type_exact        (Type I, GATES)   — step2 RXN_TYPE exactly matches GT
                                               coarse_rxn_cls (one of 9 categories).
  S3_mechanism             (info only)       — MECHANISM_KWORD exists in the 92-term
                                               mechanism dictionary (from utils.py).
  S4_smarts_parseable      (Type I, GATES)   — AllChem.ReactionFromSmarts(PROPOSED_SMARTS)
                                               returns a non-None RDKit reaction object.
  S4b_smarts_exact_match   (info only)       — PROPOSED_SMARTS string-matches GT
                                               correct_template exactly (post-hoc,
                                               zero-cost).  False can indicate either
                                               (a) model generated a wrong SMARTS, or
                                               (b) model generated a valid alternative
                                               SMARTS (e.g., more specific nucleophile)
                                               — informational only; not gating because
                                               chemically equivalent SMARTS may differ
                                               in atom numbering / bracket style.
  S5_smarts_match          (info only)       — PROPOSED_SMARTS reactant template can
                                               be run on the actual reactants with
                                               RunReactants producing ≥1 product.
                                               Info-only due to specificity sensitivity.
  outcome                  (Type I, GATES)   — SELECTED_OPTION.upper() == gt_letter.

  all_pass = S1 AND S2 AND S4   (S4b and outcome are informational)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from baselines.cot_eval.rxn_predict.forward_structured.utils import (
        mechanism_keyword_score,
    )
    _HAS_UTILS = True
except ImportError:
    _HAS_UTILS = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdBase
    rdBase.DisableLog("rdApp.*")
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False


# ── Rule-based checkpoints ─────────────────────────────────────────────────────

def check_s1_bond_changes(record: dict) -> bool:
    """S1 (GATES): BOND_CHANGES non-empty, contains 'broken' AND 'formed'."""
    bc = record.get("step1_bond_changes", "") or ""
    if not bc.strip():
        return False
    low = bc.lower()
    return "broken" in low and "formed" in low


def check_s2_coarse(record: dict) -> bool:
    """S2 (GATES): predicted coarse reaction type exactly matches GT coarse category."""
    predicted = (record.get("step2_rxn_type", "") or "").strip()
    gt = (record.get("coarse_rxn_cls", "") or "").strip()
    return bool(predicted) and bool(gt) and predicted.lower() == gt.lower()


def check_s3_mechanism(record: dict) -> bool:
    """S3 (INFO ONLY): MECHANISM_KWORD in 92-term mechanism dictionary."""
    if not _HAS_UTILS:
        return False
    kw = record.get("step3_mechanism", "") or ""
    if not kw.strip():
        return False
    try:
        score = mechanism_keyword_score(kw)
        return score > 0
    except Exception:
        return False


def check_s4_smarts_parseable(record: dict) -> bool:
    """S4 (GATES): PROPOSED_SMARTS parseable by AllChem.ReactionFromSmarts()."""
    smarts = record.get("step4_proposed_smarts", "") or ""
    if not smarts.strip():
        return False
    if not _HAS_RDKIT:
        return True   # permissive fallback without RDKit
    try:
        rxn = AllChem.ReactionFromSmarts(smarts)
        return rxn is not None
    except Exception:
        return False


def check_s4b_smarts_exact_match(record: dict) -> bool:
    """S4b (GATES): PROPOSED_SMARTS string-equals GT correct_template.

    Post-hoc, zero-cost gate using the stored correct_template field.
    A False result may indicate:
      (a) Model generated a chemically wrong SMARTS — genuine reasoning error.
      (b) Model generated a valid but different SMARTS vs the GT archetype
          (e.g., uses the specific nucleophile for this reaction instance
          while the GT template uses a more archetypal substituent) — this
          reflects a dataset labelling nuance, not a model failure.
    S4b=False items are excluded from clean_dataset; manual review is
    recommended to distinguish (a) from (b).
    """
    proposed = (record.get("step4_proposed_smarts", "") or "").strip()
    gt       = (record.get("correct_template", "") or "").strip()
    if not proposed or not gt:
        return False
    return proposed == gt


def check_s5_smarts_match(record: dict) -> bool:
    """S5 (INFO ONLY): PROPOSED_SMARTS reactant pattern runs on actual reactants.

    Uses RunReactants() — info-only because specificity mismatches cause false
    negatives for otherwise valid SMARTS.
    """
    smarts          = record.get("step4_proposed_smarts", "") or ""
    reactants_smi   = record.get("reactants_smiles", "")    or ""
    if not smarts or not reactants_smi or not _HAS_RDKIT:
        return False
    try:
        rxn = AllChem.ReactionFromSmarts(smarts)
        if rxn is None:
            return False
        frags = [s for s in reactants_smi.split(".") if s]
        mols  = []
        for smi in frags:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                return False
            mols.append(m)
        if not mols:
            return False
        try:
            products = rxn.RunReactants(tuple(mols))
            return len(products) > 0
        except Exception:
            return False
    except Exception:
        return False


def check_outcome(record: dict) -> bool:
    """outcome (GATES): SELECTED_OPTION.upper() == gt_letter.upper()."""
    selected = (
        record.get("step6_selected_option", "")
        or record.get("answer_letter", "")
    ).upper()
    gt = record.get("gt_letter", "").upper()
    return bool(selected) and bool(gt) and selected == gt


# ── Single-record verifier ─────────────────────────────────────────────────────

def verify_one(record: dict) -> dict:
    """Run all checkpoints for one parsed record.

    all_pass = S1 AND S2 AND S4   (S4b and outcome are informational)
    """
    updated = dict(record)

    falsy_keys = (
        "S1_bond_changes_valid", "S2_rxn_type_exact", "S3_mechanism",
        "S4_smarts_parseable", "S4b_smarts_exact_match",
        "S5_smarts_match",
        "outcome", "all_pass",
    )

    if not record.get("api_success", False):
        for k in falsy_keys:
            updated[k] = False
        updated["verify_skip_reason"] = "api_failure"
        return updated

    if not record.get("parse_ok", False):
        for k in falsy_keys:
            updated[k] = False
        updated["verify_skip_reason"] = "parse_failed"
        return updated

    s1  = check_s1_bond_changes(record)
    s2  = check_s2_coarse(record)             # GATES
    s3  = check_s3_mechanism(record)          # info only
    s4  = check_s4_smarts_parseable(record)
    s4b = check_s4b_smarts_exact_match(record)  # info only
    s5  = check_s5_smarts_match(record)       # info only
    oc  = check_outcome(record)

    all_pass = s1 and s2 and s4 and oc

    updated.update({
        "S1_bond_changes_valid":  s1,
        "S2_rxn_type_exact":      s2,
        "S3_mechanism":           s3,
        "S4_smarts_parseable":    s4,
        "S4b_smarts_exact_match": s4b,
        "S5_smarts_match":        s5,
        "outcome":                oc,
        "all_pass":               all_pass,
        "verify_skip_reason":     "",
    })
    return updated


# ── Batch verifier ─────────────────────────────────────────────────────────────

def verify_all(
    records: list[dict],
) -> tuple[list[dict], dict]:
    """Verify all records. Returns (updated_records, summary_dict)."""
    print("\n--- Running RDKit verification (coarse rxn type exact match) ---")

    for i, rec in enumerate(records):
        records[i] = verify_one(rec)

    n = len(records)
    if n == 0:
        return records, {}

    def rate(key: str) -> float:
        vals = [r for r in records if r.get(key) is not None]
        if not vals:
            return 0.0
        return round(sum(1 for r in vals if r.get(key)) / n, 4)

    checkpoint_rates: dict[str, float] = {
        "S1_bond_changes_valid_rate":   rate("S1_bond_changes_valid"),
        "S2_rxn_type_exact_rate":       rate("S2_rxn_type_exact"),
        "S3_mechanism_rate":            rate("S3_mechanism"),
        "S4_smarts_parseable_rate":     rate("S4_smarts_parseable"),
        "S4b_smarts_exact_match_rate":  rate("S4b_smarts_exact_match"),
        "S5_smarts_match_rate":         rate("S5_smarts_match"),
        "outcome_rate":                 rate("outcome"),
    }

    # Per-difficulty breakdown
    by_diff: dict[str, list] = {}
    for r in records:
        by_diff.setdefault(r.get("difficulty", "unknown"), []).append(r)

    diff_stats: dict[str, dict] = {}
    for d, recs in sorted(by_diff.items()):
        nd    = len(recs)
        entry = {
            "n":               nd,
            "all_pass_rate":   round(sum(r.get("all_pass",                False) for r in recs) / nd, 4),
            "outcome_acc":     round(sum(r.get("outcome",                 False) for r in recs) / nd, 4),
            "S1_rate":         round(sum(r.get("S1_bond_changes_valid",   False) for r in recs) / nd, 4),
            "S2_rate":         round(sum(r.get("S2_rxn_type_exact",       False) for r in recs) / nd, 4),
            "S3_rate":         round(sum(r.get("S3_mechanism",            False) for r in recs) / nd, 4),
            "S4_rate":         round(sum(r.get("S4_smarts_parseable",     False) for r in recs) / nd, 4),
            "S4b_rate":        round(sum(r.get("S4b_smarts_exact_match",  False) for r in recs) / nd, 4),
            "S5_rate":         round(sum(r.get("S5_smarts_match",         False) for r in recs) / nd, 4),
        }
        diff_stats[d] = entry

    summary = {
        "n_total":           n,
        "n_parsed_ok":       sum(1 for r in records if r.get("parse_ok",  False)),
        "n_all_pass":        sum(1 for r in records if r.get("all_pass",  False)),
        "all_pass_rate":     rate("all_pass"),
        "outcome_acc":       rate("outcome"),
        "all_pass_formula":  "S1 AND S2 AND S4   (S4b and outcome are informational)",
        **checkpoint_rates,
        "by_difficulty": diff_stats,
        "token_totals":  {
            "reasoning":        sum(r.get("reasoning_tokens",        0) for r in records),
            "total_completion": sum(r.get("total_completion_tokens", 0) for r in records),
            "prompt":           sum(r.get("prompt_tokens",           0) for r in records),
        },
    }

    # Console summary
    info_keys = {"S3_mechanism", "S4b_smarts_exact_match", "S5_smarts_match", "outcome"}
    print("\nVerification summary:")
    for k, r_val in checkpoint_rates.items():
        label = k.replace("_rate", "")
        cnt   = sum(1 for rec in records if rec.get(label))
        tag   = " (info)" if label in info_keys else ""
        print(f"  {label:30s}: {cnt}/{n} ({100*r_val:.1f}%){tag}")
    ap      = sum(1 for r in records if r.get("all_pass"))
    print(f"  {'all_pass':30s}: {ap}/{n} ({100*ap/n:.1f}%)  [S1∧S2∧S4]")

    return records, summary
