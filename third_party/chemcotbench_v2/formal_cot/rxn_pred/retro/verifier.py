"""
Verifier for rxn_pred/retro formal A→B CoT output (v4).

Checkpoint design (v5)
─────────────────────
  S1_fg_grounding      (GATES) — step1 FG_LIST: ≥1 FG found in PRODUCT via RDKit SMARTS
                                 Exception: if outcome_pass=True (correct reactant SMILES),
                                 S1 is waived — RDKit FG SMARTS can be too strict for
                                 tertiary amines, charged aromatics, or unusual FG names.
  S2_rxn_type_exact    (GATES) — step2 RXN_TYPE exactly matches GT coarse_rxn_cls
  S3_mechanism         (info)  — MECHANISM_KWORD in 92-term MECHANISM_KEYWORDS dict
  S4_all_valid         (GATES) — all dot-separated fragments in REACTANT_SMILES are RDKit-parseable
  S5_bond_broken       (info)  — BOND_BROKEN type EXISTS in the product molecule
  S_logic_consistency  (info)  — step2 RXN_TYPE ↔ step6 BOND_BROKEN chemical compatibility
  S6_fwd_self_check    (info)  — step7 FWD_CONSISTENCY(match=yes) ← model self-assessment
  outcome_exact        (sub)   — canonical(REACTANT_SMILES) == canonical(GT reactants) [set match]
  outcome_fwd_sim      (sub)   — forward-simulate REACTANT_SMILES via RXN_TYPE template → product match
  outcome_pass         (GATES) — Formula D (see below)

  Strict formula:
    all_pass = S1 AND S2 AND S4 AND outcome_exact

  S_logic_consistency is informational only (removed from all_pass in v5).
  Reason: the retro task uses 9 coarse Schneider classes (e.g. "Acylation",
  "Heteroatom Alkylation and Arylation") that cover many bond types, so the
  fine-grained _BOND_COMPAT table causes systematic false negatives.  The check
  is still computed and stored for analysis but does not gate all_pass.
"""

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.cot_eval.rxn_predict.forward_structured.utils import (
    fg_grounding_score,
    mechanism_keyword_score,
    all_frags_valid,
    canonical_smiles,
    smiles_match_set,
    _extract_bond_types_from_text,
    _count_bond_types,
    _BOND_CHAR_ORDERS,
)


# ── S_logic_consistency (v4 new checkpoint) ───────────────────────────────────

def check_logic_consistency(record: dict) -> bool | None:
    """S_logic_consistency (GATES): step2 RXN_TYPE ↔ step6 BOND_BROKEN compatibility.

    Checks whether the BOND_BROKEN type declared in step6 is chemically compatible
    with the reaction type declared in step2.  This catches cases where the model
    states an internally inconsistent CoT (e.g. "Deprotection" with "C-S bond broken"
    is chemically impossible — deprotection never forms C-S bonds).

    Returns:
        True  — bond type IS compatible with the declared reaction type
        False — bond type is INCOMPATIBLE (logic error, gates all_pass=False)
        None  — reaction type is unknown or bond can't be parsed → no constraint

    Symmetric bond handling: "C-N" and "N-C" are treated as equivalent.
    """
    rxn_type  = record.get("step2_rxn_type", "")
    bond_text = record.get("step6_bond_broken", "")

    if not rxn_type or not bond_text:
        return None

    from formal_cot.rxn_pred.retro.rxn_template_lib import (
        get_compatible_bonds, normalize_bond_type,
    )

    compat_set = get_compatible_bonds(rxn_type)
    if compat_set is None:
        return None  # unknown rxn type → no constraint

    bond_norm = normalize_bond_type(bond_text)
    if not bond_norm:
        return None  # can't parse bond string → no constraint

    # Check both orderings (C-N == N-C)
    if bond_norm in compat_set:
        return True
    sep = "=" if "=" in bond_norm else "-"
    parts = bond_norm.split(sep, 1)
    if len(parts) == 2:
        rev = f"{parts[1]}{sep}{parts[0]}"
        if rev in compat_set:
            return True

    return False


# ── bond_broken_score (retro-specific) ───────────────────────────────────────

def bond_broken_score(bond_text: str, product_smiles: str) -> bool:
    """S5 (INFO ONLY): Verify that the declared BOND_BROKEN type exists in the product.

    Unlike forward's bond_formed_score (which checks if bond count INCREASED),
    this only checks if the bond type EXISTS in the product (count > 0).
    This is simpler and has no systematic false-negatives from oxidant artifacts.
    """
    if not bond_text or not product_smiles:
        return False

    stated = _extract_bond_types_from_text(bond_text)
    if not stated:
        return False

    product_counts = _count_bond_types([product_smiles])

    for a1, a2, bond_char in stated:
        for order in _BOND_CHAR_ORDERS.get(bond_char, [1.0]):
            key = (a1, a2, order)
            if product_counts.get(key, 0) > 0:
                return True
    return False


# ── Rule-based checkpoint functions ──────────────────────────────────────────

def check_s1(record: dict) -> bool:
    """S1 (GATES): ≥1 FG in step1 FG_LIST verified via RDKit SMARTS on PRODUCT."""
    fg_list: list[str] = record.get("step1_fg_list", [])
    if not fg_list:
        return False
    fg_text = " ".join(fg_list)
    product_smiles = record.get("product_smiles", "")
    if not product_smiles:
        return False
    # fg_grounding_score expects a list of SMILES; pass the product as a single-item list
    return bool(fg_grounding_score(fg_text, [product_smiles]))


def check_s2_coarse(record: dict) -> bool:
    """S2 (GATES): predicted coarse reaction type exactly matches GT coarse category."""
    predicted = (record.get("step2_rxn_type", "") or "").strip()
    gt = (record.get("coarse_rxn_cls", "") or "").strip()
    return bool(predicted) and bool(gt) and predicted.lower() == gt.lower()


def check_s3(record: dict) -> bool:
    """S3 (INFO ONLY): MECHANISM_KWORD is in the recognized mechanism keywords dict."""
    return bool(mechanism_keyword_score(record.get("step3_mechanism", "")))


def check_s4(record: dict) -> bool:
    """S4 (GATES): All dot-separated fragments in REACTANT_SMILES are RDKit-parseable."""
    pred = record.get("step4_reactant_smi", "")
    if not pred:
        pred = record.get("answer_smi", "")
    return bool(pred and all_frags_valid(pred))


def check_s5(record: dict) -> bool:
    """S5 (INFO ONLY): BOND_BROKEN type exists in the product molecule."""
    bond_text     = record.get("step6_bond_broken", "")
    product_smi   = record.get("product_smiles", "")
    return bond_broken_score(bond_text, product_smi)


def check_outcome(record: dict) -> bool:
    """outcome_exact (sub): canonical set of REACTANT_SMILES == canonical set of GT reactants."""
    pred = record.get("step4_reactant_smi", "")
    if not pred:
        pred = record.get("answer_smi", "")
    gt = record.get("gt_reactants", "")
    if not pred or not gt:
        return False
    return smiles_match_set(pred, gt)


def check_outcome_fwd_sim(record: dict) -> bool | None:
    """outcome_fwd_sim (sub): forward-simulate predicted reactants → check product match.

    Returns:
        True  — simulation produced a product matching GT product
        False — simulation ran but no product matched
        None  — no SMARTS template found for this reaction type (inconclusive)

    Uses rxn_template_lib.run_forward_sim internally.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from formal_cot.rxn_pred.retro.rxn_template_lib import run_forward_sim

    pred_smi    = record.get("step4_reactant_smi", "") or record.get("answer_smi", "")
    rxn_type    = record.get("step2_rxn_type", "")
    product_smi = record.get("product_smiles", "")

    if not pred_smi or not rxn_type or not product_smi:
        return None

    try:
        return run_forward_sim(pred_smi, rxn_type, product_smi)
    except Exception as exc:
        print(f"  [fwd_sim] ERROR: {exc}")
        return None


def check_s6_fwd_self_check(record: dict) -> bool:
    """S6_fwd_self_check (info): model's step7 FWD_CONSISTENCY(match=yes/no) self-assessment.

    v4: simply checks whether step7_fwd_match == "yes".
    Backward compat: if step7_fwd_match is empty but step7_fwd_product_check exists
    (old v2 data), falls back to SMILES comparison.
    """
    fwd_match = record.get("step7_fwd_match", "")
    if fwd_match:
        return fwd_match.lower() == "yes"

    # v2 backward compat: compare FWD_PRODUCT_CHECK SMILES
    fwd_check   = record.get("step7_fwd_product_check", "")
    product_smi = record.get("product_smiles", "")
    if not fwd_check or not product_smi:
        return False
    try:
        from rdkit.Chem import MolFromSmiles, MolToSmiles
        m1 = MolFromSmiles(fwd_check)
        m2 = MolFromSmiles(product_smi)
        if m1 is None or m2 is None:
            return False
        if MolToSmiles(m1, isomericSmiles=True) == MolToSmiles(m2, isomericSmiles=True):
            return True
        if MolToSmiles(m1, isomericSmiles=False) == MolToSmiles(m2, isomericSmiles=False):
            return True
        return False
    except Exception:
        return False


# ── Single-record verifier ────────────────────────────────────────────────────

def verify_one(record: dict) -> dict:
    """Run all checkpoints for one parsed record (v5).

    v5 all_pass formula:
        all_pass = S1 AND S2 AND S4
        (S_logic_consistency is informational only — coarse reaction classes
        like "Acylation" and "Heteroatom Alkylation/Arylation" cover many bond
        types, so the fine-grained bond compat table caused false negatives.)
    """
    updated = dict(record)

    _fail_keys = ("S1_fg_grounding", "S2_rxn_type_exact", "S3_mechanism",
                  "S4_all_valid", "S5_bond_broken", "S6_fwd_self_check",
                  "S_logic_consistency",
                  "outcome_exact", "outcome_fwd_sim", "outcome_pass",
                  "all_pass")

    if not record.get("api_success", False):
        for k in _fail_keys:
            updated[k] = False
        updated["verify_skip_reason"] = "api_failure"
        return updated

    if not record.get("parse_ok", False):
        for k in _fail_keys:
            updated[k] = False
        updated["verify_skip_reason"] = "parse_failed"
        return updated

    s1    = check_s1(record)
    s2    = check_s2_coarse(record)          # GATES
    s3    = check_s3(record)                 # informational
    s4    = check_s4(record)
    s5    = check_s5(record)                 # informational
    oc    = check_outcome(record)
    fwd   = check_outcome_fwd_sim(record)    # True / False / None
    s6    = check_s6_fwd_self_check(record)  # informational
    logic = check_logic_consistency(record)  # informational only (v5)

    outcome_pass = oc or (fwd is True)

    # Strict mode: outcome_exact must be True for all_pass
    formula_used = "S1∧S2∧S4∧outcome_exact"

    all_pass = s1 and s2 and s4 and oc  # logic_gate removed in v5 (informational only)

    pred_can = canonical_smiles(record.get("step4_reactant_smi", "")) or ""
    gt_can   = canonical_smiles(record.get("gt_reactants", "")) or ""

    updated.update({
        "S1_fg_grounding":     s1,
        "S2_rxn_type_exact":   s2,
        "S3_mechanism":        s3,
        "S4_all_valid":        s4,
        "S5_bond_broken":      s5,
        "S_logic_consistency": logic,
        "S6_fwd_self_check":   s6,
        "outcome_exact":       oc,
        "outcome_fwd_sim":     fwd,
        "outcome_pass":        outcome_pass,
        "all_pass":            all_pass,
        "formula_used":        formula_used,
        "predicted_can":       pred_can,
        "gt_can":              gt_can,
        "verify_skip_reason":  "",
    })
    return updated


# ── Batch verifier ────────────────────────────────────────────────────────────

def verify_all(
    records: list[dict],
) -> tuple[list[dict], dict]:
    """Verify all records. Returns (updated_records, summary_dict)."""
    print("\n--- Running RDKit + fwd_sim + logic_consistency + coarse rxn type exact match ---")

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

    # fwd_sim rate: only count records where fwd_sim is not None
    fwd_sim_records = [r for r in records if r.get("outcome_fwd_sim") is not None]
    fwd_sim_rate = (
        round(sum(1 for r in fwd_sim_records if r.get("outcome_fwd_sim")) / len(fwd_sim_records), 4)
        if fwd_sim_records else 0.0
    )
    fwd_sim_coverage = round(len(fwd_sim_records) / n, 4) if n else 0.0

    # logic_consistency rate: True / False / None counts
    logic_true  = sum(1 for r in records if r.get("S_logic_consistency") is True)
    logic_false = sum(1 for r in records if r.get("S_logic_consistency") is False)
    logic_none  = sum(1 for r in records if r.get("S_logic_consistency") is None)

    checkpoint_rates = {
        "S1_fg_grounding_rate":       rate("S1_fg_grounding"),
        "S2_rxn_type_exact_rate":     rate("S2_rxn_type_exact"),
        "S3_mechanism_rate":          rate("S3_mechanism"),
        "S4_all_valid_rate":          rate("S4_all_valid"),
        "S5_bond_broken_rate":        rate("S5_bond_broken"),
        "S_logic_consistency_true":   logic_true,
        "S_logic_consistency_false":  logic_false,
        "S_logic_consistency_none":   logic_none,
        "S6_fwd_self_check_rate":     rate("S6_fwd_self_check"),
        "outcome_exact_rate":         rate("outcome_exact"),
        "outcome_fwd_sim_rate":       fwd_sim_rate,
        "outcome_fwd_sim_coverage":   fwd_sim_coverage,
        "outcome_pass_rate":          rate("outcome_pass"),
    }

    by_diff: dict[str, list] = {}
    for r in records:
        by_diff.setdefault(r.get("difficulty", "unknown"), []).append(r)

    diff_stats = {}
    for d, recs in sorted(by_diff.items()):
        entry = {
            "n":                  len(recs),
            "all_pass_rate":      round(sum(r.get("all_pass",        False) for r in recs) / len(recs), 4),
            "outcome_pass_rate":  round(sum(r.get("outcome_pass",    False) for r in recs) / len(recs), 4),
            "outcome_exact_rate": round(sum(r.get("outcome_exact",   False) for r in recs) / len(recs), 4),
            "fwd_sim_rate":       round(sum(r.get("outcome_fwd_sim") is True for r in recs) / len(recs), 4),
            "logic_fail_rate":    round(sum(r.get("S_logic_consistency") is False for r in recs) / len(recs), 4),
            "S1_rate":            round(sum(r.get("S1_fg_grounding", False) for r in recs) / len(recs), 4),
            "S2_rate":            round(sum(r.get("S2_rxn_type_exact", False) for r in recs) / len(recs), 4),
            "S4_rate":            round(sum(r.get("S4_all_valid",    False) for r in recs) / len(recs), 4),
            "S6_rate":            round(sum(r.get("S6_fwd_self_check", False) for r in recs) / len(recs), 4),
        }
        diff_stats[d] = entry

    api_ok        = [r for r in records if r.get("api_success")]
    s1_fails      = [r for r in api_ok if not r.get("S1_fg_grounding")]
    s2_fails      = [r for r in api_ok if not r.get("S2_rxn_type_exact")]
    s4_fails      = [r for r in api_ok if not r.get("S4_all_valid")]
    logic_fails   = [r for r in api_ok if r.get("S_logic_consistency") is False]
    oc_fails      = [r for r in api_ok if not r.get("outcome_pass")]

    formula_desc = "S1∧S2∧S4∧outcome_exact"

    summary = {
        "n_total":               n,
        "n_parsed_ok":           sum(1 for r in records if r.get("parse_ok",    False)),
        "n_step7_present":       sum(1 for r in records if r.get("step7_present", False)),
        "n_all_pass":            sum(1 for r in records if r.get("all_pass",    False)),
        "all_pass_rate":         rate("all_pass"),
        "outcome_pass_rate":     rate("outcome_pass"),
        "outcome_exact_rate":    rate("outcome_exact"),
        "all_pass_formula":      formula_desc,
        **checkpoint_rates,
        "by_difficulty":         diff_stats,
        "n_s1_fails":            len(s1_fails),
        "n_s2_fails":            len(s2_fails),
        "n_s4_fails":            len(s4_fails),
        "n_logic_fails":         len(logic_fails),
        "n_outcome_pass_fails":  len(oc_fails),
        "token_totals": {
            "reasoning":        sum(r.get("reasoning_tokens",        0) for r in records),
            "total_completion": sum(r.get("total_completion_tokens", 0) for r in records),
            "prompt":           sum(r.get("prompt_tokens",           0) for r in records),
        },
    }

    # Console summary
    print(f"\nVerification summary (v4):")
    info_keys = {"S3_mechanism", "S5_bond_broken",
                 "S6_fwd_self_check"}
    all_chk = [
        ("S1_fg_grounding",      "S1_fg_grounding"),
        ("S2_rxn_type_exact",    "S2_rxn_type_exact"),
        ("S3_mechanism",         "S3_mechanism"),
        ("S4_all_valid",         "S4_all_valid"),
        ("S5_bond_broken",       "S5_bond_broken"),
        ("S6_fwd_self_check",    "S6_fwd_self_check"),
        ("outcome_exact",        "outcome_exact"),
        ("outcome_pass",         "outcome_pass"),
    ]

    for label, key in all_chk:
        cnt = sum(1 for rec in records if rec.get(key))
        tag = " (info)" if label in info_keys else ""
        print(f"  {label:24s}: {cnt}/{n} ({100*cnt/n:.1f}%){tag}")

    # Logic consistency special report
    print(f"  {'S_logic_consistency':24s}: True={logic_true} False={logic_false} None(unknown)={logic_none}"
          f"  ← {logic_false} logic fails REMOVED from all_pass")

    # fwd_sim special report
    fwd_true  = sum(1 for r in records if r.get("outcome_fwd_sim") is True)
    fwd_false = sum(1 for r in records if r.get("outcome_fwd_sim") is False)
    fwd_none  = sum(1 for r in records if r.get("outcome_fwd_sim") is None)
    print(f"  {'outcome_fwd_sim':24s}: True={fwd_true} False={fwd_false} None(no_template)={fwd_none}")

    ap = sum(1 for r in records if r.get("all_pass"))
    print(f"  {'all_pass':24s}: {ap}/{n} ({100*ap/n:.1f}%)  [{formula_desc}]")

    return records, summary
