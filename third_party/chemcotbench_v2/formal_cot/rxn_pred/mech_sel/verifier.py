"""
Verifier for rxn_pred/mech_sel formal A→B CoT output.

Checkpoint design
─────────────────
  S1_smarts_parseable (Type I, GATES)  — step1 SMARTS_LIST: every SMARTS
                                          in the list is RDKit-parseable via
                                          Chem.MolFromSmarts(); ≥1 SMARTS required.
  S2_rxn_type_exact   (Type I, GATES)  — step2 RXN_TYPE exactly matches GT
                                          coarse_rxn_cls (one of 9 categories).
  S3_elim_logic       (Type I, GATES)  — elimination logic: ELIMINATED_OPTIONS ⊆
                                          valid_choices AND SELECTED_OPTION ∉
                                          ELIMINATED_OPTIONS.
  S4_remaining_arith  (info only)      — REMAINING_OPTIONS == valid_choices −
                                          ELIMINATED_OPTIONS (set arithmetic check)
  S5_selected_valid   (info only)      — SELECTED_OPTION ∈ REMAINING_OPTIONS
                                          (derived from S3+S4 but kept for explicit
                                          tracking)
  outcome             (Type I, GATES)  — SELECTED_OPTION.upper() == GT letter

  all_pass = S1 AND S2 AND S3 AND outcome
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from rdkit import Chem
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False


# ── Rule-based checkpoint functions ───────────────────────────────────────────

def check_s1(record: dict) -> bool:
    """S1 (GATES): every SMARTS in step1 SMARTS_LIST is RDKit-parseable.

    Returns True iff:
      - ≥1 SMARTS present in the list
      - Chem.MolFromSmarts(s) is not None for every s in the list
    Falls back to True (permissive) if RDKit is unavailable.
    """
    smarts_list: list[str] = record.get("step1_smarts_list", [])
    if not smarts_list:
        return False
    if not _HAS_RDKIT:
        # Can't validate — treat as pass (don't punish for missing RDKit)
        return True
    for s in smarts_list:
        if not s:
            return False
        try:
            mol = Chem.MolFromSmarts(s)
            if mol is None:
                return False
        except Exception:
            return False
    return True


def check_s2_coarse(record: dict) -> bool:
    """S2 (GATES): predicted coarse reaction type exactly matches GT coarse category."""
    predicted = (record.get("step2_rxn_type", "") or "").strip()
    gt = (record.get("coarse_rxn_cls", "") or "").strip()
    return bool(predicted) and bool(gt) and predicted.lower() == gt.lower()


def check_s3(record: dict) -> bool:
    """S3 (GATES): elimination logic is internally consistent.

    Checks:
      1. ELIMINATED_OPTIONS ⊆ valid_choices (all eliminated letters are real options)
      2. SELECTED_OPTION ∉ ELIMINATED_OPTIONS (didn't eliminate the chosen answer)
    """
    valid_choices = set(c.upper() for c in record.get("valid_choices", []))
    elim          = set(c.upper() for c in record.get("step3_eliminated", []))
    selected      = (record.get("step5_selected", "") or "").upper()

    if not valid_choices:
        return False

    # Eliminated must be a strict subset of valid choices
    if not elim.issubset(valid_choices):
        return False

    # The selected option must NOT have been eliminated
    if selected and selected in elim:
        return False

    # Need at least a selected option present
    return bool(selected)


def check_s4(record: dict) -> bool:
    """S4 (INFO ONLY): REMAINING_OPTIONS == valid_choices − ELIMINATED_OPTIONS."""
    valid_choices = set(c.upper() for c in record.get("valid_choices", []))
    elim          = set(c.upper() for c in record.get("step3_eliminated", []))
    remaining     = set(c.upper() for c in record.get("step4_remaining", []))

    if not valid_choices:
        return False
    return remaining == (valid_choices - elim)


def check_s5(record: dict) -> bool:
    """S5 (INFO ONLY): SELECTED_OPTION ∈ REMAINING_OPTIONS."""
    remaining = set(c.upper() for c in record.get("step4_remaining", []))
    selected  = (record.get("step5_selected", "") or "").upper()
    return bool(selected) and selected in remaining


def check_outcome(record: dict) -> bool:
    """outcome (GATES): SELECTED_OPTION.upper() == GT letter."""
    selected = (
        record.get("step5_selected", "")
        or record.get("answer_letter", "")
    ).upper()
    gt = record.get("gt_letter", "").upper()
    return bool(selected) and bool(gt) and selected == gt


# ── Single-record verifier ─────────────────────────────────────────────────────

def verify_one(record: dict) -> dict:
    """Run all checkpoints for one parsed record.

    all_pass = S1 AND S2 AND S3 AND outcome
    """
    updated = dict(record)

    # Short-circuit on API or parse failure
    if not record.get("api_success", False):
        for k in ("S1_smarts_parseable", "S2_rxn_type_exact", "S3_elim_logic",
                  "S4_remaining_arith", "S5_selected_valid",
                  "outcome", "all_pass"):
            updated[k] = False
        updated["verify_skip_reason"] = "api_failure"
        return updated

    if not record.get("parse_ok", False):
        for k in ("S1_smarts_parseable", "S2_rxn_type_exact", "S3_elim_logic",
                  "S4_remaining_arith", "S5_selected_valid",
                  "outcome", "all_pass"):
            updated[k] = False
        updated["verify_skip_reason"] = "parse_failed"
        return updated

    # Rule-based checks
    s1 = check_s1(record)
    s2 = check_s2_coarse(record)
    s3 = check_s3(record)
    s4 = check_s4(record)        # informational
    s5 = check_s5(record)        # informational
    oc = check_outcome(record)

    all_pass = s1 and s2 and s3 and oc

    updated.update({
        "S1_smarts_parseable": s1,
        "S2_rxn_type_exact":   s2,
        "S3_elim_logic":       s3,
        "S4_remaining_arith":  s4,    # informational
        "S5_selected_valid":   s5,    # informational
        "outcome":             oc,
        "all_pass":            all_pass,
        "verify_skip_reason":  "",
    })
    return updated


# ── Batch verifier ─────────────────────────────────────────────────────────────

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
        if not vals:
            return 0.0
        return round(sum(1 for r in vals if r.get(key)) / n, 4)

    checkpoint_rates = {
        "S1_smarts_parseable_rate": rate("S1_smarts_parseable"),
        "S2_rxn_type_exact_rate":   rate("S2_rxn_type_exact"),
        "S3_elim_logic_rate":       rate("S3_elim_logic"),
        "S4_remaining_arith_rate":  rate("S4_remaining_arith"),
        "S5_selected_valid_rate":   rate("S5_selected_valid"),
        "outcome_rate":             rate("outcome"),
    }

    # Per-difficulty breakdown
    by_diff: dict[str, list] = {}
    for r in records:
        by_diff.setdefault(r.get("difficulty", "unknown"), []).append(r)

    diff_stats = {}
    for d, recs in sorted(by_diff.items()):
        nd  = len(recs)
        entry = {
            "n":              nd,
            "all_pass_rate":  round(sum(r.get("all_pass",            False) for r in recs) / nd, 4),
            "outcome_acc":    round(sum(r.get("outcome",             False) for r in recs) / nd, 4),
            "S1_rate":        round(sum(r.get("S1_smarts_parseable", False) for r in recs) / nd, 4),
            "S2_rate":        round(sum(r.get("S2_rxn_type_exact",   False) for r in recs) / nd, 4),
            "S3_rate":        round(sum(r.get("S3_elim_logic",       False) for r in recs) / nd, 4),
            "S4_rate":        round(sum(r.get("S4_remaining_arith",  False) for r in recs) / nd, 4),
            "S5_rate":        round(sum(r.get("S5_selected_valid",   False) for r in recs) / nd, 4),
        }
        diff_stats[d] = entry

    api_ok    = [r for r in records if r.get("api_success")]
    s1_fails  = [r for r in api_ok if not r.get("S1_smarts_parseable")]
    s2_fails  = [r for r in api_ok if not r.get("S2_rxn_type_exact")]
    s3_fails  = [r for r in api_ok if not r.get("S3_elim_logic")]
    oc_fails  = [r for r in api_ok if not r.get("outcome")]

    summary = {
        "n_total":           n,
        "n_parsed_ok":       sum(1 for r in records if r.get("parse_ok",  False)),
        "n_all_pass":        sum(1 for r in records if r.get("all_pass",  False)),
        "all_pass_rate":     rate("all_pass"),
        "outcome_acc":       rate("outcome"),
        "all_pass_formula":  "S1 AND S2 AND S3 AND outcome",
        **checkpoint_rates,
        "by_difficulty":     diff_stats,
        "n_s1_fails":        len(s1_fails),
        "n_s2_fails":        len(s2_fails),
        "n_s3_fails":        len(s3_fails),
        "n_oc_fails":        len(oc_fails),
        "token_totals": {
            "reasoning":        sum(r.get("reasoning_tokens",        0) for r in records),
            "total_completion": sum(r.get("total_completion_tokens", 0) for r in records),
            "prompt":           sum(r.get("prompt_tokens",           0) for r in records),
        },
    }

    # Console summary
    print("\nVerification summary:")
    info_keys = {"S4_remaining_arith", "S5_selected_valid"}
    for k, r_val in checkpoint_rates.items():
        label = k.replace("_rate", "")
        cnt   = sum(1 for rec in records if rec.get(label))
        tag   = " (info)" if label in info_keys else ""
        print(f"  {label:26s}: {cnt}/{n} ({100*r_val:.1f}%){tag}")
    ap      = sum(1 for r in records if r.get("all_pass"))
    print(f"  {'all_pass':26s}: {ap}/{n} ({100*ap/n:.1f}%)  [S1∧S2∧S3∧outcome]")

    return records, summary
