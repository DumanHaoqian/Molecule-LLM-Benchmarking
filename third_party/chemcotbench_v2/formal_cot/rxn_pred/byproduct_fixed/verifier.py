"""
Verifier for rxn_pred/byproduct_fixed formal A→B CoT output.

Checkpoint design
─────────────────
  S1_fg_grounding         (Type I, info)   — step1 FG_LIST: ≥1 FG found in reactants
                                              via RDKit SMARTS; informational (not gating)
  S2_rxn_type_exact       (Type I, GATES)  — step2 RXN_TYPE exactly matches GT
                                              coarse_rxn_cls (one of 9 categories).
  S3_atomic_delta_grounded (Type I, info)  — ATOMIC_DELTA element symbols ⊆
                                              heavy_atom_elements(reactants+reagents);
                                              informational (a weak grounding check)
  S4_fragment_parseable   (Type I, GATES)  — LEAVING_FRAGMENT_SMILES is RDKit-parseable
  S5_fragment_in_reactant (Type I, GATES)  — heavy_atom_elements(LF) ⊆
                                              heavy_atom_elements(reactants)  [≡ V4b]
  S6_element_coherence    (Type I, GATES)  — heavy_atom_elements(LF) ⊆
                                              heavy_atom_elements(BYPRODUCT)  [≡ V4c]
  outcome                 (Type I, GATES)  — fragment-aware InChI match:
                                              inchi(BYPRODUCT) == inchi(GT)  [Tier 1]
                                              OR inchi(BYPRODUCT) ∈ inchi-fragments(GT) [Tier 2]

  all_pass = S4 AND S5 AND S6 AND S2 AND outcome
"""

import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.cot_eval.rxn_predict.forward_structured.utils import (
    fg_grounding_score,
)
from baselines.cot_eval.rxn_predict.byproduct_fixed_structured_v3.utils import (
    heavy_atom_elements,
    leaving_fragment_in_reactant,
    fragment_subset_of_byproduct,
    smiles_match,
    canonical_with_ions,
)


# ── Rule-based checkpoint functions ──────────────────────────────────────────

def check_s1(record: dict) -> bool:
    """S1 (INFO): ≥1 FG in step1 FG_LIST verified via RDKit SMARTS on reactants."""
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
    """S3 (INFO ONLY): ATOMIC_DELTA element symbols ⊆ heavy_atom_elements(reactants).

    A weak grounding check — catches hallucinated elements not present in
    any reactant.  Not a gate because the set-subset check is permissive.
    """
    delta: list[str] = record.get("step3_atomic_delta", [])
    if not delta:
        return False

    reactants_list: list[str] = record.get("reactants_list", [])
    if not reactants_list:
        reactants_smi = record.get("reactants_smiles", "")
        reactants_list = [s for s in reactants_smi.split(".") if s.strip()]
    # Also include reagents if any
    reagents_smi = record.get("reagents_smiles", "")
    reagents_list = [s for s in reagents_smi.split(".") if s.strip() and s != "(none)"]
    all_sources = reactants_list + reagents_list

    all_elements: set[str] = set()
    for smi in all_sources:
        all_elements |= heavy_atom_elements(smi)

    delta_set = {sym.strip() for sym in delta if sym.strip()}
    return delta_set.issubset(all_elements)


def check_s4(record: dict) -> bool:
    """S4 (GATES): LEAVING_FRAGMENT_SMILES is a valid RDKit-parseable molecule."""
    lf_smi = record.get("step4_lf_smiles", "")
    if not lf_smi:
        return False
    try:
        mol = Chem.MolFromSmiles(lf_smi)
        return mol is not None
    except Exception:
        return False


def check_s5(record: dict) -> bool:
    """S5 (GATES): heavy_atom_elements(LF) ⊆ heavy_atom_elements(reactants).

    Equivalent to V4b from the structured evaluation.  Verifies that the
    leaving fragment is chemically grounded in the input reactants.
    """
    lf_smi = record.get("step4_lf_smiles", "")
    if not lf_smi:
        return False
    reactants_list: list[str] = record.get("reactants_list", [])
    if not reactants_list:
        reactants_smi = record.get("reactants_smiles", "")
        reactants_list = [s for s in reactants_smi.split(".") if s.strip()]
    return leaving_fragment_in_reactant(lf_smi, reactants_list)


def check_s6(record: dict) -> bool:
    """S6 (GATES): heavy_atom_elements(LF) ⊆ heavy_atom_elements(BYPRODUCT).

    Equivalent to V4c from the structured evaluation.  Verifies coherence
    between the leaving fragment and the predicted byproduct.
    """
    lf_smi = record.get("step4_lf_smiles", "")
    bp_smi = record.get("step6_byproduct_smiles", "")
    if not lf_smi or not bp_smi:
        return False
    return fragment_subset_of_byproduct(lf_smi, bp_smi)


def _smiles_to_inchi(smi: str) -> str | None:
    """Convert SMILES to InChI string. Returns None if unparseable.

    InChI handles metal bond disconnection and charge-separation conventions
    automatically: e.g. Cl[Mg]Cl and [Cl-].[Mg+]Cl both map to the same
    InChI (same MgCl2), but CCO.[Li+] (EtOH + Li+) and [Li]OCC (LiOEt)
    give different InChI strings because their proton counts differ.
    This makes InChI the correct large-scale metric for byproduct matching.
    """
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return MolToInchi(mol)


def inchi_match(pred_smi: str, gt_smi: str) -> bool:
    """True iff both SMILES map to the same InChI string."""
    pi = _smiles_to_inchi(pred_smi)
    gi = _smiles_to_inchi(gt_smi)
    if pi is None or gi is None:
        return False
    return pi == gi


def check_outcome(record: dict) -> bool:
    """outcome (GATES): fragment-aware InChI match of BYPRODUCT vs GT byproduct.

    Two-tier matching strategy:
      Tier 1 (exact):   canonical InChI(pred) == canonical InChI(GT)
                        Handles metal bond disconnection automatically:
                        Cl[Mg]Cl ≡ [Cl-].[Mg+]Cl (same MgCl₂);
                        CCO[Li] ≠ CCO.[Li+] (different proton count → kept strict).
      Tier 2 (fragment): InChI(pred) ∈ { InChI(frag) for each dot-fragment of GT }
                        For multi-component GT SMILES (e.g. Dess-Martin 3 products,
                        reductive-amination water + amine), correctly predicting any
                        one component is chemically meaningful and counts as a pass.

    Scales to any dataset size with no manual inspection.
    """
    pred = record.get("answer_smiles", "") or record.get("step6_byproduct_smiles", "")
    gt   = record.get("gt_smiles", "")
    if not pred or not gt:
        return False

    # Tier 1: exact InChI match
    if inchi_match(pred, gt):
        return True

    # Tier 2: pred InChI ∈ { InChI of each fragment of GT mol }
    pred_inchi = _smiles_to_inchi(pred)
    if not pred_inchi:
        return False
    gt_mol = Chem.MolFromSmiles(gt)
    if gt_mol is None:
        return False
    for frag_mol in Chem.GetMolFrags(gt_mol, asMols=True):
        fi = _smiles_to_inchi(Chem.MolToSmiles(frag_mol))
        if fi and fi == pred_inchi:
            return True
    return False


def check_outcome_exact_inchi(record: dict) -> bool:
    """outcome_exact_inchi (reference): strict InChI match, no fragment expansion.

    Kept as a reference/comparison field alongside the primary outcome gate
    (which uses fragment-aware matching).
    """
    pred = record.get("answer_smiles", "") or record.get("step6_byproduct_smiles", "")
    gt   = record.get("gt_smiles", "")
    if not pred or not gt:
        return False
    return inchi_match(pred, gt)


def check_outcome_strict(record: dict) -> bool:
    """outcome_strict: canonical SMILES exact match (kept for reference/comparison)."""
    pred = record.get("answer_smiles", "") or record.get("step6_byproduct_smiles", "")
    gt   = record.get("gt_smiles", "")
    if not pred or not gt:
        return False
    return smiles_match(pred, gt)


# ── Single-record verifier ────────────────────────────────────────────────────

def verify_one(record: dict) -> dict:
    """Run all checkpoints for one parsed record.

    all_pass = S4 AND S5 AND S6 AND S2 AND outcome
    """
    updated = dict(record)

    if not record.get("api_success", False):
        for k in ("S1_fg_grounding", "S2_rxn_type_exact", "S3_atomic_delta_grounded",
                  "S4_fragment_parseable", "S5_fragment_in_reactant",
                  "S6_element_coherence", "outcome", "all_pass"):
            updated[k] = False
        updated["verify_skip_reason"] = "api_failure"
        return updated

    if not record.get("parse_ok", False):
        for k in ("S1_fg_grounding", "S2_rxn_type_exact", "S3_atomic_delta_grounded",
                  "S4_fragment_parseable", "S5_fragment_in_reactant",
                  "S6_element_coherence", "outcome", "all_pass"):
            updated[k] = False
        updated["verify_skip_reason"] = "parse_failed"
        return updated

    # Rule-based checks
    s1 = check_s1(record)
    s2 = check_s2_coarse(record)                  # GATES
    s3 = check_s3(record)        # informational
    s4 = check_s4(record)
    s5 = check_s5(record)
    s6 = check_s6(record)
    oc = check_outcome(record)                    # fragment-aware InChI (primary gate)
    oc_exact_inchi = check_outcome_exact_inchi(record)  # strict InChI (reference)
    oc_strict = check_outcome_strict(record)      # exact SMILES (reference)

    # Canonical SMILES for logging
    pred_can = canonical_with_ions(
        record.get("answer_smiles", "") or record.get("step6_byproduct_smiles", "")
    ) or ""
    gt_can = canonical_with_ions(record.get("gt_smiles", "")) or ""

    all_pass = s4 and s5 and s6 and s2 and oc

    updated.update({
        "S1_fg_grounding":         s1,
        "S2_rxn_type_exact":       s2,
        "S3_atomic_delta_grounded": s3,   # informational
        "S4_fragment_parseable":   s4,
        "S5_fragment_in_reactant": s5,
        "S6_element_coherence":    s6,
        "outcome":                 oc,                # fragment-aware InChI (primary gate)
        "outcome_exact_inchi":     oc_exact_inchi,    # strict InChI match (reference)
        "outcome_strict":          oc_strict,         # exact SMILES match (reference)
        "all_pass":                all_pass,
        "predicted_can":           pred_can,
        "gt_can":                  gt_can,
        "verify_skip_reason":      "",
    })
    return updated


# ── Batch verifier ────────────────────────────────────────────────────────────

def verify_all(
    records: list[dict],
) -> tuple[list[dict], dict]:
    """Verify all records. Returns (updated_records, summary_dict).

    all_pass = S4∧S5∧S6∧S2∧outcome(InChI+fragment).
    """
    print("\n--- Running coarse rxn type exact match + RDKit verification ---")

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
        "S1_fg_grounding_rate":          rate("S1_fg_grounding"),
        "S2_rxn_type_exact_rate":        rate("S2_rxn_type_exact"),
        "S3_atomic_delta_grounded_rate": rate("S3_atomic_delta_grounded"),
        "S4_fragment_parseable_rate":    rate("S4_fragment_parseable"),
        "S5_fragment_in_reactant_rate":  rate("S5_fragment_in_reactant"),
        "S6_element_coherence_rate":     rate("S6_element_coherence"),
        "outcome_rate":                  rate("outcome"),                # fragment-aware InChI
        "outcome_exact_inchi_rate":      rate("outcome_exact_inchi"),    # strict InChI (reference)
        "outcome_strict_rate":           rate("outcome_strict"),         # exact SMILES (reference)
    }

    # Per-difficulty breakdown
    by_diff: dict[str, list] = {}
    for r in records:
        by_diff.setdefault(r.get("difficulty", "unknown"), []).append(r)

    diff_stats = {}
    for d, recs in sorted(by_diff.items()):
        entry = {
            "n":             len(recs),
            "all_pass_rate": round(sum(r.get("all_pass",                  False) for r in recs) / len(recs), 4),
            "outcome_acc":   round(sum(r.get("outcome",                   False) for r in recs) / len(recs), 4),
            "S1_rate":       round(sum(r.get("S1_fg_grounding",           False) for r in recs) / len(recs), 4),
            "S2_rate":       round(sum(r.get("S2_rxn_type_exact",         False) for r in recs) / len(recs), 4),
            "S3_rate":       round(sum(r.get("S3_atomic_delta_grounded",  False) for r in recs) / len(recs), 4),
            "S4_rate":       round(sum(r.get("S4_fragment_parseable",     False) for r in recs) / len(recs), 4),
            "S5_rate":       round(sum(r.get("S5_fragment_in_reactant",   False) for r in recs) / len(recs), 4),
            "S6_rate":       round(sum(r.get("S6_element_coherence",      False) for r in recs) / len(recs), 4),
        }
        diff_stats[d] = entry

    api_ok   = [r for r in records if r.get("api_success")]
    s4_fails = [r for r in api_ok if not r.get("S4_fragment_parseable")]
    s5_fails = [r for r in api_ok if not r.get("S5_fragment_in_reactant")]
    s6_fails = [r for r in api_ok if not r.get("S6_element_coherence")]
    s2_fails = [r for r in api_ok if not r.get("S2_rxn_type_exact")]
    oc_fails = [r for r in api_ok if not r.get("outcome")]

    summary = {
        "n_total":                n,
        "n_parsed_ok":            sum(1 for r in records if r.get("parse_ok",  False)),
        "n_all_pass":             sum(1 for r in records if r.get("all_pass",  False)),
        "all_pass_rate":          rate("all_pass"),
        "outcome_acc":            rate("outcome"),              # fragment-aware InChI (primary)
        "outcome_exact_inchi_acc": rate("outcome_exact_inchi"), # strict InChI (reference)
        "outcome_strict_acc":     rate("outcome_strict"),       # exact SMILES (reference)
        "all_pass_formula":       "S4 AND S5 AND S6 AND S2_rxn_type AND outcome(InChI+fragment)",
        **checkpoint_rates,
        "by_difficulty":      diff_stats,
        "n_s4_fails":         len(s4_fails),
        "n_s5_fails":         len(s5_fails),
        "n_s6_fails":         len(s6_fails),
        "n_s2_fails":         len(s2_fails),
        "n_oc_fails":         len(oc_fails),
        "token_totals": {
            "reasoning":        sum(r.get("reasoning_tokens",        0) for r in records),
            "total_completion": sum(r.get("total_completion_tokens", 0) for r in records),
            "prompt":           sum(r.get("prompt_tokens",           0) for r in records),
        },
    }

    # Console summary
    print("\nVerification summary:")
    info_keys = {"S1_fg_grounding", "S3_atomic_delta_grounded"}
    for k, r_val in checkpoint_rates.items():
        if k in ("outcome_exact_inchi_rate", "outcome_strict_rate"):
            continue
        cnt   = sum(1 for rec in records if rec.get(k.replace("_rate", "")))
        label = k.replace("_rate", "")
        tag   = " (info)" if label in info_keys else ""
        print(f"  {label:30s}: {cnt}/{n} ({100*r_val:.1f}%){tag}")
    cnt_ei = sum(1 for r in records if r.get("outcome_exact_inchi"))
    print(f"  {'outcome_exact_inchi':30s}: {cnt_ei}/{n} ({100*rate('outcome_exact_inchi'):.1f}%)  (ref: strict InChI)")
    cnt_strict = sum(1 for r in records if r.get("outcome_strict"))
    print(f"  {'outcome_strict (SMILES)':30s}: {cnt_strict}/{n} ({100*rate('outcome_strict'):.1f}%)  (ref: exact SMILES)")
    cnt_s2 = sum(1 for r in records if r.get("S2_rxn_type_exact"))
    print(f"  {'S2_rxn_type_exact':30s}: {cnt_s2}/{n} ({100*rate('S2_rxn_type_exact'):.1f}%)")
    ap = sum(1 for r in records if r.get("all_pass"))
    formula = "S4∧S5∧S6∧S2∧outcome(InChI+frag)"
    print(f"  {'all_pass':30s}: {ap}/{n} ({100*ap/n:.1f}%)  [{formula}]")

    return records, summary
