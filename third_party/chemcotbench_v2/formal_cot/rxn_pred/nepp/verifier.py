"""
Verifier for rxn_pred/nepp formal A→B CoT output.

Checkpoint design
─────────────────
  S1_reactant_charge  (Type I, info)   — step1: declared SMILES parseable AND
                                          declared net_charge == RDKit computed charge
  S2_elem_mech        (info only)      — step2 ELEM_MECH is non-empty (presence check)
  S3_bond_change      (Type I, info)   — element pairs in BOND_CHANGE are present in
                                          the reactant/product molecules
  S4_mol_valid        (Type I, GATES)  — step4 PREDICTED_SMILES is fully RDKit-parseable
  S5_product_charge   (Type I, info)   — step5 declared product charge == RDKit computed
  S6_charge_balanced  (Type I, GATES)  — RDKit net_charge(product) == net_charge(reactants);
                                          independent of what model declared in step6
  S7_atom_conserved   (Type I, GATES)  — RDKit heavy_atom_formula(product) ==
                                          heavy_atom_formula(reactants)
  scaffold_match      (Type I, info)   — Murcko scaffold of largest product frag ==
                                          Murcko scaffold of largest GT frag
  outcome             (Type I, GATES)  — canonical(PREDICTED_SMILES) == canonical(GT)

  all_pass = S4 AND S6 AND S7 AND outcome
"""

import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.cot_eval.rxn_predict.nepp_structured.utils import (
    all_frags_valid,
    canonical_smiles,
    smiles_match_set,
    net_charge_from_smiles,
    heavy_atom_formula,
    scaffold_match,
    fts,
)


# ── Rule-based checkpoint functions ──────────────────────────────────────────

def check_s1(record: dict) -> bool:
    """S1 (info): step1 SMILES parseable AND declared net_charge == RDKit computed.

    Uses current_reactants from the dataset (ground truth for the reactant SMILES)
    if step1_reactants_smi is empty; otherwise validates the model's own SMILES.
    """
    smi = record.get("step1_reactants_smi", "") or record.get("current_reactants", "")
    if not smi:
        return False
    if not all_frags_valid(smi):
        return False
    declared = record.get("step1_net_charge")
    if declared is None:
        return False
    computed = net_charge_from_smiles(smi)
    return computed is not None and int(computed) == int(declared)


def check_s2(record: dict) -> bool:
    """S2 (info): ELEM_MECH is non-empty (presence check only)."""
    return bool((record.get("step2_elem_mech", "") or "").strip())


def check_s3(record: dict) -> bool:
    """S3 (info): bond element pairs in BOND_CHANGE are present in reactant/product.

    Checks that the element symbols mentioned in "break:X-Y, form:A-B" each appear
    in the respective molecule.  Informational — not a gate for all_pass.
    """
    bond_str = record.get("step3_bond_change", "")
    if not bond_str:
        return False

    react_smi = record.get("current_reactants", "")
    pred_smi  = record.get("step4_predicted_smi", "")
    if not react_smi or not pred_smi:
        return False

    from rdkit import Chem

    def _mol_elements(smiles: str) -> set:
        elems: set = set()
        for frag in smiles.split("."):
            mol = Chem.MolFromSmiles(frag.strip())
            if mol:
                elems.update(a.GetSymbol() for a in mol.GetAtoms())
        return elems

    react_elems = _mol_elements(react_smi)
    prod_elems  = _mol_elements(pred_smi)
    all_elems   = react_elems | prod_elems

    # Parse element pairs like "C-O" from bond_str
    pairs = re.findall(r'([A-Z][a-z]?)-([A-Z][a-z]?)', bond_str)
    if not pairs:
        return False
    return all(e1 in all_elems and e2 in all_elems for e1, e2 in pairs)


def check_s4(record: dict) -> bool:
    """S4 (GATES): PREDICTED_SMILES is fully RDKit-parseable."""
    pred = record.get("step4_predicted_smi", "")
    return bool(pred and all_frags_valid(pred))


def check_s5(record: dict) -> bool:
    """S5 (info): step5 declared product charge == RDKit computed charge."""
    pred = record.get("step4_predicted_smi", "")
    declared = record.get("step5_product_charge")
    if not pred or declared is None:
        return False
    computed = net_charge_from_smiles(pred)
    return computed is not None and int(computed) == int(declared)


def check_s6(record: dict) -> bool:
    """S6 (GATES): RDKit net_charge(product) == RDKit net_charge(current reactants).

    Computed independently from RDKit — does NOT rely on the model's declared values.
    Uses current_reactants from the dataset as the authoritative reactant SMILES.
    """
    pred   = record.get("step4_predicted_smi", "")
    react  = record.get("current_reactants", "")
    if not pred or not react:
        return False
    q_pred  = net_charge_from_smiles(pred)
    q_react = net_charge_from_smiles(react)
    if q_pred is None or q_react is None:
        return False
    return int(q_pred) == int(q_react)


def check_s7(record: dict) -> bool:
    """S7 (GATES): RDKit heavy_atom_formula(product) == heavy_atom_formula(reactants).

    Computed independently — does NOT rely on the model's ATOM_CONSERVED declaration.
    """
    pred  = record.get("step4_predicted_smi", "")
    react = record.get("current_reactants", "")
    if not pred or not react:
        return False
    f_pred  = heavy_atom_formula(pred)
    f_react = heavy_atom_formula(react)
    if f_pred is None or f_react is None:
        return False
    return f_pred == f_react


def check_scaffold(record: dict) -> bool:
    """Scaffold (info): Murcko scaffold of largest product fragment == GT."""
    pred = record.get("step4_predicted_smi", "")
    gt   = record.get("gt_product_smiles", "")
    if not pred or not gt:
        return False
    return scaffold_match(pred, gt)


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

    all_pass = S4 AND S6 AND S7 AND outcome
    """
    updated = dict(record)

    # Short-circuit on API / parse failure
    if not record.get("api_success", False):
        for k in ("S1_reactant_charge", "S2_elem_mech", "S3_bond_change",
                  "S4_mol_valid", "S5_product_charge", "S6_charge_balanced",
                  "S7_atom_conserved", "scaffold_match", "outcome",
                  "all_pass"):
            updated[k] = False
        updated["verify_skip_reason"] = "api_failure"
        return updated

    if not record.get("parse_ok", False):
        for k in ("S1_reactant_charge", "S2_elem_mech", "S3_bond_change",
                  "S4_mol_valid", "S5_product_charge", "S6_charge_balanced",
                  "S7_atom_conserved", "scaffold_match", "outcome",
                  "all_pass"):
            updated[k] = False
        updated["verify_skip_reason"] = "parse_failed"
        return updated

    # Rule-based checks
    s1 = check_s1(record)            # info
    s2 = check_s2(record)            # info
    s3 = check_s3(record)            # info
    s4 = check_s4(record)            # GATES
    s5 = check_s5(record)            # info
    s6 = check_s6(record)            # GATES
    s7 = check_s7(record)            # GATES
    sc = check_scaffold(record)      # info
    oc = check_outcome(record)       # GATES

    all_pass = s4 and s6 and s7 and oc

    # Compute FTS for reference
    pred_can  = canonical_smiles(record.get("step4_predicted_smi", "")) or ""
    gt_can    = canonical_smiles(record.get("gt_product_smiles",   "")) or ""
    fts_score = fts(record.get("step4_predicted_smi", ""),
                    record.get("gt_product_smiles",   ""))

    updated.update({
        "S1_reactant_charge":  s1,
        "S2_elem_mech":        s2,     # informational
        "S3_bond_change":      s3,     # informational
        "S4_mol_valid":        s4,
        "S5_product_charge":   s5,     # informational
        "S6_charge_balanced":  s6,
        "S7_atom_conserved":   s7,
        "scaffold_match":      sc,     # informational
        "outcome":             oc,
        "all_pass":            all_pass,
        "predicted_can":       pred_can,
        "gt_can":              gt_can,
        "fts_score":           round(fts_score, 4),
        "verify_skip_reason":  "",
    })
    return updated


# ── Batch verifier ────────────────────────────────────────────────────────────

def verify_all(
    records: list[dict],
) -> tuple[list[dict], dict]:
    """Verify all records. Returns (updated_records, summary_dict)."""
    print("\n--- Running RDKit verification (no LLM judge) ---")

    for i, rec in enumerate(records):
        records[i] = verify_one(rec)

    n = len(records)
    if n == 0:
        return records, {}

    def rate(key: str) -> float:
        vals = [r for r in records if r.get(key) is not None]
        denom = max(len(vals), 1)
        return round(sum(1 for r in vals if r.get(key)) / denom, 4)

    checkpoint_rates = {
        "S1_reactant_charge_rate":  rate("S1_reactant_charge"),
        "S2_elem_mech_rate":        rate("S2_elem_mech"),
        "S3_bond_change_rate":      rate("S3_bond_change"),
        "S4_mol_valid_rate":        rate("S4_mol_valid"),
        "S5_product_charge_rate":   rate("S5_product_charge"),
        "S6_charge_balanced_rate":  rate("S6_charge_balanced"),
        "S7_atom_conserved_rate":   rate("S7_atom_conserved"),
        "scaffold_match_rate":      rate("scaffold_match"),
        "outcome_rate":             rate("outcome"),
    }

    # Per-difficulty breakdown
    by_diff: dict[str, list] = {}
    for r in records:
        by_diff.setdefault(r.get("difficulty", "unknown"), []).append(r)

    diff_stats = {}
    for d, recs in sorted(by_diff.items()):
        nd = len(recs)
        entry = {
            "n":              nd,
            "all_pass_rate":  round(sum(r.get("all_pass",          False) for r in recs) / nd, 4),
            "outcome_acc":    round(sum(r.get("outcome",            False) for r in recs) / nd, 4),
            "S4_rate":        round(sum(r.get("S4_mol_valid",       False) for r in recs) / nd, 4),
            "S6_rate":        round(sum(r.get("S6_charge_balanced", False) for r in recs) / nd, 4),
            "S7_rate":        round(sum(r.get("S7_atom_conserved",  False) for r in recs) / nd, 4),
            "scaffold_rate":  round(sum(r.get("scaffold_match",     False) for r in recs) / nd, 4),
            "fts_mean":       round(sum(r.get("fts_score",          0.0)   for r in recs) / nd, 4),
        }
        diff_stats[d] = entry

    # Failure counts
    api_ok    = [r for r in records if r.get("api_success")]
    s4_fails  = [r for r in api_ok if not r.get("S4_mol_valid")]
    s6_fails  = [r for r in api_ok if not r.get("S6_charge_balanced")]
    s7_fails  = [r for r in api_ok if not r.get("S7_atom_conserved")]
    oc_fails  = [r for r in api_ok if not r.get("outcome")]

    summary = {
        "n_total":          n,
        "n_parsed_ok":      sum(1 for r in records if r.get("parse_ok",  False)),
        "n_all_pass":       sum(1 for r in records if r.get("all_pass",  False)),
        "all_pass_rate":    rate("all_pass"),
        "outcome_acc":      rate("outcome"),
        "fts_mean":         round(sum(r.get("fts_score", 0) for r in records) / n, 4),
        "all_pass_formula": "S4 AND S6 AND S7 AND outcome",
        **checkpoint_rates,
        "by_difficulty":    diff_stats,
        "n_s4_fails":       len(s4_fails),
        "n_s6_fails":       len(s6_fails),
        "n_s7_fails":       len(s7_fails),
        "n_oc_fails":       len(oc_fails),
        "token_totals": {
            "reasoning":        sum(r.get("reasoning_tokens",        0) for r in records),
            "total_completion": sum(r.get("total_completion_tokens", 0) for r in records),
            "prompt":           sum(r.get("prompt_tokens",           0) for r in records),
        },
    }

    # Console summary
    print("\nVerification summary:")
    INFO_KEYS = {
        "S1_reactant_charge", "S2_elem_mech",
        "S3_bond_change", "S5_product_charge", "scaffold_match",
    }
    for k, r_val in checkpoint_rates.items():
        label = k.replace("_rate", "")
        cnt   = sum(1 for r in records if r.get(label))
        tag   = " (info)" if label in INFO_KEYS else ""
        print(f"  {label:26s}: {cnt}/{n} ({100*r_val:.1f}%){tag}")
    ap = sum(1 for r in records if r.get("all_pass"))
    formula = "S4∧S6∧S7∧outcome"
    print(f"  {'all_pass':26s}: {ap}/{n} ({100*ap/n:.1f}%)  [{formula}]")
    print(f"  {'fts_mean':26s}: {summary['fts_mean']:.4f}")

    return records, summary
