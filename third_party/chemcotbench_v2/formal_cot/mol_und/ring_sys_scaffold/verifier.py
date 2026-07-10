"""
Verifier for ring_sys_scaffold formal CoT — all checkpoints Type I.

Verification checkpoints (ALL are Type I — RDKit ground truth, no GT label needed):
  S1_mol_rings  — check step1_n_mol  == CalcNumRings(molecule)
  S2_scaf_rings — check step2_n_scaf == CalcNumRings(scaffold)
  S3_non_ring   — check step3_non_ring matches scaffold_has_non_ring_atoms(scaffold)
  S4_predict    — check step4_predict follows the deterministic logic rule
                    (NON_RING_ATOMS_EXIST=yes → No)
                    (NON_RING_ATOMS_EXIST=no AND n_mol==n_scaf → Yes)
                    (NON_RING_ATOMS_EXIST=no AND n_mol!=n_scaf → No)

Additional outcome metric (uses GT label):
  outcome — step4_predict == gt_label  (classification accuracy)

V-point ↔ S-point correspondence:
  V1 mol_ring_count_correct  ↔ S1_mol_rings
  V2 scaf_ring_count_correct ↔ S2_scaf_rings
  V3 non_ring_correct        ↔ S3_non_ring
  V4+V5 logic+answer         ↔ S4_predict
"""
from typing import Optional
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Import utilities from the structured-eval package
from baselines.cot_eval.mol_und.ring_sys_scaffold_structured.utils import (
    scaffold_has_non_ring_atoms,
    get_ring_count,
)


# ────────────────────────────────────────────────────────────────────────────
# Low-level RDKit helpers
# ────────────────────────────────────────────────────────────────────────────

def _rdkit_ring_count(smiles: str) -> Optional[int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return rdMolDescriptors.CalcNumRings(mol)


def _rdkit_has_non_ring(smiles: str) -> Optional[bool]:
    try:
        return scaffold_has_non_ring_atoms(smiles)
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────────────
# Single-record verification
# ────────────────────────────────────────────────────────────────────────────

def verify_one(record: dict) -> dict:
    """Compute S1–S4 checkpoints for a single parsed record.

    Required fields in `record`:
      mol_smiles, scaffold_smiles, gt_label,
      step1_n_mol, step2_n_scaf, step3_non_ring, step4_predict, parse_ok

    Returns a dict with:
      s1_mol_rings, s2_scaf_rings, s3_non_ring, s4_predict,
      all_pass, outcome,
      rdkit_mol_rc, rdkit_scaf_rc, rdkit_has_non_ring  (RDKit reference values)
    """
    if not record.get("parse_ok", False):
        return {
            "s1_mol_rings": None, "s2_scaf_rings": None,
            "s3_non_ring":  None, "s4_predict":    None,
            "all_pass": False,    "outcome": None,
            "rdkit_mol_rc": None, "rdkit_scaf_rc": None,
            "rdkit_has_non_ring": None,
        }

    mol_smi  = record["mol_smiles"]
    scaf_smi = record["scaffold_smiles"]
    gt_label = record.get("gt_label", "")

    # ── RDKit ground truths ──────────────────────────────────────────────────
    rdkit_mol_rc   = _rdkit_ring_count(mol_smi)
    rdkit_scaf_rc  = _rdkit_ring_count(scaf_smi)
    rdkit_has_nr   = _rdkit_has_non_ring(scaf_smi)   # True = has non-ring atoms
    rdkit_non_ring_str = None
    if rdkit_has_nr is not None:
        rdkit_non_ring_str = "yes" if rdkit_has_nr else "no"

    # ── Parsed values ────────────────────────────────────────────────────────
    n_mol    = record["step1_n_mol"]
    n_scaf   = record["step2_n_scaf"]
    nr_str   = record["step3_non_ring"]   # "yes" / "no"
    predict  = record["step4_predict"]    # "Yes" / "No"

    # ── S1: molecule ring count ───────────────────────────────────────────────
    s1 = (rdkit_mol_rc is not None) and (n_mol == rdkit_mol_rc)

    # ── S2: scaffold ring count ───────────────────────────────────────────────
    s2 = (rdkit_scaf_rc is not None) and (n_scaf == rdkit_scaf_rc)

    # ── S3: non-ring atom check ───────────────────────────────────────────────
    s3 = (rdkit_non_ring_str is not None) and (nr_str == rdkit_non_ring_str)

    # ── S4: logical derivation (PREDICT consistent with step1–step3) ─────────
    if n_mol is None or n_scaf is None or nr_str is None or predict is None:
        s4 = False
    else:
        if nr_str == "yes":
            expected_predict = "No"
        elif n_mol == n_scaf:
            expected_predict = "Yes"
        else:
            expected_predict = "No"
        s4 = (predict == expected_predict)

    all_pass = s1 and s2 and s3 and s4

    # ── Outcome: predict vs GT ─────────────────────────────────────────────
    outcome = (predict == gt_label) if (predict and gt_label) else None

    return {
        "s1_mol_rings":       s1,
        "s2_scaf_rings":      s2,
        "s3_non_ring":        s3,
        "s4_predict":         s4,
        "all_pass":           all_pass,
        "outcome":            outcome,
        "rdkit_mol_rc":       rdkit_mol_rc,
        "rdkit_scaf_rc":      rdkit_scaf_rc,
        "rdkit_has_non_ring": rdkit_has_nr,
    }


# ────────────────────────────────────────────────────────────────────────────
# Batch verification + summary
# ────────────────────────────────────────────────────────────────────────────

def verify_batch(records: list[dict]) -> tuple[list[dict], dict]:
    """Run verify_one on all records; return (records_with_results, summary).

    Summary fields:
      total, api_success, parse_ok_count,
      s1_pass, s2_pass, s3_pass, s4_pass, all_pass,
      correct_outcome, accuracy,
      per_difficulty breakdown
    """
    for rec in records:
        vr = verify_one(rec)
        rec.update(vr)

    n_total       = len(records)
    n_api_success = sum(1 for r in records if r.get("api_success", False))
    n_parse_ok    = sum(1 for r in records if r.get("parse_ok", False))
    n_s1          = sum(1 for r in records if r.get("s1_mol_rings") is True)
    n_s2          = sum(1 for r in records if r.get("s2_scaf_rings") is True)
    n_s3          = sum(1 for r in records if r.get("s3_non_ring") is True)
    n_s4          = sum(1 for r in records if r.get("s4_predict") is True)
    n_all_pass    = sum(1 for r in records if r.get("all_pass") is True)
    n_correct     = sum(1 for r in records if r.get("outcome") is True)

    def pct(n, d):
        return round(100 * n / d, 1) if d > 0 else 0.0

    per_difficulty = {}
    for diff in ["easy", "medium", "hard"]:
        sub = [r for r in records if r.get("difficulty") == diff]
        nd  = len(sub)
        per_difficulty[diff] = {
            "total":   nd,
            "all_pass": sum(1 for r in sub if r.get("all_pass") is True),
            "accuracy": pct(sum(1 for r in sub if r.get("outcome") is True), nd),
        }

    summary = {
        "total":           n_total,
        "api_success":     n_api_success,
        "parse_ok":        n_parse_ok,
        # V-point correspondence labels
        "S1_mol_rings_pass  [V1]":  f"{n_s1}/{n_total} ({pct(n_s1, n_total)}%)",
        "S2_scaf_rings_pass [V2]":  f"{n_s2}/{n_total} ({pct(n_s2, n_total)}%)",
        "S3_non_ring_pass   [V3]":  f"{n_s3}/{n_total} ({pct(n_s3, n_total)}%)",
        "S4_predict_pass    [V4+5]": f"{n_s4}/{n_total} ({pct(n_s4, n_total)}%)",
        "all_pass":        f"{n_all_pass}/{n_total} ({pct(n_all_pass, n_total)}%)",
        "accuracy":        f"{n_correct}/{n_total} ({pct(n_correct, n_total)}%)",
        "per_difficulty":  per_difficulty,
    }
    return records, summary
