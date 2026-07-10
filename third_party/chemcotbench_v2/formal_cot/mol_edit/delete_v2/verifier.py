"""
Verifier for mol_edit/delete_v2 formal A→B CoT — RDKit-based checkpoints.

Checkpoint summary (11 Type I + 1 Type II outcome):

  S1 — step1 anchor
    s1_idx_valid         [Type I]  ANCHOR idx ∈ [1, n_heavy_atoms(src)]
    s1_element_match     [Type I]  src.atom[idx-1].GetSymbol() == declared element (case-insensitive)

  S2 — step2 remove group
    s2_group_valid       [Type I]  REMOVE_GROUP SMILES is RDKit-parseable
    s2_heavy_atoms_ok    [Type I]  declared k == group_mol.GetNumHeavyAtoms()

  S3 — step3 product
    s3_product_valid     [Type I]  PRODUCT_SMILES is RDKit-parseable

  S4 — step4 heavy-atom delta
    s4_src_heavy_ok      [Type I]  declared a == src_mol.GetNumHeavyAtoms()
    s4_prod_heavy_ok     [Type I]  declared b == product_mol.GetNumHeavyAtoms() [requires s3]
    s4_delta_arithmetic  [Type I]  declared delta == b − a  (typically negative)

  S5 — step5 ring delta
    s5_src_rings_ok      [Type I]  declared c == CalcNumRings(src)
    s5_prod_rings_ok     [Type I]  declared d == CalcNumRings(product)          [requires s3]
    s5_delta_arithmetic  [Type I]  declared ring_delta == d − c

  outcome                [Type II] smiles_match_main_frag(product, GT_from_dataset)

  all_pass = s1_idx_valid ∧ s1_element_match ∧ s2_group_valid ∧ s2_heavy_atoms_ok
             ∧ s3_product_valid ∧ s4_src_heavy_ok ∧ s4_prod_heavy_ok ∧ s4_delta_arithmetic
             ∧ s5_src_rings_ok ∧ s5_prod_rings_ok ∧ s5_delta_arithmetic
"""
import json
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from baselines.cot_eval.mol_edit.mol_edit_structured.utils import smiles_match_main_frag

RDLogger.DisableLog("rdApp.*")

RESULTS_DIR = PROJECT_ROOT / "results" / "formal_cot" / "mol_edit" / "delete_v2"


# ────────────────────────────────────────────────────────────────────────────
# RDKit helpers
# ────────────────────────────────────────────────────────────────────────────

def _mol(smiles: str) -> Optional[Chem.Mol]:
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles.strip())
    except Exception:
        return None


def _ring_count(mol: Chem.Mol) -> Optional[int]:
    if mol is None:
        return None
    try:
        return rdMolDescriptors.CalcNumRings(mol)
    except Exception:
        return None


def _n_heavy(mol: Chem.Mol) -> Optional[int]:
    if mol is None:
        return None
    return mol.GetNumHeavyAtoms()


def _element_at(mol: Chem.Mol, one_based_idx: int) -> Optional[str]:
    """Return uppercase element symbol of atom at 1-based index, or None."""
    if mol is None or one_based_idx is None:
        return None
    zero_idx = one_based_idx - 1
    if zero_idx < 0 or zero_idx >= mol.GetNumAtoms():
        return None
    return mol.GetAtomWithIdx(zero_idx).GetSymbol().upper()


# ────────────────────────────────────────────────────────────────────────────
# Single-record verification
# ────────────────────────────────────────────────────────────────────────────

def verify_one(record: dict) -> dict:
    """Compute all checkpoints for a single parsed mol_edit/delete_v2 record."""
    if not record.get("parse_ok", False):
        return _null_result()

    src_smi  = record.get("src_smiles", "")
    gt_smi   = record.get("gt_smiles", "")

    anchor_idx   = record.get("step1_anchor_idx")
    anchor_elem  = record.get("step1_anchor_element")

    group_smi  = record.get("step1_remove_group")   # primary source for group SMILES
    # Fall back to step2 if step1 group is missing
    if not group_smi:
        group_smi = record.get("step2_remove_smiles")
    decl_k     = record.get("step2_heavy_atoms")

    prod_smi   = record.get("step3_product_smiles")

    decl_a      = record.get("step4_n_heavy_src")
    decl_b      = record.get("step4_n_heavy_prod")
    decl_hdelta = record.get("step4_heavy_delta")

    decl_c      = record.get("step5_n_rings_src")
    decl_d      = record.get("step5_n_rings_prod")
    decl_rdelta = record.get("step5_ring_delta")

    # ── Build mol objects ─────────────────────────────────────────────────────
    src_mol   = _mol(src_smi)
    group_mol = _mol(group_smi) if group_smi else None
    prod_mol  = _mol(prod_smi)  if prod_smi  else None

    rdkit_src_heavy      = _n_heavy(src_mol)
    rdkit_prod_heavy     = _n_heavy(prod_mol)
    rdkit_src_rings      = _ring_count(src_mol)
    rdkit_prod_rings     = _ring_count(prod_mol)
    rdkit_group_heavy    = _n_heavy(group_mol)
    rdkit_elem_at_anchor = _element_at(src_mol, anchor_idx)

    # ── S1 ────────────────────────────────────────────────────────────────────
    if anchor_idx is not None and rdkit_src_heavy is not None:
        s1_idx_valid = (1 <= anchor_idx <= rdkit_src_heavy)
    else:
        s1_idx_valid = False

    if rdkit_elem_at_anchor is not None and anchor_elem is not None:
        s1_element_match = (rdkit_elem_at_anchor.upper() == anchor_elem.upper())
    else:
        s1_element_match = False

    # ── S2 ────────────────────────────────────────────────────────────────────
    s2_group_valid    = (group_mol is not None)
    s2_heavy_atoms_ok = (
        s2_group_valid and decl_k is not None and rdkit_group_heavy is not None
        and decl_k == rdkit_group_heavy
    )

    # ── S3 ────────────────────────────────────────────────────────────────────
    s3_product_valid = (prod_mol is not None)

    # ── S4 ────────────────────────────────────────────────────────────────────
    s4_src_heavy_ok = (
        rdkit_src_heavy is not None and decl_a is not None
        and decl_a == rdkit_src_heavy
    )
    s4_prod_heavy_ok = (
        s3_product_valid and rdkit_prod_heavy is not None and decl_b is not None
        and decl_b == rdkit_prod_heavy
    )
    s4_delta_arithmetic = (
        decl_a is not None and decl_b is not None and decl_hdelta is not None
        and decl_hdelta == (decl_b - decl_a)
    )

    # ── S5 ────────────────────────────────────────────────────────────────────
    s5_src_rings_ok = (
        rdkit_src_rings is not None and decl_c is not None
        and decl_c == rdkit_src_rings
    )
    s5_prod_rings_ok = (
        s3_product_valid and rdkit_prod_rings is not None and decl_d is not None
        and decl_d == rdkit_prod_rings
    )
    s5_delta_arithmetic = (
        decl_c is not None and decl_d is not None and decl_rdelta is not None
        and decl_rdelta == (decl_d - decl_c)
    )

    # ── all_pass ──────────────────────────────────────────────────────────────
    all_pass = all([
        s1_idx_valid, s1_element_match,
        s2_group_valid, s2_heavy_atoms_ok,
        s3_product_valid,
        s4_src_heavy_ok, s4_prod_heavy_ok, s4_delta_arithmetic,
        s5_src_rings_ok, s5_prod_rings_ok, s5_delta_arithmetic,
    ])

    # ── outcome (Type II — compare to dataset GT) ─────────────────────────────
    outcome = None
    if s3_product_valid and gt_smi:
        outcome = bool(smiles_match_main_frag(prod_smi, gt_smi))

    return {
        "s1_idx_valid":          s1_idx_valid,
        "s1_element_match":      s1_element_match,
        "s2_group_valid":        s2_group_valid,
        "s2_heavy_atoms_ok":     s2_heavy_atoms_ok,
        "s3_product_valid":      s3_product_valid,
        "s4_src_heavy_ok":       s4_src_heavy_ok,
        "s4_prod_heavy_ok":      s4_prod_heavy_ok,
        "s4_delta_arithmetic":   s4_delta_arithmetic,
        "s5_src_rings_ok":       s5_src_rings_ok,
        "s5_prod_rings_ok":      s5_prod_rings_ok,
        "s5_delta_arithmetic":   s5_delta_arithmetic,
        "all_pass":              all_pass,
        "outcome":               outcome,
        # RDKit reference values (useful for debugging)
        "rdkit_src_heavy":       rdkit_src_heavy,
        "rdkit_prod_heavy":      rdkit_prod_heavy,
        "rdkit_group_heavy":     rdkit_group_heavy,
        "rdkit_src_rings":       rdkit_src_rings,
        "rdkit_prod_rings":      rdkit_prod_rings,
        "rdkit_elem_at_anchor":  rdkit_elem_at_anchor,
    }


def _null_result() -> dict:
    return {
        "s1_idx_valid":          None,
        "s1_element_match":      None,
        "s2_group_valid":        None,
        "s2_heavy_atoms_ok":     None,
        "s3_product_valid":      None,
        "s4_src_heavy_ok":       None,
        "s4_prod_heavy_ok":      None,
        "s4_delta_arithmetic":   None,
        "s5_src_rings_ok":       None,
        "s5_prod_rings_ok":      None,
        "s5_delta_arithmetic":   None,
        "all_pass":              False,
        "outcome":               None,
        "rdkit_src_heavy":       None,
        "rdkit_prod_heavy":      None,
        "rdkit_group_heavy":     None,
        "rdkit_src_rings":       None,
        "rdkit_prod_rings":      None,
        "rdkit_elem_at_anchor":  None,
    }


# ────────────────────────────────────────────────────────────────────────────
# Batch verification + summary
# ────────────────────────────────────────────────────────────────────────────

def verify_batch(records: list[dict]) -> tuple[list[dict], dict]:
    """Run verify_one on all records; return (records_with_results, summary)."""
    for rec in records:
        vr = verify_one(rec)
        rec.update(vr)

    n = len(records)

    def _pass(key):
        return sum(1 for r in records if r.get(key) is True)

    def pct(k, d):
        return round(100 * k / d, 1) if d > 0 else 0.0

    n_api   = sum(1 for r in records if r.get("api_success", False))
    n_parse = sum(1 for r in records if r.get("parse_ok", False))

    chk_names = [
        "s1_idx_valid", "s1_element_match",
        "s2_group_valid", "s2_heavy_atoms_ok",
        "s3_product_valid",
        "s4_src_heavy_ok", "s4_prod_heavy_ok", "s4_delta_arithmetic",
        "s5_src_rings_ok", "s5_prod_rings_ok", "s5_delta_arithmetic",
    ]
    n_all_pass = _pass("all_pass")
    n_outcome  = sum(1 for r in records if r.get("outcome") is True)

    per_diff = {}
    for diff in ["easy", "medium", "hard"]:
        sub = [r for r in records if r.get("difficulty") == diff]
        nd  = len(sub)
        per_diff[diff] = {
            "total":    nd,
            "all_pass": sum(1 for r in sub if r.get("all_pass") is True),
            "outcome":  sum(1 for r in sub if r.get("outcome") is True),
            "accuracy": pct(sum(1 for r in sub if r.get("outcome") is True), nd),
        }

    summary = {
        "total":       n,
        "api_success": n_api,
        "parse_ok":    n_parse,
        **{f"{k} [Type I]": f"{_pass(k)}/{n} ({pct(_pass(k),n)}%)" for k in chk_names},
        "all_pass":    f"{n_all_pass}/{n} ({pct(n_all_pass, n)}%)",
        "outcome [Type II — GT match]": f"{n_outcome}/{n} ({pct(n_outcome, n)}%)",
        "per_difficulty": per_diff,
    }
    return records, summary


# ────────────────────────────────────────────────────────────────────────────
# Clean dataset export
# ────────────────────────────────────────────────────────────────────────────

def save_clean_dataset(
    records: list[dict],
    save_path: Path | None = None,
) -> int:
    """Save all_pass=True records as clean PRM training data.

    Returns the count of saved records.
    """
    if save_path is None:
        save_path = RESULTS_DIR / "clean_dataset_all.json"

    clean = [r for r in records if r.get("all_pass") is True]
    with open(save_path, "w") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(clean)}/{len(records)} clean records → {save_path}")
    return len(clean)
