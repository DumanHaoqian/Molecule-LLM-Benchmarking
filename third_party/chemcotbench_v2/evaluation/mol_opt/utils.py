"""Shared utilities for MolOpt evaluation (reuses baselines/cot_eval utilities)."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, QED
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

# Re-use baseline utilities
from baselines.cot_eval.mol_opt.mol_opt_single_structured.utils import (
    calculate_solubility,
    check_improvement,
    compute_delta,
    scaffold_preserved_per_sample,
    check_fg_v6_smiles,
    check_fg_change_consistent,
    tanimoto_fts,
    canon_smiles,
    THRESHOLD,
    morgan_fp,
)
from baselines.cot_eval.mol_opt.mol_opt_multi_structured.utils import (
    check_dual_improvement,
    FIXED_THRESHOLDS as MULTI_THRESHOLDS,
    SUBTASK_TO_PROPS as _BASE_MULTI_SUBTASK_TO_PROPS,
)

# Baseline uses "+" separator (drd+logp); registry uses "_" (drd_logp)
MULTI_SUBTASK_TO_PROPS = {
    k.replace("+", "_"): v for k, v in _BASE_MULTI_SUBTASK_TO_PROPS.items()
}

TDC_ORACLE_NAME = {
    "drd": "drd2",
    "gsk": "gsk3b",
    "jnk": "jnk3",
}


def _rdkit_logp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolLogP(mol) if mol is not None else None


def _rdkit_qed(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return QED.qed(mol) if mol is not None else None


def get_oracle(prop: str):
    if prop == "solubility":
        return calculate_solubility
    if prop == "logp":
        return _rdkit_logp
    if prop == "qed":
        return _rdkit_qed
    import tdc

    return tdc.Oracle(name=TDC_ORACLE_NAME[prop])


def _canonical(smiles: str) -> str | None:
    """Return canonical SMILES or None on failure."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _scaffold(smiles: str) -> str:
    """Return canonical Murcko scaffold SMILES or empty string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def _parse_smiles_fragment(text: str):
    """Parse text as SMILES or SMARTS fragment for substructure matching."""
    if not text:
        return None
    text = text.strip()
    if "[" in text and ("+" in text or "-" in text):
        return Chem.MolFromSmarts(text)
    mol = Chem.MolFromSmiles(text)
    if mol is not None:
        return mol
    return Chem.MolFromSmarts(text)


def fragment_equivalent(frag_a: str, frag_b: str) -> bool:
    """Check if two fragment SMILES are chemically equivalent via substructure match."""
    if not frag_a or not frag_b:
        return False
    a = _parse_smiles_fragment(frag_a)
    b = _parse_smiles_fragment(frag_b)
    if a is None or b is None:
        return False
    try:
        # Check mutual substructure match
        a_in_b = b.HasSubstructMatch(a)
        b_in_a = a.HasSubstructMatch(b)
        return a_in_b and b_in_a
    except Exception:
        return False


def smiles_equivalent(smiles_a: str, smiles_b: str, tanimoto_threshold: float = 0.85) -> bool:
    """Check if two SMILES are equivalent: canonical match first, then Tanimoto fallback."""
    ca = _canonical(smiles_a)
    cb = _canonical(smiles_b)
    if ca and cb and ca == cb:
        return True
    # Tanimoto fallback for reasonable alternatives
    ts = tanimoto_fts(smiles_a, smiles_b)
    if ts is not None and ts >= tanimoto_threshold:
        return True
    return False


def step2_match(gt_removed: str, gt_added: str, pred_removed: str, pred_added: str,
                src_smiles: str, pred_smiles: str) -> bool:
    """
    Step 2 semantic direction matching.
    Returns True if predicted edit plan is chemically equivalent to GT edit plan.
    """
    # Case 1: Exact canonical fragment match
    removed_match = fragment_equivalent(gt_removed, pred_removed) if gt_removed and pred_removed else (
        (not gt_removed or gt_removed.strip().lower() in {"none", "n/a", "na", "-"}) and
        (not pred_removed or pred_removed.strip().lower() in {"none", "n/a", "na", "-"})
    )
    added_match = fragment_equivalent(gt_added, pred_added) if gt_added and pred_added else (
        (not gt_added or gt_added.strip().lower() in {"none", "n/a", "na", "-"}) and
        (not pred_added or pred_added.strip().lower() in {"none", "n/a", "na", "-"})
    )

    if removed_match and added_match:
        return True

    # Case 2: Direction consistency check via RDKit substructure counts
    mol_src = Chem.MolFromSmiles(src_smiles) if src_smiles else None
    mol_pred = Chem.MolFromSmiles(pred_smiles) if pred_smiles else None
    if mol_src is None or mol_pred is None:
        return False

    # Verify removed direction: gt_removed should decrease from src->pred
    removed_ok = False
    if gt_removed and gt_removed.strip().lower() not in {"none", "n/a", "na", "-"}:
        frag = _parse_smiles_fragment(gt_removed)
        if frag is not None:
            try:
                c_src = len(mol_src.GetSubstructMatches(frag))
                c_pred = len(mol_pred.GetSubstructMatches(frag))
                removed_ok = c_src > c_pred
            except Exception:
                pass

    # Verify added direction: gt_added should increase from src->pred
    added_ok = False
    if gt_added and gt_added.strip().lower() not in {"none", "n/a", "na", "-"}:
        frag = _parse_smiles_fragment(gt_added)
        if frag is not None:
            try:
                c_src = len(mol_src.GetSubstructMatches(frag))
                c_pred = len(mol_pred.GetSubstructMatches(frag))
                added_ok = c_pred > c_src
            except Exception:
                pass

    # If the predicted direction is at least consistent with the chemical transformation
    return removed_ok or added_ok


def _tanimoto_similarity(smiles_a: str, smiles_b: str) -> float:
    """Compute Tanimoto similarity between two SMILES, returning a continuous score in [0, 1]."""
    if not smiles_a or not smiles_b:
        return 0.0
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, 1024)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, 1024)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def scaffold_similarity(scaffold_a: str, scaffold_b: str) -> float:
    """Compute Tanimoto similarity between two Murcko scaffold SMILES."""
    return _tanimoto_similarity(scaffold_a, scaffold_b)


def molecule_similarity(smiles_a: str, smiles_b: str) -> float:
    """Compute Tanimoto similarity between two molecule SMILES."""
    return _tanimoto_similarity(smiles_a, smiles_b)
