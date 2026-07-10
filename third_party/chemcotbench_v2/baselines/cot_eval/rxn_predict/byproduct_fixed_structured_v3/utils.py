"""
Shared utilities for byproduct_fixed_structured_v3.

New in v2 (retained in v3):
  - parse_reactants()           : extract reactant SMILES list from dataset meta
  - heavy_atom_elements()       : get set of element symbols for non-H atoms
  - leaving_fragment_in_reactant() : V4b — all leaving fragment elements present in reactants
  - fragment_subset_of_byproduct() : V4c — leaving fragment elements ⊆ byproduct elements
"""
import json
import random
import re as _re
from pathlib import Path
from typing import Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import rdBase
rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_PATH = PROJECT_ROOT / "dataset" / "rxn_predict" / "byproduct_v2.json"
RESULTS_DIR  = (PROJECT_ROOT / "results" / "cot_eval" / "rxn_predict"
                / "byproduct_fixed_structured_v3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def parse_gt(item: dict) -> str:
    """Parse GT byproduct SMILES from the 'gt' field (JSON string)."""
    gt_raw = item.get("gt", "{}")
    try:
        gt_dict = json.loads(gt_raw)
        return gt_dict.get("By Product", "")
    except (json.JSONDecodeError, AttributeError):
        return gt_raw


def parse_reactants(item: dict) -> list[str]:
    """Extract reactant SMILES list from the 'meta' field."""
    meta_raw = item.get("meta", "")
    if isinstance(meta_raw, str):
        try:
            meta_dict = json.loads(meta_raw)
        except (json.JSONDecodeError, TypeError):
            return []
    elif isinstance(meta_raw, dict):
        meta_dict = meta_raw
    else:
        return []
    reactants = meta_dict.get("reactants", [])
    # Filter out empty strings (reagent placeholders)
    return [r for r in reactants if r and r.strip()]


def select_sample(data: list[dict], n: int = 60, seed: int = 42) -> list[dict]:
    """Stratified sample: 20 each from easy/medium/hard."""
    random.seed(seed)
    per_difficulty = n // 3

    by_diff: dict[str, list[dict]] = {}
    for item in data:
        d = item.get("difficulty", "unknown")
        by_diff.setdefault(d, []).append(item)

    sampled = []
    for diff in ["easy", "medium", "hard"]:
        pool = by_diff.get(diff, [])
        pool_sorted = sorted(pool, key=lambda x: x.get("rxn_cls", ""))
        step = max(1, len(pool_sorted) // per_difficulty)
        picked = pool_sorted[::step][:per_difficulty]
        if len(picked) < per_difficulty:
            picked = pool_sorted[:per_difficulty]
        sampled.extend(picked)

    random.shuffle(sampled)
    return sampled[:n]


# ---------------------------------------------------------------------------
# RDKit helpers (original)
# ---------------------------------------------------------------------------

def canonical_with_ions(smiles: str) -> Optional[str]:
    """Get canonical SMILES. Returns None if invalid."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def smiles_match(pred: str, gt: str) -> bool:
    """Compare SMILES canonically with fragment-set matching."""
    c_pred = canonical_with_ions(pred)
    c_gt   = canonical_with_ions(gt)
    if c_pred is None or c_gt is None:
        return False
    if c_pred == c_gt:
        return True
    pred_frags = {canonical_with_ions(f) for f in pred.split(".") if f}
    gt_frags   = {canonical_with_ions(f) for f in gt.split(".") if f}
    pred_frags.discard(None)
    gt_frags.discard(None)
    return pred_frags == gt_frags


# ---------------------------------------------------------------------------
# Reaction-type matching (original)
# ---------------------------------------------------------------------------

_RXN_STOPWORDS = frozenset({
    'reaction', 'the', 'a', 'an', 'of', 'with', 'and', 'or', 'type', 'via',
    'using', 'from', 'to', 'by', 'in', 'for', 'catalyzed', 'mediated',
    'assisted', 'promoted', 'based',
})


def rxn_type_match(pred: str, gt: str, threshold: float = 0.5) -> bool:
    """Keyword-overlap reaction type matching (>= threshold fraction of GT words in pred)."""
    if not pred or not gt:
        return False

    def _key_words(s: str) -> list:
        s_norm = _re.sub(r'[^\w\s]', ' ', s.lower())
        return [w for w in _re.findall(r'\b\w+\b', s_norm)
                if len(w) >= 3 and w not in _RXN_STOPWORDS]

    gt_words  = _key_words(gt)
    pred_norm = _re.sub(r'[^\w\s]', ' ', pred.lower())

    if not gt_words:
        return bool(pred.strip())

    matching = sum(1 for gw in gt_words if gw in pred_norm)
    return matching / len(gt_words) >= threshold


# ---------------------------------------------------------------------------
# New helpers for V4a / V4b / V4c
# ---------------------------------------------------------------------------

def heavy_atom_elements(smiles: str) -> set[str]:
    """
    Return the set of element symbols for all non-hydrogen atoms in SMILES.
    Handles multi-component (dot-separated) SMILES. Returns empty set if
    SMILES is invalid or contains only H atoms.
    """
    if not smiles:
        return set()
    elements: set[str] = set()
    for fragment in smiles.split("."):
        fragment = fragment.strip()
        if not fragment:
            continue
        mol = Chem.MolFromSmiles(fragment)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() > 1:   # exclude H (atomic num = 1)
                elements.add(atom.GetSymbol())
    return elements


def leaving_fragment_in_reactant(fragment_smiles: str,
                                  reactant_smiles_list: list[str]) -> bool:
    """
    V4b check: all heavy-atom elements of the leaving fragment must be
    present in the combined element pool of the reactants.

    This verifies that the model's claimed leaving fragment is chemically
    grounded in the input (not hallucinated from nowhere).

    Examples:
      fragment="[Br-]",  reactants=["CCBr"]         → {Br} ⊆ {C,Br}     → True
      fragment="O",      reactants=["CC(=O)O.CCO"]  → {O} ⊆ {C,O}       → True
      fragment="F",      reactants=["CCBr"]          → {F} ⊄ {C,Br}      → False
    """
    frag_elements = heavy_atom_elements(fragment_smiles)
    if not frag_elements:
        # Fragment has no heavy atoms (e.g. pure H) — trivially satisfied
        return True

    all_reactant_elements: set[str] = set()
    for r_smi in reactant_smiles_list:
        if r_smi:
            all_reactant_elements |= heavy_atom_elements(r_smi)

    return frag_elements.issubset(all_reactant_elements)


def fragment_subset_of_byproduct(fragment_smiles: str,
                                  byproduct_smiles: str) -> bool:
    """
    V4c check: heavy-atom elements of the leaving fragment are a subset of
    heavy-atom elements of the predicted byproduct.

    This enforces coherence between [LEAVING_FRAGMENT_SMILES] and
    [BYPRODUCT_SMILES]: if the model says Br leaves, the byproduct must
    contain Br.

    Examples:
      fragment="[Br-]",  byproduct="[Na+].[Br-]"  → {Br} ⊆ {Na,Br} → True
      fragment="Cl",     byproduct="[H]Cl"          → {Cl} ⊆ {Cl}   → True
      fragment="Cl",     byproduct="O"              → {Cl} ⊄ {O}    → False
    """
    frag_elements   = heavy_atom_elements(fragment_smiles)
    byprod_elements = heavy_atom_elements(byproduct_smiles)

    if not frag_elements:
        return True   # no heavy atoms in fragment — trivially satisfied

    return frag_elements.issubset(byprod_elements)


# ---------------------------------------------------------------------------
# FTS helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _first_valid_mol(smiles: str):
    if not smiles:
        return None
    for frag in smiles.split("."):
        frag = frag.strip()
        if frag:
            mol = Chem.MolFromSmiles(frag)
            if mol is not None:
                return mol
    return None


def maccs_fts(pred_smiles: str, gt_smiles: str) -> float:
    from rdkit.Chem import MACCSkeys
    from rdkit import DataStructs
    if not pred_smiles or not gt_smiles:
        return 0.0
    try:
        mol_pred = _first_valid_mol(pred_smiles)
        mol_gt   = _first_valid_mol(gt_smiles)
        if mol_pred is None or mol_gt is None:
            return 0.0
        fp_pred = MACCSkeys.GenMACCSKeys(mol_pred)
        fp_gt   = MACCSkeys.GenMACCSKeys(mol_gt)
        return float(DataStructs.TanimotoSimilarity(fp_pred, fp_gt))
    except Exception:
        return 0.0


def rdkit_fts(pred_smiles: str, gt_smiles: str) -> float:
    from rdkit.Chem import RDKFingerprint
    from rdkit import DataStructs
    if not pred_smiles or not gt_smiles:
        return 0.0
    try:
        mol_pred = _first_valid_mol(pred_smiles)
        mol_gt   = _first_valid_mol(gt_smiles)
        if mol_pred is None or mol_gt is None:
            return 0.0
        fp_pred = RDKFingerprint(mol_pred)
        fp_gt   = RDKFingerprint(mol_gt)
        return float(DataStructs.TanimotoSimilarity(fp_pred, fp_gt))
    except Exception:
        return 0.0
