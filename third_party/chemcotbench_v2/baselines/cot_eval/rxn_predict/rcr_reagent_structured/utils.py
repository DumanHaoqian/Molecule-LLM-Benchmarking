"""
Shared utilities for rcr_reagent structured CoT evaluation pipeline.
Paths, data loading, RDKit helpers, and reaction-info extraction.
"""
from pathlib import Path
import json
import re
import random
from collections import Counter
from typing import Optional

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import ExplicitBitVect

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_PATH = PROJECT_ROOT / "dataset" / "rxn_predict" / "rcr_reagent_v2.json"
RESULTS_DIR  = PROJECT_ROOT / "results" / "cot_eval" / "rxn_predict" / "rcr_reagent_structured"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def select_sample(data: list[dict], n: int = 60, seed: int = 42) -> list[dict]:
    """Stratified sample: n//3 each from easy/medium/hard, maximising rxn_cls diversity."""
    random.seed(seed)
    per_diff = n // 3

    by_diff: dict[str, list] = {}
    for item in data:
        d = item.get("difficulty", "unknown")
        by_diff.setdefault(d, []).append(item)

    sampled = []
    for diff in ["easy", "medium", "hard"]:
        pool = sorted(by_diff.get(diff, []), key=lambda x: x.get("rxn_cls", ""))
        step = max(1, len(pool) // per_diff)
        picked = pool[::step][:per_diff]
        if len(picked) < per_diff:
            picked = pool[:per_diff]
        sampled.extend(picked)

    random.shuffle(sampled)
    return sampled[:n]

# ---------------------------------------------------------------------------
# Reaction-info extraction from original query
# ---------------------------------------------------------------------------

def parse_rxn_info(query: str) -> tuple[str, str]:
    """Extract (rxn_smiles, rxn_type) from the end of an rcr query string.

    The query ends with:
        Reaction: {rxn_SMILES}. Reaction Tye: {rxn_type}.
    ("Tye" is a dataset typo; we match both spellings.)
    """
    m = re.search(
        r'Reaction:\s*(.+?)\.\s*Reaction\s+Ty[ep]e?:?\s*(.+?)\.?\s*$',
        query.strip(), re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()
    lines = [l.strip() for l in query.strip().split('\n') if l.strip()]
    last = lines[-1] if lines else ""
    return last, "unknown"


def parse_gt(item: dict) -> str:
    """Return the GT SMILES string for an rcr_reagent item."""
    return item.get("gt", "").strip()

# ---------------------------------------------------------------------------
# RDKit helpers
# ---------------------------------------------------------------------------

def canonical_smiles(smiles: str) -> Optional[str]:
    """Return canonical SMILES, or None if invalid."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def smiles_match_set(pred: str, gt: str) -> bool:
    """Canonical set-match for (possibly multi-fragment) SMILES."""
    if not pred or not gt:
        return False

    def canon_set(s: str) -> set:
        result = set()
        for frag in s.split("."):
            frag = frag.strip()
            if frag:
                mol = Chem.MolFromSmiles(frag)
                if mol:
                    result.add(Chem.MolToSmiles(mol))
        return result

    return canon_set(pred) == canon_set(gt)


def union_morgan_fp(smiles: str) -> Optional[ExplicitBitVect]:
    """Union Morgan fingerprint for a (possibly multi-fragment) SMILES."""
    if not smiles:
        return None
    fps = []
    for frag in smiles.split("."):
        frag = frag.strip()
        if not frag:
            continue
        mol = Chem.MolFromSmiles(frag)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    if not fps:
        return None
    if len(fps) == 1:
        return fps[0]
    n_bits = fps[0].GetNumBits()
    result = ExplicitBitVect(n_bits)
    for fp in fps:
        result |= fp
    return result


def fts(pred_smiles: str, gt_smiles: str) -> float:
    """Fingerprint Tanimoto Similarity between two (possibly multi-fragment) SMILES."""
    fp_pred = union_morgan_fp(pred_smiles) if pred_smiles else None
    fp_gt   = union_morgan_fp(gt_smiles)   if gt_smiles   else None
    if fp_pred is None or fp_gt is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp_pred, fp_gt))


def all_frags_valid(smiles: str) -> bool:
    """True if every dot-split fragment is RDKit-parseable."""
    if not smiles:
        return False
    for frag in smiles.split("."):
        frag = frag.strip()
        if frag and Chem.MolFromSmiles(frag) is None:
            return False
    return True


# ---------------------------------------------------------------------------
# Reaction-type matching
# ---------------------------------------------------------------------------
import re as _re

_RXN_STOPWORDS = frozenset({
    'reaction', 'the', 'a', 'an', 'of', 'with', 'and', 'or', 'type', 'via',
    'using', 'from', 'to', 'by', 'in', 'for', 'catalyzed', 'mediated',
    'assisted', 'promoted', 'based',
})


def rxn_type_match(pred: str, gt: str, threshold: float = 0.5) -> bool:
    """
    Keyword-overlap based reaction type matching.
    Returns True if >= threshold fraction of GT key words appear (as substrings)
    in the normalized prediction string.
    """
    if not pred or not gt:
        return False

    def _key_words(s: str) -> list:
        s_norm = _re.sub(r'[^\w\s]', ' ', s.lower())
        return [w for w in _re.findall(r'\b\w+\b', s_norm)
                if len(w) >= 3 and w not in _RXN_STOPWORDS]

    gt_words = _key_words(gt)
    pred_norm = _re.sub(r'[^\w\s]', ' ', pred.lower())

    if not gt_words:
        return bool(pred.strip())

    matching = sum(1 for gw in gt_words if gw in pred_norm)
    return matching / len(gt_words) >= threshold


def _first_valid_mol(smiles: str):
    """Return the first parseable RDKit Mol from a dot-separated SMILES string."""
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
    """MACCS key fingerprint Tanimoto Similarity (first valid fragment)."""
    from rdkit.Chem import MACCSkeys
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
    """RDKit topological fingerprint Tanimoto Similarity (first valid fragment)."""
    from rdkit.Chem import RDKFingerprint
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


# ---------------------------------------------------------------------------
# Atomic delta analysis (for V2)
# ---------------------------------------------------------------------------

def get_atom_delta(rxn_smiles: str) -> dict:
    """Returns dict of {element: count_delta} for product - reactant (ignoring H).

    Only positive deltas (atoms gained in products) are returned.
    """
    try:
        parts = rxn_smiles.split('>>')
        if len(parts) != 2:
            return {}
        reactant_smi, product_smi = parts

        def count_atoms(smi_block: str) -> Counter:
            counts: Counter = Counter()
            for smi in smi_block.split('.'):
                m = Chem.MolFromSmiles(smi.strip())
                if m:
                    for atom in m.GetAtoms():
                        sym = atom.GetSymbol()
                        if sym != 'H':
                            counts[sym] += 1
            return counts

        r_counts = count_atoms(reactant_smi)
        p_counts = count_atoms(product_smi)
        delta = {}
        all_elements = set(r_counts.keys()) | set(p_counts.keys())
        for elem in all_elements:
            d = p_counts.get(elem, 0) - r_counts.get(elem, 0)
            if d > 0:
                delta[elem] = d
        return delta
    except Exception:
        return {}
