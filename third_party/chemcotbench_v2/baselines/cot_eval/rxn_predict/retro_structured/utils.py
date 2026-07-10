"""
Shared utilities: paths, RDKit helpers, data loading.
"""
from pathlib import Path
import json
import random
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
DATASET_PATH = PROJECT_ROOT / "dataset" / "rxn_predict" / "retro_v2.json"
RESULTS_DIR  = PROJECT_ROOT / "results" / "cot_eval" / "rxn_predict" / "retro_structured"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def select_sample(data: list[dict], n: int = 60, seed: int = 42) -> list[dict]:
    """
    Stratified sample: 20 each from easy/medium/hard.
    Maximises rxn_cls diversity within each stratum.
    """
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
# RDKit helpers
# ---------------------------------------------------------------------------

def canonical_smiles(smiles: str) -> Optional[str]:
    """Return canonical SMILES, or None if invalid."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def smiles_match_set(pred: str, gt: str) -> bool:
    """
    Compare sets of dot-separated SMILES canonically.
    Returns True if both sides produce the same set of canonical fragments.
    """
    def canon_set(s: str) -> set:
        result = set()
        for frag in s.split("."):
            frag = frag.strip()
            if frag:
                m = Chem.MolFromSmiles(frag)
                if m:
                    result.add(Chem.MolToSmiles(m))
        return result

    return canon_set(pred) == canon_set(gt)


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
    """RDKit topological fingerprint Tanimoto Similarity (first valid fragment)."""
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
