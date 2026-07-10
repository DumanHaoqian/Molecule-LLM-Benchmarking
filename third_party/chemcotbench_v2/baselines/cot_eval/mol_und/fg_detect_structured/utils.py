"""Shared utilities: paths, RDKit helpers, data loading."""
from pathlib import Path
import json
import random
from typing import Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import rdBase, rdMolDescriptors
rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_PATH = PROJECT_ROOT / "dataset" / "mol_understanding" / "1-fg_detect" / "fg_samples_v2.json"
RESULTS_DIR  = PROJECT_ROOT / "results" / "cot_eval" / "mol_und" / "fg_detect_structured_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_raw_data() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)

def select_sample(data: list[dict], n: int = 60, seed: int = 42) -> list[dict]:
    random.seed(seed)
    per_difficulty = n // 3
    by_diff: dict[str, list[dict]] = {}
    for item in data:
        d = item.get("difficulty", "unknown")
        by_diff.setdefault(d, []).append(item)
    sampled = []
    for diff in ["easy", "medium", "hard"]:
        pool = by_diff.get(diff, [])
        pool_sorted = sorted(pool, key=lambda x: x.get("fg_name", ""))
        step = max(1, len(pool_sorted) // per_difficulty)
        picked = pool_sorted[::step][:per_difficulty]
        if len(picked) < per_difficulty:
            picked = pool_sorted[:per_difficulty]
        sampled.extend(picked)
    random.shuffle(sampled)
    return sampled[:n]

_TEST_SMILES = [
    "c1ccccc1", "c1ccncc1", "c1ccoc1", "c1ccsc1", "c1cc[nH]c1",
    "c1cnc[nH]1", "c1ccc2ccccc2c1", "c1ccc2[nH]ccc2c1", "C1CCNCC1", "c1cnccn1",
]
_TEST_MOLS = [Chem.MolFromSmiles(s) for s in _TEST_SMILES]

def smarts_is_valid(smarts: str) -> bool:
    try:
        return Chem.MolFromSmarts(smarts) is not None
    except Exception:
        return False

def smarts_fingerprint(smarts: str) -> Optional[tuple]:
    try:
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            return None
        return tuple(len(m.GetSubstructMatches(pat)) if m else 0 for m in _TEST_MOLS)
    except Exception:
        return None

def smarts_semantically_matches_gt(candidate: str, gt_smarts: str) -> bool:
    fp_c = smarts_fingerprint(candidate)
    fp_g = smarts_fingerprint(gt_smarts)
    if fp_c is None or fp_g is None:
        return False
    return fp_c == fp_g

def apply_smarts_to_mol(smarts: str, smiles: str) -> Optional[int]:
    """Apply SMARTS pattern to a molecule SMILES; return number of non-overlapping matches."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    pat = Chem.MolFromSmarts(smarts) if smarts else None
    if pat is None:
        return None
    try:
        return len(mol.GetSubstructMatches(pat))
    except Exception:
        return None
