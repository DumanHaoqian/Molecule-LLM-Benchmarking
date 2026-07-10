"""Shared utilities: paths, RDKit helpers, data loading."""
from pathlib import Path
import json
import random
from typing import Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import rdBase, rdMolDescriptors
rdBase.DisableLog("rdApp.error"); rdBase.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_PATH = (PROJECT_ROOT/"dataset"/"mol_understanding"/"2-frag_detect"/"ring_system_scaffold_v2.json")
RESULTS_DIR  = PROJECT_ROOT/"results"/"cot_eval"/"mol_und"/"ring_sys_scaffold_structured_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_raw_data() -> list[dict]:
    with open(DATASET_PATH) as f: return json.load(f)

def select_sample(data: list[dict], n: int = 60, seed: int = 42) -> list[dict]:
    random.seed(seed)
    per_difficulty = n // 3
    by_diff = {}
    for item in data: by_diff.setdefault(item.get("difficulty","unknown"),[]).append(item)
    sampled = []
    for diff in ["easy","medium","hard"]:
        pool = by_diff.get(diff,[])
        random.shuffle(pool); sampled.extend(pool[:per_difficulty])
    random.shuffle(sampled)
    return sampled[:n]

def scaffold_has_non_ring_atoms(scaffold_smiles: str) -> Optional[bool]:
    """Returns True if any atom in scaffold_smiles is NOT in a ring."""
    mol = Chem.MolFromSmiles(scaffold_smiles)
    if mol is None: return None
    for atom in mol.GetAtoms():
        if not atom.IsInRing(): return True
    return False

def get_ring_count(smiles: str) -> Optional[int]:
    """Return SSSR ring count for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None: return None
    return rdMolDescriptors.CalcNumRings(mol)
