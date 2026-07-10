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
DATASET_PATH = (PROJECT_ROOT/"dataset"/"mol_understanding"/"3-permute_smiles"/"permutated_v2.json")
RESULTS_DIR  = PROJECT_ROOT/"results"/"cot_eval"/"mol_und"/"permutated_structured_v2"
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

def canonical_smiles(smiles: str) -> Optional[str]:
    if not smiles: return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        return Chem.MolToSmiles(mol)
    except: return None

def get_num_heavy_atoms(smiles: str) -> Optional[int]:
    """Return heavy atom count for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None: return None
    return mol.GetNumHeavyAtoms()

def get_ring_count(smiles: str) -> Optional[int]:
    """Return SSSR ring count for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None: return None
    return rdMolDescriptors.CalcNumRings(mol)
