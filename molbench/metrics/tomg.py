"""TOMG-Bench (S2) metrics: per-subtask Success Rate (SR) + Weighted SR (WSR).

Ported from phenixace/S2-TOMG-Bench (evaluate.py + utils/evaluation.py), adapted
to the *S2* CSV columns (e.g. FunctionalGroup uses `benzene_ring` + `thioether`).

  SR_t  = successes / total                     (per subtask)
  WSR_t = quality_t * SR_t                       quality = novelty (MolCustom)
                                                          or similarity (MolEdit/MolOpt)
Quality is averaged over *valid* generated molecules (matching upstream).
Novelty needs a ZINC250k reference (TOMG_ZINC_PATH); if absent it degrades to 0.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")

ATOM_COLS = [
    "carbon", "oxygen", "nitrogen", "sulfur", "fluorine", "chlorine", "bromine",
    "iodine", "phosphorus", "boron", "silicon", "selenium", "tellurium",
    "arsenic", "antimony", "bismuth", "polonium",
]
BOND_COLS = ["single", "double", "triple", "rotatable", "aromatic"]
FG_COLS = [
    "benzene_ring", "hydroxyl", "anhydride", "aldehyde", "ketone", "carboxyl",
    "ester", "amide", "amine", "nitro", "halo", "thioether", "nitrile", "thiol",
    "sulfide", "disulfide", "sulfoxide", "sulfone", "borane",
]

_ATOMIC_NUM = {
    "carbon": 6, "nitrogen": 7, "oxygen": 8, "fluorine": 9, "phosphorus": 15,
    "sulfur": 16, "chlorine": 17, "bromine": 35, "iodine": 53, "boron": 5,
    "silicon": 14, "selenium": 34, "tellurium": 52, "arsenic": 33,
    "antimony": 51, "bismuth": 83, "polonium": 84,
}
_BOND_TYPE = {
    "single": Chem.rdchem.BondType.SINGLE,
    "double": Chem.rdchem.BondType.DOUBLE,
    "triple": Chem.rdchem.BondType.TRIPLE,
    "aromatic": Chem.rdchem.BondType.AROMATIC,
}
_FG_SMARTS = {
    "benzene_ring": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
    "hydroxyl": "[OX2H]",
    "anhydride": "[CX3](=[OX1])[OX2][CX3](=[OX1])",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxyl": "[CX3](=O)[OX2H1]",
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "amide": "[NX3][CX3](=[OX1])[#6]",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
    "halo": "[F,Cl,Br,I]",
    "thioether": "[SX2][CX4]",
    "nitrile": "[NX1]#[CX2]",
    "thiol": "[#16X2H]",
    "sulfide": "[#16X2H0]",  # minus disulfides (handled below)
    "disulfide": "[#16X2H0][#16X2H0]",
    "sulfoxide": "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",
    "sulfone": "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",
    "borane": "[BX3]",
}


def _to_mol(smiles):
    if not isinstance(smiles, str) or not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def mol_prop(smiles: str, prop: str):
    """Compute a property of a SMILES string (None if invalid/unknown)."""
    mol = _to_mol(smiles)
    if mol is None:
        return None
    if prop == "validity":
        return True
    if prop == "logP":
        return Descriptors.MolLogP(mol)
    if prop == "MR":
        return Descriptors.MolMR(mol)
    if prop == "qed":
        return Descriptors.qed(mol)
    if prop == "rot_bonds":
        return Descriptors.NumRotatableBonds(mol)
    if prop.startswith("num_") and prop.endswith("_bonds"):
        bt = _BOND_TYPE.get(prop[len("num_"):-len("_bonds")])
        return sum(b.GetBondType() == bt for b in mol.GetBonds()) if bt else None
    if prop.startswith("num_"):
        key = prop[len("num_"):]
        if key in _ATOMIC_NUM:
            z = _ATOMIC_NUM[key]
            return sum(a.GetAtomicNum() == z for a in mol.GetAtoms())
        if key in _FG_SMARTS:
            n = len(mol.GetSubstructMatches(Chem.MolFromSmarts(_FG_SMARTS[key])))
            if key == "sulfide":  # exclude disulfides, per upstream
                n -= len(mol.GetSubstructMatches(Chem.MolFromSmarts(_FG_SMARTS["disulfide"])))
            return n
    return None


def _fp(smiles, n_bits=2048):
    mol = _to_mol(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits) if mol else None


def calculate_similarity(smiles1: str, smiles2: str) -> float:
    f1, f2 = _fp(smiles1), _fp(smiles2)
    if f1 is None or f2 is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(f1, f2)


# ---- novelty (vs ZINC250k reference) ------------------------------------
_ZINC_FPS: Optional[list] = None
_ZINC_ERROR: Optional[str] = None


def _load_zinc_fps():
    global _ZINC_FPS, _ZINC_ERROR
    if _ZINC_FPS is not None or _ZINC_ERROR is not None:
        return _ZINC_FPS
    path = os.environ.get("TOMG_ZINC_PATH")
    if not path or not os.path.isfile(path):
        _ZINC_ERROR = "TOMG_ZINC_PATH not set / not found"
        return None
    fps = []
    with open(path) as f:
        for line in f:
            smi = line.strip().split()[0] if line.strip() else ""
            fp = _fp(smi)
            if fp is not None:
                fps.append(fp)
    _ZINC_FPS = fps
    print(f"[tomg] loaded {len(fps)} ZINC reference fingerprints for novelty")
    return _ZINC_FPS


def novelty(valid_smiles: List[str]) -> Optional[List[float]]:
    """1 - max Tanimoto to the ZINC250k reference, per valid molecule."""
    ref = _load_zinc_fps()
    if ref is None:
        print(f"[tomg] novelty unavailable ({_ZINC_ERROR}); MolCustom WSR will be 0")
        return None
    out = []
    for smi in valid_smiles:
        fp = _fp(smi)
        if fp is None:
            out.append(0.0)
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref)
        out.append(1.0 - max(sims) if sims else 1.0)
    return out


# ---- per-subtask success checks -----------------------------------------

def _norm(group: str) -> str:
    return "benzene_ring" if group == "benzene ring" else group


def _success(subtask: str, row: Dict, pred: str) -> bool:
    if not mol_prop(pred, "validity"):
        return False
    if subtask == "AtomNum":
        return all(mol_prop(pred, "num_" + a) == int(row[a]) for a in ATOM_COLS)
    if subtask == "BondNum":
        for b in BOND_COLS:
            v = int(row[b])
            if v == 0:
                continue
            key = "rot_bonds" if b == "rotatable" else "num_" + b + "_bonds"
            if mol_prop(pred, key) != v:
                return False
        return True
    if subtask == "FunctionalGroup":
        return all(mol_prop(pred, "num_" + g) == int(row[g]) for g in FG_COLS)
    if subtask == "AddComponent":
        g = _norm(row["added_group"])
        return mol_prop(pred, "num_" + g) == mol_prop(row["molecule"], "num_" + g) + 1
    if subtask == "DelComponent":
        g = _norm(row["removed_group"])
        return mol_prop(pred, "num_" + g) == mol_prop(row["molecule"], "num_" + g) - 1
    if subtask == "SubComponent":
        a, r, raw = _norm(row["added_group"]), _norm(row["removed_group"]), row["molecule"]
        return (mol_prop(pred, "num_" + r) == mol_prop(raw, "num_" + r) - 1
                and mol_prop(pred, "num_" + a) == mol_prop(raw, "num_" + a) + 1)
    if subtask in ("LogP", "MR", "QED"):
        prop = {"LogP": "logP", "MR": "MR", "QED": "qed"}[subtask]
        raw = row["molecule"]
        inst = str(row["Instruction"]).lower()
        lower = "lower" in inst or "decrease" in inst
        pv, rv = mol_prop(pred, prop), mol_prop(raw, prop)
        if pv is None or rv is None:
            return False
        return pv < rv if lower else pv > rv
    raise ValueError(f"unknown subtask {subtask}")


# subtask -> task group and quality type ("novelty" | "similarity")
SUBTASKS = {
    "AtomNum": ("MolCustom", "novelty"),
    "BondNum": ("MolCustom", "novelty"),
    "FunctionalGroup": ("MolCustom", "novelty"),
    "AddComponent": ("MolEdit", "similarity"),
    "DelComponent": ("MolEdit", "similarity"),
    "SubComponent": ("MolEdit", "similarity"),
    "LogP": ("MolOpt", "similarity"),
    "MR": ("MolOpt", "similarity"),
    "QED": ("MolOpt", "similarity"),
}
TASK_GROUPS = ["MolCustom", "MolEdit", "MolOpt"]


def score_subtask(subtask: str, rows: List[Dict], preds: List[str]) -> Dict[str, float]:
    """Return {sr, quality, wsr, n, validity} for one subtask."""
    group, qtype = SUBTASKS[subtask]
    successes, valid_preds, valid_rows = [], [], []
    for row, pred in zip(rows, preds):
        valid = bool(mol_prop(pred, "validity"))
        successes.append(1 if (valid and _success(subtask, row, pred)) else 0)
        if valid:
            valid_preds.append(pred)
            valid_rows.append(row)

    n = len(preds)
    sr = sum(successes) / n if n else 0.0
    validity = len(valid_preds) / n if n else 0.0

    if not valid_preds:
        quality = 0.0
    elif qtype == "similarity":
        sims = [calculate_similarity(r["molecule"], p) for r, p in zip(valid_rows, valid_preds)]
        quality = sum(sims) / len(sims)
    else:  # novelty
        nov = novelty(valid_preds)
        quality = sum(nov) / len(nov) if nov else 0.0

    return {"sr": sr, "quality": quality, "wsr": quality * sr, "n": n, "validity": validity}
