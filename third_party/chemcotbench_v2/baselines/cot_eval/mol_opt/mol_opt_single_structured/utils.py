"""
utils.py — mol_opt_single_structured
=====================================
Shared utilities: data loading, stratified sampling, oracle evaluation,
FTS computation, and path constants.

V1 alignment:
  - TDC oracle for logp/qed/drd/gsk/jnk (names: logp/qed/drd2/gsk3b/jnk3)
  - QSPR formula for solubility (same as V1 eval_metric.py)
  - SR% criterion: oracle(pred) - oracle(src) > 0
  - best_rate criterion: oracle(pred) - oracle(src) >= THRESHOLD[prop]
  - scaffold consistency: hard (identical Murcko scaffold fraction),
                          soft (mean Tanimoto of Murcko scaffold fps)
"""

import json
import os
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdFMCS
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.environ.get("CHEMCOTBENCH_ROOT", Path(__file__).resolve().parents[4]))
DATASET_BASE = PROJECT_ROOT / "dataset" / "deep_mol_opt"
RESULTS_BASE = PROJECT_ROOT / "results" / "cot_eval" / "mol_opt" / "single"

# ─── Property metadata ────────────────────────────────────────────────────────
PROP_DESC = {
    "logp":       "LogP (lipophilicity / distribution coefficient)",
    "qed":        "QED (drug-likeness)",
    "solubility": "Aqueous Solubility (logS, QSPR-based)",
    "drd":        "DRD2 activity (Dopamine D2 Receptor)",
    "gsk":        "GSK3β inhibition (Glycogen Synthase Kinase 3-beta)",
    "jnk":        "JNK3 inhibition (c-Jun N-terminal kinase 3)",
}

# TDC oracle name mapping (V1 aligned)
TDC_ORACLE_NAME = {
    "logp": "logp",
    "qed":  "qed",
    "drd":  "drd2",
    "gsk":  "gsk3b",
    "jnk":  "jnk3",
}

VALID_PROPS = ["logp", "qed", "solubility", "drd", "gsk", "jnk"]

# V1-aligned improvement thresholds for best_rate
THRESHOLD = {
    "logp":       0.5,
    "solubility": 0.5,
    "qed":        0.3,
    "drd":        0.3,
    "gsk":        0.3,
    "jnk":        0.3,
}


# ─── Oracle / solubility ──────────────────────────────────────────────────────
def calculate_solubility(smiles: str):
    """V1-aligned QSPR logS formula."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw  = Descriptors.MolWt(mol)
    lp  = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    return 0.16 - 0.63 * lp - 0.0062 * mw + 0.066 * hbd - 0.074 * hba


def get_oracle(prop: str):
    """Return a callable oracle for the given property (V1-aligned)."""
    if prop == "solubility":
        return calculate_solubility
    import tdc
    return tdc.Oracle(name=TDC_ORACLE_NAME[prop])


def check_improvement(oracle_fn, src_smiles: str, pred_smiles: str) -> bool:
    """V1 SR% criterion: any positive delta > 0."""
    if not pred_smiles:
        return False
    if Chem.MolFromSmiles(pred_smiles) is None:
        return False
    try:
        src_val  = oracle_fn(src_smiles)
        pred_val = oracle_fn(pred_smiles)
        if src_val is None or pred_val is None:
            return False
        return float(pred_val) - float(src_val) > 0
    except Exception:
        return False


def compute_delta(oracle_fn, src_smiles: str, pred_smiles: str) -> float:
    """Return oracle(pred) - oracle(src) as a float. Returns 0.0 on any error."""
    if not pred_smiles or Chem.MolFromSmiles(pred_smiles) is None:
        return 0.0
    try:
        src_val  = oracle_fn(src_smiles)
        pred_val = oracle_fn(pred_smiles)
        if src_val is None or pred_val is None:
            return 0.0
        return float(pred_val) - float(src_val)
    except Exception:
        return 0.0


def scaffold_consistency(src_list: list, pred_list: list) -> dict:
    """
    V1-aligned scaffold consistency metrics over a list of (src, pred) pairs.

    Args:
        src_list:  list of source SMILES (all n samples, including invalid preds)
        pred_list: list of predicted SMILES (invalid/empty → score 0)

    Returns:
        hard: fraction of pairs sharing an identical Murcko scaffold
        soft: mean Tanimoto similarity of Murcko scaffold fingerprints
    """
    assert len(src_list) == len(pred_list)
    total = len(src_list)
    if total == 0:
        return {"hard": 0.0, "soft": 0.0}

    count_same = 0
    scaffold_scores = []

    for src_smi, pred_smi in zip(src_list, pred_list):
        m_src  = Chem.MolFromSmiles(src_smi)  if src_smi  else None
        m_pred = Chem.MolFromSmiles(pred_smi) if pred_smi else None

        if m_src is None or m_pred is None:
            scaffold_scores.append(0.0)
            continue

        sc_src  = MurckoScaffoldSmiles(src_smi)
        sc_pred = MurckoScaffoldSmiles(pred_smi)

        if sc_src == sc_pred:
            count_same += 1
            scaffold_scores.append(1.0)
        else:
            msc_src  = Chem.MolFromSmiles(sc_src)  if sc_src  else None
            msc_pred = Chem.MolFromSmiles(sc_pred) if sc_pred else None
            if msc_src is None or msc_pred is None:
                scaffold_scores.append(0.0)
                continue
            mcs = rdFMCS.FindMCS([msc_src, msc_pred])
            if mcs.numAtoms > 0:
                fp1 = AllChem.GetMorganFingerprintAsBitVect(msc_src,  2, 1024)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(msc_pred, 2, 1024)
                scaffold_scores.append(float(DataStructs.TanimotoSimilarity(fp1, fp2)))
            else:
                scaffold_scores.append(0.0)

    return {
        "hard": round(count_same / total, 4),
        "soft": round(sum(scaffold_scores) / total, 4),
    }


# ─── Per-sample scaffold check (V5) ──────────────────────────────────────────

def scaffold_preserved_per_sample(src_smi: str, pred_smi: str) -> bool:
    """
    V5 (per-sample): True if Murcko scaffold of pred == Murcko scaffold of src.
    Returns False if either SMILES is invalid.
    """
    mol_src  = Chem.MolFromSmiles(src_smi)  if src_smi  else None
    mol_pred = Chem.MolFromSmiles(pred_smi) if pred_smi else None
    if mol_src is None or mol_pred is None:
        return False
    try:
        sc_src  = MurckoScaffoldSmiles(mol=mol_src,  includeChirality=False)
        sc_pred = MurckoScaffoldSmiles(mol=mol_pred, includeChirality=False)
    except Exception:
        return False
    return sc_src == sc_pred


# ─── Functional-group change consistency (V6) ────────────────────────────────

# (keyword_in_text_lowercase, SMARTS) — more-specific patterns listed first
_FG_KEYWORD_SMARTS = [
    ("-cf3",            "[CX4](F)(F)F"),
    ("trifluoromethyl", "[CX4](F)(F)F"),
    ("-och3",           "[OX2][CH3]"),
    ("methoxy",         "[OX2][CH3]"),
    ("-cooh",           "[CX3](=O)[OX2H1]"),
    ("carboxylic acid", "[CX3](=O)[OX2H1]"),
    ("carboxyl",        "[CX3](=O)[OX2H1]"),
    ("ester",           "[CX3](=O)O[#6]"),
    ("amide",           "[NX3][CX3]=[OX1]"),
    ("-no2",            "[N+](=O)[O-]"),
    ("nitro",           "[N+](=O)[O-]"),
    ("-cn",             "[CX2]#N"),
    ("nitrile",         "[CX2]#N"),
    ("cyano",           "[CX2]#N"),
    ("sulfonyl",        "[SX4](=O)(=O)"),
    ("sulfone",         "[SX4](=O)(=O)"),
    ("-oh",             "[OX2H]"),
    ("hydroxyl",        "[OX2H]"),
    ("-nh2",            "[NH2]"),
    ("primary amine",   "[NH2;!$(NC=O)]"),
    ("amino",           "[NX3;H2,H1;!$(NC=O)]"),
    ("amine",           "[NX3;H2,H1;!$(NC=O)]"),
    ("-cl",             "[Cl]"),
    ("chloro",          "[Cl]"),
    ("chlorine",        "[Cl]"),
    ("-f",              "[F]"),
    ("fluoro",          "[F]"),
    ("fluorine",        "[F]"),
    ("-br",             "[Br]"),
    ("bromo",           "[Br]"),
    ("bromine",         "[Br]"),
    ("-i",              "[I]"),
    ("iodo",            "[I]"),
    ("iodine",          "[I]"),
    ("halogen",         "[F,Cl,Br,I]"),
    ("methyl",          "[CH3]"),
    ("piperazine",      "N1CCNCC1"),
    ("morpholine",      "O1CCNCC1"),
    ("phenyl",          "c1ccccc1"),
    ("tert-butyl",      "C(C)(C)C"),
]


def _count_smarts_matches(mol, smarts: str) -> int:
    """Count non-overlapping SMARTS matches in mol; returns -1 on error."""
    try:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            return -1
        return len(mol.GetSubstructMatches(patt))
    except Exception:
        return -1


def check_fg_change_consistent(text: str, mol_src, mol_pred) -> bool:
    """
    V6: Return True if at least one functional group keyword mentioned in `text`
    has a changed SMARTS match count between mol_src and mol_pred.

    `text` is from [FG_CHANGE] (or modification strategy field as fallback).
    mol_src / mol_pred must be valid RDKit Mol objects.
    """
    if not text:
        return False
    text_lower = text.lower()
    for kw, smarts in _FG_KEYWORD_SMARTS:
        if kw not in text_lower:
            continue
        c_src  = _count_smarts_matches(mol_src,  smarts)
        c_pred = _count_smarts_matches(mol_pred, smarts)
        if c_src < 0 or c_pred < 0:
            continue
        if c_src != c_pred:
            return True
    return False



# ─── SMILES-based FG verification (V6 improved) ──────────────────────────────

def _parse_smiles_fragment(text: str):
    """Parse a FG fragment for use as a substructure query.

    For fragments containing formal charges (e.g. [N+](=O)[O-], [O-], [NH4+]),
    MolFromSmiles produces a molecule whose internal representation prevents
    substructure matching from working correctly in RDKit.  Using MolFromSmarts
    for those cases fixes the issue while remaining compatible with plain SMILES.
    """
    if not text:
        return None
    text = text.strip()
    # Use SMARTS path for fragments with formal charges so that substructure
    # matching works correctly (e.g. nitro groups, carboxylates, ammonium).
    if "[" in text and ("+" in text or "-" in text):
        return Chem.MolFromSmarts(text)
    mol = Chem.MolFromSmiles(text)
    if mol is not None:
        return mol
    return Chem.MolFromSmarts(text)


def check_fg_v6_smiles(fg_removed: str, fg_added: str, mol_src, mol_pred) -> bool:
    """
    V6 (SMILES-based): Verify claimed FG changes match actual SMILES changes.

    Direction-aware:
    - fg_removed verified if: count(fg_removed in src) > count(fg_removed in pred)
    - fg_added   verified if: count(fg_added   in pred) > count(fg_added   in src)

    Returns True if at least one direction verifies correctly.
    """
    if mol_src is None or mol_pred is None:
        return False

    _NONE = {"none", "n/a", "na", "-", ""}
    removed_ok = False
    added_ok   = False

    if fg_removed and fg_removed.strip().lower() not in _NONE:
        frag = _parse_smiles_fragment(fg_removed)
        if frag is not None:
            try:
                c_src  = len(mol_src.GetSubstructMatches(frag))
                c_pred = len(mol_pred.GetSubstructMatches(frag))
                removed_ok = c_src > c_pred
            except Exception:
                pass

    if fg_added and fg_added.strip().lower() not in _NONE:
        frag = _parse_smiles_fragment(fg_added)
        if frag is not None:
            try:
                c_src  = len(mol_src.GetSubstructMatches(frag))
                c_pred = len(mol_pred.GetSubstructMatches(frag))
                added_ok = c_pred > c_src
            except Exception:
                pass

    return removed_ok or added_ok

# ─── FTS (Fingerprint Tanimoto Similarity) ───────────────────────────────────
def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto_fts(smiles_a: str, smiles_b: str) -> float | None:
    """Tanimoto similarity between two SMILES strings."""
    fp_a = morgan_fp(smiles_a)
    fp_b = morgan_fp(smiles_b)
    if fp_a is None or fp_b is None:
        return None
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def canon_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_dataset(prop: str) -> list[dict]:
    path = DATASET_BASE / prop / "final_mmp_v2.json"
    return json.load(open(path))


def select_sample(dataset: list[dict],
                  n_per_diff: int = 20,
                  seed: int = 42) -> list[dict]:
    """Stratified sample: n_per_diff items per difficulty level."""
    rng = random.Random(seed)
    by_diff: dict[str, list] = {"easy": [], "medium": [], "hard": []}
    for item in dataset:
        d = item.get("difficulty", "medium")
        if d in by_diff:
            by_diff[d].append(item)
    sampled = []
    for d, items in by_diff.items():
        k = min(n_per_diff, len(items))
        sampled.extend(rng.sample(items, k))
    return sampled


def get_results_dir(prop: str) -> Path:
    d = RESULTS_BASE / prop
    d.mkdir(parents=True, exist_ok=True)
    return d
