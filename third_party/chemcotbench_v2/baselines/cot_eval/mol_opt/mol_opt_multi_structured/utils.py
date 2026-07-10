"""
utils.py — mol_opt_multi_structured
=====================================
Shared utilities for multi-objective mol_opt evaluation.

Dataset: dataset/deep_mol_opt_multi/final_multi_mmp_v2.json (300 items)
Filter by objective_props to get each subtask's 50 items.

V1 alignment:
  - TDC oracle for drd/gsk/logp/qed; QSPR for solubility
  - SR% criterion (dual): BOTH delta(pred, src) >= threshold for each property
    Thresholds are fixed per-property (stored in dataset objective_thresholds):
      drd=0.3, gsk=0.3, logp=0.5, qed=0.3, solubility=0.5
"""

import json
import os
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.environ.get("CHEMCOTBENCH_ROOT", Path(__file__).resolve().parents[4]))
DATASET_PATH = PROJECT_ROOT / "dataset" / "deep_mol_opt_multi" / "final_multi_mmp_v2.json"
RESULTS_BASE = PROJECT_ROOT / "results" / "cot_eval" / "mol_opt" / "multi"

# ─── Subtask definitions ──────────────────────────────────────────────────────
VALID_SUBTASKS = [
    "drd+logp", "drd+solubility", "gsk+logp",
    "logp+qed", "logp+solubility", "qed+solubility",
]

SUBTASK_TO_PROPS = {
    "drd+logp":        ["drd", "logp"],
    "drd+solubility":  ["drd", "solubility"],
    "gsk+logp":        ["gsk", "logp"],
    "logp+qed":        ["logp", "qed"],
    "logp+solubility": ["logp", "solubility"],
    "qed+solubility":  ["qed", "solubility"],
}

PROP_DESC = {
    "logp":       "LogP (lipophilicity / distribution coefficient)",
    "qed":        "QED (drug-likeness)",
    "solubility": "Aqueous Solubility (logS, QSPR-based)",
    "drd":        "DRD2 activity (Dopamine D2 Receptor)",
    "gsk":        "GSK3β inhibition (Glycogen Synthase Kinase 3-beta)",
    "jnk":        "JNK3 inhibition (c-Jun N-terminal kinase 3)",
}

TDC_ORACLE_NAME = {
    "logp": "logp",
    "qed":  "qed",
    "drd":  "drd2",
    "gsk":  "gsk3b",
    "jnk":  "jnk3",
}

# Fixed per-property thresholds used across all items (from dataset objective_thresholds)
FIXED_THRESHOLDS: dict[str, float] = {
    "drd":        0.3,
    "gsk":        0.3,
    "logp":       0.5,
    "qed":        0.3,
    "solubility": 0.5,
    "jnk":        0.3,
}


# ─── Oracle functions ─────────────────────────────────────────────────────────
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
    """Return a callable oracle for the given property."""
    if prop == "solubility":
        return calculate_solubility
    import tdc
    return tdc.Oracle(name=TDC_ORACLE_NAME[prop])


def check_single_improvement(oracle_fn, src: str, pred: str) -> bool:
    """Check if oracle(pred) > oracle(src) for one property (unconstrained)."""
    if not pred or Chem.MolFromSmiles(pred) is None:
        return False
    try:
        sv = oracle_fn(src)
        pv = oracle_fn(pred)
        if sv is None or pv is None:
            return False
        return float(pv) - float(sv) > 0
    except Exception:
        return False


def check_dual_improvement(
    oracle_fns: dict,
    src: str,
    pred: str,
    thresholds: dict | None = None,
) -> tuple[bool, dict]:
    """
    Check if both properties meet the improvement threshold.
    Returns (both_correct, {prop: delta}).

    Args:
        oracle_fns:  {prop_name: oracle_callable}
        src:         source SMILES
        pred:        predicted SMILES
        thresholds:  {prop_name: min_required_delta}
                     Defaults to FIXED_THRESHOLDS if None.
    """
    if thresholds is None:
        thresholds = FIXED_THRESHOLDS

    if not pred or Chem.MolFromSmiles(pred) is None:
        return False, {}

    deltas: dict = {}
    for prop, fn in oracle_fns.items():
        try:
            sv = fn(src)
            pv = fn(pred)
            deltas[prop] = float(pv) - float(sv) if (sv is not None and pv is not None) else None
        except Exception:
            deltas[prop] = None

    both_ok = all(
        deltas.get(p) is not None and deltas[p] >= thresholds.get(p, 0.0)
        for p in oracle_fns
    )
    return both_ok, deltas


# ─── Scaffold consistency ─────────────────────────────────────────────────────
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


def scaffold_consistency(src_list: list[str], pred_list: list[str]) -> dict:
    """
    Compute scaffold hard / soft consistency (V1-aligned).
    Returns {"hard": float, "soft": float} over all (src, pred) pairs.
    Invalid SMILES contribute 0.0 to both metrics.
    """
    n = len(src_list)
    if n == 0:
        return {"hard": 0.0, "soft": 0.0}

    count_same = 0
    soft_scores: list[float] = []

    for src_smi, pred_smi in zip(src_list, pred_list):
        m_src  = Chem.MolFromSmiles(src_smi)
        m_pred = Chem.MolFromSmiles(pred_smi) if pred_smi else None
        if m_src is None or m_pred is None:
            soft_scores.append(0.0)
            continue
        try:
            scf_src  = MurckoScaffoldSmiles(src_smi)
            scf_pred = MurckoScaffoldSmiles(pred_smi)
        except Exception:
            soft_scores.append(0.0)
            continue

        if scf_src == scf_pred:
            count_same += 1
            soft_scores.append(1.0)
        else:
            try:
                fp1 = AllChem.GetMorganFingerprintAsBitVect(
                    Chem.MolFromSmiles(scf_src), 2, 1024)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(
                    Chem.MolFromSmiles(scf_pred), 2, 1024)
                soft_scores.append(float(DataStructs.TanimotoSimilarity(fp1, fp2)))
            except Exception:
                soft_scores.append(0.0)

    return {
        "hard": count_same / n,
        "soft": sum(soft_scores) / n,
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
    """Try to parse text as SMILES first, then as SMARTS. Returns None on failure."""
    if not text:
        return None
    text = text.strip()
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

# ─── FTS ─────────────────────────────────────────────────────────────────────
def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto_fts(smiles_a: str, smiles_b: str):
    fp_a = morgan_fp(smiles_a)
    fp_b = morgan_fp(smiles_b)
    if fp_a is None or fp_b is None:
        return None
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def canon_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


# ─── Data loading ─────────────────────────────────────────────────────────────
def _subtask_key(item: dict) -> str:
    """Convert objective_props list to canonical subtask string."""
    props = sorted(item.get("objective_props", []))
    return "+".join(props)


def build_threshold_lookup() -> dict[str, dict]:
    """
    Build {sample_id: {prop: threshold}} from the full dataset.
    Used to join thresholds back into sampled_cot / direct_sampled records
    that were saved without thresholds.
    """
    all_data = json.load(open(DATASET_PATH))
    return {item["id"]: item["objective_thresholds"] for item in all_data}


def load_subtask_data(subtask: str) -> list[dict]:
    """Load all items for a given subtask from the multi-objective dataset."""
    props_set = set(SUBTASK_TO_PROPS[subtask])
    all_data  = json.load(open(DATASET_PATH))
    return [
        item for item in all_data
        if set(item.get("objective_props", [])) == props_set
    ]


def select_sample(subtask_data: list[dict],
                  n_per_diff: int = 15,
                  seed: int = 42) -> list[dict]:
    """Stratified sample: n_per_diff per difficulty."""
    rng = random.Random(seed)
    by_diff: dict[str, list] = {"easy": [], "medium": [], "hard": []}
    for item in subtask_data:
        d = item.get("difficulty", "medium")
        if d in by_diff:
            by_diff[d].append(item)
    sampled = []
    for d, items in by_diff.items():
        k = min(n_per_diff, len(items))
        sampled.extend(rng.sample(items, k))
    return sampled


def subtask_to_dir(subtask: str) -> str:
    """Convert 'drd+logp' → 'drd_logp' for directory naming."""
    return subtask.replace("+", "_")


def get_results_dir(subtask: str) -> Path:
    d = RESULTS_BASE / subtask_to_dir(subtask)
    d.mkdir(parents=True, exist_ok=True)
    return d
