"""
Shared utilities for nepp_structured CoT evaluation pipeline (v2 design).
Paths, RDKit helpers, GT parsing, stratified sampling.

New in v2:
  - extract_reactant_smiles_from_query: parse current-step reactants from query text
  - net_charge_from_smiles: RDKit formal charge sum (multi-fragment)
  - heavy_atom_formula: element count dict for heavy atoms
  - formula_match: compare heavy atom formulas (V4 atom conservation)
  - charge_balance: compare net formal charges (V2 charge balance)
  - scaffold_match: Murcko scaffold comparison (V6 scaffold match)
"""
from pathlib import Path
import json
import random
import re as _re2
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
DATASET_PATH = PROJECT_ROOT / "dataset" / "rxn_predict" / "nepp_v2.json"
RESULTS_DIR  = PROJECT_ROOT / "results" / "cot_eval" / "rxn_predict" / "nepp_structured"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def parse_gt(item: dict) -> str:
    """
    Extract GT SMILES for NEPP task.
    The 'gt' field is already a SMILES string (possibly multi-fragment).
    """
    gt = item.get("gt", "")
    if isinstance(gt, str):
        return gt.strip()
    return str(gt).strip()


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
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def smiles_match_set(pred: str, gt: str) -> bool:
    """
    Canonical set-match for (possibly multi-fragment) SMILES.
    Returns True if both sides yield the same set of canonical fragments.
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

    if not pred or not gt:
        return False
    return canon_set(pred) == canon_set(gt)


def all_frags_valid(smiles: str) -> bool:
    """True if every dot-split fragment of smiles is RDKit-parseable."""
    if not smiles:
        return False
    for frag in smiles.split("."):
        frag = frag.strip()
        if frag and Chem.MolFromSmiles(frag) is None:
            return False
    return True


def union_morgan_fp(smiles: str):
    """
    Union Morgan fingerprint for a (possibly multi-fragment) SMILES.
    Each fragment fingerprinted separately; result is bitwise OR.
    Returns None if no fragment is parseable.
    """
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import ExplicitBitVect

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
    from rdkit import DataStructs
    fp_pred = union_morgan_fp(pred_smiles) if pred_smiles else None
    fp_gt   = union_morgan_fp(gt_smiles)   if gt_smiles   else None
    if fp_pred is None or fp_gt is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp_pred, fp_gt))


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


def _largest_valid_mol(smiles: str):
    """Return the heaviest (most atoms) parseable fragment from a SMILES string."""
    if not smiles:
        return None
    best_mol, best_n = None, -1
    for frag in smiles.split("."):
        frag = frag.strip()
        if frag:
            mol = Chem.MolFromSmiles(frag)
            if mol is not None and mol.GetNumAtoms() > best_n:
                best_mol, best_n = mol, mol.GetNumAtoms()
    return best_mol


# ---------------------------------------------------------------------------
# NEPP-specific helpers (v2)
# ---------------------------------------------------------------------------

def extract_reactant_smiles_from_query(query: str) -> str:
    """
    Extract current elementary step reactant SMILES from the NEPP query text.
    Looks for the 'reactants' key inside the 'current_step_info' block.
    """
    m = _re2.search(
        r'"current_step_info".*?"reactants":\s*([^\n,}]+)',
        query,
        _re2.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return ""


def net_charge_from_smiles(smiles: str) -> Optional[int]:
    """
    Compute net formal charge (sum of all atom formal charges) for a
    possibly multi-fragment SMILES.  Returns None if any fragment is invalid.
    """
    if not smiles:
        return None
    total, any_valid = 0, False
    for frag in smiles.split("."):
        frag = frag.strip()
        if not frag:
            continue
        mol = Chem.MolFromSmiles(frag)
        if mol is None:
            return None
        total += sum(a.GetFormalCharge() for a in mol.GetAtoms())
        any_valid = True
    return total if any_valid else None


def heavy_atom_formula(smiles: str) -> Optional[dict]:
    """
    Return a {element: count} dict for all heavy (non-H) atoms in a
    possibly multi-fragment SMILES.  Returns None if any fragment is invalid.
    """
    if not smiles:
        return None
    counts: dict[str, int] = {}
    any_valid = False
    for frag in smiles.split("."):
        frag = frag.strip()
        if not frag:
            continue
        mol = Chem.MolFromSmiles(frag)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            counts[sym] = counts.get(sym, 0) + 1
        any_valid = True
    return counts if any_valid else None


def charge_balance(pred_smiles: str, reactant_smiles: str) -> bool:
    """
    V2 check: net formal charge of predicted product == net formal charge
    of current-step reactants (extracted from query).
    """
    pred_q   = net_charge_from_smiles(pred_smiles)
    react_q  = net_charge_from_smiles(reactant_smiles)
    if pred_q is None or react_q is None:
        return False
    return pred_q == react_q


def formula_match(pred_smiles: str, reactant_smiles: str) -> bool:
    """
    V4 check: heavy-atom formula of predicted product == heavy-atom formula
    of current-step reactants.  Catches hallucinated / missing atoms.
    """
    pred_f  = heavy_atom_formula(pred_smiles)
    react_f = heavy_atom_formula(reactant_smiles)
    if pred_f is None or react_f is None:
        return False
    return pred_f == react_f


def scaffold_match(pred_smiles: str, gt_smiles: str) -> bool:
    """
    V6 check: Murcko scaffold of the largest fragment of pred_smiles ==
    Murcko scaffold of the largest fragment of gt_smiles.
    Acyclic molecules both yield an empty scaffold → treated as a match.
    Returns False if either SMILES is unparseable.
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold

    def _scaffold(smiles: str) -> Optional[str]:
        mol = _largest_valid_mol(smiles)
        if mol is None:
            return None
        try:
            sc = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(sc) if sc is not None else ""
        except Exception:
            return None

    ps = _scaffold(pred_smiles)
    gs = _scaffold(gt_smiles)
    if ps is None or gs is None:
        return False
    return ps == gs


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
