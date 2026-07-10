"""
Shared utilities for forward_structured CoT evaluation pipeline.
Paths, RDKit helpers, GT parsing, stratified sampling.
V2 additions: FG_SMARTS_DICT, MECHANISM_KEYWORDS, chemical validity helpers.
"""
from pathlib import Path
import json
import re as _re
import random
from collections import Counter
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
DATASET_PATH = PROJECT_ROOT / "dataset" / "rxn_predict" / "forward_v2.json"
RESULTS_DIR  = PROJECT_ROOT / "results" / "cot_eval" / "rxn_predict" / "forward_structured"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def parse_gt(item: dict) -> str:
    """Extract major product SMILES from forward task gt (JSON string)."""
    gt_raw = item.get("gt", "{}")
    if isinstance(gt_raw, str):
        try:
            gt_dict = json.loads(gt_raw)
            return gt_dict.get("Major Product", "")
        except (json.JSONDecodeError, AttributeError):
            return gt_raw
    elif isinstance(gt_raw, dict):
        return gt_raw.get("Major Product", "")
    return str(gt_raw)


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


# ---------------------------------------------------------------------------
# Functional Group SMARTS dictionary  (V2 / V3 grounding)
# ---------------------------------------------------------------------------

_FG_RAW: dict[str, str] = {
    # Nitrogen
    "amine":            "[NX3;H2,H1;!$(NC=O);!$(NS(=O)=O)]",
    "primary amine":    "[NX3H2;!$(NC=O);!$(NS(=O)=O)]",
    "secondary amine":  "[NX3H1;!$(NC=O);!$(NS(=O)=O)]",
    "aniline":          "[NH2,NHX3]c",
    "amide":            "[CX3](=[OX1])[NX3]",
    "sulfonamide":      "[SX4](=O)(=O)[NX3]",
    "carbamate":        "[NX3][CX3](=O)[OX2][#6]",
    "urea":             "[NX3][CX3](=O)[NX3]",
    "nitro":            "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "nitrile":          "[NX1]#[CX2]",
    "imine":            "[CX3]=[NX2]",
    "azide":            "[$([NX2]=[NX2+]=[NX1-]),$([NX2-][NX2+]=[NX2])]",
    # Oxygen
    "alcohol":          "[OX2H][CX4]",
    "phenol":           "[OX2H]c",
    "ether":            "[OD2;!$(OC=O);!$(Oc)]([#6])[#6]",
    "ketone":           "[#6][CX3](=[OX1])[#6]",
    "aldehyde":         "[CX3H1](=[OX1])",
    "carboxylic acid":  "[CX3](=O)[OX2H1]",
    "ester":            "[CX3](=[OX1])[OX2][#6]",
    "epoxide":          "[C]1[O][C]1",
    "peroxide":         "[OX2][OX2]",
    "hydroperoxide":    "[OX2H][OX2]",
    "anhydride":        "[CX3](=O)[OX2][CX3](=O)",
    "carbonate":        "[OX2][CX3](=O)[OX2]",
    "acyl chloride":    "[CX3](=O)[Cl]",
    "carbonyl":         "[CX3]=[OX1]",
    # Sulfur
    "thiol":            "[SX2H]",
    "sulfide":          "[SX2]([#6])[#6]",
    "thioether":        "[SX2]([#6])[#6]",
    "sulfoxide":        "[SX3](=[OX1])([#6])[#6]",
    "sulfone":          "[SX4](=[OX1])(=[OX1])([#6])[#6]",
    "sulfonyl":         "[SX4](=[OX1])(=[OX1])",
    "sulfonic acid":    "[SX4](=O)(=O)[OX2H]",
    # Halogens
    "halide":           "[F,Cl,Br,I]",
    "fluoride":         "[F]",
    "chloride":         "[Cl]",
    "bromide":          "[Br]",
    "iodide":           "[I]",
    "aryl halide":      "c[F,Cl,Br,I]",
    "aryl bromide":     "c[Br]",
    "aryl chloride":    "c[Cl]",
    "aryl iodide":      "c[I]",
    "aryl fluoride":    "c[F]",
    "alkyl halide":     "[CX4][F,Cl,Br,I]",
    "vinyl halide":     "C=C[F,Cl,Br,I]",
    # Boron
    "boronic acid":     "[B]([OX2H])[OX2H]",
    "boronate ester":   "[BX3]([OX2][#6])[OX2][#6]",
    "boronic ester":    "[BX3]([OX2][#6])[OX2][#6]",
    "organoboron":      "[#6][B]",
    "boron":            "[#5]",
    # Carbon
    "alkene":           "C=C",
    "alkyne":           "C#C",
    "terminal alkyne":  "[CX2H]#[CX2]",
    "aromatic":         "c1ccccc1",
    "arene":            "c1ccccc1",
    "phenyl":           "c1ccccc1",
    # Phosphorus
    "phosphine":        "[PX3]([#6])",
    "phosphate":        "[PX4](=O)([OX2])[OX2]",
    # Misc
    "trifluoromethyl":  "[CX4](F)(F)F",
    "tosyl":            "[SD4](=O)(=O)c1ccc([CH3])cc1",
    "mesyl":            "[SD4](=O)(=O)[CH3]",
    "triflate":         "[OX2]S(=O)(=O)C(F)(F)F",
}

# Compile SMARTS → only keep those that RDKit accepts
FG_MOLS: dict[str, "Chem.Mol"] = {}
FG_SMARTS_DICT: dict[str, str] = {}
for _name, _smarts in _FG_RAW.items():
    _p = Chem.MolFromSmarts(_smarts)
    if _p is not None:
        FG_MOLS[_name]        = _p
        FG_SMARTS_DICT[_name] = _smarts

# Sort by length (longest first) so multi-word names take priority in text scan
_FG_NAMES_SORTED: list[str] = sorted(FG_MOLS.keys(), key=len, reverse=True)


def _find_fgs_in_text(text: str) -> list[str]:
    """Return recognized FG names found in text (word-boundary match)."""
    text_lower = text.lower()
    found = []
    for name in _FG_NAMES_SORTED:
        if _re.search(r"\b" + _re.escape(name) + r"\b", text_lower):
            found.append(name)
    return found


def fg_grounding_score(fg_text: str, reactant_smiles_list: list[str]) -> int:
    """
    V2 / V3: Verify FG names in text actually exist in reactant SMILES.

    Returns 1 if:
      - at least 1 standard FG name is recognised in fg_text, AND
      - >= 50 % of recognised FGs are confirmed by RDKit substructure match
        against any reactant molecule.
    Returns 0 otherwise.
    """
    if not fg_text or not reactant_smiles_list:
        return 0

    reactant_mols = [Chem.MolFromSmiles(s) for s in reactant_smiles_list if s]
    reactant_mols = [m for m in reactant_mols if m is not None]
    if not reactant_mols:
        return 0

    recognised = _find_fgs_in_text(fg_text)
    if not recognised:
        return 0

    n_verified = sum(
        1 for name in recognised
        if any(mol.HasSubstructMatch(FG_MOLS[name]) for mol in reactant_mols)
    )
    return int(n_verified / len(recognised) >= 0.5)


# ---------------------------------------------------------------------------
# Mechanism keyword set  (V4)
# ---------------------------------------------------------------------------

MECHANISM_KEYWORDS: frozenset[str] = frozenset({
    # Organometallic catalytic cycle
    "oxidative addition", "reductive elimination", "transmetalation",
    "migratory insertion", "beta-hydride elimination", "beta hydride elimination",
    "ligand exchange", "ligand dissociation", "coordination",
    # Substitution / elimination
    "sn2", "sn1", "e2", "e1", "e1cb",
    "nucleophilic substitution", "electrophilic substitution",
    "nucleophilic attack", "electrophilic attack",
    "nucleophilic addition", "electrophilic addition",
    "electrophilic aromatic", "nucleophilic aromatic",
    "backside attack", "inversion", "retention",
    # Radical
    "radical", "homolysis", "homolytic", "hydrogen abstraction",
    "chain reaction", "chain mechanism",
    # Pericyclic
    "pericyclic", "cycloaddition", "diels-alder",
    "sigmatropic", "cope", "claisen", "ene reaction", "electrocyclic",
    # Acid / base
    "proton transfer", "deprotonation", "protonation",
    "acid-catalyzed", "base-catalyzed", "acid catalyzed", "base catalyzed",
    "acid-promoted", "base-promoted",
    # Redox / transfer
    "oxygen transfer", "hydride transfer", "electron transfer",
    "single electron", "oxidation", "reduction",
    "oxidative", "reductive",
    # Intermediates / species
    "carbocation", "carbanion", "carbene", "enolate", "enol",
    "zwitterion", "ylide", "radical anion", "radical cation",
    # General mechanistic terms (specific enough to not be filler)
    "concerted", "stepwise", "transition state",
    "rearrangement", "1,2-shift", "1,3-shift",
    "cyclization", "ring-opening", "ring opening",
    "condensation", "hydrolysis", "acylation", "alkylation", "arylation",
    "nucleophilic", "electrophilic",
    "metal-catalyzed", "palladium-catalyzed", "copper-catalyzed",
    "cross-coupling", "c-h activation",
    # Named reagent/mechanism eponyms
    "wittig", "reformatsky", "grignard", "buchwald-hartwig",
    # Protecting group
    "deprotection", "protection",
})


def mechanism_keyword_score(mech_text: str) -> int:
    """
    V4: Returns 1 if mech_text contains at least one recognised mechanism keyword.
    """
    if not mech_text:
        return 0
    text_lower = mech_text.lower()
    for kw in MECHANISM_KEYWORDS:
        if kw in text_lower:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Bond formation graph-diff validation  (V5)
# ---------------------------------------------------------------------------

# Synonyms for written-out bond names → element-pair notation
BOND_TEXT_SYNONYMS: dict[str, str] = {
    "carbon-carbon":              "C-C",
    "carbon-nitrogen":            "C-N",
    "carbon-oxygen":              "C-O",
    "carbon-sulfur":              "C-S",
    "carbon-phosphorus":          "C-P",
    "carbon-boron":               "C-B",
    "carbon-halogen":             "C-X",
    "nitrogen-oxygen":            "N-O",
    "sulfur-oxygen":              "S-O",
    "phosphorus-oxygen":          "P-O",
    "carbon carbon":              "C-C",
    "carbon nitrogen":            "C-N",
    "carbon oxygen":              "C-O",
    "carbon sulfur":              "C-S",
    "nitrogen oxygen":            "N-O",
    "sulfur oxygen":              "S-O",
    "amide bond":                 "C-N",
    "ester bond":                 "C-O",
    "ether bond":                 "C-O",
    "c-c bond":                   "C-C",
    "c-n bond":                   "C-N",
    "c-o bond":                   "C-O",
    "c-s bond":                   "C-S",
    "c=o bond":                   "C=O",
    "c=c bond":                   "C=C",
    "c=n bond":                   "C=N",
    "s=o bond":                   "S=O",
    "p=o bond":                   "P=O",
    "n-o bond":                   "N-O",
}

# Bond char → list of bond-type-as-double values to check
# Single bond "-" also checks aromatic (1.5) as a fallback
_BOND_CHAR_ORDERS: dict[str, list[float]] = {
    "-": [1.0, 1.5],
    "=": [2.0],
    "#": [3.0],
}


def _extract_bond_types_from_text(text: str) -> list[tuple[str, str, float]]:
    """
    Extract (atom1, atom2, bond_order_float) tuples from text.
    Normalises text synonyms first, then applies regex for X-Y / X=Y / X#Y.
    atom1 <= atom2 (alphabetically normalised).
    """
    t = text.lower()
    for phrase, notation in BOND_TEXT_SYNONYMS.items():
        t = t.replace(phrase, notation)

    # Match element symbols separated by bond chars (handles e.g. "C-N", "S=O")
    pattern = r'([A-Z][a-z]?)\s*([=\-#])\s*([A-Z][a-z]?)'
    results = []
    for a1, bond_char, a2 in _re.findall(pattern, t, _re.IGNORECASE):
        a1 = a1.capitalize()
        a2 = a2.capitalize()
        # Normalise pair alphabetically
        a1, a2 = min(a1, a2), max(a1, a2)
        results.append((a1, a2, bond_char))
    return list(set(results))


def _count_bond_types(smiles_list: list[str]) -> Counter:
    """
    Count (atom1, atom2, bond_order_float) tuples across all molecules.
    atom1 <= atom2 alphabetically.
    """
    counts: Counter = Counter()
    for smi in smiles_list:
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetSymbol()
            a2 = bond.GetEndAtom().GetSymbol()
            order = bond.GetBondTypeAsDouble()  # 1.0, 2.0, 3.0, or 1.5 (aromatic)
            a1, a2 = min(a1, a2), max(a1, a2)
            counts[(a1, a2, order)] += 1
    return counts


def bond_formed_score(bond_text: str,
                      reactant_smiles_list: list[str],
                      gt_product_smiles: str) -> int:
    """
    V5: Verify that the bond type stated in [BOND_FORMED] was actually
    newly formed in the GT product.

    Strategy: for each stated bond type, check if its count in the GT product
    is GREATER than the total count across all reactant molecules.
    Returns 1 if at least one stated bond passes this check; 0 otherwise.
    """
    if not bond_text or not gt_product_smiles or not reactant_smiles_list:
        return 0

    stated = _extract_bond_types_from_text(bond_text)
    if not stated:
        return 0

    reactant_counts = _count_bond_types(reactant_smiles_list)
    product_counts  = _count_bond_types([gt_product_smiles])

    for a1, a2, bond_char in stated:
        for order in _BOND_CHAR_ORDERS.get(bond_char, [1.0]):
            key = (a1, a2, order)
            if product_counts.get(key, 0) > reactant_counts.get(key, 0):
                return 1
    return 0


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
