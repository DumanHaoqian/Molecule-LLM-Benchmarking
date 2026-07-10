"""
Shared utilities: paths, data loading, sampling, yield class helpers.
V2: added ligand GT classification lookup (build_ligand_class_lookup).
"""
from pathlib import Path
import json
import random
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_PATH = PROJECT_ROOT / "dataset" / "rxn_predict" / "yield_pred_v2.json"
RESULTS_DIR  = PROJECT_ROOT / "results" / "cot_eval" / "rxn_predict" / "yield_pred_structured"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Yield class boundaries (from data inspection)
# ---------------------------------------------------------------------------
YIELD_LOW_MAX    = 16.04
YIELD_MEDIUM_MAX = 45.05


def yield_to_class(value: float) -> str:
    """Map a numeric yield to its class label."""
    if value <= YIELD_LOW_MAX:
        return "low"
    elif value <= YIELD_MEDIUM_MAX:
        return "medium"
    else:
        return "high"


def parse_gt(item: dict) -> tuple[float, str]:
    """Returns (gt_float, gt_class)."""
    return float(item["gt"]), item["yield_class"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Ligand GT classification lookup
# ---------------------------------------------------------------------------

# perera_suzuki_ord: human-readable ligand name → class
_PERERA_LIGAND_CLASS: dict[str, str] = {
    "":             "poor",
    "AmPhos":       "high",
    "CataCXium A":  "high",
    "P(Cy)3":       "standard",
    "P(Ph)3":       "standard",
    "P(o-Tol)3":    "standard",
    "P(tBu)3":      "high",
    "SPhos":        "high",
    "XPhos":        "high",
    "Xantphos":     "standard",
    "dppf":         "standard",
    "dtbpf":        "high",
}

# ahneman_cn_coupling: all 4 unique ligand SMILES are high-performance biaryl phosphines
_AHNEMAN_LIGAND_SMILES_CLASS: dict[str, str] = {}   # populated lazily by canonical key

# suzuki_hte_t5: catalyst SMILES (Pd + ligand) → class
_SUZUKI_HTE_CATALYST_CLASS: dict[str, str] = {}     # populated lazily

# buchwald_hte_t5: catalyst SMILES → class (both high)
_BUCHWALD_HTE_CATALYST_CLASS: dict[str, str] = {}   # populated lazily

_LOOKUP_BUILT: dict = {}          # {item_id: class}  — cached after first call
_LOOKUP_READY: list = [False]     # mutable sentinel


def _try_canonical(smiles: str) -> str:
    """Return RDKit canonical SMILES, or raw string on parse failure."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return smiles


def _classify_smiles_ligand(raw_smiles: str) -> str:
    """
    Classify a raw ligand/catalyst SMILES string as 'high'|'standard'|'poor'.
    Uses pattern heuristics for common Buchwald-Hartwig / Suzuki ligands.
    """
    s = raw_smiles.lower()

    # No ligand / Pd(OAc)2 only
    if not raw_smiles or raw_smiles.strip() in ("", "[pd]"):
        return "poor"
    if raw_smiles.strip() == "CC(=O)O.CC(=O)O.[Pd]":
        return "poor"

    # ----- HIGH-PERFORMANCE indicators -----

    # Buchwald palladacycle pre-catalysts: [Pd]1[c] (N→Pd cyclometallated core)
    # These are always high-performance; both buchwald_hte_t5 catalysts fall here.
    if "[pd]1[c]" in s or ("[o][pd]" in s and "[c]2ccccc" in s):
        return "high"

    # Biaryl phosphines: aryl carbon directly bonded to P atom (=C(P( or c(P( pattern)
    # Also catches 2,6-diisopropylphenyl backbone: cc(c)c...c=c1c(c)c)=c1
    has_aryl_p_bond = ("=c(p(" in s or "-c(p(" in s or "c2=c(p(" in s
                       or "c1=c(p(" in s)
    # Or: biaryl backbone + ortho-PCy2 (RuPhos variant)
    has_biaryl_phosphine = (
        has_aryl_p_bond
        or ("p(c(c)(c)c)" in s or "p(c(c)(c)c)c(c)(c)c" in s)    # PtBu groups
        or ("p(c1ccccc1)c(c)(c)c" in s)
        or ("-c2ccccc2p(" in s or "c2ccccc2p(" in s)
        or ("adamant" in s
            or "c12cc3cc(cc(c3)c1)c2" in s
            or "c34cc5cc(cc(c5)c3)cc4" in s)
        or ("c1ccc(p(c(c)(c)c)c(c)(c)c)cc1" in s)
        or ("cn(c)c1ccc(p(" in s)                                  # AmPhos-like
    )
    # Ferrocene-based high-perf (dtbpf)
    has_dtbpf = ("[fe+2]" in s or "[fe]" in s) and "c(c)(c)c" in s

    if has_biaryl_phosphine or has_dtbpf:
        return "high"

    # PtBu3 (no biaryl): P(C(C)(C)C)...
    if "p(c(c)(c)c)c(c)(c)c" in s and "c1ccccc1" not in s:
        return "high"

    # ----- STANDARD indicators -----
    # PCy3: P(C2CCCCC2)C2CCCCC2
    if "p(c1ccccc1)c1ccccc1" in s or "p(c2ccccc2)c2ccccc2" in s:
        # could be PPh3 or P(o-Tol)3 — both standard
        return "standard"
    # PCy3 (cyclohexyl): P(C2CCCCC2)
    if "p(c1ccccc1)" not in s and ("ccccc" in s or "cccc" in s) and "[pd]" in s:
        # Xantphos: has xanthene backbone cc1(c)c2cccc(p...)c2oc2c(...
        if "oc2c" in s or "c1(c)c" in s:
            return "standard"
    # dppf: [Fe] + Ph2P groups
    if ("[fe+2]" in s or "[fe]" in s) and "p(c1ccccc1)" in s:
        return "standard"
    # Simple cyclohexyl phosphine: C1CCC(P(...)...)CC1
    if "c1ccc(p(" in s and "cc1" in s:
        return "standard"

    # fallback: if Pd is present but no identifiable high-perf → standard
    if "[pd]" in s:
        return "standard"

    return "poor"


def build_ligand_class_lookup(data: list[dict] | None = None) -> dict[str, str]:
    """
    Returns {item_id → gt_ligand_class} for all items in yield_pred_v2.json.
    Result is cached after first call.
    """
    if _LOOKUP_READY[0]:
        return _LOOKUP_BUILT

    if data is None:
        data = load_raw_data()

    for item in data:
        item_id = item["id"]
        src     = item["meta"].get("source_dataset", "")
        cond    = item["meta"].get("conditions", {})
        ligand  = cond.get("ligand", cond.get("catalyst", ""))

        if src == "perera_suzuki_ord":
            cls = _PERERA_LIGAND_CLASS.get(ligand, "poor")
        else:
            # SMILES-based: ahneman, suzuki_hte_t5, buchwald_hte_t5
            # All ahneman ligands are high-performance biaryl phosphines.
            if src == "ahneman_cn_coupling":
                # 4 known ligands — all high-performance biaryl phosphines
                cls = "high"
            else:
                # suzuki_hte_t5 / buchwald_hte_t5: classify from SMILES
                cls = _classify_smiles_ligand(ligand)

        _LOOKUP_BUILT[item_id] = cls

    _LOOKUP_READY[0] = True
    return _LOOKUP_BUILT


def extract_ligand_class_from_score_field(text: str) -> Optional[str]:
    """
    Parse model's [LIGAND_SYSTEM_SCORE] field text into 'high'|'standard'|'poor'.
    Returns None if the text is missing or unrecognisable.
    """
    if not text:
        return None
    t = text.lower()
    if "high" in t or "high-perf" in t or "high perf" in t:
        return "high"
    if "poor" in t:
        return "poor"
    if "standard" in t or "moderate" in t or "medium" in t:
        return "standard"
    return None


def select_sample(data: list[dict], n: int = 60, seed: int = 42) -> list[dict]:
    """
    Stratified sample: 20 each from easy/medium/hard.
    No rxn_cls available → random shuffle within each stratum.
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
        random.shuffle(pool)
        picked = pool[:per_difficulty]
        sampled.extend(picked)

    random.shuffle(sampled)
    return sampled[:n]


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
