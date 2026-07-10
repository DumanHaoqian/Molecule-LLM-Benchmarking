"""
Shared utilities for mol_edit structured CoT evaluation.

Provides:
  - Path constants (DATASET_DIR, COT_DATA_DIR, RESULTS_DIR)
  - Data loading: load_raw_data (mini-dataset), load_step1_data (60-item eval set)
  - Gold template hint building: build_system_prompt(edit_type)
  - RDKit helpers: canonical_smiles, smiles_valid, fts, smiles_match_exact
  - Verification helpers: v1_target_group_present, v2_atom_ids_valid, v3_transformation_check, v4_ring_count_check
"""
from pathlib import Path
import json
import re
import random
from typing import Optional

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdFMCS
from rdkit.DataStructs import ExplicitBitVect

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR  = PROJECT_ROOT / "dataset" / "mol_edit" / "1-instruct_to_edit"
COT_DATA_DIR = PROJECT_ROOT / "cot_data" / "mol_edit"
RESULTS_DIR  = PROJECT_ROOT / "results" / "cot_eval" / "mol_edit" / "mol_edit_structured"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EDIT_TYPES = ["add", "delete", "substitute"]

# ---------------------------------------------------------------------------
# Gold template reasoning hints (condensed, per edit_type)
# ---------------------------------------------------------------------------

_GOLD_HINTS: dict[str, str] = {
    "add": "ADD a new group to the source molecule; no atoms are removed.",
    "delete": "DELETE (remove) a group from the source molecule; no new atoms are added.",
    "substitute": "SUBSTITUTE (replace) one group in the source molecule with another.",
}

# ---------------------------------------------------------------------------
# Per-edit-type system prompts (section names aligned with A→B formal CoT)
# ---------------------------------------------------------------------------

_ADD_PROMPT_TEMPLATE = """\
You are an expert computational chemist. {hint}

Output EXACTLY the sections below. Do NOT write any introductory text, analysis, or reasoning paragraphs. Output ONLY the section headers and field values.

[ANCHOR_IDENTIFICATION]
Target Atom: <element symbol, e.g. "C", "N", "O">
Attachment Index: <1-based index>
Free Valence: <e.g. "Yes, 1 implicit H">

[FRAGMENT_IDENTIFICATION]
Fragment: <group to attach, e.g. "methyl (-CH₃)">

[PRODUCT_CONSTRUCTION]
Action: <e.g. "Form bond between atom {{k}} and fragment">
Product SMILES: <SMILES>
Answer: <same SMILES>

[HEAVY_ATOM_VERIFICATION]
Source Heavy Atoms: <count>
Product Heavy Atoms: <count>
Delta: <product − source>

[RING_VERIFICATION]
Source Rings: <count>
Product Rings: <count>
Check: <e.g. "Ring count: 2 → 2 (unchanged)">

Rules:
- ONLY the stated edit; do not modify any other part of the molecule.
- Product SMILES must equal Answer.
- Each field is ONE line (≤ 20 words). No paragraphs.
- No text before [ANCHOR_IDENTIFICATION] and no text after [RING_VERIFICATION].
"""

_DELETE_PROMPT_TEMPLATE = """\
You are an expert computational chemist. {hint}

Output EXACTLY the sections below. Do NOT write any introductory text, analysis, or reasoning paragraphs. Output ONLY the section headers and field values.

[ANCHOR_IDENTIFICATION]
Leaving Group: <e.g. "Boc protecting group">
Attachment Atom Index: <1-based index of atom that stays>

[GROUP_SIZE_VERIFICATION]
Group Atoms: <indices, e.g. [2, 3, 4]>
Group Heavy Atoms: <count>

[PRODUCT_CONSTRUCTION]
Action: <e.g. "Break atom {{k}}–{{leaving group}}">
Product SMILES: <SMILES>
Answer: <same SMILES>

[HEAVY_ATOM_VERIFICATION]
Source Heavy Atoms: <count>
Product Heavy Atoms: <count>
Delta: <product − source>

[RING_VERIFICATION]
Source Rings: <count>
Product Rings: <count>
Check: <e.g. "Ring count: 2 → 2 (unchanged)">

Rules:
- ONLY the stated edit; do not modify any other part of the molecule.
- Product SMILES must equal Answer.
- Each field is ONE line (≤ 20 words). No paragraphs.
- No text before [ANCHOR_IDENTIFICATION] and no text after [RING_VERIFICATION].
"""

_SUBSTITUTE_PROMPT_TEMPLATE = """\
You are an expert computational chemist. {hint}

Output EXACTLY the sections below. Do NOT write any introductory text, analysis, or reasoning paragraphs. Output ONLY the section headers and field values.

[ANCHOR_IDENTIFICATION]
Attachment Atom: <element symbol, e.g. "C", "O">
Attachment Index: <1-based index>

[REMOVE_GROUP_SIZE]
Leaving Group: <e.g. "hydroxyl (-OH)">
Leaving Group Indices: <e.g. [2, 3]>

[ADD_FRAGMENT_SIZE]
Incoming Group: <e.g. "fluorine (-F)">
Incoming Fragment SMILES: <SMILES>

[PRODUCT_CONSTRUCTION]
Action: <e.g. "Break atom {{k}}–X; Form atom {{k}}–Y">
Product SMILES: <SMILES>
Answer: <same SMILES>

[HEAVY_ATOM_VERIFICATION]
Source Heavy Atoms: <count>
Product Heavy Atoms: <count>
Delta: <product − source>

[RING_VERIFICATION]
Source Rings: <count>
Product Rings: <count>
Check: <e.g. "Ring count: 2 → 2 (unchanged)">

Rules:
- ONLY the stated edit; do not modify any other part of the molecule.
- Product SMILES must equal Answer.
- Each field is ONE line (≤ 20 words). No paragraphs.
- No text before [ANCHOR_IDENTIFICATION] and no text after [RING_VERIFICATION].
"""

_SYSTEM_PROMPT_DIRECT = """\
You are an expert computational chemist. Apply a specified molecular edit to produce a new molecule.

Output ONLY the product SMILES. Nothing else.

Answer: <SMILES>
"""

_SYSTEM_PROMPT_TEMPLATES: dict[str, str] = {
    "add": _ADD_PROMPT_TEMPLATE,
    "delete": _DELETE_PROMPT_TEMPLATE,
    "substitute": _SUBSTITUTE_PROMPT_TEMPLATE,
}


def build_system_prompt(edit_type: Optional[str] = None) -> str:
    """Return the structured CoT system prompt for a given edit_type (or generic if None)."""
    et = edit_type or ""
    hint = _GOLD_HINTS.get(et, "")
    tmpl = _SYSTEM_PROMPT_TEMPLATES.get(et, _ADD_PROMPT_TEMPLATE)
    return tmpl.format(hint=hint).strip()


def get_indexed_smiles(smiles: str) -> str:
    """Return RDKit canonical SMILES with 1-based atom map numbers for site anchoring.

    Atom map number k in the output corresponds to 0-based canonical index k-1.
    Example: CC[OH] → [CH3:1][CH2:2][OH:3]
    Returns the original SMILES string unchanged if it is unparseable.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    # RDKit silently drops map-num 0 in SMILES output; use 1-based map numbers.
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return Chem.MolToSmiles(mol)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data(edit_type: Optional[str] = None) -> list[dict]:
    """
    Load mol_edit mini-dataset (17 items total).
    edit_type: "add" / "delete" / "substitute" / None (load all combined).
    Fields: molecule, Instruction, reference, edit_type, difficulty, ...
    """
    types_to_load = [edit_type] if edit_type is not None else EDIT_TYPES
    data = []
    for t in types_to_load:
        path = DATASET_DIR / f"{t}_v2.json"
        with open(path) as f:
            items = json.load(f)
        for item in items:
            item["edit_type"] = t
        data.extend(items)
    return data


def load_step1_data(edit_type: Optional[str] = None) -> list[dict]:
    """
    Load the 60-item step1 CoT eval dataset (from gold template generation).
    edit_type: "add" / "delete" / "substitute" / None (load all combined).

    Normalises fields to the common schema used by the samplers:
      molecule   ← src
      Instruction ← instruction
      reference   ← tgt
      difficulty  ← derived from mol_complexity (tertile split per edit_type)
    """
    types_to_load = [edit_type] if edit_type is not None else EDIT_TYPES
    data = []
    for t in types_to_load:
        path = COT_DATA_DIR / f"step1_cot_{t}.json"
        with open(path) as f:
            items = json.load(f)
        # Derive difficulty thresholds from mol_complexity tertiles
        vals = sorted(x.get("mol_complexity", 0) for x in items)
        n = len(vals)
        t33 = vals[max(0, n // 3 - 1)]
        t67 = vals[min(n - 1, 2 * n // 3)]
        for item in items:
            c = item.get("mol_complexity", 0)
            item["difficulty"] = "easy" if c <= t33 else ("medium" if c <= t67 else "hard")
            item["molecule"]    = item["src"]
            item["Instruction"] = item["instruction"]
            item["reference"]   = item["tgt"]
            item.setdefault("edit_type", t)
        data.extend(items)
    return data


def select_sample(data: list[dict], n: int = 999, seed: int = 42) -> list[dict]:
    """
    Return up to n items, stratified by edit_type × difficulty.
    If n >= len(data), return all items (shuffled).
    """
    random.seed(seed)
    if n >= len(data):
        items = list(data)
        random.shuffle(items)
        return items

    # Group by (edit_type, difficulty) and allocate proportionally
    groups: dict[tuple, list] = {}
    for item in data:
        key = (item.get("edit_type", "?"), item.get("difficulty", "unknown"))
        groups.setdefault(key, []).append(item)

    total = len(data)
    sampled: list[dict] = []
    for key, pool in groups.items():
        k = max(1, round(n * len(pool) / total))
        picked = random.sample(pool, min(k, len(pool)))
        sampled.extend(picked)

    random.shuffle(sampled)
    return sampled[:n]


# ---------------------------------------------------------------------------
# RDKit helpers
# ---------------------------------------------------------------------------

def canonical_smiles(smiles: str) -> Optional[str]:
    """Return RDKit canonical SMILES, or None if invalid."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def smiles_valid(smiles: str) -> bool:
    """True if SMILES is parseable by RDKit."""
    return canonical_smiles(smiles) is not None


def largest_fragment_smiles(smiles: str) -> Optional[str]:
    """Return canonical SMILES of the largest fragment (by heavy atom count).
    For single-fragment SMILES this is equivalent to canonical_smiles().
    Handles GT products that include byproducts / counterions (e.g. 'CC.Cl').
    """
    if not smiles:
        return None
    best_mol, best_n = None, -1
    for frag in smiles.split("."):
        frag = frag.strip()
        if not frag:
            continue
        mol = Chem.MolFromSmiles(frag)
        if mol is None:
            continue
        n = mol.GetNumHeavyAtoms()
        if n > best_n:
            best_mol, best_n = mol, n
    if best_mol is None:
        return None
    return Chem.MolToSmiles(best_mol)


def smiles_match_main_frag(pred: str, gt: str) -> bool:
    """
    Compare the largest fragment of pred against the largest fragment of gt.

    This is the primary correctness check for mol_edit: the model is asked to
    produce the *main* product; byproducts / counterions in the GT (e.g. '.Cl'
    from HCl elimination) should not penalise an otherwise correct answer.
    """
    c_pred = largest_fragment_smiles(pred)
    c_gt   = largest_fragment_smiles(gt)
    if c_pred is None or c_gt is None:
        return False
    return c_pred == c_gt


def smiles_match_exact(pred: str, gt: str) -> bool:
    """
    Canonical SMILES equality check.
    Handles equivalent representations of the same molecule.
    """
    c_pred = canonical_smiles(pred)
    c_gt   = canonical_smiles(gt)
    if c_pred is None or c_gt is None:
        return False
    return c_pred == c_gt


def _union_morgan_fp(smiles: str) -> Optional[ExplicitBitVect]:
    """Union Morgan fingerprint over all dot-split fragments (handles salts/mixtures)."""
    fp = None
    for frag in smiles.split("."):
        frag = frag.strip()
        if not frag:
            continue
        mol = Chem.MolFromSmiles(frag)
        if mol is None:
            continue
        frag_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        if fp is None:
            fp = ExplicitBitVect(2048)
        fp |= frag_fp
    return fp


def fts(pred: str, gt: str) -> float:
    """Fingerprint Tanimoto Similarity (Morgan r=2, 2048 bits). Returns 0.0 on failure."""
    if not pred or not gt:
        return 0.0
    fp_pred = _union_morgan_fp(pred)
    fp_gt   = _union_morgan_fp(gt)
    if fp_pred is None or fp_gt is None:
        return 0.0
    try:
        return DataStructs.TanimotoSimilarity(fp_pred, fp_gt)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Ordered specific → general.  Only patterns specific enough to be informative.
_SITE_KEYWORD_SMARTS: list[tuple[list[str], str]] = [
    # ── Protecting groups ─────────────────────────────────────────────────────
    (["boc", "tert-butoxycarbonyl", "tert-butyloxycarbonyl", "t-boc"],
     "NC(=O)OC(C)(C)C"),
    (["cbz", "carboxybenzyl", "benzyloxycarbonyl"],
     "NC(=O)OCc1ccccc1"),
    (["pmb", "para-methoxybenzyl", "4-methoxybenzyl"],
     "OCc1ccc(OC)cc1"),
    (["weinreb amide", "weinreb", "n(ome)me", "n-methoxy-n-methyl"],
     "C(=O)N(OC)C"),
    (["tips", "triisopropylsilyl"],
     "O[Si](C(C)C)(C(C)C)C(C)C"),
    (["tbs", "tbdms", "tert-butyldimethylsilyl", "tert-butyl dimethylsilyl"],
     "O[Si](C)(C)C(C)(C)C"),
    (["tms", "trimethylsilyl"],
     "O[Si](C)(C)C"),
    (["silyl", "o–si", "o-si"],
     "[Si]"),
    # ── Halogens ─────────────────────────────────────────────────────────────
    (["chloro", "chloride", "chlorine"],
     "[Cl]"),
    (["bromo", "bromide", "bromine"],
     "[Br]"),
    (["fluoro", "fluoride", "fluorine"],
     "[F]"),
    (["trifluoromethyl", "cf3", "cf₃"],
     "C(F)(F)F"),
    (["iodo", "iodide", "iodine"],
     "[I]"),
    # ── Esters (specific before generic) ─────────────────────────────────────
    (["hexafluoropropyl ester", "hexafluoro"],
     "OC(C(F)(F)F)C(F)(F)F"),
    (["tert-butyl ester", "t-butyl ester"],
     "C(=O)OC(C)(C)C"),
    (["benzyl ester"],
     "C(=O)OCc1ccccc1"),
    (["methyl ester", "–cooch₃", "–cooch3", "cooch3"],
     "C(=O)OC"),
    (["ethyl ester", "–cooet", "cooet"],
     "C(=O)OCC"),
    (["propyl ester", "–occc"],
     "C(=O)OCCC"),
    (["ester"],
     "C(=O)OC"),
    # ── Other functional groups ───────────────────────────────────────────────
    (["carboxylic acid", "cooh", "–cooh", "carboxyl"],
     "C(=O)O"),
    (["amide"],
     "C(=O)N"),
    (["benzyl ether", "benzyl group attached to ether", "och₂ph", "och2ph",
      "–occ3ccccc3", "–och₂c6h5", "–och2c6h5"],
     "OCc1ccccc1"),
    (["hydroxymethyl", "–ch2oh", "–ch₂oh", "ch2oh"],
     "[CH2]O"),
    (["hydroxyl", "–oh", "hydroxy"],
     "[OH]"),
    (["aldehyde", "–cho", "formyl"],
     "[CH]=O"),
    (["nitro group", "nitro-", "–no2", "–no₂", "-no2"],
     "[N+](=O)[O-]"),
    (["sulfonyl", "sulfone", "–so2", "sulfonamide"],
     "S(=O)(=O)"),
    (["sulfoxide"],
     "[S]=O"),
    (["alkyne", "triple bond", "ethynyl", "–c≡c", "–c#c"],
     "C#C"),
    (["epoxide", "oxirane"],
     "C1OC1"),
    (["methoxy", "–ome", "methyl ether"],
     "OC"),
    (["primary amine", "–nh₂", "–nh2", "nh₂"],
     "[NH2]"),
    (["secondary amine", "secondary nitrogen"],
     "[NH]"),
    (["indole nh", "indole n–h", "benzimidazole n–h", "benzimidazole nh",
      "imidazole n–h", "imidazole nh"],
     "[nH]"),
    (["lactam", "lactam nh", "lactam nitrogen"],
     "[n,N;H1]"),
    (["urea"],
     "NC(=O)N"),
    (["carbamate"],
     "NC(=O)O"),
    # ── Heterocyclic rings ────────────────────────────────────────────────────
    (["piperidine", "piperidinyl"],
     "N1CCCCC1"),
    (["piperazine", "piperazinyl"],
     "N1CCNCC1"),
    (["morpholine", "morpholinyl"],
     "N1CCOCC1"),
    (["pyrrolidine", "pyrrolidinyl"],
     "N1CCCC1"),
    (["azetidine", "azetidinyl"],
     "N1CCC1"),
    (["imidazole", "imidazolyl"],
     "c1cnc[nH]1"),
    (["pyridine", "pyridyl"],
     "c1ccncc1"),
    (["thiophene", "thienyl"],
     "c1ccsc1"),
    (["furan", "furanyl"],
     "c1ccoc1"),
    (["indole", "indolyl"],
     "c1ccc2[nH]ccc2c1"),
]


# V1–V4 helpers: new step-ordered verification points
# ---------------------------------------------------------------------------

def _fg_count_in_mol(text: Optional[str], smiles: Optional[str]) -> Optional[int]:
    """
    Count occurrences of the functional group described by *text* in *smiles*.

    Returns:
        int   – match count (0 = absent, ≥1 = present) if a keyword was recognised
        None  – text is blank / "None" / unrecognisable, or smiles is invalid

    Uses _SITE_KEYWORD_SMARTS for keyword → SMARTS look-up.
    """
    if not text or not smiles:
        return None
    text_lower = text.lower()
    # "None" / "none" / "N/A" → no functional group specified
    if re.search(r'\bnone\b', text_lower) or re.search(r'\bn/a\b', text_lower):
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    for keywords, smarts_str in _SITE_KEYWORD_SMARTS:
        if any(kw in text_lower for kw in keywords):
            try:
                pat = Chem.MolFromSmarts(smarts_str)
                if pat is not None:
                    return len(mol.GetSubstructMatches(pat))
            except Exception:
                pass
    return None  # no keyword recognised


def _parse_attachment_point(ap_text: Optional[str]) -> Optional[int]:
    """Parse a 1-based integer index from an Attachment Point field string."""
    if not ap_text:
        return None
    m = re.search(r'\d+', str(ap_text))
    return int(m.group()) if m else None


# Priority: longer / 2-letter symbols first to avoid partial matches
_ELEMENT_WORDS: list[tuple[list[str], str]] = [
    (["chlorine", "chlorin", "cl"],                          "Cl"),
    (["bromine",  "bromin",  "br"],                          "Br"),
    (["silicon",  "si"],                                     "Si"),
    (["selenium", "se"],                                     "Se"),
    (["phosphorus", "phosphor", " p ", "^p$"],               "P"),
    (["sulfur", "sulphur", "thio", " s ", "^s$"],            "S"),
    (["fluorine", "fluor", " f ", "^f$"],                    "F"),
    (["iodine",  "iod", " i ", "^i$"],                       "I"),
    (["nitrogen", "amine", "amino", "amide", " n ", "^n$"],  "N"),
    (["oxygen",  "hydroxyl", " o ", "^o$"],                  "O"),
    (["carbon",  " c ", "^c$"],                              "C"),
]


def _parse_element_from_text(text: str) -> Optional[str]:
    """
    Parse an element symbol from a Target Group field for add tasks.
    Handles bare symbols ('C', 'Cl'), word forms ('nitrogen', 'aromatic carbon'),
    and hybrid descriptions ('sp3 N', 'aromatic C').
    """
    if not text:
        return None
    t = text.strip()

    # 1. Direct 2-letter symbol (Cl, Br, Si, Se)
    m = re.match(r'^(Cl|Br|Si|Se|Te)(\W|$)', t, re.IGNORECASE)
    if m:
        return m.group(1).capitalize() if len(m.group(1)) == 2 else m.group(1).upper()

    # 2. Direct 1-letter symbol as full token
    m = re.match(r'^([CNOSPFI])(\W|$)', t)
    if m:
        return m.group(1).upper()

    # 3. Word / keyword matching (case-insensitive)
    t_lower = t.lower()
    for keywords, symbol in _ELEMENT_WORDS:
        for kw in keywords:
            if re.search(rf'(^|\s){re.escape(kw)}(\s|$)', t_lower):
                return symbol

    # 4. Inline symbol: "sp3 C", "aromatic N", "atom C", etc.
    m = re.search(r'\b(Cl|Br|Si|Se|[CNOSPFI])\b', t)
    if m:
        sym = m.group(1)
        return sym if sym in ("Cl", "Br", "Si", "Se") else sym.upper()

    return None


def v1_target_group_present(target_group: Optional[str],
                             source_smiles: Optional[str],
                             edit_type: Optional[str] = None,
                             attachment_point: Optional[str] = None) -> int:
    """
    V1 – Step 1 [ANCHOR_IDENTIFICATION]: Verify the stated Target Group / attachment site.

    • delete / substitute: Target Group (leaving group) must appear in source mol via SMARTS.
    • add: Target Group = element symbol of the attachment atom.
        Verify stated element matches the actual element of the atom at Attachment Point.
    • Fallback (old template / non-compliance, Target Group = "None"):
        Free-valence check — atom at Attachment Point must have ≥1 implicit H.

    Returns 0 only when a meaningful, verifiable check fails. Defaults to 1 on parse errors
    or unrecognisable groups (benefit of the doubt).
    """
    if not target_group or not source_smiles:
        return 1

    if edit_type == "add" or re.search(r'\bnone\b', target_group, re.IGNORECASE):
        if re.search(r'\bnone\b', target_group, re.IGNORECASE):
            # Old format / non-compliance → free-valence fallback
            ap_idx = _parse_attachment_point(attachment_point)
            if ap_idx is None:
                return 1
            src_mol = Chem.MolFromSmiles(source_smiles.strip())
            if src_mol is None:
                return 1
            if not (1 <= ap_idx <= src_mol.GetNumHeavyAtoms()):
                return 0
            return int(src_mol.GetAtomWithIdx(ap_idx - 1).GetTotalNumHs() > 0)

        if edit_type == "add":
            # New template: stated element must match actual atom at Attachment Point
            stated_el = _parse_element_from_text(target_group)
            if stated_el is None:
                return 1  # unrecognisable element → benefit of the doubt
            ap_idx = _parse_attachment_point(attachment_point)
            if ap_idx is None:
                return 1
            src_mol = Chem.MolFromSmiles(source_smiles.strip())
            if src_mol is None:
                return 1
            if not (1 <= ap_idx <= src_mol.GetNumHeavyAtoms()):
                return 0
            actual_el = src_mol.GetAtomWithIdx(ap_idx - 1).GetSymbol()
            return int(actual_el == stated_el)

    # delete / substitute: SMARTS presence check
    count = _fg_count_in_mol(target_group, source_smiles)
    if count is None:
        return 1  # unrecognisable → benefit of the doubt
    return int(count > 0)


def v2_atom_ids_valid(atom_ids_text: Optional[str],
                      source_smiles: Optional[str],
                      attachment_point: Optional[str] = None) -> int:
    """
    V2 – Step 2 [FRAGMENT_IDENTIFICATION / GROUP_SIZE_VERIFICATION / REMOVE_GROUP_SIZE]:
    Are the stated atom indices within the molecule?

    • delete / substitute: validates Target Atom IDs (1-based) against heavy-atom count.
    • add: Target Atom IDs = [None] (no atoms removed). Instead, validates that
      Attachment Point is an integer in [1, n_heavy_atoms].

    Returns 0 when a verifiable index is out of range; returns 1 otherwise.
    """
    if not atom_ids_text or not source_smiles:
        return 1
    text_lower = atom_ids_text.lower()
    if re.search(r'\bnone\b', text_lower) or re.search(r'\bn/a\b', text_lower):
        # add subtask: no leaving-group atoms; check Attachment Point range instead
        ap_idx = _parse_attachment_point(attachment_point)
        if ap_idx is None:
            return 1  # attachment_point missing → can't verify
        src_mol = Chem.MolFromSmiles(source_smiles.strip())
        if src_mol is None:
            return 1
        n_atoms = src_mol.GetNumHeavyAtoms()
        return int(1 <= ap_idx <= n_atoms)
    src_mol = Chem.MolFromSmiles(source_smiles.strip())
    if src_mol is None:
        return 1
    n_atoms = src_mol.GetNumHeavyAtoms()
    ints = [int(x) for x in re.findall(r'\d+', atom_ids_text)]
    if not ints:
        return 1  # no integers found → can't verify
    return int(all(1 <= idx <= n_atoms for idx in ints))


def v3_transformation_check(target_group: Optional[str],
                             replacement: Optional[str],
                             source_smiles: Optional[str],
                             pred_smiles: Optional[str],
                             edit_type: str) -> int:
    """
    V3 – Step 3 [PRODUCT_CONSTRUCTION]: Does the product reflect the correct transformation?

    delete:     target group count must DECREASE from source to product.
    add:        replacement count must INCREASE (or appear) in product vs. source.
    substitute: target count decreases AND/OR replacement count increases.

    Returns 1 (pass) or 0 (fail).
    Returns 1 (benefit of the doubt) when both groups are unrecognisable via SMARTS,
    or when the predicted SMILES is invalid.
    """
    if not pred_smiles or not source_smiles:
        return 0
    if not smiles_valid(pred_smiles):
        return 0

    src_target  = _fg_count_in_mol(target_group,  source_smiles)
    pred_target = _fg_count_in_mol(target_group,  pred_smiles)
    src_repl    = _fg_count_in_mol(replacement,   source_smiles)
    pred_repl   = _fg_count_in_mol(replacement,   pred_smiles)

    checks: list[bool] = []

    if edit_type == "delete":
        if src_target is not None and pred_target is not None:
            checks.append(pred_target < src_target)
    elif edit_type == "add":
        if pred_repl is not None and src_repl is not None:
            checks.append(pred_repl > src_repl)
        elif pred_repl is not None:
            checks.append(pred_repl > 0)
    elif edit_type == "substitute":
        if src_target is not None and pred_target is not None:
            checks.append(pred_target < src_target)
        if pred_repl is not None and src_repl is not None:
            checks.append(pred_repl > src_repl)
        elif pred_repl is not None:
            checks.append(pred_repl > 0)

    if not checks:
        return 1  # can't verify either group → benefit of the doubt
    return int(all(checks))


def v4_ring_count_check(check_field: Optional[str],
                        source_smiles: Optional[str],
                        pred_smiles: Optional[str]) -> int:
    """
    V4 – Step 5 [RING_VERIFICATION]: Is the stated ring count claim consistent?

    Recognises patterns such as:
      "Ring count: 2 → 2 (unchanged)"   → expects src=2 and pred=2
      "Ring count unchanged"             → expects src == pred
      "Ring count remains 2"             → expects pred=2
      "Ring count: 1 → 0"               → expects src=1 and pred=0

    Returns 1 (consistent) or 0 (inconsistent).
    Returns 1 (benefit of the doubt) when the pattern is unrecognisable or SMILES invalid.
    """
    if not check_field or not source_smiles or not pred_smiles:
        return 1
    src_mol  = Chem.MolFromSmiles(source_smiles.strip())
    pred_mol = Chem.MolFromSmiles(pred_smiles.strip())
    if src_mol is None or pred_mol is None:
        return 1

    n_src  = src_mol.GetRingInfo().NumRings()
    n_pred = pred_mol.GetRingInfo().NumRings()

    check_lower = check_field.lower()

    if "unchanged" in check_lower:
        return int(n_src == n_pred)

    m = re.search(r'remains\s+(\d+)', check_lower)
    if m:
        return int(n_pred == int(m.group(1)))

    # "X → Y" or "X -> Y" or "X to Y"
    m = re.search(r'(\d+)\s*(?:→|->|to)\s*(\d+)', check_field, re.IGNORECASE)
    if m:
        claimed_src  = int(m.group(1))
        claimed_pred = int(m.group(2))
        return int(n_src == claimed_src and n_pred == claimed_pred)

    # Single number: "Ring count: 2" — ambiguous (src or pred?); accept either
    m = re.search(r'ring\s*count\s*[:\s]+(\d+)', check_lower)
    if m:
        claimed = int(m.group(1))
        return int(n_pred == claimed or n_src == claimed)

    return 1  # unrecognisable → benefit of the doubt


# ---------------------------------------------------------------------------
# Supplementary: MCS-based edit site correctness
# ---------------------------------------------------------------------------

def compute_edit_site(src_smiles: str,
                      prod_smiles: str,
                      edit_type: str) -> Optional[frozenset]:
    """
    Identify the edit site as a frozenset of 1-indexed atom map numbers in src.

    Internally assigns atom map numbers 1..N to src atoms; map num 0 means
    "no mapping" in RDKit and is avoided.

    For 'delete'/'substitute':
        Returns src atom map nums NOT covered by MCS(src, prod) — the
        removed / replaced atoms.

    For 'add':
        Returns src atom map nums that serve as attachment points — src atoms
        that gained at least one new neighbour in prod.

    Returns None if the computation fails (invalid SMILES, MCS timeout,
    no change detected, or unsupported edit_type).
    """
    src_mol  = Chem.MolFromSmiles(src_smiles)
    prod_mol = Chem.MolFromSmiles(prod_smiles)
    if src_mol is None or prod_mol is None:
        return None

    # Assign 1-indexed atom maps to src (RDKit map 0 = no map)
    for atom in src_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    try:
        mcs = rdFMCS.FindMCS(
            [src_mol, prod_mol],
            timeout=10,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            ringMatchesRingOnly=False,
            completeRingsOnly=False,
        )
    except Exception:
        return None

    if mcs.numAtoms == 0:
        return None

    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    if mcs_mol is None:
        return None

    src_match = src_mol.GetSubstructMatch(mcs_mol)
    if not src_match:
        return None

    if edit_type in ("delete", "substitute"):
        mcs_src_idx = set(src_match)
        edited_idx  = set(range(src_mol.GetNumAtoms())) - mcs_src_idx
        if edited_idx:
            return frozenset(src_mol.GetAtomWithIdx(i).GetAtomMapNum() for i in edited_idx)
        # Fallthrough: substitution added atoms without removing any (e.g. COOH→COOMe).
        # Treat as add: find the src attachment point that gained a new neighbour in tgt.
        edit_type = "add"  # local rebind; handled by the add branch below

    if edit_type == "add":
        prod_match = prod_mol.GetSubstructMatch(mcs_mol)
        if not prod_match:
            return None
        prod_match_set  = set(prod_match)
        new_in_prod     = set(range(prod_mol.GetNumAtoms())) - prod_match_set
        if not new_in_prod:
            return None  # nothing added

        prod_match_list = list(prod_match)
        attach_map_nums: set[int] = set()
        for new_idx in new_in_prod:
            for bond in prod_mol.GetAtomWithIdx(new_idx).GetBonds():
                nb = bond.GetOtherAtomIdx(new_idx)
                if nb in prod_match_set:
                    pm_pos  = prod_match_list.index(nb)
                    src_pos = src_match[pm_pos]
                    attach_map_nums.add(src_mol.GetAtomWithIdx(src_pos).GetAtomMapNum())
        return frozenset(attach_map_nums) if attach_map_nums else None

    return None


def edit_site_overlap(gt_site: Optional[frozenset],
                      pred_site: Optional[frozenset]) -> float:
    """
    Recall-based site overlap: |gt ∩ pred| / |gt|.

    Ranges 0.0–1.0; returns 0.0 when either site is None/empty.
    V4 = 1 when overlap >= 0.5 (correct for majority of GT edit atoms).
    """
    if not gt_site or not pred_site:
        return 0.0
    return len(gt_site & pred_site) / len(gt_site)
