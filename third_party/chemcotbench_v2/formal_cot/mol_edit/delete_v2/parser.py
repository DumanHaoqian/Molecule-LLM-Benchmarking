"""
Parser for mol_edit/delete_v2 formal A→B CoT output (unified step format).

Expected format (5 unified steps + Answer):

  Step 1 [ANCHOR_IDENTIFICATION]: ...
    FORMAL: INDEXED_SMILES + INSTRUCTION --> ANCHOR(idx=<n>, element="<X>") + REMOVE_GROUP(smiles="<group>")
  Step 2 [GROUP_SIZE_VERIFICATION]: ...
    FORMAL: REMOVE_GROUP(smiles="<group>") --> HEAVY_ATOMS(<k>)
  Step 3 [PRODUCT_CONSTRUCTION]: ...
    FORMAL: SMILES + ANCHOR(idx=<n>) + REMOVE_GROUP(smiles="<group>") --> PRODUCT_SMILES("<product>")
  Step 4 [HEAVY_ATOM_VERIFICATION]: ...
    FORMAL: SMILES[n_heavy=<a>] + PRODUCT_SMILES[n_heavy=<b>] --> HEAVY_ATOM_DELTA(<delta>)
  Step 5 [RING_VERIFICATION]: ...
    FORMAL: SMILES[n_rings=<c>] + PRODUCT_SMILES[n_rings=<d>] --> RING_DELTA(<delta>)
  Answer: <product_smiles>

Parsed fields (returned in dict, None if not found):
  step1_anchor_idx        int
  step1_anchor_element    str
  step1_remove_group      str    (SMILES of removed fragment)
  step2_remove_smiles     str    (redundant but useful for cross-check)
  step2_heavy_atoms       int
  step3_product_smiles    str
  step4_n_heavy_src       int
  step4_n_heavy_prod      int
  step4_heavy_delta       int
  step5_n_rings_src       int
  step5_n_rings_prod      int
  step5_ring_delta        int
  answer_smiles           str
  parse_ok                bool   (True only if all core fields extracted without error)
  parse_note              str    (description of any parsing failures)
  raw_output_steps        list   (raw text of each extracted step block)
  formal_format_compliance  dict  (which steps used strict template markers)
"""
import re
from typing import Optional


# ── Compiled regexes ────────────────────────────────────────────────────────

# Unified step 1: Step 1 [ANCHOR_IDENTIFICATION]: ... FORMAL: ... --> ANCHOR(idx=8, element="N") + REMOVE_GROUP(smiles="C(=O)OC(C)(C)C")
_RE_STEP1_UNIFIED = re.compile(
    r'Step\s*1\s*\[ANCHOR_IDENTIFICATION\].*?'
    r'FORMAL:\s*.*?-->'
    r'\s*ANCHOR\s*\(\s*idx\s*=\s*(\d+)\s*,\s*element\s*=\s*["\']([A-Za-z]+)["\']\s*\)'
    r'\s*\+\s*REMOVE_GROUP\s*\(\s*smiles\s*=\s*["\']([^"\']*)["\']\s*\)',
    re.IGNORECASE | re.DOTALL,
)

# Fallback old format: step1: ... --> ANCHOR(idx=8, element="N") + REMOVE_GROUP(smiles="C(=O)OC(C)(C)C")
_RE_STEP1_OLD = re.compile(
    r'step1\s*:.*?-->\s*ANCHOR\s*\(\s*idx\s*=\s*(\d+)\s*,\s*element\s*=\s*["\']([A-Za-z]+)["\']\s*\)'
    r'(?:\s*\+\s*REMOVE_GROUP\s*\(\s*smiles\s*=\s*["\']([^"\']*)["\']\s*\))?',
    re.IGNORECASE,
)

# Unified step 2: Step 2 [GROUP_SIZE_VERIFICATION]: ... FORMAL: REMOVE_GROUP(smiles="...") --> HEAVY_ATOMS(7)
_RE_STEP2_UNIFIED = re.compile(
    r'Step\s*2\s*\[GROUP_SIZE_VERIFICATION\].*?FORMAL:\s*.*?'
    r'REMOVE_GROUP\s*\(\s*smiles\s*=\s*["\']([^"\']*)["\']\s*\)'
    r'\s*-->\s*HEAVY_ATOMS\s*\(\s*(\d+)\s*\)',
    re.IGNORECASE | re.DOTALL,
)

# Fallback old step2: step2: REMOVE_GROUP(smiles="...") --> HEAVY_ATOMS(7)
_RE_STEP2_OLD = re.compile(
    r'step2\s*:.*?REMOVE_GROUP\s*\(\s*smiles\s*=\s*["\']([^"\']*)["\']\s*\)'
    r'\s*-->\s*HEAVY_ATOMS\s*\(\s*(\d+)\s*\)',
    re.IGNORECASE,
)

# Unified step 3: Step 3 [PRODUCT_CONSTRUCTION]: ... FORMAL: ... --> PRODUCT_SMILES("NCCc1ccccc1")
_RE_STEP3_UNIFIED = re.compile(
    r'Step\s*3\s*\[PRODUCT_CONSTRUCTION\].*?FORMAL:\s*.*?-->\s*PRODUCT_SMILES\s*\(\s*["\']([^"\']+)["\']\s*\)',
    re.IGNORECASE | re.DOTALL,
)

# Fallback old step3: step3: ... --> PRODUCT_SMILES("NCCc1ccccc1")
_RE_STEP3_OLD = re.compile(
    r'step3\s*:.*?-->\s*PRODUCT_SMILES\s*\(\s*["\']([^"\']+)["\']\s*\)',
    re.IGNORECASE,
)

# ── Step 4: strict vs relaxed ──────────────────────────────────────────────
_RE_STEP4_STRICT = re.compile(
    r'Step\s*4\s*\[HEAVY_ATOM_VERIFICATION\].*?FORMAL:\s*.*?'
    r'SMILES\s*\[n_heavy\s*=\s*(\d+)\]'
    r'.*?PRODUCT_SMILES\s*\[n_heavy\s*=\s*(\d+)\]'
    r'.*?-->\s*HEAVY_ATOM_DELTA\s*\(\s*([+-]?\d+)\s*\)',
    re.IGNORECASE | re.DOTALL,
)
_RE_STEP4_RELAXED = re.compile(
    r'Step\s*4\s*\[HEAVY_ATOM_VERIFICATION\].*?FORMAL:\s*.*?'
    r'\[n_heavy\s*=\s*(\d+)\]'
    r'.*?\[n_heavy\s*=\s*(\d+)\]'
    r'.*?-->\s*HEAVY_ATOM_DELTA\s*\(\s*([+-]?\d+)\s*\)',
    re.IGNORECASE | re.DOTALL,
)
_RE_STEP4_OLD = re.compile(
    r'step4\s*:.*?\[n_heavy\s*=\s*(\d+)\]'
    r'.*?\[n_heavy\s*=\s*(\d+)\]'
    r'.*?-->\s*HEAVY_ATOM_DELTA\s*\(\s*([+-]?\d+)\s*\)',
    re.IGNORECASE,
)

# ── Step 5: strict vs relaxed ──────────────────────────────────────────────
_RE_STEP5_STRICT = re.compile(
    r'Step\s*5\s*\[RING_VERIFICATION\].*?FORMAL:\s*.*?'
    r'SMILES\s*\[n_rings\s*=\s*(\d+)\]'
    r'.*?PRODUCT_SMILES\s*\[n_rings\s*=\s*(\d+)\]'
    r'.*?-->\s*RING_DELTA\s*\(\s*([+-]?\d+)\s*\)',
    re.IGNORECASE | re.DOTALL,
)
_RE_STEP5_RELAXED = re.compile(
    r'Step\s*5\s*\[RING_VERIFICATION\].*?FORMAL:\s*.*?'
    r'\[n_rings\s*=\s*(\d+)\]'
    r'.*?\[n_rings\s*=\s*(\d+)\]'
    r'.*?-->\s*RING_DELTA\s*\(\s*([+-]?\d+)\s*\)',
    re.IGNORECASE | re.DOTALL,
)
_RE_STEP5_OLD = re.compile(
    r'step5\s*:.*?\[n_rings\s*=\s*(\d+)\]'
    r'.*?\[n_rings\s*=\s*(\d+)\]'
    r'.*?-->\s*RING_DELTA\s*\(\s*([+-]?\d+)\s*\)',
    re.IGNORECASE,
)

# Answer: <smiles>
_RE_ANSWER = re.compile(
    r'^Answer\s*:\s*(.+)$',
    re.IGNORECASE | re.MULTILINE,
)


def _extract_step_text(raw_output: str, step_num: int, step_name: str) -> str:
    """Extract the raw text block for a given unified step."""
    pattern = rf'Step\s*{step_num}\s*\[{re.escape(step_name)}\]:.*?(?=Step\s*{step_num + 1}\s*\[|$)'
    m = re.search(pattern, raw_output, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(0).strip()
    return ""


def parse_output(raw_output: str) -> dict:
    """Parse Gemini's raw output for a mol_edit/delete_v2 sample.

    Returns a dict of parsed fields plus parse_ok and parse_note.
    Compatible with both unified step format and old stepN: format.
    """
    if not raw_output or not raw_output.strip():
        return _empty_parse("empty output")

    failures = []
    raw_output_steps = []
    formal_format_compliance = {}

    # Try to extract unified step text blocks
    step_names = [
        (1, "ANCHOR_IDENTIFICATION"),
        (2, "GROUP_SIZE_VERIFICATION"),
        (3, "PRODUCT_CONSTRUCTION"),
        (4, "HEAVY_ATOM_VERIFICATION"),
        (5, "RING_VERIFICATION"),
    ]
    for num, name in step_names:
        txt = _extract_step_text(raw_output, num, name)
        if txt:
            raw_output_steps.append(txt)

    # ── step1 ────────────────────────────────────────────────────────────────
    m1 = _RE_STEP1_UNIFIED.search(raw_output)
    if not m1:
        m1 = _RE_STEP1_OLD.search(raw_output)
    if m1:
        try:
            anchor_idx     = int(m1.group(1))
            anchor_element = m1.group(2).upper()
            remove_group   = m1.group(3)
            if remove_group is not None:
                remove_group = remove_group.strip()
        except (ValueError, AttributeError):
            anchor_idx = anchor_element = remove_group = None
            failures.append("step1_parse_error")
    else:
        anchor_idx = anchor_element = remove_group = None
        failures.append("step1_not_found")

    # ── step2 ────────────────────────────────────────────────────────────────
    m2 = _RE_STEP2_UNIFIED.search(raw_output)
    if not m2:
        m2 = _RE_STEP2_OLD.search(raw_output)
    if m2:
        try:
            step2_remove_smiles = m2.group(1).strip()
            heavy_atoms         = int(m2.group(2))
        except (ValueError, AttributeError):
            step2_remove_smiles = None
            heavy_atoms = None
            failures.append("step2_parse_error")
    else:
        step2_remove_smiles = None
        heavy_atoms = None
        failures.append("step2_not_found")

    # ── step3 ────────────────────────────────────────────────────────────────
    m3 = _RE_STEP3_UNIFIED.search(raw_output)
    if not m3:
        m3 = _RE_STEP3_OLD.search(raw_output)
    if m3:
        product_smiles = m3.group(1).strip()
        if not product_smiles:
            product_smiles = None
            failures.append("step3_empty_smiles")
    else:
        product_smiles = None
        failures.append("step3_not_found")

    # ── step4 ────────────────────────────────────────────────────────────────
    m4_strict = _RE_STEP4_STRICT.search(raw_output)
    m4 = m4_strict
    formal_format_compliance["step4"] = bool(m4_strict)
    if not m4:
        m4 = _RE_STEP4_RELAXED.search(raw_output)
    if not m4:
        m4 = _RE_STEP4_OLD.search(raw_output)
    if m4:
        try:
            n_heavy_src  = int(m4.group(1))
            n_heavy_prod = int(m4.group(2))
            heavy_delta  = int(m4.group(3))
        except (ValueError, AttributeError):
            n_heavy_src = n_heavy_prod = heavy_delta = None
            failures.append("step4_parse_error")
    else:
        n_heavy_src = n_heavy_prod = heavy_delta = None
        failures.append("step4_not_found")

    # ── step5 ────────────────────────────────────────────────────────────────
    m5_strict = _RE_STEP5_STRICT.search(raw_output)
    m5 = m5_strict
    formal_format_compliance["step5"] = bool(m5_strict)
    if not m5:
        m5 = _RE_STEP5_RELAXED.search(raw_output)
    if not m5:
        m5 = _RE_STEP5_OLD.search(raw_output)
    if m5:
        try:
            n_rings_src  = int(m5.group(1))
            n_rings_prod = int(m5.group(2))
            ring_delta   = int(m5.group(3))
        except (ValueError, AttributeError):
            n_rings_src = n_rings_prod = ring_delta = None
            failures.append("step5_parse_error")
    else:
        n_rings_src = n_rings_prod = ring_delta = None
        failures.append("step5_not_found")

    # ── Answer line ──────────────────────────────────────────────────────────
    ma = _RE_ANSWER.search(raw_output)
    if ma:
        answer_smiles = ma.group(1).strip()
        if not answer_smiles:
            answer_smiles = None
            failures.append("answer_empty")
    else:
        answer_smiles = None
        failures.append("answer_not_found")

    parse_ok   = len(failures) == 0
    parse_note = "; ".join(failures) if failures else "ok"

    return {
        "step1_anchor_idx":      anchor_idx,
        "step1_anchor_element":  anchor_element,
        "step1_remove_group":    remove_group,
        "step2_remove_smiles":   step2_remove_smiles,
        "step2_heavy_atoms":     heavy_atoms,
        "step3_product_smiles":  product_smiles,
        "step4_n_heavy_src":     n_heavy_src,
        "step4_n_heavy_prod":    n_heavy_prod,
        "step4_heavy_delta":     heavy_delta,
        "step5_n_rings_src":     n_rings_src,
        "step5_n_rings_prod":    n_rings_prod,
        "step5_ring_delta":      ring_delta,
        "answer_smiles":         answer_smiles,
        "parse_ok":              parse_ok,
        "parse_note":            parse_note,
        "raw_output_steps":      raw_output_steps,
        "formal_format_compliance": formal_format_compliance,
    }


def _empty_parse(reason: str) -> dict:
    return {
        "step1_anchor_idx":      None,
        "step1_anchor_element":  None,
        "step1_remove_group":    None,
        "step2_remove_smiles":   None,
        "step2_heavy_atoms":     None,
        "step3_product_smiles":  None,
        "step4_n_heavy_src":     None,
        "step4_n_heavy_prod":    None,
        "step4_heavy_delta":     None,
        "step5_n_rings_src":     None,
        "step5_n_rings_prod":    None,
        "step5_ring_delta":      None,
        "answer_smiles":         None,
        "parse_ok":              False,
        "parse_note":            reason,
        "raw_output_steps":      [],
        "formal_format_compliance": {},
    }


def parse_batch(records: list[dict]) -> list[dict]:
    """Parse raw_output for each record in-place, adding parsed fields."""
    for rec in records:
        if not rec.get("api_success", False):
            rec.update(_empty_parse("api_failed"))
        else:
            rec.update(parse_output(rec.get("raw_output", "")))
    return records
