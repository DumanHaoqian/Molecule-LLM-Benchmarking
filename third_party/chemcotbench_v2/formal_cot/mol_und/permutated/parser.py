"""
Parser for permutated formal A→B CoT output (unified step format).

Expected model output format (4 steps):
  Step 1 [CANONICAL_A]: ... FORMAL: SMILES("{mol_A}") --> CANONICAL_SMILES("{canonical_A}")
  Step 2 [CANONICAL_B]: ... FORMAL: SMILES("{mol_B}") --> CANONICAL_SMILES("{canonical_B}")
  Step 3 [SMILES_IDENTICAL]: ... FORMAL: CANONICAL_SMILES("{canonical_A}") + CANONICAL_SMILES("{canonical_B}") --> SMILES_IDENTICAL(yes/no)
  Step 4 [PREDICT]: ... FORMAL: SMILES_IDENTICAL(yes/no) --> PREDICT(Same/Different)
  Answer: Same/Different

CANONICAL_SMILES strings may contain special chars: [, ], (, ), /, \\, @, =, #, +, -.
The regex uses [^"]+ to match everything except the enclosing double-quote.

Compatibility: works with both old dual-part format and new unified step format.
"""
import re

# ---------------------------------------------------------------------------
# Regex patterns — search entire output
# ---------------------------------------------------------------------------

# First CANONICAL_SMILES output (step1)
_STEP1_RE = re.compile(
    r'SMILES\s*\(\s*"[^"]+"\s*\)\s*-->\s*CANONICAL_SMILES\s*\(\s*"([^"]+)"\s*\)',
    re.IGNORECASE,
)

# Second CANONICAL_SMILES output (step2)
_STEP2_RE = re.compile(
    r'CANONICAL_SMILES\s*\(\s*"([^"]+)"\s*\)\s*\+\s*CANONICAL_SMILES\s*\(\s*"([^"]+)"\s*\)',
    re.IGNORECASE,
)

# step3: SMILES_IDENTICAL(yes|no)
_STEP3_RE = re.compile(
    r'-->\s*SMILES_IDENTICAL\s*\(\s*(yes|no)\s*\)',
    re.IGNORECASE,
)

# step4: PREDICT(Same|Different)
_STEP4_RE = re.compile(
    r'-->\s*PREDICT\s*\(\s*(Same|Different)\s*\)',
    re.IGNORECASE,
)

# Answer: Same / Different (tolerates trailing punctuation/whitespace)
_ANSWER_RE = re.compile(
    r'Answer\s*:\s*(Same|Different)\b',
    re.IGNORECASE,
)


def parse_formal_output(raw_output: str) -> dict:
    """
    Extract structured fields from the model's raw text output.
    Returns a dict; any unmatched field is None.
    parse_ok is True only when all 5 required fields are present.
    """
    m1 = _STEP1_RE.search(raw_output)
    m3 = _STEP3_RE.search(raw_output)
    m4 = _STEP4_RE.search(raw_output)
    ma = _ANSWER_RE.search(raw_output)

    # For step2, we can either use _STEP2_RE or simply reuse the canonical string
    # from step1 and extract the second canonical from the SMILES_IDENTICAL line.
    # A simpler approach: find all CANONICAL_SMILES("...") outputs.
    all_canonical = re.findall(
        r'CANONICAL_SMILES\s*\(\s*"([^"]+)"\s*\)',
        raw_output,
        re.IGNORECASE,
    )

    step1_canonical_a = m1.group(1).strip() if m1 else None
    step2_canonical_b = all_canonical[1].strip() if len(all_canonical) >= 2 else None
    step3_identical   = m3.group(1).lower().strip() if m3 else None     # "yes" or "no"
    step4_predict     = m4.group(1).strip().capitalize() if m4 else None # "Same" or "Different"
    answer            = ma.group(1).strip().capitalize() if ma else None  # "Same" or "Different"

    parse_ok = (
        step1_canonical_a is not None
        and step2_canonical_b is not None
        and step3_identical is not None
        and step4_predict is not None
        and answer is not None
    )

    return {
        "step1_canonical_a": step1_canonical_a,
        "step2_canonical_b": step2_canonical_b,
        "step3_identical":   step3_identical,
        "step4_predict":     step4_predict,
        "answer":            answer,
        "parse_ok":          parse_ok,
    }


def parse_all(records: list[dict]) -> list[dict]:
    """Apply parse_formal_output to all records in-place."""
    for rec in records:
        rec.update(parse_formal_output(rec.get("raw_output", "")))
    return records
