"""
Parser for ring_sys_scaffold formal CoT output (unified step format).

Extracts from model output:
  step1_n_mol    (int)   — RING_COUNT(n) from step1 line after SMILES(molecule)
  step2_n_scaf   (int)   — RING_COUNT(n) from step2 line after SMILES(scaffold)
  step3_non_ring (str)   — "yes" or "no" from NON_RING_ATOMS_EXIST(...)
  step4_predict  (str)   — "Yes" or "No" from PREDICT(...)
  answer         (str)   — "Yes" or "No" from Answer: line

All fields set to None if not found.

Compatibility: works with both old dual-part format and new unified step format.
"""
import re
from typing import Optional

# ────────────────────────────────────────────────────────────────────────────
# Regexes — search entire output for each pattern
# ────────────────────────────────────────────────────────────────────────────

# step1: SMILES(molecule) --> RING_COUNT(n)
_RE_STEP1 = re.compile(
    r'SMILES\s*\(\s*"[^"]*"\s*\)\s*-->\s*RING_COUNT\((\d+)\)',
    re.IGNORECASE,
)
_RE_STEP1_FB = re.compile(
    r'SMILES\s*\([^)]+\)\s*-->\s*RING_COUNT\((\d+)\)',
    re.IGNORECASE,
)

# step2: SMILES(scaffold) --> RING_COUNT(n)
_RE_STEP2 = re.compile(
    r'SMILES\s*\(\s*"[^"]*"\s*\)\s*-->\s*RING_COUNT\((\d+)\)',
    re.IGNORECASE,
)
_RE_STEP2_FB = re.compile(
    r'SMILES\s*\([^)]+\)\s*-->\s*RING_COUNT\((\d+)\)',
    re.IGNORECASE,
)

_RE_STEP3 = re.compile(
    r'NON_RING_ATOMS_EXIST\((yes|no)\)',
    re.IGNORECASE,
)

_RE_STEP4 = re.compile(
    r'PREDICT\((Yes|No)\)',
    re.IGNORECASE,
)

_RE_ANSWER = re.compile(
    r'Answer:\s*(Yes|No)',
    re.IGNORECASE,
)


def _match(regex, text: str) -> Optional[str]:
    m = regex.search(text)
    return m.group(1) if m else None


def parse(raw_output: str) -> dict:
    """Parse model output and return dict of extracted fields.

    Returns:
      {
        "step1_n_mol":    int | None,
        "step2_n_scaf":   int | None,
        "step3_non_ring": "yes" | "no" | None,
        "step4_predict":  "Yes" | "No" | None,
        "answer":         "Yes" | "No" | None,
        "parse_ok":       bool,
      }
    """
    if not raw_output:
        return _empty_parse()

    # step1 and step2 both match SMILES(...) --> RING_COUNT(n).
    # We need the first match (step1, molecule) and the second match (step2, scaffold).
    # Try quoted SMILES first (handles nested parens), fallback to unquoted.
    all_ring_counts = _RE_STEP1.findall(raw_output)
    if len(all_ring_counts) < 2:
        all_ring_counts_fb = _RE_STEP1_FB.findall(raw_output)
        # Merge: prefer quoted results, fill gaps with fallback
        if len(all_ring_counts) == 0 and len(all_ring_counts_fb) >= 1:
            all_ring_counts = [all_ring_counts_fb[0]]
        if len(all_ring_counts) <= 1 and len(all_ring_counts_fb) >= 2:
            all_ring_counts = all_ring_counts + [all_ring_counts_fb[1]] if len(all_ring_counts) == 1 else all_ring_counts_fb[:2]
    step1_raw = all_ring_counts[0] if len(all_ring_counts) >= 1 else None
    step2_raw = all_ring_counts[1] if len(all_ring_counts) >= 2 else None

    step3_raw = _match(_RE_STEP3, raw_output)
    step4_raw = _match(_RE_STEP4, raw_output)
    answer_raw = _match(_RE_ANSWER, raw_output)

    step1_n_mol  = int(step1_raw) if step1_raw is not None else None
    step2_n_scaf = int(step2_raw) if step2_raw is not None else None

    # Normalise case
    step3_non_ring = step3_raw.lower() if step3_raw else None
    step4_predict  = step4_raw.capitalize() if step4_raw else None
    answer         = answer_raw.capitalize() if answer_raw else None

    parse_ok = all(v is not None for v in [
        step1_n_mol, step2_n_scaf, step3_non_ring, step4_predict, answer
    ])

    return {
        "step1_n_mol":    step1_n_mol,
        "step2_n_scaf":   step2_n_scaf,
        "step3_non_ring": step3_non_ring,
        "step4_predict":  step4_predict,
        "answer":         answer,
        "parse_ok":       parse_ok,
    }


def _empty_parse() -> dict:
    return {
        "step1_n_mol":    None,
        "step2_n_scaf":   None,
        "step3_non_ring": None,
        "step4_predict":  None,
        "answer":         None,
        "parse_ok":       False,
    }


def parse_batch(records: list[dict]) -> list[dict]:
    """Apply parse() to each record and merge parsed fields in-place."""
    for rec in records:
        parsed = parse(rec.get("raw_output", ""))
        rec.update(parsed)
    return records
