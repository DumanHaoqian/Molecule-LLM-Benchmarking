"""
Parser for mutated formal A→B CoT output (unified step format).

Extracts from model output:
  - step1_formula_a       : molecular formula string from --> FORMULA("...")
  - step2_formula_b       : molecular formula string from --> FORMULA("...")
  - step3_formula_match   : "same" or "different" from FORMULA_MATCH in step3
  - step4_canonical_equal : "yes" or "no" from CANONICAL_EQUAL in step4
  - step5_predict         : "Same" or "Different" from PREDICT in step5
  - answer                : "Same" or "Different" from Answer line

Compatibility: works with both old dual-part format and new unified step format.
"""
import re
from typing import Optional


def _extract_step1_formula_a(text: str) -> Optional[str]:
    """Extract formula from first --> FORMULA("...") (mol_A)"""
    m = re.search(
        r'-->\s*FORMULA\(\s*["\']([A-Za-z0-9+-]+)["\']\s*\)',
        text,
    )
    if m:
        return m.group(1).strip()
    # Fallback: unquoted
    m2 = re.search(r'-->\s*FORMULA\(\s*([A-Za-z0-9+-]+)\s*\)', text)
    if m2:
        return m2.group(1).strip()
    return None


def _extract_step2_formula_b(text: str) -> Optional[str]:
    """Extract formula from second --> FORMULA("...") (mol_B)"""
    matches = list(re.finditer(
        r'-->\s*FORMULA\(\s*["\']?([A-Za-z0-9+-]+)["\']?\s*\)',
        text,
    ))
    if len(matches) >= 2:
        return matches[1].group(1).strip()
    return None


def _extract_step3_formula_match(text: str) -> Optional[str]:
    """Extract 'same' or 'different' from: --> FORMULA_MATCH(same/different)"""
    m = re.search(
        r'-->\s*FORMULA_MATCH\(\s*(same|different)\s*\)',
        text,
        re.IGNORECASE,
    )
    return m.group(1).lower() if m else None


def _extract_step4_canonical_equal(text: str) -> Optional[str]:
    """Extract 'yes' or 'no' from: --> CANONICAL_EQUAL(yes/no)"""
    m = re.search(
        r'-->\s*CANONICAL_EQUAL\(\s*(yes|no)\s*\)',
        text,
        re.IGNORECASE,
    )
    return m.group(1).lower() if m else None


def _extract_step5_predict(text: str) -> Optional[str]:
    """Extract 'Same' or 'Different' from: --> PREDICT(Same/Different)"""
    m = re.search(
        r'-->\s*PREDICT\(\s*(Same|Different)\s*\)',
        text,
        re.IGNORECASE,
    )
    return m.group(1) if m else None


def _extract_answer(text: str) -> Optional[str]:
    """Extract 'Same' or 'Different' from 'Answer: ...' anywhere in text."""
    m = re.search(r'(?im)^Answer\s*:\s*(Same|Different)', text)
    return m.group(1) if m else None


def parse_formal_output(raw_output: str) -> dict:
    step1_formula_a       = _extract_step1_formula_a(raw_output)
    step2_formula_b       = _extract_step2_formula_b(raw_output)
    step3_formula_match   = _extract_step3_formula_match(raw_output)
    step4_canonical_equal = _extract_step4_canonical_equal(raw_output)
    step5_predict         = _extract_step5_predict(raw_output)
    answer                = _extract_answer(raw_output)

    parse_ok = (
        step1_formula_a is not None
        and step2_formula_b is not None
        and step3_formula_match is not None
        and step4_canonical_equal is not None
        and step5_predict is not None
        and answer is not None
    )

    return {
        "part1_text":            raw_output,
        "part2_text":            raw_output,
        "step1_formula_a":       step1_formula_a,
        "step2_formula_b":       step2_formula_b,
        "step3_formula_match":   step3_formula_match,
        "step4_canonical_equal": step4_canonical_equal,
        "step5_predict":         step5_predict,
        "answer":                answer,
        "parse_ok":              parse_ok,
    }


def parse_all(records: list[dict]) -> list[dict]:
    for rec in records:
        rec.update(parse_formal_output(rec.get("raw_output", "")))
    return records
