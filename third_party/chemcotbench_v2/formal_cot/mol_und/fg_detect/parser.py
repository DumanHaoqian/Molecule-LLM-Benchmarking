"""
Parser for fg_detect formal A→B CoT output (unified step format).

Extracts from model output:
  - step1_smarts : SMARTS string from --> SMARTS("...")
  - step1_alts   : list of EQUIV alternative SMARTS
  - step2_n      : integer match count from MATCH_ATOMS([N match(es): ...])
  - step3_count  : integer from --> COUNT(N)
  - answer       : integer from "Answer: N"

Compatibility: works with both old dual-part format (PART 1 / PART 2)
and new unified step format (Step N [FIELD_NAME]: ... FORMAL: ...).
"""
import re
from typing import Optional


def _extract_step1_smarts(text: str) -> Optional[str]:
    """Extract primary SMARTS from: --> SMARTS("...")"""
    m = re.search(r'-->\s*SMARTS\(\s*["\']([^"\']+)["\']\s*\)', text)
    if m:
        return m.group(1).strip()
    # Fallback: unquoted
    m2 = re.search(r'-->\s*SMARTS\(([^)]+)\)', text)
    if m2:
        return m2.group(1).strip().strip('"\'')
    return None


def _extract_step1_alts(text: str) -> list[str]:
    """Extract alternative SMARTS from optional EQUIV("alt1", "alt2").

    Handles:
      ... --> SMARTS("primary") EQUIV("alt1")
      ... --> SMARTS("primary") EQUIV("alt1", "alt2")
    Returns a (possibly empty) list of alternative SMARTS strings.
    """
    m = re.search(r'EQUIV\s*\(([^)]+)\)', text)
    if not m:
        return []
    return re.findall(r'["\']([^"\']+)["\']', m.group(1))


def _extract_step2_n(text: str) -> Optional[int]:
    """Extract N from: MATCH_ATOMS([N match(es): ...])"""
    m = re.search(
        r'MATCH_ATOMS\(\[\s*(\d+)\s*match',
        text,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1))
    return None


def _extract_step3_count(text: str) -> Optional[int]:
    """Extract N from: --> COUNT(N)"""
    m = re.search(r'-->\s*COUNT\(\s*(\d+)\s*\)', text)
    if m:
        return int(m.group(1))
    return None


def _extract_answer(text: str) -> Optional[int]:
    """Extract integer from 'Answer: N' anywhere in the text."""
    m = re.search(r'(?im)^Answer\s*:\s*(\d+)', text)
    if m:
        return int(m.group(1))
    # Fallback: last standalone integer near "Answer"
    m2 = re.search(r'Answer\s*:?\s*(\d+)', text, re.IGNORECASE)
    return int(m2.group(1)) if m2 else None


def parse_formal_output(raw_output: str) -> dict:
    step1_smarts = _extract_step1_smarts(raw_output)
    step1_alts   = _extract_step1_alts(raw_output)
    step2_n      = _extract_step2_n(raw_output)
    step3_count  = _extract_step3_count(raw_output)
    answer       = _extract_answer(raw_output)

    # parse_ok requires the three core fields
    parse_ok = (
        step1_smarts is not None
        and step2_n is not None
        and step3_count is not None
        and answer is not None
    )

    return {
        "part1_text":    raw_output,
        "part2_text":    raw_output,
        "step1_smarts":  step1_smarts,
        "step1_alts":    step1_alts,
        "step2_n":       step2_n,
        "step3_count":   step3_count,
        "answer":        answer,
        "parse_ok":      parse_ok,
    }


def parse_all(records: list[dict]) -> list[dict]:
    for rec in records:
        rec.update(parse_formal_output(rec.get("raw_output", "")))
    return records
