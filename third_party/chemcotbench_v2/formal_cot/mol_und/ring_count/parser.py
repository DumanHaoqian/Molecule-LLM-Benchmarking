"""
Parser for ring_count formal A→B CoT output (unified step format).

Extracts from model output:
  - step1_smarts     : primary SMARTS from --> SMARTS("...")
  - step1_alts       : list of EQUIV alternative SMARTS
  - step2_total      : integer RING_COUNT_TOTAL from --> RING_COUNT_TOTAL(n)
  - step3_n          : integer match count from MATCH_ATOMS in step3
  - step4_count      : integer COUNT from --> COUNT(N)
  - step5_rejected   : integer REJECTED from --> REJECTED(N)
  - answer           : integer from "Answer: N"

Compatibility: works with both old dual-part format and new unified step format.
"""
import re
from typing import Optional


def _extract_step1_smarts(text: str) -> Optional[str]:
    """Extract primary SMARTS from: --> SMARTS("...")"""
    m = re.search(r'-->\s*SMARTS\(\s*["\']([^"\']+)["\']\s*\)', text)
    if m:
        return m.group(1).strip()
    m2 = re.search(r'-->\s*SMARTS\(([^)]+)\)', text)
    if m2:
        return m2.group(1).strip().strip('"\'')
    return None


def _extract_step1_alts(text: str) -> list[str]:
    """Extract EQUIV alternatives."""
    m = re.search(r'EQUIV\s*\(([^)]+)\)', text)
    if not m:
        return []
    return re.findall(r'["\']([^"\']+)["\']', m.group(1))


def _extract_step2_total(text: str) -> Optional[int]:
    """Extract n from: --> RING_COUNT_TOTAL(n)"""
    m = re.search(
        r'-->\s*RING_COUNT_TOTAL\(\s*(\d+)\s*\)',
        text,
    )
    if m:
        return int(m.group(1))
    return None


def _extract_step3_n(text: str) -> Optional[int]:
    """Extract N from: MATCH_ATOMS([N match(es): ...])"""
    m = re.search(
        r'MATCH_ATOMS\(\[\s*(\d+)\s*match',
        text,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1))
    return None


def _extract_step4_count(text: str) -> Optional[int]:
    """Extract N from: --> COUNT(N)"""
    m = re.search(r'-->\s*COUNT\(\s*(\d+)\s*\)', text)
    if m:
        return int(m.group(1))
    return None


def _extract_step5_rejected(text: str) -> Optional[int]:
    """Extract N from: --> REJECTED(N)"""
    m = re.search(r'-->\s*REJECTED\(\s*(-?\d+)\s*\)', text)
    if m:
        return int(m.group(1))
    return None


def _extract_answer(text: str) -> Optional[int]:
    """Extract integer from 'Answer: N' anywhere in the text."""
    m = re.search(r'(?im)^Answer\s*:\s*(\d+)', text)
    if m:
        return int(m.group(1))
    m2 = re.search(r'Answer\s*:?\s*(\d+)', text, re.IGNORECASE)
    return int(m2.group(1)) if m2 else None


def parse_formal_output(raw_output: str) -> dict:
    step1_smarts   = _extract_step1_smarts(raw_output)
    step1_alts     = _extract_step1_alts(raw_output)
    step2_total    = _extract_step2_total(raw_output)
    step3_n        = _extract_step3_n(raw_output)
    step4_count    = _extract_step4_count(raw_output)
    step5_rejected = _extract_step5_rejected(raw_output)
    answer         = _extract_answer(raw_output)

    parse_ok = (
        step1_smarts is not None
        and step2_total is not None
        and step3_n is not None
        and step4_count is not None
        and step5_rejected is not None
        and answer is not None
    )

    return {
        "part1_text":      raw_output,
        "part2_text":      raw_output,
        "step1_smarts":    step1_smarts,
        "step1_alts":      step1_alts,
        "step2_total":     step2_total,
        "step3_n":         step3_n,
        "step4_count":     step4_count,
        "step5_rejected":  step5_rejected,
        "answer":          answer,
        "parse_ok":        parse_ok,
    }


def parse_all(records: list[dict]) -> list[dict]:
    for rec in records:
        rec.update(parse_formal_output(rec.get("raw_output", "")))
    return records
