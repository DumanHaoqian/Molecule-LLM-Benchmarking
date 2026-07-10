"""
Parser for murcko_scaffold formal A→B CoT output (unified step format).

Extracts from model output:
  - step1_n_mol_rings   : integer RING_COUNT from --> RING_COUNT(n) after SMILES(...)
  - step2_scaffold      : scaffold SMILES string from --> SCAFFOLD_SMILES("...")
  - step3_n_scaf_rings  : integer RING_COUNT from --> RING_COUNT(n) after SCAFFOLD_SMILES(...)
  - step4_ring_match    : "yes" or "no" from RING_MATCH in step4
  - step5_substructure  : "yes" or "no" from SUBSTRUCTURE_MATCH in step5
  - answer              : scaffold SMILES from Answer line

Compatibility: works with both old dual-part format and new unified step format.
"""
import re
from typing import Optional


def _extract_step1_n_mol_rings(text: str) -> Optional[int]:
    """Extract n from: SMILES(...) --> RING_COUNT(n)"""
    # Try quoted SMILES("...") first (new unified format with nested parens)
    m = re.search(r'SMILES\s*\(\s*"[^"]*"\s*\)\s*-->\s*RING_COUNT\(\s*(\d+)\s*\)', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Fallback: unquoted SMILES(...) (old format, simple cases only)
    m2 = re.search(r'SMILES\s*\([^)]+\)\s*-->\s*RING_COUNT\(\s*(\d+)\s*\)', text, re.IGNORECASE)
    return int(m2.group(1)) if m2 else None


def _extract_step2_scaffold(text: str) -> Optional[str]:
    """Extract scaffold SMILES from: --> SCAFFOLD_SMILES("...")"""
    m = re.search(
        r'-->\s*SCAFFOLD_SMILES\(\s*["\']([^"\']+)["\']\s*\)',
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    # Fallback: unquoted
    m2 = re.search(r'-->\s*SCAFFOLD_SMILES\(([^)]+)\)', text, re.IGNORECASE)
    if m2:
        return m2.group(1).strip().strip('"\'')
    return None


def _extract_step3_n_scaf_rings(text: str) -> Optional[int]:
    """Extract n from: SCAFFOLD_SMILES(...) --> RING_COUNT(n) (step3)"""
    # Try quoted SCAFFOLD_SMILES("...") first (new unified format with nested parens)
    m = re.search(
        r'SCAFFOLD_SMILES\s*\(\s*"[^"]*"\s*\)\s*-->\s*RING_COUNT\(\s*(\d+)\s*\)',
        text,
        re.IGNORECASE,
    )
    if m:
        return int(m.group(1))
    # Fallback: unquoted (old format)
    m2 = re.search(
        r'SCAFFOLD_SMILES\s*\([^)]+\)\s*-->\s*RING_COUNT\(\s*(\d+)\s*\)',
        text,
        re.IGNORECASE,
    )
    return int(m2.group(1)) if m2 else None


def _extract_step4_ring_match(text: str) -> Optional[str]:
    """Extract 'yes' or 'no' from: --> RING_MATCH(yes/no)"""
    m = re.search(r'-->\s*RING_MATCH\(\s*(yes|no)\s*\)', text, re.IGNORECASE)
    return m.group(1).lower() if m else None


def _extract_step5_substructure(text: str) -> Optional[str]:
    """Extract 'yes' or 'no' from: --> SUBSTRUCTURE_MATCH(yes/no)"""
    m = re.search(r'-->\s*SUBSTRUCTURE_MATCH\(\s*(yes|no)\s*\)', text, re.IGNORECASE)
    return m.group(1).lower() if m else None


def _extract_answer(text: str) -> Optional[str]:
    """Extract scaffold SMILES from 'Answer: <smiles>' anywhere in text."""
    m = re.search(r'(?im)^Answer\s*:\s*(\S+)', text)
    if m:
        return m.group(1).strip()
    return None


def parse_formal_output(raw_output: str) -> dict:
    step1_n_mol_rings  = _extract_step1_n_mol_rings(raw_output)
    step2_scaffold     = _extract_step2_scaffold(raw_output)
    step3_n_scaf_rings = _extract_step3_n_scaf_rings(raw_output)
    step4_ring_match   = _extract_step4_ring_match(raw_output)
    step5_substructure = _extract_step5_substructure(raw_output)
    answer             = _extract_answer(raw_output)

    parse_ok = (
        step1_n_mol_rings is not None
        and step2_scaffold is not None
        and step3_n_scaf_rings is not None
        and step4_ring_match is not None
        and step5_substructure is not None
        and answer is not None
    )

    return {
        "part1_text":          raw_output,
        "part2_text":          raw_output,
        "step1_n_mol_rings":   step1_n_mol_rings,
        "step2_scaffold":      step2_scaffold,
        "step3_n_scaf_rings":  step3_n_scaf_rings,
        "step4_ring_match":    step4_ring_match,
        "step5_substructure":  step5_substructure,
        "answer":              answer,
        "parse_ok":            parse_ok,
    }


def parse_all(records: list[dict]) -> list[dict]:
    for rec in records:
        rec.update(parse_formal_output(rec.get("raw_output", "")))
    return records
