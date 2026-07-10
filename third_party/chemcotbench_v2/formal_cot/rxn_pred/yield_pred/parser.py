"""
Parser for rxn_pred/yield_pred formal A→B CoT output (regression, unified step format).

Extracts the 6-step formal verification chain.

  Step 1 [RXN_CLASS]:             FORMAL: TASK("...") --> RXN_CLASS("...")
  Step 2 [HALIDE_IDENTITY]:       FORMAL: ELECTROPHILE("...") --> HALIDE_TYPE("...")
  Step 3 [NUCLEOPHILE_CHARACTER]: FORMAL: NUCLEOPHILE("...") --> NUCLEOPHILE_TYPE("...") + NUCLEOPHILE_FORM("...")
  Step 4 [LIGAND_SYSTEM_SCORE]:   FORMAL: CONDITIONS("...") --> LIGAND_CLASS("...")
  Step 5 [YIELD_PREDICTION]:      FORMAL: ... --> PREDICTED_YIELD("<numeric>")
  Step 6 [SELF_CONSISTENCY]:      FORMAL: ... --> SELF_CONSISTENT("yes|no")
  Answer: <numeric>
"""

import re
from typing import Optional


def _extract_step_text(raw_output: str) -> list[str]:
    """Split raw output into individual step texts + Answer."""
    if not raw_output:
        return []
    pattern = re.compile(
        r'(Step\s*\d+\s*\[[^\]]+\]:.*?FORMAL:\s*.+?)(?=\n\s*Step\s*\d+|\n\s*Answer:|\Z)',
        re.DOTALL | re.IGNORECASE,
    )
    steps = [m.group(1).strip() for m in pattern.finditer(raw_output)]
    ans_m = re.search(r'^Answer\s*:\s*(.+)$', raw_output, re.MULTILINE | re.IGNORECASE)
    if ans_m:
        steps.append(f"Answer: {ans_m.group(1).strip()}")
    return steps


def _isolate_formal_lines(raw_output: str) -> str:
    """Extract all FORMAL: lines from the raw output."""
    lines = []
    for m in re.finditer(r'^\s*FORMAL:\s*(.+)$', raw_output, re.MULTILINE | re.IGNORECASE):
        lines.append(m.group(1).strip())
    return '\n'.join(lines)


def _extract_quoted_value(text: str, tag: str) -> Optional[str]:
    """Extract a quoted value after a tag like RXN_CLASS("<value>")."""
    pattern = rf'{re.escape(tag)}\(\s*"([^"]*)"\s*\)'
    m = re.search(pattern, text)
    if m:
        return m.group(1).strip()
    return None


def _extract_numeric_value(text: str, tag: str) -> Optional[str]:
    """Extract a numeric value after a tag like PREDICTED_YIELD("85.3")."""
    pattern = rf'{re.escape(tag)}\(\s*"?([+-]?\d+(?:\.\d+)?)"?\s*\)'
    m = re.search(pattern, text)
    if m:
        return m.group(1).strip()
    return None


def _extract_answer(raw_output: str) -> Optional[str]:
    """Extract the Answer line numeric value."""
    m = re.search(r'^Answer\s*:\s*([+-]?\d+(?:\.\d+)?)', raw_output, re.MULTILINE | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _is_numeric(val: str) -> bool:
    if not val:
        return False
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def _normalise_halide_type(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.lower().strip().rstrip('.,;:').strip('"\'')
    if "chloride" in t or "ar-cl" in t:
        return "aryl chloride"
    if "bromide" in t or "ar-br" in t:
        return "aryl bromide"
    if "iodide" in t or "ar-i" in t:
        return "aryl iodide"
    if "triflate" in t or "otf" in t:
        return "aryl triflate"
    return t


def _normalise_ligand_class(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.lower().strip().rstrip('.,;:').strip('"\'')
    if "high" in t:
        return "high-performance"
    if "poor" in t:
        return "poor"
    if "standard" in t or "moderate" in t:
        return "standard"
    return t


def _normalise_nucleophile_form(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.lower().strip().rstrip('.,;:').strip('"\'')
    if "trifluoroborate" in t or "bf3" in t:
        return "trifluoroborate"
    if "boronate ester" in t or "boronate" in t or "bpin" in t or "b(or" in t:
        return "boronate ester"
    if "boronic acid" in t or "b(oh)" in t:
        return "free boronic acid"
    if "amine" in t:
        return "amine"
    if "not provided" in t or "not separately" in t:
        return "not provided"
    return t


def parse_one(raw_output: str) -> dict:
    """Parse a single Gemini output into structured fields."""
    if not raw_output:
        return {
            "step1_rxn_class": None, "step2_halide_type": None,
            "step3_nucleophile_type": None, "step3_nucleophile_form": None,
            "step4_ligand_class": None, "step5_predicted_yield": None,
            "step6_self_consistent": None, "answer_yield": None,
            "parse_ok": False, "raw_output_steps": [],
        }

    formal_block = _isolate_formal_lines(raw_output)

    step1_rxn_class = _extract_quoted_value(formal_block, "RXN_CLASS")
    step2_halide_raw = _extract_quoted_value(formal_block, "HALIDE_TYPE")
    step2_halide_type = _normalise_halide_type(step2_halide_raw) if step2_halide_raw else ""
    step3_nucleophile_type = _extract_quoted_value(formal_block, "NUCLEOPHILE_TYPE")
    step3_form_raw = _extract_quoted_value(formal_block, "NUCLEOPHILE_FORM")
    step3_nucleophile_form = _normalise_nucleophile_form(step3_form_raw) if step3_form_raw else ""
    step4_ligand_raw = _extract_quoted_value(formal_block, "LIGAND_CLASS")
    step4_ligand_class = _normalise_ligand_class(step4_ligand_raw) if step4_ligand_raw else ""
    step5_predicted_yield = _extract_numeric_value(formal_block, "PREDICTED_YIELD")
    step6_self_consistent = _extract_quoted_value(formal_block, "SELF_CONSISTENT")
    answer_yield = _extract_answer(raw_output)

    parse_ok = (
        step1_rxn_class is not None and step1_rxn_class.strip() != ""
        and step2_halide_type in ("aryl chloride", "aryl bromide", "aryl iodide", "aryl triflate")
        and step3_nucleophile_form in ("free boronic acid", "boronate ester", "trifluoroborate", "amine", "not provided")
        and step4_ligand_class in ("high-performance", "standard", "poor")
        and step5_predicted_yield is not None and _is_numeric(step5_predicted_yield)
        and answer_yield is not None and _is_numeric(answer_yield)
    )

    return {
        "step1_rxn_class": step1_rxn_class or "",
        "step2_halide_type": step2_halide_type,
        "step3_nucleophile_type": step3_nucleophile_type or "",
        "step3_nucleophile_form": step3_nucleophile_form,
        "step4_ligand_class": step4_ligand_class,
        "step5_predicted_yield": step5_predicted_yield,
        "step6_self_consistent": step6_self_consistent,
        "answer_yield": answer_yield,
        "parse_ok": parse_ok,
        "raw_output_steps": _extract_step_text(raw_output),
    }


def parse_all(records: list[dict]) -> list[dict]:
    """Add parsed fields to each record in-place."""
    for record in records:
        parsed = parse_one(record.get("raw_output", ""))
        record.update(parsed)
    return records
