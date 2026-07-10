"""
Parser for rxn_pred/rcr_solvent formal A→B CoT output (unified step format).

Extracts the 6-step formal verification chain from FORMAL lines.
"""

import re
from typing import Optional


def _isolate_formal_lines(raw_output: str) -> str:
    """Extract all FORMAL: lines from the raw output."""
    lines = []
    for m in re.finditer(r'^\s*FORMAL:\s*(.+)$', raw_output, re.MULTILINE | re.IGNORECASE):
        lines.append(m.group(1).strip())
    return '\n'.join(lines)


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


def _extract_quoted_value(text: str, tag: str) -> Optional[str]:
    """Extract a quoted value after a tag like RXN_CLASS("<value>")."""
    pattern = rf'{re.escape(tag)}\(\s*"([^"]*)"\s*\)'
    m = re.search(pattern, text)
    if m:
        return m.group(1).strip()
    return None


def _extract_answer(text: str) -> Optional[str]:
    m = re.search(r"(?i)^Answer\s*:\s*(.+)$", text, re.MULTILINE)
    if not m:
        return None
    return m.group(1).strip().rstrip(".,;")


_PROTICITY_VALUES = {"protic", "aprotic"}
_POLARITY_VALUES = {"polar", "nonpolar"}


def _normalise_proticity(text: str | None) -> str:
    if not text:
        return ""
    t = text.strip().lower().rstrip(".,;:").strip('"\'')
    if t in _PROTICITY_VALUES:
        return t
    return ""


def _normalise_polarity(text: str | None) -> str:
    if not text:
        return ""
    t = text.strip().lower().rstrip(".,;:").strip('"\'')
    if t in _POLARITY_VALUES:
        return t
    return ""


def _normalise_yes_no(text: str | None) -> str:
    if not text:
        return ""
    t = text.strip().lower().rstrip(".,;:").strip('"\'')
    if t in {"yes", "no"}:
        return t
    return ""


def parse_one(record: dict) -> dict:
    text = record.get("raw_output", "") or ""
    parsed = dict(record)
    if not text:
        parsed.update({
            "step1_rxn_class": "", "step2_rxn_smiles": "",
            "step2_core_transformation": "", "step3_proticity": "",
            "step4_polarity": "", "step5_predicted_solvent_smiles": "",
            "step6_self_consistent": "", "answer_smiles": "",
            "parse_ok": False, "raw_output_steps": [],
        })
        return parsed

    formal_block = _isolate_formal_lines(text)

    step1_rxn_class = _extract_quoted_value(formal_block, "RXN_CLASS")
    step2_rxn_smiles = _extract_quoted_value(formal_block, "RXN_SMILES")
    step2_core_transformation = _extract_quoted_value(formal_block, "CORE_TRANSFORMATION")
    step3_proticity_raw = _extract_quoted_value(formal_block, "PROTICITY")
    step3_proticity = _normalise_proticity(step3_proticity_raw)
    step4_polarity_raw = _extract_quoted_value(formal_block, "POLARITY")
    step4_polarity = _normalise_polarity(step4_polarity_raw)
    step5_pred_solvent = _extract_quoted_value(formal_block, "PREDICTED_SOLVENT_SMILES")
    step6_self_raw = _extract_quoted_value(formal_block, "SELF_CONSISTENT")
    step6_self_consistent = _normalise_yes_no(step6_self_raw)
    answer_smiles = _extract_answer(text)

    parse_ok = bool(
        step1_rxn_class
        and step2_rxn_smiles
        and step2_core_transformation
        and step3_proticity in _PROTICITY_VALUES
        and step4_polarity in _POLARITY_VALUES
        and step5_pred_solvent
        and step6_self_consistent in {"yes", "no"}
        and answer_smiles
    )

    parsed.update({
        "step1_rxn_class": step1_rxn_class or "",
        "step2_rxn_smiles": step2_rxn_smiles or "",
        "step2_core_transformation": step2_core_transformation or "",
        "step3_proticity": step3_proticity or "",
        "step4_polarity": step4_polarity or "",
        "step5_predicted_solvent_smiles": step5_pred_solvent or "",
        "step6_self_consistent": step6_self_consistent or "",
        "answer_smiles": answer_smiles or "",
        "parse_ok": parse_ok,
        "raw_output_steps": _extract_step_text(text),
    })
    return parsed


def parse_all(records: list[dict]) -> list[dict]:
    n_ok = 0
    for i, rec in enumerate(records):
        records[i] = parse_one(rec)
        if records[i]["parse_ok"]:
            n_ok += 1
    print(f"parse_all: {n_ok}/{len(records)} parse_ok")
    return records
