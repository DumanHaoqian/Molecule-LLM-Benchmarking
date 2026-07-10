"""
Parser for rxn_pred/mech_sel formal A→B CoT output.

Parses Unified Step Format:
  step1_smarts_list   — SMARTS_LIST([...])
  step1_reagent_types — REAGENT_TYPES([...])
  step2_rxn_type      — RXN_TYPE("...")
  step3_eliminated    — ELIMINATED_OPTIONS([...])
  step4_remaining     — REMAINING_OPTIONS([...])
  step5_selected      — SELECTED_OPTION("...")
  answer_letter       — letter after "Answer:"
  parse_ok            — True if all essential fields extracted
"""

from .._parse_utils import (
    extract_formal_line,
    extract_quoted_value,
    extract_bracket_list,
    extract_answer,
)


def parse_one(record: dict) -> dict:
    text = record.get("raw_output", "") or ""

    s1 = extract_formal_line(text, 1)
    s2 = extract_formal_line(text, 2)
    s3 = extract_formal_line(text, 3)
    s4 = extract_formal_line(text, 4)
    s5 = extract_formal_line(text, 5)

    step1_smarts    = extract_bracket_list(s1, "SMARTS_LIST")       if s1 else None
    step1_reagents  = extract_bracket_list(s1, "REAGENT_TYPES")     if s1 else None
    step2_rxn_type  = extract_quoted_value(s2, "RXN_TYPE")          if s2 else None
    step3_eliminated = extract_bracket_list(s3, "ELIMINATED_OPTIONS") if s3 else None
    step4_remaining = extract_bracket_list(s4, "REMAINING_OPTIONS")  if s4 else None
    step5_selected  = extract_quoted_value(s5, "SELECTED_OPTION")   if s5 else None
    answer          = extract_answer(text)

    # Normalise selected option to uppercase single letter
    if step5_selected:
        step5_selected = step5_selected.strip().upper()
        if len(step5_selected) > 1:
            step5_selected = step5_selected[0]

    parse_ok = bool(
        step1_smarts is not None
        and step2_rxn_type
        and step3_eliminated is not None
        and step4_remaining is not None
        and step5_selected
        and answer
    )

    parsed = dict(record)
    parsed.update({
        "step1_smarts_list":   step1_smarts or [],
        "step1_reagent_types": step1_reagents or [],
        "step2_rxn_type":      step2_rxn_type or "",
        "step3_eliminated":    step3_eliminated or [],
        "step4_remaining":     step4_remaining or [],
        "step5_selected":      step5_selected or "",
        "answer_letter":       answer or "",
        "parse_ok":            parse_ok,
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
