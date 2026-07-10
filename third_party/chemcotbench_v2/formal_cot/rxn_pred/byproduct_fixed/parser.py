"""
Parser for rxn_pred/byproduct_fixed formal A→B CoT output.

Parses Unified Step Format:
  step1_fg_list             — FG_LIST([...])
  step2_rxn_type            — RXN_TYPE("...")
  step3_atomic_delta        — ATOMIC_DELTA([...])
  step4_lf_smiles           — LEAVING_FRAGMENT_SMILES("...")
  step5_fragment_in_reactant — FRAGMENT_IN_REACTANT(yes/no)
  step6_byproduct_smiles    — BYPRODUCT_SMILES("...")
  answer_smiles             — SMILES after "Answer:"
  parse_ok                  — True if all essential fields extracted
"""

from .._parse_utils import (
    extract_formal_line,
    extract_quoted_value,
    extract_bracket_list,
    extract_bool_field,
    extract_answer,
)


def parse_one(record: dict) -> dict:
    text = record.get("raw_output", "") or ""

    s1 = extract_formal_line(text, 1)
    s2 = extract_formal_line(text, 2)
    s3 = extract_formal_line(text, 3)
    s4 = extract_formal_line(text, 4)
    s5 = extract_formal_line(text, 5)
    s6 = extract_formal_line(text, 6)

    step1_fg_list   = extract_bracket_list(s1, "FG_LIST")         if s1 else None
    step2_rxn_type  = extract_quoted_value(s2, "RXN_TYPE")        if s2 else None
    step3_atomic_delta = extract_bracket_list(s3, "ATOMIC_DELTA") if s3 else None
    step4_lf_smiles = extract_quoted_value(s4, "LEAVING_FRAGMENT_SMILES") if s4 else None
    step5_fir       = extract_bool_field(s5, "FRAGMENT_IN_REACTANT")      if s5 else None
    step6_byproduct = extract_quoted_value(s6, "BYPRODUCT_SMILES")        if s6 else None
    answer          = extract_answer(text)

    fg_list_str    = "; ".join(step1_fg_list)    if step1_fg_list    else ""
    delta_list_str = "; ".join(step3_atomic_delta) if step3_atomic_delta else ""

    parse_ok = bool(
        step1_fg_list is not None
        and step2_rxn_type
        and step3_atomic_delta is not None
        and step4_lf_smiles
        and step6_byproduct
        and answer
    )

    parsed = dict(record)
    parsed.update({
        "step1_fg_list":             step1_fg_list or [],
        "step1_fg_list_str":         fg_list_str,
        "step2_rxn_type":            step2_rxn_type or "",
        "step3_atomic_delta":        step3_atomic_delta or [],
        "step3_atomic_delta_str":    delta_list_str,
        "step4_lf_smiles":           step4_lf_smiles or "",
        "step5_fragment_in_reactant": step5_fir or "",
        "step6_byproduct_smiles":    step6_byproduct or "",
        "answer_smiles":             answer or "",
        "parse_ok":                  parse_ok,
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
