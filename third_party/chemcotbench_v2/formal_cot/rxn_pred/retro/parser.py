"""
Parser for rxn_pred/retro formal A→B CoT output.

Parses Unified Step Format:
  step1_fg_list       — FG_LIST([...])
  step2_rxn_type      — RXN_TYPE("...")
  step3_mechanism     — MECHANISM_KWORD("...")
  step4_reactant_smi  — REACTANT_SMILES("...")
  step5_all_valid     — ALL_FRAGS_VALID(yes/no)
  step6_bond_broken   — BOND_BROKEN("...")
  step7_fwd_match     — "yes"/"no" from FWD_CONSISTENCY(match=yes/no, note="...")
  step7_fwd_note      — note string from FWD_CONSISTENCY
  answer_smi          — SMILES after "Answer:"
  parse_ok            — True if essential fields extracted
"""

import re

from .._parse_utils import (
    extract_formal_line,
    extract_quoted_value,
    extract_bracket_list,
    extract_bool_field,
    extract_answer,
    extract_fwd_consistency,
)


def parse_one(record: dict) -> dict:
    text = record.get("raw_output", "") or ""

    s1 = extract_formal_line(text, 1)
    s2 = extract_formal_line(text, 2)
    s3 = extract_formal_line(text, 3)
    s4 = extract_formal_line(text, 4)
    s5 = extract_formal_line(text, 5)
    s6 = extract_formal_line(text, 6)
    s7 = extract_formal_line(text, 7)

    step1_fg_list   = extract_bracket_list(s1, "FG_LIST")       if s1 else None
    step2_rxn_type  = extract_quoted_value(s2, "RXN_TYPE")       if s2 else None
    step3_mechanism = extract_quoted_value(s3, "MECHANISM_KWORD") if s3 else None
    step4_reactant  = extract_quoted_value(s4, "REACTANT_SMILES") if s4 else None
    step5_all_valid = extract_bool_field(s5, "ALL_FRAGS_VALID")  if s5 else None
    step6_bond      = extract_quoted_value(s6, "BOND_BROKEN")    if s6 else None

    step7_fwd_match = ""
    step7_fwd_note = ""
    if s7:
        fwd = extract_fwd_consistency(s7)
        if fwd:
            step7_fwd_match, step7_fwd_note = fwd

    answer = extract_answer(text)

    parse_ok = bool(
        step1_fg_list is not None
        and step2_rxn_type
        and step4_reactant
        and answer
    )

    parsed = dict(record)
    parsed.update({
        "step1_fg_list":      step1_fg_list or [],
        "step2_rxn_type":     step2_rxn_type or "",
        "step3_mechanism":    step3_mechanism or "",
        "step4_reactant_smi": step4_reactant or "",
        "step5_all_valid":    step5_all_valid or "",
        "step6_bond_broken":  step6_bond or "",
        "step7_fwd_match":    step7_fwd_match,
        "step7_fwd_note":     step7_fwd_note,
        "answer_smi":         answer or "",
        "parse_ok":           parse_ok,
    })
    return parsed


def parse_all(records: list[dict]) -> list[dict]:
    return [parse_one(r) for r in records]
