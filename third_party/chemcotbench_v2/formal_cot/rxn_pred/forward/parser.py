"""
Parser for rxn_pred/forward formal A→B CoT output.

Parses Gemini's raw text output into structured fields from Unified Step Format:
  step1_fg_list       — list of FG names from FG_LIST([...]) in step1
  step2_rxn_type      — RXN_TYPE("...") string from step2
  step3_mechanism     — MECHANISM_KWORD("...") string from step3
  step4_predicted_smi — PREDICTED_SMILES("...") from step4
  step5_mol_valid     — "yes"/"no" from MOL_VALID(...) in step5
  step6_bond_formed   — BOND_FORMED("...") string from step6
  answer_smiles       — SMILES after "Answer:"
  parse_ok            — True if all essential fields extracted
"""

from typing import Optional

from .._parse_utils import (
    extract_formal_line,
    extract_quoted_value,
    extract_bool_field,
    extract_bracket_list,
    extract_answer,
)


def parse_one(record: dict) -> dict:
    """Parse raw_output for one record. Returns updated dict with parsed fields."""
    text = record.get("raw_output", "") or ""

    # Extract FORMAL lines for each step
    s1 = extract_formal_line(text, 1)
    s2 = extract_formal_line(text, 2)
    s3 = extract_formal_line(text, 3)
    s4 = extract_formal_line(text, 4)
    s5 = extract_formal_line(text, 5)
    s6 = extract_formal_line(text, 6)

    step1_fg_list   = extract_bracket_list(s1, "FG_LIST")       if s1 else None
    step2_rxn_type  = extract_quoted_value(s2, "RXN_TYPE")       if s2 else None
    step3_mechanism = extract_quoted_value(s3, "MECHANISM_KWORD") if s3 else None
    step4_predicted = extract_quoted_value(s4, "PREDICTED_SMILES") if s4 else None
    step5_mol_valid = extract_bool_field(s5, "MOL_VALID")         if s5 else None
    step6_bond      = extract_quoted_value(s6, "BOND_FORMED")     if s6 else None
    answer          = extract_answer(text)

    fg_list_str = "; ".join(step1_fg_list) if step1_fg_list else ""

    parse_ok = bool(
        step1_fg_list is not None
        and step2_rxn_type
        and step3_mechanism
        and step4_predicted
        and answer
    )

    parsed = dict(record)
    parsed.update({
        "step1_line": s1 or "",
        "step2_line": s2 or "",
        "step3_line": s3 or "",
        "step4_line": s4 or "",
        "step5_line": s5 or "",
        "step6_line": s6 or "",
        "step1_fg_list":      step1_fg_list  or [],
        "step1_fg_list_str":  fg_list_str,
        "step2_rxn_type":     step2_rxn_type  or "",
        "step3_mechanism":    step3_mechanism or "",
        "step4_predicted_smi": step4_predicted or "",
        "step5_mol_valid":    step5_mol_valid or "",
        "step6_bond_formed":  step6_bond      or "",
        "answer_smiles":      answer          or "",
        "parse_ok":           parse_ok,
    })
    return parsed


def parse_all(records: list[dict]) -> list[dict]:
    """Parse all records. Returns the updated list."""
    n_ok = 0
    for i, rec in enumerate(records):
        records[i] = parse_one(rec)
        if records[i]["parse_ok"]:
            n_ok += 1
    print(f"parse_all: {n_ok}/{len(records)} parse_ok")
    return records
