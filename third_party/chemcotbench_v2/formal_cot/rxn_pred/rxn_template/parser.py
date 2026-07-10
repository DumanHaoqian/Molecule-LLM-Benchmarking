"""
Parser for rxn_pred/rxn_template formal A→B CoT output.

Parses Unified Step Format:
  step1_bond_changes    — BOND_CHANGES("...")
  step2_rxn_type        — RXN_TYPE("...")
  step3_mechanism       — MECHANISM_KWORD("...")
  step4_proposed_smarts — PROPOSED_SMARTS("...")
  step5_smarts_parseable — SMARTS_PARSEABLE(yes/no)
  step6_selected_option — SELECTED_OPTION("...")
  answer_letter         — letter after "Answer:"
  parse_ok              — True if all essential fields extracted
"""

import re

from .._parse_utils import (
    extract_formal_line,
    extract_quoted_value,
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

    # step1 has two SMILES values and BOND_CHANGES
    step1_bond_changes = ""
    if s1:
        m = re.search(r'BOND_CHANGES\("([^"]*)"\)', s1)
        if m:
            step1_bond_changes = m.group(1).strip()

    step2_rxn_type  = extract_quoted_value(s2, "RXN_TYPE")       if s2 else None
    step3_mechanism = extract_quoted_value(s3, "MECHANISM_KWORD") if s3 else None
    step4_smarts    = extract_quoted_value(s4, "PROPOSED_SMARTS") if s4 else None
    step5_parseable = extract_bool_field(s5, "SMARTS_PARSEABLE")  if s5 else None
    step6_selected  = extract_quoted_value(s6, "SELECTED_OPTION") if s6 else None
    answer          = extract_answer(text)

    # Normalise selected option to uppercase single letter
    if step6_selected:
        step6_selected = step6_selected.strip().upper()
        if len(step6_selected) > 1:
            step6_selected = step6_selected[0]

    parse_ok = bool(
        step1_bond_changes
        and step2_rxn_type
        and step4_smarts
        and step6_selected
        and answer
    )

    parsed = dict(record)
    parsed.update({
        "step1_bond_changes":     step1_bond_changes,
        "step2_rxn_type":         step2_rxn_type or "",
        "step3_mechanism":        step3_mechanism or "",
        "step4_proposed_smarts":  step4_smarts or "",
        "step5_smarts_parseable": step5_parseable or "",
        "step6_selected_option":  step6_selected or "",
        "answer_letter":          answer or "",
        "parse_ok":               parse_ok,
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
