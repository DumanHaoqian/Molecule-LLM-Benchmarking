"""
Parser for rxn_pred/nepp formal A→B CoT output.

Parses Unified Step Format:
  step1_reactants_smi   — SMILES from SMILES("...") in step1
  step1_net_charge      — integer from CHARGE(net_charge=N) in step1
  step2_elem_mech       — ELEM_MECH("...") string from step2
  step3_bond_change     — BOND_CHANGE("...") string from step3
  step4_predicted_smi   — PREDICTED_SMILES("...") from step4
  step5_product_charge  — integer from CHARGE(net_charge=N) in step5
  step6_charge_balanced — "yes"/"no" from CHARGE_BALANCED(yes/no) in step6
  step7_atom_conserved  — "yes"/"no" from ATOM_CONSERVED(yes/no) in step7
  answer_smiles         — SMILES after "Answer:"
  parse_ok              — True if all essential fields extracted
"""

import re

from .._parse_utils import (
    extract_formal_line,
    extract_quoted_value,
    extract_bool_field,
    extract_answer,
)


def _extract_charge_value(formal_line: str) -> int | None:
    """Extract integer from CHARGE(net_charge=N)."""
    if not formal_line:
        return None
    m = re.search(r'CHARGE\(net_charge\s*=\s*([+-]?\d+)\)', formal_line)
    if m:
        return int(m.group(1))
    return None


def _extract_second_charge_value(formal_line: str) -> int | None:
    """Extract the SECOND CHARGE(net_charge=N) occurrence."""
    if not formal_line:
        return None
    matches = list(re.finditer(r'CHARGE\(net_charge\s*=\s*([+-]?\d+)\)', formal_line))
    if len(matches) >= 2:
        return int(matches[1].group(1))
    return None


def _extract_smiles_from_formal(formal_line: str) -> str | None:
    """Extract first SMILES(\"...\") from a FORMAL line."""
    if not formal_line:
        return None
    m = re.search(r'SMILES\("([^"]*)"\)', formal_line)
    if m:
        return m.group(1).strip()
    return None


def parse_one(record: dict) -> dict:
    text = record.get("raw_output", "") or ""

    s1 = extract_formal_line(text, 1)
    s2 = extract_formal_line(text, 2)
    s3 = extract_formal_line(text, 3)
    s4 = extract_formal_line(text, 4)
    s5 = extract_formal_line(text, 5)
    s6 = extract_formal_line(text, 6)
    s7 = extract_formal_line(text, 7)

    step1_reactants_smi = _extract_smiles_from_formal(s1) if s1 else None
    step1_net_charge    = _extract_charge_value(s1)       if s1 else None
    step2_elem_mech     = extract_quoted_value(s2, "ELEM_MECH")      if s2 else None
    step3_bond_change   = extract_quoted_value(s3, "BOND_CHANGE")    if s3 else None
    step4_predicted_smi = extract_quoted_value(s4, "PREDICTED_SMILES") if s4 else None
    step5_product_charge = _extract_charge_value(s5)      if s5 else None
    step6_charge_bal    = extract_bool_field(s6, "CHARGE_BALANCED")  if s6 else None
    step7_atom_cons     = extract_bool_field(s7, "ATOM_CONSERVED")   if s7 else None
    answer              = extract_answer(text)

    # Fallback: if step5 line didn't parse charge, try second charge on step6 line
    if step5_product_charge is None and s6:
        step5_product_charge = _extract_second_charge_value(s6)

    parse_ok = bool(
        step1_reactants_smi
        and step2_elem_mech
        and step4_predicted_smi
        and step6_charge_bal is not None
        and step7_atom_cons is not None
        and answer
    )

    parsed = dict(record)
    parsed.update({
        "step1_reactants_smi":   step1_reactants_smi or "",
        "step1_net_charge":      step1_net_charge,
        "step2_elem_mech":       step2_elem_mech or "",
        "step3_bond_change":     step3_bond_change or "",
        "step4_predicted_smi":   step4_predicted_smi or "",
        "step5_product_charge":  step5_product_charge,
        "step6_charge_balanced": step6_charge_bal or "",
        "step7_atom_conserved":  step7_atom_cons or "",
        "answer_smiles":         answer or "",
        "parse_ok":              parse_ok,
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
