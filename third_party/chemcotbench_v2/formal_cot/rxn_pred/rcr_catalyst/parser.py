"""
Parser for rxn_pred/rcr_catalyst formal A→B CoT output.

Parses Unified Step Format:
  step1_rxn_cls        — RXN_CLASS("...")
  step2_core_transform — CORE_TRANSFORMATION("...")
  step3_catalyst_role  — CATALYST_ROLE("...")
  step4_catalyst_class — CATALYST_CLASS("...")
  step5_predicted_smi  — PREDICTED_CATALYST_SMILES("...")
  step6_self_consistent— SELF_CONSISTENT("yes|no")
  answer_smiles        — SMILES after "Answer:"
  parse_ok             — True if all essential fields extracted
"""

from .._parse_utils import (
    extract_formal_line,
    extract_quoted_value,
    extract_bool_field,
    extract_answer,
)

_VALID_CLASSES = {
    "Pd", "Fe", "Cu", "Ni", "Zn", "Pt", "Ti", "Os", "Mn", "Rh", "Ag",
    "acid", "base", "organocatalyst", "aprotic activator", "crown ether", "other",
}


def _normalise_class(text: str | None) -> str:
    if not text:
        return ""
    t = text.strip().rstrip(".,;:").strip('"\'')
    tl = t.lower()
    for metal in ("Pd", "Fe", "Cu", "Ni", "Zn", "Pt", "Ti", "Os", "Mn", "Rh", "Ag"):
        if tl == metal.lower():
            return metal
    if "crown" in tl:
        return "crown ether"
    if "aprotic" in tl or "amide activator" in tl or "polar aprotic" in tl:
        return "aprotic activator"
    if "organocatal" in tl or "organic catalyst" in tl:
        return "organocatalyst"
    if "acid" in tl:
        return "acid"
    if "base" in tl:
        return "base"
    if tl == "other":
        return "other"
    return t


def parse_one(record: dict) -> dict:
    text = record.get("raw_output", "") or ""

    s1 = extract_formal_line(text, 1)
    s2 = extract_formal_line(text, 2)
    s3 = extract_formal_line(text, 3)
    s4 = extract_formal_line(text, 4)
    s5 = extract_formal_line(text, 5)
    s6 = extract_formal_line(text, 6)

    step1_rxn_cls   = extract_quoted_value(s1, "RXN_CLASS")              if s1 else None
    step2_transform = extract_quoted_value(s2, "CORE_TRANSFORMATION")    if s2 else None
    step3_role      = extract_quoted_value(s3, "CATALYST_ROLE")          if s3 else None
    step4_class_raw = extract_quoted_value(s4, "CATALYST_CLASS")         if s4 else None
    step4_class     = _normalise_class(step4_class_raw)
    step5_predicted = extract_quoted_value(s5, "PREDICTED_CATALYST_SMILES") if s5 else None
    step6_consistent = extract_bool_field(s6, "SELF_CONSISTENT")          if s6 else None
    answer          = extract_answer(text)

    parse_ok = bool(
        step1_rxn_cls
        and step2_transform
        and step3_role
        and step4_class in _VALID_CLASSES
        and step5_predicted
        and step6_consistent in {"yes", "no"}
        and answer
    )

    parsed = dict(record)
    parsed.update({
        "step1_rxn_cls":        step1_rxn_cls or "",
        "step2_core_transform": step2_transform or "",
        "step3_catalyst_role":  step3_role or "",
        "step4_catalyst_class": step4_class or "",
        "step5_predicted_smi":  step5_predicted or "",
        "step6_self_consistent": step6_consistent or "",
        "answer_smiles":        answer or "",
        "parse_ok":             parse_ok,
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
