"""
Parser for rxn_pred/rcr_reagent formal A→B CoT output (unified step format).

Extracts the 8-step formal verification chain from FORMAL lines.
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


_VALID_SLOTS = {
    "hydrogen source",
    "hydride reagent",
    "base system",
    "activation system",
    "acidic medium",
    "solvent/medium",
    "salt/additive",
    "oxidant",
    "organometallic reagent",
    "other",
}

_VALID_COMPONENTS = {"single", "multi"}

_VALID_CLASSES = {
    "hydrogen gas",
    "transfer hydrogen donor",
    "hydride reductant",
    "inorganic carbonate base",
    "inorganic phosphate base",
    "hydroxide base",
    "organic amine base",
    "fluoride salt/additive",
    "halide salt/additive",
    "acidic additive",
    "carbodiimide activator",
    "acyl/sulfonyl activating reagent",
    "mixed base system",
    "mixed activation system",
    "solvent",
    "organometallic reagent",
    "oxidant",
    "other",
}


def _normalise_slot(text: str | None) -> str:
    if not text:
        return ""
    t = text.strip().rstrip(".,;:").strip('"\'')
    tl = t.lower()
    if "hydride" in tl:
        return "hydride reagent"
    if "hydrogen" in tl:
        return "hydrogen source"
    if "base" in tl:
        return "base system"
    if "activ" in tl:
        return "activation system"
    if "acid" in tl:
        return "acidic medium"
    if "salt" in tl or "additive" in tl or "fluoride" in tl or "halide" in tl:
        return "salt/additive"
    if "solvent" in tl or "medium" in tl:
        return "solvent/medium"
    if "oxid" in tl:
        return "oxidant"
    if "organomet" in tl or "grignard" in tl or "organolith" in tl:
        return "organometallic reagent"
    if tl == "other":
        return "other"
    return t


def _normalise_component(text: str | None) -> str:
    if not text:
        return ""
    t = text.strip().lower().rstrip(".,;:").strip('"\'')
    if t in {"single", "one", "single-component", "single component"}:
        return "single"
    if t in {"multi", "multiple", "multi-component", "multicomponent", "multi component"}:
        return "multi"
    return ""


def _normalise_class(text: str | None) -> str:
    if not text:
        return ""
    t = text.strip().rstrip(".,;:").strip('"\'')
    tl = t.lower()
    if "transfer" in tl and "hydrogen" in tl:
        return "transfer hydrogen donor"
    if tl in {"hydrogen gas", "h2", "molecular hydrogen"} or ("hydrogen" in tl and "gas" in tl):
        return "hydrogen gas"
    if "hydride" in tl:
        return "hydride reductant"
    if "carbonate" in tl:
        return "inorganic carbonate base"
    if "phosphate" in tl:
        return "inorganic phosphate base"
    if "hydroxide" in tl:
        return "hydroxide base"
    if "amine" in tl and "mixed" not in tl:
        return "organic amine base"
    if "fluoride" in tl:
        return "fluoride salt/additive"
    if "halide" in tl:
        return "halide salt/additive"
    if "acid" in tl:
        return "acidic additive"
    if "oxid" in tl:
        return "oxidant"
    if "carbodiimide" in tl:
        return "carbodiimide activator"
    if "sulfonyl" in tl or "acyl/" in tl or "acyl" in tl:
        return "acyl/sulfonyl activating reagent"
    if "mixed base" in tl:
        return "mixed base system"
    if "mixed activation" in tl:
        return "mixed activation system"
    if "organomet" in tl or "grignard" in tl or "organolith" in tl:
        return "organometallic reagent"
    if "solvent" in tl or "medium" in tl:
        return "solvent"
    if tl == "other":
        return "other"
    return t


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
            "step2_core_transformation": "", "step3_reagent_slot": "",
            "step4_reagent_strategy": "", "step5_component_mode": "",
            "step6_reagent_class": "", "step7_predicted_reagent_smiles": "",
            "step8_self_consistent": "", "answer_smiles": "",
            "parse_ok": False, "raw_output_steps": [],
        })
        return parsed

    formal_block = _isolate_formal_lines(text)

    step1_rxn_class = _extract_quoted_value(formal_block, "RXN_CLASS")
    step2_rxn_smiles = _extract_quoted_value(formal_block, "RXN_SMILES")
    step2_core_transformation = _extract_quoted_value(formal_block, "CORE_TRANSFORMATION")
    step3_slot_raw = _extract_quoted_value(formal_block, "REAGENT_SLOT")
    step3_reagent_slot = _normalise_slot(step3_slot_raw)
    step4_reagent_strategy = _extract_quoted_value(formal_block, "REAGENT_STRATEGY")
    step5_component_raw = _extract_quoted_value(formal_block, "COMPONENT_MODE")
    step5_component_mode = _normalise_component(step5_component_raw)
    step6_class_raw = _extract_quoted_value(formal_block, "REAGENT_CLASS")
    step6_reagent_class = _normalise_class(step6_class_raw)
    step7_pred_smiles = _extract_quoted_value(formal_block, "PREDICTED_REAGENT_SMILES")
    step8_self_raw = _extract_quoted_value(formal_block, "SELF_CONSISTENT")
    step8_self_consistent = _normalise_yes_no(step8_self_raw)
    answer_smiles = _extract_answer(text)

    parse_ok = bool(
        step1_rxn_class
        and step2_rxn_smiles
        and step2_core_transformation
        and step3_reagent_slot in _VALID_SLOTS
        and step4_reagent_strategy
        and step5_component_mode in _VALID_COMPONENTS
        and step6_reagent_class in _VALID_CLASSES
        and step7_pred_smiles
        and step8_self_consistent in {"yes", "no"}
        and answer_smiles
    )

    parsed.update({
        "step1_rxn_class": step1_rxn_class or "",
        "step2_rxn_smiles": step2_rxn_smiles or "",
        "step2_core_transformation": step2_core_transformation or "",
        "step3_reagent_slot": step3_reagent_slot or "",
        "step4_reagent_strategy": step4_reagent_strategy or "",
        "step5_component_mode": step5_component_mode or "",
        "step6_reagent_class": step6_reagent_class or "",
        "step7_predicted_reagent_smiles": step7_pred_smiles or "",
        "step8_self_consistent": step8_self_consistent or "",
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
