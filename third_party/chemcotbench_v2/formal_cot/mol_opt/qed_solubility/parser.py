"""
Parser for mol_opt/logp formal A→B CoT output (Unified Step Format).

Parses Gemini's raw text output into structured fields:
  step1_scaffold_smiles       — SCAFFOLD_SMILES from Step 1 [SCAFFOLD_IDENTIFICATION]
  step2_fg_removed            — remove="..." from Step 2 [EDIT_PLAN]
  step2_fg_added              — add="..."    from Step 2 [EDIT_PLAN]
  step3_predicted_smiles      — PREDICTED_SMILES from Step 3 [PRODUCT_CONSTRUCTION]
  step4_scaffold_claimed      — "yes"/"no" from Step 4 [SCAFFOLD_PRESERVATION]
  step5_fg_consistent_claimed — "yes"/"no" from Step 5 [FG_CHANGE_VERIFICATION]
  answer_smiles               — SMILES after "Answer:"
  parse_ok                    — True if all essential fields extracted

Backward compatibility: also parses old dual-part format (step1:/step2:...).
"""

import re
from typing import Optional


# ── Low-level field extractors ───────────────────────────────────────────────

def _extract_smiles_from_tag(text: str, tag: str) -> Optional[str]:
    """Extract SMILES from TAG("...") pattern."""
    pattern = re.compile(rf'{re.escape(tag)}\("(.*?)"\)', re.DOTALL)
    m = pattern.search(text)
    if m:
        return m.group(1).strip()
    return None


def _extract_scaffold_smiles(step_text: str) -> Optional[str]:
    """Extract SCAFFOLD_SMILES from Step 1 formal line."""
    return _extract_smiles_from_tag(step_text, "SCAFFOLD_SMILES")


def _extract_edit_plan(step_text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract fg_removed and fg_added from EDIT_PLAN(remove="..."; add="...")."""
    m_remove = re.search(r'remove="([^"]*)"', step_text)
    m_add    = re.search(r'add="([^"]*)"',    step_text)
    fg_removed = m_remove.group(1).strip() if m_remove else None
    fg_added   = m_add.group(1).strip()    if m_add    else None
    return fg_removed, fg_added


def _extract_predicted_smiles(step_text: str) -> Optional[str]:
    """Extract PREDICTED_SMILES from Step 3 formal line."""
    return _extract_smiles_from_tag(step_text, "PREDICTED_SMILES")


def _extract_bool_field(step_text: str, tag: str) -> Optional[str]:
    """Extract yes/no from TAG(yes/no) pattern. Returns lowercase string."""
    m = re.search(rf'{re.escape(tag)}\((yes|no)\)', step_text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


def _extract_answer(text: str) -> Optional[str]:
    """Extract SMILES from 'Answer: <smiles>' line."""
    m = re.search(r'(?i)^Answer\s*:\s*(\S+)', text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return None


# ── Step text extraction ─────────────────────────────────────────────────────

def _extract_steps_unified(text: str) -> dict[int, str]:
    """Extract steps from Unified Step Format: Step N [NAME]: ... FORMAL: ..."""
    steps = {}
    # Match Step N [NAME]: ... followed by FORMAL line, until next Step or EOF
    pattern = re.compile(
        r'Step\s*(\d+)\s*\[([^\]]+)\]:.*?FORMAL:\s*(.+?)(?=\n\s*Step|\Z)',
        re.DOTALL | re.IGNORECASE
    )
    for m in pattern.finditer(text):
        step_num = int(m.group(1))
        step_text = m.group(3).strip()
        steps[step_num] = step_text
    return steps


def _extract_step_blocks_unified(text: str) -> list[str]:
    """Extract full step text blocks (including Step N [NAME]: ... FORMAL: ...) from unified format."""
    blocks = []
    pattern = re.compile(
        r'(Step\s*\d+\s*\[[^\]]+\]:.*?)(?=^\s*Step|\Z)',
        re.DOTALL | re.IGNORECASE | re.MULTILINE
    )
    for m in pattern.finditer(text):
        blocks.append(m.group(1).strip())
    return blocks

def _extract_steps_old(text: str) -> dict[int, str]:
    """Extract steps from old format: stepN: ... (single line).

    Old format has 6 steps: step1-6
    New format has 5 steps: Step 1-5
    Mapping: old step4 (MOL_VALID) is merged into step3;
             old step5 -> new step4; old step6 -> new step5.
    """
    steps = {}
    # Try to find Part 2 section first
    part2_start = None
    m = re.search(r'(?i)part\s*2', text)
    if m:
        part2_start = m.start()
    else:
        # Fallback: first line with "step1:"
        m2 = re.search(r'^step1:', text, re.MULTILINE)
        if m2:
            part2_start = m2.start()

    search_text = text[part2_start:] if part2_start is not None else text

    # Extract old steps 1-6
    old_steps = {}
    for step_num in range(1, 7):
        pattern = re.compile(rf'^step{step_num}\s*:\s*(.+?)(?=\n\s*step|\Z)',
                             re.MULTILINE | re.DOTALL | re.IGNORECASE)
        m = pattern.search(search_text)
        if m:
            old_steps[step_num] = m.group(1).strip()

    # Map old steps to new steps
    steps[1] = old_steps.get(1, "")
    steps[2] = old_steps.get(2, "")
    steps[3] = old_steps.get(3, "")
    steps[4] = old_steps.get(5, "")  # old step5 -> new step4 (SCAFFOLD_PRESERVED)
    steps[5] = old_steps.get(6, "")  # old step6 -> new step5 (FG_CHANGE_CONSISTENT)
    return steps


# ── Normalisation ────────────────────────────────────────────────────────────

_NONE_VARIANTS = {"none", "n/a", "na", "-", ""}

def _norm_none(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    return "none" if v.strip().lower() in _NONE_VARIANTS else v.strip()


# ── Single-record parser ─────────────────────────────────────────────────────

def parse_one(record: dict) -> dict:
    """Parse raw_output for one record. Returns updated dict with parsed fields."""
    text = record.get("raw_output", "") or ""

    # Try unified format first
    steps = _extract_steps_unified(text)
    format_type = "unified"

    # Fallback to old format
    if not steps:
        steps = _extract_steps_old(text)
        format_type = "old"

    # Extract fields from step texts
    step1 = steps.get(1, "")
    step2 = steps.get(2, "")
    step3 = steps.get(3, "")
    step4 = steps.get(4, "")
    step5 = steps.get(5, "")

    step1_scaffold = _extract_scaffold_smiles(step1) if step1 else None
    step2_removed, step2_added = _extract_edit_plan(step2) if step2 else (None, None)
    step3_predicted = _extract_predicted_smiles(step3) if step3 else None
    step4_scaffold = _extract_bool_field(step4, "SCAFFOLD_PRESERVED") if step4 else None
    step5_fg = _extract_bool_field(step5, "FG_CHANGE_CONSISTENT") if step5 else None
    answer = _extract_answer(text)

    # Normalise "none" variants
    step2_removed = _norm_none(step2_removed)
    step2_added = _norm_none(step2_added)

    # parse_ok: essential fields present
    parse_ok = bool(
        step1_scaffold
        and step2_removed is not None
        and step2_added is not None
        and step3_predicted
        and step4_scaffold is not None
        and step5_fg is not None
        and answer
    )

    parsed = dict(record)
    # Extract full step blocks for readability
    raw_output_steps = _extract_step_blocks_unified(text)
    
    parsed.update({
        # step lines (for debugging)
        "step1_text": step1,
        "step2_text": step2,
        "step3_text": step3,
        "step4_text": step4,
        "step5_text": step5,
        "raw_output_steps": raw_output_steps,
        # extracted fields
        "step1_scaffold_smiles": step1_scaffold or "",
        "step2_fg_removed": step2_removed or "",
        "step2_fg_added": step2_added or "",
        "step3_predicted_smiles": step3_predicted or "",
        "step4_scaffold_claimed": step4_scaffold or "",
        "step5_fg_consistent_claimed": step5_fg or "",
        "answer_smiles": answer or "",
        "parse_ok": parse_ok,
        "parse_format_type": format_type,
    })
    return parsed


# ── Batch parser ─────────────────────────────────────────────────────────────

def parse_all(records: list[dict]) -> list[dict]:
    """Parse all records in place. Returns the updated list."""
    n_ok = 0
    fmt_counts = {"unified": 0, "old": 0, "failed": 0}
    for i, rec in enumerate(records):
        records[i] = parse_one(rec)
        if records[i]["parse_ok"]:
            n_ok += 1
        fmt = records[i].get("parse_format_type", "failed")
        if fmt in fmt_counts:
            fmt_counts[fmt] += 1
        else:
            fmt_counts["failed"] += 1
    print(f"parse_all: {n_ok}/{len(records)} parse_ok (unified={fmt_counts['unified']}, old={fmt_counts['old']}, failed={fmt_counts['failed']})")
    return records
