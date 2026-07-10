"""
Parser for condition_ranking top-3 ranking CoT responses (6-step format).

Extracts:
  step1_rxn_class       : e.g. "C-C Coupling"
  step2_decision_factor : e.g. "catalyst"
  step3_pair_diffs      : dict {"1/2": ["catalyst","ligand"], "1/3": ["catalyst"], "2/3": ["catalyst","ligand"]}
  step4_pairwise_prefs  : list of preferences, e.g. ["1>2","1>3","2>3"]
  step5_ranking         : list e.g. ["1","2","3"]
  step6_top2_support    : dict {"winner": "1", "loser": "2", "field": "catalyst"}
  answer                : list e.g. ["1","2","3"]
  parse_ok              : True/False
  parse_errors          : list of error strings
"""
import re
import json


# ──────────────────────────────────────────────────────────────────────────────
# Helper patterns
# ──────────────────────────────────────────────────────────────────────────────

def _find_formal(text: str, tag: str) -> str | None:
    """Return the content of  FORMAL: ... --> ...  that contains `tag`."""
    for line in text.splitlines():
        if line.strip().startswith("FORMAL:") and tag in line:
            return line.strip()[len("FORMAL:"):].strip()
    return None


def _quoted(s: str) -> str:
    """Extract the content of the first double-quoted substring."""
    m = re.search(r'"([^"]+)"', s)
    return m.group(1) if m else s.strip()


def _parse_pair_diffs_str(s: str) -> dict[str, list[str]]:
    """
    Parse PAIR_DIFFS value string, e.g.:
      1/2:catalyst+ligand; 1/3:catalyst; 2/3:ligand
    Returns dict:  {"1/2": ["catalyst","ligand"], "1/3": ["catalyst"], "2/3": ["ligand"]}
    """
    result: dict[str, list[str]] = {}
    parts = [p.strip() for p in s.split(";") if p.strip()]
    for part in parts:
        if ":" not in part:
            continue
        pair_key, fields_str = part.split(":", 1)
        pair_key = pair_key.strip().strip('"')
        fields = [f.strip().strip('"') for f in fields_str.split("+") if f.strip()]
        result[pair_key] = fields
    return result


def _parse_pairwise_prefs_str(s: str) -> list[str]:
    """Parse PAIRWISE_PREFS value string: '1>2; 1>3; 2>3' -> ['1>2','1>3','2>3']"""
    return [p.strip().strip('"') for p in s.split(";") if p.strip() and ">" in p]


def _extract_step_formal_output(text: str, step_tag: str) -> str | None:
    """Find the FORMAL line for Step N [step_tag] and return the --> RHS."""
    pattern = re.compile(
        rf"Step\s+\d+\s+\[{re.escape(step_tag)}\].*?\n\s+FORMAL:.*?-->\s*(.+)",
        re.IGNORECASE,
    )
    m = pattern.search(text)
    if m:
        return m.group(1).strip()
    return None


def _extract_formal_value(formal_rhs: str, outer_tag: str) -> str | None:
    """
    From a FORMAL RHS like  RXN_CLASS("C-C Coupling")  extract the inner value.
    Returns None if outer_tag not found.
    NOTE: returns raw inner content without stripping quotes, so callers that
    want a clean simple value should do their own .strip('"').
    """
    pattern = re.compile(rf'{re.escape(outer_tag)}\(([^)]*)\)', re.IGNORECASE)
    m = pattern.search(formal_rhs)
    if m:
        return m.group(1).strip()
    return None


def _extract_json_list(text: str) -> list[str] | None:
    """Find and parse the first JSON-style list in text."""
    m = re.search(r'\[([^\]]+)\]', text)
    if not m:
        return None
    inner = m.group(1)
    items = [x.strip().strip('"').strip("'") for x in inner.split(",")]
    return items if items else None


# ──────────────────────────────────────────────────────────────────────────────
# Main parse function
# ──────────────────────────────────────────────────────────────────────────────

def parse_record(raw_output: str) -> dict:
    text = raw_output.strip()
    errors: list[str] = []
    result: dict = {
        "step1_rxn_class": None,
        "step2_decision_factor": None,
        "step3_pair_diffs": None,
        "step4_pairwise_prefs": None,
        "step5_ranking": None,
        "step6_top2_support": None,
        "answer": None,
        "parse_ok": False,
        "parse_errors": [],
    }

    # ── Step 1: RXN_CLASS ────────────────────────────────────────────────────
    rhs1 = _extract_step_formal_output(text, "RXN_CLASS")
    if rhs1:
        val = _extract_formal_value(rhs1, "RXN_CLASS")
        if val:
            result["step1_rxn_class"] = val.strip('"')
        else:
            errors.append("step1: cannot extract RXN_CLASS value from FORMAL")
    else:
        errors.append("step1: FORMAL line for [RXN_CLASS] not found")

    # ── Step 2: DECISION_FACTOR ──────────────────────────────────────────────
    rhs2 = _extract_step_formal_output(text, "DECISION_FACTOR")
    if rhs2:
        val = _extract_formal_value(rhs2, "DECISION_FACTOR")
        if val:
            result["step2_decision_factor"] = val.strip('"')
        else:
            errors.append("step2: cannot extract DECISION_FACTOR value from FORMAL")
    else:
        errors.append("step2: FORMAL line for [DECISION_FACTOR] not found")

    # ── Step 3: PAIR_DIFFS ───────────────────────────────────────────────────
    rhs3 = _extract_step_formal_output(text, "PAIR_DIFFS")
    if rhs3:
        val = _extract_formal_value(rhs3, "PAIR_DIFFS")
        if val:
            result["step3_pair_diffs"] = _parse_pair_diffs_str(val)
        else:
            errors.append("step3: cannot extract PAIR_DIFFS value from FORMAL")
    else:
        errors.append("step3: FORMAL line for [PAIR_DIFFS] not found")

    # ── Step 4: PAIRWISE_PREFS ───────────────────────────────────────────────
    rhs4 = _extract_step_formal_output(text, "PAIRWISE_PREFS")
    if rhs4:
        val = _extract_formal_value(rhs4, "PAIRWISE_PREFS")
        if val:
            result["step4_pairwise_prefs"] = _parse_pairwise_prefs_str(val)
        else:
            errors.append("step4: cannot extract PAIRWISE_PREFS value from FORMAL")
    else:
        errors.append("step4: FORMAL line for [PAIRWISE_PREFS] not found")

    # ── Step 5: RANKING ──────────────────────────────────────────────────────
    rhs5 = _extract_step_formal_output(text, "RANKING")
    if rhs5:
        val = _extract_formal_value(rhs5, "RANKING")
        if val:
            ranking = _extract_json_list(val)
            if ranking:
                result["step5_ranking"] = ranking
            else:
                errors.append("step5: cannot parse RANKING list")
        else:
            errors.append("step5: cannot extract RANKING value from FORMAL")
    else:
        errors.append("step5: FORMAL line for [RANKING] not found")

    # ── Step 6: TOP2_SUPPORT ─────────────────────────────────────────────────
    rhs6 = _extract_step_formal_output(text, "TOP2_SUPPORT")
    if rhs6:
        val = _extract_formal_value(rhs6, "TOP2_SUPPORT")
        if val:
            winner_m = re.search(r'WINNER="([^"]+)"', val)
            loser_m  = re.search(r'LOSER="([^"]+)"', val)
            field_m  = re.search(r'FIELD="([^"]+)"', val)
            result["step6_top2_support"] = {
                "winner": winner_m.group(1) if winner_m else None,
                "loser":  loser_m.group(1)  if loser_m  else None,
                "field":  field_m.group(1)  if field_m  else None,
            }
        else:
            errors.append("step6: cannot extract TOP2_SUPPORT value from FORMAL")
    else:
        errors.append("step6: FORMAL line for [TOP2_SUPPORT] not found")

    # ── Answer ───────────────────────────────────────────────────────────────
    ans_m = re.search(r'^Answer:\s*(\[[^\]]+\])', text, re.MULTILINE)
    if ans_m:
        try:
            answer = json.loads(ans_m.group(1))
            if isinstance(answer, list):
                result["answer"] = [str(x) for x in answer]
        except json.JSONDecodeError:
            errors.append("answer: JSON parse failed")
    else:
        errors.append("answer: 'Answer:' line not found")

    result["parse_ok"] = len(errors) == 0
    result["parse_errors"] = errors
    return result


def parse_all(records: list[dict]) -> list[dict]:
    """Parse raw_output for each record; add parsed fields in-place."""
    out = []
    for rec in records:
        parsed = parse_record(rec.get("raw_output", ""))
        merged = {**rec, **parsed}
        out.append(merged)
    return out
