"""Parser for MolOpt dual-output format (Part A structured + Part B step chain)."""
import re


# ─── Part A: structured section parser ────────────────────────────────────────

_SECTION_PATTERNS = {
    "scaffold_smiles": re.compile(
        r'\[SCAFFOLD_IDENTIFICATION\].*?Scaffold SMILES:\s*([^\n]+)', re.DOTALL | re.IGNORECASE
    ),
    "fg_removed": re.compile(
        r'\[EDIT_PLAN\].*?FG Removed:\s*([^\n]+)', re.DOTALL | re.IGNORECASE
    ),
    "fg_added": re.compile(
        r'\[EDIT_PLAN\].*?FG Added:\s*([^\n]+)', re.DOTALL | re.IGNORECASE
    ),
    "predicted_smiles": re.compile(
        r'\[PRODUCT_CONSTRUCTION\].*?Predicted SMILES:\s*([^\n]+)', re.DOTALL | re.IGNORECASE
    ),
    "answer_smiles": re.compile(
        r'\[PRODUCT_CONSTRUCTION\].*?Answer:\s*([^\n]+)', re.DOTALL | re.IGNORECASE
    ),
    "scaffold_preserved": re.compile(
        r'\[SCAFFOLD_PRESERVATION\].*?Scaffold Preserved:\s*(yes|no)', re.DOTALL | re.IGNORECASE
    ),
    "fg_consistent": re.compile(
        r'\[FG_CHANGE_VERIFICATION\].*?FG Change Consistent:\s*(yes|no)', re.DOTALL | re.IGNORECASE
    ),
}


def _extract_part_a(text: str) -> dict:
    """Extract structured fields from Part A."""
    result = {}
    for key, pattern in _SECTION_PATTERNS.items():
        m = pattern.search(text)
        result[key] = m.group(1).strip() if m else ""
    return result


# ─── Part B: Unified Step Format parser ───────────────────────────────────────

_STEP_PATTERNS = {
    "step1_scaffold_smiles": re.compile(
        r'SCAFFOLD_SMILES\s*\(\s*"([^"]*)"\s*\)', re.IGNORECASE
    ),
    "step2_fg_removed": re.compile(
        r'EDIT_PLAN\s*\(\s*remove\s*=\s*"([^"]*)"', re.IGNORECASE
    ),
    "step2_fg_added": re.compile(
        r'EDIT_PLAN\s*\([^)]*add\s*=\s*"([^"]*)"', re.IGNORECASE
    ),
    "step3_predicted_smiles": re.compile(
        r'PREDICTED_SMILES\s*\(\s*"([^"]*)"\s*\)', re.IGNORECASE
    ),
    "step4_scaffold_claimed": re.compile(
        r'SCAFFOLD_PRESERVED\s*\(\s*(yes|no)\s*\)', re.IGNORECASE
    ),
    "step5_fg_consistent_claimed": re.compile(
        r'FG_CHANGE_CONSISTENT\s*\(\s*(yes|no)\s*\)', re.IGNORECASE
    ),
}

# Also try to extract from raw Answer line
_ANSWER_PATTERN = re.compile(r'^Answer:\s*([^\n]+)', re.MULTILINE | re.IGNORECASE)


def _extract_part_b(text: str) -> dict:
    """Extract formal step fields from Part B."""
    result = {}
    for key, pattern in _STEP_PATTERNS.items():
        m = pattern.search(text)
        result[key] = m.group(1).strip() if m else ""

    # Extract answer line
    m = _ANSWER_PATTERN.search(text)
    result["answer_smiles"] = m.group(1).strip() if m else ""

    # Also extract raw step texts for inspection
    result["raw_output_steps"] = _split_steps(text)
    return result


def _split_steps(text: str) -> list[str]:
    """Split raw output into individual step texts."""
    # Find all Step N [NAME] blocks
    pattern = re.compile(
        r'(Step\s+\d+\s+\[[^\]]+\]:.*?)(?=Step\s+\d+\s+\[|\Z)',
        re.DOTALL | re.IGNORECASE
    )
    steps = [m.group(1).strip() for m in pattern.finditer(text)]

    # Also capture Answer line
    m = _ANSWER_PATTERN.search(text)
    if m:
        steps.append(m.group(0).strip())

    return steps


# ─── Main parse function ──────────────────────────────────────────────────────

def parse_record(record: dict) -> dict:
    """
    Parse a sampled record's raw_output into structured fields.
    Returns updated record with parsed fields.
    """
    raw = record.get("raw_output", "")
    if not raw:
        return {**record, "parse_ok": False, "parse_errors": ["empty raw_output"]}

    # Try to split Part A and Part B
    # Heuristic: look for "PART A" / "PART B" markers, or
    # look for transition from structured output to Step 1
    part_a = raw
    part_b = raw

    # Try explicit part markers
    part_b_marker = re.search(r'PART\s+B|Step\s+1\s+\[', raw, re.IGNORECASE)
    if part_b_marker:
        split_pos = part_b_marker.start()
        part_a = raw[:split_pos]
        part_b = raw[split_pos:]

    # Extract Part A fields
    part_a_fields = _extract_part_a(part_a)

    # Extract Part B fields
    part_b_fields = _extract_part_b(part_b)

    # Merge: Part A takes priority for structured fields, Part B for step fields
    # But we keep both for cross-checking
    merged = dict(record)
    merged.update(part_a_fields)
    merged.update(part_b_fields)

    # Cross-check: Part A predicted_smiles vs Part B step3_predicted_smiles
    cross_errors = []
    if part_a_fields.get("predicted_smiles") and part_b_fields.get("step3_predicted_smiles"):
        if part_a_fields["predicted_smiles"] != part_b_fields["step3_predicted_smiles"]:
            cross_errors.append("predicted_smiles mismatch between Part A and Part B")

    # Determine parse_ok: all critical fields must be present
    critical_part_a = ["predicted_smiles", "answer_smiles"]
    critical_part_b = ["step3_predicted_smiles"]

    has_part_a = all(bool(part_a_fields.get(k)) for k in critical_part_a)
    has_part_b = all(bool(part_b_fields.get(k)) for k in critical_part_b)

    # We require at least Part A structured fields (for Layer 2)
    # and ideally Part B step fields (for Layer 3)
    parse_ok = has_part_a
    if not has_part_a:
        parse_errors = ["missing critical Part A fields"] + cross_errors
    else:
        parse_errors = cross_errors

    merged["parse_ok"] = parse_ok
    merged["parse_has_part_a"] = has_part_a
    merged["parse_has_part_b"] = has_part_b
    merged["parse_errors"] = parse_errors

    return merged


def parse_batch(records: list[dict]) -> list[dict]:
    """Parse a batch of records."""
    results = []
    ok_count = 0
    for rec in records:
        parsed = parse_record(rec)
        results.append(parsed)
        if parsed["parse_ok"]:
            ok_count += 1
    print(f"[parser] {ok_count}/{len(records)} parsed OK  (Part A: {sum(1 for r in results if r.get('parse_has_part_a'))}, Part B: {sum(1 for r in results if r.get('parse_has_part_b'))})")
    return results
