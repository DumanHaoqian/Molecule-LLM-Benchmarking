"""Shared parser utilities for Unified Step Format.

All rxn_pred tasks now use the Unified Step Format:
  Step N [STEP_NAME]: <natural language>
    FORMAL: <INPUT> --> <OUTPUT>

This module provides helpers to extract FORMAL lines and parse common fields.
"""

import re
from typing import Optional


def extract_formal_line(text: str, step_num: int) -> Optional[str]:
    """Extract the FORMAL line for a given step number from Unified Step Format text.

    Returns the content after 'FORMAL: ' (e.g. 'SMILES("...") --> FG_LIST(["..."])')
    or None if the step/formal line is not found.
    """
    # Match Step N [NAME]: ... up to next Step or end of text
    step_pattern = re.compile(
        rf'^Step\s+{step_num}\s+\[.*?\]:(.*?)(?=^Step\s+\d+\s+\[|\Z)',
        re.MULTILINE | re.DOTALL
    )
    m = step_pattern.search(text)
    if not m:
        return None
    block = m.group(1)
    # Find FORMAL line: any leading whitespace (0–4 spaces), then "FORMAL: ..."
    formal_m = re.search(r'^\s*FORMAL:\s*(.*?)$', block, re.MULTILINE)
    if formal_m:
        return formal_m.group(1).strip()
    return None


def extract_all_formal_lines(text: str) -> dict[int, str]:
    """Extract all FORMAL lines keyed by step number."""
    result = {}
    for step_num in range(1, 20):
        line = extract_formal_line(text, step_num)
        if line is not None:
            result[step_num] = line
    return result


def extract_quoted_value(formal_line: str, tag: str) -> Optional[str]:
    """Extract value from TAG("...") pattern in a FORMAL line.

    Handles tags like RXN_TYPE, MECHANISM_KWORD, PREDICTED_SMILES, etc.
    """
    if not formal_line:
        return None
    pattern = re.compile(rf'{re.escape(tag)}\("([^"]*)"\)')
    m = pattern.search(formal_line)
    if m:
        return m.group(1).strip()
    return None


def extract_bool_field(formal_line: str, tag: str) -> Optional[str]:
    """Extract yes/no from TAG(yes/no), TAG("yes"/"no"), or TAG(yes|no) pattern."""
    if not formal_line:
        return None
    # Try unquoted first: SELF_CONSISTENT(yes)
    m = re.search(rf'{re.escape(tag)}\((yes|no)\)', formal_line, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    # Try quoted: SELF_CONSISTENT("yes")
    m = re.search(rf'{re.escape(tag)}\("(yes|no)"\)', formal_line, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


def extract_bracket_list(formal_line: str, tag: str) -> Optional[list[str]]:
    """Extract list of quoted strings from TAG(["a", "b", ...]) pattern.

    Handles nested brackets inside quoted strings (e.g. SMARTS patterns
    like "[#6]=O" that contain ']' characters).
    """
    if not formal_line:
        return None
    start = formal_line.find(f'{tag}(')
    if start == -1:
        return None
    i = formal_line.find('[', start)
    if i == -1:
        return None
    depth = 1
    j = i + 1
    while j < len(formal_line) and depth > 0:
        ch = formal_line[j]
        # Skip characters inside quoted strings
        if ch == '"':
            j += 1
            while j < len(formal_line) and formal_line[j] != '"':
                j += 1
            if j < len(formal_line):
                j += 1
            continue
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
        j += 1
    if depth != 0:
        return None
    rest = formal_line[j:].strip()
    if not rest.startswith(')'):
        return None
    inner = formal_line[i + 1:j - 1]
    quoted = re.findall(r'"([^"]*)"', inner)
    if quoted:
        return [s.strip() for s in quoted if s.strip()]
    parts = [p.strip().strip("'\"") for p in inner.split(",")]
    return [p for p in parts if p]


def extract_answer(text: str) -> Optional[str]:
    """Extract value from 'Answer: <value>' line."""
    m = re.search(r'^Answer\s*:\s*(.+)$', text, re.MULTILINE | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def extract_fwd_consistency(formal_line: str) -> Optional[tuple[str, str]]:
    """Extract (match, note) from FWD_CONSISTENCY(match=yes, note=\"...\") pattern."""
    if not formal_line:
        return None
    match_m = re.search(r'match\s*=\s*(yes|no)', formal_line, re.IGNORECASE)
    note_m = re.search(r'note\s*=\s*"([^"]*)"', formal_line)
    if match_m:
        return match_m.group(1).lower(), note_m.group(1) if note_m else ""
    return None
