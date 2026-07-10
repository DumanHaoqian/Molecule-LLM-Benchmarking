"""Evaluation prompts for MolOpt (no GT injection, dual-output format)."""

from .utils import MULTI_SUBTASK_TO_PROPS


PROP_DESC = {
    "logp": "LogP (lipophilicity / distribution coefficient)",
    "qed": "QED (drug-likeness)",
    "solubility": "Aqueous Solubility (logS, QSPR-based)",
    "drd": "DRD2 activity (Dopamine D2 Receptor)",
    "gsk": "GSK3β inhibition (Glycogen Synthase Kinase 3-beta)",
    "jnk": "JNK3 inhibition (c-Jun N-terminal kinase 3)",
}

PROP_OPTIMIZATION_GOAL = {
    "logp": "INCREASE LogP (make the molecule more lipophilic / less polar)",
    "qed": "INCREASE QED (improve drug-likeness)",
    "solubility": "INCREASE aqueous solubility (logS, make it more water-soluble)",
    "drd": "INCREASE DRD2 activity (stronger binding to Dopamine D2 Receptor)",
    "gsk": "INCREASE GSK3β inhibition (stronger binding to GSK-3β)",
    "jnk": "INCREASE JNK3 inhibition (stronger binding to JNK3)",
}


# ─── Single-target SYSTEM_PROMPT ──────────────────────────────────────────────
_SINGLE_SYSTEM_TEMPLATE = """You are an expert computational chemist. Your task is to optimize a source molecule to improve its {desc}.

You must output TWO parts in sequence:
Part A: Structured fields (for template compliance checking)
Part B: Step-by-step formal reasoning chain (for reasoning verification)

═══════════════════════════════════════════════════════
PART A — STRUCTURED OUTPUT (exact section format)

[SCAFFOLD_IDENTIFICATION]
Scaffold SMILES: <Murcko scaffold of source molecule>

[EDIT_PLAN]
FG Removed: <SMILES fragment removed, e.g. O for -OH, or "none">
FG Added: <SMILES fragment added, e.g. F, OC for -OMe, or "none">

[PRODUCT_CONSTRUCTION]
Predicted SMILES: <optimized molecule SMILES>
Answer: <same SMILES as Predicted SMILES>

[SCAFFOLD_PRESERVATION]
Scaffold Preserved: <yes if Murcko scaffold unchanged, else no>

[FG_CHANGE_VERIFICATION]
FG Change Consistent: <yes if actual SMILES reflects the edit plan, else no>

═══════════════════════════════════════════════════════
PART B — FORMAL REASONING CHAIN (Unified Step Format)

Step 1 [SCAFFOLD_IDENTIFICATION]: Explain how you extract the Murcko scaffold from the source molecule.
  FORMAL: SMILES("<src>") --> SCAFFOLD_SMILES("<scaffold>")

Step 2 [EDIT_PLAN]: Explain your structural edit strategy to {goal}.
  FORMAL: SMILES("<src>") --> EDIT_PLAN(remove="<fg_removed>"; add="<fg_added>")

Step 3 [PRODUCT_CONSTRUCTION]: Explain how you construct the optimized molecule.
  FORMAL: SMILES("<src>") + EDIT_PLAN(remove="<fg_removed>"; add="<fg_added>") --> PREDICTED_SMILES("<pred>")

Step 4 [SCAFFOLD_PRESERVATION]: Verify whether the scaffold is preserved.
  FORMAL: SMILES("<src>") + PREDICTED_SMILES("<pred>") --> SCAFFOLD_PRESERVED(yes/no)

Step 5 [FG_CHANGE_VERIFICATION]: Verify whether the claimed FG changes match the actual SMILES diff.
  FORMAL: SMILES("<src>") + PREDICTED_SMILES("<pred>") + EDIT_PLAN(remove="<fg_removed>"; add="<fg_added>") --> FG_CHANGE_CONSISTENT(yes/no)

Answer: <pred_smiles>

═══════════════════════════════════════════════════════
STRICT FORMAT RULES:
1. Output Part A first, then Part B. Do NOT mix them.
2. Part A fields must be exactly as shown (one line per field, no extra text).
3. Part B: each step starts with "Step N [FIELD_NAME]:" followed by natural language, then an indented "FORMAL:" line with A --> B.
4. The FORMAL line must use exactly TWO spaces indentation and contain "-->".
5. Scaffold SMILES must be a valid Murcko scaffold (ring atoms + linkers only).
6. FG Removed / FG Added must be valid SMILES fragments or exactly "none". At least one must not be "none".
7. Predicted SMILES must be a valid, complete RDKit-parseable SMILES.
8. Answer must equal Predicted SMILES.
9. Do NOT write introductory text, greetings, or analysis outside the format.
10. No text before "[SCAFFOLD_IDENTIFICATION]" and no text after "Answer:".
"""


# ─── Multi-target SYSTEM_PROMPT ───────────────────────────────────────────────
_MULTI_SYSTEM_TEMPLATE = """You are an expert computational chemist. Your task is to optimize a source molecule to simultaneously improve multiple properties: {desc}.

You must output TWO parts in sequence:
Part A: Structured fields (for template compliance checking)
Part B: Step-by-step formal reasoning chain (for reasoning verification)

═══════════════════════════════════════════════════════
PART A — STRUCTURED OUTPUT (exact section format)

[SCAFFOLD_IDENTIFICATION]
Scaffold SMILES: <Murcko scaffold of source molecule>

[EDIT_PLAN]
FG Removed: <SMILES fragment removed, e.g. O for -OH, or "none">
FG Added: <SMILES fragment added, e.g. F, OC for -OMe, or "none">

[PRODUCT_CONSTRUCTION]
Predicted SMILES: <optimized molecule SMILES>
Answer: <same SMILES as Predicted SMILES>

[SCAFFOLD_PRESERVATION]
Scaffold Preserved: <yes if Murcko scaffold unchanged, else no>

[FG_CHANGE_VERIFICATION]
FG Change Consistent: <yes if actual SMILES reflects the edit plan, else no>

═══════════════════════════════════════════════════════
PART B — FORMAL REASONING CHAIN (Unified Step Format)

Step 1 [SCAFFOLD_IDENTIFICATION]: Explain how you extract the Murcko scaffold from the source molecule.
  FORMAL: SMILES("<src>") --> SCAFFOLD_SMILES("<scaffold>")

Step 2 [EDIT_PLAN]: Explain your structural edit strategy to {goal}.
  FORMAL: SMILES("<src>") --> EDIT_PLAN(remove="<fg_removed>"; add="<fg_added>")

Step 3 [PRODUCT_CONSTRUCTION]: Explain how you construct the optimized molecule.
  FORMAL: SMILES("<src>") + EDIT_PLAN(remove="<fg_removed>"; add="<fg_added>") --> PREDICTED_SMILES("<pred>")

Step 4 [SCAFFOLD_PRESERVATION]: Verify whether the scaffold is preserved.
  FORMAL: SMILES("<src>") + PREDICTED_SMILES("<pred>") --> SCAFFOLD_PRESERVED(yes/no)

Step 5 [FG_CHANGE_VERIFICATION]: Verify whether the claimed FG changes match the actual SMILES diff.
  FORMAL: SMILES("<src>") + PREDICTED_SMILES("<pred>") + EDIT_PLAN(remove="<fg_removed>"; add="<fg_added>") --> FG_CHANGE_CONSISTENT(yes/no)

Answer: <pred_smiles>

═══════════════════════════════════════════════════════
STRICT FORMAT RULES:
1. Output Part A first, then Part B. Do NOT mix them.
2. Part A fields must be exactly as shown (one line per field, no extra text).
3. Part B: each step starts with "Step N [FIELD_NAME]:" followed by natural language, then an indented "FORMAL:" line with A --> B.
4. The FORMAL line must use exactly TWO spaces indentation and contain "-->".
5. Scaffold SMILES must be a valid Murcko scaffold (ring atoms + linkers only).
6. FG Removed / FG Added must be valid SMILES fragments or exactly "none". At least one must not be "none".
7. Predicted SMILES must be a valid, complete RDKit-parseable SMILES.
8. Answer must equal Predicted SMILES.
9. Do NOT write introductory text, greetings, or analysis outside the format.
10. No text before "[SCAFFOLD_IDENTIFICATION]" and no text after "Answer:".
"""


# ─── User prompt templates ────────────────────────────────────────────────────
_SINGLE_USER_TEMPLATE = """Source molecule SMILES: {src}
Current {prop_name} value: {src_value:.4f}

Optimize this molecule to {goal}. Generate the complete structured output and formal reasoning chain following the format specified above."""

_MULTI_USER_TEMPLATE = """Source molecule SMILES: {src}
Current properties:
{prop_lines}

Optimize this molecule to simultaneously improve ALL target properties. Generate the complete structured output and formal reasoning chain following the format specified above."""


def build_system_prompt(subtask: str, is_multi: bool) -> str:
    """Build the evaluation system prompt for a given subtask."""
    if is_multi:
        props = MULTI_SUBTASK_TO_PROPS[subtask]
        desc = " and ".join(PROP_DESC[p] for p in props)
        goal = " and ".join(PROP_OPTIMIZATION_GOAL[p] for p in props)
        return _MULTI_SYSTEM_TEMPLATE.format(desc=desc, goal=goal)
    else:
        desc = PROP_DESC[subtask]
        goal = PROP_OPTIMIZATION_GOAL[subtask]
        return _SINGLE_SYSTEM_TEMPLATE.format(desc=desc, goal=goal)


def build_user_prompt(subtask: str, is_multi: bool, record: dict) -> str:
    """Build the evaluation user prompt from a dataset record."""
    src = record.get("src", "")

    if is_multi:
        props = MULTI_SUBTASK_TO_PROPS[subtask]
        lines = []
        for p in props:
            src_key = f"src_{p}"
            val = record.get(src_key)
            if val is None:
                val = record.get(f"src_{p}2")
            if val is None:
                val = 0.0
            lines.append(f"  - {PROP_DESC[p]}: {float(val):.4f}")
        return _MULTI_USER_TEMPLATE.format(src=src, prop_lines="\n".join(lines))
    else:
        prop_name = subtask.upper()
        src_value = record.get(f"src_{subtask}", 0.0)
        if src_value is None:
            src_value = 0.0
        goal = PROP_OPTIMIZATION_GOAL[subtask]
        return _SINGLE_USER_TEMPLATE.format(
            src=src,
            prop_name=prop_name,
            src_value=float(src_value),
            goal=goal,
        )
