"""Instruction templates for the two ChEBI-20 tasks.

Kept in one place so prompts are easy to audit / tweak. Each template has a
single ``{smiles}`` or ``{description}`` slot.
"""

# SMILES -> natural-language description
CAPTIONING_PROMPT = (
    "Please give me some details about the molecule with the following "
    "SMILES representation.\n{smiles}"
)

# natural-language description -> SMILES
CAPTION2SMILES_PROMPT = (
    "Please give me a molecule (represented in SMILES) that fits the "
    "following description.\n{description}"
)


def captioning_instruction(smiles: str) -> str:
    return CAPTIONING_PROMPT.format(smiles=smiles)


def caption2smiles_instruction(description: str) -> str:
    return CAPTION2SMILES_PROMPT.format(description=description)
