"""Prompt adapter for the merged SMILES equivalent task."""
from formal_cot.mol_und.mutated.prompt import SYSTEM_PROMPT as MUTATED_SYSTEM_PROMPT
from formal_cot.mol_und.mutated.prompt import USER_TEMPLATE as MUTATED_USER_TEMPLATE
from formal_cot.mol_und.permutated.prompt import SYSTEM_PROMPT as PERMUTATED_SYSTEM_PROMPT
from formal_cot.mol_und.permutated.prompt import USER_TEMPLATE as PERMUTATED_USER_TEMPLATE


SYSTEM_PROMPT = MUTATED_SYSTEM_PROMPT
USER_TEMPLATE = MUTATED_USER_TEMPLATE


def system_prompt_for_source(source_subtask: str) -> str:
    if source_subtask == "permutated":
        return PERMUTATED_SYSTEM_PROMPT
    return MUTATED_SYSTEM_PROMPT


def user_template_for_source(source_subtask: str) -> str:
    if source_subtask == "permutated":
        return PERMUTATED_USER_TEMPLATE
    return MUTATED_USER_TEMPLATE
