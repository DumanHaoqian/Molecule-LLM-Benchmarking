"""Build evaluation prompts by reusing formal_cot prompts and stripping GT injection."""
from importlib import import_module

from baselines.cot_eval.mol_edit.mol_edit_structured.utils import get_indexed_smiles


class PromptBuilder:
    def __init__(self, subtask: str):
        self.prompt_mod = import_module(f"formal_cot.mol_edit.{subtask}.prompt")
        self.system_prompt = self._strip_examples(self.prompt_mod.SYSTEM_PROMPT)
        self.system_prompt = self._add_format_discipline(self.system_prompt)
        self.user_template = self._strip_gt(self.prompt_mod.USER_TEMPLATE)
        self.subtask = subtask

    @staticmethod
    def _strip_examples(system_prompt: str) -> str:
        """Remove EXAMPLES section from SYSTEM_PROMPT to avoid GT leakage in eval."""
        lines = system_prompt.splitlines()
        filtered = []
        in_examples = False
        for line in lines:
            if line.strip().startswith("EXAMPLE"):
                in_examples = True
                continue
            if in_examples and line.strip().startswith("CRITICAL OUTPUT REQUIREMENTS:"):
                in_examples = False
            if not in_examples:
                filtered.append(line)
        return "\n".join(filtered).strip()

    @staticmethod
    def _add_format_discipline(system_prompt: str) -> str:
        """Append strict output discipline."""
        appendix = (
            "\n\n═══════════════════════════════════════════════════════\n"
            "STRICT OUTPUT DISCIPLINE (EVALUATION MODE):\n"
            "- Do NOT write any introductory text, greetings, or analysis.\n"
            "- Do NOT wrap your output in markdown code blocks (```).\n"
            "- Output ONLY the steps in the exact Unified Step Format specified above.\n"
            "- Every step MUST begin with 'Step N [FIELD_NAME]:' on its own line.\n"
            "- The FORMAL line must be indented with two spaces and on a SINGLE line.\n"
            "- End with 'Answer: <product_smiles>' and nothing after it.\n"
            "- Violating these rules will cause parsing failure."
        )
        return system_prompt + appendix

    def _strip_gt(self, template: str) -> str:
        """Remove GT injection lines from USER_TEMPLATE."""
        lines = template.splitlines()
        filtered = []
        skip = False
        for line in lines:
            lower = line.lower()
            if "ground truth product smiles:" in lower:
                skip = True
                continue
            if "ground truth" in lower and "provided" in lower:
                skip = True
                continue
            if "do not simply state the ground truth" in lower:
                skip = True
                continue
            if "you must still generate" in lower and "ground truth" in lower:
                skip = True
                continue
            if skip and line.strip() == "":
                skip = False
                continue
            if not skip:
                filtered.append(line)
        return "\n".join(filtered).strip()

    def build_user_prompt(self, record: dict) -> str:
        """Format the user prompt from a dataset record."""
        src = record.get("src_smiles", "")
        indexed = record.get("indexed_smiles") or get_indexed_smiles(src)
        instruction = record.get("instruction", "")
        try:
            return self.user_template.format(
                src_smiles=src,
                indexed_smiles=indexed,
                instruction=instruction,
            )
        except KeyError:
            keys = {
                "src_smiles": src,
                "indexed_smiles": indexed,
                "instruction": instruction,
                "gt_smiles": "",
            }
            return self.user_template.format_map(_SafeDict(keys))


class _SafeDict(dict):
    def __missing__(self, key):
        return ""
