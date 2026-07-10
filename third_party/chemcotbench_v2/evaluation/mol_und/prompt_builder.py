"""Build evaluation prompts by reusing formal_cot prompts and stripping GT injection."""
from importlib import import_module


class PromptBuilder:
    def __init__(self, subtask: str):
        self.prompt_mod = import_module(f"formal_cot.mol_und.{subtask}.prompt")
        self.subtask = subtask
        if subtask == "smiles_equivalent":
            self.system_prompt = self._build_dynamic_system_prompt
            self.user_template = None
        else:
            self.system_prompt = self._prepare_system_prompt(self.prompt_mod.SYSTEM_PROMPT)
            self.user_template = self._strip_gt(self.prompt_mod.USER_TEMPLATE)

    def _prepare_system_prompt(self, system_prompt: str) -> str:
        system_prompt = self._strip_examples(system_prompt)
        return self._add_format_discipline(system_prompt)

    def _build_dynamic_system_prompt(self, record: dict) -> str:
        source = record.get("source_subtask", "mutated")
        system_prompt = self.prompt_mod.system_prompt_for_source(source)
        return self._prepare_system_prompt(system_prompt)

    @staticmethod
    def _strip_examples(system_prompt: str) -> str:
        """Remove EXAMPLES section from SYSTEM_PROMPT to avoid GT leakage in eval."""
        lines = system_prompt.splitlines()
        filtered = []
        in_examples = False
        for line in lines:
            if line.strip().startswith("EXAMPLE") or line.strip().startswith("WORKED EXAMPLE"):
                in_examples = True
                continue
            if in_examples and line.strip().startswith("═══════════════════════════════════════════════════════"):
                in_examples = False
            if not in_examples:
                filtered.append(line)
        return "\n".join(filtered).strip()

    @staticmethod
    def _add_format_discipline(system_prompt: str) -> str:
        """Append strict output discipline inspired by V1 JSON format enforcement."""
        appendix = (
            "\n\n═══════════════════════════════════════════════════════\n"
            "STRICT OUTPUT DISCIPLINE (EVALUATION MODE):\n"
            "- Do NOT write any introductory text, greetings, or analysis.\n"
            "- Do NOT wrap your output in markdown code blocks (```).\n"
            "- Output ONLY the steps in the exact Unified Step Format specified above.\n"
            "- Every step MUST begin with 'Step N [FIELD_NAME]:' on its own line.\n"
            "- The FORMAL line must be indented with two spaces and on a SINGLE line.\n"
            "- End with 'Answer: <value>' and nothing after it.\n"
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
            if "ground truth" in lower:
                skip = True
                continue
            if "gt " in lower and ("answer" in lower or "smiles" in lower or "count" in lower):
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
        keys = dict(record)
        if self.subtask == "smiles_equivalent":
            source = keys.get("source_subtask", "mutated")
            template = self._strip_gt(self.prompt_mod.user_template_for_source(source))
            if source == "permutated":
                return template.format(
                    smiles_a=keys["smiles"],
                    smiles_b=keys["permutated"],
                )
            return template.format(
                smiles_a=keys["smiles"],
                smiles_b=keys["mutated"],
            )
        # Normalize key names for templates
        if "smiles" in keys and "fg_name" in keys:
            # fg_detect
            return self.user_template.format(
                smiles=keys["smiles"],
                fg_name=keys["fg_name"],
            )
        if "smiles" in keys and "ring_name" in keys:
            # ring_count
            return self.user_template.format(
                smiles=keys["smiles"],
                ring_name=keys["ring_name"],
            )
        if "smiles" in keys and "largest_scaffold" in keys:
            # murcko_scaffold
            return self.user_template.format(
                smiles=keys["smiles"],
            )
        if "smiles" in keys and "ring_system_scaffold" in keys:
            # ring_sys_scaffold
            return self.user_template.format(
                mol_smiles=keys["smiles"],
                scaffold_smiles=keys["ring_system_scaffold"],
            )
        if "smiles" in keys and "mutated" in keys:
            # mutated
            return self.user_template.format(
                smiles_a=keys["smiles"],
                smiles_b=keys["mutated"],
            )
        if "smiles" in keys and "permutated" in keys:
            # permutated
            return self.user_template.format(
                smiles_a=keys["smiles"],
                smiles_b=keys["permutated"],
            )
        # Generic fallback: try to format with record keys
        try:
            return self.user_template.format_map(_SafeDict(keys))
        except Exception:
            # Last resort: use template as-is with basic substitutions
            tpl = self.user_template
            for k, v in keys.items():
                if isinstance(v, str):
                    tpl = tpl.replace(f"{{{k}}}", v)
            return tpl


class _SafeDict(dict):
    def __missing__(self, key):
        return ""
