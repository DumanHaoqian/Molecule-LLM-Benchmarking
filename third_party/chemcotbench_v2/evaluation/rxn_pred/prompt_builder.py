"""Build evaluation prompts by reusing formal_cot prompts and stripping GT injection."""
from importlib import import_module


class PromptBuilder:
    def __init__(self, subtask: str):
        from evaluation.core.config import resolve_module_name

        mod_name = resolve_module_name("rxn_pred", subtask)
        self.prompt_mod = import_module(f"formal_cot.rxn_pred.{mod_name}.prompt")
        self.system_prompt = self._strip_examples(self.prompt_mod.SYSTEM_PROMPT)
        self.system_prompt = self._add_format_discipline(self.system_prompt)
        self.user_template = self._strip_gt(self.prompt_mod.USER_TEMPLATE)
        self.subtask = subtask

    @staticmethod
    def _strip_examples(system_prompt: str) -> str:
        """Remove EXAMPLE INPUT section but keep EXAMPLE OUTPUT to preserve format demonstration."""
        lines = system_prompt.splitlines()
        filtered = []
        in_examples = False
        for line in lines:
            if line.strip().startswith("EXAMPLE INPUT"):
                in_examples = True
                continue
            if in_examples and line.strip().startswith("EXAMPLE OUTPUT"):
                in_examples = False
                filtered.append(line)
                continue
            if in_examples and line.strip().startswith("CRITICAL"):
                in_examples = False
            if in_examples and line.strip().startswith("STRICT FORMAT"):
                in_examples = False
            if in_examples and line.strip().startswith("═"):
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
            "- End with the Answer line and nothing after it.\n"
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
            if "ground truth" in lower and ("smiles" in lower or "temp" in lower or "ranking" in lower):
                skip = True
                continue
            if "ground truth" in lower and "provided" in lower:
                skip = True
                continue
            if "calibration" in lower:
                skip = True
                continue
            if "do not simply state" in lower:
                skip = True
                continue
            if "you must still generate" in lower and "ground truth" in lower:
                skip = True
                continue
            if "must match the calibration" in lower:
                skip = True
                continue
            if "all must be consistent with the calibration" in lower:
                skip = True
                continue
            if skip and line.strip() == "":
                skip = False
                continue
            if not skip:
                filtered.append(line)
        result = "\n".join(filtered)
        import re
        result = re.sub(r"\{gt_\w+\}", "", result)
        result = re.sub(r"\{gt_temp\}°C", "", result)
        result = re.sub(r"Ground Truth.*?:\s*\{[^}]+\}", "", result, flags=re.IGNORECASE)
        return result.strip()

    def build_user_prompt(self, record: dict) -> str:
        """Format the user prompt from a dataset record.

        Handles field-name mismatches between formal_cot prompts and evaluation
        datasets by injecting aliases and subtask-specific fallbacks.
        """
        keys = dict(record)

        # ── Common cross-dataset aliases ────────────────────────────────────
        if "reactants_smiles" in record:
            keys["reactants"] = record["reactants_smiles"]
        if "reagents_smiles" in record:
            keys["reagents"] = record["reagents_smiles"]
        if "gt_product_smiles" in record:
            keys["product_smiles"] = record["gt_product_smiles"]

        # ── Subtask-specific missing-field fallbacks ────────────────────────
        if self.subtask == "nepp" and not keys.get("context_query"):
            rxn = keys.get("rxn_cls", "")
            reactants = keys.get("current_reactants", "")
            annot = keys.get("step_annotation", "")
            keys["context_query"] = (
                f"Reaction class: {rxn}\n"
                f"Current step reactants: {reactants}\n"
                f"Step annotation: {annot}"
            )

        if self.subtask == "byproduct" and "gt_byproduct_smiles" not in keys:
            keys["gt_byproduct_smiles"] = keys.get("gt_smiles", "")

        if self.subtask == "rcr_reagent" and "gt_reagent" not in keys:
            keys["gt_reagent"] = keys.get("gt_reagent_smiles", "")

        if self.subtask == "rcr_solvent" and "gt_solvent" not in keys:
            keys["gt_solvent"] = keys.get("gt_solvent_smiles", "")

        if self.subtask == "yield_pred":
            if "gt_yield" not in keys:
                keys["gt_yield"] = keys.get("gt_float", "")
            if "source_note" not in keys:
                keys["source_note"] = ""
            if "reactants" not in keys:
                keys["reactants"] = keys.get("meta_reactants", "")

        prompt = self.user_template.format_map(_SafeDict(keys))

        # ── Safety check: warn if the prompt looks empty or truncated ───────
        stripped = prompt.strip()
        if len(stripped) < 20:
            print(f"[WARN] {self.subtask}: user prompt suspiciously short ({len(stripped)} chars). "
                  f"Check field mapping.")
        return prompt


class _SafeDict(dict):
    def __missing__(self, key):
        return ""
