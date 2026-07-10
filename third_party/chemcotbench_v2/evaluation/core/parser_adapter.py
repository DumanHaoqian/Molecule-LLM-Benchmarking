"""Unified parser adapter that dynamically imports task-specific parsers."""
from importlib import import_module

from evaluation.core.config import resolve_module_name


class ParserAdapter:
    """Thin wrapper around formal_cot parser modules."""

    def __init__(self, task: str, subtask: str):
        self.task = task
        self.subtask = subtask
        module_name = resolve_module_name(task, subtask)
        self.parser = import_module(f"formal_cot.{task}.{module_name}.parser")

    def parse_batch(self, records: list[dict]) -> list[dict]:
        if hasattr(self.parser, "parse_batch"):
            return self.parser.parse_batch(records)
        elif hasattr(self.parser, "parse_all"):
            return self.parser.parse_all(records)
        else:
            raise RuntimeError(
                f"Parser for {self.task}/{self.subtask} has no parse_batch or parse_all method"
            )
