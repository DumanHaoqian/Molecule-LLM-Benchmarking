"""Task abstraction — one evaluatable direction within a benchmark.

A Task owns everything direction-specific:
  * turning a dataset example into a prompt,
  * turning a model's answer into a prediction,
  * scoring collected predictions (delegating to the shared ``metrics`` library),
  * declaring how its results table is laid out.

Heavy metric implementations stay reusable in ``molbench.metrics``; the Task
just wires the right fields into them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EvalRecord:
    """One generated item, as persisted to / loaded from the prediction jsonl."""

    example: Dict[str, Any]   # raw dataset example (self-contained for eval)
    prompt: str
    raw_output: str           # model's clean answer (post reasoning-parse)
    prediction: Any           # task-postprocessed prediction


class Task(ABC):
    #: task identifier, unique within its benchmark
    name: str
    #: answer-content token budget (reasoning headroom added by the model)
    max_new_tokens: int = 256
    #: results-table columns as (header, metric_key); header may carry ↑/↓
    columns: List[Tuple[str, str]] = []

    @abstractmethod
    def build_prompt(self, example: Dict[str, Any]) -> str:
        """Dataset example -> instruction string for the model."""

    @abstractmethod
    def postprocess(self, answer: str) -> Any:
        """Model answer string -> task prediction (e.g. an extracted SMILES)."""

    @abstractmethod
    def evaluate(self, records: List[EvalRecord], device: str = "cpu") -> Dict[str, Any]:
        """Score records; return a metric-name -> value dict (aggregate)."""

    def score_examples(
        self, records: List[EvalRecord], device: str = "cpu"
    ) -> Optional[List[Dict[str, Any]]]:
        """Per-example scores, aligned to ``records`` (for bad-case analysis).

        Return None if the task has no per-example scoring. Each dict is merged
        into the record when the runner writes the ``__scored.jsonl`` file.
        """
        return None
