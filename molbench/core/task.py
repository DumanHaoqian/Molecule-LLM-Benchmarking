"""Task abstraction: prompting, post-processing, and resumable scoring."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class EvalRecord:
    """One generated item in artifact schema v2."""

    example: Dict[str, Any]
    prompt: str
    raw_output: str
    prediction: Any
    example_index: int = -1
    example_id: str = ""
    generation_metadata: Dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    name: str
    max_new_tokens: int = 256
    columns: List[Tuple[str, str]] = []

    @abstractmethod
    def build_prompt(self, example: Dict[str, Any]) -> str:
        """Dataset example -> model instruction."""

    @abstractmethod
    def postprocess(self, answer: str) -> Any:
        """Clean model answer -> task prediction."""

    def batch_length(self, example: Dict[str, Any], prompt: str) -> int:
        """Character-length hint used by the generic batch planner."""
        return len(prompt)

    def score_chunk(
        self, records: Sequence[EvalRecord], device: str = "cpu"
    ) -> Optional[List[Dict[str, Any]]]:
        """Return resumable per-example scores for one chunk, or None."""
        return self.score_examples(list(records), device=device)

    def aggregate(
        self,
        records: List[EvalRecord],
        scores: Optional[List[Dict[str, Any]]],
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Build aggregate metrics after all resumable chunks are complete."""
        return self.evaluate(records, device=device)

    # Legacy hooks remain as adapters for tasks that have no expensive
    # per-example path. New task implementations should override score_chunk
    # and aggregate so expensive work is never repeated.
    def evaluate(self, records: List[EvalRecord], device: str = "cpu") -> Dict[str, Any]:
        raise NotImplementedError

    def score_examples(
        self, records: List[EvalRecord], device: str = "cpu"
    ) -> Optional[List[Dict[str, Any]]]:
        return None
