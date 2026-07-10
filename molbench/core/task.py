"""Task abstraction: prompting, post-processing, and resumable scoring."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class EvalRecord:
    """One generated item in the current artifact schema."""

    example: Dict[str, Any]
    prompt: str
    raw_output: str
    prediction: Any
    answer_text: str = ""
    example_index: int = -1
    example_id: str = ""
    generation_metadata: Dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    name: str
    family: str = ""
    subtask: str = ""
    reporting_task: str = ""
    max_new_tokens: int = 256
    uses_model_reasoning_budget: bool = True
    columns: List[Tuple[str, str]] = []

    @abstractmethod
    def build_prompt(self, example: Dict[str, Any]) -> str:
        """Dataset example -> model instruction."""

    def build_system_prompt(self, example: Dict[str, Any]) -> Optional[str]:
        """Return a task-specific system prompt, or use the model default."""
        return None

    @abstractmethod
    def postprocess(self, answer: str) -> Any:
        """Clean the model's extracted answer text into a task prediction."""

    def artifact_identity(self) -> Dict[str, Any]:
        """Stable task metadata included in generation/evaluation fingerprints."""
        return {
            "task_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "family": self.family or None,
            "subtask": self.subtask or self.name,
            "reporting_task": self.reporting_task or None,
            "max_new_tokens": self.max_new_tokens,
            "uses_model_reasoning_budget": self.uses_model_reasoning_budget,
        }

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
