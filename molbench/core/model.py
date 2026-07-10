"""Benchmark-agnostic model and incremental generation contracts."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List


@dataclass(frozen=True)
class GenerationInput:
    """One indexed instruction handed from the runner to a model."""

    example_index: int
    example_id: str
    instruction: str
    size_hint: int


@dataclass
class GenerationOutput:
    """One generated answer plus enough metadata for durable persistence."""

    example_index: int
    text: str
    prompt_tokens: int
    output_tokens: int
    finish_reason: str
    size_hint: int
    stop_token_id: int | None = None


@dataclass
class GenerationBatch:
    """A completed physical batch. The runner persists each yielded batch."""

    batch_id: int
    outputs: List[GenerationOutput]
    elapsed_seconds: float
    remaining_examples: int
    eta_seconds: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 256
    max_batch_size: int = 8
    do_sample: bool = False
    batching: str = "length-aware"
    length_batch_policy: str = "128:16,256:8,384:4,512:2,inf:1"
    token_budget: int = 16384
    heartbeat_seconds: int = 30


class Model(ABC):
    reasoning: bool = False
    reasoning_budget: int = 1536

    @abstractmethod
    def iter_generate(
        self, inputs: List[GenerationInput], config: GenerationConfig
    ) -> Iterator[GenerationBatch]:
        """Yield completed batches; never retain a whole task before yielding."""

    def answer_budget(self, task_max_new_tokens: int) -> int:
        extra = self.reasoning_budget if self.reasoning else 0
        return task_max_new_tokens + extra
