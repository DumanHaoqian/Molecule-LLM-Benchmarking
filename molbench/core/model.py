"""Model abstraction — benchmark-agnostic text generation.

A ``Model`` takes plain instruction strings and returns *clean answer strings*.
Everything model-specific (chat template, system prompt, reasoning-answer
parsing) lives behind this interface, so benchmarks and the runner never care
which model they are driving.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class Model(ABC):
    #: whether the model emits a reasoning trace before its answer. The runner
    #: uses this to grant extra ``max_new_tokens`` headroom for the trace.
    reasoning: bool = False
    #: tokens of headroom to add for the reasoning trace (reasoning models only)
    reasoning_budget: int = 1536

    @abstractmethod
    def generate(
        self,
        instructions: List[str],
        max_new_tokens: int = 256,
        batch_size: int = 8,
        do_sample: bool = False,
    ) -> List[str]:
        """Return one clean answer per instruction (reasoning traces removed)."""

    def answer_budget(self, task_max_new_tokens: int) -> int:
        """Total generation budget = answer budget (+ reasoning headroom)."""
        extra = self.reasoning_budget if self.reasoning else 0
        return task_max_new_tokens + extra
