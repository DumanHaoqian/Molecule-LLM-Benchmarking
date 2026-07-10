"""Benchmark abstraction — a dataset bundled with its tasks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .task import Task


class Benchmark(ABC):
    #: benchmark identifier (used in CLI and prediction filenames)
    name: str

    @abstractmethod
    def load(self, split: str = "test", limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return dataset examples as plain dicts (self-contained for eval)."""

    @abstractmethod
    def tasks(self) -> Dict[str, Task]:
        """Return {task_name: Task} for this benchmark."""

    def task_names(self) -> List[str]:
        return list(self.tasks().keys())
