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

    def load_task(
        self, task_name: str, split: str = "test", limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load the examples for one task; legacy benchmarks share one dataset."""
        if task_name not in self.tasks():
            raise KeyError(f"unknown task {task_name!r}")
        return self.load(split=split, limit=limit)

    def artifact_identity(self, task_name: str) -> Dict[str, Any]:
        """Stable benchmark/data metadata included in run fingerprints."""
        return {"benchmark_class": f"{type(self).__module__}.{type(self).__qualname__}"}

    def aggregate_task_results(self, task_results: Dict[str, Any]) -> Dict[str, Any] | None:
        """Optionally build benchmark-level reporting views from task summaries."""
        return None

    def task_names(self) -> List[str]:
        return list(self.tasks().keys())
