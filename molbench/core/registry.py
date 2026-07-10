"""Name -> object registries for models and benchmarks.

Models are registered as lightweight *specs* (config + a factory) so importing
the registry never loads model weights. Benchmarks are registered as factories.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

from .benchmark import Benchmark
from .model import Model


@dataclass
class ModelSpec:
    key: str
    display_name: str
    params: str
    build: Callable[[], Model]  # lazily instantiates the heavy model


_MODELS: Dict[str, ModelSpec] = {}
_BENCHMARKS: Dict[str, Callable[[], Benchmark]] = {}


def register_model(spec: ModelSpec) -> None:
    _MODELS[spec.key] = spec


def register_benchmark(name: str, factory: Callable[[], Benchmark]) -> None:
    _BENCHMARKS[name] = factory


def get_model_spec(key: str) -> ModelSpec:
    if key not in _MODELS:
        raise KeyError(f"unknown model '{key}'. registered: {list(_MODELS)}")
    return _MODELS[key]


def get_benchmark(name: str) -> Benchmark:
    if name not in _BENCHMARKS:
        raise KeyError(f"unknown benchmark '{name}'. registered: {list(_BENCHMARKS)}")
    return _BENCHMARKS[name]()


def list_models() -> List[str]:
    return list(_MODELS)


def list_benchmarks() -> List[str]:
    return list(_BENCHMARKS)
