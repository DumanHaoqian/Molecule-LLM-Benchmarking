"""Pure-Python length-aware batch planning."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .model import GenerationConfig, GenerationInput


@dataclass(frozen=True)
class PreparedInput:
    item: GenerationInput
    chat_prompt: str
    prompt_tokens: int


def parse_length_batch_policy(value: str) -> List[Tuple[float, int]]:
    bands: List[Tuple[float, int]] = []
    for raw in value.split(","):
        try:
            upper_raw, batch_raw = raw.strip().split(":", 1)
            upper = float("inf") if upper_raw.lower() in {"inf", "*"} else float(upper_raw)
            batch = int(batch_raw)
        except ValueError as exc:
            raise ValueError(f"invalid length batch policy segment: {raw!r}") from exc
        if batch < 1 or upper <= 0:
            raise ValueError(f"invalid length batch policy segment: {raw!r}")
        if bands and upper <= bands[-1][0]:
            raise ValueError("length batch policy bounds must be strictly increasing")
        bands.append((upper, batch))
    if not bands or bands[-1][0] != float("inf"):
        raise ValueError("length batch policy must end with inf:<batch-size>")
    return bands


def length_band(size_hint: int, bands: Sequence[Tuple[float, int]]) -> Tuple[int, int]:
    for idx, (upper, batch_size) in enumerate(bands):
        if size_hint <= upper:
            return idx, batch_size
    raise AssertionError("policy must contain an infinite final band")


def plan_batches(
    prepared: List[PreparedInput], config: GenerationConfig
) -> List[Tuple[int, List[PreparedInput]]]:
    if config.batching not in {"fixed", "length-aware"}:
        raise ValueError(f"unknown batching mode: {config.batching}")
    if config.max_batch_size < 1 or config.token_budget < 1:
        raise ValueError("batch size and token budget must be positive")

    bands = parse_length_batch_policy(config.length_batch_policy)
    for item in prepared:
        required = item.prompt_tokens + config.max_new_tokens
        if required > config.token_budget:
            raise ValueError(
                f"example {item.item.example_index} requires {required} tokens, "
                f"exceeding token budget {config.token_budget} even at batch size 1"
            )

    if config.batching == "fixed":
        ordered = sorted(prepared, key=lambda p: p.item.example_index)
        planned: List[Tuple[int, List[PreparedInput]]] = []
        current: List[PreparedInput] = []
        for item in ordered:
            candidate = current + [item]
            max_prompt = max(x.prompt_tokens for x in candidate)
            fits_tokens = (
                len(candidate) * (max_prompt + config.max_new_tokens)
                <= config.token_budget
            )
            if current and (len(candidate) > config.max_batch_size or not fits_tokens):
                planned.append((-1, current))
                current = [item]
            else:
                current = candidate
            if len(current) == config.max_batch_size:
                planned.append((-1, current))
                current = []
        if current:
            planned.append((-1, current))
        return planned

    grouped: Dict[int, List[PreparedInput]] = defaultdict(list)
    for item in prepared:
        band, _ = length_band(item.item.size_hint, bands)
        grouped[band].append(item)

    planned: List[Tuple[int, List[PreparedInput]]] = []
    for band in sorted(grouped, reverse=True):
        values = sorted(grouped[band], key=lambda p: (-p.item.size_hint, p.item.example_index))
        band_limit = min(bands[band][1], config.max_batch_size)
        current: List[PreparedInput] = []
        for item in values:
            candidate = current + [item]
            max_prompt = max(x.prompt_tokens for x in candidate)
            fits_tokens = len(candidate) * (max_prompt + config.max_new_tokens) <= config.token_budget
            if current and (len(candidate) > band_limit or not fits_tokens):
                planned.append((band, current))
                current = [item]
            else:
                current = candidate
            if len(current) == band_limit:
                planned.append((band, current))
                current = []
        if current:
            planned.append((band, current))
    return planned
