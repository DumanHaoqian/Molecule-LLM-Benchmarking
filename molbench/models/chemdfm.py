"""ChemDFM wrappers with length-aware, incremental generation."""
from __future__ import annotations

from collections import defaultdict
from statistics import mean
import threading
import time
from typing import Dict, Iterator, List

import torch

from ..core.model import (
    GenerationBatch,
    GenerationConfig,
    GenerationInput,
    GenerationOutput,
    Model,
)
from ..core.batching import PreparedInput, plan_batches
from ..core.registry import ModelSpec, register_model
from ..utils.chem import parse_answer

_BASE = "/home/haoqian/Data/SAERAG/Open-Scopes/ChemDFM-Scope"

CHEMDFM_R_SYSTEM = (
    "You are a helpful assistant that is good at reasoning. You always reason "
    "thoroughly before giving response. The reasoning process and answer are "
    "enclosed within <think> </think> and <answer> </answer> tags, "
    "respectively.\ni.e.,\n<think>\nreasoning process here\n</think>\n"
    "<answer>\nanswer here\n</answer>"
)


class _Heartbeat:
    def __init__(self, label: str, seconds: int):
        self.label = label
        self.seconds = max(1, seconds)
        self._stop = threading.Event()
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.seconds):
            elapsed = time.monotonic() - self._started
            print(f"[heartbeat] {self.label} elapsed={elapsed:.0f}s", flush=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=1)
        return False


def cuda_preflight(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight failed: torch.cuda.is_available() is false")
    try:
        probe = torch.ones(256, device=device)
        probe = probe * 2
        torch.cuda.synchronize(device)
        del probe
    except Exception as exc:
        raise RuntimeError(f"CUDA preflight failed: {exc}") from exc


class ChemDFMModel(Model):
    def __init__(
        self,
        model_path: str,
        system: str,
        reasoning: bool = False,
        reasoning_budget: int = 1536,
        dtype=torch.bfloat16,
        device: str = "cuda",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.system = system
        self.reasoning = reasoning
        self.reasoning_budget = reasoning_budget
        self.model_path = model_path

        cuda_preflight(device)
        print(f"[model] loading {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
        self.model.eval()
        configured_eos = self.model.generation_config.eos_token_id
        configured_ids = (
            list(configured_eos)
            if isinstance(configured_eos, (list, tuple))
            else [configured_eos]
        )
        self.eos_token_ids = sorted(
            {
                int(token_id)
                for token_id in [self.tokenizer.eos_token_id, *configured_ids]
                if token_id is not None
            }
        )
        print(f"[model] ready on {next(self.model.parameters()).device}")
        print(f"[model] stop token ids={self.eos_token_ids}")

    def _build_prompt(self, instruction: str) -> str:
        msg = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": instruction},
        ]
        return self.tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        )

    def _prepare(self, inputs: List[GenerationInput]) -> List[PreparedInput]:
        prepared = []
        for item in inputs:
            prompt = self._build_prompt(item.instruction)
            tokens = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            prepared.append(PreparedInput(item=item, chat_prompt=prompt, prompt_tokens=tokens))
        return prepared

    def _decode_batch(
        self, batch: List[PreparedInput], config: GenerationConfig, batch_id: int
    ) -> GenerationBatch:
        prompts = [x.chat_prompt for x in batch]
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.device)
        kwargs = dict(
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.eos_token_ids,
        )
        if config.do_sample:
            kwargs.update(temperature=0.9, top_p=0.9, top_k=20)

        if str(self.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.monotonic()
        label = (
            f"batch={batch_id} size={len(batch)} "
            f"chars={min(x.item.size_hint for x in batch)}-{max(x.item.size_hint for x in batch)} "
            f"prompt_tokens={min(x.prompt_tokens for x in batch)}-"
            f"{max(x.prompt_tokens for x in batch)}"
        )
        with _Heartbeat(label, config.heartbeat_seconds), torch.inference_mode():
            out = self.model.generate(**enc, **kwargs)
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)
        elapsed = time.monotonic() - started
        peak_reserved = (
            torch.cuda.max_memory_reserved(self.device)
            if str(self.device).startswith("cuda")
            else 0
        )
        total_memory = (
            torch.cuda.get_device_properties(self.device).total_memory
            if str(self.device).startswith("cuda")
            else 0
        )

        new = out[:, enc["input_ids"].shape[1] :]
        outputs: List[GenerationOutput] = []
        for prepared_item, token_row in zip(batch, new):
            values = token_row.tolist()
            stops = [
                (values.index(token_id), token_id)
                for token_id in self.eos_token_ids
                if token_id in values
            ]
            if stops:
                eos_at, stop_token_id = min(stops)
                output_tokens = eos_at + 1
                finish_reason = "eos"
            else:
                stop_token_id = None
                output_tokens = len(values)
                finish_reason = "length" if len(values) >= config.max_new_tokens else "unknown"
            decoded = self.tokenizer.decode(
                token_row[:output_tokens], skip_special_tokens=True
            )
            outputs.append(
                GenerationOutput(
                    example_index=prepared_item.item.example_index,
                    text=parse_answer(decoded, self.reasoning),
                    prompt_tokens=prepared_item.prompt_tokens,
                    output_tokens=output_tokens,
                    finish_reason=finish_reason,
                    size_hint=prepared_item.item.size_hint,
                    stop_token_id=stop_token_id,
                )
            )
        return GenerationBatch(
            batch_id=batch_id,
            outputs=outputs,
            elapsed_seconds=elapsed,
            remaining_examples=0,
            metadata={
                "batch_size": len(batch),
                "size_hint_min": min(x.item.size_hint for x in batch),
                "size_hint_mean": mean(x.item.size_hint for x in batch),
                "size_hint_max": max(x.item.size_hint for x in batch),
                "prompt_tokens_min": min(x.prompt_tokens for x in batch),
                "prompt_tokens_mean": mean(x.prompt_tokens for x in batch),
                "prompt_tokens_max": max(x.prompt_tokens for x in batch),
                "peak_reserved_bytes": peak_reserved,
                "peak_memory_fraction": peak_reserved / total_memory if total_memory else 0.0,
            },
        )

    def _iter_with_oom_split(
        self, batch: List[PreparedInput], config: GenerationConfig, next_id: List[int]
    ) -> Iterator[GenerationBatch]:
        batch_id = next_id[0]
        next_id[0] += 1
        split_at = None
        try:
            yield self._decode_batch(batch, config, batch_id)
            return
        except RuntimeError as exc:
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
            if not is_oom:
                raise
            if len(batch) == 1:
                raise
            split_at = len(batch) // 2
        torch.cuda.empty_cache()
        assert split_at is not None
        print(
            f"[gen] CUDA OOM in batch {batch_id}; "
            f"retrying as {split_at}+{len(batch)-split_at}"
        )
        yield from self._iter_with_oom_split(batch[:split_at], config, next_id)
        yield from self._iter_with_oom_split(batch[split_at:], config, next_id)

    def iter_generate(
        self, inputs: List[GenerationInput], config: GenerationConfig
    ) -> Iterator[GenerationBatch]:
        prepared = self._prepare(inputs)
        planned = plan_batches(prepared, config)
        total = len(inputs)
        completed = 0
        next_id = [1]
        band_durations: Dict[int, List[float]] = defaultdict(list)
        print(f"[gen] planned {len(planned)} batches for {total} pending examples")

        for position, (band, logical_batch) in enumerate(planned):
            for result in self._iter_with_oom_split(logical_batch, config, next_id):
                completed += len(result.outputs)
                result.remaining_examples = total - completed
                result.metadata["length_band"] = band
                band_durations[band].append(result.elapsed_seconds)
                remaining_by_band = [b for b, _ in planned[position + 1 :]]
                estimates = []
                all_seen = [d for values in band_durations.values() for d in values]
                fallback = mean(all_seen) if all_seen else result.elapsed_seconds
                for future_band in remaining_by_band:
                    values = band_durations.get(future_band)
                    estimates.append(mean(values) if values else fallback)
                result.eta_seconds = sum(estimates)
                yield result


def _spec(key, display_name, subdir, system, reasoning, budget) -> ModelSpec:
    path = f"{_BASE}/{subdir}"
    return ModelSpec(
        key=key,
        display_name=display_name,
        params="14B",
        build=lambda: ChemDFMModel(
            path, system=system, reasoning=reasoning, reasoning_budget=budget
        ),
        artifact_identity={
            "model_path": path,
            "reasoning": reasoning,
            "reasoning_budget": budget,
        },
    )


register_model(
    _spec(
        "chemdfm-v2",
        "ChemDFM-v2.0-14B",
        "ChemDFM-v2.0-14B",
        "You are a helpful assistant.",
        False,
        0,
    )
)
register_model(
    _spec(
        "chemdfm-r",
        "ChemDFM-R-14B",
        "ChemDFM-R-14B",
        CHEMDFM_R_SYSTEM,
        True,
        1792,
    )
)
