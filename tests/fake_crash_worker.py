"""Subprocess worker used for hard-kill and signal checkpoint tests."""
from __future__ import annotations

import sys
import time

from molbench.core.benchmark import Benchmark
from molbench.core.model import GenerationBatch, GenerationOutput, Model
from molbench.core.registry import ModelSpec, register_benchmark, register_model
from molbench.core.runner import run_generation
from molbench.core.task import Task


class WorkerTask(Task):
    name = "fake"
    max_new_tokens = 16

    def build_prompt(self, example):
        return example["text"]

    def postprocess(self, answer):
        return answer


class WorkerBenchmark(Benchmark):
    name = "process-resume-test"

    def load(self, split="test", limit=None):
        rows = [{"text": f"item-{index}"} for index in range(5)]
        return rows[:limit] if limit is not None else rows

    def tasks(self):
        return {"fake": WorkerTask()}


class WorkerModel(Model):
    def __init__(self, delay):
        self.delay = delay

    def iter_generate(self, inputs, config):
        for batch_id, item in enumerate(inputs, 1):
            yield GenerationBatch(
                batch_id=batch_id,
                outputs=[
                    GenerationOutput(
                        example_index=item.example_index,
                        raw_text=item.instruction,
                        answer_text=item.instruction,
                        prompt_tokens=len(item.instruction),
                        output_tokens=2,
                        finish_reason="eos",
                        size_hint=item.size_hint,
                    )
                ],
                elapsed_seconds=max(self.delay, 0.01),
                remaining_examples=len(inputs) - batch_id,
                eta_seconds=0.0,
                metadata={
                    "size_hint_min": item.size_hint,
                    "size_hint_mean": item.size_hint,
                    "size_hint_max": item.size_hint,
                    "prompt_tokens_min": len(item.instruction),
                    "prompt_tokens_mean": len(item.instruction),
                    "prompt_tokens_max": len(item.instruction),
                },
            )
            if self.delay:
                time.sleep(self.delay)


def main():
    out_dir = sys.argv[1]
    delay = float(sys.argv[2])
    register_benchmark("process-resume-test", WorkerBenchmark)
    register_model(
        ModelSpec(
            "process-resume-model",
            "Process resume model",
            "0",
            lambda: WorkerModel(delay),
            artifact_identity={"test_worker": 1},
        )
    )
    run_generation(
        benchmark_name="process-resume-test",
        model_key="process-resume-model",
        task_names=["fake"],
        out_dir=out_dir,
        batch_size=1,
        batching="fixed",
    )


if __name__ == "__main__":
    main()
