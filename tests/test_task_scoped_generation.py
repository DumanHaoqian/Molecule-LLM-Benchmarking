import tempfile
import unittest

from molbench.core.benchmark import Benchmark
from molbench.core.io import paths_for, read_records
from molbench.core.model import GenerationBatch, GenerationOutput, Model
from molbench.core.registry import ModelSpec, register_benchmark, register_model
from molbench.core.runner import run_generation
from molbench.core.task import Task


class ScopedTask(Task):
    max_new_tokens = 8

    def __init__(self, name):
        self.name = name
        self.family = "family"
        self.subtask = name

    def build_prompt(self, example):
        return example["value"]

    def build_system_prompt(self, example):
        return "system:" + self.name

    def postprocess(self, answer):
        return answer.upper()


class ScopedBenchmark(Benchmark):
    name = "task-scoped-test"

    def load(self, split="test", limit=None):
        raise AssertionError("runner must use load_task")

    def load_task(self, task_name, split="test", limit=None):
        return [{"value": task_name}]

    def tasks(self):
        return {name: ScopedTask(name) for name in ("first", "second")}


class RawModel(Model):
    def iter_generate(self, inputs, config):
        for batch_id, item in enumerate(inputs, 1):
            self.assert_system(item)
            yield GenerationBatch(
                batch_id=batch_id,
                outputs=[
                    GenerationOutput(
                        example_index=item.example_index,
                        raw_text=f"<think>trace</think><answer>{item.instruction}</answer>",
                        answer_text=item.instruction,
                        prompt_tokens=4,
                        output_tokens=4,
                        finish_reason="eos",
                        size_hint=item.size_hint,
                    )
                ],
                elapsed_seconds=0.01,
                remaining_examples=len(inputs) - batch_id,
                metadata={},
            )

    @staticmethod
    def assert_system(item):
        if item.system_prompt != "system:" + item.instruction:
            raise AssertionError("task-specific system prompt was not preserved")


class TaskScopedGenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_benchmark("task-scoped-test", ScopedBenchmark)
        register_model(
            ModelSpec("raw-model", "Raw model", "0", lambda: RawModel())
        )

    def test_each_task_loads_its_own_examples_and_keeps_raw_output(self):
        with tempfile.TemporaryDirectory() as out_dir:
            run_generation(
                benchmark_name="task-scoped-test",
                model_key="raw-model",
                task_names=None,
                out_dir=out_dir,
                batch_size=2,
            )
            for name in ("first", "second"):
                path = paths_for(
                    out_dir, "task-scoped-test", "raw-model", name, "test"
                ).final
                record = read_records(path)[0]
                self.assertEqual(record.example["value"], name)
                self.assertEqual(record.answer_text, name)
                self.assertEqual(record.prediction, name.upper())
                self.assertIn("<think>trace</think>", record.raw_output)


if __name__ == "__main__":
    unittest.main()
