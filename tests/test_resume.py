import os
import tempfile
import unittest

from molbench.core.benchmark import Benchmark
from molbench.core.io import paths_for, read_records
from molbench.core.model import (
    GenerationBatch,
    GenerationInput,
    GenerationOutput,
    Model,
)
from molbench.core.registry import ModelSpec, register_benchmark, register_model
from molbench.core.runner import run_generation
from molbench.core.task import Task


class FakeTask(Task):
    name = "fake"
    max_new_tokens = 16

    def build_prompt(self, example):
        return example["text"]

    def postprocess(self, answer):
        return answer.upper()


class FakeBenchmark(Benchmark):
    name = "resume-test"

    def load(self, split="test", limit=None):
        rows = [{"text": f"item-{i}"} for i in range(5)]
        return rows[:limit] if limit is not None else rows

    def tasks(self):
        return {"fake": FakeTask()}


class FakeModel(Model):
    def __init__(self, fail_after_first=False):
        self.fail_after_first = fail_after_first

    def iter_generate(self, inputs, config):
        for batch_id, item in enumerate(inputs, 1):
            output = GenerationOutput(
                example_index=item.example_index,
                text=item.instruction,
                prompt_tokens=len(item.instruction),
                output_tokens=2,
                finish_reason="eos",
                size_hint=item.size_hint,
            )
            yield GenerationBatch(
                batch_id=batch_id,
                outputs=[output],
                elapsed_seconds=0.01,
                remaining_examples=len(inputs) - batch_id,
                eta_seconds=0.01 * (len(inputs) - batch_id),
                metadata={
                    "size_hint_min": item.size_hint,
                    "size_hint_max": item.size_hint,
                    "prompt_tokens_min": len(item.instruction),
                    "prompt_tokens_max": len(item.instruction),
                },
            )
            if self.fail_after_first:
                raise RuntimeError("injected failure")


class ResumeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_benchmark("resume-test", FakeBenchmark)

    def test_failed_run_resumes_without_duplicates(self):
        builds = {"count": 0}

        def build():
            builds["count"] += 1
            return FakeModel(fail_after_first=builds["count"] == 1)

        register_model(ModelSpec("resume-model", "Resume model", "0", build))
        with tempfile.TemporaryDirectory() as out_dir:
            kwargs = dict(
                benchmark_name="resume-test",
                model_key="resume-model",
                task_names=["fake"],
                out_dir=out_dir,
                batch_size=1,
                batching="fixed",
            )
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                run_generation(**kwargs)

            paths = paths_for(out_dir, "resume-test", "resume-model", "fake", "test")
            self.assertTrue(os.path.exists(paths.partial))
            run_generation(**kwargs)
            records = read_records(paths.final)
            self.assertEqual([r.example_index for r in records], list(range(5)))
            self.assertEqual(len({r.example_id for r in records}), 5)
            self.assertFalse(os.path.exists(paths.partial))

    def test_changed_configuration_refuses_resume(self):
        register_model(
            ModelSpec(
                "always-fails",
                "Always fails",
                "0",
                lambda: FakeModel(fail_after_first=True),
            )
        )
        with tempfile.TemporaryDirectory() as out_dir:
            base = dict(
                benchmark_name="resume-test",
                model_key="always-fails",
                task_names=["fake"],
                out_dir=out_dir,
                batch_size=1,
                batching="fixed",
            )
            with self.assertRaises(RuntimeError):
                run_generation(**base)
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                run_generation(**base, do_sample=True)

    def test_final_artifact_recovers_missing_meta_and_manifest_status(self):
        register_model(
            ModelSpec("recovery-model", "Recovery model", "0", lambda: FakeModel())
        )
        with tempfile.TemporaryDirectory() as out_dir:
            kwargs = dict(
                benchmark_name="resume-test",
                model_key="recovery-model",
                task_names=["fake"],
                out_dir=out_dir,
                batch_size=1,
                batching="fixed",
            )
            run_generation(**kwargs)
            paths = paths_for(out_dir, "resume-test", "recovery-model", "fake", "test")
            os.unlink(paths.meta)
            import json

            with open(paths.manifest, encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["status"] = "running"
            with open(paths.manifest, "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            run_generation(**kwargs)
            self.assertTrue(os.path.exists(paths.meta))
            with open(paths.manifest, encoding="utf-8") as f:
                recovered = json.load(f)
            self.assertEqual(recovered["status"], "complete")
            self.assertTrue(recovered["recovered_after_finalize"])


if __name__ == "__main__":
    unittest.main()
