import os
import tempfile
import unittest

from molbench.core.benchmark import Benchmark
from molbench.core.io import (
    atomic_write_json,
    example_id,
    paths_for,
    read_json,
    record_to_row,
    write_records,
)
from molbench.core.registry import ModelSpec, register_benchmark, register_model
from molbench.core.runner import run_evaluation
from molbench.core.task import EvalRecord, Task


class ScoreTask(Task):
    name = "score"
    columns = [("Mean", "mean")]
    score_calls = 0
    fail_on_call = None

    def build_prompt(self, example):
        return example["text"]

    def postprocess(self, answer):
        return answer

    def score_chunk(self, records, device="cpu"):
        type(self).score_calls += 1
        if type(self).score_calls == type(self).fail_on_call:
            raise RuntimeError("injected scoring failure")
        return [{"value": r.example["value"]} for r in records]

    def aggregate(self, records, scores, device="cpu"):
        return {"mean": sum(s["value"] for s in scores) / len(scores)}


class ScoreBenchmark(Benchmark):
    name = "score-test"

    def load(self, split="test", limit=None):
        return []

    def tasks(self):
        return {"score": ScoreTask()}


class EvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_benchmark("score-test", ScoreBenchmark)
        register_model(ModelSpec("score-model", "Score model", "0", lambda: None))

    def test_scoring_is_checkpointed_and_not_repeated(self):
        ScoreTask.score_calls = 0
        ScoreTask.fail_on_call = None
        with tempfile.TemporaryDirectory() as out_dir:
            paths = paths_for(out_dir, "score-test", "score-model", "score", "test")
            records = []
            for index in range(5):
                example = {"text": str(index), "value": index}
                records.append(
                    EvalRecord(
                        example=example,
                        prompt=str(index),
                        raw_output=str(index),
                        prediction=str(index),
                        example_index=index,
                        example_id=example_id(example),
                    )
                )
            write_records(paths.final, records)
            atomic_write_json(
                paths.manifest,
                {"status": "complete", "fingerprint": "generation-fingerprint"},
            )

            result = run_evaluation(
                "score-test",
                ["score-model"],
                ["score"],
                out_dir=out_dir,
                chunk_size=2,
            )
            self.assertEqual(result["tasks"]["score"][0]["metrics"]["mean"], 2)
            self.assertEqual(ScoreTask.score_calls, 3)
            self.assertTrue(os.path.exists(paths.stem + "__scored.jsonl"))

            run_evaluation(
                "score-test",
                ["score-model"],
                ["score"],
                out_dir=out_dir,
                chunk_size=2,
            )
            self.assertEqual(ScoreTask.score_calls, 3)

    def test_failed_evaluation_marks_manifest_and_resumes(self):
        ScoreTask.score_calls = 0
        ScoreTask.fail_on_call = 2
        with tempfile.TemporaryDirectory() as out_dir:
            paths = paths_for(out_dir, "score-test", "score-model", "score", "test")
            records = []
            for index in range(5):
                example = {"text": str(index), "value": index}
                records.append(
                    EvalRecord(
                        example=example,
                        prompt=str(index),
                        raw_output=str(index),
                        prediction=str(index),
                        example_index=index,
                        example_id=example_id(example),
                    )
                )
            write_records(paths.final, records)
            atomic_write_json(
                paths.manifest,
                {"status": "complete", "fingerprint": "generation-fingerprint"},
            )

            with self.assertRaisesRegex(RuntimeError, "injected scoring failure"):
                run_evaluation(
                    "score-test",
                    ["score-model"],
                    ["score"],
                    out_dir=out_dir,
                    chunk_size=2,
                )
            scored_manifest = paths.stem + "__scored.run.json"
            self.assertEqual(read_json(scored_manifest)["status"], "failed")
            self.assertEqual(read_json(scored_manifest)["completed"], 2)

            ScoreTask.fail_on_call = None
            result = run_evaluation(
                "score-test",
                ["score-model"],
                ["score"],
                out_dir=out_dir,
                chunk_size=2,
            )
            self.assertEqual(result["tasks"]["score"][0]["metrics"]["mean"], 2)
            self.assertEqual(ScoreTask.score_calls, 4)
            self.assertEqual(read_json(scored_manifest)["status"], "complete")
            self.assertTrue(os.path.exists(paths.stem + "__scored.jsonl"))

            run_evaluation(
                "score-test",
                ["score-model"],
                ["score"],
                out_dir=out_dir,
                chunk_size=2,
            )
            self.assertEqual(ScoreTask.score_calls, 4)


if __name__ == "__main__":
    unittest.main()
