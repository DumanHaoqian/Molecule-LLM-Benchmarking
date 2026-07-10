import json
import os
import tempfile
import unittest

from molbench.core.io import (
    ArtifactPaths,
    PartialWriter,
    RunLock,
    example_id,
    finalize_partial,
    load_partial_rows,
    read_records,
    record_to_row,
)
from molbench.core.task import EvalRecord


def make_record(index):
    example = {"value": index}
    return EvalRecord(
        example=example,
        prompt=f"prompt {index}",
        raw_output=f"raw {index}",
        prediction=f"prediction {index}",
        example_index=index,
        example_id=example_id(example),
        generation_metadata={"finish_reason": "eos"},
    )


class ArtifactIoTest(unittest.TestCase):
    def test_truncated_last_row_is_repaired_and_finalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = ArtifactPaths(os.path.join(tmp, "predictions"))
            rows = [record_to_row(make_record(0)), record_to_row(make_record(1))]
            with PartialWriter(paths.partial) as writer:
                writer.append(rows)
            with open(paths.partial, "ab") as f:
                f.write(b'{"schema_version":2,"example_index":2')

            loaded = load_partial_rows(paths.partial)
            self.assertEqual(set(loaded), {0, 1})
            finalize_partial(paths, loaded, [0, 1])
            self.assertFalse(os.path.exists(paths.partial))
            self.assertEqual([r.example_index for r in read_records(paths.final)], [0, 1])

    def test_conflicting_duplicate_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "partial.jsonl")
            row = record_to_row(make_record(0))
            conflict = dict(row)
            conflict["prediction"] = "different"
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
                f.write(json.dumps(conflict) + "\n")
            with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
                load_partial_rows(path)

    def test_identical_duplicate_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "partial.jsonl")
            row = record_to_row(make_record(0))
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
                f.write(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate index"):
                load_partial_rows(path)

    def test_newline_terminated_corruption_is_not_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "partial.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(record_to_row(make_record(0))) + "\n")
                f.write("not-json\n")
            with self.assertRaisesRegex(ValueError, "corrupt JSONL"):
                load_partial_rows(path)

    def test_live_lock_rejects_second_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "artifact.lock")
            with RunLock(path):
                with self.assertRaisesRegex(RuntimeError, "locked"):
                    RunLock(path).acquire()

    def test_old_ownerless_lock_is_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "artifact.lock")
            os.mkdir(path)
            os.utime(path, (1, 1))
            with RunLock(path):
                self.assertTrue(os.path.exists(os.path.join(path, "owner.json")))


if __name__ == "__main__":
    unittest.main()
