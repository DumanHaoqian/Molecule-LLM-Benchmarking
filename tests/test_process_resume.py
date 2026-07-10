import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from molbench.core.io import paths_for, read_records


class ProcessResumeTest(unittest.TestCase):
    def _start(self, out_dir, delay):
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tests.fake_crash_worker",
                out_dir,
                str(delay),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _wait_for_checkpoint(self, process, path):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if process.poll() is not None:
                self.fail(f"worker exited before checkpoint with {process.returncode}")
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                return
            time.sleep(0.02)
        process.kill()
        process.wait()
        self.fail("worker did not produce a durable checkpoint")

    def _resume_and_assert_complete(self, out_dir, paths):
        subprocess.run(
            [sys.executable, "-m", "tests.fake_crash_worker", out_dir, "0"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        records = read_records(paths.final)
        self.assertEqual([record.example_index for record in records], list(range(5)))
        self.assertEqual(len({record.example_id for record in records}), 5)
        self.assertFalse(os.path.exists(paths.partial))

    @unittest.skipUnless(hasattr(signal, "SIGKILL"), "SIGKILL is unavailable")
    def test_kill9_then_resume_has_no_duplicates_or_missing_rows(self):
        with tempfile.TemporaryDirectory() as out_dir:
            paths = paths_for(
                out_dir,
                "process-resume-test",
                "process-resume-model",
                "fake",
                "test",
            )
            process = self._start(out_dir, 60)
            self._wait_for_checkpoint(process, paths.partial)
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
            self.assertLess(process.returncode, 0)
            self._resume_and_assert_complete(out_dir, paths)

    def test_sigterm_persists_current_batch_and_marks_interrupted(self):
        with tempfile.TemporaryDirectory() as out_dir:
            paths = paths_for(
                out_dir,
                "process-resume-test",
                "process-resume-model",
                "fake",
                "test",
            )
            process = self._start(out_dir, 0.25)
            self._wait_for_checkpoint(process, paths.partial)
            process.terminate()
            process.wait(timeout=5)
            self.assertNotEqual(process.returncode, 0)
            with open(paths.manifest, encoding="utf-8") as file:
                manifest = json.load(file)
            self.assertEqual(manifest["status"], "interrupted")
            self.assertGreaterEqual(manifest["completed"], 1)
            self._resume_and_assert_complete(out_dir, paths)


if __name__ == "__main__":
    unittest.main()
