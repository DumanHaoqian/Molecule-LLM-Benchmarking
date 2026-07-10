import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from molbench.core.io import read_json, read_records


class MigrationTest(unittest.TestCase):
    def test_legacy_artifact_gets_current_rows_and_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy.jsonl"
            destination = root / "migrated.jsonl"
            rows = [
                {
                    "input": "CCO",
                    "target": "ethanol",
                    "prediction": "an alcohol",
                    "scores": {"rouge1": 0.5},
                },
                {
                    "input": "CC",
                    "target": "ethane",
                    "prediction": "an alkane",
                },
            ]
            with source.open("w", encoding="utf-8") as file:
                for row in rows:
                    file.write(json.dumps(row) + "\n")

            subprocess.run(
                [
                    sys.executable,
                    str(
                        Path(__file__).resolve().parents[1]
                        / "scripts"
                        / "migrate_legacy_artifacts.py"
                    ),
                    str(source),
                    str(destination),
                    "--task",
                    "captioning",
                    "--model-key",
                    "legacy-test",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )

            records = read_records(str(destination))
            self.assertEqual([record.example_index for record in records], [0, 1])
            self.assertEqual(records[0].example["CAN_SMILES"], "CCO")
            manifest = read_json(str(root / "migrated.run.json"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["model_key"], "legacy-test")
            self.assertTrue((root / "migrated.meta.json").is_file())
            with destination.open(encoding="utf-8") as file:
                first_row = json.loads(next(file))
            self.assertEqual(first_row["scores"], {"rouge1": 0.5})


if __name__ == "__main__":
    unittest.main()
