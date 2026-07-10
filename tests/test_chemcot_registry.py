import unittest
import json
from pathlib import Path

from molbench.benchmarks.chemcotbench.benchmark import ChemCoTBenchV1
from molbench.benchmarks.chemcotbench_v2.benchmark import (
    ChemCoTBenchV2,
    MAX_NEW_TOKENS,
    TASKS,
)
from molbench.benchmarks.chemcotbench_v2.reporting import aggregate_reporting


class ChemCoTRegistryTest(unittest.TestCase):
    def test_expected_task_counts_and_safe_file_names(self):
        v1 = ChemCoTBenchV1(data_root="/missing")
        v2 = ChemCoTBenchV2(data_root="/missing")
        self.assertEqual(len(v1.tasks()), 19)
        self.assertEqual(len(v2.tasks()), 31)
        self.assertEqual(len(TASKS), 31)
        self.assertTrue(all("/" not in name for name in [*v1.tasks(), *v2.tasks()]))

    def test_reporting_marks_missing_subtasks_instead_of_guessing(self):
        result = aggregate_reporting({})
        self.assertEqual(result["n_reporting_tasks"], 0)
        self.assertEqual(len(result["incomplete_reporting_tasks"]), 18)

    def test_all_31_subtasks_form_exactly_18_reporting_tasks(self):
        task_results = {}
        for family, subtask, _ in TASKS:
            summary = {
                "layer1": {
                    "exact_match_acc": 1.0,
                    "mean_mae": 0.0,
                    "avg_tanimoto": 1.0,
                    "top1_acc": 1.0,
                    "mae": 0.0,
                    "sr_pct": 100.0,
                    "dual_sr_pct": 100.0,
                },
                "layer2": {"state_score": 1.0, "avg_state_score": 1.0},
                "layer3": {
                    "type1": {"all_pass_rate": 1.0},
                    "type2": {"all_fields_match_rate": 1.0},
                    "avg_step_score": 1.0,
                },
            }
            task_results[f"{family}__{subtask}"] = [
                {
                    "model_key": "reference",
                    "display_name": "Reference",
                    "metrics": {"n": 1, "official_summary": summary},
                }
            ]
        result = aggregate_reporting(task_results)
        self.assertEqual(result["n_reporting_tasks"], 18)
        self.assertEqual(result["incomplete_reporting_tasks"], [])

    def test_output_budgets_cover_profiled_p99_9_margin(self):
        profile_path = (
            Path(__file__).resolve().parents[1]
            / "resources"
            / "chemcotbench"
            / "v2_token_profile.json"
        )
        profile = json.loads(profile_path.read_text(encoding="utf-8"))["tasks"]
        for family, subtask, _ in TASKS:
            configured = MAX_NEW_TOKENS[(family, subtask)]
            recommended = profile[f"{family}__{subtask}"]["recommended_max_new_tokens"]
            self.assertGreaterEqual(configured, recommended)


if __name__ == "__main__":
    unittest.main()
