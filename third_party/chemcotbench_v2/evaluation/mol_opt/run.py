"""Main entry point for MolOpt V2 three-layer evaluation."""

from pathlib import Path

from evaluation.core.runner import BaseRunner, RunnerConfig
from evaluation.core.released_data import load_released_records
from evaluation.core.utils import load_json
from evaluation.mol_opt.evaluator_layer1 import (
    evaluate_multi_layer1,
    evaluate_single_layer1,
    summarize_multi_layer1,
    summarize_single_layer1,
)
from evaluation.mol_opt.layer2_evaluator import evaluate_layer2, summarize_layer2
from evaluation.mol_opt.evaluator_layer3 import evaluate_batch_layer3, summarize_layer3
from evaluation.mol_opt.parser import parse_batch
from evaluation.mol_opt.prompt import build_system_prompt, build_user_prompt
from evaluation.mol_opt.utils import MULTI_SUBTASK_TO_PROPS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_DATASET_BASE = PROJECT_ROOT / "dataset" / "deep_mol_opt"
MULTI_DATASET_BASE = PROJECT_ROOT / "dataset" / "deep_mol_opt_multi" / "by_pair"


class _PromptBuilder:
    def __init__(self, subtask: str, is_multi: bool):
        self.system_prompt = build_system_prompt(subtask, is_multi)
        self._subtask = subtask
        self._is_multi = is_multi

    def build_user_prompt(self, record: dict) -> str:
        return build_user_prompt(self._subtask, self._is_multi, record)


class MolOptRunner(BaseRunner):
    def __init__(self, cfg: RunnerConfig):
        super().__init__(cfg)
        self.is_multi = cfg.subtask in MULTI_SUBTASK_TO_PROPS
        self.gt_by_idx: dict[int, dict] = {}

    def load_gt_records(self) -> list[dict]:
        if self.is_multi:
            dataset_path = MULTI_DATASET_BASE / f"{self.cfg.subtask.replace('_', '__')}_v2.json"
        else:
            dataset_path = SINGLE_DATASET_BASE / self.cfg.subtask / "final_mmp_v2.json"

        if not dataset_path.exists():
            records = load_released_records("mol_opt", self.cfg.subtask)
            self.gt_by_idx = {record.get("idx", i): record for i, record in enumerate(records)}
            return records

        dataset = load_json(dataset_path)
        prm_dataset = load_json(self._resolve_gt_path())

        merged_dataset = []
        self.gt_by_idx = {}
        for idx, record in enumerate(dataset):
            merged = dict(record)
            merged.setdefault("idx", idx)
            merged_dataset.append(merged)
        for idx, record in enumerate(prm_dataset):
            gt_record = dict(record)
            gt_record.setdefault("idx", idx)
            self.gt_by_idx[gt_record["idx"]] = gt_record
        return merged_dataset

    def build_prompt_builder(self):
        return _PromptBuilder(self.cfg.subtask, self.is_multi)

    def parse_batch(self, records: list[dict]) -> list[dict]:
        return parse_batch(records)

    def evaluate_layer3(self, records: list[dict]) -> list[dict]:
        gt_records = [self.gt_by_idx.get(record.get("idx"), {}) for record in records]
        return evaluate_batch_layer3(records, gt_records)

    def evaluate_layer1(self, records: list[dict]) -> list[dict]:
        evaluator = evaluate_multi_layer1 if self.is_multi else evaluate_single_layer1
        for record in records:
            record.update(evaluator(record, self.cfg.subtask))
        return records

    def evaluate_layer2(self, records: list[dict]) -> list[dict]:
        for record in records:
            record.update(evaluate_layer2(record))
        return records

    def compute_summary(self, records: list[dict]) -> dict:
        layer1 = (
            summarize_multi_layer1(records, self.cfg.subtask)
            if self.is_multi
            else summarize_single_layer1(records, self.cfg.subtask)
        )
        return {
            "layer1": layer1,
            "layer2": summarize_layer2(records),
            "layer3": summarize_layer3(records),
        }

    def print_summary(self, summary: dict) -> None:
        print("\n" + "=" * 70)
        cfg = summary["config"]
        print(f"Evaluation Summary — {cfg['subtask']} — {cfg['model']} — n={cfg['n_samples']}")
        print("=" * 70)

        layer1 = summary["layer1"]
        if self.is_multi:
            print(f"Layer 1  Dual-SR%: {layer1.get('dual_sr_pct', 0.0):.1f}%")
            for key in layer1:
                if key.startswith("sr_") and key.endswith("_pct"):
                    print(f"         {key}: {layer1[key]:.1f}%")
            print(
                f"         norm_min={layer1.get('mean_norm_min', 'N/A')}  "
                f"norm_geo={layer1.get('mean_norm_geo', 'N/A')}"
            )
        else:
            print(
                f"Layer 1  SR%: {layer1.get('sr_pct', 0.0):.1f}%  "
                f"Best Rate: {layer1.get('best_rate', 0.0):.3f}  "
                f"Mean Δ: {layer1.get('mean_delta', 0.0):.4f}  "
                f"Invalid: {layer1.get('invalid_rate', 0.0):.3f}"
            )
        print(
            f"         Avg FTS: {layer1.get('avg_fts', 0.0):.4f}  "
            f"Scaffold hard={layer1.get('scaffold_hard', 0.0):.3f} "
            f"soft={layer1.get('scaffold_soft', 0.0):.3f}"
        )

        layer2 = summary["layer2"]
        print(
            f"Layer 2  State Score: {layer2.get('avg_state_score', 0.0):.3f}  "
            f"V1={layer2.get('V1', 0.0):.3f}  "
            f"V2={layer2.get('V2', 0.0):.3f}  "
            f"V3={layer2.get('V3', 0.0):.3f}  "
            f"V4={layer2.get('V4', 0.0):.3f}  "
            f"V5={layer2.get('V5', 0.0):.3f}"
        )

        layer3 = summary["layer3"]
        print(f"Layer 3  Step Score: {layer3.get('avg_step_score', 0.0):.3f}")
        print(
            f"         S1(scaffold avg): {layer3.get('layer3_step1_scaffold_match', 0.0):.3f}  "
            f"S2(editplan pct): {layer3.get('layer3_step2_editplan_match', 0.0):.1%}  "
            f"S3(product avg): {layer3.get('layer3_step3_product_match', 0.0):.3f}"
        )

    def _resolve_gt_path(self):
        from evaluation.core.config import resolve_gt_dataset_path

        return resolve_gt_dataset_path("mol_opt", self.cfg.subtask)


def main() -> None:
    parser = MolOptRunner.build_parser("MolOpt V2 three-layer evaluation")
    parser.add_argument("--n-test", type=int, default=None, help="Alias for --n-samples")
    args = parser.parse_args()
    n_samples = args.n_samples if args.n_samples is not None else args.n_test
    cfg = RunnerConfig(
        task="mol_opt",
        subtask=args.subtask,
        model_name=args.model,
        base_url=args.base_url or "https://api.ppio.com/openai",
        api_key=args.api_key or ("dummy" if args.eval_only else ""),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        delay=args.delay,
        n_samples=n_samples,
        seed=args.seed,
        eval_only=args.eval_only,
        rerun_failed=args.rerun_failed,
        max_workers=args.max_workers,
    )
    MolOptRunner(cfg).run()


if __name__ == "__main__":
    main()
