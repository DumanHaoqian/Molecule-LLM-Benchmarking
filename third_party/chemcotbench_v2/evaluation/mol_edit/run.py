"""Main entry point for MolEdit V2 three-layer evaluation."""

from evaluation.core.runner import BaseRunner, RunnerConfig
from evaluation.mol_edit.layer1_evaluator import evaluate_layer1
from evaluation.mol_edit.layer3_evaluator import Layer3Evaluator
from evaluation.mol_edit.metrics import compute_summary
from evaluation.mol_edit.prompt_builder import PromptBuilder

from evaluation.mol_edit.layer2_evaluator import evaluate_record as eval_layer2_record


class MolEditRunner(BaseRunner):
    def __init__(self, cfg: RunnerConfig):
        super().__init__(cfg)
        self.layer3 = Layer3Evaluator(cfg.subtask)
        self.edit_type = cfg.subtask.replace("_v2", "")

    def build_prompt_builder(self):
        return PromptBuilder(self.cfg.subtask)

    def evaluate_layer3(self, records: list[dict]) -> list[dict]:
        return self.layer3.evaluate_batch(records)

    def evaluate_layer1(self, records: list[dict]) -> list[dict]:
        for record in records:
            record.setdefault("edit_type", self.edit_type)
            record.setdefault("source_smiles", record.get("src_smiles", ""))
            record.update(evaluate_layer1(record, self.edit_type))
        return records

    def evaluate_layer2(self, records: list[dict]) -> list[dict]:
        for record in records:
            result = eval_layer2_record(record, self.cfg.subtask)
            for key, value in result.items():
                record[f"layer2_{key}"] = value
        return records

    def compute_summary(self, records: list[dict]) -> dict:
        return compute_summary(records)

    def print_summary(self, summary: dict) -> None:
        print("\n" + "=" * 70)
        cfg = summary["config"]
        print(f"Evaluation Summary — {cfg['task']}/{cfg['subtask']} — {cfg['model']}")
        print("=" * 70)
        layer1 = summary["layer1"]
        print(
            f"Layer 1  Exact Match Acc:  {layer1['exact_match_acc']:.1%}   "
            f"Avg FTS: {layer1['avg_fts']:.4f}"
        )
        print(
            f"         Main-frag Acc:      {layer1['top1_acc']:.1%}   "
            f"V_consistency: {layer1['v_consistency']:.3f}   "
            f"ESO: {layer1['avg_edit_site_overlap']:.3f}"
        )
        print(
            f"Layer 2  State Score: {summary['layer2']['state_score']:.3f}   "
            f"V6 (Top-1): {summary['layer2']['V6']:.3f}"
        )
        print(f"Layer 3  Type I all_pass: {summary['layer3']['type1']['all_pass_rate']:.1%}")
        print(
            "         Type II all_fields_match: "
            f"{summary['layer3']['type2']['all_fields_match_rate']:.1%}"
        )
        print("=" * 70)


def main() -> None:
    parser = MolEditRunner.build_parser("MolEdit three-layer evaluation")
    args = parser.parse_args()
    cfg = RunnerConfig(
        task="mol_edit",
        subtask=args.subtask,
        model_name=args.model,
        base_url=args.base_url or "",
        api_key=args.api_key or ("dummy" if args.eval_only else ""),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        delay=args.delay,
        n_samples=args.n_samples,
        seed=args.seed,
        eval_only=args.eval_only,
        rerun_failed=args.rerun_failed,
        max_workers=args.max_workers,
    )
    MolEditRunner(cfg).run()


if __name__ == "__main__":
    main()
