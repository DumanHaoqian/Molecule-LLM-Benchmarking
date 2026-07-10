"""Main entry point for RxnPred three-layer evaluation."""

from evaluation.core.runner import BaseRunner, RunnerConfig
from evaluation.rxn_pred.layer1_evaluator import evaluate_layer1
from evaluation.rxn_pred.layer2_evaluator import evaluate_record as eval_layer2_record
from evaluation.rxn_pred.layer3_evaluator import Layer3Evaluator
from evaluation.rxn_pred.metrics import compute_summary
from evaluation.rxn_pred.prompt_builder import PromptBuilder


class RxnPredRunner(BaseRunner):
    def __init__(self, cfg: RunnerConfig):
        super().__init__(cfg)
        self.layer3 = Layer3Evaluator(cfg.subtask)

    def build_prompt_builder(self):
        return PromptBuilder(self.cfg.subtask)

    def evaluate_layer3(self, records: list[dict]) -> list[dict]:
        return self.layer3.evaluate_batch(records)

    def evaluate_layer1(self, records: list[dict]) -> list[dict]:
        for record in records:
            record.update(evaluate_layer1(record, self.cfg.subtask))
        return records

    def evaluate_layer2(self, records: list[dict]) -> list[dict]:
        for record in records:
            result = eval_layer2_record(record, self.cfg.subtask)
            for key, value in result.items():
                record[f"layer2_{key}"] = value
        return records

    def compute_summary(self, records: list[dict]) -> dict:
        return compute_summary(records, self.cfg.subtask)

    def print_summary(self, summary: dict) -> None:
        print("\n" + "=" * 70)
        cfg = summary["config"]
        print(f"Evaluation Summary — {cfg['task']}/{cfg['subtask']} — {cfg['model']}")
        print("=" * 70)

        layer1 = summary.get("layer1", {})
        if "top1_acc" in layer1:
            print(f"Layer 1  Top-1 Acc:        {layer1['top1_acc']:.1%}")
        if "avg_fts" in layer1:
            print(f"         Avg FTS:           {layer1['avg_fts']:.4f}")
        if "avg_ndcg" in layer1:
            print(f"         Avg NDCG:          {layer1['avg_ndcg']:.4f}")
        if "avg_mrr" in layer1:
            print(f"         Avg MRR:           {layer1['avg_mrr']:.4f}")
        if "mae" in layer1:
            print(f"         MAE:               {layer1['mae']:.4f}")
        if "within_5" in layer1:
            print(f"         Within 5°C:        {layer1['within_5']:.1%}")

        layer2 = summary.get("layer2", {})
        if "state_score" in layer2:
            print(f"Layer 2  State Score:      {layer2['state_score']:.3f}")
        v_keys = [key for key in sorted(layer2) if key.startswith("V") and key[1].isdigit()]
        if v_keys:
            print("         V-points:          " + "  ".join(f"{key}={layer2[key]:.2f}" for key in v_keys))

        layer3 = summary.get("layer3", {})
        type1 = layer3.get("type1", {})
        if "all_pass_rate" in type1:
            print(f"Layer 3  Type I all_pass:  {type1['all_pass_rate']:.1%}")
        type2 = layer3.get("type2", {})
        if "all_fields_match_rate" in type2:
            print(f"         Type II all_match: {type2['all_fields_match_rate']:.1%}")
        print("=" * 70)


def main() -> None:
    parser = RxnPredRunner.build_parser("RxnPred three-layer evaluation")
    args = parser.parse_args()
    cfg = RunnerConfig(
        task="rxn_pred",
        subtask=args.subtask,
        model_name=args.model,
        base_url=args.base_url or "https://api.ppio.com/openai",
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
    RxnPredRunner(cfg).run()


if __name__ == "__main__":
    main()
