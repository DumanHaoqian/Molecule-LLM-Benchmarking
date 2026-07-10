"""Main entry point for MolUnd V2 three-layer evaluation."""

from pathlib import Path

from evaluation.core.runner import BaseRunner, RunnerConfig
from evaluation.core.parser_adapter import ParserAdapter
from evaluation.core.released_data import load_released_records
from evaluation.mol_und.layer1_evaluator import evaluate_layer1
from evaluation.mol_und.layer3_evaluator import Layer3Evaluator
from evaluation.mol_und.metrics import compute_summary
from evaluation.mol_und.prompt_builder import PromptBuilder

from evaluation.mol_und.layer2_evaluator import evaluate_record as eval_layer2_record

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATHS = {
    "fg_detect": "dataset/mol_understanding/1-fg_detect/fg_samples_v2.json",
    "ring_count": "dataset/mol_understanding/2-frag_detect/ring_count_v2.json",
    "murcko_scaffold": "dataset/mol_understanding/2-frag_detect/Murcko_scaffold_v2.json",
    "ring_sys_scaffold": "dataset/mol_understanding/2-frag_detect/ring_system_scaffold_v2.json",
    "mutated": "dataset/mol_understanding/3-permute_smiles/mutated_v2.json",
    "permutated": "dataset/mol_understanding/3-permute_smiles/permutated_v2.json",
    "smiles_equivalent": "dataset/mol_understanding/3-permute_smiles/smiles_equivalent_v2.json",
}


class MolUndRunner(BaseRunner):
    def __init__(self, cfg: RunnerConfig):
        super().__init__(cfg)
        self.layer3 = Layer3Evaluator(cfg.subtask)
        self.parser_adapter = ParserAdapter("mol_und", cfg.subtask)
        self.dataset_by_idx: dict[int, dict] = {}

    def load_gt_records(self) -> list[dict]:
        dataset_path = PROJECT_ROOT / DATASET_PATHS[self.cfg.subtask]
        if not dataset_path.exists():
            records = load_released_records("mol_und", self.cfg.subtask)
            self.dataset_by_idx = {record.get("orig_idx", i): record for i, record in enumerate(records)}
            return records
        dataset = self._load_json(dataset_path)
        prm_dataset = self._load_json(self._resolve_gt_path())

        gt_by_idx = {record["orig_idx"]: record for record in prm_dataset if "orig_idx" in record}
        merged_dataset = []
        for idx, record in enumerate(dataset):
            merged = dict(record)
            merged["orig_idx"] = idx
            gt = gt_by_idx.get(idx, {})
            for key in ("gt_count", "gt_smarts", "gt_scaffold", "gt_label", "gt_answer"):
                if key in gt:
                    merged[key] = gt[key]
            effective_subtask = merged.get("source_subtask", self.cfg.subtask)
            if effective_subtask == "mutated" and "gt_answer" not in merged:
                merged["gt_answer"] = "Different"
            if effective_subtask == "permutated" and "gt_answer" not in merged:
                merged["gt_answer"] = "Same"
            merged_dataset.append(merged)

        self.dataset_by_idx = {record["orig_idx"]: record for record in merged_dataset}
        return merged_dataset

    def build_prompt_builder(self):
        return PromptBuilder(self.cfg.subtask)

    def parse_batch(self, records: list[dict]) -> list[dict]:
        merged_records = []
        for record in records:
            orig_idx = record.get("orig_idx")
            base = dict(self.dataset_by_idx.get(orig_idx, {}))
            base.update(record)
            self._normalize_fields(base)
            merged_records.append(base)
        return self.parser_adapter.parse_batch(merged_records)

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

        layer1 = summary["layer1"]
        primary_name = layer1.get("primary_metric_name", "exact_match_acc")
        if primary_name == "mean_mae":
            print(f"Layer 1  Mean MAE:         {layer1['mean_mae']:.4f}")
            print(f"         Exact Match Acc:  {layer1['exact_match_acc']:.1%}")
        elif primary_name == "avg_tanimoto":
            print(f"Layer 1  Avg Tanimoto:     {layer1['avg_tanimoto']:.4f}")
            print(f"         Exact Match Acc:  {layer1['exact_match_acc']:.1%}")
        else:
            print(f"Layer 1  Exact Match Acc:  {layer1['exact_match_acc']:.1%}")

        layer2 = summary["layer2"]
        if "state_score" in layer2:
            print(f"Layer 2  State Score:      {layer2['state_score']:.3f}")
            v_keys = [key for key in sorted(layer2) if key.startswith("V") and key[1:].isdigit()]
            if v_keys:
                print("         V-points:         " + "  ".join(f"{key}={layer2[key]:.2f}" for key in v_keys))

        layer3 = summary["layer3"]
        print(f"Layer 3  Type I all_pass:  {layer3['type1']['all_pass_rate']:.1%}")
        print(
            "         Type II all_match: "
            f"{layer3['type2']['all_fields_match_rate']:.1%}"
        )
        print("=" * 70)

    def _resolve_gt_path(self):
        from evaluation.core.config import resolve_gt_dataset_path

        return resolve_gt_dataset_path("mol_und", self.cfg.subtask)

    @staticmethod
    def _load_json(path):
        import json

        with open(path) as handle:
            return json.load(handle)

    def _normalize_fields(self, record: dict) -> None:
        if self.cfg.subtask == "ring_sys_scaffold":
            if "smiles" in record and "mol_smiles" not in record:
                record["mol_smiles"] = record["smiles"]
            if "ring_system_scaffold" in record and "scaffold_smiles" not in record:
                record["scaffold_smiles"] = record["ring_system_scaffold"]
        effective_subtask = record.get("source_subtask", self.cfg.subtask)
        if effective_subtask == "mutated":
            if "smiles" in record and "smiles_a" not in record:
                record["smiles_a"] = record["smiles"]
            if "mutated" in record and "smiles_b" not in record:
                record["smiles_b"] = record["mutated"]
        if effective_subtask == "permutated":
            if "smiles" in record and "smiles_a" not in record:
                record["smiles_a"] = record["smiles"]
            if "permutated" in record and "smiles_b" not in record:
                record["smiles_b"] = record["permutated"]


def main() -> None:
    parser = MolUndRunner.build_parser("MolUnd V2 three-layer evaluation")
    args = parser.parse_args()
    cfg = RunnerConfig(
        task="mol_und",
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
    MolUndRunner(cfg).run()


if __name__ == "__main__":
    main()
