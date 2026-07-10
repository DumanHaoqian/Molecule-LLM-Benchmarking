"""Base evaluation runner with a hookable 6-step pipeline.

The pipeline:
  1. Sampling (API calls)
  2. Parsing (raw_output -> structured fields)
  3. Layer 3 (Type I verifier + Type II GT comparison)
  4. Layer 1 (final answer metrics)
  5. Layer 2 (V-points / template compliance)
  6. Summary (aggregation + print)

Subclasses override hooks to inject task-specific behavior.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

from evaluation.api_client import create_client
from evaluation.config import ModelConfig
from evaluation.core.config import get_task_spec, resolve_gt_dataset_path, resolve_output_dir
from evaluation.core.released_data import load_released_records
from evaluation.core.sampler import run_sampling
from evaluation.core.parser_adapter import ParserAdapter
from evaluation.core.utils import select_n_stratified, load_json, save_json


@dataclass
class RunnerConfig:
    task: str
    subtask: str
    model_name: str
    base_url: str
    api_key: str
    max_tokens: int = 32768
    temperature: float = 0.1
    delay: float = 0.5
    n_samples: int | None = None
    seed: int = 42
    eval_only: bool = False
    rerun_failed: bool = False
    max_workers: int = 1
    reasoning_effort: str | None = None
    timeout: float = 800.0


class BaseRunner:
    """Base class for 3-layer evaluation runners."""

    def __init__(self, cfg: RunnerConfig):
        self.cfg = cfg
        self.task_spec = get_task_spec(cfg.task)
        self.output_dir = resolve_output_dir(cfg.task, cfg.subtask, cfg.model_name)
        self.sampled_path = self.output_dir / "sampled.json"
        self.parsed_path = self.output_dir / "parsed.json"
        self.layer1_path = self.output_dir / "layer1_eval.json"
        self.layer2_path = self.output_dir / "layer2_eval.json"
        self.layer3_path = self.output_dir / "layer3_eval.json"
        self.summary_path = self.output_dir / "summary.json"

    # ------------------------------------------------------------------
    # Hooks: override in subclasses
    # ------------------------------------------------------------------

    def load_gt_records(self) -> list[dict]:
        """Load evaluation records, optionally merged with GT fields."""
        path = resolve_gt_dataset_path(self.cfg.task, self.cfg.subtask)
        records = load_json(path)
        if records:
            return records
        return load_released_records(self.cfg.task, self.cfg.subtask)

    def build_prompt_builder(self):
        """Return a prompt builder object with .system_prompt and .build_user_prompt()."""
        raise NotImplementedError

    def parse_batch(self, records: list[dict]) -> list[dict]:
        """Parse raw outputs into structured fields."""
        adapter = ParserAdapter(self.cfg.task, self.cfg.subtask)
        return adapter.parse_batch(records)

    def evaluate_layer3(self, records: list[dict]) -> list[dict]:
        """Run Layer 3 evaluation (Type I + Type II)."""
        raise NotImplementedError

    def evaluate_layer1(self, records: list[dict]) -> list[dict]:
        """Run Layer 1 evaluation (final answer metrics)."""
        raise NotImplementedError

    def evaluate_layer2(self, records: list[dict]) -> list[dict]:
        """Run Layer 2 evaluation (V-points / template compliance)."""
        raise NotImplementedError

    def compute_summary(self, records: list[dict]) -> dict:
        """Compute final summary from fully-evaluated records."""
        raise NotImplementedError

    def print_summary(self, summary: dict) -> None:
        """Print summary to stdout."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self) -> dict:
        cfg = self.cfg

        # Load evaluation records (direct PRM GT or task-specific dataset merged with GT)
        source_records = self.load_gt_records()
        records = select_n_stratified(source_records, cfg.n_samples, cfg.seed)
        print(f"Loaded {len(records)} records")

        # Step 1: Sampling
        if not cfg.eval_only:
            mcfg = ModelConfig.from_args(
                cfg.model_name,
                cfg.base_url,
                cfg.api_key,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                reasoning_effort=cfg.reasoning_effort,
                timeout=cfg.timeout,
            )
            client = create_client(mcfg)
            pb = self.build_prompt_builder()
            records = run_sampling(
                records=records,
                client=client,
                system_prompt=pb.system_prompt,
                build_user_prompt=pb.build_user_prompt,
                save_path=self.sampled_path,
                delay=cfg.delay,
                rerun_failed=cfg.rerun_failed,
                id_field=self.task_spec.id_fields[0],
                max_workers=cfg.max_workers,
            )
        else:
            if not self.sampled_path.exists():
                raise RuntimeError(
                    f"[eval-only] Missing sampled outputs: {self.sampled_path}. "
                    "Run sampling first, or set CHEMCOT_OUTPUT_DIR to the output root "
                    "that contains existing evaluation artifacts."
                )
            all_sampled = load_json(self.sampled_path)
            if not isinstance(all_sampled, list):
                raise RuntimeError(
                    f"[eval-only] Expected {self.sampled_path} to contain a JSON list, "
                    f"got {type(all_sampled).__name__}"
                )
            if not all_sampled:
                raise RuntimeError(f"[eval-only] Empty sampled output file: {self.sampled_path}")
            records = select_n_stratified(all_sampled, cfg.n_samples, cfg.seed)
            print(f"[eval-only] Loaded {len(records)} sampled records")

        # Step 2: Parsing
        records = self.parse_batch(records)
        save_json(records, self.parsed_path)
        n_ok = sum(1 for r in records if r.get("parse_ok"))
        print(f"Parsed {n_ok}/{len(records)} OK")

        # Step 3: Layer 3
        records = self.evaluate_layer3(records)
        save_json(records, self.layer3_path)
        print("Layer 3 complete")

        # Step 4: Layer 1
        records = self.evaluate_layer1(records)
        save_json(records, self.layer1_path)
        print("Layer 1 complete")

        # Step 5: Layer 2
        records = self.evaluate_layer2(records)
        save_json(records, self.layer2_path)
        print("Layer 2 complete")

        # Step 6: Summary
        summary = self.compute_summary(records)
        summary["config"] = {
            "task": cfg.task,
            "subtask": cfg.subtask,
            "model": cfg.model_name,
            "n_samples": len(records),
        }
        save_json(summary, self.summary_path)
        self.print_summary(summary)
        print(f"\nAll artifacts saved to {self.output_dir}")
        return summary

    @classmethod
    def build_parser(cls, description: str) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("--subtask", required=True, help="Subtask name")
        parser.add_argument("--model", required=True, help="Model name")
        parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
        parser.add_argument("--api-key", default=None, help="API key")
        parser.add_argument("--n-samples", type=int, default=None, help="Max samples to evaluate")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--delay", type=float, default=0.5)
        parser.add_argument("--eval-only", action="store_true", help="Skip API calls")
        parser.add_argument("--rerun-failed", action="store_true")
        parser.add_argument("--max-tokens", type=int, default=32768)
        parser.add_argument("--temperature", type=float, default=0.1)
        parser.add_argument("--max-workers", type=int, default=1, help="Concurrent API workers")
        return parser
