"""Unified single-task evaluation dispatcher."""

from importlib import import_module

from evaluation.core.runner import RunnerConfig


RUNNER_CLASS_MAP = {
    "mol_edit": ("evaluation.mol_edit.run", "MolEditRunner"),
    "rxn_pred": ("evaluation.rxn_pred.run", "RxnPredRunner"),
    "mol_und": ("evaluation.mol_und.run", "MolUndRunner"),
    "mol_opt": ("evaluation.mol_opt.run", "MolOptRunner"),
}


def _load_runner_class(task: str):
    if task not in RUNNER_CLASS_MAP:
        raise ValueError(f"Unknown task: {task}")
    module_name, class_name = RUNNER_CLASS_MAP[task]
    module = import_module(module_name)
    return getattr(module, class_name)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Unified evaluation dispatcher")
    parser.add_argument("--task", required=True, choices=sorted(RUNNER_CLASS_MAP))
    parser.add_argument("--subtask", required=True, help="Subtask name")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--base-url", default="https://api.ppio.com/openai")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None, help="Alias for MolOpt compatibility")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--reasoning-effort", type=str, default=None, help="e.g. high, medium, low")
    parser.add_argument("--timeout", type=float, default=800.0, help="API call timeout in seconds")
    args = parser.parse_args()

    n_samples = args.n_samples if args.n_samples is not None else args.n_test
    cfg = RunnerConfig(
        task=args.task,
        subtask=args.subtask,
        model_name=args.model,
        base_url=args.base_url or "",
        api_key=(
            args.api_key
            if args.api_key is not None
            else ("dummy" if args.eval_only else None)
        ),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        delay=args.delay,
        n_samples=n_samples,
        seed=args.seed,
        eval_only=args.eval_only,
        rerun_failed=args.rerun_failed,
        max_workers=args.max_workers,
        reasoning_effort=args.reasoning_effort,
        timeout=args.timeout,
    )
    runner_cls = _load_runner_class(args.task)
    runner_cls(cfg).run()


if __name__ == "__main__":
    main()
