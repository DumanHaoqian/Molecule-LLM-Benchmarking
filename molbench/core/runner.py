"""Resumable generation/evaluation orchestration and table rendering."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import glob
import os
import signal
import subprocess
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .io import (
    ArtifactPaths,
    PartialWriter,
    RunLock,
    SCHEMA_VERSION,
    archive_paths,
    atomic_write_json,
    dataset_digest,
    example_id,
    finalize_partial,
    load_partial_rows,
    load_rows_by_index,
    paths_for,
    pred_stem,
    read_json,
    read_records,
    record_to_row,
    stable_hash,
)
from .model import GenerationConfig, GenerationInput
from .registry import get_benchmark, get_model_spec
from .task import EvalRecord


class GracefulStop(RuntimeError):
    pass


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _git_state() -> Dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    diff = run("diff", "--binary", "HEAD")
    return {
        "commit": commit or None,
        "dirty": bool(status),
        "worktree_hash": hashlib.sha256((status + "\n" + diff).encode()).hexdigest(),
    }


def _model_snapshot(identity: Dict[str, Any]) -> Dict[str, Any]:
    path = identity.get("model_path")
    if not isinstance(path, str) or not os.path.isdir(path):
        return identity
    files = []
    patterns = (
        "config.json",
        "tokenizer_config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "*.safetensors",
    )
    for pattern in patterns:
        for filename in sorted(glob.glob(os.path.join(path, pattern))):
            stat = os.stat(filename)
            files.append(
                {
                    "name": os.path.basename(filename),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {**identity, "files_digest": stable_hash(files), "files": files}


def _manifest_update(path: str, manifest: Dict[str, Any], **updates: Any) -> None:
    manifest.update(updates)
    manifest["updated_at"] = _now()
    atomic_write_json(path, manifest)


def _verify_fingerprint(manifest: Dict[str, Any], expected: str, path: str) -> None:
    actual = manifest.get("fingerprint")
    if actual != expected:
        raise ValueError(
            f"run fingerprint mismatch for {path}: existing={actual} requested={expected}. "
            "Use --restart to archive the old run instead of mixing artifacts."
        )


def _slice_examples(
    examples: List[Dict[str, Any]], index_slice: Optional[Tuple[int, int]]
) -> List[tuple[int, Dict[str, Any]]]:
    indexed = list(enumerate(examples))
    if index_slice is None:
        return indexed
    start, stop = index_slice
    if start < 0 or stop < start or stop > len(examples):
        raise ValueError(f"invalid slice {start}:{stop} for {len(examples)} examples")
    return indexed[start:stop]


def _prepare_generation_manifest(
    benchmark_name: str,
    model_key: str,
    task_name: str,
    split: str,
    limit: Optional[int],
    index_slice: Optional[Tuple[int, int]],
    indexed_examples: Sequence[tuple[int, Dict[str, Any]]],
    generation_config: GenerationConfig,
    model_identity: Dict[str, Any],
) -> Dict[str, Any]:
    config = {
        "schema_version": SCHEMA_VERSION,
        "kind": "generation",
        "benchmark": benchmark_name,
        "model_key": model_key,
        "model_identity": _model_snapshot(model_identity),
        "task": task_name,
        "split": split,
        "limit": limit,
        "slice": list(index_slice) if index_slice else None,
        "n": len(indexed_examples),
        "dataset_digest": dataset_digest(indexed_examples),
        "generation": {
            "max_new_tokens": generation_config.max_new_tokens,
            "max_batch_size": generation_config.max_batch_size,
            "do_sample": generation_config.do_sample,
            "batching": generation_config.batching,
            "length_batch_policy": generation_config.length_batch_policy,
            "token_budget": generation_config.token_budget,
        },
        "git": _git_state(),
    }
    return {
        **config,
        "fingerprint": stable_hash(config),
        "status": "running",
        "created_at": _now(),
        "updated_at": _now(),
    }


def run_generation(
    benchmark_name: str,
    model_key: str,
    task_names: Optional[List[str]],
    split: str = "test",
    limit: Optional[int] = None,
    out_dir: str = "results",
    batch_size: int = 8,
    do_sample: bool = False,
    resume: bool = True,
    restart: bool = False,
    batching: str = "length-aware",
    length_batch_policy: str = "128:16,256:8,384:4,512:2,inf:1",
    token_budget: int = 16384,
    heartbeat_seconds: int = 30,
    index_slice: Optional[Tuple[int, int]] = None,
) -> None:
    bench = get_benchmark(benchmark_name)
    spec = get_model_spec(model_key)
    tasks = bench.tasks()
    task_names = task_names or list(tasks)
    examples = bench.load(split=split, limit=limit)
    indexed_examples = _slice_examples(examples, index_slice)
    print(
        f"[gen] {benchmark_name} :: {model_key} :: {len(indexed_examples)} examples "
        f"slice={index_slice or 'all'}"
    )

    model = None
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"[signal] received {signum}; stopping after the current batch", flush=True)

    old_handlers: Dict[int, Any] = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, request_stop)

    try:
        for task_name in task_names:
            if task_name not in tasks:
                raise KeyError(f"unknown task {task_name!r}; available={list(tasks)}")
            task = tasks[task_name]
            reasoning_budget = (
                int(spec.artifact_identity.get("reasoning_budget", 0))
                if spec.artifact_identity.get("reasoning")
                else 0
            )
            generation_config = GenerationConfig(
                max_new_tokens=task.max_new_tokens + reasoning_budget,
                max_batch_size=batch_size,
                do_sample=do_sample,
                batching=batching,
                length_batch_policy=length_batch_policy,
                token_budget=token_budget,
                heartbeat_seconds=heartbeat_seconds,
            )
            paths = paths_for(out_dir, benchmark_name, model_key, task_name, split)
            requested_manifest = _prepare_generation_manifest(
                benchmark_name,
                model_key,
                task_name,
                split,
                limit,
                index_slice,
                indexed_examples,
                generation_config,
                spec.artifact_identity,
            )
            fingerprint = requested_manifest["fingerprint"]

            with RunLock(paths.lock, allow_stale=resume or restart):
                if restart:
                    archived = archive_paths(paths, reason="restart")
                    if archived:
                        print(f"[restart] archived prior artifacts -> {archived}")

                if not resume and not restart and any(
                    os.path.exists(path)
                    for path in (paths.final, paths.partial, paths.meta, paths.manifest)
                ):
                    raise FileExistsError(
                        f"artifacts already exist for {paths.stem}; use --resume or --restart"
                    )

                if os.path.exists(paths.final):
                    if not resume:
                        raise FileExistsError(f"final artifact already exists: {paths.final}")
                    if not os.path.exists(paths.manifest):
                        raise ValueError(f"final artifact has no run manifest: {paths.final}")
                    existing_manifest = read_json(paths.manifest)
                    _verify_fingerprint(existing_manifest, fingerprint, paths.manifest)
                    records = read_records(paths.final)
                    if len(records) != len(indexed_examples):
                        raise ValueError("completed artifact count does not match requested run")
                    expected_indexes = [idx for idx, _ in indexed_examples]
                    if [record.example_index for record in records] != sorted(expected_indexes):
                        raise ValueError("completed artifact indexes do not match requested run")
                    if existing_manifest.get("status") != "complete" or not os.path.exists(paths.meta):
                        finish_reasons = Counter(
                            record.generation_metadata.get("finish_reason", "unknown")
                            for record in records
                        )
                        atomic_write_json(
                            paths.meta,
                            {
                                "schema_version": SCHEMA_VERSION,
                                "benchmark": benchmark_name,
                                "model_key": model_key,
                                "display_name": spec.display_name,
                                "params": spec.params,
                                "task": task_name,
                                "split": split,
                                "limit": limit,
                                "slice": list(index_slice) if index_slice else None,
                                "n": len(records),
                                "do_sample": do_sample,
                                "max_new_tokens": generation_config.max_new_tokens,
                                "finish_reasons": dict(finish_reasons),
                                "fingerprint": fingerprint,
                                "completed_at": _now(),
                                "recovered_after_finalize": True,
                            },
                        )
                        _manifest_update(
                            paths.manifest,
                            existing_manifest,
                            status="complete",
                            completed=len(records),
                            recovered_after_finalize=True,
                            completed_at=_now(),
                        )
                    if os.path.exists(paths.partial):
                        os.unlink(paths.partial)
                    print(f"[resume] already complete -> {paths.final}")
                    continue

                if os.path.exists(paths.partial) and not os.path.exists(paths.manifest):
                    raise ValueError(f"partial artifact has no run manifest: {paths.partial}")

                if os.path.exists(paths.manifest):
                    manifest = read_json(paths.manifest)
                    _verify_fingerprint(manifest, fingerprint, paths.manifest)
                    _manifest_update(paths.manifest, manifest, status="running", resumed_at=_now())
                else:
                    manifest = requested_manifest
                    atomic_write_json(paths.manifest, manifest)

                rows_by_index = load_partial_rows(paths.partial)
                expected_by_index = {idx: example for idx, example in indexed_examples}
                prompts_by_index = {
                    idx: task.build_prompt(example) for idx, example in indexed_examples
                }
                for idx, row in rows_by_index.items():
                    if idx not in expected_by_index:
                        raise ValueError(f"partial artifact contains unexpected index {idx}")
                    expected_id = example_id(expected_by_index[idx])
                    if row["example_id"] != expected_id:
                        raise ValueError(f"partial artifact conflicts with dataset at index {idx}")

                pending = []
                for idx, example in indexed_examples:
                    if idx in rows_by_index:
                        continue
                    prompt = prompts_by_index[idx]
                    pending.append(
                        GenerationInput(
                            example_index=idx,
                            example_id=example_id(example),
                            instruction=prompt,
                            size_hint=task.batch_length(example, prompt),
                        )
                    )
                print(f"[resume] completed={len(rows_by_index)} pending={len(pending)}")

                started = time.monotonic()
                try:
                    if pending and model is None:
                        model = spec.build()
                    if model is not None:
                        actual_budget = model.answer_budget(task.max_new_tokens)
                        if actual_budget != generation_config.max_new_tokens:
                            raise ValueError(
                                f"model budget mismatch: manifest={generation_config.max_new_tokens} "
                                f"model={actual_budget}"
                            )

                    with PartialWriter(paths.partial) as writer:
                        if model is not None:
                            for batch in model.iter_generate(pending, generation_config):
                                persisted = []
                                for output in batch.outputs:
                                    example = expected_by_index[output.example_index]
                                    record = EvalRecord(
                                        example=example,
                                        prompt=prompts_by_index[output.example_index],
                                        raw_output=output.text,
                                        prediction=task.postprocess(output.text),
                                        example_index=output.example_index,
                                        example_id=example_id(example),
                                        generation_metadata={
                                            "batch_id": batch.batch_id,
                                            "batch_size": len(batch.outputs),
                                            "batch_elapsed_seconds": batch.elapsed_seconds,
                                            "size_hint": output.size_hint,
                                            "prompt_tokens": output.prompt_tokens,
                                            "output_tokens": output.output_tokens,
                                            "finish_reason": output.finish_reason,
                                            "stop_token_id": output.stop_token_id,
                                        },
                                    )
                                    row = record_to_row(record)
                                    rows_by_index[record.example_index] = row
                                    persisted.append(row)
                                writer.append(persisted)
                                eta = (
                                    f"{batch.eta_seconds / 60:.1f}m"
                                    if batch.eta_seconds is not None
                                    else "unknown"
                                )
                                output_lengths = [o.output_tokens for o in batch.outputs]
                                size_hints = [o.size_hint for o in batch.outputs]
                                prompt_lengths = [o.prompt_tokens for o in batch.outputs]
                                finish_reasons_batch = Counter(
                                    output.finish_reason for output in batch.outputs
                                )
                                examples_per_second = (
                                    len(batch.outputs) / batch.elapsed_seconds
                                    if batch.elapsed_seconds
                                    else float("inf")
                                )
                                tokens_per_second = (
                                    sum(output_lengths) / batch.elapsed_seconds
                                    if batch.elapsed_seconds
                                    else float("inf")
                                )
                                print(
                                    f"[gen] batch={batch.batch_id} "
                                    f"bucket={batch.metadata.get('length_band')} "
                                    f"size={len(batch.outputs)} "
                                    f"chars={min(size_hints)}/"
                                    f"{sum(size_hints) / len(size_hints):.1f}/"
                                    f"{max(size_hints)} "
                                    f"prompt_tokens={min(prompt_lengths)}/"
                                    f"{sum(prompt_lengths) / len(prompt_lengths):.1f}/"
                                    f"{max(prompt_lengths)} "
                                    f"output_tokens={min(output_lengths)}/"
                                    f"{sum(output_lengths) / len(output_lengths):.1f}/"
                                    f"{max(output_lengths)} finish={dict(finish_reasons_batch)} "
                                    f"peak_mem={100 * batch.metadata.get('peak_memory_fraction', 0):.1f}% "
                                    f"elapsed={batch.elapsed_seconds:.1f}s "
                                    f"throughput={examples_per_second:.2f}ex/s,"
                                    f"{tokens_per_second:.1f}tok/s "
                                    f"completed={len(rows_by_index)}/{len(indexed_examples)} eta={eta}",
                                    flush=True,
                                )
                                _manifest_update(
                                    paths.manifest,
                                    manifest,
                                    status="running",
                                    completed=len(rows_by_index),
                                    last_batch_id=batch.batch_id,
                                )
                                if stop_requested:
                                    raise GracefulStop("stop requested after durable batch checkpoint")

                    expected_indexes = [idx for idx, _ in indexed_examples]
                    finalize_partial(paths, rows_by_index, expected_indexes)
                    finish_reasons = Counter(
                        row.get("generation_metadata", {}).get("finish_reason", "unknown")
                        for row in rows_by_index.values()
                    )
                    meta = {
                        "schema_version": SCHEMA_VERSION,
                        "benchmark": benchmark_name,
                        "model_key": model_key,
                        "display_name": spec.display_name,
                        "params": spec.params,
                        "task": task_name,
                        "split": split,
                        "limit": limit,
                        "slice": list(index_slice) if index_slice else None,
                        "n": len(indexed_examples),
                        "do_sample": do_sample,
                        "max_new_tokens": generation_config.max_new_tokens,
                        "finish_reasons": dict(finish_reasons),
                        "fingerprint": fingerprint,
                        "completed_at": _now(),
                    }
                    atomic_write_json(paths.meta, meta)
                    _manifest_update(
                        paths.manifest,
                        manifest,
                        status="complete",
                        completed=len(indexed_examples),
                        elapsed_seconds=time.monotonic() - started,
                        completed_at=_now(),
                    )
                    print(f"[gen] wrote {len(indexed_examples)} -> {paths.final}")
                except GracefulStop as exc:
                    _manifest_update(
                        paths.manifest,
                        manifest,
                        status="interrupted",
                        completed=len(rows_by_index),
                        error=str(exc),
                    )
                    raise
                except Exception as exc:
                    _manifest_update(
                        paths.manifest,
                        manifest,
                        status="failed",
                        completed=len(rows_by_index),
                        error=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc(),
                    )
                    raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def _fmt(v: Any, decimals: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _render_table(task, rows: List[dict]) -> str:
    header = ["Method", "#Params."] + [h for h, _ in task.columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        cells = [row["display_name"], row["params"]]
        for _, key in task.columns:
            decimals = 1 if key == "levenshtein" else 3
            cells.append(_fmt(row["metrics"].get(key), decimals))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _evaluation_manifest(
    generation_manifest: Dict[str, Any], task_name: str, device: str, chunk_size: int
) -> Dict[str, Any]:
    config = {
        "schema_version": SCHEMA_VERSION,
        "kind": "evaluation",
        "generation_fingerprint": generation_manifest["fingerprint"],
        "task": task_name,
        "device": device,
        "chunk_size": chunk_size,
        "git": _git_state(),
    }
    return {
        **config,
        "fingerprint": stable_hash(config),
        "status": "running",
        "created_at": _now(),
        "updated_at": _now(),
    }


def run_evaluation(
    benchmark_name: str,
    model_keys: List[str],
    task_names: Optional[List[str]],
    split: str = "test",
    out_dir: str = "results",
    device: str = "cpu",
    resume: bool = True,
    restart: bool = False,
    chunk_size: int = 32,
) -> Dict[str, Any]:
    bench = get_benchmark(benchmark_name)
    tasks = bench.tasks()
    task_names = task_names or list(tasks)
    result: Dict[str, Any] = {"benchmark": benchmark_name, "split": split, "tasks": {}}
    markdown_sections: List[str] = []

    for task_name in task_names:
        task = tasks[task_name]
        table_rows = []
        for model_key in model_keys:
            spec = get_model_spec(model_key)
            generation_paths = paths_for(
                out_dir, benchmark_name, model_key, task_name, split
            )
            if not os.path.exists(generation_paths.final):
                raise FileNotFoundError(
                    f"complete generation artifact required: {generation_paths.final}"
                )
            if not os.path.exists(generation_paths.manifest):
                raise ValueError("generation artifact has no run manifest")
            generation_manifest = read_json(generation_paths.manifest)
            if generation_manifest.get("status") != "complete":
                raise ValueError("evaluation refuses an incomplete generation artifact")
            records = read_records(generation_paths.final)

            scored_paths = ArtifactPaths(
                pred_stem(out_dir, benchmark_name, model_key, task_name, split) + "__scored"
            )
            requested = _evaluation_manifest(
                generation_manifest, task_name, device, chunk_size
            )
            fingerprint = requested["fingerprint"]
            print(f"[eval] {benchmark_name} :: {model_key} :: {task_name}")

            with RunLock(scored_paths.lock, allow_stale=resume or restart):
                if restart:
                    archive_paths(scored_paths, reason="eval-restart")
                if not resume and not restart and any(
                    os.path.exists(path)
                    for path in (
                        scored_paths.final,
                        scored_paths.partial,
                        scored_paths.meta,
                        scored_paths.manifest,
                    )
                ):
                    raise FileExistsError(
                        f"evaluation artifacts already exist for {scored_paths.stem}"
                    )
                if os.path.exists(scored_paths.manifest):
                    manifest = read_json(scored_paths.manifest)
                    _verify_fingerprint(manifest, fingerprint, scored_paths.manifest)
                    _manifest_update(scored_paths.manifest, manifest, status="running")
                else:
                    manifest = requested
                    atomic_write_json(scored_paths.manifest, manifest)

                stop_requested = False

                def request_stop(signum, frame):
                    nonlocal stop_requested
                    stop_requested = True
                    print(
                        f"[signal] received {signum}; stopping after the current score chunk",
                        flush=True,
                    )

                old_handlers: Dict[int, Any] = {}
                for sig in (signal.SIGINT, signal.SIGTERM):
                    old_handlers[sig] = signal.getsignal(sig)
                    signal.signal(sig, request_stop)

                scored_rows: Dict[int, Dict[str, Any]] = {}
                try:
                    if os.path.exists(scored_paths.final):
                        scored_rows = load_rows_by_index(scored_paths.final)
                    else:
                        scored_rows = load_partial_rows(scored_paths.partial)
                    pending = [
                        record for record in records if record.example_index not in scored_rows
                    ]
                    supports_scores = True
                    if pending:
                        with PartialWriter(scored_paths.partial) as writer:
                            for start in range(0, len(pending), chunk_size):
                                chunk = pending[start : start + chunk_size]
                                scores = task.score_chunk(chunk, device=device)
                                if scores is None:
                                    if scored_rows:
                                        raise ValueError(
                                            "task scoring contract changed during resume"
                                        )
                                    supports_scores = False
                                    break
                                if len(scores) != len(chunk):
                                    raise ValueError(
                                        "score_chunk returned a misaligned result"
                                    )
                                rows = [
                                    record_to_row(record, score)
                                    for record, score in zip(chunk, scores)
                                ]
                                writer.append(rows)
                                scored_rows.update(
                                    {row["example_index"]: row for row in rows}
                                )
                                _manifest_update(
                                    scored_paths.manifest,
                                    manifest,
                                    status="running",
                                    completed=len(scored_rows),
                                )
                                print(
                                    f"[eval] scored {len(scored_rows)}/{len(records)}",
                                    flush=True,
                                )
                                if stop_requested:
                                    raise GracefulStop(
                                        "stop requested after durable score checkpoint"
                                    )

                    if stop_requested:
                        raise GracefulStop("stop requested before evaluation aggregate")

                    aligned_scores: Optional[List[Dict[str, Any]]]
                    if supports_scores:
                        expected = [record.example_index for record in records]
                        finalize_partial(scored_paths, scored_rows, expected)
                        aligned_scores = [scored_rows[i]["scores"] for i in expected]
                    else:
                        aligned_scores = None
                        if os.path.exists(scored_paths.partial):
                            os.unlink(scored_paths.partial)

                    if stop_requested:
                        raise GracefulStop("stop requested before evaluation aggregate")
                    metrics = task.aggregate(records, aligned_scores, device=device)
                    if stop_requested:
                        raise GracefulStop("stop requested after evaluation aggregate")
                    _manifest_update(
                        scored_paths.manifest,
                        manifest,
                        status="complete",
                        completed=len(records),
                        completed_at=_now(),
                    )
                    table_rows.append(
                        {
                            "display_name": spec.display_name,
                            "params": spec.params,
                            "metrics": metrics,
                        }
                    )
                except GracefulStop as exc:
                    _manifest_update(
                        scored_paths.manifest,
                        manifest,
                        status="interrupted",
                        completed=len(scored_rows),
                        error=str(exc),
                    )
                    raise
                except Exception as exc:
                    _manifest_update(
                        scored_paths.manifest,
                        manifest,
                        status="failed",
                        completed=len(scored_rows),
                        error=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc(),
                    )
                    raise
                finally:
                    for sig, handler in old_handlers.items():
                        signal.signal(sig, handler)

        table = _render_table(task, table_rows)
        markdown_sections.append(f"### {benchmark_name} — {task_name}\n\n{table}\n")
        result["tasks"][task_name] = table_rows

    result["markdown"] = "\n".join(markdown_sections)
    return result
