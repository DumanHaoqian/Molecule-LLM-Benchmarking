"""Unified sampling with optional concurrent execution."""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from evaluation.api_client import OpenAICompatibleClient
from evaluation.core.utils import resolve_record_id


def run_sampling(
    records: list[dict],
    client: OpenAICompatibleClient,
    system_prompt: str,
    build_user_prompt: callable,
    save_path: Path,
    delay: float = 0.5,
    rerun_failed: bool = False,
    id_field: str = "id",
    max_workers: int = 1,
) -> list[dict]:
    """Call the API for each record and incrementally save to JSON.

    Args:
        records: Dataset records to sample.
        client: OpenAI-compatible API client.
        system_prompt: System prompt string.
        build_user_prompt: Callable(record) -> str to build user prompt.
        save_path: Path to incrementally save results.
        delay: Delay between requests (only effective when max_workers=1).
        rerun_failed: If True and save_path exists, reuse successful records.
        id_field: Primary identifier field used before shared fallback IDs.
        max_workers: Number of concurrent workers. 1 = sequential.

    Returns:
        List of result records (original fields + API response fields).
    """
    existing: dict = {}
    if rerun_failed and save_path.exists():
        with open(save_path) as f:
            existing = {}
            for i, r in enumerate(json.load(f)):
                oid = resolve_record_id(r, preferred_fields=[id_field], fallback=str(i))
                existing[oid] = r

    results_map: dict[int, dict] = {}
    if rerun_failed and existing:
        for i, rec in enumerate(records):
            oid = resolve_record_id(rec, preferred_fields=[id_field], fallback=str(i))
            reused = existing.get(oid)
            if reused and reused.get("api_success"):
                results_map[i] = reused
    total_in = total_out = 0
    lock = __import__("threading").Lock()

    def _sample_one(idx_rec: tuple[int, dict]) -> dict:
        i, rec = idx_rec
        oid = resolve_record_id(rec, preferred_fields=[id_field], fallback=str(i))

        if rerun_failed and oid in existing and existing[oid].get("api_success"):
            out_rec = existing[oid]
            return out_rec

        active_system_prompt = system_prompt(rec) if callable(system_prompt) else system_prompt
        resp = client.call(
            active_system_prompt,
            build_user_prompt(rec),
        )

        out_rec = dict(rec)
        out_rec.update(
            {
                "raw_output": resp["content"],
                "reasoning_content": resp.get("reasoning_content", ""),
                "api_success": resp["success"],
                "input_tokens": resp["input_tokens"],
                "output_tokens": resp["output_tokens"],
                "model": resp["model"],
                "finish_reason": resp["finish_reason"],
                "error": resp.get("error", ""),
            }
        )

        # Incremental save (thread-safe)
        if save_path:
            with lock:
                nonlocal total_in, total_out
                total_in += resp["input_tokens"]
                total_out += resp["output_tokens"]
                results_map[i] = out_rec
                _save_current(results_map, save_path)

        status = "ok" if resp["success"] else f"ERROR: {resp.get('error', '?')}"
        print(f"  [{i+1:03d}/{len(records):03d}] {status}  ({resp['output_tokens']} tok)")

        if max_workers == 1:
            time.sleep(delay)
        return out_rec

    if max_workers == 1:
        final_results = [_sample_one((i, r)) for i, r in enumerate(records)]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_sample_one, (i, r)): i
                for i, r in enumerate(records)
            }
            final_results = [None] * len(records)
            for future in as_completed(futures):
                idx = futures[future]
                final_results[idx] = future.result()

    print(f"\nSampling done. Total tokens: input={total_in}, output={total_out}")
    return final_results


def _save_current(results_map: dict[int, dict], save_path: Path) -> None:
    """Internal: save current results while preserving input order."""
    with open(save_path, "w") as f:
        ordered = [results_map[idx] for idx in sorted(results_map)]
        json.dump(ordered, f, indent=2, ensure_ascii=False)
