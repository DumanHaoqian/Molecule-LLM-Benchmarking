"""Common utilities shared across all evaluation tasks."""
import json
import random
from pathlib import Path


COMMON_ID_FALLBACK_FIELDS = (
    "id",
    "sample_id",
    "orig_idx",
    "idx",
    "pool_id",
    "src_id",
    "orig_id",
)


def select_n_stratified(records: list[dict], n: int | None, seed: int = 42) -> list[dict]:
    """Select n records stratified by difficulty."""
    if n is None or n >= len(records):
        return list(records)
    random.seed(seed)
    by_diff: dict[str, list] = {}
    for r in records:
        by_diff.setdefault(r.get("difficulty", "unknown"), []).append(r)
    sampled = []
    for diff, pool in by_diff.items():
        k = max(1, round(n * len(pool) / len(records)))
        sampled.extend(random.sample(pool, min(k, len(pool))))
    random.shuffle(sampled)
    return sampled[:n]


def iter_record_id_candidates(
    record: dict,
    preferred_fields: list[str] | tuple[str, ...] | None = None,
):
    """Yield available identifier fields for a record in priority order."""
    seen: set[str] = set()
    for field in [*(preferred_fields or []), *COMMON_ID_FALLBACK_FIELDS]:
        if field in seen:
            continue
        seen.add(field)
        value = record.get(field)
        if value is not None:
            yield field, value


def resolve_record_id(
    record: dict,
    preferred_fields: list[str] | tuple[str, ...] | None = None,
    fallback=None,
):
    """Resolve the best available record identifier for reruns or deduplication."""
    for _, value in iter_record_id_candidates(record, preferred_fields):
        return value
    return fallback


def find_gt_record(record: dict, gt_records: list[dict], id_fields: list[str] | None = None) -> dict | None:
    """Find matching GT record by checking common ID fields."""
    for id_field, rec_id in iter_record_id_candidates(record, id_fields):
        for g in gt_records:
            if g.get(id_field) == rec_id:
                return g
    return None


def load_json(path: Path) -> list[dict] | dict:
    """Load JSON file, returning empty list/dict on error."""
    if not path.exists():
        return [] if path.suffix == ".json" else {}
    with open(path) as f:
        return json.load(f)


def save_json(data, path: Path, indent: int = 2) -> None:
    """Save data to JSON file, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
