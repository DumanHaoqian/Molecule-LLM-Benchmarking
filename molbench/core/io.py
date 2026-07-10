"""Transactional artifact I/O for resumable generation and evaluation."""
from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
import shutil
import socket
from typing import Any, Dict, Iterable, List, Sequence

from .task import EvalRecord

SCHEMA_VERSION = 2


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def example_id(example: Dict[str, Any]) -> str:
    return stable_hash(example)


def dataset_digest(indexed_examples: Sequence[tuple[int, Dict[str, Any]]]) -> str:
    h = hashlib.sha256()
    for idx, example in indexed_examples:
        h.update(f"{idx}:".encode("ascii"))
        h.update(example_id(example).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def pred_stem(out_dir: str, benchmark: str, model: str, task: str, split: str) -> str:
    return os.path.join(out_dir, f"{benchmark}__{model}__{task}__{split}")


def pred_paths(out_dir: str, benchmark: str, model: str, task: str, split: str):
    stem = pred_stem(out_dir, benchmark, model, task, split)
    return stem + ".jsonl", stem + ".meta.json"


@dataclass(frozen=True)
class ArtifactPaths:
    stem: str

    @property
    def final(self) -> str:
        return self.stem + ".jsonl"

    @property
    def partial(self) -> str:
        return self.stem + ".partial.jsonl"

    @property
    def meta(self) -> str:
        return self.stem + ".meta.json"

    @property
    def manifest(self) -> str:
        return self.stem + ".run.json"

    @property
    def lock(self) -> str:
        return self.stem + ".lock"


def paths_for(out_dir: str, benchmark: str, model: str, task: str, split: str) -> ArtifactPaths:
    return ArtifactPaths(pred_stem(out_dir, benchmark, model, task, split))


def record_to_row(record: EvalRecord, scores: Dict[str, Any] | None = None) -> Dict[str, Any]:
    row = {
        "schema_version": SCHEMA_VERSION,
        "example_index": record.example_index,
        "example_id": record.example_id,
        "example": record.example,
        "prompt": record.prompt,
        "raw_output": record.raw_output,
        "prediction": record.prediction,
        "generation_metadata": record.generation_metadata,
    }
    if scores is not None:
        row["scores"] = scores
    return row


def row_to_record(row: Dict[str, Any]) -> EvalRecord:
    if row.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported artifact schema {row.get('schema_version')!r}; "
            "run scripts/migrate_legacy_artifacts.py first"
        )
    required = {"example_index", "example_id", "example", "prediction"}
    missing = required - row.keys()
    if missing:
        raise ValueError(f"artifact row missing fields: {sorted(missing)}")
    expected = example_id(row["example"])
    if row["example_id"] != expected:
        raise ValueError(
            f"example hash mismatch at index {row['example_index']}: "
            f"{row['example_id']} != {expected}"
        )
    return EvalRecord(
        example=row["example"],
        prompt=row.get("prompt", ""),
        raw_output=row.get("raw_output", ""),
        prediction=row.get("prediction"),
        example_index=int(row["example_index"]),
        example_id=row["example_id"],
        generation_metadata=row.get("generation_metadata", {}),
    )


def _fsync_directory(path: str) -> None:
    try:
        fd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        # WSL 9p may reject directory fsync. File fsync remains mandatory.
        print(f"[io] warning: directory fsync unavailable: {exc}")


def atomic_write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_directory(path)


def atomic_write_text(path: str, value: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_directory(path)


def atomic_write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_directory(path)


def _load_jsonl(path: str, repair_trailing: bool = False) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    mode = "rb+" if repair_trailing else "rb"
    rows: List[Dict[str, Any]] = []
    with open(path, mode) as f:
        lines = f.readlines()
        valid_bytes = 0
        for i, raw in enumerate(lines):
            if not raw.strip():
                valid_bytes += len(raw)
                continue
            try:
                rows.append(json.loads(raw.decode("utf-8")))
                valid_bytes += len(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if repair_trailing and i == len(lines) - 1 and not raw.endswith(b"\n"):
                    print(f"[resume] truncating incomplete final JSONL row in {path}")
                    f.seek(valid_bytes)
                    f.truncate()
                    f.flush()
                    os.fsync(f.fileno())
                    break
                raise ValueError(f"corrupt JSONL row {i + 1} in {path}: {exc}") from exc
    return rows


def load_rows_by_index(path: str, repair_trailing: bool = False) -> Dict[int, Dict[str, Any]]:
    by_index: Dict[int, Dict[str, Any]] = {}
    for row in _load_jsonl(path, repair_trailing=repair_trailing):
        record = row_to_record(row)
        previous = by_index.get(record.example_index)
        if previous is not None:
            if canonical_json(previous) != canonical_json(row):
                raise ValueError(f"conflicting duplicate index {record.example_index} in {path}")
            raise ValueError(f"duplicate index {record.example_index} in {path}")
        by_index[record.example_index] = row
    return by_index


def load_partial_rows(path: str) -> Dict[int, Dict[str, Any]]:
    return load_rows_by_index(path, repair_trailing=True)


def read_records(path: str) -> List[EvalRecord]:
    rows = _load_jsonl(path)
    records = [row_to_record(row) for row in rows]
    indexes = [r.example_index for r in records]
    if indexes != sorted(indexes) or len(indexes) != len(set(indexes)):
        raise ValueError(f"final artifact is not uniquely sorted by example_index: {path}")
    return records


class PartialWriter(AbstractContextManager):
    """Append-only JSONL writer durable at every completed batch."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._file = open(path, "a", encoding="utf-8")

    def append(self, rows: Iterable[Dict[str, Any]]) -> None:
        for row in rows:
            self._file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class RunLock(AbstractContextManager):
    """Cross-process lock implemented with atomic directory creation."""

    def __init__(self, path: str, allow_stale: bool = True):
        self.path = path
        self.allow_stale = allow_stale
        self.owner = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.acquired = False

    def _owner_is_alive(self, owner: Dict[str, Any]) -> bool:
        if owner.get("hostname") != socket.gethostname():
            return True
        pid = owner.get("pid")
        if not isinstance(pid, int):
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def acquire(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        try:
            os.mkdir(self.path)
        except FileExistsError:
            owner_path = os.path.join(self.path, "owner.json")
            try:
                with open(owner_path, encoding="utf-8") as f:
                    owner = json.load(f)
            except (OSError, json.JSONDecodeError):
                age = max(0.0, datetime.now().timestamp() - os.stat(self.path).st_mtime)
                if age < 30:
                    raise RuntimeError(
                        f"artifact lock is still being initialized: {self.path}"
                    )
                owner = {"hostname": socket.gethostname(), "pid": None}
            if self._owner_is_alive(owner) or not self.allow_stale:
                raise RuntimeError(f"artifact is locked by {owner}: {self.path}")
            stale = (
                self.path
                + ".stale."
                + datetime.now().strftime("%Y%m%d-%H%M%S")
                + f".{os.getpid()}"
            )
            os.replace(self.path, stale)
            os.mkdir(self.path)
        atomic_write_json(os.path.join(self.path, "owner.json"), self.owner)
        self.acquired = True

    def release(self) -> None:
        if self.acquired:
            shutil.rmtree(self.path, ignore_errors=False)
            self.acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


def finalize_partial(
    paths: ArtifactPaths, rows_by_index: Dict[int, Dict[str, Any]], expected: Sequence[int]
) -> None:
    expected_set = set(expected)
    actual_set = set(rows_by_index)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)[:20]
        extra = sorted(actual_set - expected_set)[:20]
        raise ValueError(f"cannot finalize; missing={missing} extra={extra}")
    ordered = [rows_by_index[i] for i in sorted(expected)]
    atomic_write_jsonl(paths.final, ordered)
    if os.path.exists(paths.partial):
        os.unlink(paths.partial)
        _fsync_directory(paths.partial)


def write_records(path: str, records: List[EvalRecord]) -> None:
    atomic_write_jsonl(path, (record_to_row(r) for r in records))


def write_meta(path: str, meta: Dict[str, Any]) -> None:
    atomic_write_json(path, meta)


def archive_paths(paths: ArtifactPaths, reason: str = "restart") -> str | None:
    existing = [
        p for p in (paths.final, paths.partial, paths.meta, paths.manifest) if os.path.exists(p)
    ]
    if not existing:
        return None
    root = os.path.dirname(paths.stem)
    run_name = os.path.basename(paths.stem)
    history = os.path.join(
        root,
        "history",
        datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{reason}-{run_name}",
    )
    os.makedirs(history, exist_ok=False)
    for path in existing:
        shutil.move(path, os.path.join(history, os.path.basename(path)))
    _fsync_directory(history)
    return history


def read_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
