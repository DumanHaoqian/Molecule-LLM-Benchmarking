#!/usr/bin/env python3
"""Explicitly migrate legacy prediction JSONL files to artifact schema v2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from molbench.core.io import (  # noqa: E402
    SCHEMA_VERSION,
    atomic_write_json,
    atomic_write_jsonl,
    example_id,
    stable_hash,
)


def legacy_example(row, task):
    if "example" in row:
        return row["example"]
    if not {"input", "target"} <= row.keys():
        raise ValueError("legacy row has neither example nor input/target fields")
    if task == "captioning":
        return {"CAN_SMILES": row["input"], "DESCRIPTION": row["target"]}
    if task == "caption2smiles":
        return {"DESCRIPTION": row["input"], "CAN_SMILES": row["target"]}
    raise ValueError("input/target migration only supports ChEBI captioning tasks")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_paths(destination: Path):
    raw = str(destination)
    if not raw.endswith(".jsonl"):
        raise ValueError("destination must end with .jsonl")
    stem = raw[: -len(".jsonl")]
    return Path(stem + ".run.json"), Path(stem + ".meta.json")


def migrate(
    source: Path,
    destination: Path,
    task: str,
    benchmark: str,
    model_key: str,
    split: str,
) -> None:
    rows = []
    with source.open(encoding="utf-8") as f:
        for fallback_index, line in enumerate(f):
            if not line.strip():
                continue
            old = json.loads(line)
            example = legacy_example(old, task)
            index = int(old.get("example_index", old.get("idx", fallback_index)))
            migrated = {
                "schema_version": SCHEMA_VERSION,
                "example_index": index,
                "example_id": example_id(example),
                "example": example,
                "prompt": old.get("prompt", ""),
                "raw_output": old.get("raw_output", ""),
                "prediction": old.get("prediction"),
                "generation_metadata": {"migrated_from": str(source)},
            }
            if "scores" in old:
                migrated["scores"] = old["scores"]
            rows.append(migrated)
    rows.sort(key=lambda row: row["example_index"])
    if len({row["example_index"] for row in rows}) != len(rows):
        raise ValueError("legacy artifact contains duplicate indexes")
    source_sha = file_sha256(source)
    migration = {
        "schema_version": SCHEMA_VERSION,
        "kind": "generation",
        "benchmark": benchmark,
        "model_key": model_key,
        "task": task,
        "split": split,
        "n": len(rows),
        "source": str(source.resolve()),
        "source_sha256": source_sha,
        "migration_tool": "scripts/migrate_legacy_artifacts.py",
    }
    manifest_path, meta_path = sidecar_paths(destination)
    manifest = {
        **migration,
        "fingerprint": stable_hash(migration),
        "status": "complete",
    }
    atomic_write_jsonl(str(destination), rows)
    atomic_write_json(str(meta_path), migration)
    atomic_write_json(str(manifest_path), manifest)
    print(f"migrated {len(rows)} rows -> {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--task", required=True, choices=("captioning", "caption2smiles"))
    parser.add_argument("--benchmark", default="chebi20")
    parser.add_argument("--model-key", default="legacy")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    manifest_path, meta_path = sidecar_paths(args.destination)
    existing = [path for path in (args.destination, manifest_path, meta_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing artifacts: {existing}")
    migrate(
        args.source,
        args.destination,
        args.task,
        args.benchmark,
        args.model_key,
        args.split,
    )


if __name__ == "__main__":
    main()
