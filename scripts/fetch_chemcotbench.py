#!/usr/bin/env python3
"""Download pinned ChemCoTBench snapshots and verify their canonical files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = REPO / "resources" / "chemcotbench"
SOURCES = json.loads((RESOURCE_ROOT / "sources.json").read_text(encoding="utf-8"))
V1_FILES = {
    "chemcotbench/mol_und/fg_count.json": 100,
    "chemcotbench/mol_und/ring_count.json": 20,
    "chemcotbench/mol_und/Murcko_scaffold.json": 40,
    "chemcotbench/mol_und/ring_system_scaffold.json": 60,
    "chemcotbench/mol_und/equivalence.json": 100,
    "chemcotbench/mol_edit/add.json": 20,
    "chemcotbench/mol_edit/delete.json": 20,
    "chemcotbench/mol_edit/sub.json": 60,
    "chemcotbench/mol_opt/drd.json": 100,
    "chemcotbench/mol_opt/gsk.json": 100,
    "chemcotbench/mol_opt/jnk.json": 100,
    "chemcotbench/mol_opt/logp.json": 100,
    "chemcotbench/mol_opt/qed.json": 100,
    "chemcotbench/mol_opt/solubility.json": 100,
    "chemcotbench/reaction/fs.json": 100,
    "chemcotbench/reaction/retro.json": 100,
    "chemcotbench/reaction/rcr.json": 90,
    "chemcotbench/reaction/nepp.json": 85,
    "chemcotbench/reaction/mechsel.json": 100,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def add_snapshot_hashes(root: Path, files: list[dict]) -> None:
    included = {entry["path"] for entry in files}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in included or relative == ".molbench-source.json":
            continue
        if ".cache" in path.relative_to(root).parts:
            continue
        files.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )


def validate_v1(root: Path) -> dict:
    files = []
    total = 0
    for relative, expected in V1_FILES.items():
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"V1 snapshot is missing {relative}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or len(rows) != expected:
            raise ValueError(f"V1 count mismatch for {relative}: {len(rows)} != {expected}")
        total += len(rows)
        files.append(
            {
                "path": relative,
                "n": len(rows),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    if total != SOURCES["chemcotbench"]["expected_samples"]:
        raise ValueError(f"V1 total mismatch: {total}")
    add_snapshot_hashes(root, files)
    return {"total_samples": total, "files": files}


def validate_v2(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("V2 snapshot is missing manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = SOURCES["chemcotbench-v2"]
    if manifest.get("total_samples") != source["expected_samples"]:
        raise ValueError("V2 manifest sample count mismatch")
    if len(manifest.get("files", [])) != source["expected_subtasks"]:
        raise ValueError("V2 manifest subtask count mismatch")
    seen = set()
    total = 0
    files = [
        {
            "path": "manifest.json",
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256(manifest_path),
        }
    ]
    for entry in manifest["files"]:
        raw_path = root / entry["raw_file"]
        process_path = root / entry["process_file"]
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        process = json.loads(process_path.read_text(encoding="utf-8"))
        raw_ids = [row.get("anonymous_sample_id") for row in raw]
        process_ids = [row.get("anonymous_sample_id") for row in process]
        expected = int(entry["n_samples"])
        if len(raw) != expected or len(process) != expected or raw_ids != process_ids:
            raise ValueError(
                f"V2 raw/process alignment failure for {entry['family']}/{entry['subtask']}"
            )
        duplicates = seen.intersection(raw_ids)
        if duplicates:
            raise ValueError(f"V2 duplicate sample IDs: {sorted(duplicates)[:5]}")
        seen.update(raw_ids)
        total += expected
        files.extend(
            [
                {"path": entry["raw_file"], "n": expected, "bytes": raw_path.stat().st_size, "sha256": sha256(raw_path)},
                {"path": entry["process_file"], "n": expected, "bytes": process_path.stat().st_size, "sha256": sha256(process_path)},
            ]
        )
    if total != source["expected_samples"] or len(seen) != total:
        raise ValueError(f"V2 total/ID mismatch: total={total} ids={len(seen)}")
    add_snapshot_hashes(root, files)
    return {
        "total_samples": total,
        "sample_ids": len(seen),
        "known_input_repairs": [
            {
                "anonymous_sample_id": "mol_und.smiles_equivalent.0097",
                "reason": "raw/viewer omit both model-facing SMILES; loader restores released canonical_a/canonical_b",
            }
        ],
        "files": files,
    }


def fetch(version: str, root: Path, token: str | None) -> None:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except ImportError as exc:
        raise RuntimeError("install huggingface_hub before fetching benchmark data") from exc

    key = "chemcotbench" if version == "v1" else "chemcotbench-v2"
    source = SOURCES[key]
    destination = root / version
    patterns = ["README.md"]
    if version == "v1":
        patterns.append("chemcotbench/**/*.json")
    else:
        patterns.extend(
            [
                "manifest.json",
                "raw_benchmark_data/**/*.json",
                "process_evaluation_data/**/*.json",
                "formal_templates/**",
                "prompt_templates/**",
                "verifier_rule_descriptions/**",
                "task_schema/**",
                "evaluation_split_metadata/**",
                "sample_examples/**",
            ]
        )
    try:
        snapshot_download(
            repo_id=source["repo_id"],
            repo_type="dataset",
            revision=source["revision"],
            local_dir=destination,
            allow_patterns=patterns,
            token=token,
        )
    except (GatedRepoError, HfHubHTTPError) as exc:
        if version == "v1":
            raise RuntimeError(
                "ChemCoTBench V1 is gated. Accept the dataset terms on Hugging Face "
                "and export an authorized HF_TOKEN before retrying."
            ) from exc
        raise
    validation = validate_v1(destination) if version == "v1" else validate_v2(destination)
    receipt = {
        "schema_version": 1,
        "version": version,
        "repo_id": source["repo_id"],
        "revision": source["revision"],
        **validation,
    }
    atomic_json(root / "manifests" / f"{version}.json", receipt)
    atomic_json(destination / ".molbench-source.json", receipt)
    print(f"[fetch] {key}: {validation['total_samples']} verified -> {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=("v1", "v2", "both"), default="both")
    parser.add_argument("--root", type=Path, default=RESOURCE_ROOT)
    args = parser.parse_args()
    versions = ("v1", "v2") if args.version == "both" else (args.version,)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    for version in versions:
        fetch(version, args.root.resolve(), token)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[fetch] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
