"""Shared file-path conventions for prediction / metadata artifacts.

Kept dependency-free so the evaluation stage (ChEBI-20-Eva venv, no `datasets`)
can import it without pulling in the generation stack.
"""
from __future__ import annotations

import os


def pred_paths(out_dir: str, model_key: str, task: str, split: str):
    """Return (jsonl_path, meta_json_path) for a (model, task, split)."""
    stem = os.path.join(out_dir, f"{model_key}__{task}__{split}")
    return stem + ".jsonl", stem + ".meta.json"
