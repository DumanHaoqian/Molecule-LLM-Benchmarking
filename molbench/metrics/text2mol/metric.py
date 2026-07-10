"""Text2Mol metric entry point — lazy singleton around the scorer.

Resolves resources from the ``TEXT2MOL_DIR`` env var. Returns None (metric
shown as "—") when the checkpoint/mol2vec model are unavailable, so evaluation
never hard-fails on a missing optional dependency.
"""
from __future__ import annotations

import os
from typing import List, Optional

_MODEL = None
_LOAD_ERROR: Optional[str] = None


def _try_load():
    global _MODEL, _LOAD_ERROR
    if _MODEL is not None or _LOAD_ERROR is not None:
        return _MODEL
    t2m_dir = os.environ.get("TEXT2MOL_DIR")
    if not t2m_dir or not os.path.isdir(t2m_dir):
        _LOAD_ERROR = "TEXT2MOL_DIR not set / not found"
        return None
    try:
        from .scorer import load_text2mol

        _MODEL = load_text2mol(t2m_dir)
        return _MODEL
    except Exception as e:  # pragma: no cover
        _LOAD_ERROR = f"failed to load Text2Mol: {e}"
        return None


def text2mol_score(
    mols: List[str], texts: List[str], device: str = "cpu"
) -> Optional[float]:
    """Mean cosine similarity between paired (molecule SMILES, caption)."""
    model = _try_load()
    if model is None:
        print(f"[text2mol] unavailable ({_LOAD_ERROR}); skipping")
        return None
    return float(model.mean_similarity(mols, texts, device=device))


def text2mol_scores(
    mols: List[str], texts: List[str], device: str = "cpu"
) -> Optional[List[Optional[float]]]:
    """Per-pair cosine similarities (None if model unavailable)."""
    model = _try_load()
    if model is None:
        return None
    return model.pair_similarities(mols, texts, device=device)
