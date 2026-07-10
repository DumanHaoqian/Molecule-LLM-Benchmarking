"""Text2Mol metric (Edwards et al., 2021) — cross-modal SMILES<->text similarity.

Text2Mol scores how well a molecule and a caption correspond by embedding both
into a shared space with a pretrained association model (SciBERT text encoder +
GCN molecule encoder over mol2vec features) and taking cosine similarity:

  * captioning     : sim(generated_caption, gold_molecule)   [higher better]
  * caption2smiles : sim(generated_molecule, gold_caption)   [higher better]

The pretrained checkpoint + embedding artifacts are NOT bundled. Point the
harness at them via the ``TEXT2MOL_DIR`` env var (a directory containing the
Text2Mol checkpoint and mol2vec/token embeddings). When unavailable this returns
``None`` and the metric is shown as "—" in the tables rather than crashing.

Resources & upstream code: https://github.com/cnedwards/text2mol
"""
from __future__ import annotations

import os
from typing import List, Optional

_MODEL = None
_LOAD_ERROR: Optional[str] = None


def _try_load():
    """Lazily load the Text2Mol model. Returns model or None (sets _LOAD_ERROR)."""
    global _MODEL, _LOAD_ERROR
    if _MODEL is not None or _LOAD_ERROR is not None:
        return _MODEL

    t2m_dir = os.environ.get("TEXT2MOL_DIR")
    if not t2m_dir or not os.path.isdir(t2m_dir):
        _LOAD_ERROR = "TEXT2MOL_DIR not set / not found"
        return None
    try:
        from .text2mol_model import load_text2mol  # local loader (eval venv only)

        _MODEL = load_text2mol(t2m_dir)
        return _MODEL
    except Exception as e:  # pragma: no cover
        _LOAD_ERROR = f"failed to load Text2Mol: {e}"
        return None


def text2mol_metric(
    mols: List[str], texts: List[str], device: str = "cuda"
) -> Optional[float]:
    """Mean cosine similarity between paired (molecule SMILES, caption).

    ``mols`` and ``texts`` are aligned. For captioning pass (gold_smiles,
    generated_caption); for caption2smiles pass (generated_smiles, gold_caption).
    Returns None if the pretrained model is unavailable.
    """
    model = _try_load()
    if model is None:
        print(f"[text2mol] unavailable ({_LOAD_ERROR}); skipping")
        return None
    return float(model.mean_similarity(mols, texts, device=device))
