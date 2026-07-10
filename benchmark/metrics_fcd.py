"""Fréchet ChemNet Distance (FCD) for text -> SMILES generation.

FCD compares the ChemNet-activation distributions of the generated molecules
and the reference molecules; lower is better. Computed only over predictions
that RDKit can canonicalize (invalid SMILES are dropped, following MolT5).

Requires ``fcd_torch`` (installed in the ChEBI-20-Eva venv). Import is lazy so
the rest of evaluation still works if it is unavailable.
"""
from __future__ import annotations

from typing import List, Optional

from .smiles_utils import canonicalize


def fcd_metric(
    preds: List[str], golds: List[str], device: str = "cuda"
) -> Optional[float]:
    """Return FCD between valid predicted SMILES and gold SMILES, or None."""
    try:
        import numpy as np

        # fcd_torch 1.0.7 calls np.row_stack, removed in NumPy 2.0
        if not hasattr(np, "row_stack"):
            np.row_stack = np.vstack
        from fcd_torch import FCD
    except Exception as e:  # pragma: no cover
        print(f"[fcd] unavailable ({e}); skipping")
        return None

    can_preds = [c for c in (canonicalize(p) for p in preds) if c]
    can_golds = [c for c in (canonicalize(g) for g in golds) if c]
    if not can_preds or not can_golds:
        return None

    fcd = FCD(device=device, n_jobs=1)
    return float(fcd(can_golds, can_preds))
