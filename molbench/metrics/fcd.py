"""Fréchet ChemNet Distance (FCD) — text -> SMILES generation. Lower is better.

Computed over predictions RDKit can canonicalize. Lazy import so evaluation
still works if fcd_torch is absent.
"""
from __future__ import annotations

from typing import List, Optional

from ..utils.chem import canonicalize


def fcd_metric(preds: List[str], golds: List[str], device: str = "cpu") -> Optional[float]:
    try:
        import numpy as np

        # fcd_torch 1.0.7 uses np.row_stack (removed in NumPy 2.0)
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
    return float(FCD(device=device, n_jobs=1)(can_golds, can_preds))
