"""Dataset loading for the ChEBI-20 benchmark (duongttr/chebi-20).

Columns of interest:
  * CAN_SMILES  : canonical SMILES        (input for captioning / target for gen)
  * DESCRIPTION : natural-language caption (target for captioning / input for gen)
"""
from __future__ import annotations

from typing import Optional

from datasets import load_dataset

CHEBI20_REPO = "duongttr/chebi-20"
SMILES_COL = "CAN_SMILES"
TEXT_COL = "DESCRIPTION"


def load_chebi20(split: str = "test", limit: Optional[int] = None):
    """Load a ChEBI-20 split, dropping the heavy PIL IMAGE column."""
    ds = load_dataset(CHEBI20_REPO, split=split)
    drop = [c for c in ("IMAGE",) if c in ds.column_names]
    if drop:
        ds = ds.remove_columns(drop)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds
