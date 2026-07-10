"""Compatibility helpers for the legacy TDC chemistry oracles."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[2]
ORACLE_DIR = REPO_ROOT / "oracle"
ORACLE_MANIFEST = REPO_ROOT / "resources" / "chemcotbench" / "oracle_sources.json"


def install_rdkit_six_compat() -> None:
    """Restore the sole rdkit.six API used by PyTDC on modern RDKit."""
    try:
        import rdkit.six  # type: ignore  # noqa: F401
        return
    except ImportError:
        pass
    module = types.ModuleType("rdkit.six")
    module.iteritems = lambda mapping: iter(mapping.items())
    sys.modules["rdkit.six"] = module


def _verify_oracle(name: str) -> None:
    manifest = json.loads(ORACLE_MANIFEST.read_text(encoding="utf-8"))
    filename = {
        "drd2": "drd2_current.pkl",
        "gsk3b": "gsk3b_current.pkl",
        "jnk3": "jnk3_current.pkl",
    }.get(name.lower())
    if filename is None:
        return
    expected = manifest["files"][filename]
    path = ORACLE_DIR / filename
    if not path.exists() or path.stat().st_size != expected["bytes"]:
        raise RuntimeError(f"unexpected TDC oracle cache file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected["sha256"]:
        raise RuntimeError(f"TDC oracle checksum mismatch: {path}")


@lru_cache(maxsize=None)
def oracle(name: str):
    install_rdkit_six_compat()
    from tdc import Oracle
    import tdc.chem_utils.oracle.oracle as evaluator_module
    import tdc.oracles as oracle_module

    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    original_load = getattr(
        oracle_module, "_molbench_original_oracle_load", oracle_module.oracle_load
    )
    oracle_module._molbench_original_oracle_load = original_load
    oracle_module.oracle_load = lambda oracle_name: original_load(
        oracle_name, path=str(ORACLE_DIR)
    )
    original_pickle_load = getattr(
        evaluator_module,
        "_molbench_original_load_pickled_model",
        evaluator_module.load_pickled_model,
    )
    evaluator_module._molbench_original_load_pickled_model = original_pickle_load
    evaluator_module.load_pickled_model = lambda path: original_pickle_load(
        str(ORACLE_DIR / Path(path).name)
    )
    result = Oracle(name=name)
    _verify_oracle(name)
    return result
