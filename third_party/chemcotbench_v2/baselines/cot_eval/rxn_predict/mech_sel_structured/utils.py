"""
Shared utilities: paths, data loading, sampling for mech_sel structured CoT.
"""
from pathlib import Path
import json
import re
import random
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_PATH = PROJECT_ROOT / "dataset" / "rxn_predict" / "mech_sel_v2.json"
RESULTS_DIR  = PROJECT_ROOT / "results" / "cot_eval" / "rxn_predict" / "mech_sel_structured"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def select_sample(data: list[dict], n: int = 60, seed: int = 42) -> list[dict]:
    """
    Stratified sample: 20 each from easy/medium/hard.
    Maximises rxn_cls diversity within each stratum.
    """
    random.seed(seed)
    per_difficulty = n // 3

    by_diff: dict[str, list[dict]] = {}
    for item in data:
        d = item.get("difficulty", "unknown")
        by_diff.setdefault(d, []).append(item)

    sampled = []
    for diff in ["easy", "medium", "hard"]:
        pool = by_diff.get(diff, [])
        pool_sorted = sorted(pool, key=lambda x: x.get("rxn_cls", ""))
        step = max(1, len(pool_sorted) // per_difficulty)
        picked = pool_sorted[::step][:per_difficulty]
        if len(picked) < per_difficulty:
            picked = pool_sorted[:per_difficulty]
        sampled.extend(picked)

    random.shuffle(sampled)
    return sampled[:n]


# ---------------------------------------------------------------------------
# GT parsing
# ---------------------------------------------------------------------------

def parse_gt(item: dict) -> str:
    """Return GT as string (capital letter)."""
    return str(item.get("gt", "")).strip()


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_valid_choices(query: str) -> set:
    """Extract set of valid choice letters from query (e.g., {'A','B','C','D','E','F'})."""
    return set(re.findall(r"\b([A-J]):", query))


# ---------------------------------------------------------------------------
# Reaction-type matching
# ---------------------------------------------------------------------------
import re as _re

_RXN_STOPWORDS = frozenset({
    'reaction', 'the', 'a', 'an', 'of', 'with', 'and', 'or', 'type', 'via',
    'using', 'from', 'to', 'by', 'in', 'for', 'catalyzed', 'mediated',
    'assisted', 'promoted', 'based',
})


def rxn_type_match(pred: str, gt: str, threshold: float = 0.5) -> bool:
    """
    Keyword-overlap based reaction type matching.
    Returns True if >= threshold fraction of GT key words appear (as substrings)
    in the normalized prediction string.
    """
    if not pred or not gt:
        return False

    def _key_words(s: str) -> list:
        s_norm = _re.sub(r'[^\w\s]', ' ', s.lower())
        return [w for w in _re.findall(r'\b\w+\b', s_norm)
                if len(w) >= 3 and w not in _RXN_STOPWORDS]

    gt_words = _key_words(gt)
    pred_norm = _re.sub(r'[^\w\s]', ' ', pred.lower())

    if not gt_words:
        return bool(pred.strip())

    matching = sum(1 for gw in gt_words if gw in pred_norm)
    return matching / len(gt_words) >= threshold
