"""
Shared utilities: paths, data loading, sampling, temperature helpers.
"""
from pathlib import Path
import json
import random

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_PATH = PROJECT_ROOT / "dataset" / "rxn_predict" / "condition_temperature_v2.json"
RESULTS_DIR  = PROJECT_ROOT / "results" / "cot_eval" / "rxn_predict" / "condition_temperature_structured"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Temperature helpers
# ---------------------------------------------------------------------------
VALID_TEMPS = {20.0, 60.0, 100.0}


def round_to_temp(value: float) -> float:
    """Round to nearest valid temperature: 20, 60, or 100."""
    return min(VALID_TEMPS, key=lambda t: abs(t - value))


def parse_gt(item: dict) -> float:
    return float(item["gt"])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def select_sample(data: list[dict], n: int = 60, seed: int = 42) -> list[dict]:
    """
    Stratified sample: 20 each from easy/medium/hard.
    No rxn_cls available → random shuffle within each stratum.
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
        random.shuffle(pool)
        picked = pool[:per_difficulty]
        sampled.extend(picked)

    random.shuffle(sampled)
    return sampled[:n]


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
