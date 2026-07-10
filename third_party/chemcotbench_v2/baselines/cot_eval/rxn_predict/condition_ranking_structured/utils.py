"""
Shared utilities: paths, GT parsing, ranking metrics, data loading.
V3: added get_top2_diff, COMPONENT_KEYWORDS, differentiator_correct,
    parse_condition_scores, scores_to_ranking.
"""
from pathlib import Path
import json
import math
import random
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_PATH = PROJECT_ROOT / "dataset" / "rxn_predict" / "condition_ranking_v2.json"
RESULTS_DIR  = PROJECT_ROOT / "results" / "cot_eval" / "rxn_predict" / "condition_ranking_structured"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def parse_gt(item: dict) -> list:
    """Returns GT ranking as list like ['A', 'C', 'B', 'D', 'E']."""
    gt = item["gt"]
    if isinstance(gt, list):
        return [str(x).strip().upper() for x in gt]
    if isinstance(gt, str):
        import ast as _ast
        try:
            return [str(x).strip().upper() for x in json.loads(gt)]
        except Exception:
            pass
        try:
            return [str(x).strip().upper() for x in _ast.literal_eval(gt)]
        except Exception:
            pass
        items = re.findall(r'[A-E]', gt)
        return [x.upper() for x in items]
    return []


def select_sample(data: list[dict], n: int = 60, seed: int = 42) -> list[dict]:
    """
    Stratified sample: n//3 each from easy/medium/hard.
    No rxn_cls field — shuffles randomly within each stratum.
    """
    random.seed(seed)
    per_diff = n // 3

    by_diff: dict[str, list[dict]] = {}
    for item in data:
        d = item.get("difficulty", "unknown")
        by_diff.setdefault(d, []).append(item)

    sampled = []
    for diff in ["easy", "medium", "hard"]:
        pool = by_diff.get(diff, [])[:]
        random.shuffle(pool)
        picked = pool[:per_diff]
        sampled.extend(picked)

    random.shuffle(sampled)
    return sampled[:n]


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def ndcg_score(pred_rank: list, gt_rank: list, k: int = 5) -> float:
    """
    NDCG score where gt_rank[0] is best (relevance n, n-1, ..., 1).
    pred_rank and gt_rank are lists of label strings like ['A', 'C', 'B', 'D', 'E'].
    """
    if not pred_rank or not gt_rank:
        return 0.0
    n = len(gt_rank)
    rel = {item: (n - i) for i, item in enumerate(gt_rank)}

    def dcg(rank: list) -> float:
        score = 0.0
        for i, item in enumerate(rank[:k], start=1):
            score += rel.get(item, 0) / math.log2(i + 1)
        return score

    ideal = dcg(gt_rank)
    actual = dcg(pred_rank)
    return round(actual / ideal if ideal > 0 else 0.0, 4)


def mrr_score(pred_rank: list, best_item: str) -> float:
    """Mean Reciprocal Rank for the GT best item (gt_rank[0])."""
    try:
        rank = [x.upper() for x in pred_rank].index(best_item.upper()) + 1
        return round(1.0 / rank, 4)
    except ValueError:
        return 0.0


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


# ---------------------------------------------------------------------------
# V3 additions: SMILES-based component diff and condition scoring utilities
# ---------------------------------------------------------------------------

COMPONENT_KEYWORDS: dict[str, list[str]] = {
    'reagent': [
        'reagent', 'base', 'additive', 'isoxazole', 'phosphazene', 'guanidine',
        'amidine', 'oxidant', 'reductant', 'activator', 'proton', 'nucleophil',
        'electrophil', 'auxiliary', 'nitrogen', 'amine',
    ],
    'catalyst': [
        'catalyst', 'ligand', 'palladium', ' pd', 'pd(', 'phosphine', 'phosphin',
        'nickel', ' ni', 'ni(', 'metal', 'precatalyst', 'ancillary', 'bidentate',
        'monodentate', 'dppf', 'binap', 'xantphos', 'davephos', 'sphos', 'ruphos',
        'biphenyl', 'bulky', 'steric', 'electronic',
    ],
    'solvent': [
        'solvent', 'polar', 'nonpolar', 'dmf', 'thf', 'toluene', 'acetonitrile',
        'dioxane', 'dmso', 'dielectric', 'protic', 'aprotic', 'ethereal',
    ],
}


def get_top2_diff(item: dict) -> list[str]:
    """
    Compare SMILES component sets between GT Rank#1 and Rank#2 conditions.
    Splits each field (reagent/catalyst/solvent) by '.' to get component sets,
    then returns the list of field names that actually differ.

    Returns: subset of ['reagent', 'catalyst', 'solvent'].
    """
    gt_rank = parse_gt(item)
    if len(gt_rank) < 2:
        return []

    label1, label2 = gt_rank[0], gt_rank[1]
    cands = {c['label']: c['conditions'] for c in item['meta']['candidates']}

    if label1 not in cands or label2 not in cands:
        return []

    diffs = []
    for field in ['reagent', 'catalyst', 'solvent']:
        s1 = set(cands[label1].get(field, '').split('.')) - {''}
        s2 = set(cands[label2].get(field, '').split('.')) - {''}
        if s1 != s2:
            diffs.append(field)
    return diffs


def differentiator_correct(key_diff_text: str, actual_diffs: list[str]) -> bool:
    """
    V4: Check whether [KEY_DIFFERENTIATOR] text mentions the correct component
    type, as determined by the actual SMILES-level diff between GT top-2.

    Returns True if any keyword belonging to an actually-differing component
    type appears (as substring) in the lowercased key_diff_text.
    """
    if not key_diff_text or not actual_diffs:
        return False

    text_lower = key_diff_text.lower()
    for diff_field in actual_diffs:
        keywords = COMPONENT_KEYWORDS.get(diff_field, [])
        if any(kw in text_lower for kw in keywords):
            return True
    return False


def parse_condition_scores(text: str) -> dict[str, float]:
    """
    Parse [CONDITION_SCORES] field text such as:
      'A:4 B:3 C:5 D:2 E:1'
      'A:[4] B:[3] C:[5] D:[2] E:[1]'
      'A=4 B=3 ...'
    Returns a dict like {'A': 4.0, 'B': 3.0, 'C': 5.0, 'D': 2.0, 'E': 1.0}.
    """
    scores: dict[str, float] = {}
    # Allow optional colon/equals/dash, then optional '[', then the number
    pattern = r'([A-E])\s*[:\-=]?\s*\[?\s*(\d+(?:\.\d+)?)'
    for m in _re.finditer(pattern, text.upper()):
        label = m.group(1)
        score = float(m.group(2))
        scores[label] = score
    return scores


def scores_consistent_with_ranking(scores: dict[str, float],
                                   ranking: list[str]) -> bool:
    """
    V2: Check that the ordering implied by scores (descending) is weakly
    consistent with ranking.  Specifically: for every pair (i < j) in
    ranking, scores[ranking[i]] >= scores[ranking[j]].
    Ties (equal scores) are acceptable; strict inversion is not.

    Returns False if any earlier-ranked label has a strictly lower score
    than a later-ranked label.
    """
    if not scores or len(scores) < 5 or not ranking or len(ranking) < 5:
        return False
    for i in range(len(ranking)):
        for j in range(i + 1, len(ranking)):
            li, lj = ranking[i], ranking[j]
            if li in scores and lj in scores:
                if scores[li] < scores[lj]:   # earlier rank has lower score → inversion
                    return False
    return True
