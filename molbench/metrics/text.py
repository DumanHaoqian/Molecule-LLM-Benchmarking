"""Text-generation metrics (molecule captioning): BLEU-2/4, ROUGE, METEOR."""
from __future__ import annotations

from typing import Dict, List

import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

_SMOOTH = SmoothingFunction().method1


def _tok(text: str) -> List[str]:
    return nltk.word_tokenize(text.lower())


def caption_metrics(
    preds: List[str],
    refs: List[str],
    per_example: List[Dict[str, float]] | None = None,
) -> Dict[str, float]:
    assert len(preds) == len(refs), "preds and refs must be aligned"
    tok_preds = [_tok(p) for p in preds]
    tok_refs = [_tok(r) for r in refs]
    list_of_refs = [[r] for r in tok_refs]

    bleu2 = corpus_bleu(list_of_refs, tok_preds, weights=(0.5, 0.5))
    bleu4 = corpus_bleu(list_of_refs, tok_preds, weights=(0.25, 0.25, 0.25, 0.25))

    n = len(preds)
    if per_example is None:
        per_example = caption_metrics_per_example(preds, refs)

    def avg(key: str) -> float:
        return sum(row[key] for row in per_example) / n if n else 0.0

    return {
        "n": n,
        "bleu2": bleu2,
        "bleu4": bleu4,
        "rouge1": avg("rouge1"),
        "rouge2": avg("rouge2"),
        "rougeL": avg("rougeL"),
        "meteor": avg("meteor"),
    }


def caption_metrics_per_example(preds: List[str], refs: List[str]) -> List[Dict[str, float]]:
    """Per-example caption scores (sentence-level BLEU, ROUGE, METEOR)."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    out = []
    for pred, ref in zip(preds, refs):
        tp, tr = _tok(pred), _tok(ref)
        rs = scorer.score(ref, pred)
        out.append({
            "bleu2": sentence_bleu([tr], tp, weights=(0.5, 0.5), smoothing_function=_SMOOTH),
            "bleu4": sentence_bleu([tr], tp, weights=(0.25,) * 4, smoothing_function=_SMOOTH),
            "rouge1": rs["rouge1"].fmeasure,
            "rouge2": rs["rouge2"].fmeasure,
            "rougeL": rs["rougeL"].fmeasure,
            "meteor": meteor_score([tr], tp),
        })
    return out
