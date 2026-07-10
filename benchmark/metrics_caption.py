"""Metrics for molecule captioning (SMILES -> text).

Follows the MolT5 / ChEBI-20 caption evaluation: BLEU-2, BLEU-4, ROUGE-1/2/L
and METEOR, all computed on lower-cased NLTK-tokenized text.
"""
from __future__ import annotations

from typing import Dict, List

import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


def _tok(text: str) -> List[str]:
    return nltk.word_tokenize(text.lower())


def caption_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """Compute captioning metrics. ``preds``/``refs`` are aligned lists."""
    assert len(preds) == len(refs), "preds and refs must be aligned"

    tok_preds = [_tok(p) for p in preds]
    tok_refs = [_tok(r) for r in refs]

    # corpus BLEU expects a list of reference-lists per hypothesis
    list_of_refs = [[r] for r in tok_refs]
    bleu2 = corpus_bleu(list_of_refs, tok_preds, weights=(0.5, 0.5))
    bleu4 = corpus_bleu(
        list_of_refs, tok_preds, weights=(0.25, 0.25, 0.25, 0.25)
    )

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    r1 = r2 = rl = 0.0
    meteor_total = 0.0
    n = len(preds)
    for pred, ref, tp, tr in zip(preds, refs, tok_preds, tok_refs):
        rs = scorer.score(ref, pred)
        r1 += rs["rouge1"].fmeasure
        r2 += rs["rouge2"].fmeasure
        rl += rs["rougeL"].fmeasure
        meteor_total += meteor_score([tr], tp)

    return {
        "n": n,
        "bleu2": bleu2,
        "bleu4": bleu4,
        "rouge1": r1 / n,
        "rouge2": r2 / n,
        "rougeL": rl / n,
        "meteor": meteor_total / n,
    }
