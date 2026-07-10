"""Metrics for text-based molecule generation (text -> SMILES).

Follows the MolT5 / ChEBI-20 generation evaluation:
  * BLEU              (atom-level SMILES BLEU)
  * exact match       (RDKit-canonical equality)
  * Levenshtein       (mean edit distance on raw SMILES)
  * validity          (fraction of predictions RDKit can parse)
  * MACCS / RDK / Morgan fingerprint Tanimoto similarity
    (averaged over pairs where both gold and prediction are valid)
"""
from __future__ import annotations

from typing import Dict, List

from nltk.translate.bleu_score import corpus_bleu
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys

from .smiles_utils import canonicalize, tokenize_smiles


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _fp_similarities(gold_mol, pred_mol) -> Dict[str, float]:
    maccs = DataStructs.FingerprintSimilarity(
        MACCSkeys.GenMACCSKeys(gold_mol), MACCSkeys.GenMACCSKeys(pred_mol)
    )
    rdk = DataStructs.FingerprintSimilarity(
        Chem.RDKFingerprint(gold_mol), Chem.RDKFingerprint(pred_mol)
    )
    morgan = DataStructs.TanimotoSimilarity(
        AllChem.GetMorganFingerprintAsBitVect(gold_mol, 2, nBits=2048),
        AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2, nBits=2048),
    )
    return {"maccs": maccs, "rdk": rdk, "morgan": morgan}


def molgen_metrics(preds: List[str], golds: List[str]) -> Dict[str, float]:
    """Compute generation metrics. ``preds``/``golds`` are raw SMILES strings."""
    assert len(preds) == len(golds), "preds and golds must be aligned"
    n = len(preds)

    # atom-level SMILES BLEU (over all pairs, on raw strings)
    tok_preds = [tokenize_smiles(p) for p in preds]
    list_of_refs = [[tokenize_smiles(g)] for g in golds]
    bleu = corpus_bleu(list_of_refs, tok_preds)

    exact = 0
    valid = 0
    lev_total = 0
    maccs_sum = rdk_sum = morgan_sum = 0.0
    fp_pairs = 0

    for pred, gold in zip(preds, golds):
        lev_total += _levenshtein(pred, gold)
        can_pred = canonicalize(pred)
        can_gold = canonicalize(gold)
        if can_pred is not None:
            valid += 1
        if can_pred is not None and can_gold is not None:
            if can_pred == can_gold:
                exact += 1
            sims = _fp_similarities(
                Chem.MolFromSmiles(can_gold), Chem.MolFromSmiles(can_pred)
            )
            maccs_sum += sims["maccs"]
            rdk_sum += sims["rdk"]
            morgan_sum += sims["morgan"]
            fp_pairs += 1

    fp_denom = fp_pairs if fp_pairs else 1
    return {
        "n": n,
        "bleu": bleu,
        "exact_match": exact / n,
        "levenshtein": lev_total / n,
        "validity": valid / n,
        "maccs_fts": maccs_sum / fp_denom,
        "rdk_fts": rdk_sum / fp_denom,
        "morgan_fts": morgan_sum / fp_denom,
        "fp_pairs": fp_pairs,
    }
