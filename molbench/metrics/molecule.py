"""Molecule-generation metrics (text -> SMILES).

BLEU (atom-level), exact match, Levenshtein, validity, and MACCS/RDK/Morgan
fingerprint Tanimoto similarity (over pairs where both molecules are valid).
"""
from __future__ import annotations

from typing import Dict, List

from nltk.translate.bleu_score import corpus_bleu
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys

from ..utils.chem import canonicalize, tokenize_smiles


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


def _fp_sims(gold_mol, pred_mol) -> Dict[str, float]:
    return {
        "maccs": DataStructs.FingerprintSimilarity(
            MACCSkeys.GenMACCSKeys(gold_mol), MACCSkeys.GenMACCSKeys(pred_mol)
        ),
        "rdk": DataStructs.FingerprintSimilarity(
            Chem.RDKFingerprint(gold_mol), Chem.RDKFingerprint(pred_mol)
        ),
        "morgan": DataStructs.TanimotoSimilarity(
            AllChem.GetMorganFingerprintAsBitVect(gold_mol, 2, nBits=2048),
            AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2, nBits=2048),
        ),
    }


def molgen_metrics(preds: List[str], golds: List[str]) -> Dict[str, float]:
    assert len(preds) == len(golds), "preds and golds must be aligned"
    n = len(preds)

    tok_preds = [tokenize_smiles(p) for p in preds]
    list_of_refs = [[tokenize_smiles(g)] for g in golds]
    bleu = corpus_bleu(list_of_refs, tok_preds)

    exact = valid = lev_total = 0
    maccs = rdk = morgan = 0.0
    fp_pairs = 0
    for pred, gold in zip(preds, golds):
        lev_total += _levenshtein(pred, gold)
        cp, cg = canonicalize(pred), canonicalize(gold)
        if cp is not None:
            valid += 1
        if cp is not None and cg is not None:
            if cp == cg:
                exact += 1
            sims = _fp_sims(Chem.MolFromSmiles(cg), Chem.MolFromSmiles(cp))
            maccs += sims["maccs"]
            rdk += sims["rdk"]
            morgan += sims["morgan"]
            fp_pairs += 1

    d = fp_pairs or 1
    return {
        "n": n,
        "bleu": bleu,
        "exact_match": exact / n,
        "levenshtein": lev_total / n,
        "validity": valid / n,
        "maccs_fts": maccs / d,
        "rdk_fts": rdk / d,
        "morgan_fts": morgan / d,
        "fp_pairs": fp_pairs,
    }
