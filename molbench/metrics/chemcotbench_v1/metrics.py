"""Pure, resumable metric primitives matching ChemCoTBench V1 semantics."""
from __future__ import annotations

from functools import lru_cache
import math
from typing import Any, Dict, Iterable, List

from nltk.translate.bleu_score import corpus_bleu
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, MACCSkeys, QED
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


GROUP_SMARTS = {
    "benzene": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
    "benzene_ring": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
    "side-chain hydroxyls": "*-[O;D1]",
    "hydroxyl": "[OX2H]",
    "anhydride": "[CX3](=[OX1])[OX2][CX3](=[OX1])",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxyl": "[CX3](=O)[OX2H1]",
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "amide": "[NX3][CX3](=[OX1])[#6]",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
    "halo": "[F,Cl,Br,I]",
    "halogen": "[F,Cl,Br,I]",
    "thioether": "[SX2][CX4]",
    "nitrile": "[NX1]#[CX2]",
    "thiol": "[#16X2H]",
    "sulfide": "[#16X2H0]",
    "disulfide": "[#16X2H0][#16X2H0]",
    "sulfoxide": "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",
    "sulfone": "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",
    "borane": "[BX3]",
}


def canonical(smiles: str | None) -> str | None:
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    except Exception:
        return None


def _levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


def molecule_pair(prediction: str, reference: str) -> Dict[str, Any]:
    pred = canonical(prediction)
    ref = canonical(reference)
    result: Dict[str, Any] = {
        "valid": pred is not None,
        "exact_match": 0.0,
        "levenshtein": 0,
        "rdk_sims": 0.0,
        "maccs_sims": 0.0,
        "morgan_sims": 0.0,
    }
    if pred is None or ref is None:
        return result
    result["levenshtein"] = _levenshtein(pred, ref)
    pred_mol = Chem.MolFromSmiles(pred)
    ref_mol = Chem.MolFromSmiles(ref)
    try:
        result["exact_match"] = float(Chem.MolToInchi(pred_mol) == Chem.MolToInchi(ref_mol))
    except Exception:
        result["exact_match"] = float(pred == ref)
    result["rdk_sims"] = float(
        DataStructs.FingerprintSimilarity(
            Chem.RDKFingerprint(ref_mol), Chem.RDKFingerprint(pred_mol)
        )
    )
    result["maccs_sims"] = float(
        DataStructs.FingerprintSimilarity(
            MACCSkeys.GenMACCSKeys(ref_mol), MACCSkeys.GenMACCSKeys(pred_mol)
        )
    )
    result["morgan_sims"] = float(
        DataStructs.TanimotoSimilarity(
            AllChem.GetMorganFingerprint(ref_mol, 2),
            AllChem.GetMorganFingerprint(pred_mol, 2),
        )
    )
    return result


def scaffold_pair(prediction: str, reference: str) -> Dict[str, Any]:
    pred = canonical(prediction)
    ref = canonical(reference)
    if pred is None or ref is None:
        return {"valid": pred is not None, "scaffold_hard": 0.0, "scaffold_soft": 0.0}
    pred_scaffold = MurckoScaffoldSmiles(smiles=pred)
    ref_scaffold = MurckoScaffoldSmiles(smiles=ref)
    hard = float(pred_scaffold == ref_scaffold)
    soft = 0.0
    if pred_scaffold and ref_scaffold:
        soft = molecule_pair(pred_scaffold, ref_scaffold)["morgan_sims"]
    return {"valid": True, "scaffold_hard": hard, "scaffold_soft": soft}


def aggregate_molecule_pairs(
    predictions: List[str], references: List[str], scores: List[Dict[str, Any]]
) -> Dict[str, Any]:
    n = len(scores)
    refs = [[[character for character in canonical(ref) or ""]] for ref in references]
    preds = [[character for character in canonical(pred) or ""] for pred in predictions]
    usable = [(ref, pred) for ref, pred in zip(refs, preds) if ref[0] and pred]
    bleu = corpus_bleu(
        [item[0] for item in usable], [item[1] for item in usable]
    ) if usable else 0.0

    def avg(key: str) -> float:
        return sum(float(score.get(key, 0.0)) for score in scores) / n if n else 0.0

    return {
        "n": n,
        "exact_match": avg("exact_match"),
        "bleu": bleu,
        "levenshtein": avg("levenshtein"),
        "rdk_sims": avg("rdk_sims"),
        "maccs_sims": avg("maccs_sims"),
        "morgan_sims": avg("morgan_sims"),
        "validity": avg("valid"),
    }


def count_group(smiles: str, group: str, smarts: str | None = None) -> int | None:
    mol = Chem.MolFromSmiles(smiles or "")
    query = Chem.MolFromSmarts(smarts or GROUP_SMARTS.get(group.lower().strip(), ""))
    if mol is None or query is None:
        return None
    count = len(mol.GetSubstructMatches(query))
    if group.lower().strip() == "sulfide":
        disulfide = Chem.MolFromSmarts(GROUP_SMARTS["disulfide"])
        count -= len(mol.GetSubstructMatches(disulfide))
    return count


def edit_score(
    source: str,
    prediction: str,
    operation: str,
    added_group: str | None,
    removed_group: str | None,
) -> Dict[str, Any]:
    pred = canonical(prediction)
    src = canonical(source)
    if pred is None or src is None:
        return {"valid": False, "success": 0.0}
    added_before = count_group(src, added_group or "")
    added_after = count_group(pred, added_group or "")
    removed_before = count_group(src, removed_group or "")
    removed_after = count_group(pred, removed_group or "")
    success = False
    if operation == "add" and added_before is not None and added_after is not None:
        success = added_after == added_before + 1
    elif operation == "delete" and removed_before is not None and removed_after is not None:
        success = removed_after == removed_before - 1
    elif operation == "sub" and None not in (
        added_before,
        added_after,
        removed_before,
        removed_after,
    ):
        success = added_after == added_before + 1 and removed_after == removed_before - 1
    return {"valid": True, "success": float(success)}


class ESOLCalculator:
    def __init__(self):
        self.aromatic_query = Chem.MolFromSmarts("a")

    def __call__(self, smiles: str) -> float | None:
        mol = Chem.MolFromSmiles(smiles or "")
        if mol is None or not mol.GetNumAtoms():
            return None
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        rotors = Lipinski.NumRotatableBonds(mol)
        aromatic = len(mol.GetSubstructMatches(self.aromatic_query)) / mol.GetNumAtoms()
        return (
            0.26121066137801696
            - 0.0066138847738667125 * mw
            - 0.7416739523408995 * logp
            + 0.003451545565957996 * rotors
            - 0.42624840441316975 * aromatic
        )


@lru_cache(maxsize=None)
def property_oracle(prop: str):
    normalized = prop.lower()
    if normalized == "qed":
        return lambda smiles: QED.qed(Chem.MolFromSmiles(smiles))
    if normalized == "logp":
        return lambda smiles: Descriptors.MolLogP(Chem.MolFromSmiles(smiles))
    if normalized == "solubility":
        return ESOLCalculator()
    try:
        from ...utils.tdc import oracle
    except ImportError as exc:
        raise RuntimeError(
            f"TDC is required for the {prop} ChemCoTBench oracle"
        ) from exc
    names = {"drd": "drd2", "gsk": "gsk3b", "jnk": "jnk3"}
    return oracle(names.get(normalized, normalized))


def optimization_score(source: str, prediction: str, prop: str) -> Dict[str, Any]:
    src = canonical(source)
    pred = canonical(prediction)
    if src is None or pred is None:
        return {
            "valid": False,
            "valid_smiles": False,
            "valid_score": False,
            "improvement": 0.0,
            "success": 0.0,
            "best": 0.0,
            "scaffold_hard": 0.0,
            "scaffold_soft": 0.0,
        }
    oracle = property_oracle(prop)
    try:
        src_value = oracle(src)
        pred_value = oracle(pred)
    except Exception:
        src_value = pred_value = None
    if src_value is None or pred_value is None or not all(
        math.isfinite(float(value)) for value in (src_value, pred_value)
    ):
        improvement = 0.0
        valid_score = False
    else:
        improvement = float(pred_value) - float(src_value)
        valid_score = True
    threshold = 0.5 if prop.lower() in {"logp", "solubility"} else 0.3
    src_scaffold = MurckoScaffoldSmiles(smiles=src)
    pred_scaffold = MurckoScaffoldSmiles(smiles=pred)
    hard = float(src_scaffold == pred_scaffold)
    soft = molecule_pair(pred_scaffold, src_scaffold)["morgan_sims"] if src_scaffold and pred_scaffold else 0.0
    return {
        "valid": valid_score,
        "valid_smiles": True,
        "valid_score": valid_score,
        "improvement": improvement,
        "success": float(improvement > 0),
        "best": float(improvement >= threshold),
        "scaffold_hard": hard,
        "scaffold_soft": soft,
    }


def mean_scores(scores: Iterable[Dict[str, Any]], keys: Iterable[str]) -> Dict[str, float]:
    rows = list(scores)
    return {
        key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows) if rows else 0.0
        for key in keys
    }


def aggregate_optimization(scores: Iterable[Dict[str, Any]], prop: str) -> Dict[str, Any]:
    """Reproduce V1's P5/P95 winsorized property-improvement summary."""
    import numpy as np

    rows = list(scores)
    improvements = [float(row.get("improvement", 0.0)) for row in rows]
    if improvements:
        lower, upper = np.percentile(improvements, [5, 95])
        winsorized = np.clip(improvements, lower, upper)
        mean = float(np.mean(winsorized))
        variance = float(np.var(winsorized))
        minimum = float(np.min(winsorized))
        maximum = float(np.max(winsorized))
    else:
        mean = variance = minimum = maximum = 0.0
    means = mean_scores(
        rows,
        [
            "valid_smiles",
            "valid_score",
            "success",
            "best",
            "scaffold_hard",
            "scaffold_soft",
        ],
    )
    return {
        "n": len(rows),
        "validity": means["valid_smiles"],
        "valid_smiles_rate": means["valid_smiles"],
        "valid_score_rate": means["valid_score"],
        "success_rate": means["success"],
        "best_rate": means["best"],
        "mean_improvement": mean,
        "improvement_variance": variance,
        "improvement_min": minimum,
        "improvement_max": maximum,
        "scaffold_hard": means["scaffold_hard"],
        "scaffold_soft": means["scaffold_soft"],
        "property": prop,
    }
