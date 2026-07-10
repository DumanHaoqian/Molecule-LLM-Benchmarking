"""
Verifier for rxn_pred/rcr_catalyst formal A->B CoT output.

Checkpoint design
─────────────────
  S1_rxn_class_exact (Type I, GATES)
    — step1 RXN_CLASS must exactly match GT coarse_rxn_cls (one of 9 categories).

  S2_transformation_logic (Type I, INFO ONLY)
    — CORE_TRANSFORMATION should contain keywords compatible with rxn_cls.
      This is useful diagnostic signal but too brittle to gate.

  S3_catalyst_role_presence (Type I, INFO ONLY)
    — CATALYST_ROLE is non-empty (presence check only).

  S4_class_grounding (Type II / GT-backed heuristic, GATES)
    — CATALYST_CLASS should match the catalyst class inferred from the GT catalyst SMILES.

  S5_class_self_consistency (Type I, GATES)
    — CATALYST_CLASS should match the class inferred from the predicted catalyst SMILES.

  S6_mol_valid (Type I, GATES)
    — predicted catalyst SMILES must be RDKit-parseable fragment-wise.

  S7_answer_consistent (Type I, GATES)
    — Answer line must match step5 PREDICTED_CATALYST_SMILES.

  outcome (Type I, GATES)
    — predicted catalyst SMILES exactly matches GT catalyst SMILES under canonical set match.

  all_pass = S1_coarse AND S4 AND S5 AND S6 AND S7 AND outcome
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem

from baselines.cot_eval.rxn_predict.rcr_catalyst_structured.utils import (
    all_frags_valid,
    fts,
    smiles_match_set,
)


_KNOWN_METALS = ("Pd", "Fe", "Cu", "Ni", "Zn", "Pt", "Ti", "Os", "Mn", "Rh", "Ag")

_ACID_SMARTS = [
    Chem.MolFromSmarts("[CX3](=O)[OX2H1]"),
    Chem.MolFromSmarts("[SX4](=O)(=O)[OX2H1]"),
]
_AMIDE_SMARTS = [
    Chem.MolFromSmarts("[NX3][CX3H1](=O)"),
    Chem.MolFromSmarts("[NX3][CX3](=O)[#6]"),
]

_TRANSFORMATION_KEYWORDS: dict[str, list[str]] = {
    "Reduction": ["reduc", "nitro", "amine", "hydrogen", "deprotect", "carbamate"],
    "Acylation": ["acyl", "ester", "amide", "sulfonate", "mesyl", "tosyl", "carbonate", "protect"],
    "C-C Coupling": ["coupl", "biaryl", "c-c", "cross", "aryl", "stille", "suzuki", "heck"],
    "Heteroatom Alkylation and Arylation": ["alkyl", "arylat", "c-n", "c-o", "ether", "amine", "substitution", "acetal"],
    "Deprotection": ["deprotect", "cleav", "hydrogenolysis", "debenzyl", "carbamate"],
    "Functional Group Interconversion": ["interconversion", "oh", "cl", "chlor", "halide", "activation"],
    "Aromatic Heterocycle Formation": ["cycl", "indole", "heterocycle", "ring", "closure", "oxazole"],
    "Oxidation": ["oxid", "aldehyde", "ketone", "sulfone", "sulfoxide", "epoxid", "diol", "dihydroxylation", "alkene"],
    "Protection": ["protect", "boc", "cbz", "protection"],
}


def _iter_mols(smiles: str):
    if not smiles:
        return
    for frag in smiles.split("."):
        frag = frag.strip()
        if not frag:
            continue
        mol = Chem.MolFromSmiles(frag)
        if mol is not None:
            yield mol


def infer_catalyst_class_from_smiles(smiles: str) -> str:
    """
    Infer a coarse catalyst class from catalyst SMILES for GT-backed or self checks.
    """
    mols = list(_iter_mols(smiles))
    if not mols:
        return "other"

    atom_symbols = [atom.GetSymbol() for mol in mols for atom in mol.GetAtoms()]
    atom_counter = Counter(atom_symbols)
    for metal in _KNOWN_METALS:
        if atom_counter.get(metal, 0) > 0:
            return metal

    for mol in mols:
        for patt in _ACID_SMARTS:
            if patt is not None and mol.HasSubstructMatch(patt):
                return "acid"

    ring_o_atoms = 0
    total_o_atoms = atom_counter.get("O", 0)
    total_n_atoms = atom_counter.get("N", 0)
    for mol in mols:
        ring_info = mol.GetRingInfo()
        ring_atom_ids = {idx for ring in ring_info.AtomRings() for idx in ring}
        ring_o_atoms += sum(
            1 for atom in mol.GetAtoms()
            if atom.GetIdx() in ring_atom_ids and atom.GetSymbol() == "O"
        )
    if total_o_atoms >= 4 and ring_o_atoms >= 4 and total_n_atoms == 0:
        return "crown ether"

    for mol in mols:
        if any(patt is not None and mol.HasSubstructMatch(patt) for patt in _AMIDE_SMARTS):
            if total_n_atoms > 0 and total_o_atoms > 0:
                return "aprotic activator"

    has_aromatic_n = any(
        atom.GetSymbol() == "N" and atom.GetIsAromatic()
        for mol in mols for atom in mol.GetAtoms()
    )
    if has_aromatic_n:
        return "organocatalyst"
    if total_n_atoms > 0:
        return "base"
    return "other"


def check_s1_coarse(record: dict) -> bool:
    """S1 (GATES): step1 RXN_CLASS exactly matches GT coarse_rxn_cls."""
    model = (record.get("step1_rxn_cls", "") or "").strip()
    gt = (record.get("coarse_rxn_cls", "") or "").strip()
    return bool(model) and bool(gt) and model.lower() == gt.lower()


def check_s2_transformation_logic(record: dict) -> bool:
    """S2 (INFO ONLY): CORE_TRANSFORMATION contains keywords compatible with rxn_cls."""
    rxn_cls = record.get("rxn_cls", "")
    text = (record.get("step2_core_transform", "") or "").lower()
    if not rxn_cls or not text:
        return False
    kws = _TRANSFORMATION_KEYWORDS.get(rxn_cls, [])
    return any(kw in text for kw in kws) if kws else len(text) >= 3


def check_s3_role_presence(record: dict) -> bool:
    """S3 (INFO ONLY): CATALYST_ROLE is non-empty."""
    return bool((record.get("step3_catalyst_role", "") or "").strip())


def check_s4_class_grounding(record: dict) -> bool:
    """S4 (GATES): predicted catalyst class matches GT catalyst class inferred from GT SMILES."""
    model_class = record.get("step4_catalyst_class", "")
    gt_smiles = record.get("gt_catalyst_smiles", "")
    gt_class = infer_catalyst_class_from_smiles(gt_smiles)
    return bool(model_class) and model_class == gt_class


def check_s5_class_self_consistency(record: dict) -> bool:
    """S5 (GATES): predicted catalyst class matches class inferred from predicted catalyst SMILES."""
    model_class = record.get("step4_catalyst_class", "")
    pred_smiles = record.get("step5_predicted_smi", "")
    if not model_class or not pred_smiles:
        return False
    pred_class = infer_catalyst_class_from_smiles(pred_smiles)
    return model_class == pred_class


def check_s6_mol_valid(record: dict) -> bool:
    """S6 (GATES): predicted catalyst SMILES is RDKit-parseable fragment-wise."""
    return all_frags_valid(record.get("step5_predicted_smi", ""))


def check_s7_answer_consistent(record: dict) -> bool:
    """S7 (GATES): Answer line matches step5 PREDICTED_CATALYST_SMILES."""
    step_smiles = record.get("step5_predicted_smi", "")
    answer = record.get("answer_smiles", "")
    if not step_smiles or not answer:
        return False
    if all_frags_valid(step_smiles) and all_frags_valid(answer):
        return smiles_match_set(step_smiles, answer)
    return step_smiles.strip() == answer.strip()


def check_outcome(record: dict) -> bool:
    """outcome (GATES): predicted catalyst SMILES matches GT catalyst SMILES under canonical set match."""
    pred = record.get("step5_predicted_smi", "") or record.get("answer_smiles", "")
    gt = record.get("gt_catalyst_smiles", "")
    return bool(pred) and bool(gt) and smiles_match_set(pred, gt)


def verify_one(record: dict) -> dict:
    updated = dict(record)

    if not record.get("api_success", False):
        for key in (
            "S1_rxn_class_exact",
            "S2_transformation_logic",
            "S3_catalyst_role_presence",
            "S4_class_grounding",
            "S5_class_self_consistency",
            "S6_mol_valid",
            "S7_answer_consistent",
            "outcome",
            "all_pass",
        ):
            updated[key] = False
        updated["verify_skip_reason"] = "api_failure"
        return updated

    if not record.get("parse_ok", False):
        for key in (
            "S1_rxn_class_exact",
            "S2_transformation_logic",
            "S3_catalyst_role_presence",
            "S4_class_grounding",
            "S5_class_self_consistency",
            "S6_mol_valid",
            "S7_answer_consistent",
            "outcome",
            "all_pass",
        ):
            updated[key] = False
        updated["verify_skip_reason"] = "parse_failed"
        return updated

    s1 = check_s1_coarse(record)
    s2 = check_s2_transformation_logic(record)
    s3 = check_s3_role_presence(record)
    s4 = check_s4_class_grounding(record)
    s5 = check_s5_class_self_consistency(record)
    s6 = check_s6_mol_valid(record)
    s7 = check_s7_answer_consistent(record)
    outcome = check_outcome(record)

    all_pass = s1 and s4 and s5 and s6 and s7 and outcome
    updated.update({
        "S1_rxn_class_exact": s1,
        "S2_transformation_logic": s2,
        "S3_catalyst_role_presence": s3,
        "S4_class_grounding": s4,
        "S5_class_self_consistency": s5,
        "S6_mol_valid": s6,
        "S7_answer_consistent": s7,
        "outcome": outcome,
        "all_pass": all_pass,
        "pred_class_from_smiles": infer_catalyst_class_from_smiles(
            record.get("step5_predicted_smi", "")
        ),
        "gt_class_from_smiles": infer_catalyst_class_from_smiles(
            record.get("gt_catalyst_smiles", "")
        ),
        "outcome_fts": round(fts(record.get("step5_predicted_smi", ""), record.get("gt_catalyst_smiles", "")), 4),
    })
    return updated


def _build_summary(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {}

    def rate(key: str) -> float:
        return round(sum(bool(r.get(key, False)) for r in records) / n, 4)

    by_diff: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_diff[r.get("difficulty", "unknown")].append(r)

    diff_stats = {}
    for diff, recs in by_diff.items():
        dn = len(recs)
        diff_stats[diff] = {
            "n": dn,
            "all_pass": round(sum(bool(r.get("all_pass", False)) for r in recs) / dn, 4),
            "outcome": round(sum(bool(r.get("outcome", False)) for r in recs) / dn, 4),
            "parse_ok": round(sum(bool(r.get("parse_ok", False)) for r in recs) / dn, 4),
        }

    return {
        "n_total": n,
        "parse_ok": rate("parse_ok"),
        "S1_rxn_class_exact": rate("S1_rxn_class_exact"),
        "S2_transformation_logic": rate("S2_transformation_logic"),
        "S3_catalyst_role_presence": rate("S3_catalyst_role_presence"),
        "S4_class_grounding": rate("S4_class_grounding"),
        "S5_class_self_consistency": rate("S5_class_self_consistency"),
        "S6_mol_valid": rate("S6_mol_valid"),
        "S7_answer_consistent": rate("S7_answer_consistent"),
        "outcome": rate("outcome"),
        "all_pass": rate("all_pass"),
        "avg_fts": round(sum(r.get("outcome_fts", 0.0) for r in records) / n, 4),
        "by_difficulty": diff_stats,
    }


def verify_all(records: list[dict]) -> tuple[list[dict], dict]:
    verified = []
    for idx, rec in enumerate(records):
        updated = verify_one(rec)
        verified.append(updated)
    summary = _build_summary(verified)
    return verified, summary
