"""
Verifier for rxn_pred/rcr_solvent formal A→B CoT output.

Checkpoints
─────────────
  S1_rxn_class_exact (Type I, GATES)
    — step1 RXN_CLASS must exactly copy the input rxn_cls.

  S2_transformation_logic (Type I, INFO ONLY)
    — CORE_TRANSFORMATION should contain keywords compatible with rxn_cls.
      Useful diagnostic signal but too brittle to gate.

  S4_proticity_consistency (Type I / RDKit, GATES)
    — If PROTICITY="protic", predicted solvent SMILES must contain at least
      one O-H or N-H group (SMARTS [O;!H0,N;!H0]).
    — If PROTICITY="aprotic", predicted solvent SMILES must contain NO O-H/N-H.

  S5_polarity_consistency (Type I / RDKit, GATES)
    — Polar heteroatom (N/O/S/F/Cl/Br/I) fraction of heavy atoms in predicted
      solvent. If ≥0.20 → polar; <0.20 → nonpolar.
    — Must match the declared POLARITY.

  S6_mol_valid (Type I / RDKit, GATES)
    — Predicted solvent SMILES must be RDKit-parseable fragment-wise.

  S7_answer_consistent (Type I, GATES)
    — Answer line must match step5 PREDICTED_SOLVENT_SMILES.

  outcome (Type I / RDKit, GATES)
    — Predicted solvent SMILES exactly matches GT solvent SMILES under
      canonical set match.

  all_pass = S1 ∧ S4 ∧ S5 ∧ S6 ∧ S7 ∧ outcome
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.cot_eval.rxn_predict.rcr_solvent_structured.utils import (
    all_frags_valid,
    fts,
    smiles_match_set,
)

# SMARTS for protic groups: O-H or N-H
_PROTIC_SMARTS = "[O;!H0,N;!H0]"
_PROTIC_PATTERN = Chem.MolFromSmarts(_PROTIC_SMARTS)

# Polar heteroatoms for polarity check
_POLAR_ATOMS = {"N", "O", "S", "F", "Cl", "Br", "I"}

_TRANSFORMATION_KEYWORDS: dict[str, list[str]] = {
    "Reduction": ["reduc", "nitro", "amine", "hydrogen", "deprotect", "carbamate",
                   "carbonyl", "aldehyde", "ketone", "alkene", "alkyne"],
    "Acylation": ["acyl", "ester", "amide", "sulfonate", "mesyl", "tosyl",
                  "carbonate", "protect", "sulfonylation", "sulfonyl"],
    "C-C Coupling": ["coupl", "biaryl", "c-c", "cross", "aryl", "stille",
                     "suzuki", "heck", "sonogashira", "negishi"],
    "Heteroatom Alkylation and Arylation": ["alkyl", "arylat", "c-n", "c-o",
                                            "ether", "amine", "substitution",
                                            "acetal", "sn2", "snar"],
    "Deprotection": ["deprotect", "cleav", "hydrogenolysis", "debenzyl",
                     "carbamate", "boc", "cbz", "silyl"],
    "Functional Group Interconversion": ["interconversion", "oh", "cl", "chlor",
                                         "halide", "activation", "fluorin",
                                         "bromin", "iodin"],
    "Aromatic Heterocycle Formation": ["cycl", "indole", "heterocycle", "ring",
                                       "closure", "oxazole", "imidazole",
                                       "pyridine", "pyrimidine"],
    "Oxidation": ["oxid", "aldehyde", "ketone", "sulfone", "sulfoxide", "epoxid",
                  "diol", "dihydroxylation", "alkene", "alcohol"],
    "Protection": ["protect", "boc", "cbz", "protection", "silyl", "acetal",
                   "benzyl"],
    "Functional Group Addition": ["addition", "halogen", "nitro", "sulfon",
                                  "bromin", "chlorin"],
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


# ---------------------------------------------------------------------------
# S1: RXN_CLASS exact match
# ---------------------------------------------------------------------------

def check_s1_rxn_class_exact(record: dict) -> bool:
    model = " ".join((record.get("step1_rxn_class", "") or "").split())
    gt = " ".join((record.get("rxn_cls", "") or "").split())
    return bool(model) and bool(gt) and model == gt


# ---------------------------------------------------------------------------
# S2: transformation logic (INFO only)
# ---------------------------------------------------------------------------

def check_s2_transformation_logic(record: dict) -> bool:
    rxn_cls = record.get("rxn_cls", "")
    text = (record.get("step2_core_transformation", "") or "").lower()
    if not rxn_cls or not text:
        return False
    kws = _TRANSFORMATION_KEYWORDS.get(rxn_cls, [])
    return any(kw in text for kw in kws) if kws else len(text) >= 3


# ---------------------------------------------------------------------------
# S4: proticity consistency (RDKit SMARTS)
# ---------------------------------------------------------------------------

def check_s4_proticity_consistency(record: dict) -> bool:
    proticity = record.get("step3_proticity", "")
    solvent_smiles = record.get("step5_predicted_solvent_smiles", "")
    if not proticity or not solvent_smiles:
        return False

    has_protic = False
    for frag in solvent_smiles.split("."):
        frag = frag.strip()
        if not frag:
            continue
        mol = Chem.MolFromSmiles(frag)
        if mol is None:
            return False
        if mol.HasSubstructMatch(_PROTIC_PATTERN):
            has_protic = True
            break

    if proticity == "protic":
        return has_protic
    else:  # aprotic
        return not has_protic


# ---------------------------------------------------------------------------
# S5: polarity consistency (polar atom ratio)
# ---------------------------------------------------------------------------

def _polar_atom_ratio(smiles: str) -> float | None:
    """Fraction of heavy atoms that are polar heteroatoms (N/O/S/F/Cl/Br/I)."""
    mols = list(_iter_mols(smiles))
    if not mols:
        return None

    total_heavy = 0
    total_polar = 0
    for mol in mols:
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                continue
            total_heavy += 1
            if atom.GetSymbol() in _POLAR_ATOMS:
                total_polar += 1

    if total_heavy == 0:
        return None
    return total_polar / total_heavy


def check_s5_polarity_consistency(record: dict) -> bool:
    polarity = record.get("step4_polarity", "")
    solvent_smiles = record.get("step5_predicted_solvent_smiles", "")
    if not polarity or not solvent_smiles:
        return False

    ratio = _polar_atom_ratio(solvent_smiles)
    if ratio is None:
        return False

    is_polar = ratio >= 0.20
    if polarity == "polar":
        return is_polar
    else:  # nonpolar
        return not is_polar


# ---------------------------------------------------------------------------
# S6: mol_valid
# ---------------------------------------------------------------------------

def check_s6_mol_valid(record: dict) -> bool:
    return all_frags_valid(record.get("step5_predicted_solvent_smiles", ""))


# ---------------------------------------------------------------------------
# S7: answer_consistent
# ---------------------------------------------------------------------------

def check_s7_answer_consistent(record: dict) -> bool:
    step_smiles = record.get("step5_predicted_solvent_smiles", "")
    answer = record.get("answer_smiles", "")
    if not step_smiles or not answer:
        return False
    if all_frags_valid(step_smiles) and all_frags_valid(answer):
        return smiles_match_set(step_smiles, answer)
    return step_smiles.strip() == answer.strip()


# ---------------------------------------------------------------------------
# outcome: predicted == GT
# ---------------------------------------------------------------------------

def check_outcome(record: dict) -> bool:
    pred = record.get("step5_predicted_solvent_smiles", "") or record.get("answer_smiles", "")
    gt = record.get("gt_solvent_smiles", "")
    return bool(pred) and bool(gt) and smiles_match_set(pred, gt)


# ---------------------------------------------------------------------------
# verify_one / verify_all
# ---------------------------------------------------------------------------

def verify_one(record: dict) -> dict:
    updated = dict(record)

    fail_keys = (
        "S1_rxn_class_exact",
        "S2_transformation_logic",
        "S4_proticity_consistency",
        "S5_polarity_consistency",
        "S6_mol_valid",
        "S7_answer_consistent",
        "outcome",
        "all_pass",
    )

    if not record.get("api_success", False):
        for key in fail_keys:
            updated[key] = False
        updated["verify_skip_reason"] = "api_failure"
        return updated

    if not record.get("parse_ok", False):
        for key in fail_keys:
            updated[key] = False
        updated["verify_skip_reason"] = "parse_failed"
        return updated

    s1 = check_s1_rxn_class_exact(record)
    s2 = check_s2_transformation_logic(record)
    s4 = check_s4_proticity_consistency(record)
    s5 = check_s5_polarity_consistency(record)
    s6 = check_s6_mol_valid(record)
    s7 = check_s7_answer_consistent(record)
    outcome = check_outcome(record)

    all_pass = s1 and s4 and s5 and s6 and s7 and outcome

    # Compute proticity/polarity of GT solvent for diagnostics
    gt_smiles = record.get("gt_solvent_smiles", "")
    gt_protic = None
    gt_polar = None
    if gt_smiles and all_frags_valid(gt_smiles):
        has_protic = any(
            mol.HasSubstructMatch(_PROTIC_PATTERN)
            for mol in _iter_mols(gt_smiles)
        )
        gt_protic = "protic" if has_protic else "aprotic"
        ratio = _polar_atom_ratio(gt_smiles)
        if ratio is not None:
            gt_polar = "polar" if ratio >= 0.20 else "nonpolar"

    updated.update({
        "S1_rxn_class_exact": s1,
        "S2_transformation_logic": s2,
        "S4_proticity_consistency": s4,
        "S5_polarity_consistency": s5,
        "S6_mol_valid": s6,
        "S7_answer_consistent": s7,
        "outcome": outcome,
        "all_pass": all_pass,
        "gt_proticity": gt_protic or "",
        "gt_polarity": gt_polar or "",
        "pred_polar_ratio": round(_polar_atom_ratio(
            record.get("step5_predicted_solvent_smiles", "")
        ) or 0.0, 4),
        "outcome_fts": round(fts(
            record.get("step5_predicted_solvent_smiles", ""),
            record.get("gt_solvent_smiles", ""),
        ), 4),
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
        by_diff.setdefault(r.get("difficulty", "unknown"), []).append(r)

    diff_stats = {}
    for diff, recs in by_diff.items():
        dn = len(recs)
        diff_stats[diff] = {
            "n": dn,
            "all_pass_rate": round(sum(bool(r.get("all_pass", False)) for r in recs) / dn, 4) if dn > 0 else 0.0,
            "outcome_acc": round(sum(bool(r.get("outcome", False)) for r in recs) / dn, 4) if dn > 0 else 0.0,
            "parse_ok_rate": round(sum(bool(r.get("parse_ok", False)) for r in recs) / dn, 4) if dn > 0 else 0.0,
        }

    return {
        "n_total": n,
        "n_parsed_ok": sum(bool(r.get("parse_ok", False)) for r in records),
        "n_all_pass": sum(bool(r.get("all_pass", False)) for r in records),
        "all_pass_rate": rate("all_pass"),
        "outcome_acc": rate("outcome"),
        "all_pass_formula": "S1 ∧ S4 ∧ S5 ∧ S6 ∧ S7 ∧ outcome",
        "S1_rxn_class_exact_rate": rate("S1_rxn_class_exact"),
        "S2_transformation_logic_rate": rate("S2_transformation_logic"),
        "S4_proticity_consistency_rate": rate("S4_proticity_consistency"),
        "S5_polarity_consistency_rate": rate("S5_polarity_consistency"),
        "S6_mol_valid_rate": rate("S6_mol_valid"),
        "S7_answer_consistent_rate": rate("S7_answer_consistent"),
        "outcome_rate": rate("outcome"),
        "avg_fts": round(sum(r.get("outcome_fts", 0.0) for r in records) / n, 4),
        "by_difficulty": diff_stats,
        "token_totals": {
            "reasoning": sum(r.get("reasoning_tokens", 0) for r in records),
            "total_completion": sum(r.get("total_completion_tokens", 0) for r in records),
            "prompt": sum(r.get("prompt_tokens", 0) for r in records),
        },
    }


def verify_all(records: list[dict]) -> tuple[list[dict], dict]:
    print("\n--- Rule verification ---")
    for i, rec in enumerate(records):
        records[i] = verify_one(rec)

    summary = _build_summary(records)

    print("\nVerification summary:")
    for key, value in summary.items():
        if key.endswith("_rate"):
            label = key.replace("_rate", "")
            count = sum(bool(r.get(label)) for r in records)
            tag = " (INFO)" if label == "outcome" else " (GATE)"
            print(f"  {label:32s}: {count}/{summary['n_total']} ({100 * value:.1f}%){tag}")
    all_pass_count = summary.get("n_all_pass", 0)
    n_total = summary.get("n_total", 0)
    print(
        f"  {'all_pass':32s}: {all_pass_count}/{n_total} ({100 * summary['all_pass_rate']:.1f}%)"
        "  [S1∧S4∧S5∧S6∧S7]"
    )

    return records, summary
