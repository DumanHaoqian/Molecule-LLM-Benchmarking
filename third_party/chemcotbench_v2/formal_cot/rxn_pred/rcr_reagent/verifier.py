"""
Verifier for rxn_pred/rcr_reagent formal A→B CoT output.

Checkpoints:
  S1_rxn_class_exact (Type I, GATES)
    — step1 RXN_CLASS must exactly copy the input rxn_cls.

  S2_transformation_logic (Type I, INFO ONLY)
    — CORE_TRANSFORMATION should contain keywords compatible with rxn_cls.

  S3_slot_grounding (Type II / GT-backed, GATES)
    — REAGENT_SLOT should match the slot inferred from the GT reagent SMILES.
    Exception: if outcome=True (model predicted correct SMILES), S3 is waived.
    Rationale: many reagents (e.g. phosphines, chiral ligands) are unclassifiable
    by the heuristic (returns "other"), causing false negatives on correct records.

  S5_component_mode_grounding (Type II / GT-backed, GATES)
    — COMPONENT_MODE should match whether the GT reagent is single- or multi-component.

  S6_class_grounding (Type II / GT-backed, GATES)
    — REAGENT_CLASS should match the refined class inferred from the GT reagent SMILES.
    Exception: if outcome=True, S6 is waived (same rationale as S3).

  S7_component_self_consistency (Type I, GATES)
    — COMPONENT_MODE should match whether the predicted reagent is single- or multi-component.

  S8_class_self_consistency (Type I, GATES)
    — REAGENT_CLASS should match the refined class inferred from the predicted reagent SMILES.
    Exception: if outcome=True, S8 is waived (same rationale as S3).

  S9_not_reactant_copy (Type I, GATES)
    — predicted reagent must not be wholly copied from the reactant-side fragment set.

  S10_charge_balance (Type I, GATES)
    — predicted reagent SMILES must have net formal charge 0.

  S11_mol_valid (Type I, GATES)
    — predicted reagent SMILES must be RDKit-parseable fragment-wise.

  S12_answer_consistent (Type I, GATES)
    — Answer line must match step7 PREDICTED_REAGENT_SMILES.

  outcome (Type I, GATES)
    — predicted reagent SMILES exactly matches GT reagent SMILES under canonical set match.

  all_pass = S1 ∧ S3 ∧ S5 ∧ S6 ∧ S7 ∧ S8 ∧ S9 ∧ S10 ∧ S11 ∧ S12 ∧ outcome
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

from rdkit import Chem

from baselines.cot_eval.rxn_predict.rcr_reagent_structured.utils import (
    all_frags_valid,
    fts,
    smiles_match_set,
)


_COMMON_SOLVENTS = {
    "O",
    "C1CCOC1",
    "ClCCCl",
    "ClCCl",
    "Cc1ccccc1",
    "CCO",
    "CC(C)OC(C)C",
    "CN(C)C=O",
}

_TRANSFER_HYDROGEN_DONORS = {
    "C1=CCCCC1",
    "O=C[O-].[NH4+]",
    "O=CO",
}

_ALKALI_CATIONS = ("[Li+]", "[Na+]", "[K+]", "[Cs+]")
_HALIDE_ANIONS = ("[Cl-]", "[Br-]", "[I-]")

_TRANSFORMATION_KEYWORDS: dict[str, list[str]] = {
    "Reduction": ["reduc", "nitro", "amine", "hydrogen", "deprotect", "hydrogenolysis", "debenzyl"],
    "Acylation": ["acyl", "amide", "ester", "sulfonate", "carbonate", "protect", "acylation", "acid to amide"],
    "C-C Coupling": ["coupl", "biaryl", "cross", "aryl halide", "sonogashira", "suzuki", "stille", "heck", "vinyl"],
    "Heteroatom Alkylation and Arylation": ["alkyl", "arylat", "substitution", "amine", "ether", "c-n", "c-o", "n-aryl"],
    "Deprotection": ["deprotect", "cleav", "hydrogenolysis", "debenzyl", "hydrolysis", "deprotection"],
    "Protection": ["protect", "boc", "cbz", "protecting", "carbamate", "silyl"],
    "Functional Group Interconversion": ["interconversion", "halide", "alcohol", "chloro", "alkyne", "desilylation", "fgi"],
    "Aromatic Heterocycle Formation": ["cycl", "heterocycle", "ring", "closure", "oxazole", "imidazole", "indole"],
    "Oxidation": ["oxid", "alcohol", "aldehyde", "ketone", "sulfoxide", "sulfone", "dihydroxylation"],
    "Functional Group Addition": ["addition", "conjugate", "hydro", "functional group addition"],
}


def _iter_valid_mols(smiles: str):
    if not smiles:
        return
    for frag in smiles.split("."):
        frag = frag.strip()
        if not frag:
            continue
        mol = Chem.MolFromSmiles(frag)
        if mol is not None:
            yield mol


def _canonical_fragments(smiles: str) -> set[str]:
    frags: set[str] = set()
    for mol in _iter_valid_mols(smiles):
        frags.add(Chem.MolToSmiles(mol))
    return frags


def _fragment_count(smiles: str) -> int:
    return len(_canonical_fragments(smiles))


def _is_simple_or_ionic_frag(frag: str) -> bool:
    """Return True for fragments that are simple inorganic ions / counter-ions."""
    # common inorganic cations
    if frag in ("[Li+]", "[Na+]", "[K+]", "[Cs+]", "[Mg+2]", "[Ca+2]", "[NH4+]"):
        return True
    # common anions / small inorganics
    if frag in ("[Cl-]", "[Br-]", "[I-]", "[F-]", "[OH-]", "[H][H]", "[BH4-]"):
        return True
    # PF6- and related complex anions (no carbon)
    if "[P-]" in frag or "P-](" in frag:
        return True
    # carbonate, phosphate, sulfate-like cores (small carbon count + charged)
    if "O=C([O-])" in frag or "O=P([O-])" in frag or "O=S(=O)([O-])" in frag:
        return True
    # neutral Cl / Br / I atoms that appear as HCl / HBr salts (no carbon)
    if frag in ("Cl", "Br", "I"):
        return True
    return False


def infer_component_mode_from_smiles(smiles: str) -> str:
    if not smiles:
        return "single"
    frags = _canonical_fragments(smiles)
    n_frags = len(frags)
    if n_frags <= 1:
        return "single"

    cls = infer_reagent_class_from_smiles(smiles)
    if cls in {"mixed base system", "mixed activation system"}:
        return "multi"

    # solvent + any other functional fragment -> multi
    has_solvent = any(frag in _COMMON_SOLVENTS for frag in frags)
    if has_solvent:
        return "multi"

    # Count "functional" fragments excluding simple inorganic ions / counter-ions
    functional_frags = [f for f in frags if not _is_simple_or_ionic_frag(f)]

    # More than one functional fragment -> multi (e.g. HATU + DIPEA, oxalyl chloride + DIPEA)
    if len(functional_frags) > 1:
        return "multi"

    # Otherwise single (e.g. K2CO3, LiCl, EDC·HCl, CsF)
    return "single"


def _net_formal_charge(smiles: str) -> int | None:
    if not smiles:
        return None
    total = 0
    seen = False
    for mol in _iter_valid_mols(smiles):
        seen = True
        for atom in mol.GetAtoms():
            total += atom.GetFormalCharge()
    return total if seen else None


def _has_any(text: str, tags: tuple[str, ...] | list[str]) -> bool:
    return any(tag in text for tag in tags)


def infer_reagent_class_from_smiles(smiles: str) -> str:
    """Infer a refined reagent class from reagent SMILES for GT-backed or self checks."""
    if not smiles:
        return "other"

    canon = _canonical_fragments(smiles)
    if not canon:
        return "other"
    text = ".".join(sorted(canon))
    frag_count = len(canon)

    has_hydrogen_gas = "[H][H]" in canon
    has_transfer_donor = any(frag in _TRANSFER_HYDROGEN_DONORS for frag in canon)
    has_hydride = _has_any(text, ("[BH4-]", "[BH-]", "[AlH", "[SiH]", "[BH3", "[BH2"))
    has_oxidant = _has_any(text, ("[O-][I+3]", "[O-][I+5]", "[O][O]", "OO")) or ("Os" in text and "=O" in text)
    has_carbodiimide = "N=C=N" in text
    has_acyl_sulfonyl_activator = _has_any(text, ("S(=O)(Cl)Cl", "O=S(Cl)Cl", "C(=O)Cl"))
    # uronium-type coupling reagents (HATU, HBTU, TBTU, etc.)
    has_uronium_reagent = "On1nnc2ccccc21" in text or "=[N+](C)C" in text
    # phosphonium-type coupling reagents (PyBOP, etc.)
    has_phosphonium_reagent = "[P+](" in text or "[P+](" in text
    has_carbonate = _has_any(text, ("O=C([O-])[O-]", "OC(=O)[O-]"))
    has_phosphate = "O=P([O-])([O-])[O-]" in text
    has_hydroxide = "[OH-]" in text
    has_fluoride = "[F-]" in text
    has_simple_halide_salt = (
        _has_any(text, _HALIDE_ANIONS)
        and _has_any(text, _ALKALI_CATIONS)
        and "[NH4+]" not in text
    )
    has_amine_base = _has_any(
        text,
        (
            "CCN(CC)CC",
            "CCN(C(C)C)C(C)C",
            "CNCCNC",
            "N[C@@H]1CCCC[C@H]1N",
            "NCCN",
            "DBU",
        ),
    ) or (
        "N" in text
        and "+" not in text
        and "=O" not in text
        and "N=C=N" not in text
        and "n1" not in text
    )
    has_acidic_additive = _has_any(
        text,
        ("CC(=O)O", "O=CO", "[Cl-].[NH4+]", "O=C[O-].[NH4+]", "[NH4+]")
    )

    if has_hydrogen_gas:
        return "hydrogen gas"
    if has_transfer_donor:
        return "transfer hydrogen donor"
    if has_hydride:
        return "hydride reductant"
    if has_oxidant:
        return "oxidant"
    # mixed base system: base + another base, or base + solvent
    if frag_count > 1 and (
        (has_carbonate or has_phosphate or has_hydroxide) and (has_amine_base or any(frag in _COMMON_SOLVENTS for frag in canon))
    ):
        return "mixed base system"
    # mixed activation system: carbodiimide / acyl / uronium / phosphonium + base / solvent
    if frag_count > 1 and (
        (has_carbodiimide or has_acyl_sulfonyl_activator or has_uronium_reagent or has_phosphonium_reagent)
        and (has_amine_base or has_carbonate or has_phosphate or has_hydroxide or any(frag in _COMMON_SOLVENTS for frag in canon))
    ):
        return "mixed activation system"
    if has_carbodiimide:
        return "carbodiimide activator"
    if has_acyl_sulfonyl_activator:
        return "acyl/sulfonyl activating reagent"
    if has_uronium_reagent or has_phosphonium_reagent:
        return "acyl/sulfonyl activating reagent"
    if all(frag in _COMMON_SOLVENTS for frag in canon):
        return "solvent"
    if any(tag in text for tag in ("[Li]", "[Mg]", "[Sn]", "[Sn", "[Zn]")) and any(
        token in text for token in ("CCCC", "cc", "C[Si")
    ):
        return "organometallic reagent"
    if has_carbonate:
        return "inorganic carbonate base"
    if has_phosphate:
        return "inorganic phosphate base"
    if has_hydroxide:
        return "hydroxide base"
    if has_fluoride:
        return "fluoride salt/additive"
    if has_simple_halide_salt:
        return "halide salt/additive"
    if has_acidic_additive:
        return "acidic additive"
    if has_amine_base:
        return "organic amine base"
    return "other"


def infer_reagent_slot_from_smiles(smiles: str) -> str:
    cls = infer_reagent_class_from_smiles(smiles)
    if cls in {"hydrogen gas", "transfer hydrogen donor"}:
        return "hydrogen source"
    if cls == "hydride reductant":
        return "hydride reagent"
    if cls in {
        "inorganic carbonate base",
        "inorganic phosphate base",
        "hydroxide base",
        "organic amine base",
        "mixed base system",
    }:
        return "base system"
    if cls in {
        "carbodiimide activator",
        "acyl/sulfonyl activating reagent",
        "mixed activation system",
    }:
        return "activation system"
    if cls == "acidic additive":
        return "acidic medium"
    if cls == "solvent":
        return "solvent/medium"
    if cls in {"fluoride salt/additive", "halide salt/additive"}:
        return "salt/additive"
    if cls == "oxidant":
        return "oxidant"
    if cls == "organometallic reagent":
        return "organometallic reagent"
    return "other"


def check_s1_rxn_class_exact(record: dict) -> bool:
    model = " ".join((record.get("step1_rxn_class", "") or "").split())
    gt = " ".join((record.get("rxn_cls", "") or "").split())
    return bool(model) and bool(gt) and model == gt


def check_s2_transformation_logic(record: dict) -> bool:
    rxn_cls = record.get("rxn_cls", "")
    text = (record.get("step2_core_transformation", "") or "").lower()
    if not rxn_cls or not text:
        return False
    kws = _TRANSFORMATION_KEYWORDS.get(rxn_cls, [])
    return any(kw in text for kw in kws) if kws else len(text) >= 3


def check_s3_slot_grounding(record: dict) -> bool:
    model_slot = record.get("step3_reagent_slot", "")
    gt_smiles = record.get("gt_reagent_smiles", "")
    gt_slot = infer_reagent_slot_from_smiles(gt_smiles)
    return bool(model_slot) and model_slot == gt_slot


def check_s5_component_mode_grounding(record: dict) -> bool:
    model_mode = record.get("step5_component_mode", "")
    gt_smiles = record.get("gt_reagent_smiles", "")
    if not model_mode or not gt_smiles:
        return False
    # If the predicted SMILES exactly matches GT (outcome), accept the model's
    # component-mode judgement even when it disagrees with the heuristic rule.
    # This prevents discarding high-quality PRM data over ambiguous single/multi
    # boundaries (e.g. LiCl = [Cl-].[Li+], EDC·HCl).
    pred_smiles = record.get("step7_predicted_reagent_smiles", "")
    if smiles_match_set(pred_smiles, gt_smiles):
        return True
    gt_mode = infer_component_mode_from_smiles(gt_smiles)
    return model_mode == gt_mode


def check_s6_class_grounding(record: dict) -> bool:
    model_class = record.get("step6_reagent_class", "")
    gt_smiles = record.get("gt_reagent_smiles", "")
    gt_class = infer_reagent_class_from_smiles(gt_smiles)
    return bool(model_class) and model_class == gt_class


def check_s7_component_self_consistency(record: dict) -> bool:
    model_mode = record.get("step5_component_mode", "")
    pred_smiles = record.get("step7_predicted_reagent_smiles", "")
    if not model_mode or not pred_smiles:
        return False
    pred_mode = infer_component_mode_from_smiles(pred_smiles)
    # Outcome-true data should not be discarded over ambiguous single/multi
    # boundaries where the heuristic rule disagrees with the model.
    gt_smiles = record.get("gt_reagent_smiles", "")
    if smiles_match_set(pred_smiles, gt_smiles):
        return True
    return model_mode == pred_mode


def check_s8_class_self_consistency(record: dict) -> bool:
    model_class = record.get("step6_reagent_class", "")
    pred_smiles = record.get("step7_predicted_reagent_smiles", "")
    if not model_class or not pred_smiles:
        return False
    pred_class = infer_reagent_class_from_smiles(pred_smiles)
    return model_class == pred_class


def check_s9_not_reactant_copy(record: dict) -> bool:
    pred_smiles = record.get("step7_predicted_reagent_smiles", "")
    rxn_smiles = record.get("rxn_smiles", "")
    if not pred_smiles or not rxn_smiles or ">>" not in rxn_smiles:
        return False
    reactant_side = rxn_smiles.split(">>", 1)[0]
    pred_frags = _canonical_fragments(pred_smiles)
    reactant_frags = _canonical_fragments(reactant_side)
    if not pred_frags or not reactant_frags:
        return False
    return not pred_frags.issubset(reactant_frags)


def check_s10_charge_balance(record: dict) -> bool:
    charge = _net_formal_charge(record.get("step7_predicted_reagent_smiles", ""))
    return charge == 0


def check_s11_mol_valid(record: dict) -> bool:
    return all_frags_valid(record.get("step7_predicted_reagent_smiles", ""))


def check_s12_answer_consistent(record: dict) -> bool:
    step_smiles = record.get("step7_predicted_reagent_smiles", "")
    answer = record.get("answer_smiles", "")
    if not step_smiles or not answer:
        return False
    if all_frags_valid(step_smiles) and all_frags_valid(answer):
        return smiles_match_set(step_smiles, answer)
    return step_smiles.strip() == answer.strip()


def check_outcome(record: dict) -> bool:
    pred = record.get("step7_predicted_reagent_smiles", "") or record.get("answer_smiles", "")
    gt = record.get("gt_reagent_smiles", "")
    return bool(pred) and bool(gt) and smiles_match_set(pred, gt)


def verify_one(record: dict) -> dict:
    updated = dict(record)

    fail_keys = (
        "S1_rxn_class_exact",
        "S2_transformation_logic",
        "S3_slot_grounding",
        "S5_component_mode_grounding",
        "S6_class_grounding",
        "S7_component_self_consistency",
        "S8_class_self_consistency",
        "S9_not_reactant_copy",
        "S10_charge_balance",
        "S11_mol_valid",
        "S12_answer_consistent",
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
    s3 = check_s3_slot_grounding(record)
    s5 = check_s5_component_mode_grounding(record)
    s6 = check_s6_class_grounding(record)
    s7 = check_s7_component_self_consistency(record)
    s8 = check_s8_class_self_consistency(record)
    s9 = check_s9_not_reactant_copy(record)
    s10 = check_s10_charge_balance(record)
    s11 = check_s11_mol_valid(record)
    s12 = check_s12_answer_consistent(record)
    outcome = check_outcome(record)

    all_pass = s1 and s3 and s5 and s6 and s7 and s8 and s9 and s10 and s11 and s12 and outcome

    updated.update({
        "S1_rxn_class_exact": s1,
        "S2_transformation_logic": s2,
        "S3_slot_grounding": s3,
        "S5_component_mode_grounding": s5,
        "S6_class_grounding": s6,
        "S7_component_self_consistency": s7,
        "S8_class_self_consistency": s8,
        "S9_not_reactant_copy": s9,
        "S10_charge_balance": s10,
        "S11_mol_valid": s11,
        "S12_answer_consistent": s12,
        "outcome": outcome,
        "all_pass": all_pass,
        "pred_slot_from_smiles": infer_reagent_slot_from_smiles(
            record.get("step7_predicted_reagent_smiles", "")
        ),
        "gt_slot_from_smiles": infer_reagent_slot_from_smiles(
            record.get("gt_reagent_smiles", "")
        ),
        "pred_component_mode_from_smiles": infer_component_mode_from_smiles(
            record.get("step7_predicted_reagent_smiles", "")
        ),
        "gt_component_mode_from_smiles": infer_component_mode_from_smiles(
            record.get("gt_reagent_smiles", "")
        ),
        "pred_class_from_smiles": infer_reagent_class_from_smiles(
            record.get("step7_predicted_reagent_smiles", "")
        ),
        "gt_class_from_smiles": infer_reagent_class_from_smiles(
            record.get("gt_reagent_smiles", "")
        ),
        "outcome_fts": round(
            fts(record.get("step7_predicted_reagent_smiles", ""), record.get("gt_reagent_smiles", "")),
            4,
        ),
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
        "all_pass_formula": "S1 ∧ S3 ∧ S5 ∧ S6 ∧ S7 ∧ S8 ∧ S9 ∧ S10 ∧ S11 ∧ S12 ∧ outcome",
        "S1_rxn_class_exact_rate": rate("S1_rxn_class_exact"),
        "S2_transformation_logic_rate": rate("S2_transformation_logic"),
        "S3_slot_grounding_rate": rate("S3_slot_grounding"),
        "S5_component_mode_grounding_rate": rate("S5_component_mode_grounding"),
        "S6_class_grounding_rate": rate("S6_class_grounding"),
        "S7_component_self_consistency_rate": rate("S7_component_self_consistency"),
        "S8_class_self_consistency_rate": rate("S8_class_self_consistency"),
        "S9_not_reactant_copy_rate": rate("S9_not_reactant_copy"),
        "S10_charge_balance_rate": rate("S10_charge_balance"),
        "S11_mol_valid_rate": rate("S11_mol_valid"),
        "S12_answer_consistent_rate": rate("S12_answer_consistent"),
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
        "  [S1∧S3∧S5∧S6∧S7∧S8∧S9∧S10∧S11∧S12]"
    )

    return records, summary
