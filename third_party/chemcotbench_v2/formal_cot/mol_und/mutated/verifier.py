"""
Verifier for mutated formal A→B CoT.

Five verification checkpoints — all Type I (RDKit-computable, no GT label needed):

  Checkpoint           | Structured eval   | Type   | Verification logic
  ─────────────────────┼───────────────────┼────────┼─────────────────────────────────────────────
  S1_formula_a         | V1 formula_a_cor. | Type I | CalcMolFormula(mol_A) == step1_formula_a
  S2_formula_b         | V2 formula_b_cor. | Type I | CalcMolFormula(mol_B) == step2_formula_b
  S3_formula_match     | V3 formula_diff   | Type I | expected_match(mol_A,mol_B) == step3_formula_match
  S4_canonical_equal   | [extra]           | Type I | (canonical(A)==canonical(B)? "yes":"no") == step4_canonical_equal
  S5_predict_logic     | V4 key_diff_coh.  | Type I | expected_predict(step3,step4) == step5_predict

  all_pass = S1 AND S2 AND S3 AND S4 AND S5
  outcome  = answer == "Different"   (GT is always "Different" in this dataset)

Notes:
  - S3 uses RDKit-computed formulas (not Gemini's), so it checks whether Gemini's
    FORMULA_MATCH declaration is objectively correct.
  - S4 uses canonical SMILES comparison; since GT is always Different, the correct
    CANONICAL_EQUAL is always "no".
  - S5 is pure logical consistency: given Gemini's own step3+step4 declarations,
    does step5_predict follow correctly?
  - The key diagnostic value: S1=S2=0 (or near 0) with high outcome accuracy
    would confirm the "SMILES shortcut" phenomenon found in the structured eval.
"""
import statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from baselines.cot_eval.mol_und.mutated_structured.utils import (
    canonical_smiles,
    get_mol_formula,
)

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "formal_cot" / "mol_und" / "mutated"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Single-record verification
# --------------------------------------------------------------------------- #

def verify_record(rec: dict) -> dict:
    smiles_a = rec.get("smiles_a", "")
    smiles_b = rec.get("smiles_b", "")
    parse_ok = rec.get("parse_ok", False)

    if not rec.get("api_success", False) or not parse_ok:
        return {
            "S1_formula_a": 0, "S2_formula_b": 0,
            "S3_formula_match": 0, "S4_canonical_equal": 0,
            "S5_predict_logic": 0,
            "all_pass": False,
            "outcome": False,
            "rdkit_formula_a": None, "rdkit_formula_b": None,
            "rdkit_canonical_a": None, "rdkit_canonical_b": None,
            "expected_formula_match": None, "expected_canonical_equal": None,
            "verify_note": "api_failed_or_parse_failed",
        }

    step1_formula_a       = rec.get("step1_formula_a")
    step2_formula_b       = rec.get("step2_formula_b")
    step3_formula_match   = rec.get("step3_formula_match")   # "same" or "different"
    step4_canonical_equal = rec.get("step4_canonical_equal") # "yes" or "no"
    step5_predict         = rec.get("step5_predict")         # "Same" or "Different"
    answer                = rec.get("answer")                # "Same" or "Different"

    # ── Compute RDKit ground truths ────────────────────────────────────────
    rdkit_formula_a   = get_mol_formula(smiles_a)
    rdkit_formula_b   = get_mol_formula(smiles_b)
    rdkit_canonical_a = canonical_smiles(smiles_a)
    rdkit_canonical_b = canonical_smiles(smiles_b)

    # Expected formula match (based on RDKit-computed formulas)
    if rdkit_formula_a is not None and rdkit_formula_b is not None:
        expected_formula_match = "same" if rdkit_formula_a == rdkit_formula_b else "different"
    else:
        expected_formula_match = None

    # Expected canonical equality (always "no" since GT is Different)
    if rdkit_canonical_a is not None and rdkit_canonical_b is not None:
        expected_canonical_equal = "yes" if rdkit_canonical_a == rdkit_canonical_b else "no"
    else:
        expected_canonical_equal = None

    # ── S1: formula_a correct (Type I) ────────────────────────────────────
    S1_formula_a = int(
        rdkit_formula_a is not None
        and step1_formula_a is not None
        and rdkit_formula_a == step1_formula_a
    )

    # ── S2: formula_b correct (Type I) ────────────────────────────────────
    S2_formula_b = int(
        rdkit_formula_b is not None
        and step2_formula_b is not None
        and rdkit_formula_b == step2_formula_b
    )

    # ── S3: formula_match declaration correct (Type I) ────────────────────
    # Check against RDKit-computed expected (not Gemini's possibly-wrong formulas)
    S3_formula_match = int(
        expected_formula_match is not None
        and step3_formula_match is not None
        and step3_formula_match == expected_formula_match
    )

    # ── S4: canonical_equal declaration correct (Type I) ──────────────────
    S4_canonical_equal = int(
        expected_canonical_equal is not None
        and step4_canonical_equal is not None
        and step4_canonical_equal == expected_canonical_equal
    )

    # ── S5: predict logic consistent with step3 + step4 declarations ──────
    # This checks internal logical consistency of Gemini's OWN declarations.
    expected_predict = None
    if step3_formula_match is not None and step4_canonical_equal is not None:
        if step3_formula_match == "different":
            expected_predict = "Different"
        elif step3_formula_match == "same" and step4_canonical_equal == "no":
            expected_predict = "Different"
        elif step3_formula_match == "same" and step4_canonical_equal == "yes":
            expected_predict = "Same"

    S5_predict_logic = int(
        expected_predict is not None
        and step5_predict is not None
        and step5_predict == expected_predict
    )

    all_pass = bool(
        S1_formula_a and S2_formula_b and S3_formula_match
        and S4_canonical_equal and S5_predict_logic
    )

    # GT is always "Different" in this dataset
    outcome = (answer == "Different") if answer else False

    return {
        "S1_formula_a":           S1_formula_a,
        "S2_formula_b":           S2_formula_b,
        "S3_formula_match":       S3_formula_match,
        "S4_canonical_equal":     S4_canonical_equal,
        "S5_predict_logic":       S5_predict_logic,
        "all_pass":               all_pass,
        "outcome":                outcome,
        "rdkit_formula_a":        rdkit_formula_a,
        "rdkit_formula_b":        rdkit_formula_b,
        "rdkit_canonical_a":      rdkit_canonical_a,
        "rdkit_canonical_b":      rdkit_canonical_b,
        "expected_formula_match":   expected_formula_match,
        "expected_canonical_equal": expected_canonical_equal,
        "expected_predict":         expected_predict,
        "verify_note": "",
    }


# --------------------------------------------------------------------------- #
# Batch verification + summary
# --------------------------------------------------------------------------- #

def verify_all(records: list[dict]) -> tuple[list[dict], dict]:
    for rec in records:
        rec.update(verify_record(rec))

    n = len(records)
    if n == 0:
        return records, {}

    def rate(key):
        return round(sum(1 for r in records if r.get(key)) / n, 4)

    checkpoint_rates = {
        "S1_formula_a_rate":       round(sum(r["S1_formula_a"]       for r in records) / n, 4),
        "S2_formula_b_rate":       round(sum(r["S2_formula_b"]       for r in records) / n, 4),
        "S3_formula_match_rate":   round(sum(r["S3_formula_match"]   for r in records) / n, 4),
        "S4_canonical_equal_rate": round(sum(r["S4_canonical_equal"] for r in records) / n, 4),
        "S5_predict_logic_rate":   round(sum(r["S5_predict_logic"]   for r in records) / n, 4),
    }

    vpoint_correspondence = {
        "V1_formula_a_correct  ≡ S1_formula_a":      checkpoint_rates["S1_formula_a_rate"],
        "V2_formula_b_correct  ≡ S2_formula_b":      checkpoint_rates["S2_formula_b_rate"],
        "V3_formula_diff_coh.  ≡ S3_formula_match":  checkpoint_rates["S3_formula_match_rate"],
        "[extra]               ≡ S4_canonical_equal": checkpoint_rates["S4_canonical_equal_rate"],
        "V4_key_diff_coh.      ≡ S5_predict_logic":  checkpoint_rates["S5_predict_logic_rate"],
    }

    all_pass_rate = rate("all_pass")
    outcome_acc   = rate("outcome")

    by_diff: dict[str, list] = {}
    for r in records:
        by_diff.setdefault(r["difficulty"], []).append(r)
    diff_stats = {
        d: {
            "n":             len(recs),
            "all_pass_rate": round(sum(r["all_pass"] for r in recs) / len(recs), 4),
            "outcome_acc":   round(sum(r["outcome"]  for r in recs) / len(recs), 4),
            "S1_rate":       round(sum(r["S1_formula_a"]     for r in recs) / len(recs), 4),
            "S2_rate":       round(sum(r["S2_formula_b"]     for r in recs) / len(recs), 4),
            "S3_rate":       round(sum(r["S3_formula_match"] for r in recs) / len(recs), 4),
        }
        for d, recs in sorted(by_diff.items())
    }

    # Failure analysis helpers
    s1_fails = [r for r in records if r["S1_formula_a"] == 0 and r.get("api_success")]
    s2_fails = [r for r in records if r["S2_formula_b"] == 0 and r.get("api_success")]
    s5_fails = [r for r in records if r["S5_predict_logic"] == 0 and r.get("api_success")]

    # "Shortcut" detection: S1=0 or S2=0 (wrong formula) but outcome=True
    shortcut_cases = [
        r for r in records
        if (r["S1_formula_a"] == 0 or r["S2_formula_b"] == 0) and r.get("outcome")
    ]

    summary = {
        "n_total":       n,
        "n_parsed_ok":   sum(1 for r in records if r.get("parse_ok", False)),
        "n_all_pass":    sum(1 for r in records if r["all_pass"]),
        "all_pass_rate": all_pass_rate,
        "outcome_acc":   outcome_acc,
        **checkpoint_rates,
        "vpoint_correspondence": vpoint_correspondence,
        "by_difficulty": diff_stats,
        "n_s1_fails":        len(s1_fails),
        "n_s2_fails":        len(s2_fails),
        "n_s5_logic_fails":  len(s5_fails),
        "n_shortcut_cases":  len(shortcut_cases),
        "shortcut_note": (
            "Records where formula wrong (S1=0 or S2=0) but outcome=True: "
            "model bypassed formula calculation and took a SMILES-string shortcut."
        ),
        "token_totals": {
            "reasoning":        sum(r.get("reasoning_tokens", 0)        for r in records),
            "total_completion": sum(r.get("total_completion_tokens", 0) for r in records),
            "prompt":           sum(r.get("prompt_tokens", 0)           for r in records),
        },
    }
    return records, summary
