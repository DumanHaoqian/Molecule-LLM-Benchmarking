"""
Verifier for permutated formal A→B CoT.

Four verification checkpoints — all Type I (RDKit-computable, no GT label needed):

  Checkpoint          | Type   | Verification logic
  ────────────────────┼────────┼──────────────────────────────────────────────────
  S1_canonical_a      | Type I | canonical_smiles(step1_canonical_a) == canonical_smiles(mol_A)
  S2_canonical_b      | Type I | canonical_smiles(step2_canonical_b) == canonical_smiles(mol_B)
  S3_identical        | Type I | (rdkit_can_A == rdkit_can_B ? "yes" : "no") == step3_identical
  S4_predict_logic    | Type I | SMILES_IDENTICAL→PREDICT mapping AND answer matches step4_predict

  all_pass = S1 AND S2 AND S3 AND S4
  outcome  = answer == "Same"   (GT is always "Same" in this dataset)

Notes:
  - S1 uses a lenient check: we re-canonicalize Gemini's output with RDKit so that any
    valid SMILES string for the same molecule passes (handles non-standard-but-correct forms).
  - S3 always expects "yes" because mol_A and mol_B are always the same molecule (permuted).
  - S4 checks: if step3_identical == "yes" then step4_predict should be "Same" (and vice versa),
    AND the final Answer must match step4_predict.
  - Shortcut detection: S1=0 or S2=0 (wrong canonicalization) but outcome=True.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from baselines.cot_eval.mol_und.permutated_structured.utils import canonical_smiles

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "formal_cot" / "mol_und" / "permutated"
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
            "S1_canonical_a":    0,
            "S2_canonical_b":    0,
            "S3_identical":      0,
            "S4_predict_logic":  0,
            "all_pass":          False,
            "outcome":           False,
            "rdkit_canonical_a": None,
            "rdkit_canonical_b": None,
            "expected_identical": None,
            "verify_note": "api_failed_or_parse_failed",
        }

    step1_canonical_a = rec.get("step1_canonical_a")
    step2_canonical_b = rec.get("step2_canonical_b")
    step3_identical   = rec.get("step3_identical")    # "yes" or "no"
    step4_predict     = rec.get("step4_predict")      # "Same" or "Different"
    answer            = rec.get("answer")             # "Same" or "Different"

    # ── Compute RDKit ground truths ────────────────────────────────────────
    rdkit_canonical_a = canonical_smiles(smiles_a)
    rdkit_canonical_b = canonical_smiles(smiles_b)

    # Expected SMILES_IDENTICAL (always "yes" since mol_A == mol_B in this dataset)
    if rdkit_canonical_a is not None and rdkit_canonical_b is not None:
        expected_identical = "yes" if rdkit_canonical_a == rdkit_canonical_b else "no"
    else:
        expected_identical = None

    # ── S1: canonical_a correct (lenient — re-canonicalize Gemini's output) ─
    gemini_can_a = canonical_smiles(step1_canonical_a) if step1_canonical_a else None
    S1_canonical_a = int(
        rdkit_canonical_a is not None
        and gemini_can_a is not None
        and rdkit_canonical_a == gemini_can_a
    )

    # ── S2: canonical_b correct (lenient — re-canonicalize Gemini's output) ─
    gemini_can_b = canonical_smiles(step2_canonical_b) if step2_canonical_b else None
    S2_canonical_b = int(
        rdkit_canonical_b is not None
        and gemini_can_b is not None
        and rdkit_canonical_b == gemini_can_b
    )

    # ── S3: SMILES_IDENTICAL declaration correct (Type I) ─────────────────
    S3_identical = int(
        expected_identical is not None
        and step3_identical is not None
        and step3_identical == expected_identical
    )

    # ── S4: predict logic consistent + answer matches step4_predict ────────
    # SMILES_IDENTICAL("yes") → PREDICT("Same"); SMILES_IDENTICAL("no") → PREDICT("Different")
    if step3_identical is not None:
        expected_predict = "Same" if step3_identical == "yes" else "Different"
    else:
        expected_predict = None

    S4_predict_logic = int(
        expected_predict is not None
        and step4_predict is not None
        and step4_predict == expected_predict
        and answer is not None
        and answer == step4_predict
    )

    all_pass = bool(S1_canonical_a and S2_canonical_b and S3_identical and S4_predict_logic)

    # GT is always "Same" in this dataset
    outcome = (answer == "Same") if answer else False

    return {
        "S1_canonical_a":     S1_canonical_a,
        "S2_canonical_b":     S2_canonical_b,
        "S3_identical":       S3_identical,
        "S4_predict_logic":   S4_predict_logic,
        "all_pass":           all_pass,
        "outcome":            outcome,
        "rdkit_canonical_a":  rdkit_canonical_a,
        "rdkit_canonical_b":  rdkit_canonical_b,
        "expected_identical": expected_identical,
        "expected_predict":   expected_predict,
        "gemini_can_a":       gemini_can_a,
        "gemini_can_b":       gemini_can_b,
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
        "S1_canonical_a_rate":    round(sum(r["S1_canonical_a"]   for r in records) / n, 4),
        "S2_canonical_b_rate":    round(sum(r["S2_canonical_b"]   for r in records) / n, 4),
        "S3_identical_rate":      round(sum(r["S3_identical"]     for r in records) / n, 4),
        "S4_predict_logic_rate":  round(sum(r["S4_predict_logic"] for r in records) / n, 4),
    }

    all_pass_rate = rate("all_pass")
    outcome_acc   = rate("outcome")

    by_diff: dict[str, list] = {}
    for r in records:
        by_diff.setdefault(r["difficulty"], []).append(r)
    diff_stats = {
        d: {
            "n":             len(recs),
            "all_pass_rate": round(sum(r["all_pass"]        for r in recs) / len(recs), 4),
            "outcome_acc":   round(sum(r["outcome"]         for r in recs) / len(recs), 4),
            "S1_rate":       round(sum(r["S1_canonical_a"]  for r in recs) / len(recs), 4),
            "S2_rate":       round(sum(r["S2_canonical_b"]  for r in recs) / len(recs), 4),
            "S3_rate":       round(sum(r["S3_identical"]    for r in recs) / len(recs), 4),
            "S4_rate":       round(sum(r["S4_predict_logic"]for r in recs) / len(recs), 4),
        }
        for d, recs in sorted(by_diff.items())
    }

    # Shortcut detection: wrong canonicalization (S1=0 or S2=0) but outcome=True
    shortcut_cases = [
        r for r in records
        if (r["S1_canonical_a"] == 0 or r["S2_canonical_b"] == 0) and r.get("outcome")
    ]

    s1_fails = [r for r in records if r["S1_canonical_a"]   == 0 and r.get("api_success")]
    s2_fails = [r for r in records if r["S2_canonical_b"]   == 0 and r.get("api_success")]
    s4_fails = [r for r in records if r["S4_predict_logic"] == 0 and r.get("api_success")]

    summary = {
        "n_total":       n,
        "n_parsed_ok":   sum(1 for r in records if r.get("parse_ok", False)),
        "n_all_pass":    sum(1 for r in records if r["all_pass"]),
        "all_pass_rate": all_pass_rate,
        "outcome_acc":   outcome_acc,
        **checkpoint_rates,
        "by_difficulty": diff_stats,
        "n_s1_fails":          len(s1_fails),
        "n_s2_fails":          len(s2_fails),
        "n_s4_logic_fails":    len(s4_fails),
        "n_shortcut_cases":    len(shortcut_cases),
        "shortcut_note": (
            "Records where canonicalization wrong (S1=0 or S2=0) but outcome=True: "
            "model bypassed SMILES canonicalization and guessed the answer directly."
        ),
        "token_totals": {
            "reasoning":        sum(r.get("reasoning_tokens", 0)        for r in records),
            "total_completion": sum(r.get("total_completion_tokens", 0) for r in records),
            "prompt":           sum(r.get("prompt_tokens", 0)           for r in records),
        },
    }
    return records, summary
