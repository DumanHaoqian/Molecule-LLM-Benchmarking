"""
Verifier for murcko_scaffold formal A→B CoT.

Six verification checkpoints — correspondence to structured eval V-points:

  Checkpoint        | Structured eval  | Type    | Verification logic
  ──────────────────┼──────────────────┼─────────┼──────────────────────────────────────────────
  S1_mol_rings      | V1 mol_ring_cnt  | Type I  | CalcNumRings(mol) == step1_n_mol_rings
  S2_scaffold_valid | V2 scaf_valid    | Type I  | RDKit can parse step2_scaffold SMILES
  S2_scaffold_match | V3 scaf_correct  | Type II | canonical(step2_scaffold) == canonical(gt_scaffold)
  S3_scaf_rings     | V4 scaf_ring_cnt | Type I  | CalcNumRings(scaffold) == step3_n_scaf_rings
  S4_ring_match     | V5 ring_preserved| Type I  | (n_mol_rings == n_scaf_rings) == (step4 == "yes")
  S5_substructure   | [extra]          | Type I  | mol.HasSubstructMatch(scaffold) == (step5 == "yes")

  all_pass = S1 AND S2_valid AND S2_match AND S3 AND S4 AND S5
  outcome  = canonical(answer) == canonical(gt_scaffold)

  Type II note — S2_scaffold_match:
    The scaffold prediction IS the task answer. We compare Gemini's output with the dataset GT
    using canonical SMILES equality. Tanimoto similarity ≥ 0.9 is used as a lenient fallback
    (captures minor SMILES writing differences that don't affect molecular structure).
    For the formal CoT data generation, S2_scaffold_match must pass for the record to be clean.
"""
import statistics
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from baselines.cot_eval.mol_und.murcko_scaffold_structured.utils import (
    canonical_smiles,
    smiles_are_equal,
    get_ring_count,
    is_substructure,
)

try:
    from rdkit import Chem
    from rdkit.Chem import DataStructs
    from rdkit.Chem import rdMolDescriptors
    _RDKIT = True
except ImportError:
    _RDKIT = False

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "formal_cot" / "mol_und" / "murcko_scaffold"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _tanimoto(smiles_a: str, smiles_b: str) -> float:
    """Morgan fingerprint Tanimoto similarity between two SMILES."""
    if not _RDKIT:
        return 0.0
    try:
        from rdkit.Chem import AllChem
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)
        if mol_a is None or mol_b is None:
            return 0.0
        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, radius=2, nBits=2048)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, radius=2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp_a, fp_b)
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# Single-record verification
# --------------------------------------------------------------------------- #

def verify_record(rec: dict) -> dict:
    gt_scaffold = rec.get("gt_scaffold", "")
    mol_smiles  = rec.get("smiles", "")
    parse_ok    = rec.get("parse_ok", False)

    if not rec.get("api_success", False) or not parse_ok:
        return {
            "S1_mol_rings": 0, "S2_scaffold_valid": 0, "S2_scaffold_match": 0,
            "S3_scaf_rings": 0, "S4_ring_match": 0, "S5_substructure": 0,
            "all_pass": False, "rdkit_ok": False,
            "outcome": False, "tanimoto": 0.0,
            "verify_note": "api_failed_or_parse_failed",
        }

    step1_n_mol   = rec.get("step1_n_mol_rings")
    step2_scaf    = rec.get("step2_scaffold")
    step3_n_scaf  = rec.get("step3_n_scaf_rings")
    step4_match   = rec.get("step4_ring_match")   # "yes" / "no"
    step5_subst   = rec.get("step5_substructure")  # "yes" / "no"
    answer        = rec.get("answer")

    # ── S1_mol_rings (Type I) ──────────────────────────────────────────────
    rdkit_mol_rings = get_ring_count(mol_smiles)
    S1_mol_rings = int(
        rdkit_mol_rings is not None
        and step1_n_mol is not None
        and rdkit_mol_rings == step1_n_mol
    )

    # ── S2_scaffold_valid (Type I) ─────────────────────────────────────────
    scaf_canonical = canonical_smiles(step2_scaf) if step2_scaf else None
    S2_scaffold_valid = int(scaf_canonical is not None)

    # ── S2_scaffold_match (Type II): compare with GT ───────────────────────
    # Primary: canonical SMILES equality
    # Fallback: Tanimoto >= 0.9 (allows minor SMILES writing variants)
    tanimoto = 0.0
    S2_scaffold_match = 0
    if S2_scaffold_valid and gt_scaffold:
        if smiles_are_equal(step2_scaf, gt_scaffold):
            S2_scaffold_match = 1
            tanimoto = 1.0
        else:
            tanimoto = _tanimoto(step2_scaf, gt_scaffold)
            # Lenient fallback: Tanimoto ≥ 0.9 and same ring count
            rdkit_scaf_rings_for_gt = get_ring_count(gt_scaffold)
            rdkit_scaf_rings_pred   = get_ring_count(step2_scaf)
            if (tanimoto >= 0.9
                    and rdkit_scaf_rings_for_gt is not None
                    and rdkit_scaf_rings_pred is not None
                    and rdkit_scaf_rings_for_gt == rdkit_scaf_rings_pred):
                S2_scaffold_match = 1

    # ── S3_scaf_rings (Type I): ring count of predicted scaffold ──────────
    rdkit_scaf_rings = get_ring_count(step2_scaf) if step2_scaf else None
    S3_scaf_rings = int(
        rdkit_scaf_rings is not None
        and step3_n_scaf is not None
        and rdkit_scaf_rings == step3_n_scaf
    )

    # ── S4_ring_match (Type I): logical consistency ────────────────────────
    # Gemini says RING_MATCH(yes) iff mol_rings == scaf_rings
    # We check: (step1_n_mol == step3_n_scaf) == (step4_match == "yes")
    expected_ring_match = (
        "yes" if (step1_n_mol is not None
                  and step3_n_scaf is not None
                  and step1_n_mol == step3_n_scaf)
        else "no"
    )
    S4_ring_match = int(
        step4_match is not None
        and step4_match == expected_ring_match
    )

    # ── S5_substructure (Type I) ───────────────────────────────────────────
    rdkit_subst = is_substructure(step2_scaf, mol_smiles) if step2_scaf else None
    expected_subst = "yes" if rdkit_subst else "no"
    S5_substructure = int(
        step5_subst is not None
        and rdkit_subst is not None
        and step5_subst == expected_subst
    )

    all_pass = bool(
        S1_mol_rings and S2_scaffold_valid and S2_scaffold_match
        and S3_scaf_rings and S4_ring_match and S5_substructure
    )
    rdkit_ok = bool(S2_scaffold_valid and S5_substructure)

    outcome = smiles_are_equal(answer, gt_scaffold) if answer and gt_scaffold else False
    if not outcome and answer and gt_scaffold:
        # Lenient outcome: Tanimoto ≥ 0.9
        ans_tan = _tanimoto(answer, gt_scaffold)
        if ans_tan >= 0.9:
            outcome = True

    return {
        "S1_mol_rings":        S1_mol_rings,
        "S2_scaffold_valid":   S2_scaffold_valid,
        "S2_scaffold_match":   S2_scaffold_match,
        "S3_scaf_rings":       S3_scaf_rings,
        "S4_ring_match":       S4_ring_match,
        "S5_substructure":     S5_substructure,
        "all_pass":            all_pass,
        "rdkit_ok":            rdkit_ok,
        "outcome":             outcome,
        "tanimoto":            round(tanimoto, 4),
        "rdkit_mol_rings":     rdkit_mol_rings,
        "rdkit_scaf_rings":    rdkit_scaf_rings,
        "scaf_canonical":      scaf_canonical,
        "verify_note":         "",
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
        "S1_mol_rings_rate":      round(sum(r["S1_mol_rings"]      for r in records) / n, 4),
        "S2_valid_rate":          round(sum(r["S2_scaffold_valid"]  for r in records) / n, 4),
        "S2_match_rate":          round(sum(r["S2_scaffold_match"]  for r in records) / n, 4),
        "S3_scaf_rings_rate":     round(sum(r["S3_scaf_rings"]      for r in records) / n, 4),
        "S4_ring_match_rate":     round(sum(r["S4_ring_match"]      for r in records) / n, 4),
        "S5_substructure_rate":   round(sum(r["S5_substructure"]    for r in records) / n, 4),
    }

    vpoint_correspondence = {
        "V1_mol_ring_count     ≡ S1_mol_rings":    checkpoint_rates["S1_mol_rings_rate"],
        "V2_scaffold_valid     ≡ S2_valid":        checkpoint_rates["S2_valid_rate"],
        "V3_scaffold_correct   ≡ S2_match":        checkpoint_rates["S2_match_rate"],
        "V4_scaffold_ring_cnt  ≡ S3_scaf_rings":   checkpoint_rates["S3_scaf_rings_rate"],
        "V5_ring_preserved     ≡ S4_ring_match":   checkpoint_rates["S4_ring_match_rate"],
    }

    all_pass_rate = rate("all_pass")
    outcome_acc   = rate("outcome")
    avg_tanimoto  = statistics.mean(r.get("tanimoto", 0.0) for r in records)

    by_diff: dict[str, list] = {}
    for r in records:
        by_diff.setdefault(r["difficulty"], []).append(r)
    diff_stats = {
        d: {
            "n":             len(recs),
            "all_pass_rate": round(sum(r["all_pass"] for r in recs) / len(recs), 4),
            "outcome_acc":   round(sum(r["outcome"]  for r in recs) / len(recs), 4),
            "avg_tanimoto":  round(statistics.mean(r.get("tanimoto", 0.0) for r in recs), 4),
        }
        for d, recs in sorted(by_diff.items())
    }

    s2_fails        = [r for r in records if r["S2_scaffold_valid"] == 1 and r["S2_scaffold_match"] == 0]
    s5_fails        = [r for r in records if r["S5_substructure"] == 0]
    high_tanimoto   = [r for r in records if 0.7 <= r.get("tanimoto", 0.0) < 1.0]

    summary = {
        "n_total":        n,
        "n_parsed_ok":    sum(1 for r in records if r.get("parse_ok", False)),
        "n_all_pass":     sum(1 for r in records if r["all_pass"]),
        "all_pass_rate":  all_pass_rate,
        "outcome_acc":    outcome_acc,
        "avg_tanimoto":   round(avg_tanimoto, 4),
        **checkpoint_rates,
        "vpoint_correspondence": vpoint_correspondence,
        "by_difficulty": diff_stats,
        "n_s2_match_fails":    len(s2_fails),
        "n_s5_subst_fails":    len(s5_fails),
        "n_high_tanimoto":     len(high_tanimoto),
        "token_totals": {
            "reasoning":        sum(r.get("reasoning_tokens", 0)        for r in records),
            "total_completion": sum(r.get("total_completion_tokens", 0) for r in records),
            "prompt":           sum(r.get("prompt_tokens", 0)           for r in records),
        },
    }
    return records, summary
