"""
Verifier for ring_count formal A→B CoT.

Five verification checkpoints — correspondence to structured eval V-points:

  Checkpoint      | Structured eval | Type    | Verification logic
  ────────────────┼─────────────────┼─────────┼──────────────────────────────────────────────────
  S1_syntax       | V1 smarts_valid | Type I  | RDKit can parse step1_smarts (primary)
  S1_semantic     | V2 smarts_seman | Type II | rdkit_count(primary OR EQUIV) == gt_count
  S2_total        | V3 ring_total   | Type I  | CalcNumRings(mol) == step2_total
  S3_match        | V4 apply_coher  | Type I  | rdkit_match_count(active_smarts, mol) == step3_n
  S4_count        | [arithmetic]    | Type I  | step4_count == step3_n
  S5_rejected     | [arithmetic]    | Type I  | step5_rejected == step2_total - step4_count

  all_pass = S1_syntax AND S1_semantic AND S2_total AND S3_match AND S4_count AND S5_rejected
  outcome  = answer == gt_count  (implied by all_pass)

  S1_semantic uses the same EQUIV multi-candidate logic as fg_detect:
    Try primary SMARTS first; if rdkit_count != gt_count, try EQUIV alternatives.
    The winning candidate becomes active_smarts, used for S3_match.

  Note: S2_total is a Type I check that provides context but is independent of the
  core prediction (S1_semantic). All other post-S1 checks are Type I arithmetic.
"""
import statistics
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from baselines.cot_eval.mol_und.ring_count_structured.utils import (
    smarts_is_valid,
    apply_smarts_to_mol,
    get_ring_count,
)

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "formal_cot" / "mol_und" / "ring_count"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Single-record verification
# --------------------------------------------------------------------------- #

def verify_record(rec: dict) -> dict:
    gt_count = rec.get("gt_count", -1)
    smiles   = rec.get("smiles", "")
    parse_ok = rec.get("parse_ok", False)

    if not rec.get("api_success", False) or not parse_ok:
        return {
            "S1_syntax": 0, "S1_semantic": 0, "S2_total": 0,
            "S3_match": 0, "S4_count": 0, "S5_rejected": 0,
            "all_pass": False, "rdkit_ok": False,
            "outcome": False,
            "outcome_mae": abs(int(gt_count)) if gt_count not in (-1, None) else 0,
            "rdkit_applied_count": None,
            "active_smarts": None,
            "used_equiv": False,
            "verify_note": "api_failed_or_parse_failed",
        }

    step1_smarts   = rec.get("step1_smarts")
    step1_alts     = rec.get("step1_alts") or []
    step2_total    = rec.get("step2_total")
    step3_n        = rec.get("step3_n")
    step4_count    = rec.get("step4_count")
    step5_rejected = rec.get("step5_rejected")
    answer         = rec.get("answer")

    # ── S1_syntax (Type I) ─────────────────────────────────────────────────
    S1_syntax = int(smarts_is_valid(step1_smarts) if step1_smarts else False)

    # ── S1_semantic (Type II, multi-candidate) ─────────────────────────────
    S1_semantic        = 0
    active_smarts      = step1_smarts
    active_rdkit_count = None
    used_equiv         = False

    if S1_syntax:
        primary_count = apply_smarts_to_mol(step1_smarts, smiles)
        if primary_count is not None and primary_count == int(gt_count):
            S1_semantic        = 1
            active_smarts      = step1_smarts
            active_rdkit_count = primary_count
        else:
            for alt in step1_alts:
                if smarts_is_valid(alt):
                    alt_count = apply_smarts_to_mol(alt, smiles)
                    if alt_count is not None and alt_count == int(gt_count):
                        S1_semantic        = 1
                        active_smarts      = alt
                        active_rdkit_count = alt_count
                        used_equiv         = True
                        break
            if not S1_semantic:
                active_rdkit_count = primary_count

    # ── S2_total (Type I): RDKit CalcNumRings == step2_total ───────────────
    rdkit_total = get_ring_count(smiles)
    S2_total = int(
        rdkit_total is not None
        and step2_total is not None
        and rdkit_total == step2_total
    )

    # ── S3_match (Type I): rdkit_match_count(active_smarts) == step3_n ────
    S3_match = int(
        active_rdkit_count is not None
        and step3_n is not None
        and active_rdkit_count == step3_n
    )

    # ── S4_count (Type I): step4_count == step3_n (arithmetic) ────────────
    S4_count = int(
        step4_count is not None
        and step3_n is not None
        and step4_count == step3_n
    )

    # ── S5_rejected (Type I): step5 == step2_total - step4_count ──────────
    S5_rejected = int(
        step5_rejected is not None
        and step2_total is not None
        and step4_count is not None
        and step5_rejected == step2_total - step4_count
    )

    all_pass = bool(S1_syntax and S1_semantic and S2_total and S3_match and S4_count and S5_rejected)
    rdkit_ok = bool(S1_syntax and S3_match)

    gt_int  = int(gt_count) if gt_count not in (-1, None) else None
    outcome = (answer == gt_count)
    mae_val = abs(answer - gt_int) if (answer is not None and gt_int is not None) else (gt_int or 0)

    return {
        "S1_syntax":           S1_syntax,
        "S1_semantic":         S1_semantic,
        "S2_total":            S2_total,
        "S3_match":            S3_match,
        "S4_count":            S4_count,
        "S5_rejected":         S5_rejected,
        "all_pass":            all_pass,
        "rdkit_ok":            rdkit_ok,
        "outcome":             outcome,
        "outcome_mae":         mae_val,
        "rdkit_applied_count": active_rdkit_count,
        "rdkit_total_rings":   rdkit_total,
        "active_smarts":       active_smarts,
        "used_equiv":          used_equiv,
        "verify_note":         "equiv_used" if used_equiv else "",
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
        "S1_syntax_rate":   round(sum(r["S1_syntax"]   for r in records) / n, 4),
        "S1_semantic_rate": round(sum(r["S1_semantic"] for r in records) / n, 4),
        "S2_total_rate":    round(sum(r["S2_total"]    for r in records) / n, 4),
        "S3_match_rate":    round(sum(r["S3_match"]    for r in records) / n, 4),
        "S4_count_rate":    round(sum(r["S4_count"]    for r in records) / n, 4),
        "S5_rejected_rate": round(sum(r["S5_rejected"] for r in records) / n, 4),
    }

    vpoint_correspondence = {
        "V1_smarts_valid    ≡ S1_syntax":   checkpoint_rates["S1_syntax_rate"],
        "V2_smarts_semantic ≡ S1_semantic": checkpoint_rates["S1_semantic_rate"],
        "V3_ring_total      ≡ S2_total":    checkpoint_rates["S2_total_rate"],
        "V4_apply_coherence ≡ S3_match":    checkpoint_rates["S3_match_rate"],
    }

    all_pass_rate = rate("all_pass")
    rdkit_ok_rate = rate("rdkit_ok")
    outcome_acc   = rate("outcome")
    outcome_mae   = statistics.mean(r["outcome_mae"] for r in records)

    by_diff: dict[str, list] = {}
    for r in records:
        by_diff.setdefault(r["difficulty"], []).append(r)
    diff_stats = {
        d: {
            "n":             len(recs),
            "all_pass_rate": round(sum(r["all_pass"] for r in recs) / len(recs), 4),
            "outcome_acc":   round(sum(r["outcome"]  for r in recs) / len(recs), 4),
            "outcome_mae":   round(statistics.mean(r["outcome_mae"] for r in recs), 4),
        }
        for d, recs in sorted(by_diff.items())
    }

    by_ring: dict[str, list] = {}
    for r in records:
        by_ring.setdefault(r.get("ring_name", "unknown"), []).append(r)
    ring_stats = {
        ring: {
            "n":           len(recs),
            "all_pass":    sum(r["all_pass"] for r in recs),
            "outcome_acc": round(sum(r["outcome"] for r in recs) / len(recs), 4),
        }
        for ring, recs in sorted(by_ring.items())
    }

    s1_sem_fails = [r for r in records if r["S1_syntax"] == 1 and r["S1_semantic"] == 0]
    s2_fails     = [r for r in records if r["S2_total"] == 0]
    equiv_saves  = [r for r in records if r.get("used_equiv")]

    summary = {
        "n_total":       n,
        "n_parsed_ok":   sum(1 for r in records if r.get("parse_ok", False)),
        "n_all_pass":    sum(1 for r in records if r["all_pass"]),
        "all_pass_rate": all_pass_rate,
        "rdkit_ok_rate": rdkit_ok_rate,
        "outcome_acc":   outcome_acc,
        "outcome_mae":   round(outcome_mae, 4),
        **checkpoint_rates,
        "vpoint_correspondence": vpoint_correspondence,
        "by_difficulty":   diff_stats,
        "by_ring_name":    ring_stats,
        "n_equiv_saves":   len(equiv_saves),
        "equiv_save_rings": Counter(r.get("ring_name") for r in equiv_saves).most_common(),
        "n_s1_semantic_fails_given_syntax_ok": len(s1_sem_fails),
        "s1_semantic_fail_rings": Counter(r.get("ring_name") for r in s1_sem_fails).most_common(),
        "n_s2_total_fails":       len(s2_fails),
        "token_totals": {
            "reasoning":        sum(r.get("reasoning_tokens", 0)         for r in records),
            "total_completion": sum(r.get("total_completion_tokens", 0)  for r in records),
            "prompt":           sum(r.get("prompt_tokens", 0)            for r in records),
        },
    }
    return records, summary
