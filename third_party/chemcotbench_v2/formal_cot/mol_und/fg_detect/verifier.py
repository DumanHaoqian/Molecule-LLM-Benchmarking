"""
Verifier for fg_detect formal A→B CoT.

Five verification checkpoints — direct correspondence to structured eval V-points:

  Checkpoint      | Structured eval | Type    | Verification logic
  ────────────────┼─────────────────┼─────────┼──────────────────────────────────────────────────
  S1_syntax       | V1 smarts_valid | Type I  | RDKit can parse step1_smarts (primary)
  S1_semantic     | V2 smarts_seman | Type II | rdkit_count (primary OR any EQUIV alt) == gt_count
  S2_count        | V3 apply_coher  | Type I  | rdkit_count(active_smarts) == step2_n
  S3_arith        | V4 count_answer | Type I  | step3_count == step2_n  (arithmetic)
  S4_ans          | V4 count_answer | Type I  | answer == step3_count   (arithmetic)

  all_pass  = S1_syntax AND S1_semantic AND S2_count AND S3_arith AND S4_ans
  outcome   = answer == gt_count  (implied by all_pass)

  S1_semantic — multi-candidate logic (Approach 2):
    Gemini outputs a primary SMARTS plus optional EQUIV alternatives for notation variants
    (e.g., [SH][#6] vs [S;D1] for thiols).  S1_semantic passes if ANY candidate gives
    rdkit_count == gt_count.  The winning candidate becomes `active_smarts`, which is
    then used for S2_count validation (ensuring the formal chain remains self-consistent).

  Note on Type II failures:
    If S1_semantic fails even after trying all EQUIV candidates, it indicates one of two cases:
    (a) Gemini's SMARTS reasoning diverged from the dataset's FG definition — a genuine failure.
    (b) The GT SMARTS itself is semantically ambiguous (e.g., counting a nitro group as 2 nitroso
        matches) — a data quality issue, not a pipeline failure.  These are reported separately.
"""
import statistics
from collections import Counter
from pathlib import Path

# Reuse the exact same RDKit helpers as in the structured eval framework
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from baselines.cot_eval.mol_und.fg_detect_structured.utils import (
    smarts_is_valid,
    apply_smarts_to_mol,
)

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "formal_cot" / "mol_und" / "fg_detect"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Single-record verification
# --------------------------------------------------------------------------- #

def verify_record(rec: dict) -> dict:
    gt_count   = rec.get("gt_count", -1)
    smiles     = rec.get("smiles", "")
    parse_ok   = rec.get("parse_ok", False)

    if not rec.get("api_success", False) or not parse_ok:
        return {
            "S1_syntax": 0, "S1_semantic": 0,
            "S2_count":  0, "S3_arith":   0, "S4_ans": 0,
            "all_pass": False, "rdkit_ok": False,
            "outcome": False, "outcome_mae": abs(int(gt_count)) if gt_count != -1 else 0,
            "rdkit_applied_count": None,
            "active_smarts": None,
            "used_equiv": False,
            "verify_note": "api_failed_or_parse_failed",
        }

    step1_smarts = rec.get("step1_smarts")
    step1_alts   = rec.get("step1_alts") or []
    step2_n      = rec.get("step2_n")
    step3_count  = rec.get("step3_count")
    answer       = rec.get("answer")

    # ── S1_syntax (Type I): RDKit can parse the primary SMARTS ─────────────
    S1_syntax = int(smarts_is_valid(step1_smarts) if step1_smarts else False)

    # ── S1_semantic (Type II, multi-candidate): try primary then EQUIV alts ─
    #    active_smarts = whichever candidate first gives rdkit_count == gt_count
    #    active_rdkit_count = rdkit_count(active_smarts) on this molecule
    S1_semantic       = 0
    active_smarts     = step1_smarts
    active_rdkit_count = None
    used_equiv        = False

    if S1_syntax:
        primary_count = apply_smarts_to_mol(step1_smarts, smiles)
        if primary_count is not None and primary_count == int(gt_count):
            S1_semantic        = 1
            active_smarts      = step1_smarts
            active_rdkit_count = primary_count
        else:
            # Try EQUIV alternatives
            for alt in step1_alts:
                if smarts_is_valid(alt):
                    alt_count = apply_smarts_to_mol(alt, smiles)
                    if alt_count is not None and alt_count == int(gt_count):
                        S1_semantic        = 1
                        active_smarts      = alt
                        active_rdkit_count = alt_count
                        used_equiv         = True
                        break
            # If still not found, record primary count for diagnostics
            if not S1_semantic:
                active_rdkit_count = primary_count

    # ── S2_count (Type I): model's step2_n matches active_smarts RDKit count ─
    #    Uses active_smarts count so the chain remains self-consistent even when
    #    an EQUIV alt was selected.
    S2_count = int(
        active_rdkit_count is not None
        and step2_n is not None
        and active_rdkit_count == step2_n
    )

    # ── S3_arith (Type I): step3_count == step2_n (arithmetic) ─────────────
    S3_arith = int(
        step3_count is not None
        and step2_n is not None
        and step3_count == step2_n
    )

    # ── S4_ans (Type I): answer == step3_count (arithmetic) ─────────────────
    S4_ans = int(
        answer is not None
        and step3_count is not None
        and answer == step3_count
    )

    all_pass = bool(S1_syntax and S1_semantic and S2_count and S3_arith and S4_ans)
    rdkit_ok = bool(S1_syntax and S2_count)

    gt_int  = int(gt_count) if gt_count not in (-1, None) else None
    outcome = (answer == gt_count)
    mae_val = abs(answer - gt_int) if (answer is not None and gt_int is not None) else (gt_int or 0)

    return {
        "S1_syntax":           S1_syntax,
        "S1_semantic":         S1_semantic,
        "S2_count":            S2_count,
        "S3_arith":            S3_arith,
        "S4_ans":              S4_ans,
        "all_pass":            all_pass,
        "rdkit_ok":            rdkit_ok,
        "outcome":             outcome,
        "outcome_mae":         mae_val,
        "rdkit_applied_count": active_rdkit_count,
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
        "S2_count_rate":    round(sum(r["S2_count"]    for r in records) / n, 4),
        "S3_arith_rate":    round(sum(r["S3_arith"]    for r in records) / n, 4),
        "S4_ans_rate":      round(sum(r["S4_ans"]      for r in records) / n, 4),
    }

    # Correspondence labels for paper
    vpoint_correspondence = {
        "V1_smarts_valid   ≡ S1_syntax":   checkpoint_rates["S1_syntax_rate"],
        "V2_smarts_semantic ≡ S1_semantic": checkpoint_rates["S1_semantic_rate"],
        "V3_apply_coherence ≡ S2_count":   checkpoint_rates["S2_count_rate"],
        "V4_count_answer   ≡ S3+S4":       round(
            (checkpoint_rates["S3_arith_rate"] + checkpoint_rates["S4_ans_rate"]) / 2, 4
        ),
    }

    all_pass_rate  = rate("all_pass")
    rdkit_ok_rate  = rate("rdkit_ok")
    outcome_acc    = rate("outcome")
    outcome_mae    = statistics.mean(r["outcome_mae"] for r in records)

    # By difficulty
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

    # By fg_name
    by_fg: dict[str, list] = {}
    for r in records:
        by_fg.setdefault(r["fg_name"], []).append(r)
    fg_stats = {
        fg: {
            "n":           len(recs),
            "all_pass":    sum(r["all_pass"] for r in recs),
            "outcome_acc": round(sum(r["outcome"] for r in recs) / len(recs), 4),
        }
        for fg, recs in sorted(by_fg.items())
    }

    # Failure analysis
    s1_sem_fails = [r for r in records if r["S1_syntax"] == 1 and r["S1_semantic"] == 0]
    s2_fails     = [r for r in records if r["S1_syntax"] == 1 and r["S2_count"] == 0]
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
        "by_difficulty": diff_stats,
        "by_fg_name":    fg_stats,
        "n_equiv_saves": len(equiv_saves),
        "equiv_save_fgs": Counter(r["fg_name"] for r in equiv_saves).most_common(),
        "n_s1_semantic_fails_given_syntax_ok": len(s1_sem_fails),
        "s1_semantic_fail_fgs": Counter(r["fg_name"] for r in s1_sem_fails).most_common(),
        "n_s2_count_fails_given_syntax_ok": len(s2_fails),
        "s2_count_fail_fgs": Counter(r["fg_name"] for r in s2_fails).most_common(),
        "token_totals": {
            "reasoning": sum(r.get("reasoning_tokens", 0) for r in records),
            "total_completion": sum(r.get("total_completion_tokens", 0) for r in records),
            "prompt": sum(r.get("prompt_tokens", 0) for r in records),
        },
    }
    return records, summary
