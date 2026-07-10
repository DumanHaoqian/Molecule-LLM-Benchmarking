"""Summary aggregation for MolUnd three-layer evaluation."""
import statistics


def _resolve_primary_metric(layer1_summary: dict, subtask: str) -> tuple[str, float | None, str]:
    """Return the finalized V1-aligned primary metric for a MolUnd subtask."""
    if subtask in ("fg_detect", "ring_count"):
        return ("mean_mae", layer1_summary.get("mean_mae"), "lower_better")
    if subtask == "murcko_scaffold":
        return ("avg_tanimoto", layer1_summary.get("avg_tanimoto"), "higher_better")
    return ("exact_match_acc", layer1_summary.get("exact_match_acc"), "higher_better")


def compute_summary(records: list[dict], subtask: str) -> dict:
    """Aggregate Layer 1, Layer 2, Layer 3 into final summary."""
    n = len(records)
    if n == 0:
        return {"n_total": 0}

    # Layer 1
    l1_exact = sum(r.get("layer1_exact_match", False) for r in records) / n
    l1_summary = {"exact_match_acc": round(l1_exact, 4)}
    if subtask in ("fg_detect", "ring_count"):
        l1_summary["mean_mae"] = round(statistics.mean(r.get("layer1_mae", 0) for r in records), 4)
    if subtask == "murcko_scaffold":
        l1_summary["avg_tanimoto"] = round(statistics.mean(r.get("layer1_tanimoto", 0.0) for r in records), 4)
    primary_name, primary_value, primary_direction = _resolve_primary_metric(l1_summary, subtask)
    l1_summary["primary_metric_name"] = primary_name
    l1_summary["primary_metric_value"] = primary_value
    l1_summary["primary_metric_direction"] = primary_direction

    # Layer 2 — dynamic V-point detection
    l2 = {}
    v_keys = set()
    for r in records:
        for k in r.keys():
            if k.startswith("layer2_V") and k[8:].split("_")[0].isdigit():
                v_keys.add(k)
    for k in sorted(v_keys):
        short = k.replace("layer2_", "")
        vals = [r.get(k) for r in records if r.get(k) is not None]
        if vals:
            l2[short] = round(sum(v for v in vals if v is not None) / len(vals), 4)
    ss_vals = [r.get("layer2_state_score") for r in records if r.get("layer2_state_score") is not None]
    if ss_vals:
        l2["state_score"] = round(sum(ss_vals) / len(ss_vals), 4)

    # Layer 3 Type I — dynamic checkpoint names
    def _is_checkpoint_key(k: str) -> bool:
        if k in ("all_pass", "outcome", "layer3_type1_outcome"):
            return False
        # fg_detect/ring_count: S1_syntax, S1_semantic, S2_count, etc.
        # murcko: S1_mol_rings, S2_scaffold_valid, etc.
        # ring_sys: s1_mol_rings, s2_scaf_rings, etc.
        # mutated: S1_formula_a, etc.
        # permutated: S1_canonical_a, etc.
        if k.startswith(("S1_", "S2_", "S3_", "S4_", "S5_", "s1_", "s2_", "s3_", "s4_")):
            return True
        return False

    sample_keys = list(records[0].keys()) if records else []
    t1_names = [k for k in sample_keys if _is_checkpoint_key(k)]
    t1 = {}
    for k in t1_names:
        # Some checkpoints are int (0/1), some are bool
        rate = sum(1 for r in records if r.get(k)) / n
        t1[f"{k}_rate"] = round(rate, 4)
    t1["all_pass_rate"] = round(sum(1 for r in records if r.get("all_pass")) / n, 4)
    t1["outcome_rate"] = round(sum(1 for r in records if r.get("layer3_type1_outcome")) / n, 4)

    # Layer 3 Type II
    gt_fields = [
        k for k in sample_keys
        if k.startswith("gt_match_")
        and k not in ("gt_match_all_fields", "gt_match_count", "gt_match_total")
    ]
    t2 = {}
    for k in gt_fields:
        t2[f"{k}_rate"] = round(sum(1 for r in records if r.get(k) is True) / n, 4)
    t2["all_fields_match_rate"] = round(
        sum(1 for r in records if r.get("gt_match_all_fields") is True) / n, 4
    )

    # Format compliance
    n_parse_ok = sum(1 for r in records if r.get("parse_ok"))
    n_parse_fail = n - n_parse_ok

    fmt = {
        "parse_ok_rate": round(n_parse_ok / n, 4) if n else 0.0,
        "parse_fail_rate": round(n_parse_fail / n, 4) if n else 0.0,
    }

    # Per difficulty
    by_diff: dict[str, list] = {}
    for r in records:
        by_diff.setdefault(r.get("difficulty", "unknown"), []).append(r)

    per_diff = {}
    for diff, recs in by_diff.items():
        nd = len(recs)
        per_diff[diff] = {
            "n": nd,
            "layer1_exact_match_acc": round(
                sum(r.get("layer1_exact_match", False) for r in recs) / nd, 4
            ),
            "layer2_state_score": round(
                statistics.mean(r.get("layer2_state_score", 0.0) for r in recs), 4
            ),
            "layer3_type1_all_pass": round(
                sum(r.get("all_pass") is True for r in recs) / nd, 4
            ),
            "layer3_type2_all_match": round(
                sum(r.get("gt_match_all_fields") is True for r in recs) / nd, 4
            ),
        }

    return {
        "n_total": n,
        "layer1": l1_summary,
        "layer2": l2,
        "layer3": {
            "type1": t1,
            "type2": t2,
        },
        "format_compliance": fmt,
        "by_difficulty": per_diff,
    }
