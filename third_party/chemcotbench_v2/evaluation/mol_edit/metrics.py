"""Summary aggregation for MolEdit three-layer evaluation."""
import statistics


def compute_summary(records: list[dict]) -> dict:
    """Aggregate Layer 1, Layer 2, Layer 3 into final summary."""
    n = len(records)
    if n == 0:
        return {"n_total": 0}

    # Layer 1
    l1_acc = sum(r["layer1_top1_acc"] for r in records) / n
    l1_fts = statistics.mean(r["layer1_fts"] for r in records)
    l1_acc_strict = sum(r["layer1_top1_acc_strict"] for r in records) / n
    l1_v_consistency = statistics.mean(r["layer1_v_consistency"] for r in records)
    l1_eso = statistics.mean(r["layer1_edit_site_overlap"] for r in records)

    # Layer 2
    l2_keys = ["V1", "V2", "V3", "V4", "V5", "V6", "state_score"]
    l2 = {}
    for k in l2_keys:
        vals = [r.get(f"layer2_{k}", 0) for r in records if f"layer2_{k}" in r]
        l2[k] = statistics.mean(vals) if vals else 0.0

    # Layer 3 Type I — dynamic checkpoint names
    def _is_checkpoint_key(k: str) -> bool:
        if k in ("all_pass", "outcome", "layer3_type1_outcome"):
            return False
        parts = k.split("_")
        if len(parts) < 3:
            return False
        return (
            len(parts[0]) == 2
            and parts[0][0] == "s"
            and parts[0][1].isdigit()
            and parts[-1] in ("valid", "ok", "arithmetic", "match")
        )

    sample_keys = list(records[0].keys()) if records else []
    t1_names = [k for k in sample_keys if _is_checkpoint_key(k)]
    t1 = {}
    for k in t1_names:
        t1[f"{k}_rate"] = sum(r.get(k) is True for r in records) / n
    t1["all_pass_rate"] = sum(r.get("all_pass") is True for r in records) / n
    t1["outcome_rate"] = sum(r.get("layer3_type1_outcome") is True for r in records) / n

    # Layer 3 Type II
    gt_fields = [
        k for k in sample_keys
        if k.startswith("gt_match_")
        and k not in ("gt_match_all_fields", "gt_match_count", "gt_match_total")
    ]
    t2 = {}
    for k in gt_fields:
        t2[f"{k}_rate"] = sum(r.get(k) is True for r in records) / n
    t2["all_fields_match_rate"] = (
        sum(r.get("gt_match_all_fields") is True for r in records) / n
    )

    # Format compliance
    def _is_format_strict(r: dict) -> bool:
        comp = r.get("formal_format_compliance")
        if not comp:
            return False
        return all(v is True for v in comp.values())

    n_strict = sum(1 for r in records if _is_format_strict(r))
    n_relaxed = sum(
        1 for r in records
        if r.get("parse_ok") and not _is_format_strict(r)
    )
    n_parse_fail = sum(1 for r in records if not r.get("parse_ok"))

    fmt = {
        "strict_rate": round(n_strict / n, 4) if n else 0.0,
        "relaxed_rate": round(n_relaxed / n, 4) if n else 0.0,
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
            "layer1_top1_acc": round(
                sum(r["layer1_top1_acc"] for r in recs) / nd, 4
            ),
            "layer1_avg_fts": round(
                statistics.mean(r["layer1_fts"] for r in recs), 4
            ),
            "layer1_top1_acc_strict": round(
                sum(r["layer1_top1_acc_strict"] for r in recs) / nd, 4
            ),
            "layer1_v_consistency": round(
                statistics.mean(r["layer1_v_consistency"] for r in recs), 4
            ),
            "layer1_avg_edit_site_overlap": round(
                statistics.mean(r["layer1_edit_site_overlap"] for r in recs), 4
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
        "layer1": {
            "primary_metric_name": "exact_match_acc",
            "primary_metric_value": round(l1_acc_strict, 4),
            "primary_metric_direction": "higher_better",
            "top1_acc": round(l1_acc, 4),
            "avg_fts": round(l1_fts, 4),
            "top1_acc_strict": round(l1_acc_strict, 4),
            "exact_match_acc": round(l1_acc_strict, 4),
            "v_consistency": round(l1_v_consistency, 4),
            "avg_edit_site_overlap": round(l1_eso, 4),
        },
        "layer2": {k: round(v, 4) for k, v in l2.items()},
        "layer3": {
            "type1": {k: round(v, 4) for k, v in t1.items()},
            "type2": {k: round(v, 4) for k, v in t2.items()},
        },
        "format_compliance": fmt,
        "by_difficulty": per_diff,
    }
