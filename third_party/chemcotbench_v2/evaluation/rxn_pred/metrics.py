"""Summary aggregation for RxnPred three-layer evaluation."""
import statistics


def compute_summary(records: list[dict], subtask: str) -> dict:
    """Aggregate Layer 1, Layer 2, Layer 3 into final summary."""
    n = len(records)
    if n == 0:
        return {"n_total": 0}

    # Layer 1
    l1 = _aggregate_layer1(records)

    # Layer 2: dynamic V-point detection
    l2 = _aggregate_layer2(records)

    # Layer 3 Type I — dynamic checkpoint names
    t1 = _aggregate_layer3_type1(records)

    # Layer 3 Type II
    t2 = _aggregate_layer3_type2(records)

    # Format compliance
    fmt = _aggregate_format_compliance(records)

    # Per difficulty
    by_diff = _aggregate_by_difficulty(records, l1, l2, t1, t2)

    return {
        "n_total": n,
        "layer1": l1,
        "layer2": l2,
        "layer3": {"type1": t1, "type2": t2},
        "format_compliance": fmt,
        "by_difficulty": by_diff,
    }


def _aggregate_layer1(records: list[dict]) -> dict:
    n = len(records)
    result = {}

    # Top-1 Acc
    acc_vals = [r.get("layer1_top1_acc") for r in records]
    acc_vals = [v for v in acc_vals if v is not None]
    if acc_vals:
        result["top1_acc"] = round(sum(acc_vals) / n, 4)

    strict_vals = [r.get("layer1_top1_acc_strict") for r in records]
    strict_vals = [v for v in strict_vals if v is not None]
    if strict_vals:
        result["top1_acc_strict"] = round(sum(strict_vals) / n, 4)

    # FTS
    fts_vals = [r.get("layer1_fts") for r in records if r.get("layer1_fts") is not None]
    if fts_vals:
        result["avg_fts"] = round(statistics.mean(fts_vals), 4)

    # NDCG / MRR
    ndcg_vals = [r.get("layer1_ndcg") for r in records if r.get("layer1_ndcg") is not None]
    if ndcg_vals:
        result["avg_ndcg"] = round(statistics.mean(ndcg_vals), 4)
    mrr_vals = [r.get("layer1_mrr") for r in records if r.get("layer1_mrr") is not None]
    if mrr_vals:
        result["avg_mrr"] = round(statistics.mean(mrr_vals), 4)

    # MAE / RMSE / within_5
    mae_vals = [r.get("layer1_mae") for r in records if r.get("layer1_mae") is not None]
    if mae_vals:
        result["mae"] = round(statistics.mean(mae_vals), 4)
        result["rmse"] = round(
            (sum(v ** 2 for v in mae_vals) / len(mae_vals)) ** 0.5, 4
        )
    within5_vals = [r.get("layer1_within_5") for r in records if r.get("layer1_within_5") is not None]
    if within5_vals:
        result["within_5"] = round(sum(within5_vals) / n, 4)

    return result


def _aggregate_layer2(records: list[dict]) -> dict:
    n = len(records)
    # Detect V-point keys dynamically (handle both "V1" and "layer2_V1")
    v_keys = set()
    for r in records:
        for k in r.keys():
            if k.startswith("layer2_V") and k[8:].split("_")[0].isdigit():
                v_keys.add(k)
            elif k.startswith("V") and k[1:].split("_")[0].isdigit():
                v_keys.add(k)
    v_keys = sorted(v_keys)

    result = {}
    for k in v_keys:
        vals = [r.get(k) for r in records if r.get(k) is not None]
        if vals:
            # V-points are typically int (0/1) or float
            numeric_vals = []
            for v in vals:
                if isinstance(v, (int, float)):
                    numeric_vals.append(v)
                elif isinstance(v, bool):
                    numeric_vals.append(1 if v else 0)
            if numeric_vals:
                short = k.replace("layer2_", "")
                result[short] = round(sum(numeric_vals) / n, 4)

    # State score (handle both "state_score" and "layer2_state_score")
    for ss_key in ["layer2_state_score", "state_score"]:
        ss_vals = [r.get(ss_key) for r in records if r.get(ss_key) is not None]
        if ss_vals:
            result["state_score"] = round(statistics.mean(ss_vals), 4)
            break

    return result


def _aggregate_layer3_type1(records: list[dict]) -> dict:
    def _is_checkpoint_key(k: str) -> bool:
        if k in ("all_pass", "outcome", "layer3_type1_outcome"):
            return False
        parts = k.split("_")
        if len(parts) < 2:
            return False
        return (
            len(parts[0]) == 2
            and parts[0][0] == "S"
            and parts[0][1].isdigit()
        )

    n = len(records)
    sample_keys = list(records[0].keys()) if records else []
    t1_names = [k for k in sample_keys if _is_checkpoint_key(k)]
    t1 = {}
    for k in t1_names:
        rate = sum(r.get(k) is True for r in records) / n
        t1[f"{k}_rate"] = round(rate, 4)

    t1["all_pass_rate"] = sum(r.get("all_pass") is True for r in records) / n
    t1["outcome_rate"] = sum(r.get("layer3_type1_outcome") is True for r in records) / n
    return {k: round(v, 4) for k, v in t1.items()}


def _aggregate_layer3_type2(records: list[dict]) -> dict:
    n = len(records)
    gt_fields = [
        k for k in (records[0].keys() if records else [])
        if k.startswith("gt_match_")
        and k not in ("gt_match_all_fields", "gt_match_count", "gt_match_total")
    ]
    t2 = {}
    for k in gt_fields:
        t2[f"{k}_rate"] = sum(r.get(k) is True for r in records) / n
    t2["all_fields_match_rate"] = (
        sum(r.get("gt_match_all_fields") is True for r in records) / n
    )
    return {k: round(v, 4) for k, v in t2.items()}


def _aggregate_format_compliance(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"strict_rate": 0.0, "relaxed_rate": 0.0, "parse_fail_rate": 0.0}

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

    return {
        "strict_rate": round(n_strict / n, 4),
        "relaxed_rate": round(n_relaxed / n, 4),
        "parse_fail_rate": round(n_parse_fail / n, 4),
    }


def _aggregate_by_difficulty(records, l1, l2, t1, t2):
    by_diff: dict[str, list] = {}
    for r in records:
        by_diff.setdefault(r.get("difficulty", "unknown"), []).append(r)

    per_diff = {}
    for diff, recs in by_diff.items():
        nd = len(recs)
        entry = {"n": nd}

        # Layer 1 per-difficulty
        if "top1_acc" in l1:
            entry["layer1_top1_acc"] = round(
                sum(r.get("layer1_top1_acc", False) for r in recs) / nd, 4
            )
        if "avg_fts" in l1:
            fts_vals = [r.get("layer1_fts", 0.0) for r in recs]
            entry["layer1_avg_fts"] = round(statistics.mean(fts_vals), 4)
        if "top1_acc_strict" in l1:
            entry["layer1_top1_acc_strict"] = round(
                sum(r.get("layer1_top1_acc_strict", False) for r in recs) / nd, 4
            )

        # Layer 2 per-difficulty
        if "state_score" in l2:
            ss_vals = [r.get("layer2_state_score", r.get("state_score", 0.0)) for r in recs]
            entry["layer2_state_score"] = round(statistics.mean(ss_vals), 4)

        # Layer 3 Type I per-difficulty
        entry["layer3_type1_all_pass"] = round(
            sum(r.get("all_pass") is True for r in recs) / nd, 4
        )

        # Layer 3 Type II per-difficulty
        entry["layer3_type2_all_match"] = round(
            sum(r.get("gt_match_all_fields") is True for r in recs) / nd, 4
        )

        per_diff[diff] = entry

    return per_diff
