"""
Verifier for condition_ranking top-3 ranking CoT (6-step format).

all_pass formula (strict evaluation mode):
  S1 ∧ S2 ∧ S3 ∧ S4 ∧ S5
where:
  S1 (decision_factor_valid)   : step2_decision_factor ∈ valid fields
  S2 (pair_diffs_valid)        : step3_pair_diffs covers all three pairs {1/2, 1/3, 2/3}
  S3 (pairwise_prefs_complete) : step4_pairwise_prefs has exactly 3 comparisons covering all pairs
  S4 (pairwise_prefs_consistent): the three pairwise prefs are mutually consistent (no cycles)
  S5 (ranking_valid)           : step5_ranking is a permutation of ["1","2","3"]
  S6 (top2_support_valid)      : step6_top2_support.winner == ranking[0], loser == ranking[1], field valid

Outcome (answer == gt_ranking) is tracked but NOT part of all_pass.
"""

VALID_FIELDS = {"catalyst", "ligand", "base", "reagent", "additive", "solvent"}
REQUIRED_PAIRS = {"1/2", "1/3", "2/3"}
VALID_LABELS = {"1", "2", "3"}


def _prefs_consistent(prefs: list[str]) -> bool:
    """Check that 3 pairwise prefs form a consistent total order (no cycles)."""
    if len(prefs) != 3:
        return False
    edges: set[tuple[str, str]] = set()
    for p in prefs:
        if ">" not in p:
            return False
        parts = p.split(">")
        if len(parts) != 2:
            return False
        a, b = parts[0].strip(), parts[1].strip()
        if a not in VALID_LABELS or b not in VALID_LABELS:
            return False
        edges.add((a, b))
    # Exactly 3 edges, and no cycle: there must exist a total order among 3 nodes
    # Enumerate all consistent orders and check if edges match one
    valid_orders = [
        [("1","2"),("1","3"),("2","3")],
        [("1","2"),("1","3"),("3","2")],
        [("1","2"),("3","1"),("3","2")],
        [("2","1"),("1","3"),("1","3")],
        [("2","1"),("2","3"),("1","3")],
        [("2","1"),("2","3"),("3","1")],
        [("1","3"),("2","3"),("1","2")],
        [("3","1"),("3","2"),("1","2")],
        [("3","1"),("3","2"),("2","1")],
        [("3","2"),("1","2"),("1","3")],
        [("2","3"),("1","3"),("2","1")],
        [("1","2"),("3","2"),("1","3")],
    ]
    # Better: just check there are 3 distinct edges and derive implied order
    # A cycle among 3 nodes would mean A>B, B>C, C>A
    nodes = VALID_LABELS.copy()
    beats: dict[str, set[str]] = {n: set() for n in nodes}
    for a, b in edges:
        beats[a].add(b)
    for n in nodes:
        if n in beats[n]:
            return False  # self-loop
    # Check no cycle: A beats B, B beats C, C beats A
    for a in nodes:
        for b in beats[a]:
            for c in beats[b]:
                if c in beats and a in beats[c]:
                    return False  # 3-cycle
    return True


def verify_record(rec: dict) -> dict:
    """Add verification fields to a parsed record."""
    checks: dict[str, bool] = {}
    msgs: list[str] = []

    # S1: decision_factor_valid
    df = rec.get("step2_decision_factor")
    s1 = isinstance(df, str) and df.lower() in VALID_FIELDS
    checks["S1_decision_factor"] = s1
    if not s1:
        msgs.append(f"S1 fail: decision_factor={df!r} not in {VALID_FIELDS}")

    # S2: pair_diffs covers all three pairs
    pd_dict = rec.get("step3_pair_diffs")
    if isinstance(pd_dict, dict):
        covered = set(pd_dict.keys())
        s2 = REQUIRED_PAIRS.issubset(covered)
    else:
        s2 = False
    checks["S2_pair_diffs_valid"] = s2
    if not s2:
        msgs.append(f"S2 fail: pair_diffs keys={set(pd_dict.keys()) if isinstance(pd_dict, dict) else None}, need {REQUIRED_PAIRS}")

    # S3: pairwise_prefs has 3 comparisons covering all pair combinations
    prefs = rec.get("step4_pairwise_prefs")
    if isinstance(prefs, list) and len(prefs) == 3:
        # Check all three pairs are covered (regardless of direction)
        covered_pairs = set()
        valid_prefs = True
        for p in prefs:
            if ">" not in p:
                valid_prefs = False
                break
            parts = p.split(">")
            if len(parts) != 2:
                valid_prefs = False
                break
            a, b = parts[0].strip(), parts[1].strip()
            if a not in VALID_LABELS or b not in VALID_LABELS:
                valid_prefs = False
                break
            pair_key = "/".join(sorted([a, b]))
            covered_pairs.add(pair_key)
        s3 = valid_prefs and REQUIRED_PAIRS.issubset(covered_pairs)
    else:
        s3 = False
        valid_prefs = False
    checks["S3_pairwise_complete"] = s3
    if not s3:
        msgs.append(f"S3 fail: pairwise_prefs={prefs!r}")

    # S4: pairwise prefs are mutually consistent (no cycles)
    if s3:
        s4 = _prefs_consistent(prefs)
    else:
        s4 = False
    checks["S4_pairwise_consistent"] = s4
    if not s4 and s3:
        msgs.append(f"S4 fail: pairwise prefs cycle detected in {prefs!r}")

    # S5: ranking is a valid permutation of ["1","2","3"]
    ranking = rec.get("step5_ranking")
    if isinstance(ranking, list) and sorted(ranking) == ["1", "2", "3"]:
        s5 = True
    else:
        s5 = False
    checks["S5_ranking_valid"] = s5
    if not s5:
        msgs.append(f"S5 fail: ranking={ranking!r}")

    # S6: top2_support correct
    t2 = rec.get("step6_top2_support")
    if isinstance(t2, dict) and s5:
        winner = t2.get("winner")
        loser  = t2.get("loser")
        field  = t2.get("field")
        s6 = (
            winner == ranking[0]
            and loser == ranking[1]
            and isinstance(field, str)
            and field.lower() in VALID_FIELDS
        )
    else:
        s6 = False
    checks["S6_top2_support_valid"] = s6
    if not s6:
        msgs.append(f"S6 fail: top2_support={t2!r}, ranking[0]={ranking[0] if s5 else '?'}")

    # Outcome: informational only (not in all_pass)
    gt_ranking = rec.get("gt_ranking", ["1", "2", "3"])
    answer = rec.get("answer")
    outcome = isinstance(answer, list) and answer == gt_ranking
    checks["outcome"] = outcome

    all_pass = s1 and s2 and s3 and s4 and s5 and s6

    return {
        **rec,
        **checks,
        "all_pass": all_pass,
        "verify_msgs": msgs,
    }


def verify_all(records: list[dict]) -> tuple[list[dict], dict]:
    """Verify all records. Returns (updated_records, summary_dict)."""
    verified = [verify_record(r) for r in records]

    n = len(verified)
    if n == 0:
        return verified, {}

    def rate(key: str) -> float:
        vals = [r for r in verified if r.get(key) is not None]
        if not vals:
            return 0.0
        return round(sum(1 for r in vals if r.get(key)) / n, 4)

    checkpoint_rates = {
        "S1_decision_factor_rate":     rate("S1_decision_factor"),
        "S2_pair_diffs_rate":          rate("S2_pair_diffs_valid"),
        "S3_pairwise_complete_rate":   rate("S3_pairwise_complete"),
        "S4_pairwise_consistent_rate": rate("S4_pairwise_consistent"),
        "S5_ranking_valid_rate":       rate("S5_ranking_valid"),
        "S6_top2_support_rate":        rate("S6_top2_support_valid"),
        "outcome_rate":                rate("outcome"),
    }

    all_pass_n = sum(1 for r in verified if r.get("all_pass"))
    api_ok_n   = sum(1 for r in verified if r.get("api_success"))
    parse_ok_n = sum(1 for r in verified if r.get("parse_ok"))

    by_diff: dict[str, list] = {}
    for r in verified:
        by_diff.setdefault(r.get("difficulty", "unknown"), []).append(r)

    diff_stats = {}
    for d, recs in sorted(by_diff.items()):
        diff_stats[d] = {
            "n":             len(recs),
            "all_pass_rate": round(sum(r.get("all_pass", False) for r in recs) / len(recs), 4),
            "outcome_acc":   round(sum(r.get("outcome", False) for r in recs) / len(recs), 4),
        }

    summary = {
        "total":              n,
        "api_success":        api_ok_n,
        "parse_ok":           parse_ok_n,
        "all_pass":           all_pass_n,
        "all_pass_rate":      round(all_pass_n / n, 4),
        "checkpoint_rates":   checkpoint_rates,
        "by_difficulty":      diff_stats,
        "formula":            "all_pass = S1 AND S2 AND S3 AND S4 AND S5 AND S6 (outcome tracked only)",
    }

    return verified, summary
