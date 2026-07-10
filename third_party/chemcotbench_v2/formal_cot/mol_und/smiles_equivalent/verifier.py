"""Verifier adapter for merged SMILES equivalent records."""
from formal_cot.mol_und.mutated import verifier as mutated_verifier
from formal_cot.mol_und.permutated import verifier as permutated_verifier


def verify_record(record: dict) -> dict:
    source = record.get("source_subtask")
    if source == "mutated":
        out = dict(record)
        out.update(mutated_verifier.verify_record(record))
        return out
    if source == "permutated":
        out = dict(record)
        out.update(permutated_verifier.verify_record(record))
        return out
    out = dict(record)
    out.update({"all_pass": False, "outcome": False, "verify_error": f"unknown source_subtask: {source}"})
    return out


def verify_all(records: list[dict]) -> tuple[list[dict], dict]:
    verified = [verify_record(record) for record in records]
    n = len(verified)
    summary = {
        "n": n,
        "n_all_pass": sum(1 for r in verified if r.get("all_pass")),
        "all_pass_rate": round(sum(1 for r in verified if r.get("all_pass")) / n, 4) if n else 0.0,
        "outcome_acc": round(sum(1 for r in verified if r.get("outcome")) / n, 4) if n else 0.0,
        "source_subtask": {
            source: {
                "n": sum(1 for r in verified if r.get("source_subtask") == source),
                "all_pass_rate": round(
                    sum(1 for r in verified if r.get("source_subtask") == source and r.get("all_pass"))
                    / max(1, sum(1 for r in verified if r.get("source_subtask") == source)),
                    4,
                ),
                "outcome_acc": round(
                    sum(1 for r in verified if r.get("source_subtask") == source and r.get("outcome"))
                    / max(1, sum(1 for r in verified if r.get("source_subtask") == source)),
                    4,
                ),
            }
            for source in ("mutated", "permutated")
        },
    }
    return verified, summary


def verify_batch(records: list[dict]) -> tuple[list[dict], dict]:
    return verify_all(records)
