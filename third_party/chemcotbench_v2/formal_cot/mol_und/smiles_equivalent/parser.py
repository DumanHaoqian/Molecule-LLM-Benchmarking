"""Parser adapter for merged SMILES equivalent records.

The merged task intentionally keeps the original mutated/permutated templates.
Each record carries source_subtask and is parsed by the corresponding parser.
"""
from formal_cot.mol_und.mutated import parser as mutated_parser
from formal_cot.mol_und.permutated import parser as permutated_parser


def _parse_record(record: dict) -> dict:
    source = record.get("source_subtask")
    if source == "mutated":
        return mutated_parser.parse_formal_output(record.get("raw_output", ""))
    if source == "permutated":
        return permutated_parser.parse_formal_output(record.get("raw_output", ""))
    return {"parse_ok": False, "parse_error": f"unknown source_subtask: {source}"}


def parse_all(records: list[dict]) -> list[dict]:
    for rec in records:
        rec.update(_parse_record(rec))
    return records


def parse_batch(records: list[dict]) -> list[dict]:
    return parse_all(records)
