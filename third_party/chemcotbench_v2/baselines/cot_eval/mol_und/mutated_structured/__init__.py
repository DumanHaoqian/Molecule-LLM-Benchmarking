"""
mutated Structured CoT Evaluation Pipeline
==========================================

Task description:
  Given two SMILES strings (mol A and mol B), determine if they represent the same
  or different molecules. In this dataset, mol B is always a chemically MUTATED
  version of mol A (single atom substitution or similar), so GT is always "Different".
  The diagnostic value is whether the model correctly identifies WHERE the mutation is.

Structured output format (enforced via system prompt):
  [COMPARISON_APPROACH]: <strategy used, e.g. "canonicalize then compare">
  [DIFF_DETECTED]:       <Yes/No — was any structural difference found?>
  [DIFF_LOCATION]:       <specific position or atom where difference occurs>
  [DIFF_CHEMISTRY]:      <chemical nature of the change, e.g. "O replaced by S">
  Answer:                <Same/Different>

Verification points:
  V1: [DIFF_DETECTED] and [DIFF_LOCATION] both extracted
  V2: Both smiles (mol A) and mutated (mol B) from dataset parseable by RDKit
  V3: RDKit confirms canonical(smiles) != canonical(mutated)
      (True for all valid dataset items; dataset anomaly if canonicals are equal)
  V4: diff_detected_field == "Yes" AND diff_location_field is not a none-equivalent
      (model correctly found the mutation location)
  V5: answer == "Different" (GT, always "Different" in this dataset)

State Score = (V1 + V2 + V3 + V4) / 4

What State Score measures:
  Measures the model's ability to detect and locate chemical mutations.
  V4 specifically tests whether the model found WHERE the difference is (not just
  asserting "Different"), while V5 tests final outcome correctness.
"""
