"""
permutated Structured CoT Evaluation Pipeline
=============================================

Task description:
  Given two SMILES strings that look DIFFERENT (different notation/traversal order)
  but actually represent the SAME molecule, determine Same or Different. GT answer
  is always "Same". The diagnostic value is whether the model explains WHY they are
  the same (equivalence proof) and notices the notation differences.

Structured output format (enforced via system prompt):
  [CANONICAL_A]:        <canonical form or key structural summary of SMILES A>
  [CANONICAL_B]:        <canonical form or key structural summary of SMILES B>
  [NOTATION_DIFFERENCES]: <apparent differences in notation, e.g. ring closure numbering>
  [EQUIVALENCE_PROOF]:  <why the two SMILES represent the same structure>
  Answer:               <Same/Different>

Verification points:
  V1: [EQUIVALENCE_PROOF] extracted and non-empty (more than 5 characters)
  V2: Both smiles (mol A) and permutated (mol B) from dataset parseable by RDKit
  V3: RDKit confirms canonical(smiles) == canonical(permutated)
      (True for all valid dataset items)
  V4: [NOTATION_DIFFERENCES] extracted and non-empty
      (model noticed there ARE notation differences)
  V5: answer == "Same" (GT, always "Same" in this dataset)

State Score = (V1 + V2 + V3 + V4) / 4

What State Score measures:
  Evaluates whether the model can recognize that two different-looking SMILES encode
  the same molecule AND provide a chemical justification. V1 checks that an equivalence
  proof was given; V4 checks that the model acknowledges notation differences exist;
  V5 checks final answer correctness.
"""
