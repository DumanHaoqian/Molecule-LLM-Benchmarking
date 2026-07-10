"""
fg_detect Structured CoT Evaluation Pipeline
=============================================

Task description:
  Count how many times a specific functional group appears in a given molecule.
  The model is given a SMILES string and a functional group name, and must output
  both a SMARTS pattern and the integer count.

Structured output format (enforced via system prompt):
  [TARGET_SMARTS]: <SMARTS for the functional group, e.g. [NH2;D1]>
  [CANDIDATES]:    <atoms/positions that could potentially match>
  [CONFIRMED]:     <matches confirmed with all SMARTS constraints>
  [REJECTED]:      <rejected candidates with specific constraint failure>
  [COUNT]:         <integer>
  Answer:          <integer>

Verification points:
  V1: [TARGET_SMARTS] field present and non-empty
  V2: Value is RDKit-parseable (valid SMARTS)
  V3: SMARTS is semantically equivalent to GT fg_smarts
      (same substructure match counts on 10 diverse test molecules)
  V4: [COUNT] field matches GT fg_num (count)
  V5: [COUNT] == Answer (internal self-consistency check)

State Score = (V1 + V2 + V3 + V4) / 4

What State Score measures:
  A weighted proxy for reasoning quality at the intermediate state level.
  A model that outputs a valid, semantically correct SMARTS and the right count
  achieves full State Score = 1.0, regardless of final answer formatting.
  V3 specifically tests whether the model's SMARTS captures the correct chemistry,
  not just a superficially similar pattern.
"""
