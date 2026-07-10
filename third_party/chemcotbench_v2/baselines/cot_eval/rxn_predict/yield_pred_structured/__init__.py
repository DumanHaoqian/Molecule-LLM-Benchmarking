"""
yield_pred Structured CoT Evaluation Pipeline
===============================================

Task description:
  Given a reaction description (type, reactants, conditions, product), predict
  the expected reaction yield percentage (0–100). GT is a float.

Structured output format (enforced via system prompt):
  [REACTION_TYPE]: <reaction type and coupling class>
  [YIELD_FACTORS]: <key factors: catalyst, ligand, solvent, base>
  [YIELD_CLASS]:   <low (0-16%) / medium (16-45%) / high (>45%)>
  Answer:          <numeric yield percentage, e.g. 72.5>

Verification points:
  V1: Answer (answer_float) is present and in [0, 100]
  V2: [YIELD_CLASS] field is present and is one of low/medium/high
  V3: yield_to_class(answer_float) matches GT yield_class
  V4: [REACTION_TYPE] field present and non-empty
  V5: [YIELD_CLASS] field text matches yield_to_class(answer_float) (internal consistency)

State Score = (V1 + V2 + V3 + V4) / 4

Primary outcome metrics:
  - Yield class accuracy (V3 rate): fraction where predicted class == GT class
  - MAE: mean |answer_float - gt_float| over records with valid answer_float

has_rxn_cls = False → select_sample uses random stratified sampling by difficulty.
Dataset: yield_pred_v2.json (2400 samples, 20 per difficulty in sample of 60).
"""
