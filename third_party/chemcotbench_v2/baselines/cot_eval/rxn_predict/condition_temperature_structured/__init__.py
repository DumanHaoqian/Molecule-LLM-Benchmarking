"""
condition_temperature Structured CoT Evaluation Pipeline
=========================================================

Task description:
  Given a reaction description (type, reactants, conditions), predict the
  optimal reaction temperature in Celsius. GT is one of {20.0, 60.0, 100.0}.

Structured output format (enforced via system prompt):
  [REACTION_TYPE]: <reaction type>
  [TEMP_REASONING]: <key temperature determining factors>
  [TEMP_CLASS]:    <low (20°C) / medium (60°C) / high (100°C)>
  Answer:          <temperature: 20, 60, or 100>

Verification points:
  V1: answer_float is present (not None)
  V2: round_to_temp(answer_float) is close to a valid temp (|diff| < 15)
  V3: round_to_temp(answer_float) == gt_float  [primary outcome — 3-class accuracy]
  V4: [REACTION_TYPE] field present and non-empty
  V5: [TEMP_CLASS] field text is consistent with round_to_temp(answer_float)

State Score = (V1 + V2 + V3 + V4) / 4

Primary outcome metric:
  - Accuracy (3-class temp): fraction where round_to_temp(answer_float) == gt_float

has_rxn_cls = False → select_sample uses random stratified sampling by difficulty.
Dataset: condition_temperature_v2.json (900 samples, 20 per difficulty in sample of 60).
"""
