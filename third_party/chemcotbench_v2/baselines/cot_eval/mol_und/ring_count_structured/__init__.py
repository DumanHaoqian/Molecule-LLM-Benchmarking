"""
ring_count Structured CoT Evaluation Pipeline
==============================================

Core innovation (Feishu Doc §Chain-of-Thoughts 构建与评估, Method 2):
  Instead of free-text CoT → regex-scan (brittle), we require the model to output
  structured intermediate states (A→B formal language), which are then verified
  deterministically by RDKit.

Structured output format (enforced via prompt):
  [TARGET_SMARTS]: <SMARTS pattern the model believes corresponds to ring_name>
  [RINGS_FOUND]: <all ring systems identified in the molecule>
  [ACCEPTED]: <ring systems confirmed to match target, or "none">
  [REJECTED]: <ring systems rejected and why>
  [COUNT]: <integer>
  Answer: <integer>

Evaluation:
  V1: [TARGET_SMARTS] field present and non-empty
  V2: Value is RDKit-parseable (valid SMARTS)
  V3: SMARTS is semantically equivalent to GT SMARTS (fingerprint on test set)
  V4: [COUNT] matches GT count
  Outcome: Answer matches GT count

Reference: Feishu Doc "通过(A->B)形式化语言验证" + VPRM (2025)
"""
