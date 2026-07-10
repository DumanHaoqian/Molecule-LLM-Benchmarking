"""
condition_ranking_structured
============================
Structured CoT evaluation pipeline for the condition_ranking task.

Compares:
  - Direct baseline (Qwen with raw query)
  - Structured CoT  (Qwen with [REACTION_TYPE] / [BEST_CONDITION] / [RANKING] / Answer format)

Primary metrics: NDCG@5, Top-1 Accuracy, MRR
"""
