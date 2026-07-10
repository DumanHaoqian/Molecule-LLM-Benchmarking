"""Model registry: per-model paths, params, system prompts and output format.

Generation is model-specific (system prompt, reasoning vs. direct, answer
parsing); metrics are not. Keeping all model quirks here lets ``generate.py``
stay generic and lets the eval stage treat every model's predictions uniformly.
"""
from __future__ import annotations

CHEMDFM_R_SYSTEM = (
    "You are a helpful assistant that is good at reasoning. You always reason "
    "thoroughly before giving response. The reasoning process and answer are "
    "enclosed within <think> </think> and <answer> </answer> tags, "
    "respectively.\ni.e.,\n<think>\nreasoning process here\n</think>\n"
    "<answer>\nanswer here\n</answer>"
)

_BASE = "/home/haoqian/Data/SAERAG/Open-Scopes/ChemDFM-Scope"

MODEL_REGISTRY = {
    # ChemDFM-v2.0-14B: direct-answer chat model (Qwen2.5-14B based).
    "chemdfm-v2": {
        "display_name": "ChemDFM-v2.0-14B",
        "path": f"{_BASE}/ChemDFM-v2.0-14B",
        "params": "14B",
        "system": "You are a helpful assistant.",
        "reasoning": False,
        # generous caps: captions are long, SMILES answers short
        "max_new_tokens": {"captioning": 512, "caption2smiles": 256},
    },
    # ChemDFM-R-14B: reasoning model, emits <think>..</think><answer>..</answer>.
    "chemdfm-r": {
        "display_name": "ChemDFM-R-14B",
        "path": f"{_BASE}/ChemDFM-R-14B",
        "params": "14B",
        "system": CHEMDFM_R_SYSTEM,
        "reasoning": True,
        # reasoning chains need a lot of room before the <answer> block
        "max_new_tokens": {"captioning": 2048, "caption2smiles": 2048},
    },
}


def get_model_cfg(key: str) -> dict:
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown model '{key}'. choices: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[key]
