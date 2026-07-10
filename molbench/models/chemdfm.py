"""ChemDFM model wrappers (Qwen2 chat models) — v2.0 (direct) and R (reasoning).

Each owns its chat template, system prompt, and answer parsing, and returns
clean answer strings. Registered as ModelSpecs so weights load lazily.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from ..core.model import Model
from ..core.registry import ModelSpec, register_model
from ..utils.chem import parse_answer

_BASE = "/home/haoqian/Data/SAERAG/Open-Scopes/ChemDFM-Scope"

CHEMDFM_R_SYSTEM = (
    "You are a helpful assistant that is good at reasoning. You always reason "
    "thoroughly before giving response. The reasoning process and answer are "
    "enclosed within <think> </think> and <answer> </answer> tags, "
    "respectively.\ni.e.,\n<think>\nreasoning process here\n</think>\n"
    "<answer>\nanswer here\n</answer>"
)


class ChemDFMModel(Model):
    def __init__(
        self,
        model_path: str,
        system: str,
        reasoning: bool = False,
        reasoning_budget: int = 1536,
        dtype=torch.bfloat16,
        device: str = "cuda",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.system = system
        self.reasoning = reasoning
        self.reasoning_budget = reasoning_budget

        print(f"[model] loading {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
        self.model.eval()
        print(f"[model] ready on {next(self.model.parameters()).device}")

    def _build_prompt(self, instruction: str) -> str:
        msg = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": instruction},
        ]
        return self.tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def generate(
        self,
        instructions: List[str],
        max_new_tokens: int = 256,
        batch_size: int = 8,
        do_sample: bool = False,
    ) -> List[str]:
        prompts = [self._build_prompt(x) for x in instructions]
        answers: List[str] = []
        n = len(prompts)
        for start in range(0, n, batch_size):
            enc = self.tokenizer(
                prompts[start : start + batch_size], return_tensors="pt", padding=True
            ).to(self.device)
            kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            if do_sample:
                kwargs.update(temperature=0.9, top_p=0.9, top_k=20)
            out = self.model.generate(**enc, **kwargs)
            new = out[:, enc["input_ids"].shape[1] :]
            texts = self.tokenizer.batch_decode(new, skip_special_tokens=True)
            answers.extend(parse_answer(t, self.reasoning) for t in texts)
            print(f"[gen] {min(start + batch_size, n)}/{n}", flush=True)
        return answers


def _spec(key, display_name, subdir, system, reasoning, budget) -> ModelSpec:
    path = f"{_BASE}/{subdir}"
    return ModelSpec(
        key=key,
        display_name=display_name,
        params="14B",
        build=lambda: ChemDFMModel(
            path, system=system, reasoning=reasoning, reasoning_budget=budget
        ),
    )


register_model(
    _spec("chemdfm-v2", "ChemDFM-v2.0-14B", "ChemDFM-v2.0-14B",
          "You are a helpful assistant.", False, 0)
)
register_model(
    _spec("chemdfm-r", "ChemDFM-R-14B", "ChemDFM-R-14B",
          CHEMDFM_R_SYSTEM, True, 1792)
)
