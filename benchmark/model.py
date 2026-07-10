"""ChemDFM model wrapper for batched, deterministic benchmark generation.

Reuses the official chat-template / Qwen2 loading pattern from
``Stage1_layer_selection_v2/infer_hook.py`` but is specialised for
benchmarking: left-padded batched generation, greedy decoding by default
(reproducible), and decoding of *only* the newly generated tokens.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_PATH = (
    "/home/haoqian/Data/SAERAG/Open-Scopes/ChemDFM-Scope/ChemDFM-v2.0-14B"
)
DEFAULT_SYSTEM = "You are a helpful assistant."


class ChemDFMModel:
    """Load a ChemDFM (Qwen2ForCausalLM) checkpoint and generate completions."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        dtype=torch.bfloat16,
        device: str = "cuda",
        system: str = DEFAULT_SYSTEM,
    ):
        self.device = device
        self.system = system
        print(f"[model] tokenizer <- {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # left padding is required for correct batched decoder-only generation
        self.tokenizer.padding_side = "left"

        print(f"[model] weights <- {model_path} (dtype={dtype})")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
        self.model.eval()
        print(
            f"[model] ready. n_layers={self.model.config.num_hidden_layers} "
            f"device={next(self.model.parameters()).device}"
        )

    def build_prompt(self, instruction: str, system: Optional[str] = None) -> str:
        message = [
            {"role": "system", "content": system or self.system},
            {"role": "user", "content": instruction},
        ]
        return self.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def generate(
        self,
        instructions: List[str],
        max_new_tokens: int = 256,
        batch_size: int = 8,
        do_sample: bool = False,
        temperature: float = 0.9,
        top_p: float = 0.9,
        top_k: int = 20,
        repetition_penalty: float = 1.05,
        system: Optional[str] = None,
        progress: bool = True,
    ) -> List[str]:
        """Generate a completion for each instruction (returns new text only)."""
        prompts = [self.build_prompt(x, system) for x in instructions]
        outputs: List[str] = []
        n = len(prompts)
        for start in range(0, n, batch_size):
            batch = prompts[start : start + batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True
            ).to(self.device)
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            if do_sample:
                gen_kwargs.update(
                    temperature=temperature, top_p=top_p, top_k=top_k
                )
            out = self.model.generate(**enc, **gen_kwargs)
            # keep only newly generated tokens (strip the prompt)
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            texts = self.tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True
            )
            outputs.extend(t.strip() for t in texts)
            if progress:
                print(f"[gen] {min(start + batch_size, n)}/{n}", flush=True)
        return outputs
