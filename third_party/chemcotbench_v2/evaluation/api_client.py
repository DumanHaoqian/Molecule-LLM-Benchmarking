"""Generic API clients used by the evaluation framework."""
from openai import OpenAI
import os
import requests

from .config import ModelConfig


MODEL_NAME_MAP = {
    "dpsk-V4-Pro": "deepseek/deepseek-v4-pro",
    "yunwu-claude-sonnet-4-6": "claude-sonnet-4-6",
    "gemini-3.1-pro-preview": "gemini-3-pro-preview",
}

MODEL_NAME_MAP_DASHSCOPE = {
    "dpsk-V4-Pro": "deepseek-v4-pro",
    "zai-org/glm-5.1": "glm-5.1",
}


def _resolve_api_model_name(model_name: str, base_url: str) -> str:
    if "dashscope.aliyuncs.com" in (base_url or ""):
        return MODEL_NAME_MAP_DASHSCOPE.get(model_name, model_name)
    return MODEL_NAME_MAP.get(model_name, model_name)


def _normalize_openai_base_url(base_url: str) -> str:
    """Normalize known OpenAI-compatible gateway roots without affecting others."""
    trimmed = (base_url or "").rstrip("/")
    if trimmed == "https://api.bltcy.ai":
        return f"{trimmed}/v1"
    return base_url


def create_client(cfg: ModelConfig):
    """Create the correct client without changing existing OpenAI-compatible paths."""
    base_url = cfg.base_url or ""
    model_name = cfg.model_name or ""
    is_openai_compatible_claude_gateway = any(
        host in base_url.lower()
        for host in ("yunwu.ai", "api.bltcy.ai", "infra.chatexcel.com")
    )
    if (
        "anthropic" in base_url.lower()
        or "apiany.org" in base_url.lower()
        or (model_name.startswith("claude") and not is_openai_compatible_claude_gateway)
    ):
        return AnthropicMessagesClient(cfg)
    return OpenAICompatibleClient(cfg)


class OpenAICompatibleClient:
    def __init__(self, cfg: ModelConfig):
        cfg.base_url = _normalize_openai_base_url(cfg.base_url)
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout)
        self.cfg = cfg
        # Resolve actual API model name if a display alias is used
        self.api_model_name = _resolve_api_model_name(cfg.model_name, cfg.base_url)

    def call(self, system_prompt: str, user_prompt: str) -> dict:
        """Call the model and return a normalized dict."""
        try:
            if self._use_dashscope_glm_controls():
                return self._call_dashscope_glm(
                    self._add_glm_concise_hint(system_prompt),
                    user_prompt,
                )

            kwargs = {
                "model": self.api_model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": self.cfg.max_tokens,
                "temperature": self.cfg.temperature,
            }
            extra_body = {}
            if self.cfg.reasoning_effort:
                if self._use_bltcy_gemini_controls():
                    thinking = self._gemini_thinking_config(self.cfg.reasoning_effort)
                    if thinking:
                        extra_body["thinking"] = thinking
                else:
                    extra_body["reasoning_effort"] = self.cfg.reasoning_effort
            if (
                "dashscope.aliyuncs.com" in (self.cfg.base_url or "")
                and self.api_model_name == "deepseek-v4-pro"
            ):
                extra_body["enable_thinking"] = False
            if self._use_dashscope_glm_controls():
                extra_body["enable_thinking"] = False
            if extra_body:
                kwargs["extra_body"] = extra_body
            if self._use_bltcy_gemini_controls():
                kwargs["messages"][0]["content"] = self._add_gemini_concise_hint(system_prompt)
            elif self._use_chatexcel_gpt_controls():
                kwargs["messages"][0]["content"] = self._add_gpt_concise_hint(system_prompt)

            try:
                resp = self.client.chat.completions.create(**kwargs)
            except Exception:
                if extra_body and self._use_bltcy_gemini_controls():
                    kwargs.pop("extra_body", None)
                    resp = self.client.chat.completions.create(**kwargs)
                else:
                    raise

            msg = resp.choices[0].message
            content = msg.content or ""

            # DeepSeek v4 Pro high-reasoning mode returns reasoning separately
            reasoning = ""
            if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                reasoning = msg.reasoning_content

            # Keep only the final answer in raw_output so strict parsers see the
            # structured response first.
            if reasoning and not self._use_dashscope_glm_controls():
                full_output = f"{reasoning}\n\n{content}".strip()
            else:
                full_output = content

            return {
                "success": True,
                "content": full_output,
                "reasoning_content": reasoning,
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
                "model": resp.model,
                "finish_reason": resp.choices[0].finish_reason,
            }
        except Exception as e:
            return {
                "success": False,
                "content": "",
                "error": str(e),
                "reasoning_content": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "model": self.api_model_name,
                "finish_reason": None,
            }

    def _call_dashscope_glm(self, system_prompt: str, user_prompt: str) -> dict:
        payload = {
            "model": self.api_model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "enable_thinking": False,
        }
        if self.cfg.reasoning_effort:
            payload["reasoning_effort"] = self.cfg.reasoning_effort

        response = requests.post(
            f"{self.cfg.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.cfg.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.cfg.timeout,
        )
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        usage = body.get("usage") or {}
        return {
            "success": True,
            "content": content,
            "reasoning_content": reasoning,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "model": body.get("model", self.api_model_name),
            "finish_reason": choice.get("finish_reason"),
        }

    def _use_dashscope_glm_controls(self) -> bool:
        return (
            "dashscope.aliyuncs.com" in (self.cfg.base_url or "")
            and self.api_model_name == "glm-5.1"
        )

    def _use_bltcy_gemini_controls(self) -> bool:
        return (
            "api.bltcy.ai" in (self.cfg.base_url or "")
            and "gemini" in (self.api_model_name or "").lower()
        )

    def _use_chatexcel_gpt_controls(self) -> bool:
        return (
            "infra.chatexcel.com" in (self.cfg.base_url or "")
            and "gpt" in (self.api_model_name or "").lower()
        )

    @staticmethod
    def _gemini_thinking_config(reasoning_effort: str) -> dict | None:
        effort = reasoning_effort.strip().lower()
        if effort in {"none", "off", "false", "disabled", "disable", "no"}:
            return None
        budget_by_effort = {
            "low": 1024,
            "short": 1024,
            "medium": 4096,
            "high": 8192,
        }
        budget = budget_by_effort.get(effort)
        if budget is None:
            return None
        return {"type": "enabled", "budget_tokens": budget}

    @staticmethod
    def _add_glm_concise_hint(system_prompt: str) -> str:
        appendix = (
            "\n\nOUTPUT BUDGET:\n"
            "- Reasoning mode is disabled.\n"
            "- Keep the response concise.\n"
            "- Do not add extra explanation beyond the required structured answer.\n"
            "- Prioritize the shortest correct formatted output."
        )
        if "OUTPUT BUDGET:" in system_prompt:
            return system_prompt
        return system_prompt + appendix

    @staticmethod
    def _add_gemini_concise_hint(system_prompt: str) -> str:
        appendix = (
            "\n\nOUTPUT BUDGET:\n"
            "- Do not use hidden or extended thinking.\n"
            "- Answer directly and concisely.\n"
            "- Do not add commentary outside the required structured answer.\n"
            "- Preserve the required step names, FORMAL lines, and final Answer exactly.\n"
            "- Keep the answer concise to reduce completion tokens."
        )
        if "OUTPUT BUDGET:" in system_prompt:
            return system_prompt
        return system_prompt + appendix

    @staticmethod
    def _add_gpt_concise_hint(system_prompt: str) -> str:
        appendix = (
            "\n\nOUTPUT BUDGET:\n"
            "- Use no extended thinking.\n"
            "- Answer directly and concisely.\n"
            "- Do not add commentary outside the required structured answer.\n"
            "- Preserve the required step names, FORMAL lines, and final Answer exactly.\n"
            "- Keep the answer concise to reduce completion tokens."
        )
        if "OUTPUT BUDGET:" in system_prompt:
            return system_prompt
        return system_prompt + appendix


class AnthropicMessagesClient:
    """Minimal Anthropic Messages API client for Claude-style endpoints."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.api_model_name = cfg.model_name
        self.base_url = (cfg.base_url or os.environ.get("ANTHROPIC_BASE_URL", "")).rstrip("/")
        self.api_key = (
            cfg.api_key
            or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        if not self.base_url:
            self.base_url = "https://api.anthropic.com"

    def call(self, system_prompt: str, user_prompt: str) -> dict:
        """Call Anthropic /v1/messages and return the same normalized shape."""
        payload = {
            "model": self.api_model_name,
            "system": self._add_claude_concise_hint(system_prompt),
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
        }
        if self.cfg.reasoning_effort:
            thinking = self._thinking_config(self.cfg.reasoning_effort)
            if thinking:
                payload["thinking"] = thinking
                # Anthropic extended thinking is most compatible with temperature=1.
                payload["temperature"] = 1

        try:
            body = self._post_messages(payload)
            return self._normalize_response(body)
        except Exception as first_error:
            # Some Claude-Code-oriented proxies reject explicit thinking controls.
            # Retry once without them before reporting failure.
            if "thinking" in payload:
                payload.pop("thinking", None)
                payload["temperature"] = self.cfg.temperature
                try:
                    body = self._post_messages(payload)
                    result = self._normalize_response(body)
                    result["warning"] = f"retried_without_thinking: {first_error}"
                    return result
                except Exception as second_error:
                    first_error = second_error
            return {
                "success": False,
                "content": "",
                "error": str(first_error),
                "reasoning_content": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "model": self.api_model_name,
                "finish_reason": None,
            }

    def _post_messages(self, payload: dict) -> dict:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.post(
            f"{self.base_url}/v1/messages",
            headers=headers,
            json=payload,
            timeout=self.cfg.timeout,
        )
        if not response.ok:
            raise RuntimeError(
                f"{response.status_code} {response.reason}: {response.text[:1000]}"
            )
        return response.json()

    def _normalize_response(self, body: dict) -> dict:
        content_blocks = body.get("content") or []
        text_parts = []
        thinking_parts = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(block.get("text") or "")
            elif block_type in {"thinking", "redacted_thinking"}:
                thinking_parts.append(block.get("thinking") or block.get("text") or "")

        usage = body.get("usage") or {}
        content = "\n".join(part for part in text_parts if part).strip()
        reasoning = "\n".join(part for part in thinking_parts if part).strip()
        return {
            "success": True,
            "content": content,
            "reasoning_content": reasoning,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "model": body.get("model", self.api_model_name),
            "finish_reason": body.get("stop_reason"),
        }

    @staticmethod
    def _thinking_config(reasoning_effort: str) -> dict | None:
        effort = reasoning_effort.strip().lower()
        if effort in {"none", "off", "false", "disabled", "disable"}:
            return None
        budget_by_effort = {
            "low": 1024,
            "short": 1024,
            "medium": 4096,
            "high": 8192,
        }
        budget = budget_by_effort.get(effort)
        if budget is None:
            return None
        return {"type": "enabled", "budget_tokens": budget}

    @staticmethod
    def _add_claude_concise_hint(system_prompt: str) -> str:
        appendix = (
            "\n\nOUTPUT BUDGET:\n"
            "- Use brief internal reasoning and keep the visible response concise.\n"
            "- Do not add commentary outside the required structured answer.\n"
            "- Preserve the required step names, FORMAL lines, and final Answer exactly."
        )
        if "OUTPUT BUDGET:" in system_prompt:
            return system_prompt
        return system_prompt + appendix
