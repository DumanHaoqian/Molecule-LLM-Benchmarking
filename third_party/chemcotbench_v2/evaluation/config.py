"""Configuration and path resolution for the evaluation framework."""
from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ModelConfig:
    model_name: str
    base_url: str
    api_key: str
    max_tokens: int = 32768
    temperature: float = 0.1
    reasoning_effort: str | None = None
    timeout: float = 800.0

    @classmethod
    def from_args(
        cls,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ) -> "ModelConfig":
        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL", "")
        if api_key is None:
            if "anthropic" in (base_url or "").lower() or "apiany.org" in (base_url or "").lower():
                api_key = (
                    os.environ.get("ANTHROPIC_AUTH_TOKEN")
                    or os.environ.get("ANTHROPIC_API_KEY", "")
                )
            elif "dashscope.aliyuncs.com" in (base_url or ""):
                api_key = (
                    os.environ.get("DASHSCOPE_API_KEY")
                    or os.environ.get("OPENAI_API_KEY", "")
                    or os.environ.get("PPIO_API_KEY", "")
                )
            elif "infra.chatexcel.com" in (base_url or ""):
                api_key = (
                    os.environ.get("GPT52_API_KEY")
                    or os.environ.get("CHATEXCEL_API_KEY")
                    or os.environ.get("OPENAI_API_KEY", "")
                )
            elif "yunwu.ai" in (base_url or ""):
                api_key = (
                    os.environ.get("CLAUDE46_API_KEY")
                    or os.environ.get("YUNWU_API_KEY")
                    or os.environ.get("OPENAI_API_KEY", "")
                    or os.environ.get("BLTCY_API_KEY", "")
                    or os.environ.get("GEMINI_API_KEY", "")
                )
            else:
                api_key = (
                    os.environ.get("GEMINI_API_KEY")
                    or os.environ.get("BLTCY_API_KEY", "")
                    or os.environ.get("OPENAI_API_KEY")
                    or os.environ.get("ARK_API_KEY", "")
                    or os.environ.get("PPIO_API_KEY", "")
                    or os.environ.get("DASHSCOPE_API_KEY", "")
                )
        if not api_key:
            raise EnvironmentError(
                "API key required. Set --api-key or GEMINI_API_KEY / BLTCY_API_KEY / OPENAI_API_KEY / ARK_API_KEY / PPIO_API_KEY / DASHSCOPE_API_KEY / ANTHROPIC_AUTH_TOKEN env var."
            )
        return cls(
            model_name=model,
            base_url=base_url,
            api_key=api_key,
            **kwargs,
        )


@dataclass
class EvalConfig:
    task: str
    subtask: str
    model_config: ModelConfig
    n_samples: int | None = None
    seed: int = 42
    delay: float = 0.5
    rerun_failed: bool = False
    eval_only: bool = False

    def __post_init__(self):
        self.gt_dataset_path = (
            PROJECT_ROOT
            / "results"
            / "formal_cot"
            / self.task
            / self.subtask
            / "clean_dataset.json"
        )
        safe_model = self.model_config.model_name.replace("/", "_")
        self.output_dir = (
            PROJECT_ROOT
            / "results"
            / "evaluation"
            / self.task
            / self.subtask
            / safe_model
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
