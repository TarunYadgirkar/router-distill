import json
import os
import time
from pathlib import Path
from typing import Any

import anthropic

from .config import MODELS, cost_usd

_client: anthropic.Anthropic | None = None


def _load_project_env() -> None:
    # Tarun pastes the key into the project .env by hand (agent hooks block .env writes);
    # load it here so scripts work without exporting anything.
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _load_project_env()
        _client = anthropic.Anthropic(max_retries=4)
    return _client


class CallResult:
    def __init__(self, text: str, input_tokens: int, output_tokens: int, latency_ms: int, cost: float):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms
        self.cost_usd = cost


def complete(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 4000,
    effort: str | None = None,
) -> CallResult:
    kwargs: dict[str, Any] = {}
    if effort is not None and MODELS[model].supports_effort:
        kwargs["output_config"] = {"effort": effort}
    start = time.monotonic()
    response = client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        **kwargs,
    )
    latency_ms = int((time.monotonic() - start) * 1000)
    if response.stop_reason == "refusal":
        raise RuntimeError(f"{model} refused: {response.stop_details}")
    text = "".join(b.text for b in response.content if b.type == "text")
    return CallResult(
        text,
        response.usage.input_tokens,
        response.usage.output_tokens,
        latency_ms,
        cost_usd(model, response.usage.input_tokens, response.usage.output_tokens),
    )


def extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in: {text[:200]!r}")
    slice_ = text[start : end + 1]
    try:
        return json.loads(slice_)
    except json.JSONDecodeError:
        # Models occasionally emit python-dict style (single quotes / True) — accept it.
        import ast

        value = ast.literal_eval(slice_)
        if not isinstance(value, dict):
            raise
        return value
