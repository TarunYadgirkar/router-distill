import ast
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic

from .config import MODELS, cost_usd
from .schema import utc_now

_client: anthropic.Anthropic | None = None


def _load_project_env() -> None:
    # The key lives in the project .env, pasted by hand; load it here so scripts work without exporting.
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


@dataclass
class CallResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float
    stop_reason: str | None = None
    response_id: str | None = None
    max_tokens: int | None = None
    created_at: str | None = None


def complete(
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    effort: str | None = None,
) -> CallResult:
    kwargs: dict[str, Any] = {}
    if effort is not None and MODELS[model].supports_effort:
        kwargs["output_config"] = {"effort": effort}
    created_at = utc_now()
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
        text=text,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd(model, response.usage.input_tokens, response.usage.output_tokens),
        stop_reason=response.stop_reason,
        response_id=response.id,
        max_tokens=max_tokens,
        created_at=created_at,
    )


def extract_json(text: str) -> dict:
    """First JSON object in a model reply: tolerates prose, code fences, trailing text and
    python-dict style ({'a': True})."""
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in: {text[:200]!r}")
    try:
        value = ast.literal_eval(text[start : end + 1])
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"unparseable object in: {text[:200]!r}") from e
    if not isinstance(value, dict):
        raise ValueError(f"not an object: {text[:200]!r}")
    return value
