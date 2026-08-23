from dataclasses import dataclass

TIERS = ["quick", "thoughtful", "deep"]
KINDS = ["lookup", "synthesis", "comparison", "reasoning", "code", "creative", "other"]
SLICES = ["clean-quick", "clean-thoughtful", "clean-deep", "ambiguous", "ood"]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    tier: str
    input_per_mtok: float
    output_per_mtok: float
    supports_effort: bool


# Rung ladder mirrors Bonsai's quick/thoughtful/deep. Haiku 4.5 rejects output_config.effort.
MODELS = {
    "claude-haiku-4-5": ModelSpec("claude-haiku-4-5", "quick", 1.0, 5.0, False),
    "claude-sonnet-5": ModelSpec("claude-sonnet-5", "thoughtful", 3.0, 15.0, True),
    "claude-opus-5": ModelSpec("claude-opus-5", "deep", 5.0, 25.0, True),
}

TIER_MODEL = {spec.tier: spec.id for spec in MODELS.values()}
TIER_EFFORT = {"quick": "low", "thoughtful": "medium", "deep": "high"}

TEACHERS = ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"]
JUDGE_MODEL = "claude-opus-5"
GENERATOR_MODEL = "claude-opus-5"


def cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    spec = MODELS[model_id]
    return (
        input_tokens * spec.input_per_mtok + output_tokens * spec.output_per_mtok
    ) / 1_000_000
