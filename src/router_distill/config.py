from dataclasses import dataclass

TIERS = ["quick", "thoughtful", "deep"]
KINDS = ["lookup", "synthesis", "comparison", "reasoning", "code", "creative", "other"]
SLICES = ["clean-quick", "clean-thoughtful", "clean-deep", "ambiguous", "ood"]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    tier: str | None  # rung it answers at; None for a model used only as a judge or teacher
    input_per_mtok: float
    output_per_mtok: float
    supports_effort: bool
    name: str


# List prices checked on PRICES_AS_OF against PRICE_SOURCE. Claude Sonnet 5 launched at $2/$10 as
# introductory pricing through 2026-08-31 and that became the standard price, so the 2026-08-22 run was
# billed at $2/$10. v0.1 of this repo priced it at $3/$15 (the cancelled post-intro price), which made
# the thoughtful rung look 1.5x more expensive than it was.
PRICES_AS_OF = "2026-10-18"
PRICE_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
BATCH_DISCOUNT = 0.5

# Rung ladder: quick / thoughtful / deep. Haiku 4.5 rejects output_config.effort. Claude Sonnet 5.5 is
# listed for the planned second judge (README "Next steps"); it answers at no rung.
MODELS = {
    "claude-haiku-4-5": ModelSpec("claude-haiku-4-5", "quick", 1.0, 5.0, False, "Claude Haiku 4.5"),
    "claude-sonnet-5": ModelSpec("claude-sonnet-5", "thoughtful", 2.0, 10.0, True, "Claude Sonnet 5"),
    "claude-opus-5": ModelSpec("claude-opus-5", "deep", 5.0, 25.0, True, "Claude Opus 5"),
    "claude-sonnet-5-5": ModelSpec("claude-sonnet-5-5", None, 2.0, 10.0, True, "Claude Sonnet 5.5"),
}

# Prices v0.1 used when it wrote each row's cost_usd. Teacher labels store only cost_usd (no token
# counts), so their cost is rescaled by current/recorded price; input and output moved by the same ratio.
RECORDED_PRICES = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-5": (5.0, 25.0),
}

TIER_MODEL = {spec.tier: spec.id for spec in MODELS.values() if spec.tier is not None}
TIER_EFFORT = {"quick": "low", "thoughtful": "medium", "deep": "high"}

TEACHERS = ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"]
SHORT = {"claude-haiku-4-5": "haiku-4.5", "claude-sonnet-5": "sonnet-5", "claude-opus-5": "opus-5"}
JUDGE_MODEL = "claude-opus-5"
GENERATOR_MODEL = "claude-opus-5"

# Adaptive thinking spends from max_tokens. v0.1 ran rungs at 4000 first, which truncated 21 answers;
# those were re-run at 16000 (see data/README.md).
RUNG_MAX_TOKENS = 16000
LABEL_MAX_TOKENS = 1500
JUDGE_MAX_TOKENS = 2000
GENERATOR_MAX_TOKENS = 4000


def cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    spec = MODELS[model_id]
    return (input_tokens * spec.input_per_mtok + output_tokens * spec.output_per_mtok) / 1_000_000


def reprice_recorded(model_id: str, recorded_cost: float) -> float:
    recorded_input, _ = RECORDED_PRICES[model_id]
    return recorded_cost * MODELS[model_id].input_per_mtok / recorded_input
