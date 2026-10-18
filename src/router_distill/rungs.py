from .config import RUNG_MAX_TOKENS, TIER_EFFORT, TIER_MODEL
from .llm import complete
from .schema import Question, RungAnswer

SYSTEM = (
    "Answer the question directly and completely. If it requires reasoning, show the key steps "
    "concisely. Do not pad."
)


def answer(question: Question, tier: str, sample: int = 0) -> RungAnswer:
    model = TIER_MODEL[tier]
    effort = TIER_EFFORT[tier]
    result = complete(model, SYSTEM, question.text, max_tokens=RUNG_MAX_TOKENS, effort=effort)
    return RungAnswer(
        question_id=question.id,
        tier=tier,
        model=model,
        effort=effort,
        answer=result.text,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
        cost_usd=result.cost_usd,
        sample=sample,
        max_tokens=result.max_tokens,
        stop_reason=result.stop_reason,
        response_id=result.response_id,
        created_at=result.created_at,
    )
