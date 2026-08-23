from .config import TIER_EFFORT, TIER_MODEL
from .llm import complete
from .schema import Question, RungAnswer

SYSTEM = (
    "Answer the question directly and completely. If it requires reasoning, show the key steps "
    "concisely. Do not pad."
)


def answer(question: Question, tier: str) -> RungAnswer:
    model = TIER_MODEL[tier]
    effort = TIER_EFFORT[tier]
    result = complete(model, SYSTEM, question.text, max_tokens=4000, effort=effort)
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
    )
