from .config import JUDGE_MODEL
from .llm import complete, extract_json
from .schema import JudgedAnswer, RungAnswer

SYSTEM = (
    "You grade an answer to a question. Score 0-10 on correctness first, then completeness and "
    "usefulness. A wrong or misleading answer scores 0-2 regardless of polish. A correct but "
    "thin answer scores 5-7. A correct, complete, well-reasoned answer scores 8-10.\n"
    "punt: true if the answer dodges, says it cannot determine, or asks for missing context "
    "instead of answering.\n"
    "You are not told which model wrote the answer; grade the content only.\n"
    'Respond with JSON only: {"score": 0-10, "punt": true|false, "rationale": "<15 words>"}'
)


def judge(question_text: str, rung: RungAnswer) -> JudgedAnswer:
    user = f"Question:\n{question_text}\n\nAnswer:\n{rung.answer}"
    result = complete(JUDGE_MODEL, SYSTEM, user, max_tokens=2000, effort="medium")
    parsed = extract_json(result.text)
    score = float(parsed["score"])
    if not 0 <= score <= 10:
        raise ValueError(f"judge score out of range: {score}")
    return JudgedAnswer(
        question_id=rung.question_id,
        tier=rung.tier,
        score=score,
        punt=bool(parsed.get("punt", False)),
        judge_cost_usd=result.cost_usd,
        rationale=str(parsed.get("rationale", "")),
    )
