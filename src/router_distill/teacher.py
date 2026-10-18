from .config import KINDS, LABEL_MAX_TOKENS, TIERS
from .llm import complete, extract_json
from .schema import Question, TeacherLabel

SYSTEM = (
    "You are a routing classifier for an inference router. Given a question, decide which model "
    "tier should answer it and what kind of question it is.\n"
    "Tiers: quick = a lookup or single-fact restatement, a small fast model suffices; "
    "thoughtful = reasoning or synthesis over several facts, a mid model; "
    "deep = multi-constraint reasoning, proofs, design tradeoffs, or questions where being wrong "
    "is expensive, the strongest model.\n"
    f"Kinds: one of {', '.join(KINDS)}.\n"
    "Judge the QUESTION itself, not its length.\n"
    "Report your genuine uncertainty as a probability distribution over tiers — do not put 1.0 on "
    "one tier unless the call is truly unambiguous.\n"
    'Respond with JSON only: {"kind": "...", "tier_probs": {"quick": p, "thoughtful": p, "deep": p}, '
    '"reason": "<8 words>"}'
)


def label(question: Question, teacher: str) -> TeacherLabel:
    result = complete(
        teacher, SYSTEM, f"Question: {question.text}", max_tokens=LABEL_MAX_TOKENS, effort="low"
    )
    parsed = extract_json(result.text)

    raw = parsed.get("tier_probs", {})
    probs = {t: max(0.0, float(raw.get(t, 0.0))) for t in TIERS}
    total = sum(probs.values())
    if total <= 0:
        raise ValueError(f"degenerate tier_probs from {teacher}: {parsed}")
    probs = {t: p / total for t, p in probs.items()}

    kind = parsed.get("kind")
    if kind not in KINDS:
        kind = "other"

    return TeacherLabel(
        question_id=question.id,
        teacher=teacher,
        kind=kind,
        tier_probs=probs,
        reason=str(parsed.get("reason", "")),
        cost_usd=result.cost_usd,
        stop_reason=result.stop_reason,
        response_id=result.response_id,
        created_at=result.created_at,
    )
