from dataclasses import dataclass

import numpy as np

from .config import JUDGE_MODEL, MODELS, TIERS, cost_usd
from .schema import JudgedAnswer, Question, RungAnswer


@dataclass
class OutcomeGrid:
    """The ground truth every router is scored against: for each eval question with all three rungs
    answered and judged, a (question, tier) matrix of judge score, answer cost and latency. Costs are
    recomputed from token counts at current list prices, not read from the stored cost_usd."""

    questions: list[Question]
    score: np.ndarray  # (n, 3)
    cost: np.ndarray  # (n, 3) USD per answer
    latency_ms: np.ndarray  # (n, 3)
    answer_chars: np.ndarray  # (n, 3)
    truncated: np.ndarray  # (n, 3) bool: hit max_tokens
    punt: np.ndarray  # (n, 3) bool

    @property
    def n(self) -> int:
        return len(self.questions)

    @property
    def ids(self) -> list[str]:
        return [q.id for q in self.questions]

    @property
    def texts(self) -> list[str]:
        return [q.text for q in self.questions]

    @property
    def slices(self) -> np.ndarray:
        return np.array([q.slice for q in self.questions])

    def oracle_choice(self) -> np.ndarray:
        """Best score per question; ties go to the cheapest rung."""
        best = self.score.max(axis=1, keepdims=True)
        return np.where(self.score == best, self.cost, np.inf).argmin(axis=1)

    def pick(self, matrix: np.ndarray, choice: np.ndarray) -> np.ndarray:
        return matrix[np.arange(self.n), choice]

    def subset(self, mask: np.ndarray) -> "OutcomeGrid":
        return OutcomeGrid(
            [q for q, keep in zip(self.questions, mask, strict=True) if keep],
            self.score[mask],
            self.cost[mask],
            self.latency_ms[mask],
            self.answer_chars[mask],
            self.truncated[mask],
            self.punt[mask],
        )

    def with_cost(self, cost: np.ndarray) -> "OutcomeGrid":
        return OutcomeGrid(
            self.questions, self.score, cost, self.latency_ms, self.answer_chars, self.truncated, self.punt
        )


def _is_truncated(r: RungAnswer) -> bool:
    if r.stop_reason is not None:
        return r.stop_reason == "max_tokens"
    return r.max_tokens is not None and r.output_tokens >= r.max_tokens


def build_grid(
    questions: list[Question],
    rungs: list[RungAnswer],
    judged: list[JudgedAnswer],
    sample: int = 0,
    judge: str = JUDGE_MODEL,
    judge_pass: int = 0,
    prices: dict[str, tuple[float, float]] | None = None,
) -> OutcomeGrid:
    answers = {(r.question_id, r.tier): r for r in rungs if r.sample == sample}
    scores = {
        (j.question_id, j.tier): j
        for j in judged
        if j.sample == sample and j.judge == judge and j.judge_pass == judge_pass
    }
    kept = [q for q in questions if all((q.id, t) in answers and (q.id, t) in scores for t in TIERS)]
    if not kept:
        raise ValueError("outcome grid is empty: run run_rungs.py and judge_answers.py first")

    def matrix(fn) -> np.ndarray:
        return np.array([[fn(q.id, t) for t in TIERS] for q in kept])

    def price(q: str, t: str) -> float:
        r = answers[(q, t)]
        if prices is None:
            return cost_usd(r.model, r.input_tokens, r.output_tokens)
        rate_in, rate_out = prices.get(
            r.model, (MODELS[r.model].input_per_mtok, MODELS[r.model].output_per_mtok)
        )
        return (r.input_tokens * rate_in + r.output_tokens * rate_out) / 1_000_000

    return OutcomeGrid(
        questions=kept,
        score=matrix(lambda q, t: scores[(q, t)].score),
        cost=matrix(price),
        latency_ms=matrix(lambda q, t: answers[(q, t)].latency_ms),
        answer_chars=matrix(lambda q, t: len(answers[(q, t)].answer)),
        truncated=matrix(lambda q, t: _is_truncated(answers[(q, t)])).astype(bool),
        punt=matrix(lambda q, t: scores[(q, t)].punt).astype(bool),
    )
