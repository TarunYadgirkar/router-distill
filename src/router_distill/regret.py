from dataclasses import dataclass

from .config import TIERS
from .schema import JudgedAnswer, Question, RungAnswer


@dataclass
class PolicyEval:
    name: str
    mean_score: float
    mean_cost_usd: float
    mean_latency_ms: float
    regret: float  # oracle mean score minus this policy's mean score
    score_per_dollar: float
    n: int
    by_slice: dict[str, float]  # slice -> mean regret on that slice


class OutcomeGrid:
    """question_id x tier -> (score, cost, latency). The ground truth every router is judged
    against; only questions with all three rungs scored participate."""

    def __init__(self, questions: list[Question], rungs: list[RungAnswer], judged: list[JudgedAnswer]):
        self.questions = {q.id: q for q in questions}
        self.cost = {(r.question_id, r.tier): r.cost_usd for r in rungs}
        self.latency = {(r.question_id, r.tier): r.latency_ms for r in rungs}
        self.score = {(j.question_id, j.tier): j.score for j in judged}
        self.complete_ids = [
            qid for qid in self.questions
            if all((qid, t) in self.score and (qid, t) in self.cost for t in TIERS)
        ]

    def oracle_tier(self, qid: str) -> str:
        # Best score wins; ties go to the cheapest rung — quality first, cost as tiebreak,
        # same ordering Mahogany's evidence scorer uses.
        return max(TIERS, key=lambda t: (self.score[(qid, t)], -self.cost[(qid, t)]))

    def evaluate(self, name: str, policy: dict[str, str]) -> PolicyEval:
        ids = [qid for qid in self.complete_ids if qid in policy]
        if not ids:
            raise ValueError(f"policy {name} covers none of the {len(self.complete_ids)} scored questions")

        scores, costs, latencies = [], [], []
        slice_regrets: dict[str, list[float]] = {}
        for qid in ids:
            tier = policy[qid]
            oracle = self.score[(qid, self.oracle_tier(qid))]
            scores.append(self.score[(qid, tier)])
            costs.append(self.cost[(qid, tier)])
            latencies.append(self.latency[(qid, tier)])
            regret = oracle - self.score[(qid, tier)]
            slice_regrets.setdefault(self.questions[qid].slice, []).append(regret)

        mean_score = sum(scores) / len(ids)
        mean_cost = sum(costs) / len(ids)
        oracle_mean = sum(self.score[(qid, self.oracle_tier(qid))] for qid in ids) / len(ids)
        return PolicyEval(
            name=name,
            mean_score=round(mean_score, 3),
            mean_cost_usd=round(mean_cost, 6),
            mean_latency_ms=round(sum(latencies) / len(ids), 0),
            regret=round(oracle_mean - mean_score, 3),
            score_per_dollar=round(mean_score / mean_cost, 1) if mean_cost > 0 else 0.0,
            n=len(ids),
            by_slice={s: round(sum(v) / len(v), 3) for s, v in sorted(slice_regrets.items())},
        )
