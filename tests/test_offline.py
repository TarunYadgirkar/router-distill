"""Offline smoke: no API calls. Run with `uv run python tests/test_offline.py`."""

import numpy as np

from router_distill import heuristic
from router_distill.config import TIERS
from router_distill.regret import OutcomeGrid
from router_distill.schema import JudgedAnswer, Question, RungAnswer
from router_distill.students import TierStudent, one_hot, soft_targets


def test_students():
    rng = np.random.default_rng(0)
    quick = [f"what is the capital of country {i}" for i in range(40)]
    deep = [f"prove that algorithm {i} converges under adversarial multi constraint load" for i in range(40)]
    thoughtful = [f"explain how system {i} handles concurrent writes and why" for i in range(40)]
    texts = quick + thoughtful + deep
    tiers = ["quick"] * 40 + ["thoughtful"] * 40 + ["deep"] * 40

    sft = TierStudent(epochs=200)
    sft.fit(texts, one_hot(tiers))
    acc = sum(p == t for p, t in zip(sft.predict(texts), tiers)) / len(tiers)
    assert acc > 0.95, f"sft train acc {acc}"

    soft = soft_targets([{"quick": 0.6, "thoughtful": 0.3, "deep": 0.1}] * len(texts))
    distilled = TierStudent(epochs=50)
    distilled.fit(texts, soft)
    probs = distilled.predict_proba(texts[:5])
    assert np.allclose(probs.sum(axis=1), 1.0)
    print(f"students ok (sft train acc {acc:.2f})")


def test_regret():
    questions = [Question(id=f"q{i}", text=f"question {i}", slice="clean-quick") for i in range(3)]
    rungs = [
        RungAnswer(q.id, t, "m", None, "a", 10, 10, 100, cost)
        for q in questions
        for t, cost in zip(TIERS, (0.001, 0.01, 0.05))
    ]
    judged = [
        JudgedAnswer(q.id, t, score, False, 0.0)
        for q in questions
        for t, score in zip(TIERS, (8.0, 8.0, 9.0))
    ]
    grid = OutcomeGrid(questions, rungs, judged)
    assert len(grid.complete_ids) == 3
    assert grid.oracle_tier("q0") == "deep"
    always_quick = grid.evaluate("always-quick", {q.id: "quick" for q in questions})
    oracle = grid.evaluate("oracle", {q.id: grid.oracle_tier(q.id) for q in questions})
    assert oracle.regret == 0.0
    assert always_quick.regret == 1.0
    assert always_quick.mean_cost_usd < oracle.mean_cost_usd
    print("regret ok")


def test_heuristic():
    assert heuristic.route("What is the capital of France?")["tier"] == "quick"
    assert heuristic.route("Prove that comparison sorting is Omega(n log n).")["tier"] == "deep"
    assert heuristic.route("Explain how TCP handshakes work and why.")["tier"] == "thoughtful"
    print("heuristic ok")


if __name__ == "__main__":
    test_students()
    test_regret()
    test_heuristic()
    print("all offline checks passed")
