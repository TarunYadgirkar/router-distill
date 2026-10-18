import numpy as np
import pytest

from router_distill.config import TIERS
from router_distill.grid import build_grid
from router_distill.schema import JudgedAnswer, Question, RungAnswer
from router_distill.stats import (
    Policy,
    bootstrap_indices,
    candidate_choices,
    curve,
    cv_tuned,
    holm,
    p_two_sided,
    quantile_choice,
    split_score,
    summarize,
)

MODELS = {"quick": "claude-haiku-4-5", "thoughtful": "claude-sonnet-5", "deep": "claude-opus-5"}


def make_grid(scores: list[tuple[float, float, float]]):
    questions = [Question(f"q{i}", f"question {i}", "clean-quick") for i in range(len(scores))]
    rungs = [
        RungAnswer(q.id, t, MODELS[t], None, "answer", 100, tokens, 100, 0.0)
        for q in questions
        for t, tokens in zip(TIERS, (100, 300, 900), strict=True)
    ]
    judged = [
        JudgedAnswer(q.id, t, s, False, 0.0)
        for q, row in zip(questions, scores, strict=True)
        for t, s in zip(TIERS, row, strict=True)
    ]
    return build_grid(questions, rungs, judged)


def test_grid_reprices_from_tokens_and_keeps_only_complete_questions():
    grid = make_grid([(8, 8, 9), (5, 9, 9)])
    assert grid.n == 2
    assert grid.cost[0, 1] == pytest.approx((100 * 2.0 + 300 * 10.0) / 1e6)


def test_oracle_takes_best_score_and_cheapest_on_ties():
    grid = make_grid([(8, 8, 9), (5, 9, 9), (7, 7, 7)])
    assert grid.oracle_choice().tolist() == [2, 1, 0]


def test_split_score_interpolates_upper_hull():
    cost = np.array([1.0, 2.0, 10.0])
    score = np.array([6.0, 8.0, 9.0])
    assert split_score(cost, score, np.array(1.5)) == pytest.approx(7.0)
    assert split_score(cost, score, np.array(6.0)) == pytest.approx(8.5)
    assert split_score(cost, score, np.array(0.5)) == pytest.approx(6.0)
    assert split_score(cost, score, np.array(20.0)) == pytest.approx(9.0)


def test_fixed_tiers_have_zero_gain_and_oracle_positive_gain():
    grid = make_grid([(8, 8, 9), (5, 9, 9), (7, 7, 7), (9, 6, 9)])
    idx = bootstrap_indices(grid.n, 500)
    zeros = np.zeros(grid.n)
    for i in range(3):
        row = summarize(grid, Policy("fixed", "baseline", np.full(grid.n, i), zeros), idx)
        assert row["gain"] == pytest.approx(0.0, abs=1e-12)
    oracle = summarize(grid, Policy("oracle", "reference", grid.oracle_choice(), zeros), idx)
    assert oracle["gain"] > 0 and oracle["regret"] == 0


def test_route_cost_is_charged_to_the_router():
    grid = make_grid([(8, 8, 9), (5, 9, 9)])
    idx = bootstrap_indices(grid.n, 200)
    free = summarize(grid, Policy("t", "teacher", np.array([2, 1]), np.zeros(2)), idx)
    paid = summarize(grid, Policy("t", "teacher", np.array([2, 1]), np.full(2, 0.01)), idx)
    assert paid["cost"] == pytest.approx(free["cost"] + 0.01)
    assert paid["gain"] < free["gain"]


def test_quantile_choice_endpoints_and_median():
    probs = np.array([[0.2, 0.5, 0.3], [0.7, 0.3, 0.0]])
    assert quantile_choice(probs, 0.0).tolist() == [0, 0]
    assert quantile_choice(probs, 0.5).tolist() == [1, 0]
    assert quantile_choice(probs, 1.0).tolist() == [2, 1]


def test_holm_and_bootstrap_p():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert p_two_sided(np.zeros(100)) == 1.0
    assert p_two_sided(np.ones(100)) == pytest.approx(0.01)


def test_cv_tuned_routes_every_question_with_a_rule_chosen_without_it():
    rng = np.random.default_rng(1)
    grid = make_grid([tuple(rng.integers(4, 11, 3)) for _ in range(40)])
    probs = rng.dirichlet(np.ones(3), size=grid.n)
    policy = Policy("r", "student", probs.argmax(axis=1), np.zeros(grid.n), probs)
    idx = bootstrap_indices(grid.n, 200)
    tuned = cv_tuned(grid, policy, idx)
    assert sum(tuned["routes"]) == grid.n
    assert candidate_choices(probs).shape == (101 + 41 * 41, grid.n)
    assert set(curve(grid, policy, idx)["cv_tuned_gain"]) == set(tuned)
