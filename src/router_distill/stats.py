"""Paired bootstrap over eval questions, the cost-matched random-split baseline, and quality-cost curves."""

from dataclasses import dataclass, field, replace

import numpy as np
from sklearn.model_selection import KFold

from .grid import OutcomeGrid

N_BOOT = 10_000
SEED = 0
TAUS = np.linspace(0.0, 1.0, 101)
THRESHOLDS = np.linspace(0.0, 1.0, 41)
GAP_TARGETS = (0.5, 0.8)
# An operating point counts as "interior" (a real mix of tiers) when no tier gets more than this share.
INTERIOR_MAX_SHARE = 0.9
# Folds for choosing an operating point out of sample: pick on 9/10 of the eval questions, apply to the rest.
CV_TUNE_FOLDS = 10


@dataclass
class Policy:
    name: str
    group: str  # baseline | reference | teacher | student
    choice: np.ndarray  # (n,) tier index per question
    route_cost: np.ndarray  # (n,) USD spent deciding the route
    probs: np.ndarray | None = None  # (n, 3) router's tier distribution, for curves
    meta: dict = field(default_factory=dict)


def bootstrap_indices(n: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def split_score(fixed_cost: np.ndarray, fixed_score: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Expected score of the best random split between two fixed tiers at mean cost c (the upper
    hull of the three always-<tier> points). Below the cheapest tier's cost the cheapest tier is the
    comparison; above the dearest tier's cost, the best fixed tier is. Shapes: (..., 3), (..., 3), (...)."""
    c = np.asarray(c, dtype=float)
    best = np.full(c.shape, -np.inf)
    for i, j in ((0, 1), (1, 2), (0, 2)):
        ci, cj = fixed_cost[..., i], fixed_cost[..., j]
        si, sj = fixed_score[..., i], fixed_score[..., j]
        lo, hi = np.minimum(ci, cj), np.maximum(ci, cj)
        span = np.where(cj != ci, cj - ci, 1.0)
        value = si + (c - ci) / span * (sj - si)
        inside = (c >= lo) & (c <= hi)
        best = np.where(inside, np.maximum(best, value), best)
    cheapest = np.take_along_axis(fixed_score, fixed_cost.argmin(axis=-1)[..., None], axis=-1)[..., 0]
    below = c < fixed_cost.min(axis=-1)
    above = c > fixed_cost.max(axis=-1)
    best = np.where(below, cheapest, best)
    return np.where(above, fixed_score.max(axis=-1), best)


def ci(samples: np.ndarray) -> list[float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return [float(lo), float(hi)]


def p_two_sided(samples: np.ndarray) -> float:
    """Bootstrap two-sided p-value for 'mean differs from 0'."""
    if np.all(samples == 0):
        return 1.0
    p = 2 * min((samples <= 0).mean(), (samples >= 0).mean())
    return float(min(1.0, max(p, 1.0 / len(samples))))


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def per_question(grid: OutcomeGrid, policy: Policy) -> tuple[np.ndarray, np.ndarray]:
    return grid.pick(grid.score, policy.choice), grid.pick(grid.cost, policy.choice) + policy.route_cost


def gain_samples(grid: OutcomeGrid, policy: Policy, idx: np.ndarray) -> np.ndarray:
    score, cost = per_question(grid, policy)
    split = split_score(grid.cost[idx].mean(axis=1), grid.score[idx].mean(axis=1), cost[idx].mean(axis=1))
    return score[idx].mean(axis=1) - split


def point_gain(grid: OutcomeGrid, score_mean: float, cost_mean: float) -> float:
    return float(
        score_mean - split_score(grid.cost.mean(axis=0), grid.score.mean(axis=0), np.array(cost_mean))
    )


def summarize(grid: OutcomeGrid, policy: Policy, idx: np.ndarray) -> dict:
    score, cost = per_question(grid, policy)
    oracle = grid.pick(grid.score, grid.oracle_choice())
    regret = oracle - score
    gains = gain_samples(grid, policy, idx)
    vs_deep = score - grid.score[:, 2]
    slices = grid.slices
    return {
        "name": policy.name,
        "group": policy.group,
        "meta": policy.meta,
        "routes": np.bincount(policy.choice, minlength=3).tolist(),
        "score": float(score.mean()),
        "score_ci": ci(score[idx].mean(axis=1)),
        "cost": float(cost.mean()),
        "cost_ci": ci(cost[idx].mean(axis=1)),
        "route_cost": float(policy.route_cost.mean()),
        "gain": point_gain(grid, score.mean(), cost.mean()),
        "gain_ci": ci(gains),
        "gain_p": p_two_sided(gains),
        "vs_deep": float(vs_deep.mean()),
        "vs_deep_ci": ci(vs_deep[idx].mean(axis=1)),
        "regret": float(regret.mean()),
        "regret_ci": ci(regret[idx].mean(axis=1)),
        "latency_ms": float(grid.pick(grid.latency_ms, policy.choice).mean()),
        "regret_by_slice": {s: float(regret[slices == s].mean()) for s in sorted(set(slices))},
    }


def compare(grid: OutcomeGrid, a: Policy, b: Policy, idx: np.ndarray) -> dict:
    """Paired difference a - b in gain over the random split (and in mean score)."""
    diff = gain_samples(grid, a, idx) - gain_samples(grid, b, idx)
    score_a, cost_a = per_question(grid, a)
    score_b, cost_b = per_question(grid, b)
    point = point_gain(grid, score_a.mean(), cost_a.mean()) - point_gain(grid, score_b.mean(), cost_b.mean())
    return {
        "a": a.name,
        "b": b.name,
        "gain_diff": point,
        "gain_diff_ci": ci(diff),
        "p": p_two_sided(diff),
        "score_diff": float((score_a - score_b).mean()),
        "disagree": int((a.choice != b.choice).sum()),
    }


def quantile_choice(probs: np.ndarray, tau: float) -> np.ndarray:
    """Route to the tau-quantile tier of the router's distribution: tau=0 sends everything to quick,
    tau=0.5 is the median tier, tau=1 sends anything with any deep mass to deep."""
    cdf = np.cumsum(probs, axis=1)
    return (cdf >= tau - 1e-12).argmax(axis=1)


def threshold_choice(probs: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.where(probs[:, 2] > a, 2, np.where(probs[:, 0] > b, 0, 1))


def is_interior(choice: np.ndarray) -> bool:
    return np.bincount(choice, minlength=3).max() <= INTERIOR_MAX_SHARE * len(choice)


def _operating_point(grid: OutcomeGrid, policy: Policy, choice: np.ndarray) -> tuple[float, float, float]:
    score = grid.pick(grid.score, choice).mean()
    cost = (grid.pick(grid.cost, choice) + policy.route_cost).mean()
    return float(cost), float(score), point_gain(grid, score, cost)


def curve(grid: OutcomeGrid, policy: Policy, idx: np.ndarray) -> dict:
    """Sweep the router's operating point. The best points are chosen on the eval set itself, so they
    are the most favourable view of a router, not an out-of-sample estimate."""
    points = []
    best_interior, best_choice = None, None
    for tau in TAUS:
        choice = quantile_choice(policy.probs, tau)
        cost, score, gain = _operating_point(grid, policy, choice)
        points.append([cost, score])
        if is_interior(choice) and (best_interior is None or gain > best_interior["gain"]):
            best_interior = {"tau": float(tau), "cost": cost, "score": score, "gain": gain}
            best_choice = choice
    if best_interior is not None:
        best_interior["gain_ci"] = ci(gain_samples(grid, replace(policy, choice=best_choice), idx))
    best_2d, best_2d_choice = None, None
    for a in THRESHOLDS:
        for b in THRESHOLDS:
            choice = threshold_choice(policy.probs, a, b)
            if not is_interior(choice):
                continue
            cost, score, gain = _operating_point(grid, policy, choice)
            if best_2d is None or gain > best_2d["gain"]:
                best_2d = {
                    "deep_above": float(a),
                    "quick_above": float(b),
                    "cost": cost,
                    "score": score,
                    "gain": gain,
                }
                best_2d_choice = choice
    if best_2d is not None:
        best_2d["gain_ci"] = ci(gain_samples(grid, replace(policy, choice=best_2d_choice), idx))
    return {
        "points": points,
        "best_interior_gain": best_interior,
        "best_threshold_gain": best_2d,
        "cv_tuned_gain": cv_tuned(grid, policy, idx),
        "cost_to_reach": {str(f): cost_to_reach(grid, points, f) for f in GAP_TARGETS},
    }


def candidate_choices(probs: np.ndarray) -> np.ndarray:
    """(k, n) routing of every candidate operating point: the quantile rules, then the threshold pairs."""
    quantile = [quantile_choice(probs, tau) for tau in TAUS]
    pairs = [threshold_choice(probs, a, b) for a in THRESHOLDS for b in THRESHOLDS]
    return np.array(quantile + pairs)


def best_candidate(grid: OutcomeGrid, policy: Policy, candidates: np.ndarray, mask: np.ndarray) -> int:
    """Index of the interior candidate with the highest gain on the questions in mask."""
    rows = np.flatnonzero(mask)
    sub = candidates[:, rows]
    score = grid.score[rows, sub].mean(axis=1)
    cost = (grid.cost[rows, sub] + policy.route_cost[rows]).mean(axis=1)
    k = len(candidates)
    fixed_cost = np.broadcast_to(grid.cost[rows].mean(axis=0), (k, 3))
    fixed_score = np.broadcast_to(grid.score[rows].mean(axis=0), (k, 3))
    gain = score - split_score(fixed_cost, fixed_score, cost)
    counts = (sub[..., None] == np.arange(3)).sum(axis=1)
    interior = counts.max(axis=1) <= INTERIOR_MAX_SHARE * len(rows)
    return int(np.where(interior, gain, -np.inf).argmax())


def cv_tuned(grid: OutcomeGrid, policy: Policy, idx: np.ndarray, folds: int = CV_TUNE_FOLDS) -> dict:
    """Gain of an operating point chosen out of sample: for each fold, pick the best candidate on the
    other folds and route the held-out questions with it. Unlike the in-sample best point, this pays
    for the search over candidates."""
    candidates = candidate_choices(policy.probs)
    choice = np.empty(grid.n, dtype=int)
    for train, test in KFold(folds, shuffle=True, random_state=SEED).split(np.arange(grid.n)):
        mask = np.zeros(grid.n, dtype=bool)
        mask[train] = True
        choice[test] = candidates[best_candidate(grid, policy, candidates, mask), test]
    tuned = replace(policy, choice=choice)
    score, cost = per_question(grid, tuned)
    gains = gain_samples(grid, tuned, idx)
    return {
        "routes": np.bincount(choice, minlength=3).tolist(),
        "score": float(score.mean()),
        "cost": float(cost.mean()),
        "gain": point_gain(grid, score.mean(), cost.mean()),
        "gain_ci": ci(gains),
        "gain_p": p_two_sided(gains),
    }


def gap_target(grid: OutcomeGrid, fraction: float) -> float:
    fixed = grid.score.mean(axis=0)
    return float(fixed[0] + fraction * (fixed[2] - fixed[0]))


def cost_to_reach(grid: OutcomeGrid, points: list[list[float]], fraction: float) -> float | None:
    target = gap_target(grid, fraction)
    costs = [c for c, s in points if s >= target - 1e-9]
    return min(costs) if costs else None


def split_cost_to_reach(grid: OutcomeGrid, fraction: float) -> float:
    target = gap_target(grid, fraction)
    fixed_cost, fixed_score = grid.cost.mean(axis=0), grid.score.mean(axis=0)
    costs = np.linspace(fixed_cost.min(), fixed_cost.max(), 20001)
    reach = split_score(
        np.broadcast_to(fixed_cost, (len(costs), 3)), np.broadcast_to(fixed_score, (len(costs), 3)), costs
    )
    return float(costs[np.argmax(reach >= target - 1e-9)])
