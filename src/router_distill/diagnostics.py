"""Measurement checks: how the judge scores, how much data a comparison needs, whether conclusions
survive the v0.1 prices or dropping the truncated answer, and what the committed data cost."""

from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import spearmanr

from .config import RECORDED_PRICES, SHORT, TEACHERS, TIERS, cost_usd, reprice_recorded
from .data import load_answers, load_eval, load_judged, load_labels
from .grid import OutcomeGrid, build_grid
from .stats import Policy, bootstrap_indices, ci, gain_samples, point_gain

if TYPE_CHECKING:
    from .analysis import Inputs

SENSITIVITY_BOOT = 2000
Z95 = 1.96
HALF_WIDTHS = (0.05, 0.03)


def judge_diagnostics(grid: OutcomeGrid) -> dict:
    score = grid.score
    centered_len = np.log(grid.answer_chars) - np.log(grid.answer_chars).mean(axis=1, keepdims=True)
    centered_score = score - score.mean(axis=1, keepdims=True)
    rho, p = spearmanr(centered_len.ravel(), centered_score.ravel())
    best = score.max(axis=1)
    quick_slice = grid.slices == "clean-quick"
    truncated = [
        {"question_id": grid.ids[i], "tier": TIERS[j], "score": float(score[i, j])}
        for i, j in zip(*np.nonzero(grid.truncated), strict=True)
    ]
    punts = Counter(f"{grid.slices[i]}/{TIERS[j]}" for i, j in zip(*np.nonzero(grid.punt), strict=True))
    return {
        "score_counts": {
            t: {str(int(k)): int(v) for k, v in sorted(Counter(score[:, i]).items())}
            for i, t in enumerate(TIERS)
        },
        "deep_ties_or_beats": int((score[:, 2] >= best).sum()),
        "deep_9_or_10": int((score[:, 2] >= 9).sum()),
        "clean_quick_n": int(quick_slice.sum()),
        "clean_quick_deep_beats_quick": int((score[quick_slice, 2] > score[quick_slice, 0]).sum()),
        "length_rho": float(rho),
        "length_p": f"{p:.0e}",
        "truncated": truncated,
        "punts": dict(sorted(punts.items())),
        "slice_n": {s: int((grid.slices == s).sum()) for s in sorted(set(grid.slices))},
    }


def power(grid: OutcomeGrid, policies: dict[str, Policy], comparisons: list[dict]) -> dict:
    """Per-question SD of paired score differences, and the eval size a 95% CI of a given half-width needs."""
    sds = []
    for row in comparisons:
        a, b = policies[row["a"]], policies[row["b"]]
        diff = grid.pick(grid.score, a.choice) - grid.pick(grid.score, b.choice)
        sds.append(float(diff.std(ddof=1)))
    sd = float(np.median(sds))
    return {
        "median_sd": sd,
        "max_sd": float(max(sds)),
        "n_for_half_width": {str(h): int(np.ceil((Z95 * sd / h) ** 2)) for h in HALF_WIDTHS},
        "half_width_now": float(Z95 * sd / np.sqrt(grid.n)),
    }


def _gain_row(grid: OutcomeGrid, policy: Policy, idx: np.ndarray) -> dict:
    score = grid.pick(grid.score, policy.choice).mean()
    cost = (grid.pick(grid.cost, policy.choice) + policy.route_cost).mean()
    return {"gain": point_gain(grid, score, cost), "gain_ci": ci(gain_samples(grid, policy, idx))}


def sensitivity(inp: "Inputs", policies: list[Policy]) -> dict:
    grid = inp.grid
    v01 = build_grid(grid.questions, load_answers(), load_judged(), prices=RECORDED_PRICES)
    recorded_route = {
        f"teacher:{SHORT[teacher]}": np.array([label.cost_usd for label in inp.eval_labels[teacher]])
        for teacher in TEACHERS
    }
    idx = bootstrap_indices(grid.n, SENSITIVITY_BOOT)
    at_v01 = {}
    for p in policies:
        if p.group == "reference":
            continue
        priced = replace(p, route_cost=recorded_route.get(p.name, p.route_cost))
        at_v01[p.name] = _gain_row(v01, priced, idx)

    keep = ~grid.truncated.any(axis=1)
    trimmed = grid.subset(keep)
    idx_trim = bootstrap_indices(trimmed.n, SENSITIVITY_BOOT)
    no_trunc = {}
    for p in policies:
        if p.group == "reference":
            continue
        sub = replace(p, choice=p.choice[keep], route_cost=p.route_cost[keep], probs=None)
        no_trunc[p.name] = _gain_row(trimmed, sub, idx_trim)
    return {
        "v01_prices": at_v01,
        "v01_fixed_cost": v01.cost.mean(axis=0).tolist(),
        "without_truncated": no_trunc,
        "without_truncated_n": int(trimmed.n),
    }


def spend() -> dict:
    """What the committed rows cost, at the prices recorded in the rows and at current list prices."""
    answers = load_answers()
    judged = load_judged()
    labels = {
        (stem, teacher): list(load_labels(stem, teacher).values())
        for stem in ("questions", "questions_train")
        for teacher in TEACHERS
    }
    label_recorded = sum(label.cost_usd for rows in labels.values() for label in rows)
    label_current = sum(
        reprice_recorded(teacher, label.cost_usd) for (_, teacher), rows in labels.items() for label in rows
    )
    rungs_recorded = sum(r.cost_usd for r in answers)
    rungs_current = sum(cost_usd(r.model, r.input_tokens, r.output_tokens) for r in answers)
    judge_cost = sum(j.judge_cost_usd for j in judged)
    return {
        "rungs_recorded": rungs_recorded,
        "rungs_current": rungs_current,
        "judge": judge_cost,
        "labels_recorded": label_recorded,
        "labels_current": label_current,
        "total_recorded": rungs_recorded + judge_cost + label_recorded,
        "total_current": rungs_current + judge_cost + label_current,
        "n_eval_questions": len(load_eval()),
        "n_answers": len(answers),
        "n_judgments": len(judged),
        "n_labels": sum(len(rows) for rows in labels.values()),
        "this_version_api_spend": 0.0,
    }
