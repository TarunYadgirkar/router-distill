import numpy as np

from router_distill import heuristic
from router_distill.students import (
    C_GRID,
    TierStudent,
    adjust_prior,
    featurize,
    fixed_step_proba,
    fixed_step_softmax,
    kl_divergence,
    make_folds,
    make_vectorizers,
    one_hot,
    soft_targets,
    tune_c,
)

QUICK = [f"what is the capital of country {i}" for i in range(40)]
THOUGHTFUL = [f"explain how system {i} handles concurrent writes and why" for i in range(40)]
DEEP = [f"prove that algorithm {i} converges under adversarial multi constraint load" for i in range(40)]
TEXTS = QUICK + THOUGHTFUL + DEEP
TIERS = ["quick"] * 40 + ["thoughtful"] * 40 + ["deep"] * 40


def test_sft_student_learns_separable_tiers():
    student = TierStudent(c=10.0).fit(TEXTS, one_hot(TIERS))
    accuracy = np.mean([p == t for p, t in zip(student.predict(TEXTS), TIERS, strict=True)])
    assert accuracy > 0.95
    assert np.allclose(student.train_prior, [1 / 3] * 3)


def test_distilled_student_matches_constant_soft_target():
    target = {"quick": 0.6, "thoughtful": 0.3, "deep": 0.1}
    student = TierStudent(c=1.0).fit(TEXTS, soft_targets([target] * len(TEXTS)))
    probs = student.predict_proba(TEXTS[:5])
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(probs.mean(axis=0), [0.6, 0.3, 0.1], atol=0.02)


def test_fixed_step_trainer_learns_but_stays_softer_than_converged():
    x = featurize(make_vectorizers(), TEXTS, fit=True)
    probs = fixed_step_proba(fixed_step_softmax(x, one_hot(TIERS)), x)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs.argmax(axis=1) == one_hot(TIERS).argmax(axis=1)).mean() > 0.95
    converged = TierStudent(c=100.0).fit(TEXTS, one_hot(TIERS)).predict_proba(TEXTS)
    assert probs.max(axis=1).mean() < converged.max(axis=1).mean()


def test_adjust_prior_moves_mass_toward_target():
    probs = np.array([[0.2, 0.3, 0.5]])
    out = adjust_prior(probs, source=np.array([0.2, 0.2, 0.6]), target=np.array([0.3, 0.5, 0.2]))
    assert np.isclose(out.sum(), 1.0)
    assert out[0, 2] < probs[0, 2] and out[0, 1] > probs[0, 1]
    unchanged = adjust_prior(probs, source=np.array([0.3, 0.3, 0.4]), target=np.array([0.3, 0.3, 0.4]))
    assert np.allclose(unchanged, probs)


def test_tune_c_picks_from_grid_and_reports_each_value():
    folds = make_folds(TEXTS, n_folds=3)
    best, table = tune_c(folds, one_hot(TIERS))
    assert best in C_GRID and set(table) == set(C_GRID)
    assert all(row["oof"].shape == (len(TEXTS), 3) for row in table.values())


def test_kl_is_zero_for_identical_rows():
    p = np.array([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]])
    assert abs(kl_divergence(p, p)) < 1e-9
    assert kl_divergence(p, np.full_like(p, 1 / 3)) > 0


def test_heuristic_tiers():
    assert heuristic.route("What is the capital of France?")["tier"] == "quick"
    assert heuristic.route("Prove that comparison sorting is Omega(n log n).")["tier"] == "deep"
    assert heuristic.route("Explain how TCP handshakes work and why.")["tier"] == "thoughtful"
