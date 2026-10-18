"""Builds every router, scores it against the outcome grid and collects the numbers REPORT.md prints.
Offline: reads the committed data and the students trained by scripts/train_students.py."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from . import heuristic
from .config import PRICE_SOURCE, PRICES_AS_OF, RECORDED_PRICES, SHORT, TEACHERS, TIERS, reprice_recorded
from .data import ROOT, load_answers, load_eval, load_exclusions, load_judged, load_labels, load_train_pool
from .diagnostics import judge_diagnostics, power, sensitivity, spend
from .grid import OutcomeGrid, build_grid
from .schema import Question, TeacherLabel
from .stats import (
    GAP_TARGETS,
    N_BOOT,
    SEED,
    Policy,
    bootstrap_indices,
    compare,
    curve,
    gap_target,
    holm,
    point_gain,
    split_cost_to_reach,
    summarize,
)
from .students import (
    TierStudent,
    adjust_prior,
    featurize,
    fit_softmax,
    kl_divergence,
    make_folds,
    make_vectorizers,
    one_hot,
    one_hot_from_indices,
    soft_targets,
    softmax_proba,
    tune_c,
)

MODELS_DIR = ROOT / "models"
TARGETS = ("sft", "distilled")
TRAIN_BOOT = 50
OUTCOME_CV_FOLDS = 10
OUTCOME_C = 10.0


def rounded(value, digits: int = 6):
    """Round every float in a JSON-able structure so committed results diff cleanly."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [rounded(v, digits) for v in value]
    return value


@dataclass
class Inputs:
    grid: OutcomeGrid
    eval_labels: dict[str, list[TeacherLabel]]  # aligned with grid.questions
    pool: list[Question]
    train_labels: dict[str, list[TeacherLabel]]  # aligned with pool
    students: dict[tuple[str, str], TierStudent]


def load_inputs(models_dir: Path = MODELS_DIR) -> Inputs:
    grid = build_grid(load_eval(), load_answers(), load_judged())
    eval_labels, train_labels, students = {}, {}, {}
    pool = load_train_pool()
    for teacher in TEACHERS:
        by_id = load_labels("questions", teacher)
        eval_labels[teacher] = [by_id[qid] for qid in grid.ids]
        train_by_id = load_labels("questions_train", teacher)
        train_labels[teacher] = [train_by_id[q.id] for q in pool]
        for target in TARGETS:
            path = models_dir / f"{target}.{teacher}.pkl"
            if not path.exists():
                raise SystemExit(f"missing {path}: run `make train` (scripts/train_students.py) first")
            students[(target, teacher)] = TierStudent.load(path)
    return Inputs(grid, eval_labels, pool, train_labels, students)


def teacher_probs(labels: list[TeacherLabel]) -> np.ndarray:
    return soft_targets([label.tier_probs for label in labels])


def teacher_choice(labels: list[TeacherLabel]) -> np.ndarray:
    return np.array([TIERS.index(label.tier) for label in labels])


def target_prior(target: str, labels: list[TeacherLabel]) -> np.ndarray:
    """The teacher's tier distribution on the eval questions, in the form the student was trained on."""
    if target == "sft":
        return np.bincount(teacher_choice(labels), minlength=3) / len(labels)
    return teacher_probs(labels).mean(axis=0)


def student_name(target: str, teacher: str, adjusted: bool) -> str:
    return f"{target}{'+prior' if adjusted else ''}:{SHORT[teacher]}"


def build_policies(inp: Inputs) -> list[Policy]:
    grid, zeros = inp.grid, np.zeros(inp.grid.n)
    policies = [Policy(f"always-{t}", "baseline", np.full(grid.n, i), zeros) for i, t in enumerate(TIERS)]
    policies.append(Policy("oracle", "reference", grid.oracle_choice(), zeros))
    heur = np.array([TIERS.index(heuristic.classify_tier(t)) for t in grid.texts])
    policies.append(Policy("heuristic", "baseline", heur, zeros))
    for teacher in TEACHERS:
        labels = inp.eval_labels[teacher]
        route_cost = np.array([reprice_recorded(teacher, label.cost_usd) for label in labels])
        policies.append(
            Policy(
                f"teacher:{SHORT[teacher]}",
                "teacher",
                teacher_choice(labels),
                route_cost,
                teacher_probs(labels),
            )
        )
    for adjusted in (False, True):
        for teacher in TEACHERS:
            for target in TARGETS:
                student = inp.students[(target, teacher)]
                prior = target_prior(target, inp.eval_labels[teacher]) if adjusted else None
                probs = student.predict_proba(grid.texts, prior)
                meta = {"teacher": teacher, "target": target, "adjusted": adjusted, "c": student.c}
                policies.append(
                    Policy(
                        student_name(target, teacher, adjusted),
                        "student",
                        probs.argmax(axis=1),
                        zeros,
                        probs,
                        meta,
                    )
                )
    policies.append(generator_label_policy(inp))
    policies.append(outcome_cv_policy(grid))
    return policies


def generator_label_policy(inp: Inputs) -> Policy:
    """A student trained on the tier the question generator was asked for: no teacher, $0 labels."""
    texts = [q.text for q in inp.pool]
    y = one_hot([q.slice.removeprefix("train-") for q in inp.pool])
    c, _ = tune_c(make_folds(texts), y)
    probs = TierStudent(c=c).fit(texts, y).predict_proba(inp.grid.texts)
    return Policy("generator-label", "baseline", probs.argmax(axis=1), np.zeros(inp.grid.n), probs, {"c": c})


def outcome_cv_policy(grid: OutcomeGrid) -> Policy:
    """Trained on the eval set's own oracle tiers, scored out-of-fold: what outcome supervision learns."""
    y = one_hot_from_indices(grid.oracle_choice())
    probs = np.zeros((grid.n, 3))
    for train, test in KFold(OUTCOME_CV_FOLDS, shuffle=True, random_state=SEED).split(grid.texts):
        vecs = make_vectorizers()
        x_train = featurize(vecs, [grid.texts[i] for i in train], fit=True)
        model = fit_softmax(x_train, y[train], OUTCOME_C)
        probs[test] = softmax_proba(model, featurize(vecs, [grid.texts[i] for i in test]))
    return Policy("outcome-cv", "baseline", probs.argmax(axis=1), np.zeros(grid.n), probs, {"c": OUTCOME_C})


def fidelity(inp: Inputs, policies: dict[str, Policy]) -> dict:
    rows = []
    for adjusted in (False, True):
        for teacher in TEACHERS:
            labels = inp.eval_labels[teacher]
            for target in TARGETS:
                p = policies[student_name(target, teacher, adjusted)]
                cv = inp.students[(target, teacher)].cv
                rows.append(
                    {
                        "name": p.name,
                        "eval_agreement": float((p.choice == teacher_choice(labels)).mean()),
                        "eval_kl": kl_divergence(teacher_probs(labels), p.probs),
                        "deep_share": float((p.choice == 2).mean()),
                        "cv_agreement": cv["oof_agreement"],
                        "cv_kl": cv["oof_kl"],
                        "c": cv["c"],
                    }
                )
    teachers = {}
    for teacher in TEACHERS:
        eval_choice = teacher_choice(inp.eval_labels[teacher])
        train_choice = teacher_choice(inp.train_labels[teacher])
        teachers[SHORT[teacher]] = {
            "eval_prior": (np.bincount(eval_choice, minlength=3) / len(eval_choice)).tolist(),
            "train_prior": (np.bincount(train_choice, minlength=3) / len(train_choice)).tolist(),
            "majority_agreement": float(np.bincount(eval_choice, minlength=3).max() / len(eval_choice)),
            "agreement": {
                SHORT[other]: float((eval_choice == teacher_choice(inp.eval_labels[other])).mean())
                for other in TEACHERS
            },
            "generator_agreement": float(
                (
                    train_choice == np.array([TIERS.index(q.slice.removeprefix("train-")) for q in inp.pool])
                ).mean()
            ),
        }
    return {"students": rows, "teachers": teachers}


def comparison_family(inp: Inputs, policies: dict[str, Policy], idx: np.ndarray) -> list[dict]:
    """The pre-registered set of teacher comparisons (Q2), Holm-adjusted together."""
    pairs = []
    for i, a in enumerate(TEACHERS):
        for b in TEACHERS[i + 1 :]:
            pairs.append((f"teacher:{SHORT[a]}", f"teacher:{SHORT[b]}"))
            for target in TARGETS:
                pairs.append((student_name(target, a, False), student_name(target, b, False)))
    for teacher in TEACHERS:
        for target in TARGETS:
            pairs.append((student_name(target, teacher, False), f"teacher:{SHORT[teacher]}"))
        pairs.append((student_name("distilled", teacher, False), student_name("sft", teacher, False)))
    rows = [compare(inp.grid, policies[a], policies[b], idx) for a, b in pairs]
    for row, adjusted in zip(rows, holm([r["p"] for r in rows]), strict=True):
        row["p_holm"] = adjusted
    return rows


def training_bootstrap(inp: Inputs, n_boot: int = TRAIN_BOOT) -> dict:
    """Refit every student on bootstrap resamples of the training pool (same resamples for every
    teacher) to show how much of each result is training-set luck."""
    grid = inp.grid
    rng = np.random.default_rng(SEED)
    texts = [q.text for q in inp.pool]
    records: dict[str, list[dict]] = {}
    targets = {
        (target, teacher): one_hot([lab.tier for lab in inp.train_labels[teacher]])
        if target == "sft"
        else soft_targets([lab.tier_probs for lab in inp.train_labels[teacher]])
        for teacher in TEACHERS
        for target in TARGETS
    }
    for _ in range(n_boot):
        rows = rng.integers(0, len(texts), len(texts))
        vecs = make_vectorizers()
        x_train = featurize(vecs, [texts[i] for i in rows], fit=True)
        x_eval = featurize(vecs, grid.texts)
        for (target, teacher), full in targets.items():
            labels = inp.eval_labels[teacher]
            y = full[rows]
            probs = softmax_proba(fit_softmax(x_train, y, inp.students[(target, teacher)].c), x_eval)
            for adjusted in (False, True):
                p = adjust_prior(probs, y.mean(axis=0), target_prior(target, labels)) if adjusted else probs
                choice = p.argmax(axis=1)
                score = grid.pick(grid.score, choice).mean()
                cost = grid.pick(grid.cost, choice).mean()
                records.setdefault(student_name(target, teacher, adjusted), []).append(
                    {
                        "agreement": float((choice == teacher_choice(labels)).mean()),
                        "deep_share": float((choice == 2).mean()),
                        "score": float(score),
                        "gain": point_gain(grid, score, cost),
                    }
                )
    out = {}
    for name, recs in records.items():
        out[name] = {
            key: [float(np.percentile([r[key] for r in recs], q)) for q in (2.5, 50, 97.5)]
            for key in ("agreement", "deep_share", "score", "gain")
        }
    best_counts = {}
    for adjusted in (False, True):
        for target in TARGETS:
            names = [student_name(target, t, adjusted) for t in TEACHERS]
            gains = np.array([[r["gain"] for r in records[n]] for n in names])
            winners = np.bincount(gains.argmax(axis=0), minlength=len(TEACHERS))
            best_counts[f"{target}{'+prior' if adjusted else ''}"] = {
                SHORT[t]: int(w) for t, w in zip(TEACHERS, winners, strict=True)
            }
    return {"n_boot": n_boot, "students": out, "best_teacher_counts": best_counts}


def run(models_dir: Path = MODELS_DIR, n_boot: int = N_BOOT, train_boot: int = TRAIN_BOOT) -> dict:
    inp = load_inputs(models_dir)
    grid = inp.grid
    idx = bootstrap_indices(grid.n, n_boot)
    policies = build_policies(inp)
    by_name = {p.name: p for p in policies}
    summaries = [summarize(grid, p, idx) for p in policies]
    curves = {p.name: curve(grid, p, idx) for p in policies if p.probs is not None}
    comparisons = comparison_family(inp, by_name, idx)
    return {
        "meta": {
            "n_eval": grid.n,
            "n_pool": len(inp.pool),
            "n_excluded": len(load_exclusions()),
            "prices_as_of": PRICES_AS_OF,
            "price_source": PRICE_SOURCE,
            "recorded_prices": RECORDED_PRICES,
            "n_boot": n_boot,
            "seed": SEED,
        },
        "fixed": {
            t: {"cost": float(grid.cost[:, i].mean()), "score": float(grid.score[:, i].mean())}
            for i, t in enumerate(TIERS)
        },
        "gap_targets": {str(f): gap_target(grid, f) for f in GAP_TARGETS},
        "split_cost_to_reach": {str(f): split_cost_to_reach(grid, f) for f in GAP_TARGETS},
        "policies": summaries,
        "curves": curves,
        "fidelity": fidelity(inp, by_name),
        "comparisons": comparisons,
        "training_bootstrap": training_bootstrap(inp, train_boot),
        "diagnostics": judge_diagnostics(grid),
        "power": power(grid, by_name, comparisons),
        "sensitivity": sensitivity(inp, policies),
        "spend": spend(),
    }
