import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from .config import TIERS

C_GRID = (1.0, 3.0, 10.0, 30.0, 100.0)
CV_FOLDS = 5
SEED = 0
MAX_ITER = 10000
# Converge well past the default (1e-4) so platform floating-point differences cannot flip a prediction.
TOL = 1e-8
# Independent fits run in parallel; capped so the rebuild leaves cores for other work on the machine.
N_JOBS = min(4, os.cpu_count() or 1)
EPS = 1e-6


def make_vectorizers() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=20000)
    return word, char


def featurize(
    vecs: tuple[TfidfVectorizer, TfidfVectorizer], texts: list[str], fit: bool = False
) -> csr_matrix:
    word, char = vecs
    if fit:
        return hstack([word.fit_transform(texts), char.fit_transform(texts)]).tocsr()
    return hstack([word.transform(texts), char.transform(texts)]).tocsr()


def fit_softmax(features: csr_matrix, targets: np.ndarray, c: float) -> LogisticRegression:
    """Multinomial logistic regression on soft targets: each row is repeated once per tier with the
    target probability as its sample weight, which is exactly cross-entropy against the targets."""
    n = features.shape[0]
    stacked = vstack([features] * len(TIERS)).tocsr()
    labels = np.repeat(np.arange(len(TIERS)), n)
    weights = targets.T.reshape(-1)
    keep = weights > 0
    model = LogisticRegression(C=c, max_iter=MAX_ITER, tol=TOL)
    model.fit(stacked[keep], labels[keep], sample_weight=weights[keep])
    return model


def softmax_proba(model: LogisticRegression, features: csr_matrix) -> np.ndarray:
    out = np.zeros((features.shape[0], len(TIERS)))
    out[:, model.classes_] = model.predict_proba(features)
    return out


def adjust_prior(probs: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Post-hoc label-shift correction: reweight each tier by target/source prior and renormalize."""
    adjusted = probs * (np.asarray(target) + EPS) / (np.asarray(source) + EPS)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


class TierStudent:
    """TF-IDF word 1-2-grams + char 3-5-grams into multinomial logistic regression. SFT trains on the
    teacher's argmax as a one-hot target, distillation on the teacher's full tier distribution; the
    target is the only difference between the two."""

    def __init__(self, c: float = 10.0):
        self.c = c
        self.vecs = make_vectorizers()
        self.model: LogisticRegression | None = None
        self.train_prior: np.ndarray | None = None

    def fit(self, texts: list[str], targets: np.ndarray) -> "TierStudent":
        self.model = fit_softmax(featurize(self.vecs, texts, fit=True), targets, self.c)
        self.train_prior = targets.mean(axis=0)
        return self

    def predict_proba(self, texts: list[str], target_prior: np.ndarray | None = None) -> np.ndarray:
        probs = softmax_proba(self.model, featurize(self.vecs, texts))
        if target_prior is None:
            return probs
        return adjust_prior(probs, self.train_prior, target_prior)

    def predict(self, texts: list[str], target_prior: np.ndarray | None = None) -> list[str]:
        return [TIERS[i] for i in self.predict_proba(texts, target_prior).argmax(axis=1)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "TierStudent":
        # Only load pickles this repo wrote locally; models/ is gitignored for that reason.
        with path.open("rb") as f:
            return pickle.load(f)


@dataclass
class Fold:
    train: np.ndarray
    test: np.ndarray
    x_train: csr_matrix
    x_test: csr_matrix


def make_folds(texts: list[str], n_folds: int = CV_FOLDS, seed: int = SEED) -> list[Fold]:
    """Featurize each CV fold once so every teacher, target and C reuses it."""
    folds = []
    for train, test in KFold(n_folds, shuffle=True, random_state=seed).split(texts):
        vecs = make_vectorizers()
        x_train = featurize(vecs, [texts[i] for i in train], fit=True)
        x_test = featurize(vecs, [texts[i] for i in test])
        folds.append(Fold(train, test, x_train, x_test))
    return folds


def _fold_proba(fold: Fold, targets: np.ndarray, c: float) -> np.ndarray:
    return softmax_proba(fit_softmax(fold.x_train, targets[fold.train], c), fold.x_test)


def out_of_fold_grid(
    folds: list[Fold], targets: np.ndarray, grid: tuple[float, ...]
) -> dict[float, np.ndarray]:
    """Out-of-fold probabilities for every C, fitting the len(grid) x len(folds) models in parallel."""
    jobs = [(c, fold) for c in grid for fold in folds]
    parts = Parallel(n_jobs=N_JOBS)(delayed(_fold_proba)(fold, targets, c) for c, fold in jobs)
    out = {c: np.zeros_like(targets, dtype=float) for c in grid}
    for (c, fold), probs in zip(jobs, parts, strict=True):
        out[c][fold.test] = probs
    return out


def cross_entropy(targets: np.ndarray, probs: np.ndarray) -> float:
    return float(-(targets * np.log(np.clip(probs, EPS, 1.0))).sum(axis=1).mean())


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Mean KL(p || q) over rows, in nats."""
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    return float((p * np.log(p / q)).sum(axis=1).mean())


def tune_c(folds: list[Fold], targets: np.ndarray, grid: tuple[float, ...] = C_GRID) -> tuple[float, dict]:
    """Pick C by out-of-fold cross-entropy against the student's own training target."""
    oof = out_of_fold_grid(folds, targets, grid)
    table = {c: {"cross_entropy": cross_entropy(targets, oof[c]), "oof": oof[c]} for c in grid}
    best = min(table, key=lambda c: table[c]["cross_entropy"])
    return best, table


def one_hot_from_indices(indices: np.ndarray) -> np.ndarray:
    out = np.zeros((len(indices), len(TIERS)))
    out[np.arange(len(indices)), indices] = 1.0
    return out


def one_hot(tiers: list[str]) -> np.ndarray:
    return one_hot_from_indices(np.array([TIERS.index(t) for t in tiers]))


def soft_targets(prob_dicts: list[dict[str, float]]) -> np.ndarray:
    return np.array([[p[t] for t in TIERS] for p in prob_dicts])
