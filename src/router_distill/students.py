import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import TIERS


class TierStudent:
    """TF-IDF features + softmax regression. SFT trains on one-hot argmax labels; distilled
    trains on the teacher's full tier distribution — the only difference between the two rows
    of the comparison table, by design."""

    def __init__(self, l2: float = 1e-3, lr: float = 0.5, epochs: int = 300):
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=20000)
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        self.weights: np.ndarray | None = None

    def _featurize(self, texts: list[str], fit: bool = False):
        if fit:
            word = self.word_vec.fit_transform(texts)
            char = self.char_vec.fit_transform(texts)
        else:
            word = self.word_vec.transform(texts)
            char = self.char_vec.transform(texts)
        return hstack([word, char]).tocsr()

    def fit(self, texts: list[str], targets: np.ndarray) -> None:
        """targets: (n, 3) rows summing to 1 — one-hot for SFT, soft for distillation."""
        features = self._featurize(texts, fit=True)
        n, d = features.shape
        self.weights = np.zeros((d + 1, len(TIERS)))
        for _ in range(self.epochs):
            probs = self._forward(features)
            grad_logits = (probs - targets) / n
            grad_w = features.T @ grad_logits + self.l2 * self.weights[:-1]
            grad_b = grad_logits.sum(axis=0)
            self.weights[:-1] -= self.lr * grad_w
            self.weights[-1] -= self.lr * grad_b

    def _forward(self, features) -> np.ndarray:
        logits = features @ self.weights[:-1] + self.weights[-1]
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self._forward(self._featurize(texts))

    def predict(self, texts: list[str]) -> list[str]:
        probs = self.predict_proba(texts)
        return [TIERS[i] for i in probs.argmax(axis=1)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "TierStudent":
        with path.open("rb") as f:
            return pickle.load(f)


def one_hot(tiers: list[str]) -> np.ndarray:
    out = np.zeros((len(tiers), len(TIERS)))
    for i, tier in enumerate(tiers):
        out[i, TIERS.index(tier)] = 1.0
    return out


def soft_targets(prob_dicts: list[dict[str, float]]) -> np.ndarray:
    return np.array([[p[t] for t in TIERS] for p in prob_dicts])
