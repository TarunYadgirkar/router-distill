import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .schema import JudgedAnswer, Question, RungAnswer, TeacherLabel, read_jsonl

ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH = ROOT / "data/questions.jsonl"
TRAIN_PATH = ROOT / "data/questions_train.jsonl"
EXCLUSIONS_PATH = ROOT / "data/train_exclusions.json"
ANSWERS_PATH = ROOT / "data/outcomes/answers.jsonl"
JUDGED_PATH = ROOT / "data/outcomes/judged.jsonl"
LABELS_DIR = ROOT / "data/labels"

# Character n-gram cosine at or above this is treated as the same question (catches contractions and
# small rewordings). Paraphrases score lower (0.4-0.5 here) and are only reported for review, because
# unrelated questions on the same topic score in that range too.
SAME_QUESTION_SIM = 0.8
REVIEW_SIM = 0.4


def labels_path(stem: str, teacher: str) -> Path:
    return LABELS_DIR / f"{stem}.{teacher}.jsonl"


def load_eval() -> list[Question]:
    return read_jsonl(EVAL_PATH, Question)


def load_exclusions() -> dict[str, dict]:
    if not EXCLUSIONS_PATH.exists():
        return {}
    return {row["train_id"]: row for row in json.loads(EXCLUSIONS_PATH.read_text())}


def load_train_pool(include_excluded: bool = False) -> list[Question]:
    pool = read_jsonl(TRAIN_PATH, Question)
    if include_excluded:
        return pool
    excluded = load_exclusions()
    return [q for q in pool if q.id not in excluded]


def load_labels(stem: str, teacher: str) -> dict[str, TeacherLabel]:
    return {label.question_id: label for label in read_jsonl(labels_path(stem, teacher), TeacherLabel)}


def load_answers() -> list[RungAnswer]:
    return read_jsonl(ANSWERS_PATH, RungAnswer)


def load_judged() -> list[JudgedAnswer]:
    return read_jsonl(JUDGED_PATH, JudgedAnswer)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def similarity(candidates: list[str], references: list[str]) -> np.ndarray:
    """(len(candidates), len(references)) char 3-5-gram TF-IDF cosine."""
    if not candidates or not references:
        return np.zeros((len(candidates), len(references)))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    vec.fit(candidates + references)
    a, b = vec.transform(candidates), vec.transform(references)
    return (a @ b.T).toarray()


def is_same_question(candidate: str, references: list[str], sims: np.ndarray) -> bool:
    norm = normalize(candidate)
    return any(normalize(r) == norm for r in references) or bool((sims >= SAME_QUESTION_SIM).any())
