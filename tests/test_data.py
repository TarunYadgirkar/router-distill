"""Invariants of the committed data."""

import pytest

from router_distill import analysis
from router_distill.config import TEACHERS
from router_distill.data import (
    is_same_question,
    load_eval,
    load_exclusions,
    load_labels,
    load_train_pool,
    normalize,
    similarity,
)


def test_training_pool_has_no_copies_of_eval_questions():
    eval_texts = [q.text for q in load_eval()]
    pool = load_train_pool()
    sims = similarity([q.text for q in pool], eval_texts)
    leaks = [q.id for q, row in zip(pool, sims, strict=True) if is_same_question(q.text, eval_texts, row)]
    assert leaks == []


def test_exclusions_point_at_real_rows():
    eval_ids = {q.id for q in load_eval()}
    pool_ids = {q.id for q in load_train_pool(include_excluded=True)}
    for train_id, row in load_exclusions().items():
        assert train_id in pool_ids and row["eval_id"] in eval_ids


def test_every_pool_and_eval_question_has_a_label_from_every_teacher():
    pool = load_train_pool()
    evals = load_eval()
    for teacher in TEACHERS:
        assert {q.id for q in pool} <= set(load_labels("questions_train", teacher))
        assert {q.id for q in evals} <= set(load_labels("questions", teacher))


def test_ids_are_unique():
    for questions in (load_eval(), load_train_pool(include_excluded=True)):
        ids = [q.id for q in questions]
        assert len(ids) == len(set(ids))


def test_same_question_detection():
    refs = ["What is the chemical symbol for tungsten?"]
    assert normalize("What is the chemical symbol for TUNGSTEN") == normalize(refs[0])
    candidate = "What's the chemical symbol for tungsten?"
    assert is_same_question(candidate, refs, similarity([candidate], refs)[0])
    other = "Which is denser, osmium or tungsten?"
    assert not is_same_question(other, refs, similarity([other], refs)[0])


def test_report_refuses_to_run_without_trained_students(tmp_path):
    with pytest.raises(SystemExit, match="make train"):
        analysis.load_inputs(models_dir=tmp_path)
