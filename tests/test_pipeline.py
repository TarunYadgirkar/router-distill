"""The paid scripts' bookkeeping, exercised with a fake model call: nothing here touches the API."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from router_distill import teacher
from router_distill.llm import CallResult, extract_json
from router_distill.schema import Question, TeacherLabel, drop_truncated_tail, read_jsonl, write_jsonl

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "text",
    [
        '{"score": 9, "punt": false}',
        'Here you go:\n```json\n{"score": 9, "punt": false}\n```',
        '{"score": 9, "punt": false} (note: {braces} in trailing prose)',
        "{'score': 9, 'punt': False}",
    ],
)
def test_extract_json_variants(text):
    assert extract_json(text) == {"score": 9, "punt": False}


def test_extract_json_rejects_text_without_object():
    with pytest.raises(ValueError):
        extract_json("no object here")


def test_teacher_tie_escalates_to_more_capable_tier():
    label = TeacherLabel("q", "t", "other", {"quick": 0.4, "thoughtful": 0.2, "deep": 0.4}, "", 0.0)
    assert label.tier == "deep"


def test_drop_truncated_tail(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n{"a": 2')
    drop_truncated_tail(path)
    assert path.read_text() == '{"a": 1}\n'
    path.write_text('{"a": 1}')
    drop_truncated_tail(path)
    assert path.read_text() == '{"a": 1}\n'


def fake_label_call(fail_ids: set[str]):
    def complete(model, system, user, max_tokens, effort=None):
        if any(qid in user for qid in fail_ids):
            raise RuntimeError("simulated API failure")
        text = json.dumps({"kind": "lookup", "tier_probs": {"quick": 0.7, "thoughtful": 0.2, "deep": 0.1}})
        return CallResult(
            text, 10, 5, 1, 0.001, "end_turn", "msg_test", max_tokens, "2026-01-01T00:00:00+00:00"
        )

    return complete


def test_label_teacher_appends_as_it_goes_and_resumes(tmp_path, monkeypatch):
    questions = [Question(f"q{i}", f"question q{i}", "clean-quick") for i in range(5)]
    qpath, out = tmp_path / "questions.jsonl", tmp_path / "labels.jsonl"
    write_jsonl(qpath, questions)
    script = load_script("label_teacher")
    argv = ["label_teacher", "--teacher", "claude-haiku-4-5", "--questions", str(qpath), "--out", str(out)]

    monkeypatch.setattr(teacher, "complete", fake_label_call({"q3"}))
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        script.main()
    first = read_jsonl(out, TeacherLabel)
    assert sorted(r.question_id for r in first) == ["q0", "q1", "q2", "q4"]
    assert first[0].response_id == "msg_test" and first[0].stop_reason == "end_turn"

    calls = []
    real = fake_label_call(set())

    def counting(*args, **kwargs):
        calls.append(args[2])
        return real(*args, **kwargs)

    monkeypatch.setattr(teacher, "complete", counting)
    script.main()
    assert calls == ["Question: question q3"]
    assert len(read_jsonl(out, TeacherLabel)) == 5


def test_limit_restricts_work(tmp_path, monkeypatch):
    questions = [Question(f"q{i}", f"question q{i}", "clean-quick") for i in range(5)]
    qpath, out = tmp_path / "questions.jsonl", tmp_path / "labels.jsonl"
    write_jsonl(qpath, questions)
    monkeypatch.setattr(teacher, "complete", fake_label_call(set()))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "label_teacher",
            "--teacher",
            "claude-haiku-4-5",
            "--questions",
            str(qpath),
            "--out",
            str(out),
            "--limit",
            "2",
        ],
    )
    load_script("label_teacher").main()
    assert [r.question_id for r in read_jsonl(out, TeacherLabel)] == ["q0", "q1"]


def test_expand_questions_mints_ids_past_the_highest_existing_id():
    script = load_script("expand_questions")
    pool = [Question("syn-0001", "a", "train-quick"), Question("syn-0483", "b", "train-quick")]
    assert script.next_id(pool) == 484
    assert script.next_id([]) == 1


def test_rungs_and_judge_passes_key_by_sample_and_pass(tmp_path, monkeypatch):
    from router_distill import judge, rungs
    from router_distill.schema import JudgedAnswer, RungAnswer

    questions = [Question(f"q{i}", f"question q{i}", "clean-quick") for i in range(2)]
    qpath, answers, judged = tmp_path / "q.jsonl", tmp_path / "answers.jsonl", tmp_path / "judged.jsonl"
    write_jsonl(qpath, questions)

    def answer_call(model, system, user, max_tokens, effort=None):
        return CallResult(
            "an answer", 10, 20, 1, 0.002, "end_turn", "msg_a", max_tokens, "2026-01-01T00:00:00+00:00"
        )

    def judge_call(model, system, user, max_tokens, effort=None):
        text = '{"score": 8, "punt": false, "rationale": "fine"}'
        return CallResult(text, 10, 5, 1, 0.001, "end_turn", "msg_j", max_tokens, "2026-01-01T00:00:00+00:00")

    monkeypatch.setattr(rungs, "complete", answer_call)
    monkeypatch.setattr(judge, "complete", judge_call)
    run, grade = load_script("run_rungs"), load_script("judge_answers")
    base = ["--questions", str(qpath)]
    for argv in (["run_rungs", *base, "--out", str(answers)], ["run_rungs", *base, "--out", str(answers)]):
        monkeypatch.setattr(sys, "argv", argv)
        run.main()
    rows = read_jsonl(answers, RungAnswer)
    assert len(rows) == 6 and rows[0].max_tokens == 16000 and rows[0].stop_reason == "end_turn"

    for judge_pass in ("0", "1", "1"):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "judge_answers",
                *base,
                "--answers",
                str(answers),
                "--out",
                str(judged),
                "--judge-pass",
                judge_pass,
            ],
        )
        grade.main()
    graded = read_jsonl(judged, JudgedAnswer)
    assert len(graded) == 12
    assert sorted({j.judge_pass for j in graded}) == [0, 1]
