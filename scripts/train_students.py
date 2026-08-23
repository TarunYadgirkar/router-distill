"""Train SFT and distilled students from one teacher's labels over the synthetic pool.

Usage: uv run scripts/train_students.py --teacher claude-opus-5
Expects data/labels/questions_train.<teacher>.jsonl (run label_teacher.py with
--questions data/questions_train.jsonl first).
"""

import argparse
from pathlib import Path

from router_distill.config import TEACHERS
from router_distill.schema import Question, TeacherLabel, read_jsonl
from router_distill.students import TierStudent, one_hot, soft_targets

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, choices=TEACHERS)
    args = parser.parse_args()

    labels_path = ROOT / f"data/labels/questions_train.{args.teacher}.jsonl"
    labels = [TeacherLabel(**row) for row in read_jsonl(labels_path)]
    if not labels:
        raise SystemExit(f"no labels at {labels_path} — run label_teacher.py on the training pool first")

    questions = {q.id: q for q in read_jsonl(ROOT / "data/questions_train.jsonl", Question)}
    pairs = [(questions[l.question_id].text, l) for l in labels if l.question_id in questions]
    texts = [t for t, _ in pairs]
    print(f"training on {len(pairs)} labeled questions from {args.teacher}")

    sft = TierStudent()
    sft.fit(texts, one_hot([l.tier for _, l in pairs]))
    sft_path = ROOT / f"models/sft.{args.teacher}.pkl"
    sft.save(sft_path)

    distilled = TierStudent()
    distilled.fit(texts, soft_targets([l.tier_probs for _, l in pairs]))
    distilled_path = ROOT / f"models/distilled.{args.teacher}.pkl"
    distilled.save(distilled_path)

    train_preds_sft = sft.predict(texts)
    train_preds_dis = distilled.predict(texts)
    agree_sft = sum(p == l.tier for p, (_, l) in zip(train_preds_sft, pairs)) / len(pairs)
    agree_dis = sum(p == l.tier for p, (_, l) in zip(train_preds_dis, pairs)) / len(pairs)
    print(f"train agreement with teacher argmax — sft: {agree_sft:.3f}, distilled: {agree_dis:.3f}")
    print(f"saved {sft_path} and {distilled_path}")


if __name__ == "__main__":
    main()
