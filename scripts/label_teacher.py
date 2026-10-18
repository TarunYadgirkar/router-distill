"""Label questions with a teacher router's soft tier distribution. Paid API calls.

Usage: uv run scripts/label_teacher.py --teacher claude-opus-5 [--questions data/questions.jsonl] [--limit N]

Labels are appended to data/labels/<questions-stem>.<teacher>.jsonl as they return; re-running skips
questions that already have a label.
"""

import argparse
import sys
from pathlib import Path

from router_distill.config import TEACHERS
from router_distill.data import labels_path
from router_distill.runner import run_parallel
from router_distill.schema import Question, TeacherLabel, append_jsonl, drop_truncated_tail, read_jsonl
from router_distill.teacher import label

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, choices=TEACHERS)
    parser.add_argument("--questions", default=str(ROOT / "data/questions.jsonl"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    questions = read_jsonl(Path(args.questions), Question)[: args.limit]
    out = Path(args.out) if args.out else labels_path(Path(args.questions).stem, args.teacher)
    drop_truncated_tail(out)
    done_ids = {row.question_id for row in read_jsonl(out, TeacherLabel)}
    todo = [q for q in questions if q.id not in done_ids]
    print(f"{len(done_ids)} already labeled, {len(todo)} to go -> {out}")

    spent = 0.0

    def save(q: Question, result: TeacherLabel) -> None:
        nonlocal spent
        append_jsonl(out, result)
        spent += result.cost_usd
        print(f"  {q.id}: {result.tier} {result.tier_probs} - {result.reason} (run ${spent:.4f})")

    failures = run_parallel(
        todo, lambda q: label(q, args.teacher), save, args.workers, describe=lambda q: q.id
    )
    print(f"done: {len(todo) - failures} new labels, {failures} failures, ${spent:.4f} this run -> {out}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
