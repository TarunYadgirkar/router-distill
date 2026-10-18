"""Score rung answers with a judge model. Paid API calls.

Usage: uv run scripts/judge_answers.py [--judge claude-opus-5] [--judge-pass P] [--sample K] [--limit N]

Rows are keyed by (question, tier, sample, judge, judge_pass) and appended as they return. The original
run is sample 0, judge claude-opus-5, pass 0; --judge-pass 1 re-judges the same answers for a
test-retest check, and --judge picks a second judge, without touching existing rows.
"""

import argparse
import sys
from pathlib import Path

from router_distill.config import JUDGE_MODEL, MODELS
from router_distill.data import ANSWERS_PATH, JUDGED_PATH
from router_distill.judge import judge
from router_distill.runner import run_parallel
from router_distill.schema import (
    JudgedAnswer,
    Question,
    RungAnswer,
    append_jsonl,
    drop_truncated_tail,
    read_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(ROOT / "data/questions.jsonl"))
    parser.add_argument("--answers", default=str(ANSWERS_PATH))
    parser.add_argument("--out", default=str(JUDGED_PATH))
    parser.add_argument("--judge", default=JUDGE_MODEL, choices=sorted(MODELS))
    parser.add_argument("--judge-pass", type=int, default=0)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=None, help="judge answers for the first N questions only"
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    out = Path(args.out)
    questions = read_jsonl(Path(args.questions), Question)[: args.limit]
    by_id = {q.id: q for q in questions}
    answers = [a for a in read_jsonl(Path(args.answers), RungAnswer) if a.sample == args.sample]
    drop_truncated_tail(out)
    done = {j.key for j in read_jsonl(out, JudgedAnswer)}
    todo = [
        a for a in answers if a.question_id in by_id and (*a.key, args.judge, args.judge_pass) not in done
    ]
    print(
        f"{len(done)} judgments on disk, {len(todo)} to judge with {args.judge} pass {args.judge_pass} -> {out}"
    )

    spent = 0.0

    def save(a: RungAnswer, j: JudgedAnswer) -> None:
        nonlocal spent
        append_jsonl(out, j)
        spent += j.judge_cost_usd
        print(
            f"  {a.question_id} @ {a.tier}: {j.score}{' PUNT' if j.punt else ''} - {j.rationale} (run ${spent:.2f})"
        )

    failures = run_parallel(
        todo,
        lambda a: judge(by_id[a.question_id].text, a, judge_model=args.judge, judge_pass=args.judge_pass),
        save,
        args.workers,
        describe=lambda a: f"{a.question_id} @ {a.tier}",
    )
    print(f"done: {len(todo) - failures} new judgments, {failures} failures, ${spent:.2f} this run")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
