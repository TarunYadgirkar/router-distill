"""Answer every eval question at every rung: the outcome grid's raw material. Paid API calls.

Usage: uv run scripts/run_rungs.py [--limit N] [--sample K] [--workers 4]

Each answer is appended to data/outcomes/answers.jsonl as soon as it returns, so an interrupted run
keeps what it paid for. Re-running skips (question, tier, sample) rows that already exist.
--sample K draws an independent K-th answer per cell (sample 0 is the original run).
"""

import argparse
import sys
from pathlib import Path

from router_distill.config import TIERS
from router_distill.data import ANSWERS_PATH
from router_distill.rungs import answer
from router_distill.runner import run_parallel
from router_distill.schema import Question, RungAnswer, append_jsonl, drop_truncated_tail, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(ROOT / "data/questions.jsonl"))
    parser.add_argument("--out", default=str(ANSWERS_PATH))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    out = Path(args.out)
    questions = read_jsonl(Path(args.questions), Question)[: args.limit]
    drop_truncated_tail(out)
    done = {r.key for r in read_jsonl(out, RungAnswer)}
    todo = [(q, t) for q in questions for t in TIERS if (q.id, t, args.sample) not in done]
    print(f"{len(done)} rung answers on disk, {len(todo)} to run (sample {args.sample}) -> {out}")

    spent = 0.0

    def save(item: tuple[Question, str], r: RungAnswer) -> None:
        nonlocal spent
        append_jsonl(out, r)
        spent += r.cost_usd
        flag = " TRUNCATED" if r.stop_reason == "max_tokens" else ""
        print(
            f"  {r.question_id} @ {r.tier}: {r.output_tokens} tok, ${r.cost_usd:.4f}{flag} (run ${spent:.2f})"
        )

    failures = run_parallel(
        todo,
        lambda item: answer(item[0], item[1], sample=args.sample),
        save,
        args.workers,
        describe=lambda item: f"{item[0].id} @ {item[1]}",
    )
    print(f"done: {len(todo) - failures} new answers, {failures} failures, ${spent:.2f} this run")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
