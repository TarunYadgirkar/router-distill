"""Label questions with a teacher router's soft tier distribution.

Usage: uv run scripts/label_teacher.py --teacher claude-opus-5 [--questions data/questions.jsonl] [--limit N]
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from router_distill.config import TEACHERS
from router_distill.schema import Question, TeacherLabel, read_jsonl, write_jsonl
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

    questions = read_jsonl(Path(args.questions), Question)
    if args.limit:
        questions = questions[: args.limit]

    qfile_stem = Path(args.questions).stem
    out = Path(args.out) if args.out else ROOT / f"data/labels/{qfile_stem}.{args.teacher}.jsonl"
    done_ids = {row["question_id"] for row in read_jsonl(out)}
    todo = [q for q in questions if q.id not in done_ids]
    print(f"{len(done_ids)} already labeled, {len(todo)} to go -> {out}")

    labels: list[TeacherLabel] = [TeacherLabel(**row) for row in read_jsonl(out)]
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(label, q, args.teacher): q for q in todo}
        for future in as_completed(futures):
            q = futures[future]
            try:
                result = future.result()
                labels.append(result)
                print(f"  {q.id}: {result.tier} ({result.tier_probs}) — {result.reason}")
            except Exception as e:
                failures += 1
                print(f"  {q.id}: FAILED — {e}", file=sys.stderr)

    write_jsonl(out, labels)
    total_cost = sum(l.cost_usd for l in labels)
    print(f"done: {len(labels)} labels, {failures} failures, ${total_cost:.4f} total -> {out}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
