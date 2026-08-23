"""Score every rung answer with the judge model.

Usage: uv run scripts/judge_answers.py [--workers 4]
Resumable: already-judged (question, tier) pairs are skipped.
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from router_distill.judge import judge
from router_distill.schema import JudgedAnswer, Question, RungAnswer, read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
ANSWERS = ROOT / "data/outcomes/answers.jsonl"
OUT = ROOT / "data/outcomes/judged.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(ROOT / "data/questions.jsonl"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    questions = {q.id: q for q in read_jsonl(Path(args.questions), Question)}
    answers = [RungAnswer(**row) for row in read_jsonl(ANSWERS)]
    existing = [JudgedAnswer(**row) for row in read_jsonl(OUT)]
    done = {(j.question_id, j.tier) for j in existing}
    todo = [a for a in answers if (a.question_id, a.tier) not in done and a.question_id in questions]
    print(f"{len(done)} judged, {len(todo)} to judge")

    results = list(existing)
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(judge, questions[a.question_id].text, a): a for a in todo}
        for future in as_completed(futures):
            a = futures[future]
            try:
                j = future.result()
                results.append(j)
                print(f"  {a.question_id} @ {a.tier}: {j.score}{' PUNT' if j.punt else ''} — {j.rationale}")
            except Exception as e:
                failures += 1
                print(f"  {a.question_id} @ {a.tier}: FAILED — {e}", file=sys.stderr)

    write_jsonl(OUT, results)
    print(f"done: {len(results)} judged, {failures} failures, ${sum(j.judge_cost_usd for j in results):.2f} judge cost")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
