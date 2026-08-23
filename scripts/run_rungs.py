"""Answer every eval question at every rung — the outcome grid's raw material.

Usage: uv run scripts/run_rungs.py [--limit N] [--workers 4]
Resumable: already-answered (question, tier) pairs are skipped.
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from router_distill.config import TIERS
from router_distill.rungs import answer
from router_distill.schema import Question, RungAnswer, read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/outcomes/answers.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(ROOT / "data/questions.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    questions = read_jsonl(Path(args.questions), Question)
    if args.limit:
        questions = questions[: args.limit]

    existing = [RungAnswer(**row) for row in read_jsonl(OUT)]
    done = {(r.question_id, r.tier) for r in existing}
    todo = [(q, t) for q in questions for t in TIERS if (q.id, t) not in done]
    print(f"{len(done)} rung answers exist, {len(todo)} to run")

    results = list(existing)
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(answer, q, t): (q, t) for q, t in todo}
        for future in as_completed(futures):
            q, t = futures[future]
            try:
                r = future.result()
                results.append(r)
                print(f"  {q.id} @ {t}: {r.output_tokens} tok, ${r.cost_usd:.4f}, {r.latency_ms}ms")
            except Exception as e:
                failures += 1
                print(f"  {q.id} @ {t}: FAILED — {e}", file=sys.stderr)

    write_jsonl(OUT, results)
    print(f"done: {len(results)} answers, {failures} failures, ${sum(r.cost_usd for r in results):.2f} total")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
