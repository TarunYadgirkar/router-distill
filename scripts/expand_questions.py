"""Generate a synthetic training pool of questions (students train on these; the curated set
stays as the held-out, outcome-scored test set).

Usage: uv run scripts/expand_questions.py --n 500
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from router_distill.config import GENERATOR_MODEL, KINDS, TIERS
from router_distill.llm import complete
from router_distill.schema import Question, read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/questions_train.jsonl"
BATCH = 10

SYSTEM = (
    "You generate realistic user questions for training an inference router. Given a target "
    "question kind and difficulty tier, produce diverse, natural questions a real user might ask "
    "an AI assistant. Vary domain, length, and phrasing aggressively — include tech, science, "
    "business, hobbies, law, cooking, sports, arts. Do not reuse sentence templates.\n"
    "Tiers: quick = single-fact lookup or trivial transform; thoughtful = reasoning/synthesis "
    "over several facts; deep = proofs, multi-constraint design, high-stakes tradeoffs.\n"
    'Respond with JSON only: {"questions": ["...", ...]}'
)


def generate(kind: str, tier: str, n: int, batch_idx: int) -> list[str]:
    user = (
        f"Generate {n} questions of kind '{kind}' at tier '{tier}'. Batch {batch_idx} — make these "
        f"distinct from anything an earlier batch would obviously produce."
    )
    result = complete(GENERATOR_MODEL, SYSTEM, user, max_tokens=4000, effort="medium")
    start = result.text.find("{")
    parsed = json.loads(result.text[start : result.text.rfind("}") + 1])
    return [q for q in parsed["questions"] if isinstance(q, str) and q.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    existing = read_jsonl(OUT, Question)
    combos = [(k, t) for k in KINDS for t in TIERS]
    per_combo = max(1, args.n // len(combos))
    batches = []
    for kind, tier in combos:
        have = sum(1 for q in existing if q.kind_hint == kind and q.slice == f"train-{tier}")
        need = per_combo - have
        idx = 0
        while need > 0:
            take = min(BATCH, need)
            batches.append((kind, tier, take, idx))
            need -= take
            idx += 1

    print(f"{len(existing)} existing, generating ~{sum(b[2] for b in batches)} across {len(batches)} batches")
    questions = list(existing)
    counter = len(existing)
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(generate, k, t, n, i): (k, t) for k, t, n, i in batches}
        for future in as_completed(futures):
            kind, tier = futures[future]
            try:
                for text in future.result():
                    counter += 1
                    questions.append(
                        Question(
                            id=f"syn-{counter:04d}",
                            text=text.strip(),
                            slice=f"train-{tier}",
                            kind_hint=kind,
                            source="synthetic",
                        )
                    )
                print(f"  {kind}/{tier}: ok ({counter} total)")
            except Exception as e:
                failures += 1
                print(f"  {kind}/{tier}: FAILED — {e}", file=sys.stderr)

    seen: set[str] = set()
    deduped = []
    for q in questions:
        key = q.text.lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(q)

    write_jsonl(OUT, deduped)
    print(f"done: {len(deduped)} training questions ({failures} batch failures) -> {OUT}")


if __name__ == "__main__":
    main()
