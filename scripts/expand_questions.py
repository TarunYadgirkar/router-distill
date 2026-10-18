"""Generate the synthetic training pool (students train on these; the curated set stays the held-out,
outcome-scored eval set). Paid API calls.

Usage: uv run scripts/expand_questions.py --n 500 [--limit BATCHES] [--workers 4]

New questions are appended to data/questions_train.jsonl batch by batch. A generated question is
dropped if it matches an eval question or an existing training question (same normalized text, or
character n-gram cosine >= SAME_QUESTION_SIM); closer-than-usual matches to eval questions are printed
for manual review and can be added to data/train_exclusions.json.
"""

import argparse
import sys

from router_distill.config import GENERATOR_MAX_TOKENS, GENERATOR_MODEL, KINDS, TIERS
from router_distill.data import REVIEW_SIM, TRAIN_PATH, is_same_question, load_eval, similarity
from router_distill.llm import complete, extract_json
from router_distill.runner import run_parallel
from router_distill.schema import Question, append_jsonl, drop_truncated_tail, read_jsonl

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
    result = complete(GENERATOR_MODEL, SYSTEM, user, max_tokens=GENERATOR_MAX_TOKENS, effort="medium")
    parsed = extract_json(result.text)
    return [q.strip() for q in parsed["questions"] if isinstance(q, str) and q.strip()]


def next_id(questions: list[Question]) -> int:
    nums = [int(q.id.split("-")[1]) for q in questions if q.id.startswith("syn-")]
    return max(nums, default=0) + 1


def plan_batches(existing: list[Question], n: int) -> list[tuple[str, str, int, int]]:
    combos = [(k, t) for k in KINDS for t in TIERS]
    per_combo = max(1, n // len(combos))
    batches = []
    for kind, tier in combos:
        have = [q for q in existing if q.kind_hint == kind and q.slice == f"train-{tier}"]
        need = per_combo - len(have)
        idx = len(have) // BATCH
        while need > 0:
            take = min(BATCH, need)
            batches.append((kind, tier, take, idx))
            need -= take
            idx += 1
    return batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None, help="run at most this many batches")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    drop_truncated_tail(TRAIN_PATH)
    pool = read_jsonl(TRAIN_PATH, Question)
    eval_texts = [q.text for q in load_eval()]
    batches = plan_batches(pool, args.n)[: args.limit]
    print(f"{len(pool)} existing, generating ~{sum(b[2] for b in batches)} across {len(batches)} batches")

    counter = next_id(pool)
    kept = dropped = 0

    def save(batch: tuple[str, str, int, int], texts: list[str]) -> None:
        nonlocal counter, kept, dropped
        kind, tier, _, _ = batch
        references = eval_texts + [q.text for q in pool]
        sims = similarity(texts, references)
        for text, row in zip(texts, sims, strict=True):
            if is_same_question(text, references, row):
                dropped += 1
                print(f"    dropped duplicate: {text[:80]!r}")
                continue
            review = row[: len(eval_texts)].max(initial=0.0)
            if review >= REVIEW_SIM:
                close = eval_texts[int(row[: len(eval_texts)].argmax())]
                print(f"    review (sim {review:.2f} to eval {close[:60]!r}): {text[:80]!r}")
            question = Question(f"syn-{counter:04d}", text, f"train-{tier}", kind, "synthetic")
            counter += 1
            append_jsonl(TRAIN_PATH, question)
            pool.append(question)
            kept += 1
        print(f"  {kind}/{tier}: {len(pool)} in pool")

    failures = run_parallel(
        batches, lambda b: generate(*b), save, args.workers, describe=lambda b: f"{b[0]}/{b[1]}"
    )
    print(f"done: {kept} added, {dropped} duplicates dropped, {failures} batch failures -> {TRAIN_PATH}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
