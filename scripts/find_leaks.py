"""List training questions that may copy or paraphrase an eval question, for manual review. Offline.

Usage: uv run --group embed scripts/find_leaks.py [--top 30]   (or `make leaks`)

Two signals per training question, each against its nearest eval question: character 3-5-gram TF-IDF
cosine (catches copies and small rewordings) and all-MiniLM-L6-v2 embedding cosine (catches paraphrases
that share few characters). Neither decides anything: a person reads the list and adds real leaks to
data/train_exclusions.json. Rows already excluded are marked.
"""

import argparse

import numpy as np

from router_distill.data import load_eval, load_exclusions, load_train_pool, similarity
from router_distill.encoder import embed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=30, help="pairs to print per signal")
    args = parser.parse_args()

    evals = load_eval()
    pool = load_train_pool(include_excluded=True)
    excluded = load_exclusions()
    eval_texts, pool_texts = [q.text for q in evals], [q.text for q in pool]
    signals = {
        "character n-gram cosine": similarity(pool_texts, eval_texts),
        "embedding cosine": embed(pool_texts) @ embed(eval_texts).T,
    }
    for name, sims in signals.items():
        nearest = sims.argmax(axis=1)
        best = sims[np.arange(len(pool)), nearest]
        print(f"## top {args.top} by {name}")
        for i in np.argsort(-best)[: args.top]:
            mark = " [excluded]" if pool[i].id in excluded else ""
            q = evals[nearest[i]]
            print(f"{best[i]:.2f} {pool[i].id}{mark}: {pool_texts[i][:110]!r}")
            print(f"     {q.id}: {q.text[:110]!r}")
        print()


if __name__ == "__main__":
    main()
