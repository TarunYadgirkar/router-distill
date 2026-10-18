"""Optional: does a stronger text encoder close the imitation gap? Offline, no API calls.

Embeds every question with all-MiniLM-L6-v2 (a 22M-parameter sentence encoder read from the local
Hugging Face cache; nothing is downloaded when it is already cached), trains the same logistic-regression
students on the embeddings instead of TF-IDF, and writes results/embedding_students.json, which
scripts/report.py folds into REPORT.md.

Usage: uv run --group embed scripts/embed_students.py   (or `make embed`)
"""

import json
from importlib.metadata import version

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold

from router_distill.analysis import (
    TARGETS,
    load_inputs,
    rounded,
    target_prior,
    teacher_choice,
    teacher_probs,
)
from router_distill.config import SHORT, TEACHERS
from router_distill.data import ROOT
from router_distill.encoder import ENCODER, embed
from router_distill.stats import Policy, bootstrap_indices, summarize
from router_distill.students import (
    C_GRID,
    CV_FOLDS,
    SEED,
    adjust_prior,
    cross_entropy,
    fit_softmax,
    kl_divergence,
    one_hot,
    soft_targets,
    softmax_proba,
)

OUT = ROOT / "results/embedding_students.json"


def cv_probs(x: csr_matrix, y: np.ndarray, c: float) -> np.ndarray:
    out = np.zeros_like(y)
    for train, test in KFold(CV_FOLDS, shuffle=True, random_state=SEED).split(np.arange(x.shape[0])):
        out[test] = softmax_proba(fit_softmax(x[train], y[train], c), x[test])
    return out


def main() -> None:
    inp = load_inputs()
    grid = inp.grid
    pool_x = csr_matrix(embed([q.text for q in inp.pool]))
    eval_x = csr_matrix(embed(grid.texts))
    idx = bootstrap_indices(grid.n)
    rows = []
    for teacher in TEACHERS:
        labels = inp.train_labels[teacher]
        hard = one_hot([label.tier for label in labels])
        soft = soft_targets([label.tier_probs for label in labels])
        eval_labels = inp.eval_labels[teacher]
        for target in TARGETS:
            y = hard if target == "sft" else soft
            table = {c: cv_probs(pool_x, y, c) for c in C_GRID}
            best = min(table, key=lambda c: cross_entropy(y, table[c]))
            probs = softmax_proba(fit_softmax(pool_x, y, best), eval_x)
            for adjusted in (False, True):
                p = (
                    adjust_prior(probs, y.mean(axis=0), target_prior(target, eval_labels))
                    if adjusted
                    else probs
                )
                name = f"embed-{target}{'+prior' if adjusted else ''}:{SHORT[teacher]}"
                summary = summarize(grid, Policy(name, "student", p.argmax(axis=1), np.zeros(grid.n), p), idx)
                summary.update(
                    {
                        "c": best,
                        "cv_agreement": float((table[best].argmax(axis=1) == hard.argmax(axis=1)).mean()),
                        "cv_kl": kl_divergence(soft, table[best]),
                        "eval_agreement": float((p.argmax(axis=1) == teacher_choice(eval_labels)).mean()),
                        "eval_kl": kl_divergence(teacher_probs(eval_labels), p),
                    }
                )
                rows.append(summary)
                print(
                    f"  {name:28s} C={best:g} CV agree {summary['cv_agreement']:.2f}  eval agree "
                    f"{summary['eval_agreement']:.2f}  gain {summary['gain']:+.2f}"
                )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "encoder": ENCODER,
        "torch": version("torch"),
        "transformers": version("transformers"),
        "device": "cpu",
        "students": rows,
    }

    OUT.write_text(json.dumps(rounded(result), indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
