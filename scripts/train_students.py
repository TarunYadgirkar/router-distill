"""Train the SFT and distilled students for each teacher on the synthetic pool. Offline, no API calls.

Usage: uv run scripts/train_students.py [--teacher claude-opus-5 ...]   (default: all teachers)

For each (teacher, target) the regularisation strength C is chosen by 5-fold cross-validation on the
pool (out-of-fold cross-entropy against the target), then the student is refit on the whole pool and
saved to models/<target>.<teacher>.pkl together with its cross-validation results.
"""

import argparse

import numpy as np

from router_distill.config import TEACHERS
from router_distill.data import ROOT, load_labels, load_train_pool
from router_distill.schema import TeacherLabel
from router_distill.students import (
    TierStudent,
    kl_divergence,
    make_folds,
    one_hot,
    soft_targets,
    tune_c,
)

MODELS_DIR = ROOT / "models"
TARGETS = ("sft", "distilled")


def student_path(target: str, teacher: str):
    return MODELS_DIR / f"{target}.{teacher}.pkl"


def targets_for(target: str, labels: list[TeacherLabel]) -> np.ndarray:
    if target == "sft":
        return one_hot([label.tier for label in labels])
    return soft_targets([label.tier_probs for label in labels])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", nargs="*", choices=TEACHERS, default=TEACHERS)
    args = parser.parse_args()

    pool = load_train_pool()
    texts = [q.text for q in pool]
    folds = make_folds(texts)
    print(f"training pool: {len(pool)} questions (leak exclusions applied), {len(folds)} CV folds")

    for teacher in args.teacher:
        by_id = load_labels("questions_train", teacher)
        missing = [q.id for q in pool if q.id not in by_id]
        if missing:
            raise SystemExit(
                f"{teacher}: {len(missing)} pool questions lack labels (e.g. {missing[:3]}); run label_teacher.py"
            )
        labels = [by_id[q.id] for q in pool]
        hard = one_hot([label.tier for label in labels])
        soft = soft_targets([label.tier_probs for label in labels])
        for target in TARGETS:
            y = targets_for(target, labels)
            best_c, table = tune_c(folds, y)
            oof = table[best_c]["oof"]
            student = TierStudent(c=best_c).fit(texts, y)
            student.cv = {
                "c_grid": {str(c): round(v["cross_entropy"], 4) for c, v in table.items()},
                "c": best_c,
                "oof_agreement": float((oof.argmax(axis=1) == hard.argmax(axis=1)).mean()),
                "oof_kl": kl_divergence(soft, oof),
                "train_agreement": float(
                    (student.predict_proba(texts).argmax(axis=1) == hard.argmax(axis=1)).mean()
                ),
            }
            student.save(student_path(target, teacher))
            cv = student.cv
            print(
                f"  {target}:{teacher} C={best_c:g}  CV agreement {cv['oof_agreement']:.3f}  "
                f"CV KL {cv['oof_kl']:.3f}  train agreement {cv['train_agreement']:.3f}"
            )


if __name__ == "__main__":
    main()
