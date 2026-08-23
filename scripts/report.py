"""Evaluate every router against the outcome grid and write REPORT.md.

Routers compared: fixed-tier baselines, heuristic, each teacher (as a live router),
and each teacher's SFT + distilled students. Regret is against the per-question oracle.

Usage: uv run scripts/report.py
"""

from pathlib import Path

from router_distill import heuristic
from router_distill.config import TEACHERS, TIERS
from router_distill.regret import OutcomeGrid, PolicyEval
from router_distill.schema import JudgedAnswer, Question, RungAnswer, TeacherLabel, read_jsonl
from router_distill.students import TierStudent

ROOT = Path(__file__).resolve().parents[1]


def teacher_policy(teacher: str, ids: list[str]) -> dict[str, str] | None:
    labels = [TeacherLabel(**row) for row in read_jsonl(ROOT / f"data/labels/questions.{teacher}.jsonl")]
    if not labels:
        return None
    by_id = {l.question_id: l.tier for l in labels}
    return {qid: by_id[qid] for qid in ids if qid in by_id}


def student_policy(kind: str, teacher: str, questions: dict[str, Question], ids: list[str]) -> dict[str, str] | None:
    path = ROOT / f"models/{kind}.{teacher}.pkl"
    if not path.exists():
        return None
    student = TierStudent.load(path)
    preds = student.predict([questions[qid].text for qid in ids])
    return dict(zip(ids, preds))


def render(evals: list[PolicyEval], grid: OutcomeGrid) -> str:
    lines = [
        "# Router regret report",
        "",
        f"Scored questions: {len(grid.complete_ids)} (all 3 rungs answered and judged).",
        "Regret = oracle mean score − policy mean score. Oracle picks the best-scoring rung per question, cheapest on ties.",
        "",
        "| router | mean score | regret | mean cost/q | score per $ | latency ms | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in sorted(evals, key=lambda e: e.regret):
        lines.append(
            f"| {e.name} | {e.mean_score} | {e.regret} | ${e.mean_cost_usd:.4f} | {e.score_per_dollar} | {int(e.mean_latency_ms)} | {e.n} |"
        )

    slices = sorted({s for e in evals for s in e.by_slice})
    lines += ["", "## Regret by slice", "", "| router | " + " | ".join(slices) + " |", "|---" * (len(slices) + 1) + "|"]
    for e in sorted(evals, key=lambda e: e.regret):
        lines.append("| " + e.name + " | " + " | ".join(str(e.by_slice.get(s, "—")) for s in slices) + " |")

    lines += [
        "",
        "## Reading it",
        "",
        "- `always-deep` is the quality ceiling money can buy; its regret is what judge noise plus occasional small-model wins look like.",
        "- The interesting rows are ambiguous/ood slice regret: clean slices are easy for everyone.",
        "- Teacher rows are live-routing cost-free here (label cost not included in cost/q); student rows route for ~free at inference.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    questions = {q.id: q for q in read_jsonl(ROOT / "data/questions.jsonl", Question)}
    rungs = [RungAnswer(**row) for row in read_jsonl(ROOT / "data/outcomes/answers.jsonl")]
    judged = [JudgedAnswer(**row) for row in read_jsonl(ROOT / "data/outcomes/judged.jsonl")]
    grid = OutcomeGrid(list(questions.values()), rungs, judged)
    ids = grid.complete_ids
    if not ids:
        raise SystemExit("outcome grid is empty — run run_rungs.py and judge_answers.py first")

    evals = []
    for tier in TIERS:
        evals.append(grid.evaluate(f"always-{tier}", {qid: tier for qid in ids}))
    evals.append(grid.evaluate("oracle", {qid: grid.oracle_tier(qid) for qid in ids}))
    evals.append(grid.evaluate("heuristic", {qid: heuristic.route(questions[qid].text)["tier"] for qid in ids}))

    for teacher in TEACHERS:
        policy = teacher_policy(teacher, ids)
        if policy:
            evals.append(grid.evaluate(f"teacher:{teacher}", policy))
        for kind in ("sft", "distilled"):
            policy = student_policy(kind, teacher, questions, ids)
            if policy:
                evals.append(grid.evaluate(f"{kind}:{teacher}", policy))

    report = render(evals, grid)
    (ROOT / "REPORT.md").write_text(report)
    print(report)
    print(f"\nwrote {ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
