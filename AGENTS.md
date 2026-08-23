# AGENTS.md — router-distill

Research project: distill routing decisions (which model tier + effort should answer a
question) into tiny students, and test whether the best solver is the best routing teacher.
Read README.md for the experiment design and pipeline order.

## Rules

- Python + uv. `uv sync` to install; run everything as `uv run scripts/<name>.py`.
- All model calls go through `complete()` in `src/router_distill/llm.py`; every script prints
  running cost and is resumable (existing JSONL rows are skipped).
- `data/questions.jsonl` (curated eval set) is hand-written, entity-diverse, and is the
  held-out test set — never train on it, never regenerate it mechanically.
- `data/questions_train.jsonl` is synthetic (expand_questions.py) — train-only.
- Agents cannot read or write `.env*` on this machine (hooks). Tarun pastes keys via
  `! touch .env && open -e .env`. Required: `ANTHROPIC_API_KEY` (OPENAI_API_KEY reserved for
  the cross-provider extension).
- Commit style: conventional prefix, short imperative subject, no trailers/footers.

## Lane coordination (2026-08-22)

- This repo is session -50's exclusive write lane (coordinator: tarunyadgirkar-a6).
- `~/TarunsCode/multi-teacher-kd` (-3e) does LLM-scale multi-teacher GKD via trl>=1.0
  `trl.experimental.gkd`; `~/TarunsCode/bonsai-distill` (-62) has a synthetic data engine at
  `src/bonsai_distill/datagen.py` (entity-split). Both read-only from here. This repo's
  students are deliberately sklearn-scale (routing must cost ~nothing at inference); if a
  small-LM student row is added later, mirror -3e's TRL/GKD setup instead of inventing one.

## Ongoing

Updated: 2026-08-22 by session -50 (initial scaffold)

- Scaffold complete: config/schema/llm/heuristic/teacher/rungs/judge/students/regret +
  5 pipeline scripts + curated 114-question eval set (24 quick / 24 thoughtful / 18 deep /
  24 ambiguous / 18 ood... see data/questions.jsonl), offline smoke passing.
- BLOCKED on Tarun pasting ANTHROPIC_API_KEY into .env, then: run_rungs -> judge_answers ->
  label_teacher x3 -> expand_questions -> label_teacher (train pool) x3 -> train_students x3
  -> report.
- Open extensions: OpenAI rung/teacher (add provider field to ModelSpec + dispatch in llm.py);
  small-LM student via -3e's GKD stack; kind-prediction head; Mahogany-style merged/corrected
  outcome signal once Bonsai traffic exists.
