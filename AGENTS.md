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

Updated: 2026-08-23 by session -50 — EXPERIMENT COMPLETE, see REPORT.md

- Full pipeline ran: 108-question outcome grid (324 answers + judged), 3 teachers labeled
  eval + 475-question synthetic train pool, 6 students trained. Total spend ~$15.
- Headline: every student BEATS its own teacher as a live router (sft:opus regret 0.213 vs
  teacher:opus 0.556). Best-solver-as-best-teacher partially holds: opus students best (0.213),
  haiku close (0.259), sonnet — the MID tier — worst (0.481). Distilled (soft-label) never beat
  SFT — consistent small negative. Judge scores compress 8-10, so always-deep nearly matches
  oracle on regret while always-thoughtful dominates score-per-dollar 806 vs 172 — the regret
  metric and the efficiency metric pick opposite winners; say which you're optimizing.
- Mid-run fixes (committed): rung max_tokens 4k→16k (adaptive thinking truncated 21 deep-rung
  answers, biasing the oracle against opus), judge extract_json tolerates python-dict output.
- DEFERRED (Tarun's no-new-spend directive): OpenAI rung/teacher, small-LM student via -3e's
  GKD stack, kind-prediction head, real Bonsai-traffic outcome signal, judge-ensemble to
  de-compress scores.
