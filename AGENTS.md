# AGENTS.md: router-distill

Research code for distilling an LLM router's tier choices into a tiny classifier and testing whether the best solver is the best routing teacher. Read README.md for the question and results, data/README.md for the data, and REPORT.md for every number.

## Rules

- Python 3.12 and uv. `uv sync` installs; run scripts as `uv run scripts/<name>.py` or through the Makefile.
- `make lint test check` must pass before a commit. `make check` retrains the students and fails if REPORT.md or results/metrics.json changes, so commit regenerated outputs together with the code that changed them.
- REPORT.md and results/*.json are generated. Edit `scripts/report.py` or the analysis modules, never the outputs.
- All model calls go through `complete()` in `src/router_distill/llm.py`. Scripts that call the API append each row as it returns, skip existing rows on re-run, and take `--limit`.
- API spend needs the owner's approval first. Estimate it from the per-call costs in REPORT.md section 8 and say it before running. The offline analysis costs nothing.
- `data/questions.jsonl` is the held-out eval set: never train on it, never regenerate it mechanically. `data/questions_train.jsonl` is train-only; add any new eval leak to `data/train_exclusions.json` rather than deleting rows.
- Never rewrite committed data rows. New samples, judges or judge passes are new rows keyed by `sample`, `judge` and `judge_pass`.
- Keep `models/` out of git: the students rebuild in seconds and pickles are unsafe to load from untrusted sources.
- The API key lives in `.env` (gitignored) or the environment. Never print or commit it.
- Commits: conventional prefix, short imperative subject, no trailers.

## Open work

See "Next steps" in README.md. The most informative cheap step is a test-retest re-judge (about $1.50 with the Batch API, $3 without), which the scripts already support with `--judge-pass 1`; the Batch API itself is not wired in yet.
