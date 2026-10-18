.PHONY: install test lint train report figure reproduce check embed leaks

install:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check src scripts tests
	uv run ruff format --check src scripts tests

# Offline: retrain the students and rebuild REPORT.md, results/metrics.json, the README results block and the figure (~2 min on 4 cores, no API calls).
reproduce: train report figure

train:
	uv run scripts/train_students.py

report:
	uv run scripts/report.py

figure:
	uv run scripts/plot.py

# CI: fail if a clean rebuild does not reproduce the committed outputs (see scripts/check_reproduction.py
# for the one tolerance it allows). Run on a clean checkout: it compares against HEAD.
check: train report
	uv run scripts/check_reproduction.py

# Optional encoder students; needs torch + transformers and the all-MiniLM-L6-v2 weights in the HF cache.
embed:
	uv run --group embed scripts/embed_students.py

# List training questions that may copy or paraphrase an eval question (character n-grams and the same
# encoder); a person reads the list and adds real leaks to data/train_exclusions.json.
leaks:
	uv run --group embed scripts/find_leaks.py
