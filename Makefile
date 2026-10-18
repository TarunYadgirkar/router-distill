.PHONY: install test lint train report figure reproduce check embed

install:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check src scripts tests
	uv run ruff format --check src scripts tests

# Offline: retrain the students and rebuild REPORT.md, results/metrics.json and the figure (~1.5 min, no API calls).
reproduce: train report figure

train:
	uv run scripts/train_students.py

report:
	uv run scripts/report.py

figure:
	uv run scripts/plot.py

# CI: fail if a clean rebuild does not reproduce the committed report byte for byte.
check: train report
	git diff --exit-code REPORT.md results/metrics.json

# Optional encoder students; needs torch + transformers and the all-MiniLM-L6-v2 weights in the HF cache.
embed:
	uv run --group embed scripts/embed_students.py
