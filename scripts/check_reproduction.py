"""Compare a fresh offline rebuild with the committed outputs. Run after `make train report`.

Usage: uv run scripts/check_reproduction.py

Everything must match the committed results/metrics.json to 1e-4, except the training-set bootstrap:
its 300 refits on resampled data are not bit-identical across BLAS builds (macOS Accelerate vs the
OpenBLAS in Linux wheels), so its percentiles may move by up to TRAIN_BOOT_TOL and its win counts by
TRAIN_BOOT_COUNT_TOL. REPORT.md and README.md's generated results block must match line for line outside
that table.
"""

import json
import subprocess
import sys

from router_distill.data import ROOT

README_START = "<!-- results:start"
README_END = "<!-- results:end -->"

DEFAULT_TOL = 1e-4
TRAIN_BOOT_TOL = 0.03
TRAIN_BOOT_COUNT_TOL = 5
BOOT_SECTION = ("Spread over", "## 5.")


def committed(path: str) -> str:
    return subprocess.run(
        ["git", "show", f"HEAD:{path}"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def leaves(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaves(child, (*path, key))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from leaves(child, (*path, i))
    else:
        yield path, value


def tolerance(path: tuple, value) -> float:
    if path and path[0] == "training_bootstrap":
        return TRAIN_BOOT_COUNT_TOL if isinstance(value, int) else TRAIN_BOOT_TOL
    return DEFAULT_TOL


def compare_metrics() -> list[str]:
    old = dict(leaves(json.loads(committed("results/metrics.json"))))
    new = dict(leaves(json.loads((ROOT / "results/metrics.json").read_text())))
    problems = [f"metrics: {'/'.join(map(str, p))} missing" for p in old.keys() ^ new.keys()]
    worst = {"exact": 0.0, "training_bootstrap": 0.0}
    for path in old.keys() & new.keys():
        a, b = old[path], new[path]
        if isinstance(a, int | float) and isinstance(b, int | float) and not isinstance(a, bool):
            delta = abs(a - b)
            section = "training_bootstrap" if path[0] == "training_bootstrap" else "exact"
            worst[section] = max(worst[section], delta)
            if delta > tolerance(path, a):
                problems.append(f"metrics: {'/'.join(map(str, path))} {a} -> {b}")
        elif a != b:
            problems.append(f"metrics: {'/'.join(map(str, path))} {a!r} -> {b!r}")
    print(
        f"max deviation: {worst['exact']:.2g} outside the training bootstrap, {worst['training_bootstrap']:.2g} inside"
    )
    return problems


def outside_boot_section(lines: list[str]) -> list[str]:
    out, inside = [], False
    for line in lines:
        if line.startswith(BOOT_SECTION[0]):
            inside = True
        elif line.startswith(BOOT_SECTION[1]):
            inside = False
        if not inside:
            out.append(line)
    return out


def generated_part(path: str, text: str) -> list[str]:
    if path == "README.md":
        text = text[text.index(README_START) : text.index(README_END)]
    return outside_boot_section(text.splitlines())


def compare_text(path: str) -> list[str]:
    old = generated_part(path, committed(path))
    new = generated_part(path, (ROOT / path).read_text())
    if old == new:
        return []
    diffs = [f"{path}: {a!r} -> {b!r}" for a, b in zip(old, new, strict=False) if a != b]
    return diffs or [f"{path}: line count {len(old)} -> {len(new)}"]


def main() -> None:
    problems = compare_metrics() + compare_text("REPORT.md") + compare_text("README.md")
    for line in problems[:40]:
        print(line, file=sys.stderr)
    if problems:
        sys.exit(f"rebuild does not match the committed outputs ({len(problems)} differences)")
    print("rebuild matches the committed outputs")


if __name__ == "__main__":
    main()
