"""Compare a fresh offline rebuild with the committed outputs. Run after `make train report`.

Usage: uv run scripts/check_reproduction.py

Everything must match the committed results/metrics.json to 1e-4, except the training-set bootstrap:
its 300 refits on resampled data are not bit-identical across platforms (macOS arm64 vs Linux x86-64
wheels), so its percentiles may move by up to TRAIN_BOOT_TOL and its win counts by TRAIN_BOOT_COUNT_TOL.
Both limits sit just above the noise seen between the two platforms (up to 0.007 on a percentile, one
win); one question flipping in one refit moves an agreement percentile by at most 1/108. REPORT.md and
README.md's generated blocks must match line for line, except that inside the training-bootstrap table
each printed number may differ by one unit in its last printed digit.
"""

import json
import re
import subprocess
import sys

from router_distill.data import ROOT

README_BLOCKS = ("headline", "results", "tuned")

DEFAULT_TOL = 1e-4
TRAIN_BOOT_TOL = 0.01
TRAIN_BOOT_COUNT_TOL = 1
BOOT_SECTION = ("Spread over", "### Student regularisation")
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


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
    worst = {"exact": 0.0, "training_bootstrap": 0.0, "win counts": 0.0}
    for path in old.keys() & new.keys():
        a, b = old[path], new[path]
        if isinstance(a, int | float) and isinstance(b, int | float) and not isinstance(a, bool):
            delta = abs(a - b)
            section = "exact"
            if path[0] == "training_bootstrap":
                section = "win counts" if isinstance(a, int) else "training_bootstrap"
            worst[section] = max(worst[section], delta)
            if delta > tolerance(path, a):
                problems.append(f"metrics: {'/'.join(map(str, path))} {a} -> {b}")
        elif a != b:
            problems.append(f"metrics: {'/'.join(map(str, path))} {a!r} -> {b!r}")
    print(
        f"max deviation: {worst['exact']:.2g} outside the training bootstrap, {worst['training_bootstrap']:.2g} on "
        f"its percentiles, {worst['win counts']:.0f} on its win counts"
    )
    return problems


def in_boot_section(lines: list[str]) -> list[bool]:
    flags, inside = [], False
    for line in lines:
        if line.startswith(BOOT_SECTION[0]):
            inside = True
        elif line.startswith(BOOT_SECTION[1]):
            inside = False
        flags.append(inside)
    return flags


def nearly_equal(a: str, b: str) -> bool:
    """Same text, and every number within one unit of its last printed digit."""
    na, nb = NUMBER.findall(a), NUMBER.findall(b)
    if NUMBER.sub("#", a) != NUMBER.sub("#", b) or len(na) != len(nb):
        return False
    for x, y in zip(na, nb, strict=True):
        decimals = len(x.split(".")[1]) if "." in x else 0
        if abs(float(x) - float(y)) > 10**-decimals + 1e-9:
            return False
    return True


def generated_part(path: str, text: str) -> list[str]:
    if path != "README.md":
        return text.splitlines()
    lines = []
    for name in README_BLOCKS:
        start, end = f"<!-- {name}:start", f"<!-- {name}:end -->"
        lines += text[text.index(start) : text.index(end)].splitlines()
    return lines


def compare_text(path: str) -> list[str]:
    old = generated_part(path, committed(path))
    new = generated_part(path, (ROOT / path).read_text())
    if len(old) != len(new):
        return [f"{path}: line count {len(old)} -> {len(new)}"]
    return [
        f"{path}: {a!r} -> {b!r}"
        for a, b, boot in zip(old, new, in_boot_section(old), strict=True)
        if a != b and not (boot and nearly_equal(a, b))
    ]


def main() -> None:
    problems = compare_metrics() + compare_text("REPORT.md") + compare_text("README.md")
    for line in problems[:40]:
        print(line, file=sys.stderr)
    if problems:
        sys.exit(f"rebuild does not match the committed outputs ({len(problems)} differences)")
    print("rebuild matches the committed outputs")


if __name__ == "__main__":
    main()
