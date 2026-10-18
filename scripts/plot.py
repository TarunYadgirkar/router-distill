"""Draw assets/generated/frontier.png from results/metrics.json. Offline.

Usage: uv run scripts/plot.py   (after scripts/report.py; `make reproduce` runs both)
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from router_distill.data import ROOT  # noqa: E402

RESULTS = ROOT / "results/metrics.json"
OUT = ROOT / "assets/generated/frontier.png"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
NEUTRAL = "#9a9993"
# Categorical slots 1-3 of the reference palette (validated all-pairs); shape is the secondary channel.
STYLE = {
    "teacher": ("#2a78d6", "o", "LLM teacher (routing call included)"),
    "student": ("#eb6834", "s", "TF-IDF student"),
    "student+prior": ("#1baf7a", "D", "TF-IDF student, prior-corrected"),
    "baseline": (NEUTRAL, "^", "heuristic / generator-label / outcome-cv"),
}


def group_of(p: dict) -> str:
    if p["name"].startswith("always-"):
        return "fixed"
    if p["group"] == "student":
        return "student+prior" if "+prior" in p["name"] else "student"
    return p["group"]


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def frontier(ax, res: dict) -> None:
    fixed = res["fixed"]
    xs = [fixed[t]["cost"] for t in ("quick", "thoughtful", "deep")]
    ys = [fixed[t]["score"] for t in ("quick", "thoughtful", "deep")]
    ax.plot(xs, ys, color=INK, linewidth=2, solid_capstyle="round", zorder=2)
    ax.scatter(xs, ys, s=64, color=INK, edgecolor=SURFACE, linewidth=2, zorder=4)
    offsets = {"always-quick": (8, -4), "always-thoughtful": (8, -22), "always-deep": (8, -4)}
    for x, y, t in zip(xs, ys, ("always-quick", "always-thoughtful", "always-deep"), strict=True):
        ax.annotate(
            t, (x, y), xytext=offsets[t], textcoords="offset points", fontsize=9, color=INK, ha="left"
        )
    ax.annotate(
        "random split between two fixed rungs\n(expected score at each cost)",
        (0.006, 9.15),
        fontsize=9,
        color=INK_2,
        ha="left",
    )
    for p in res["policies"]:
        group = group_of(p)
        if group not in STYLE:
            continue
        color, marker, _ = STYLE[group]
        ax.scatter(
            p["cost"],
            p["score"],
            s=56,
            color=color,
            marker=marker,
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=5,
        )
    oracle = next(p for p in res["policies"] if p["name"] == "oracle")
    ax.scatter(
        oracle["cost"],
        oracle["score"],
        s=110,
        marker="*",
        color=INK,
        edgecolor=SURFACE,
        linewidth=1,
        zorder=6,
    )
    ax.annotate(
        "oracle (needs every answer\nand its score)",
        (oracle["cost"], oracle["score"]),
        xytext=(-150, 4),
        textcoords="offset points",
        fontsize=9,
        color=INK,
    )
    ax.set_xlabel("mean cost per question (USD)", color=INK_2, fontsize=10)
    ax.set_ylabel("mean judge score (0-10)", color=INK_2, fontsize=10)
    ax.set_xlim(0, 0.068)
    ax.set_title("Score vs cost", loc="left", color=INK, fontsize=12, fontweight="bold")


def forest(ax, res: dict) -> None:
    rows = [p for p in res["policies"] if group_of(p) in STYLE or p["name"] == "oracle"]
    for i, p in enumerate(rows):
        y = len(rows) - 1 - i
        if p["name"] == "oracle":
            color, marker = INK, "*"
        else:
            color, marker, _ = STYLE[group_of(p)]
        lo, hi = p["gain_ci"]
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=3)
        ax.scatter(
            p["gain"],
            y,
            s=56 if marker != "*" else 110,
            color=color,
            marker=marker,
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=4,
        )
    ax.axvline(0, color=INK, linewidth=1, zorder=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([p["name"] for p in reversed(rows)], fontsize=8.5, color=INK)
    ax.set_xlabel(
        "gain vs random split at the same cost (judge points, 95% CI);\nfixed rungs are 0 by construction",
        color=INK_2,
        fontsize=10,
    )
    ax.set_title(
        "Gain over a cost-matched random split", loc="left", color=INK, fontsize=12, fontweight="bold"
    )
    ax.grid(axis="y", visible=False)


def main() -> None:
    res = json.loads(RESULTS.read_text())
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (left, right):
        style_axes(ax)
    frontier(left, res)
    forest(right, res)
    handles = [
        Line2D([], [], color=c, marker=m, linestyle="", markersize=8, markeredgecolor=SURFACE, label=label)
        for c, m, label in STYLE.values()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=INK)
    n = res["meta"]["n_eval"]
    fig.suptitle(
        f"No learned router beats a random split between fixed rungs ({n} eval questions, Claude Opus 5 judge)",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, facecolor=SURFACE)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
