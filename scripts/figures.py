"""Matplotlib figures for the multi-model comparison.

Usage: python scripts/figures.py results/ figures/
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, RULE = "#1f1f1f", "#6b6b6b", "#cfcdc8"
PASS, FIT, NOFILE = "#2a78d6", "#e8730e", "#a3a3a3"  # same outcome colours as the draft's Figure 4
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
})

summary = list(csv.DictReader(open(RES / "summary.csv")))
trials = list(csv.DictReader(open(RES / "trials.csv")))
models = [s["model"] for s in summary]


def style(ax, xmax=1.0, pct=True, pad=0.0):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-pad, xmax + pad)
    if pct:
        ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color="#ecebe8", linewidth=0.8)
    ax.set_axisbelow(True)


# --- Figure 5: mean with clustered SE, and the pass^3 to pass@3 range, per model -------------
fig, ax = plt.subplots(figsize=(7.6, 0.55 * len(models) + 1.5), dpi=200)
for i, s in enumerate(summary):
    n = int(s["n_tasks"])
    lo, hi = int(s["pass_hat_3"]) / n, int(s["pass_at_3"]) / n
    mean, se = float(s["mean"]), float(s["clustered_se"])
    ax.plot([lo, hi], [i, i], color=RULE, linewidth=6, solid_capstyle="round", zorder=1)
    ax.errorbar(mean, i, xerr=se, fmt="none", ecolor=INK, elinewidth=1.4, capsize=4, zorder=2)
    ax.scatter([lo], [i], s=70, marker="o", facecolor="white", edgecolor=PASS, linewidth=2, zorder=3)
    ax.scatter([hi], [i], s=70, marker="o", facecolor=PASS, edgecolor="white", linewidth=1.5, zorder=3)
    ax.scatter([mean], [i], s=80, marker="D", facecolor=INK, edgecolor="white", linewidth=1.5, zorder=4)
    ax.text(1.06, i, f"{s['raman_lab_passed']}/{s['raman_lab_rollouts']}", va="center", ha="left",
            fontsize=11, fontweight="bold", color=INK, transform=ax.get_yaxis_transform())
ax.set_yticks(range(len(models)), models)
ax.invert_yaxis()
style(ax, pad=0.03)
handles = [
    plt.Line2D([], [], marker="D", color="none", markerfacecolor=INK, markeredgecolor="white", markersize=8),
    plt.Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=PASS, markeredgewidth=2, markersize=8),
    plt.Line2D([], [], marker="o", color="none", markerfacecolor=PASS, markeredgecolor="white", markersize=8),
]
ax.legend(handles, ["Mean ± clustered SE", "pass^3 (every rollout passed)", "pass@3 (any rollout passed)"],
          ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), fontsize=9.5, handletextpad=0.3,
          columnspacing=1.2)
fig.suptitle("raman-lab: 8 tasks, 3 rollouts each, per model", x=0.01, ha="left", fontsize=12.5, y=0.99)
fig.tight_layout()
fig.savefig(OUT / "fig5_models_pass.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# --- Figure 6: outcome of every rollout, split by whether the model said "done" first ------
fig, axes = plt.subplots(1, 2, figsize=(9.2, 0.5 * len(models) + 1.6), dpi=200, sharey=True,
                         gridspec_kw={"wspace": 0.08})
cats = [("passed", "Passed", PASS, "white"), ("fit check failed", "Fit check failed", FIT, INK),
        ("no valid results file", "No results file", NOFILE, INK)]
groups = [("False", "Checked its output first"), ("True", 'Said "done" in its first response')]
counts = defaultdict(lambda: defaultdict(int))
for t in trials:
    if t["dataset"] == "raman-lab" and t["model"] in models:
        counts[(t["model"], t["done_in_first_response"])][t["outcome"]] += 1
xmax = max(sum(c.values()) for c in counts.values())
for ax, (flag, title) in zip(axes, groups):
    for i, m in enumerate(models):
        left, c = 0, counts[(m, flag)]
        for key, _, colour, text_colour in cats:
            v = c.get(key, 0)
            if v:
                ax.barh(i, v, left=left, color=colour, edgecolor="white", linewidth=2, height=0.62)
                if v >= 1:
                    ax.text(left + v / 2, i, str(v), ha="center", va="center", color=text_colour,
                            fontsize=9.5, fontweight="bold")
            left += v
        if left == 0:
            ax.text(0.3, i, "none", va="center", ha="left", fontsize=9.5, color=MUTED)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    style(ax, xmax=xmax, pct=False)
    ax.set_xticks(range(0, xmax + 1, 6))
    ax.set_xlabel("rollouts (of 24)", color=MUTED, fontsize=9.5)
axes[0].set_yticks(range(len(models)), models)
axes[0].invert_yaxis()
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c, _ in cats]
fig.legend(handles, [label for _, label, _, _ in cats], ncol=3, frameon=False, loc="lower left",
           bbox_to_anchor=(0.005, 0.965), fontsize=10, handlelength=1.2)
fig.suptitle("raman-lab rollouts by outcome and by how the model finished", x=0.01, ha="left", fontsize=12.5, y=1.07)
fig.savefig(OUT / "fig6_models_behaviour.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# --- Repo-only: passes per task and model (out of 3) ---------------------------------------
tasks = sorted({t["task"] for t in trials if t["dataset"] == "raman-lab"})
grid = [[sum(1 for t in trials if t["model"] == m and t["task"] == k and t["reward"] == "1.0") for k in tasks]
        for m in models]
fig, ax = plt.subplots(figsize=(8.2, 0.5 * len(models) + 1.4), dpi=200)
ax.imshow(grid, cmap="Blues", vmin=-0.4, vmax=3, aspect="auto")
for i, row in enumerate(grid):
    for j, v in enumerate(row):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=10, color="white" if v >= 2 else INK)
ax.set_xticks(range(len(tasks)), [k.split("-")[-1] for k in tasks])
ax.set_yticks(range(len(models)), models)
ax.tick_params(which="both", length=0)
for side in ax.spines.values():
    side.set_visible(False)
ax.set_xticks([x - 0.5 for x in range(1, len(tasks))], minor=True)
ax.set_yticks([y - 0.5 for y in range(1, len(models))], minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.set_xlabel("raman-lab task (sample)", color=MUTED)
fig.suptitle("Passed rollouts per task (of 3)", x=0.01, ha="left", fontsize=12.5)
fig.tight_layout()
fig.savefig(OUT / "fig_tasks_by_model.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
