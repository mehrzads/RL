"""Training charts for the video.

    python plots.py      -> charts/<run>_dashboard.png, charts/compare_*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from detect import diagnose, load
from recipes import RECIPES

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ALARM = "#e34948"
CAUSES = {"wall": SERIES[0], "self": SERIES[1], "starved": SERIES[2], "timeout": SERIES[3], "win": SERIES[4]}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "legend.frameon": False,
})


def smooth(y, n):
    if len(y) < 3:
        return y, 1
    w = max(1, min(len(y) // 30, n))
    return np.convolve(y, np.ones(w) / w, mode="valid"), w


def curve(ax, x, y, color=SERIES[0], label=None, raw=True):
    if raw:
        ax.plot(x, y, color=color, alpha=0.15, linewidth=1)
    s, w = smooth(y, 200)
    ax.plot(x[w - 1:], s, color=color, label=label)


def kfmt(ax):
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.set_xlabel("environment steps")


def dashboard(name, runs="runs", out="charts"):
    rows, col = load(Path(runs) / name)
    rc = RECIPES.get(name, {})
    x = col("step")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    (a1, a2, a3), (a4, a5, a6) = axes

    curve(a1, x, col("score")); a1.set_title("Score (food eaten) — what we want")
    curve(a2, x, col("return")); a2.set_title("Episode return — what the agent optimises")
    curve(a3, x, col("length")); a3.set_title("Episode length (steps)")

    # death-cause share over training (stacked area of rolling fractions)
    w = max(20, len(rows) // 40)
    xs = x[w - 1:]
    shares = {c: np.convolve([r["cause"] == c for r in rows], np.ones(w) / w, "valid") for c in CAUSES}
    shares = {c: s for c, s in shares.items() if s.max() > 0.01}
    a4.stackplot(xs, *shares.values(), labels=shares.keys(), colors=[CAUSES[c] for c in shares],
                 edgecolor=SURFACE, linewidth=0.5)
    a4.set_ylim(0, 1); a4.set_title("How episodes end"); a4.legend(loc="lower left", ncol=len(shares), frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=1)
    a4.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))

    curve(a5, x, col("loop_ratio"), SERIES[0], "loop ratio (revisited cells)", raw=False)
    curve(a5, x, col("action_entropy"), SERIES[1], "action entropy (normalised)", raw=False)
    a5.set_ylim(0, 1.05); a5.set_title("Behaviour fingerprints"); a5.legend(loc="center right")

    if rc.get("agent") == "ppo":
        e = col("entropy")
        a6.plot(np.arange(1, len(e) + 1), e, color=SERIES[0], marker="o", markersize=3)
        a6.axhline(np.log(3), color=INK2, linestyle=":", linewidth=1)
        a6.text(1, np.log(3), " max (uniform)", va="bottom", color=INK2, fontsize=9)
        a6.set_title("Policy entropy per PPO update (nats)"); a6.set_xlabel("PPO update")
        a6.set_ylim(bottom=0)
    else:
        qm = col("q_max")
        curve(a6, x[-len(qm):], qm, SERIES[0], raw=True)
        a6.set_title("Max Q-value (agent's own value estimate)")
        if qm.max() > 100:
            a6.set_yscale("symlog")
        kfmt(a6)
    for a in (a1, a2, a3, a4, a5):
        kfmt(a)

    flags = diagnose(name, runs)
    head = f"{name}  ·  {rc.get('agent', '?').upper()}  ·  {rc.get('story', '')}"
    fig.suptitle(head, x=0.01, ha="left", fontsize=15, fontweight="bold")
    if flags:
        txt = "DETECTED:  " + "   |   ".join(f"{t}: {why}" for t, why in flags[:3])
        fig.text(0.01, 0.925, txt, color=ALARM, fontsize=10.5, ha="left", wrap=True)
    else:
        fig.text(0.01, 0.925, "HEALTHY: no alarms raised", color=SERIES[5], fontsize=10.5, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    Path(out).mkdir(exist_ok=True)
    fig.savefig(Path(out) / f"{name}_dashboard.png", dpi=120)
    plt.close(fig)


def compare(names, title, fname, runs="runs", out="charts", metric="score"):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for i, n in enumerate(names):
        rows, col = load(Path(runs) / n)
        x, y = col("step"), col(metric)
        s, w = smooth(y, 200)
        ax.plot(x[w - 1:], s, color=SERIES[i], label=n)
        ax.annotate(n, (x[-1], s[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=10, va="center")
    ax.set_title(title); ax.set_ylabel(metric); kfmt(ax)
    ax.legend(loc="lower right")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.12)
    fig.tight_layout()
    fig.savefig(Path(out) / fname, dpi=120)
    plt.close(fig)


def proxy_vs_goal(name, runs="runs", out="charts"):
    """The money shot for reward hacking: reward climbs, score doesn't. Two panels, never two y-axes."""
    rows, col = load(Path(runs) / name)
    x = col("step")
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.5))
    curve(a, x, col("return"), SERIES[0]); a.set_title("Return: 'it's learning!'")
    curve(b, x, col("score"), ALARM); b.set_title("Score: ...it is not")
    b.set_ylim(0, max(3, col("score").max() + 0.5))
    kfmt(a); kfmt(b)
    fig.suptitle(f"{name}: reward goes up, goal stays flat", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(Path(out) / f"{name}_proxy_vs_goal.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    have = [n for n in RECIPES if (Path("runs") / n / "metrics.csv").exists()]
    for n in have:
        dashboard(n)
    good = [n for n in have if RECIPES[n]["kind"] == "good"]
    compare(good, "Good recipes: score while training", "compare_good.png")
    for n in have:
        if RECIPES[n]["kind"] == "bad":
            base = RECIPES[n]["agent"]
            compare([base, n], f"{n} vs healthy {base}", f"compare_{n}.png")
    for n in ("bad_survival_reward", "bad_step_penalty"):
        if n in have:
            proxy_vs_goal(n)
    print("charts written to charts/")
