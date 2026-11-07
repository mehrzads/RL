"""Charts + automatic diagnosis for the LLM-style RL runs.

    python report.py           -> prints a health check, writes charts/*.png
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))  # after llm/, so train.py here wins
import matplotlib.pyplot as plt  # noqa: E402
from plots import ALARM, INK2, SERIES, SURFACE, smooth  # noqa: E402  (shared chart style)
from train import RECIPES  # noqa: E402

CAUSE_COLORS = {"wall": SERIES[0], "self": SERIES[1], "starved": SERIES[2], "timeout": SERIES[3],
                "invalid": SERIES[7], "win": SERIES[4]}


def load(name, runs="runs"):
    rows = list(csv.DictReader(open(Path(runs) / name / "metrics.csv")))
    def col(k, x="games"):
        pts = [(float(r[x]), float(r[k])) for r in rows if r.get(k) not in (None, "")]
        return (np.array([p[0] for p in pts]), np.array([p[1] for p in pts])) if pts else (np.array([]), np.array([]))
    return rows, col


# ------------------------------------------------------------------ diagnosis
def diagnose(name, runs="runs"):
    _, col = load(name, runs)
    flags = []
    def ends(k):
        y = col(k)[1]
        return (y[: max(1, len(y) // 10)].mean(), y[-max(1, len(y) // 10):].mean()) if len(y) else (np.nan, np.nan)

    s0, s1 = ends("score")
    r0, r1 = ends("reward")
    if r1 - r0 > 0.2 * abs(r0) + 0.1 and s1 < s0 - 0.3:
        flags.append(("REWARD HACKING", f"optimised reward {r0:.2f} -> {r1:.2f} while true score {s0:.2f} -> {s1:.2f}"))
    else:
        sc, rw = col("score")[1], col("reward")[1]
        w = max(1, len(sc) // 10)
        sm_s, sm_r = np.convolve(sc, np.ones(w) / w, "valid"), np.convolve(rw, np.ones(w) / w, "valid")
        if len(sm_s) > 5 and sm_s.max() > 1 and sm_s[-1] < 0.8 * sm_s.max() and sm_r[-1] > 0.9 * sm_r.max():
            flags.append(("REWARD HACKING", f"true score peaked at {sm_s.max():.1f} and fell to {sm_s[-1]:.1f} "
                          "while the optimised reward stayed near its max"))
        elif s1 < s0 + 0.5:
            flags.append(("NO IMPROVEMENT", f"score {s0:.2f} -> {s1:.2f} over training"))
    k0, k1 = ends("kl")
    if k1 > 0.5:
        flags.append(("KL BLOW-UP", f"KL to SFT reference {k1:.2f} nats/token (a healthy run stays well under 0.2)"))
    e0, e1 = ends("entropy")
    if e1 < 0.05 and e1 < 0.3 * e0:
        flags.append(("ENTROPY COLLAPSE", f"token entropy {e0:.2f} -> {e1:.3f}: the model stopped exploring"))
    z0, z1 = ends("zero_signal_groups")
    if np.nanmean([z0, z1]) > 0.6:
        flags.append(("NO LEARNING SIGNAL", f"{np.nanmean([z0, z1]):.0%} of GRPO groups had identical rewards (advantage = 0)"))
    i0, i1 = ends("end_invalid")
    if max(i0, i1) > 0.3:
        flags.append(("FORMAT ERRORS", f"{i0:.0%} -> {i1:.0%} of games end on a non-move token"))
    st = sum(ends(k)[1] for k in ("end_starved", "end_timeout"))
    if st > 0.3:
        flags.append(("STALLING", f"{st:.0%} of games end by starving / timeout (loop ratio {ends('loop_ratio')[1]:.0%})"))
    c0, c1 = ends("chosen_logp_shift")
    a0, a1 = ends("pair_acc")
    if c1 < -0.1:
        flags.append(("LIKELIHOOD DISPLACEMENT", f"chosen answers got LESS likely ({c1:+.2f} nats/token vs reference): "
                      "DPO is pushing rejected down faster, drifting off the data"))
    l0, l1 = ends("dpo_loss")
    if abs(l1 - np.log(2)) < 0.01 and a1 < 0.6:
        flags.insert(0, ("NO PREFERENCE SIGNAL", f"DPO loss stuck at {l1:.3f} (ln 2 = coin flip), pair accuracy {a1:.0%}"))
    v0, v1 = ends("explained_var")
    if v1 < 0:
        flags.append(("CRITIC FAILING", f"explained variance {v1:.2f}"))
    return flags


# ------------------------------------------------------------------ charts
def line(ax, x, y, color=SERIES[0], label=None, raw=True):
    if not len(y):
        return
    if raw and len(y) > 15:
        ax.plot(x, y, color=color, alpha=0.18, linewidth=1)
        s, w = smooth(y, 8)
        ax.plot(x[w - 1:], s, color=color, label=label)
    else:
        ax.plot(x, y, color=color, label=label, marker="o" if len(y) < 20 else None, markersize=4)


def dashboard(name, runs="runs", out="charts"):
    rows, col = load(name, runs)
    rc = RECIPES[name]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    (a1, a2, a3), (a4, a5, a6) = axes
    xl = "games played" if rc["algo"] != "dpo" else "preference games seen"

    line(a1, *col("score")); a1.set_title("True score (food eaten)"); a1.set_ylim(bottom=0)
    rlabel = {"rm": "reward-model score", "length": "length reward", "score>=10": "pass rate (score >= 10)",
              "score>=20": "pass rate (score >= 20)"}
    line(a2, *col("reward")); a2.set_title(f"Reward being optimised: {rlabel.get(rc.get('reward'), 'score - format penalty')}")
    if rc["algo"] == "dpo":
        line(a3, *col("chosen_logp_shift"), SERIES[0], "chosen", raw=False)
        line(a3, *col("rejected_logp_shift"), SERIES[1], "rejected", raw=False)
        a3.axhline(0, color=INK2, linewidth=1, linestyle=":")
        a3.set_title("DPO: log-prob/token vs reference"); a3.legend()
    else:
        line(a3, *col("kl")); a3.set_title("KL to SFT reference (nats/token)"); a3.set_ylim(bottom=0)
    line(a4, *col("entropy")); a4.set_title("Token entropy (exploration)"); a4.set_ylim(bottom=0)

    x = col("end_wall")[0]
    shares = {c: col(f"end_{c}")[1] for c in CAUSE_COLORS}
    shares = {c: s for c, s in shares.items() if len(s) and s.max() > 0.01}
    if len(x):
        w = 5 if len(x) > 20 else 1
        sm = {c: np.convolve(s, np.ones(w) / w, "valid") for c, s in shares.items()}
        a4b = a5
        a4b.stackplot(x[w - 1:], *sm.values(), labels=sm.keys(), colors=[CAUSE_COLORS[c] for c in sm],
                      edgecolor=SURFACE, linewidth=0.5)
        a4b.set_ylim(0, 1); a4b.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=len(sm), fontsize=9, handlelength=1)
    a5.set_title("How games end")

    if rc["algo"] == "grpo":
        line(a6, *col("zero_signal_groups"), SERIES[0], "groups with zero signal", raw=False)
        line(a6, *col("clipfrac"), SERIES[1], "clip fraction", raw=False)
        a6.set_ylim(-0.02, 1.02); a6.set_title("GRPO internals"); a6.legend()
    elif rc["algo"] == "ppo":
        line(a6, *col("explained_var")); a6.set_title("Critic explained variance (1 = perfect)")
        a6.axhline(0, color=INK2, linewidth=1, linestyle=":")
    elif rc["algo"] == "dpo":
        line(a6, *col("pair_acc"), raw=True); a6.set_title("DPO pair accuracy (train)"); a6.set_ylim(0, 1.02)
    else:
        line(a6, *col("grad_norm")); a6.set_title("Gradient norm (before clipping at 1.0)")
    for a in axes.ravel():
        a.set_xlabel(xl)

    flags = diagnose(name, runs)
    fig.suptitle(f"{name}  ·  {rc['story']}", x=0.01, ha="left", fontsize=14, fontweight="bold")
    txt = ("DETECTED:  " + "   |   ".join(f"{t}: {w}" for t, w in flags[:3])) if flags else "HEALTHY: no alarms raised"
    fig.text(0.01, 0.925, txt, color=ALARM if flags else SERIES[5], fontsize=10.5, ha="left", wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(Path(out) / f"{name}_dashboard.png", dpi=120)
    plt.close(fig)


def compare(names, fname, title, metric="score", out="charts", hline=None):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for i, n in enumerate(names):
        x, y = load(n)[1](metric)
        if not len(y):
            continue
        s, w = smooth(y, 8) if len(y) > 15 else (y, 1)
        ax.plot(x[w - 1:], s, color=SERIES[i], label=n, marker="o" if len(y) < 20 else None, markersize=4)
        ax.annotate(n, (x[-1], s[-1]), xytext=(6, 0), textcoords="offset points", fontsize=10, va="center")
    if hline:
        ax.axhline(hline[0], color=INK2, linestyle=":", linewidth=1)
        ax.text(ax.get_xlim()[0], hline[0], f" {hline[1]}", color=INK2, va="bottom", fontsize=10)
    ax.set_title(title); ax.set_ylabel(metric); ax.set_xlabel("games played")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.15); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(Path(out) / fname, dpi=120)
    plt.close(fig)


def overoptimisation(out="charts"):
    """Gao et al. style: proxy (reward-model) score vs true score, with and without the KL penalty."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.5))
    for i, n in enumerate(["ppo_rlhf", "bad_rlhf_no_kl"]):
        if not (Path("runs") / n / "metrics.csv").exists():
            continue
        _, col = load(n)
        rm, sc = col("reward")[1], col("score")[1]
        a.plot(*col("reward"), color=SERIES[i], alpha=0.25, linewidth=1)
        s, w = smooth(rm, 8); a.plot(col("reward")[0][w - 1:], s, color=SERIES[i], label=n)
        s, w = smooth(sc, 8); b.plot(col("score")[0][w - 1:], s, color=SERIES[i], label=n)
    a.set_title("Reward-model score: 'it's improving!'"); b.set_title("True score: ...is it?")
    for ax in (a, b):
        ax.set_xlabel("games played"); ax.legend()
    fig.suptitle("Reward-model over-optimisation: KL penalty vs none", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(Path(out) / "rlhf_overoptimisation.png", dpi=120)
    plt.close(fig)


def sft_chart(out="charts"):
    rows = list(csv.DictReader(open("runs/sft/metrics.csv")))
    expert = float(Path("runs/sft/expert_score.txt").read_text())
    step = np.array([float(r["step"]) for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(step, [float(r["loss"]) for r in rows], color=SERIES[0]); axes[0].set_title("SFT loss (next-move cross-entropy)")
    axes[1].plot(step, [float(r["p_valid_format"]) for r in rows], color=SERIES[0]); axes[1].set_title("P(answer is a move token)")
    axes[1].set_ylim(0, 1.02)
    pts = [(float(r["step"]), float(r["score"])) for r in rows if r["score"]]
    axes[2].plot(*zip(*pts), color=SERIES[0], marker="o", markersize=5)
    axes[2].axhline(expert, color=INK2, linestyle=":", linewidth=1)
    axes[2].text(pts[0][0], expert, " expert (with 10% noise)", color=INK2, va="bottom", fontsize=10)
    axes[2].set_title("Score when the model plays"); axes[2].set_ylim(bottom=0)
    for a in axes:
        a.set_xlabel("SFT step")
    fig.suptitle("Stage 1 · SFT: imitate the expert's moves", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(Path(out) / "sft.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    Path("charts").mkdir(exist_ok=True)
    have = [n for n in RECIPES if (Path("runs") / n / "metrics.csv").exists()]
    for n in have:
        flags = diagnose(n)
        print(f"\n{n}  [{RECIPES[n]['kind']}]  ->  {'UNHEALTHY' if flags else 'HEALTHY'}")
        for t, w in flags:
            print(f"   ! {t:24s} {w}")
        dashboard(n)
    sft_start = float(load(have[0])[1]("score")[1][:3].mean()) if have else None
    good = [n for n in have if RECIPES[n]["kind"] == "good"]
    compare(good, "compare_good.png", "LLM-style RL on top of SFT: true score",
            hline=(sft_start, "SFT model") if sft_start else None)
    compare([n for n in ("grpo", "reinforce", "rloo") if n in have], "compare_baselines.png",
            "Where GRPO comes from: REINFORCE -> RLOO -> GRPO")
    compare([n for n in ("grpo", "bad_grpo_no_kl_high_lr") if n in have], "compare_kl.png",
            "KL to reference: with vs without", metric="kl")
    overoptimisation()
    if Path("runs/sft/metrics.csv").exists():
        sft_chart()
    print("\ncharts written to charts/")
