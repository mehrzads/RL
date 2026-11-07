"""Automatic 'is my RL run healthy?' checks. Each rule mirrors something you'd look for by eye.

    python detect.py              # check every run in runs/
    python detect.py bad_myopic   # check one
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from recipes import RECIPES


def load(run):
    rows = list(csv.DictReader(open(Path(run) / "metrics.csv")))
    def col(k, rs=rows):
        return np.array([float(r[k]) for r in rs if r.get(k) not in (None, "")])
    return rows, col


def diagnose(name, runs="runs"):
    rows, col = load(Path(runs) / name)
    baseline = RECIPES.get(name, {}).get("agent")  # the good recipe for the same algorithm
    q = max(1, len(rows) // 10)
    first, last = rows[:q], rows[-q:]
    m = lambda k, rs: col(k, rs).mean() if len(col(k, rs)) else np.nan
    causes = Counter(r["cause"] for r in last)
    n_last = len(last)
    flags = []

    # 1. Proxy reward goes up, the thing we actually care about doesn't.
    ret_gain = m("return", last) - m("return", first)
    score_gain = m("score", last) - m("score", first)
    if ret_gain > 20 and score_gain < 1:
        flags.append(("REWARD HACKING", f"return +{ret_gain:.0f} but score {score_gain:+.2f}: "
                      "the agent optimises the reward, not the goal"))

    # 2. Going in circles.
    if m("loop_ratio", last) > 0.5:
        flags.append(("LOOPING", f"{m('loop_ratio', last):.0%} of moves revisit cells since last food"))

    # 3. Episodes end by running out the clock instead of playing.
    stall = (causes["timeout"] + causes["starved"]) / n_last
    if stall > 0.3:
        flags.append(("STALLING", f"{stall:.0%} of episodes end by timeout/starvation"))

    # 4. Dying on purpose.
    if m("length", last) < 15 and causes["wall"] / n_last > 0.8:
        flags.append(("SUICIDE", f"avg episode {m('length', last):.1f} steps, "
                      f"{causes['wall'] / n_last:.0%} wall deaths: dying is cheaper than living"))

    # 5. Value estimates impossible given the returns ever seen.
    qmax, max_ret = col("q_max"), np.abs(col("return")).max()
    if len(qmax) and qmax.max() > max(50, 3 * max_ret):
        flags.append(("Q DIVERGENCE", f"peak Q {qmax.max():.0f} vs best |episode return| {max_ret:.0f}"))
    gn = col("grad_norm")
    if len(gn) and gn.max() > 20 * np.median(gn):
        flags.append(("GRADIENT SPIKES", f"max grad norm {gn.max():.1f} vs median {np.median(gn):.2f}"))

    # 6. Policy stopped exploring.
    ent = col("entropy")
    if len(ent) and ent[-1] < 0.01:
        k = int(np.argmax(ent < 0.01))
        flags.append(("ENTROPY COLLAPSE", f"policy entropy < 0.01 nats by update {k + 1}/{len(ent)}"))
    if m("action_entropy", last) < 0.2:
        flags.append(("ONE-ACTION POLICY", f"normalised action entropy {m('action_entropy', last):.2f}"))

    # 7. Not learning at all / underperforming a known-good recipe.
    final = m("score", last)
    if final < 2:
        flags.append(("NOT LEARNING", f"final score {final:.2f}"))
    elif name != baseline and (Path(runs) / baseline / "metrics.csv").exists():
        b_rows, b_col = load(Path(runs) / baseline)
        b = b_col("score", b_rows[-max(1, len(b_rows) // 10):]).mean()
        if final < 0.85 * b:
            flags.append(("UNDERPERFORMING", f"final score {final:.1f} vs {baseline} baseline {b:.1f}"))

    return flags


def main(names, runs="runs"):
    for n in names:
        flags = diagnose(n, runs)
        kind = RECIPES.get(n, {}).get("kind", "?")
        verdict = "HEALTHY" if not flags else "UNHEALTHY"
        print(f"\n{n}  [{kind} recipe]  ->  {verdict}")
        for tag, why in flags:
            print(f"   ! {tag:17s} {why}")


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.name for p in Path("runs").iterdir() if (p / "metrics.csv").exists())
    main(names)
