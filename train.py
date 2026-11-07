"""Train one or more recipes and log everything needed for charts, diagnostics and replays.

    python train.py qtable dqn            # specific recipes
    python train.py --all                 # everything (runs in parallel)

Writes runs/<recipe>/metrics.csv (one row per episode) and runs/<recipe>/replays/*.json.
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from agents import AGENTS
from recipes import RECIPES
from snake_env import SnakeEnv

REPLAY_AT = [0.0, 0.05, 0.15, 0.4, 1.0]  # fractions of training at which to record a greedy game


def play_greedy(agent, env_kwargs, seed=123, max_steps=500):
    env = SnakeEnv(seed=seed, **{**env_kwargs, "max_steps": max_steps})
    o, frames, done, info = env.reset(), [env.snapshot()], False, {}
    while not done:
        o, _, done, info = env.step(agent.greedy(o))
        frames.append(env.snapshot())
    return {"size": env.size, "score": info["score"], "cause": info["cause"], "frames": frames}


def record(agent, env_kwargs, path, n=5):
    """Play n greedy games and keep the median-scoring one: representative, not cherry-picked."""
    games = sorted((play_greedy(agent, env_kwargs, seed=100 + i) for i in range(n)), key=lambda g: g["score"])
    pick = games[n // 2]
    pick["all_scores"] = [g["score"] for g in games]
    path.write_text(json.dumps(pick))


def train(name, seed=0, out="runs"):
    torch.set_num_threads(1)
    rc = RECIPES[name]
    run = Path(out) / name
    (run / "replays").mkdir(parents=True, exist_ok=True)
    env = SnakeEnv(seed=seed, **rc["env"])
    agent = AGENTS[rc["agent"]](seed=seed, **rc["hp"])
    total = rc["steps"]
    replay_steps = [int(f * total) for f in REPLAY_AT]

    rows, step, ep, t0 = [], 0, 0, time.time()
    while step < total:
        o, done, ret, acts, stats = env.reset(), False, 0.0, np.zeros(3), defaultdict(list)
        while not done:
            if replay_steps and step >= replay_steps[0]:
                pct = round(100 * replay_steps.pop(0) / total)
                record(agent, rc["env"], run / "replays" / f"{pct:03d}.json")
            a = agent.act(o)
            o2, r, done, info = env.step(a)
            for k, v in agent.observe(o, a, r, o2, done).items():
                stats[k].append(v)
            o, ret, step = o2, ret + r, step + 1
            acts[a] += 1
        for k, v in agent.end_episode().items():
            stats[k].append(v)
        p = acts / acts.sum()
        rows.append({
            "episode": ep, "step": step, "score": info["score"], "length": info["steps"],
            "return": ret, "cause": info["cause"], "loop_ratio": info["loop_ratio"],
            "p_straight": p[0], "p_right": p[1], "p_left": p[2],
            "action_entropy": float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(3)),
            **{k: float(np.mean(v)) for k, v in stats.items()},
        })
        ep += 1
    for f in replay_steps:  # anything not yet recorded (e.g. 100%)
        record(agent, rc["env"], run / "replays" / f"{round(100 * f / total):03d}.json")

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(run / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    last = rows[-max(1, len(rows) // 10):]
    msg = (f"{name:22s} {time.time() - t0:6.0f}s  episodes={ep:5d}  "
           f"last-10% mean score={np.mean([r['score'] for r in last]):5.2f}  "
           f"len={np.mean([r['length'] for r in last]):6.1f}")
    print(msg, flush=True)
    return msg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("recipes", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    names = list(RECIPES) if args.all else args.recipes
    if not names:
        ap.error("name some recipes or pass --all; available: " + ", ".join(RECIPES))
    with Pool(min(args.jobs, len(names))) as pool:
        pool.starmap(train, [(n, args.seed) for n in names])
