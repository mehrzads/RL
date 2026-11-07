"""Stage 1: supervised fine-tuning (the 'pretraining' of our tiny LLM).

A tabular Q-learning expert plays with a little noise (imperfect demonstrations, like web data);
the GPT learns to predict its next move from the board tokens.

    python sft.py      -> ckpt/sft.pt, runs/sft/metrics.csv
"""
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import ACT0, SIZE, HUNGER, TinyGPT, encode, n_params, rollout
from agents import QTable
from snake_env import SnakeEnv

torch.manual_seed(0)


def train_expert(episodes=3000):
    env, q = SnakeEnv(size=SIZE, seed=1, hunger_limit=HUNGER), QTable(eps_decay=0.995)
    for _ in range(episodes):
        o, done = env.reset(), False
        while not done:
            a = q.act(o)
            o2, r, done, _ = env.step(a)
            q.observe(o, a, r, o2, done)
            o = o2
        q.end_episode()
    return q


def demonstrations(q, n_games=3000, noise=0.1, seed=10_000):
    rng = np.random.default_rng(0)
    X, Y, scores = [], [], []
    for g in range(n_games):
        env = SnakeEnv(size=SIZE, seed=seed + g, hunger_limit=HUNGER)
        o, done = env.reset(), False
        while not done:
            a = q.greedy(o) if rng.random() > noise else int(rng.integers(3))
            X.append(encode(env))
            Y.append(ACT0 + (env.dir + (0, 1, -1)[a]) % 4)   # relative -> absolute move token
            o, _, done, info = env.step(a)
        scores.append(info["score"])
    return torch.as_tensor(np.array(X)), torch.as_tensor(np.array(Y)), float(np.mean(scores))


def main(epochs=3, batch=256, lr=1e-3, out=Path("runs/sft")):
    torch.set_num_threads(8)
    t0 = time.time()
    q = train_expert()
    X, Y, expert_score = demonstrations(q)
    print(f"expert avg score {expert_score:.2f}, {len(X)} (prompt, answer) pairs, {time.time() - t0:.0f}s")

    model = TinyGPT()
    print(f"TinyGPT: {n_params(model) / 1e3:.0f}k parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * (len(X) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    out.mkdir(parents=True, exist_ok=True)
    rows, step = [], 0
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X) - batch + 1, batch):
            idx = perm[i:i + batch]
            logits = model(X[idx])
            loss = F.cross_entropy(logits, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 100 == 0 or step == steps:
                row = {"step": step, "loss": loss.item(),
                       "acc": (logits.argmax(-1) == Y[idx]).float().mean().item(),
                       "p_valid_format": logits.softmax(-1)[:, ACT0:].sum(-1).mean().item()}
                if step % 500 == 0 or step == steps:
                    games = rollout(model, range(64))
                    row["score"] = np.mean([g["score"] for g in games])
                    row["invalid_rate"] = np.mean([g["cause"] == "invalid" for g in games])
                    print(f"step {step}/{steps} loss {row['loss']:.3f} acc {row['acc']:.2f} "
                          f"sampled score {row['score']:.2f} invalid {row['invalid_rate']:.0%}", flush=True)
                rows.append(row)
    Path("ckpt").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "ckpt/sft.pt")
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss", "acc", "p_valid_format", "score", "invalid_rate"])
        w.writeheader()
        w.writerows(rows)
    (out / "expert_score.txt").write_text(f"{expert_score}\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
