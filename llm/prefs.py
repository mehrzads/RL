"""Stage 2 (for RLHF and DPO): preference pairs + a learned reward model.

Two games are sampled from the SFT model on the same start (same 'prompt'). A 'rater' prefers
the game with the higher score: ties are a coin flip and 10% of labels are flipped (rater noise).

The reward model scores every (board, move) step; a game's reward is the sum over its steps,
trained with the Bradley-Terry loss  -log sigmoid(R(chosen) - R(rejected)).

    python prefs.py                   -> ckpt/prefs.pt, ckpt/dpo_pairs.pt, ckpt/rm.pt, runs/reward_model/metrics.csv
    python prefs.py --dpo-pairs-only  -> regenerate ckpt/dpo_pairs.pt only
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import TinyGPT, rollout

torch.manual_seed(0)


def rm_input(obs, acts):
    """Reward model reads the prompt with the answer in place of <sep>."""
    x = obs.clone()
    x[:, -1] = acts
    return x


def traj_reward(rm, eps):
    """Sum of per-step rewards for each episode in a list, with grad."""
    lens = [len(e["acts"]) for e in eps]
    obs = torch.as_tensor(np.concatenate([e["obs"] for e in eps]).astype(np.int64))
    acts = torch.as_tensor(np.concatenate([e["acts"] for e in eps]))
    r = rm(rm_input(obs, acts))
    return torch.stack([s.sum() for s in torch.split(r, lens)])


def make_pairs(policy, n=2000, noise=0.1, seed=50_000):
    rng = np.random.default_rng(0)
    games = rollout(policy, range(seed, seed + n), group=2)
    pairs = []
    for a, b in zip(games[0::2], games[1::2]):
        prefer_a = a["score"] > b["score"] if a["score"] != b["score"] else rng.random() < 0.5
        if rng.random() < noise:
            prefer_a = not prefer_a
        pairs.append((a, b) if prefer_a else (b, a))
    return pairs


def make_contrastive_pairs(policy, n=1500, k=4, seed=80_000):
    """Cleaner DPO data: sample k games per start, chosen = best, rejected = worst (skip if tied)."""
    games = rollout(policy, range(seed, seed + n), group=k)
    pairs = []
    for i in range(0, len(games), k):
        g = sorted(games[i:i + k], key=lambda e: e["score"])
        if g[-1]["score"] > g[0]["score"]:
            pairs.append((g[-1], g[0]))
    return pairs


def slim(e):
    return {k: e[k] for k in ("obs", "acts", "score", "length")}


def load_sft():
    sft = TinyGPT()
    sft.load_state_dict(torch.load("ckpt/sft.pt"))
    return sft


def save_dpo_pairs(sft):
    best_worst = make_contrastive_pairs(sft)
    torch.save([(slim(c), slim(r)) for c, r in best_worst], "ckpt/dpo_pairs.pt")
    print(f"{len(best_worst)} best-vs-worst pairs, mean score gap "
          f"{np.mean([c['score'] - r['score'] for c, r in best_worst]):.1f}")


def main(epochs=2, batch=16, lr=3e-4):
    torch.set_num_threads(8)
    sft = load_sft()
    pairs = make_pairs(sft)
    torch.save([(slim(c), slim(r)) for c, r in pairs], "ckpt/prefs.pt")
    save_dpo_pairs(sft)
    gap = np.mean([c["score"] > r["score"] for c, r in pairs])
    print(f"{len(pairs)} pairs; chosen has strictly higher score in {gap:.0%}")

    train, held = pairs[:1800], pairs[1800:]
    rm = TinyGPT(out=1)  # initialised from the SFT model's body, like InstructGPT
    rm.load_state_dict({k: v for k, v in sft.state_dict().items() if not k.startswith("head")}, strict=False)
    opt = torch.optim.AdamW(rm.parameters(), lr=lr)
    rows, step = [], 0

    def accuracy(ps):
        with torch.no_grad():
            c, r = traj_reward(rm, [p[0] for p in ps]), traj_reward(rm, [p[1] for p in ps])
        return (c > r).float().mean().item()

    for ep in range(epochs):
        order = np.random.permutation(len(train))
        for i in range(0, len(train), batch):
            ps = [train[j] for j in order[i:i + batch]]
            rc, rr = traj_reward(rm, [p[0] for p in ps]), traj_reward(rm, [p[1] for p in ps])
            loss = -F.logsigmoid(rc - rr).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 25 == 0:
                row = {"step": step, "loss": loss.item(), "train_acc": (rc > rr).float().mean().item(),
                       "heldout_acc": accuracy(held)}
                rows.append(row)
                print(row, flush=True)
    torch.save(rm.state_dict(), "ckpt/rm.pt")

    # what did it learn? correlation of RM total with true score and with game length
    with torch.no_grad():
        eps = [g for p in held for g in p]
        R = traj_reward(rm, eps).numpy()
    s, L = np.array([e["score"] for e in eps]), np.array([e["length"] for e in eps])
    print(f"held-out: corr(RM, score) = {np.corrcoef(R, s)[0, 1]:.2f}, corr(RM, length) = {np.corrcoef(R, L)[0, 1]:.2f}")
    Path("runs/reward_model").mkdir(parents=True, exist_ok=True)
    with open("runs/reward_model/metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    if sys.argv[1:] == ["--dpo-pairs-only"]:
        save_dpo_pairs(load_sft())
    else:
        main()
