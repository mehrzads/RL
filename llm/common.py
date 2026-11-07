"""Snake as a language-model task.

The board is a 'prompt' of tokens; the model 'answers' with one token, its next move.
Exactly like an LLM, it samples from the FULL vocabulary, so it can answer with garbage
(a board token instead of a move). That ends the game: a format error.

    prompt:  <bos> c00 c01 ... c77 <dir> <sep>      (67 tokens on an 8x8 board)
    answer:  <U> | <R> | <D> | <L>
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))  # after llm/, so train.py here wins
from snake_env import SnakeEnv  # noqa: E402

SIZE = 8
HUNGER = 64           # steps without food before starving (at length 3)
MAX_STEPS = 150

VOCAB = ["<bos>", ".", "o", "H", "t", "*", "^", ">", "v", "<", "<sep>", "U", "R", "D", "L"]
BOS, EMPTY, BODY, HEAD, TAIL, FOOD = range(6)
DIR0, SEP, ACT0 = 6, 10, 11
V = len(VOCAB)
CTX = 1 + SIZE * SIZE + 2
CAUSES = ["wall", "self", "starved", "timeout", "invalid", "win"]


def encode(env):
    g = np.full((SIZE, SIZE), EMPTY, np.int64)
    g[env.food[1], env.food[0]] = FOOD
    for x, y in env.body[1:-1]:
        g[y, x] = BODY
    tx, ty = env.body[-1]
    g[ty, tx] = TAIL
    hx, hy = env.body[0]
    g[hy, hx] = HEAD
    return np.concatenate([[BOS], g.ravel(), [DIR0 + env.dir, SEP]])


def show(tokens):
    """Pretty-print a prompt, handy on camera: python -c 'from common import *; ...'"""
    t = [VOCAB[i] for i in tokens]
    rows = [" ".join(t[1 + r * SIZE: 1 + (r + 1) * SIZE]) for r in range(SIZE)]
    return "\n".join(rows) + f"\nheading {t[-2]}"


def new_env(seed):
    return SnakeEnv(size=SIZE, seed=seed, hunger_limit=HUNGER, max_steps=MAX_STEPS)


def env_step(env, tok):
    """Apply an answer token. Returns (done, cause)."""
    if not ACT0 <= tok < ACT0 + 4:
        return True, "invalid"
    a = tok - ACT0
    if a == (env.dir + 2) % 4:  # reversing into your own neck
        env.steps += 1
        return True, "self"
    rel = {0: 0, 1: 1, 3: 2}[(a - env.dir) % 4]
    _, _, done, info = env.step(rel)
    return done, info["cause"]


# ------------------------------------------------------------------ model
class TinyGPT(nn.Module):
    """Decoder-only transformer. out=V -> language model head; out=1 -> scalar head (critic / reward model)."""

    def __init__(self, out=V, d=64, layers=3, heads=4):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(CTX, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, out)
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(CTX), persistent=False)

    def forward(self, x):
        h = self.tok(x) + self.pos
        h = self.blocks(h, mask=self.mask, is_causal=True)
        y = self.head(self.ln(h[:, -1]))
        return y.squeeze(-1) if y.shape[-1] == 1 else y


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def token_logp(policy, obs, acts, chunk=2048):
    """log pi(answer | prompt) for many (prompt, answer) pairs, with grad."""
    out = []
    for i in range(0, len(obs), chunk):
        lp = F.log_softmax(policy(obs[i:i + chunk]), -1)
        out.append(lp.gather(1, acts[i:i + chunk, None]).squeeze(1))
    return torch.cat(out)


# ---------------------------------------------------------------- rollouts
@torch.no_grad()
def rollout(policy, seeds, group=1, greedy=False, record=False):
    """Play len(seeds)*group games in lock-step, batching the model calls.
    Games sharing a seed share their start (the 'prompt group' in GRPO/RLOO)."""
    policy.eval()
    envs = [new_env(s) for s in seeds for _ in range(group)]
    eps = [{"seed": s, "obs": [], "acts": [], "logp": [], "ent": [], "frames": []}
           for s in seeds for _ in range(group)]
    active = list(range(len(envs)))
    while active:
        X = torch.as_tensor(np.stack([encode(envs[i]) for i in active]))
        logp_all = F.log_softmax(policy(X), -1)
        acts = logp_all.argmax(-1) if greedy else torch.multinomial(logp_all.exp(), 1).squeeze(1)
        lp = logp_all.gather(1, acts[:, None]).squeeze(1)
        ent = -(logp_all.exp() * logp_all).sum(-1)
        still = []
        for k, i in enumerate(active):
            e, env = eps[i], envs[i]
            if record:
                e["frames"].append(env.snapshot())
            e["obs"].append(X[k].numpy().astype(np.int8))
            e["acts"].append(int(acts[k]))
            e["logp"].append(float(lp[k]))
            e["ent"].append(float(ent[k]))
            done, cause = env_step(env, int(acts[k]))
            if done:
                if record:
                    e["frames"].append(env.snapshot())
                e.update(score=env.score, length=len(e["acts"]), cause=cause,
                         loop_ratio=env.revisits / max(1, env.steps))
                for k2 in ("obs", "acts", "logp", "ent"):
                    e[k2] = np.array(e[k2])
            else:
                still.append(i)
        active = still
    policy.train()
    return eps


def flatten(eps):
    obs = torch.as_tensor(np.concatenate([e["obs"] for e in eps]).astype(np.int64))
    acts = torch.as_tensor(np.concatenate([e["acts"] for e in eps]))
    return obs, acts
