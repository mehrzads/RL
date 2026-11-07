"""Stage 3: RL on top of the SFT model. REINFORCE, RLOO, GRPO, PPO-RLHF (critic + reward model), DPO.

    python train.py grpo rloo           # specific recipes
    python train.py --all               # everything, in parallel

Every recipe starts from ckpt/sft.pt; the frozen SFT model is the KL reference.
Logs runs/<recipe>/metrics.csv (one row per iteration) and runs/<recipe>/replays/*.json.
"""
import argparse
import csv
import json
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import CAUSES, TinyGPT, flatten, rollout, token_logp
from prefs import rm_input, traj_reward

# --------------------------------------------------------------------------- recipes
BASE = dict(iters=80, prompts=8, group=8, lr=1e-4, kl=0.0, epochs=1, clip=0.2,
            max_tokens=4096, reward="score", init="sft")
RECIPES = {
    # good
    "reinforce": dict(kind="good", algo="reinforce",
                      story="REINFORCE: push up log-prob of every move in a game, scaled by (reward - batch mean)."),
    "rloo": dict(kind="good", algo="rloo",
                 story="RLOO: 8 games per start; baseline = mean reward of the OTHER 7 games."),
    "grpo": dict(kind="good", algo="grpo", kl=0.04, epochs=2,
                 story="GRPO: group-normalised advantage, PPO clipping, KL to the SFT model. No critic."),
    "ppo_rlhf": dict(kind="good", algo="ppo", reward="rm", kl=0.2, epochs=2, max_tokens=2048, iters=50,
                     story="PPO-RLHF: learned reward model + critic network + per-step KL penalty."),
    "grpo_sparse_reachable": dict(kind="good", algo="grpo", kl=0.04, epochs=2, reward="score>=10",
                                  story="GRPO with a pass/fail reward (score >= 10). SFT passes 3% of the time: enough to learn from."),
    "dpo": dict(kind="good", algo="dpo", beta=0.1, dpo_epochs=2, lr=3e-4, pairs="ckpt/dpo_pairs.pt",
                story="DPO: no rollouts, no reward model: learn directly from best-vs-worst game pairs."),
    # bad
    "bad_grpo_no_kl_high_lr": dict(kind="bad", algo="grpo", kl=0.0, epochs=4, lr=3e-3, clip=10.0,
                                   story="GRPO with no KL, 30x LR, clipping loosened to 10, 4 epochs: the policy collapses.",
                                   symptom="entropy -> 0 and KL explodes within a few iterations"),
    "bad_grpo_sparse": dict(kind="bad", algo="grpo", kl=0.04, epochs=2, reward="score>=20",
                            story="GRPO with an unreachable pass/fail reward (score >= 20): no group ever has any signal.",
                            symptom="~100% of groups have all-equal rewards -> advantage 0 -> no gradient"),
    "bad_grpo_length_reward": dict(kind="bad", algo="grpo", kl=0.0, epochs=2, reward="length",
                                   story="GRPO rewarded for game length (think 'longer answer = better'): stalls.",
                                   symptom="length and reward up, score down, loop ratio and starvation up"),
    "bad_rlhf_no_kl": dict(kind="bad", algo="ppo", reward="rm", kl=0.0, epochs=4, lr=3e-4, iters=50, max_tokens=2048,
                           story="PPO-RLHF with no KL leash, 3x LR and 4 epochs: the policy over-optimises the reward model.",
                           symptom="reward-model score keeps rising while the true score falls"),
    "bad_dpo_noisy_pairs": dict(kind="bad", algo="dpo", beta=0.1, dpo_epochs=1, lr=1e-4, pairs="ckpt/prefs.pt",
                                story="DPO on noisy pairs (2 random games, 26% ties, 10% flipped labels): nothing to learn.",
                                symptom="DPO loss stuck at ln 2 = 0.693, pair accuracy ~50%"),
    "bad_dpo_overtrained": dict(kind="bad", algo="dpo", beta=0.01, dpo_epochs=8, lr=1e-3, n_pairs=600,
                                pairs="ckpt/dpo_pairs.pt",
                                story="DPO with tiny beta, ~3x LR, 8 epochs on 600 pairs: chosen answers get LESS likely (warning sign).",
                                symptom="chosen log-prob falls well below the reference (likelihood displacement); here play did not degrade yet"),
    "bad_no_sft": dict(kind="bad", algo="grpo", kl=0.0, epochs=2, init="random",
                       story="GRPO from a random model (no SFT): most answers aren't even moves.",
                       symptom="invalid-format rate near 100%, reward near zero for a long time"),
}
REPLAY_AT = [0.0, 0.25, 0.5, 1.0]


def reward_fn(name, e):
    if name == "score":
        return e["score"] - (1.0 if e["cause"] == "invalid" else 0.0)
    if name.startswith("score>="):
        return float(e["score"] >= int(name[7:]))
    if name == "length":
        return e["length"] / 50
    raise ValueError(name)


# --------------------------------------------------------------------------- losses
def k3(logp, ref_logp):
    """Unbiased, always-positive estimate of KL(pi || pi_ref) per token (Schulman's k3)."""
    d = ref_logp - logp
    return d.exp() - d - 1


def policy_gradient_step(policy, ref, opt, eps, adv_per_ep, cfg):
    """Shared by REINFORCE / RLOO / GRPO: token-level clipped surrogate + optional k3 KL to reference.

    One call = one batch of games, but several optimizer steps: up to max_tokens moves, shuffled,
    one step per minibatch of 1,024 moves (times `epochs`).
    """
    obs, acts = flatten(eps)
    adv = torch.as_tensor(np.concatenate([np.full(len(e["acts"]), a, np.float32) for e, a in zip(eps, adv_per_ep)]))
    old = torch.as_tensor(np.concatenate([e["logp"] for e in eps]), dtype=torch.float32)
    n = len(acts)
    stats = {"kl": [], "clipfrac": [], "grad_norm": []}
    for _ in range(cfg["epochs"]):
        idx = torch.randperm(n)[:cfg["max_tokens"]]
        for mb in torch.split(idx, 1024):
            logp = token_logp(policy, obs[mb], acts[mb])
            with torch.no_grad():
                ref_logp = token_logp(ref, obs[mb], acts[mb])
            ratio = (logp - old[mb]).exp()
            a = adv[mb]
            if cfg["algo"] == "grpo":
                pg = -torch.min(ratio * a, ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * a)
            else:  # plain policy gradient: grad of ratio * a equals grad of log pi * a at ratio 1. The ratio is
                # exactly 1 only on the first minibatch; later minibatches see a slightly updated policy.
                pg = -ratio * a
            kl = k3(logp, ref_logp)
            loss = (pg + cfg["kl"] * kl).mean()
            opt.zero_grad()
            loss.backward()
            stats["grad_norm"].append(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0).item())
            opt.step()
            stats["kl"].append(kl.mean().item())
            stats["clipfrac"].append(((ratio - 1).abs() > cfg["clip"]).float().mean().item())
    return {k: float(np.mean(v)) for k, v in stats.items()}


def advantages(cfg, R, group):
    G = R.reshape(-1, group)
    if cfg["algo"] == "reinforce":
        return (R - R.mean()), {}
    if cfg["algo"] == "rloo":
        loo = (G.sum(1, keepdims=True) - G) / (group - 1)
        return (G - loo).ravel(), {}
    std = G.std(1, keepdims=True)
    adv = ((G - G.mean(1, keepdims=True)) / (std + 1e-4)).ravel()
    return adv, {"zero_signal_groups": float((std.squeeze(1) < 1e-6).mean())}


def ppo_rlhf_step(policy, ref, critic, rm, opt, copt, eps, cfg, gamma=0.99, lam=0.95):
    """Classic RLHF PPO: per-step reward = RM(s, a) - beta * (log pi - log pi_ref); GAE from a critic."""
    obs, acts = flatten(eps)
    with torch.no_grad():
        r_rm = rm(rm_input(obs, acts))
        ref_logp = token_logp(ref, obs, acts)
        values = critic(obs)
    old = torch.as_tensor(np.concatenate([e["logp"] for e in eps]), dtype=torch.float32)
    r = r_rm - cfg["kl"] * (old - ref_logp)
    adv, ret, i = torch.zeros_like(r), torch.zeros_like(r), 0
    for e in eps:  # GAE per episode
        L, gae = len(e["acts"]), 0.0
        for t in reversed(range(i, i + L)):
            nxt = values[t + 1] if t + 1 < i + L else 0.0
            delta = r[t] + gamma * nxt - values[t]
            gae = delta + gamma * lam * gae
            adv[t] = gae
        ret[i:i + L] = adv[i:i + L] + values[i:i + L]
        i += L
    adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"kl": [], "clipfrac": [], "grad_norm": [], "value_loss": []}
    for _ in range(cfg["epochs"]):
        idx = torch.randperm(len(acts))[:cfg["max_tokens"]]
        for mb in torch.split(idx, 1024):
            logp = token_logp(policy, obs[mb], acts[mb])
            ratio = (logp - old[mb]).exp()
            a = adv_n[mb]
            loss = -torch.min(ratio * a, ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * a).mean()
            opt.zero_grad()
            loss.backward()
            stats["grad_norm"].append(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0).item())
            opt.step()
            vl = F.mse_loss(critic(obs[mb]), ret[mb])
            copt.zero_grad()
            vl.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            copt.step()
            stats["value_loss"].append(vl.item())
            stats["kl"].append(k3(logp.detach(), ref_logp[mb]).mean().item())
            stats["clipfrac"].append(((ratio - 1).abs() > cfg["clip"]).float().mean().item())
    ev = 1 - (ret - values).var() / (ret.var() + 1e-8)
    out = {k: float(np.mean(v)) for k, v in stats.items()}
    out["explained_var"] = float(ev)
    lens = [len(e["acts"]) for e in eps]
    out["rm_score"] = float(np.mean([s.sum().item() for s in torch.split(r_rm, lens)]))
    return out


def dpo_seq_logp(model, eps):
    obs, acts = flatten(eps)
    lp = token_logp(model, obs, acts)
    return torch.stack([s.sum() for s in torch.split(lp, [len(e["acts"]) for e in eps])])


# --------------------------------------------------------------------------- loop
def episode_stats(eps, rewards):
    causes = Counter(e["cause"] for e in eps)
    return {"score": np.mean([e["score"] for e in eps]), "reward": float(np.mean(rewards)),
            "length": np.mean([e["length"] for e in eps]),
            "entropy": np.mean(np.concatenate([e["ent"] for e in eps])),
            "loop_ratio": np.mean([e["loop_ratio"] for e in eps]),
            **{f"end_{c}": causes[c] / len(eps) for c in CAUSES}}


def record(policy, path, n=5):
    games = sorted(rollout(policy, range(900, 900 + n), greedy=False, record=True), key=lambda g: g["score"])
    g = games[n // 2]
    path.write_text(json.dumps({"size": 8, "score": g["score"], "cause": g["cause"], "frames": g["frames"],
                                "all_scores": [x["score"] for x in games]}))


def save_csv(rows, run):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(run / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def train(name, seed=0, out="runs"):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = {**BASE, **RECIPES[name]}
    run = Path(out) / name
    (run / "replays").mkdir(parents=True, exist_ok=True)

    policy, ref = TinyGPT(), TinyGPT()
    ref.load_state_dict(torch.load("ckpt/sft.pt"))
    if cfg["init"] == "sft":
        policy.load_state_dict(ref.state_dict())
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])
    rm = critic = copt = None
    if cfg["reward"] == "rm" or cfg["algo"] == "dpo":
        rm = TinyGPT(out=1)
        rm.load_state_dict(torch.load("ckpt/rm.pt"))
    if cfg["algo"] == "ppo":
        critic = TinyGPT(out=1)
        critic.load_state_dict(rm.state_dict())  # critic initialised from the reward model
        copt = torch.optim.Adam(critic.parameters(), lr=1e-4)

    rows, t0 = [], time.time()
    if cfg["algo"] == "dpo":
        rows = train_dpo(policy, ref, rm, opt, cfg, run)
    else:
        rng = np.random.default_rng(seed)
        marks = {round(f * (cfg["iters"] - 1)): f for f in REPLAY_AT}
        for it in range(cfg["iters"]):
            if it in marks:
                record(policy, run / "replays" / f"{round(100 * marks[it]):03d}.json")
            seeds = rng.integers(1_000_000, size=cfg["prompts"]).tolist()
            eps = rollout(policy, seeds, group=cfg["group"])
            if cfg["algo"] == "ppo":
                upd = ppo_rlhf_step(policy, ref, critic, rm, opt, copt, eps, cfg)
                R = np.full(len(eps), upd["rm_score"])
            else:
                R = np.array([reward_fn(cfg["reward"], e) for e in eps], dtype=np.float32)
                adv, extra = advantages(cfg, R, cfg["group"])
                upd = {**policy_gradient_step(policy, ref, opt, eps, adv, cfg), **extra}
            rows.append({"iter": it, "games": (it + 1) * len(eps), **episode_stats(eps, R), **upd})
            save_csv(rows, run)  # every iteration, so a run can be stopped early without losing it
            if it % 10 == 0:
                r = rows[-1]
                print(f"{name:24s} it {it:3d} score {r['score']:5.2f} reward {r['reward']:7.2f} "
                      f"kl {r['kl']:.3f} ent {r['entropy']:.3f} invalid {r['end_invalid']:.0%}", flush=True)

    save_csv(rows, run)
    last = [r["score"] for r in rows if "score" in r][-5:]
    msg = f"{name:24s} done {time.time() - t0:5.0f}s  final score {np.mean(last):.2f}"
    print(msg, flush=True)
    return msg


def train_dpo(policy, ref, rm, opt, cfg, run, batch=16):
    pairs = torch.load(cfg["pairs"], weights_only=False)
    pairs = pairs[:cfg.get("n_pairs", 1800)]
    rows, step = [], 0
    total = cfg["dpo_epochs"] * (len(pairs) // batch)
    evals = {round(f * total) for f in np.linspace(0, 1, 9)}
    marks = {round(f * total): f for f in REPLAY_AT}
    for ep in range(cfg["dpo_epochs"]):
        order = np.random.permutation(len(pairs))
        for i in range(0, len(pairs) - batch + 1, batch):
            if step in marks:
                record(policy, run / "replays" / f"{round(100 * marks[step]):03d}.json")
            ch = [pairs[j][0] for j in order[i:i + batch]]
            rj = [pairs[j][1] for j in order[i:i + batch]]
            pc, pr = dpo_seq_logp(policy, ch), dpo_seq_logp(policy, rj)
            with torch.no_grad():
                rc, rr = dpo_seq_logp(ref, ch), dpo_seq_logp(ref, rj)
            margin = cfg["beta"] * ((pc - rc) - (pr - rr))
            loss = -F.logsigmoid(margin).mean()
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0).item()
            opt.step()
            row = {"iter": step, "games": (step + 1) * batch * 2, "dpo_loss": loss.item(),
                   "pair_acc": (margin > 0).float().mean().item(), "margin": margin.mean().item(),
                   "chosen_logp_shift": (pc - rc).mean().item() / np.mean([len(e["acts"]) for e in ch]),
                   "rejected_logp_shift": (pr - rr).mean().item() / np.mean([len(e["acts"]) for e in rj]),
                   "grad_norm": gn}
            if step in evals:  # DPO never plays during training: evaluate by sampling games
                eps = rollout(policy, range(700, 764))
                R = traj_reward(rm, eps).detach().numpy()
                row.update(episode_stats(eps, R))
                print(f"dpo step {step}/{total} score {row['score']:.2f} acc {row['pair_acc']:.2f} "
                      f"chosen shift {row['chosen_logp_shift']:.3f}", flush=True)
            rows.append(row)
            if step in evals:
                save_csv(rows, run)
            step += 1
    record(policy, run / "replays" / "100.json")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("recipes", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    names = list(RECIPES) if args.all else args.recipes
    slow_first = sorted(names, key=lambda n: (RECIPES[n]["algo"] != "ppo", RECIPES[n]["algo"] != "dpo"))
    with Pool(min(args.jobs, len(names))) as pool:  # ~500 MB per worker: keep jobs low on 8 GB machines
        for _ in pool.imap_unordered(train, slow_first):
            pass
