"""Three RL recipes behind one tiny interface.

    act(obs)        -> action             (exploratory, used for training)
    greedy(obs)     -> action             (used for eval replays)
    observe(o, a, r, o2, done) -> dict    (training stats from any update that ran)
    end_episode()                          (schedules, e.g. epsilon decay)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def bits_to_index(obs):
    return int(obs.astype(np.int64) @ (1 << np.arange(len(obs))))


class QTable:
    """Tabular Q-learning. 11 binary features -> 2048 discrete states."""

    def __init__(self, n_obs=11, n_act=3, lr=0.1, gamma=0.9, eps_start=1.0, eps_end=0.01,
                 eps_decay=0.995, seed=0, **_):
        self.q = np.zeros((2 ** n_obs, n_act))
        self.lr, self.gamma = lr, gamma
        self.eps, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.rng = np.random.default_rng(seed)
        self.n_act = n_act

    def greedy(self, obs):
        return int(np.argmax(self.q[bits_to_index(obs)]))

    def act(self, obs):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_act))
        return self.greedy(obs)

    def observe(self, o, a, r, o2, done):
        s, s2 = bits_to_index(o), bits_to_index(o2)
        target = r + (0.0 if done else self.gamma * self.q[s2].max())
        td = target - self.q[s, a]
        self.q[s, a] += self.lr * td
        return {"td_error": abs(td), "q_mean": self.q[s].mean(), "q_max": self.q[s].max()}

    def end_episode(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        return {"epsilon": self.eps}


def mlp(n_in, n_out, hidden=128):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, n_out))


class DQN:
    """Deep Q-Network: replay buffer + target network + epsilon-greedy."""

    def __init__(self, n_obs=11, n_act=3, lr=1e-3, gamma=0.9, eps_start=1.0, eps_end=0.01,
                 eps_decay=0.99, buffer=50_000, batch=64, warmup=1000, target_every=500,
                 grad_clip=10.0, seed=0, **_):
        torch.manual_seed(seed)
        self.net, self.tgt = mlp(n_obs, n_act), mlp(n_obs, n_act)
        self.tgt.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.batch, self.warmup = gamma, batch, warmup
        self.target_every, self.grad_clip = target_every, grad_clip  # target_every=None -> no target net
        self.eps, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.n_act = n_act
        self.rng = np.random.default_rng(seed)
        self.buf_o = np.zeros((buffer, n_obs), np.float32)
        self.buf_o2 = np.zeros((buffer, n_obs), np.float32)
        self.buf_a = np.zeros(buffer, np.int64)
        self.buf_r = np.zeros(buffer, np.float32)
        self.buf_d = np.zeros(buffer, np.float32)
        self.size, self.ptr, self.cap, self.t = 0, 0, buffer, 0

    @torch.no_grad()
    def greedy(self, obs):
        return int(self.net(torch.as_tensor(obs)).argmax())

    def act(self, obs):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_act))
        return self.greedy(obs)

    def observe(self, o, a, r, o2, done):
        i = self.ptr
        self.buf_o[i], self.buf_a[i], self.buf_r[i], self.buf_o2[i], self.buf_d[i] = o, a, r, o2, done
        self.ptr, self.size, self.t = (i + 1) % self.cap, min(self.size + 1, self.cap), self.t + 1
        if self.size < self.warmup:
            return {}

        idx = self.rng.integers(self.size, size=self.batch)
        o, a, r, o2, d = (torch.as_tensor(x[idx]) for x in
                          (self.buf_o, self.buf_a, self.buf_r, self.buf_o2, self.buf_d))
        q = self.net(o).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            boot = (self.tgt if self.target_every else self.net)(o2).max(1).values
            target = r + self.gamma * (1 - d) * boot
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        gn = nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip or float("inf"))
        self.opt.step()
        if self.target_every and self.t % self.target_every == 0:
            self.tgt.load_state_dict(self.net.state_dict())
        return {"loss": loss.item(), "q_mean": q.mean().item(), "q_max": q.max().item(),
                "grad_norm": gn.item()}

    def end_episode(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        return {"epsilon": self.eps}


class ActorCritic(nn.Module):
    def __init__(self, n_obs, n_act, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_obs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi, self.v = nn.Linear(hidden, n_act), nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return torch.distributions.Categorical(logits=self.pi(h)), self.v(h).squeeze(-1)


class PPO:
    """Proximal Policy Optimization with GAE. Updates every `rollout` env steps."""

    def __init__(self, n_obs=11, n_act=3, lr=3e-4, gamma=0.95, lam=0.95, clip=0.2, ent_coef=0.01,
                 vf_coef=0.5, rollout=2048, epochs=4, minibatch=256, seed=0, **_):
        torch.manual_seed(seed)
        self.ac = ActorCritic(n_obs, n_act)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.rollout, self.epochs, self.minibatch = rollout, epochs, minibatch
        self.traj = []
        self._last = None  # (logp, value) of the most recent act()

    @torch.no_grad()
    def greedy(self, obs):
        dist, _ = self.ac(torch.as_tensor(obs))
        return int(dist.probs.argmax())

    @torch.no_grad()
    def act(self, obs):
        dist, v = self.ac(torch.as_tensor(obs))
        a = dist.sample()
        self._last = (dist.log_prob(a).item(), v.item())
        return int(a)

    def observe(self, o, a, r, o2, done):
        self.traj.append((o, a, r, done, *self._last))
        if len(self.traj) < self.rollout:
            return {}
        with torch.no_grad():
            _, last_v = self.ac(torch.as_tensor(o2))
        stats = self._update(last_v.item() * (1 - done))
        self.traj = []
        return stats

    def _update(self, last_v):
        o, a, r, d, logp_old, v = (np.array(x) for x in zip(*self.traj))
        adv = np.zeros_like(r, dtype=np.float32)
        gae, nxt = 0.0, last_v
        for t in reversed(range(len(r))):
            nonterm = 1.0 - d[t]
            delta = r[t] + self.gamma * nxt * nonterm - v[t]
            gae = delta + self.gamma * self.lam * nonterm * gae
            adv[t], nxt = gae, v[t]
        ret = adv + v
        o, a = torch.as_tensor(o, dtype=torch.float32), torch.as_tensor(a)
        logp_old, adv_t = torch.as_tensor(logp_old, dtype=torch.float32), torch.as_tensor(adv)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)

        log = {"entropy": [], "approx_kl": [], "clipfrac": [], "value_loss": []}
        n = len(r)
        for _ in range(self.epochs):
            for idx in np.array_split(np.random.permutation(n), max(1, n // self.minibatch)):
                dist, val = self.ac(o[idx])
                logp = dist.log_prob(a[idx])
                ratio = (logp - logp_old[idx]).exp()
                ad = adv_t[idx]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                pg = -torch.min(ratio * ad, ratio.clamp(1 - self.clip, 1 + self.clip) * ad).mean()
                vl = F.mse_loss(val, ret_t[idx])
                ent = dist.entropy().mean()
                loss = pg + self.vf_coef * vl - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.opt.step()
                with torch.no_grad():
                    log["entropy"].append(ent.item())
                    log["approx_kl"].append(((ratio - 1) - (logp - logp_old[idx])).mean().item())
                    log["clipfrac"].append(((ratio - 1).abs() > self.clip).float().mean().item())
                    log["value_loss"].append(vl.item())
        explained = 1 - np.var(ret - v) / (np.var(ret) + 1e-8)
        return {**{k: float(np.mean(x)) for k, x in log.items()}, "explained_var": float(explained)}

    def end_episode(self):
        return {}


AGENTS = {"qtable": QTable, "dqn": DQN, "ppo": PPO}
