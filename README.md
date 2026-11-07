# Snake RL: good recipes, bad recipes, and how to tell them apart

Small, CPU-friendly reinforcement-learning experiments on Snake, built to show what healthy and broken
training runs look like, and how to detect the difference automatically.

Two parts:

- **Classic RL** (this folder): Q-table, DQN and PPO on a 10×10 Snake with hand-made features.
- **[Snake as an LLM](llm/)**: a 156k-parameter transformer plays Snake by reading the board as a token
  prompt, trained with the LLM post-training stack: SFT → REINFORCE / RLOO / GRPO → preferences → DPO →
  reward model → PPO-RLHF, plus a failure gallery (collapse, reward hacking, sparse reward, likelihood displacement).

## Classic RL

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # ffmpeg optional, for .mp4
.venv/bin/python train.py --all --jobs 8   # ~2.5 min on 8 cores -> runs/<recipe>/
.venv/bin/python detect.py                 # automatic health check, prints alarms
.venv/bin/python plots.py                  # charts/  (dashboards + comparisons)
.venv/bin/python render.py                 # videos/  (GIF per checkpoint + progression .mp4)
```

| file | what it is |
|---|---|
| `snake_env.py` | 10×10 Snake, 11-feature observation, 3 relative actions, **reward functions** |
| `agents.py` | Q-table, DQN, PPO behind one interface |
| `recipes.py` | every experiment: algorithm + reward + hyperparameters + the "story" |
| `train.py` | trains, logs one CSV row per episode, records greedy replays at 0/5/15/40/100% |
| `detect.py` | rule-based diagnostics, the "how to detect bad behaviour" part |
| `plots.py` / `render.py` | charts and game clips for the video |

## Results (seed 0)

| recipe | algo | final score | detector says |
|---|---|---|---|
| qtable | Q-table | ~16 | healthy |
| dqn | DQN | ~16 | healthy |
| dqn_shaped | DQN + symmetric distance hint | ~16 | healthy |
| ppo | PPO (300k steps) | ~12–14 | healthy |
| bad_survival_reward | DQN | 0.2 | REWARD HACKING, LOOPING, STALLING |
| bad_step_penalty | DQN | 0.5 | SUICIDE |
| bad_lr_too_high | DQN | 0.6 | Q DIVERGENCE, GRADIENT SPIKES |
| bad_myopic (γ=0) | DQN | 11.5 | UNDERPERFORMING vs baseline |
| bad_entropy_collapse | PPO | ~0.1 | ENTROPY COLLAPSE, ONE-ACTION POLICY, LOOPING |

## Honest notes worth mentioning on camera
- **ε = 0 (no exploration) did *not* break DQN here.** With such informative features and death penalties,
  greedy actions from a random network explore enough. Exploration failures need harder, sparser tasks.
- `dqn_shaped` uses a *symmetric* ±0.1 distance hint, so moving toward and then away from the food nets
  zero and loops don't pay. An asymmetric hint (reward for getting closer, nothing for moving away) is a
  classic way to invite loop-hacking. It isn't tested here, but it would make a good extra experiment.
- PPO results vary more across seeds than DQN. Try `--seed 1`, `--seed 2` and show the spread.
- Replays are the **median of 5 greedy games** at each checkpoint, not the best one.

## What's in the repo

- Code, per-run metrics (`runs/*/metrics.csv`, `llm/runs/*/metrics.csv`) and the charts made from them.
- Replay JSONs (`runs/*/replays/`), so `render.py` can make the clips without retraining.
- Not included (regenerate with the commands above): model checkpoints (`llm/ckpt/`) and GIF/MP4 clips.
