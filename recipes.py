"""Training recipes. 'good' ones should learn; 'bad' ones each break in one instructive way."""

RECIPES = {
    # ------------------------------------------------------------------ good
    "qtable": dict(
        kind="good", agent="qtable", steps=150_000,
        env=dict(reward="standard"),
        hp=dict(lr=0.1, gamma=0.9, eps_decay=0.99),
        story="Tabular Q-learning: 2048 states, a lookup table, learns in seconds.",
    ),
    "dqn": dict(
        kind="good", agent="dqn", steps=150_000,
        env=dict(reward="standard"),
        hp=dict(lr=1e-3, gamma=0.9),
        story="DQN: neural net Q-function + replay buffer + target network.",
    ),
    "dqn_shaped": dict(
        kind="good", agent="dqn", steps=150_000,
        env=dict(reward="shaped"),
        hp=dict(lr=1e-3, gamma=0.9),
        story="DQN with a small symmetric 'move toward food' hint: faster early learning.",
    ),
    "ppo": dict(
        kind="good", agent="ppo", steps=300_000,
        env=dict(reward="standard"),
        hp=dict(lr=1e-3, ent_coef=0.01, epochs=10),
        story="PPO: on-policy policy gradient with clipping. Slower per step, very stable.",
    ),

    # ------------------------------------------------------------------- bad
    "bad_survival_reward": dict(
        kind="bad", agent="dqn", steps=150_000,
        env=dict(reward="survival", hunger_limit=None, max_steps=500),
        hp=dict(lr=1e-3, gamma=0.95),
        story="REWARD MISSPECIFICATION: +1 per step alive, nothing for food, no hunger -> snake circles forever.",
        symptom="episode length pinned at the cap, score flat, loop_ratio near 1, deaths = timeout",
    ),
    "bad_step_penalty": dict(
        kind="bad", agent="dqn", steps=150_000,
        env=dict(reward="step_penalty"),
        hp=dict(lr=1e-3, gamma=0.9),
        story="SUICIDAL AGENT: -1 per step but only -1 for dying -> fastest exit is the wall.",
        symptom="episode length collapses to a few steps, return rises while score falls, deaths = wall",
    ),
    "bad_lr_too_high": dict(
        kind="bad", agent="dqn", steps=150_000,
        env=dict(reward="standard"),
        hp=dict(lr=3e-2, gamma=0.99, target_every=None, grad_clip=None),
        story="DIVERGENCE: huge learning rate, no target net, no grad clipping -> Q-values blow up.",
        symptom="q_mean far outside the possible return range, loss/grad_norm spikes, score never improves",
    ),
    "bad_myopic": dict(
        kind="bad", agent="dqn", steps=150_000,
        env=dict(reward="standard"),
        hp=dict(lr=1e-3, gamma=0.0),
        story="MYOPIA: gamma = 0 -> greedy for the next bite, never plans an escape route, traps itself.",
        symptom="looks OK-ish (score plateaus lower), but death causes shift to 'self' as the snake grows",
    ),
    "bad_entropy_collapse": dict(
        kind="bad", agent="ppo", steps=300_000,
        env=dict(reward="standard"),
        hp=dict(lr=1e-2, ent_coef=0.0, epochs=30, clip=1.0),
        story="PREMATURE CONVERGENCE: aggressive PPO updates, no entropy bonus -> policy goes deterministic too early.",
        symptom="policy entropy -> 0 in the first updates, approx_kl and clipfrac high, score plateaus",
    ),
}
