"""Minimal Snake environment with pluggable reward functions.

Observation (11 binary features, the classic compact Snake encoding):
    danger straight / right / left            (3)
    moving up / right / down / left           (4)
    food up / right / down / left of head     (4)

Actions are relative to the current heading:
    0 = straight, 1 = turn right, 2 = turn left
"""
import numpy as np

DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left (x, y)


class SnakeEnv:
    def __init__(self, size=10, reward="standard", hunger_limit=100, max_steps=1000, seed=0):
        self.size = size
        self.reward_name = reward
        self.reward_fn = REWARDS[reward]
        self.hunger_limit = hunger_limit  # steps without food before starving (None = never)
        self.max_steps = max_steps        # hard episode cap ("timeout")
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        c = self.size // 2
        self.dir = 1
        self.body = [(c, c), (c - 1, c), (c - 2, c)]  # head first
        self.steps = 0
        self.hunger = 0
        self.score = 0
        self.visited = {self.body[0]}  # cells visited since last food (loop detector)
        self.revisits = 0
        self._place_food()
        return self.obs()

    def _place_food(self):
        occupied = set(self.body)
        free = [(x, y) for x in range(self.size) for y in range(self.size) if (x, y) not in occupied]
        self.food = free[self.rng.integers(len(free))]

    def _collides(self, p):
        x, y = p
        return not (0 <= x < self.size and 0 <= y < self.size) or p in self.body[:-1]

    def obs(self):
        hx, hy = self.body[0]
        d = self.dir
        def danger(rel):
            dx, dy = DIRS[(d + rel) % 4]
            return self._collides((hx + dx, hy + dy))
        fx, fy = self.food
        return np.array([
            danger(0), danger(1), danger(-1),
            d == 0, d == 1, d == 2, d == 3,
            fy < hy, fx > hx, fy > hy, fx < hx,
        ], dtype=np.float32)

    def step(self, action):
        self.dir = (self.dir + (0, 1, -1)[action]) % 4
        dx, dy = DIRS[self.dir]
        hx, hy = self.body[0]
        new = (hx + dx, hy + dy)
        old_dist = abs(hx - self.food[0]) + abs(hy - self.food[1])
        self.steps += 1
        self.hunger += 1

        event, cause = "move", None
        if self._collides(new):
            event = "death"
            cause = "self" if new in self.body else "wall"
        else:
            self.body.insert(0, new)
            if new == self.food:
                event = "food"
                self.score += 1
                self.hunger = 0
                self.visited = {new}
                if len(self.body) == self.size * self.size:
                    cause = "win"
                else:
                    self._place_food()
            else:
                self.body.pop()
                if new in self.visited:
                    self.revisits += 1
                self.visited.add(new)
            if event != "death" and cause is None:
                if self.hunger_limit and self.hunger >= self.hunger_limit * len(self.body) // 3:
                    event, cause = "death", "starved"
                elif self.steps >= self.max_steps:
                    cause = "timeout"

        new_dist = abs(self.body[0][0] - self.food[0]) + abs(self.body[0][1] - self.food[1])
        r = self.reward_fn(event, old_dist, new_dist)
        done = event == "death" or cause in ("timeout", "win")
        info = {"event": event, "cause": cause, "score": self.score, "steps": self.steps,
                "loop_ratio": self.revisits / self.steps}
        return self.obs(), r, done, info

    def snapshot(self):
        """Compact frame for replays: body cells and food cell."""
        return {"body": [list(p) for p in self.body], "food": list(self.food)}


# ---------------------------------------------------------------- reward recipes
# Each returns a scalar given the step event and Manhattan distance to food
# before/after the move. These are where most "bad behaviour" comes from.

def r_standard(event, d0, d1):
    return {"food": 10.0, "death": -10.0}.get(event, 0.0)

def r_shaped(event, d0, d1):
    # Small hint toward food. Symmetric (+/-) so loops don't pay off.
    base = r_standard(event, d0, d1)
    return base + (0.1 if d1 < d0 else -0.1) if event == "move" else base

def r_survival(event, d0, d1):
    # BAD: we only pay for staying alive. Circling forever is the optimal policy.
    return {"food": 0.0, "death": -10.0}.get(event, 1.0)

def r_step_penalty(event, d0, d1):
    # BAD: harsh per-step cost with a cheap death. Dying fast is the best deal.
    return {"food": 5.0, "death": -1.0}.get(event, -1.0)

REWARDS = {
    "standard": r_standard,
    "shaped": r_shaped,
    "survival": r_survival,
    "step_penalty": r_step_penalty,
}
