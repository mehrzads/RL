# Snake as an LLM: SFT → REINFORCE / RLOO / GRPO / PPO-RLHF / DPO

A 156k-parameter decoder-only transformer reads the board as a **prompt of tokens** and answers
with **one token**, its next move. It samples from the full vocabulary, so it can also answer with
garbage (a format error), exactly like an LLM.

```
prompt:  <bos>  . . . . . . . .  ← 8×8 board, one token per cell
                . . o o H . * .     (H head, o body, t tail, * food)
                ...
         >  <sep>                 ← current heading
answer:  U | R | D | L            ← anything else = format error, game over
```

## Pipeline

```bash
cd llm
../.venv/bin/python sft.py            # stage 1: imitate a noisy expert       (~5 min)
../.venv/bin/python prefs.py          # stage 2: preference pairs + reward model (~5 min)
../.venv/bin/python train.py --all    # stage 3: every RL recipe, 4 in parallel (~2 h on an M2 8 GB)
../.venv/bin/python report.py         # health check + charts/
../.venv/bin/python ../render.py --runs runs --out videos   # game clips (needs runs/*/replays)
```

Keep `--jobs` at 4 on an 8 GB Mac. Each worker needs ~500 MB, and swapping makes everything 5× slower.
MPS (the Mac GPU) crashes with this PyTorch build, so everything runs on CPU.

| file | what it is |
|---|---|
| `common.py` | tokenizer, TinyGPT, batched rollouts (games sharing a seed = one GRPO "prompt group") |
| `sft.py` | Q-table expert + 10% random moves → 154k (prompt, answer) pairs → next-token cross-entropy |
| `prefs.py` | game pairs labelled by score; Bradley-Terry reward model (initialised from SFT, like InstructGPT) |
| `train.py` | all recipes: REINFORCE, RLOO, GRPO, PPO-RLHF (critic initialised from RM, per-token KL in reward), DPO |
| `report.py` | automatic diagnosis + dashboards |

## Results (seed 0, start = SFT model, sampled score ≈ 4.4)

Reward for REINFORCE / RLOO / GRPO: apples eaten, minus 1 if the game ends on a format error (a non-move token).

| recipe | final score | detector |
|---|---|---|
| reinforce | 9.2 | healthy |
| rloo | 9.5 | healthy |
| grpo | 9.4 | healthy |
| grpo_sparse_reachable (pass/fail, score ≥ 10) | 9.5 | healthy |
| ppo_rlhf (learned RM, KL β=0.2) | 7.0 | healthy |
| dpo (best-vs-worst pairs) | 6.6 | healthy |
| bad_grpo_no_kl_high_lr | 0.05 | KL blow-up, entropy collapse, zero-signal groups |
| bad_grpo_sparse (score ≥ 20, never reached) | ~SFT | no learning signal |
| bad_grpo_length_reward | 10.2 peak → 5.4 | reward hacking, stalling (loops until timeout) |
| bad_rlhf_no_kl | 9.0 peak → 2.3 | reward hacking (RM score 12), KL blow-up, stalling |
| bad_dpo_noisy_pairs | ~SFT | DPO loss stuck at ln 2 |
| bad_dpo_overtrained | 6.7 | likelihood displacement (warning sign; play didn't degrade *yet*) |
| bad_no_sft (GRPO from random init) | 0 | learns the format, never the game |

Expert it imitated: 7.0. REINFORCE, RLOO and GRPO beat their teacher; PPO-RLHF (learned reward) matched it.

## Failure gallery, in one line each

- *No KL + 30× LR + clipping loosened to 10 + 4 epochs*: collapses in 3 iterations (entropy → 0, KL blow-up, every game hits the wall).
- *Reward-model over-optimisation* (`charts/rlhf_overoptimisation.png`): RM score 1.5 → 12 while the true score falls;
  the policy exploits the RM's length bias. That run also used 3× the LR and 4 epochs, so KL is not the only
  difference from the healthy run (β = 0.2, LR 1e-4, 2 epochs).
- *Length reward*: the verbosity analogue; eat just enough not to starve, then run out the clock.
- *Sparse reward*: score ≥ 10 works (SFT passes 3% of the time); score ≥ 20 is never reached, so GRPO does nothing.
- *No SFT*: GRPO from random weights learns the format but never the game.
- *DPO on noisy pairs*: loss stays at ln 2 = 0.693.
- *Over-trained DPO*: chosen games get less likely than under the reference (likelihood displacement).

## Honest notes
- Each recipe was run once (seed 0). Scores swing ±1 between iterations, so don't over-read small gaps,
  e.g. REINFORCE vs RLOO vs GRPO. On a problem this small, all three work.
- Several "bad" recipes surprised me, and the outline uses those surprises as lessons:
  the first sparse threshold worked, the length reward only hacked after ~50 iterations, and the
  over-trained DPO didn't degrade play.
- The first `ppo_rlhf` attempt with β=0.05 also drifted (true score peaked, then fell). β=0.2 was needed.
