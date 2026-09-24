# Objective 1 — Autonomous Exploration Agent

A PPO agent that learns to play the Mario Level 1-1 clone on its own and
finish the level. Its 6M brain is the baseline for everything that follows.

## Start

1. Prepared the Mario game for reinforcement learning.
2. Wrapped the game with Gymnasium.
3. Defined the action space and visual observations.
4. Initialized a PPO agent with CnnPolicy.
5. Started training with 8 parallel workers.

## During

6. The agent learned movement, jumping, navigation, and autonomous exploration.
7. Added checkpoint/resume support and basic dashboard/runtime controls.
8. Continued training until the policy became stable and capable.

## End

9. Reached a strong baseline at around **6 million training steps**.
10. The agent could navigate and complete the level autonomously.
11. This **6M brain became the starting point for Objective 2**.

**Flow:** Game → RL setup → PPO training → **6M autonomous brain**

## Key results

| | |
|---|---|
| Brain | `mario_brain_checkpoint.zip` (project root, tracked in git so a fresh clone can watch it play) |
| SHA-256 | `690d57022c1fb444454b0b47f9d1e4ff1bc1444d7110c0bc3320b7fe53a188b3` |
| Training steps | 6,000,000 (`train_agent.TOTAL_TIMESTEPS_LEGACY`) |
| Completion (500-episode protocol) | **234 / 500 = 46.8%** (95% CI 42.5–51.2%) |
| Mean progress | 0.71 |
| Endings | 234 level complete · 209 deaths (goomba 151, koopa 31, pit 27) · 57 time-outs |
| Greedy (deterministic) run | dies after 185 steps |

## How it works

* **Game:** a Python/Pygame Super Mario Bros clone (now `mario_clean/`),
  driven one 60 fps frame per `step()` by `custom_mario_env.CustomMarioEnv`.
* **Actions:** 10 discrete key combinations (walk, run, jump, crouch, left/right).
* **Observation:** 4 stacked 84×84 grayscale frames; the agent acts every 4
  engine frames (`wrap_observation`).
* **Learning:** Stable-Baselines3 PPO, `CnnPolicy`, 8 `SubprocVecEnv` workers,
  torch 2.5.1 (CUDA 12.1).
* **Reward** (`rewards/legacy.py`, `REWARD_MODE = "legacy_completion"`): new
  tiles, progress milestones, +500 for the flag; small death penalty.
* **Episode ends:** engine timer, death, a stuck rule, or a step limit.
* **Resume:** matched checkpoint files, so training can stop and continue.

## Key files

| Path | What |
|---|---|
| `train_agent.py` | training entry point (legacy and, later, QA mode) |
| `custom_mario_env.py`, `agent_logic.py` | the Gymnasium env and the reward wrapper |
| `rewards/legacy.py` | the completion reward, preserved bit for bit (`tests/test_reward_wrapper.py`) |
| `evaluation/completion_baseline_6M.json` | the frozen 500-episode protocol and thresholds, recorded from this brain |
| `app.py`, `dashboard_*.py` | the live dashboard that watches the agent play |

## Why it still matters

* **Baseline for every later brain:** a brain is HEALTHY only if it keeps at
  least this ability to finish the level (`config.BASELINE_MODEL`).
* **Seed for Objective 2:** the QA phase started from these weights.
* **Reproducible:** `REWARD_MODE = "legacy_completion"` still trains the same objective.

## Problems and fixes

* The clone's jump needed the key released between jumps; an RL agent holds
  keys, so the env forces `allow_jump` each frame (documented in the env).
* Parallel windows stacked on top of each other: window placement fixed.
* The game loaded resources relative to the working directory: paths made
  absolute so workers can start from anywhere.

## Validation

* 500-episode completion protocol with fixed seeds (the numbers above).
* Unit tests pin the legacy reward, the observation chain and the action map.

## Limitations

* 46.8% completion, not near 100%: most failures are Goomba deaths.
* The greedy run does not finish the level; the numbers come from sampled play.
* Only one copy on disk; git is its backup
  (`git checkout -- mario_brain_checkpoint.zip`). The extra copies in
  `backup_6M/` and `checkpoints/` were removed on 2026-09-24.

## Complete project flow

**Objective 1:** Game → RL setup → PPO training → **6M autonomous brain**

**Objective 2:** 6M autonomous brain → QA exploration + glitch-hunting system → training + audits/recovery → **16M final QA brain**

**Objective 3:** 16M QA brain → automated bug evidence/reporting pipeline → benchmark bug detection → **complete autonomous QA reporting system**

[Objective 1](OBJECTIVE1.md) · [Objective 2](OBJECTIVE2.md) · [Objective 3](OBJECTIVE3.md)
