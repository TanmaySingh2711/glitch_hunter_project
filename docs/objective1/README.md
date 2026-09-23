# Objective 1 — finish Level 1-1

The first objective: train a PPO agent that plays the Mario clone and
**completes Level 1-1**. It ran for 6,000,000 steps under the completion
reward (`rewards/legacy.py`, `REWARD_MODE = "legacy_completion"`) and produced
the project's first brain.

| | |
|---|---|
| Brain | `mario_brain_checkpoint.zip` at the project root - tracked in git, so a fresh clone can watch it play |
| SHA-256 | `690d57022c1fb444454b0b47f9d1e4ff1bc1444d7110c0bc3320b7fe53a188b3` (`evaluation.completion.SIX_M_SHA256`) |
| Steps | 6,000,000 (`train_agent.TOTAL_TIMESTEPS_LEGACY`) |
| Completion (500-episode protocol) | **234 / 500 = 46.8%** (95% CI 42.5-51.2%), mean progress 0.71 |
| Endings | 234 level complete, 209 deaths (goomba 151, koopa 31, pit 27), 57 time-outs |
| Greedy run | dies after 185 steps (the 16M main brain's greedy run finishes the level) |

## Why it still matters

* It is the **baseline** every later brain is measured against: the frozen
  500-episode protocol and thresholds in `evaluation/completion_baseline_6M.json`
  were recorded from it, and `config.BASELINE_MODEL` points at it. A brain is
  HEALTHY only if it keeps at least this ability to finish the level.
* Objective 2 started from it (the QA exploration phase seeded its weights).
* Its reward is preserved bit for bit (`tests/test_reward_wrapper.py`), so
  `REWARD_MODE = "legacy_completion"` still reproduces it.

It has exactly one copy on disk; git is its backup
(`git checkout -- mario_brain_checkpoint.zip`). The three byte-identical copies
it used to have in `backup_6M/` and `checkpoints/` were removed on 2026-09-24.

Next: [Objective 2](../objective2/WORKLOG.md) (explore the whole level, keep
finishing it) and [Objective 3](../objective3/README.md) (bug evidence and
reports).
