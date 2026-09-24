# Objective 2 — QA / Glitch-Hunting Agent

Turn the 6M brain into a QA explorer that covers as much of Level 1-1 as it
can reach while keeping its ability to finish the level. **Closed 2026-09-21.**

## Start

1. Loaded the **6M autonomous brain** from Objective 1.
2. Added spatial coverage tracking and testable/reachable-space logic.
3. Added QA-specific rewards and physics/anomaly detection.
4. Added persistent coverage, stuck/loop handling, and the **EXPLORE → COMPLETE** lifecycle.

## During

5. Continued QA training beyond 6M.
6. Performed hard audits for reward farming, mode collapse, bad checkpoints, PPO instability, and coverage issues.
7. Detected a major regression around 6.4M and recovered from the last healthy checkpoint.
8. Improved reward logic, resume safety, telemetry, anomaly handling, and completion-retention evaluation.
9. Repeated controlled training and validation until the agent became stable again.

## End

10. Selected the **16M QA brain** as the final Objective-2 brain.
11. Final results: **56.6% completion rate**, **0.75 average progress**, **84.25% accepted practical spatial coverage**.
12. Froze the final brain, matching coverage state, hashes, and documentation.
13. Objective 2 was formally closed.

**Flow:** 6M brain → QA coverage/reward/anomaly system → QA training + audits/recovery → **16M final QA brain**

## Key results

| | |
|---|---|
| Final brain | `glitch_hunter_main_brain.zip` (root, read-only), 16,000,000 steps, SHA-256 `d906d09e…` |
| Coverage state | `glitch_hunter_main_brain_coverage.npz`, SHA-256 `aa384707…` |
| Coverage | 3,166,235 / 3,757,990 testable px = **84.25%** (not 100%) |
| Completion (500 episodes) | **283/500 = 56.6%** (CI 52.2–60.9%) vs 46.8% at 6M → **HEALTHY** |
| Mean progress | **0.750** vs 0.710; the greedy run finishes the level in 434 steps |
| Backup + evidence | `checkpoints_qa/final_objective2_16000000/` (backup copy, `FINAL_OBJECTIVE2.json`, logs, telemetry) |

## How it works

* **Coverage:** a 1-px world bitmap, shared by the 8 workers, saved with every
  checkpoint and never reset. **Testable mask** = space Mario can physically
  reach (3 reachability methods + real jump arcs, both Mario sizes).
* **Reward** (`rewards/qa.py`): new testable pixels in EXPLORE, then progress
  in COMPLETE; interaction/locomotion bonuses cut off when discovery dries up.
* **Lifecycle:** EXPLORE → COMPLETE only on evidence (target met, exhausted
  windows, nothing reachable ahead) — never on time alone.
* **Anchor consolidation:** after each update, KL to a frozen healthy policy is
  measured on fixed anchor states and pulled back if it exceeds 0.08. This is
  what kept the level-finishing skill alive.
* **Safety:** health-aware resume (a REGRESSED checkpoint is never resumed),
  matched model/coverage pairs with fingerprints, 500-episode retention test.

## Problems found and fixed

| Problem | Cause | Fix |
|---|---|---|
| Completion collapsed at 6.4M (43.8% → 6.0%) | discount horizon (γ 0.99 ≈ 100 steps) hides the finish | anchor consolidation; resumed from the healthy checkpoint |
| Reward farming (jumping at ? blocks) | secondary reward never ran out | drought gate, hard cut-off |
| Wrong denominator (twice) | mask counted unreachable space / flagpole trigger | real-arc mask; flag-trigger corridor; lossless migration |
| Stuck at tall pipes | 172 px walls need a run-up (standing jump 166 px, running 183 px) | run-up/retreat shaping |
| Anomaly counter mostly noise | brick changes, sweep artifacts, pit rule | reclassified at mask build, tolerance kept |
| Silent resume hazards, launcher config bug | wrong checkpoint / parent-only config | explicit `--resume-from`, dry-run, per-worker config |

**Tried and rejected:** reward re-weighting, completion rehearsal (30.2%, worse),
KL budget 0.13, run-up reward alone, stagnation escalation.

**Engine anomaly found:** acceleration state leaks from the ground into the air
(turnaround/sprint settings survive a jump), letting Mario exceed the walk
speed cap in mid-air. Reproducible, not fixed (the game is read-only).
Scripts: `tools/objective2_evidence/`.

## Why training stopped at 16M

Gains had become tiny (+0.27 pt in the last 0.8M steps). 85% of the remaining
591,755 px are high open air (y < 300) that Mario rarely uses; near solids
coverage is 92.14%. The user chose this as the practical stopping point.

## Resume command (inspect only)

```
venv_gpu\Scripts\python.exe train_agent.py --dry-run-resume --resume-from glitch_hunter_main_brain.zip
```

A real QA launch still needs `--safety-cap-timesteps N` or `--unrestricted`.

## Limitations

* 84.25% is a practical stopping point, not a proven ceiling (strict bound
  95.31%); uncovered space is not proof of "no bugs there".
* 56.6% is one 500-episode measurement; 25% of episodes still end in death.
* 21,473 px flagged "anomalous" were never triaged one by one.
* Artifacts are git-ignored; git tracks only their hashes (`artifacts.json`).

## Complete project flow

**Objective 1:** Game → RL setup → PPO training → **6M autonomous brain**

**Objective 2:** 6M autonomous brain → QA exploration + glitch-hunting system → training + audits/recovery → **16M final QA brain**

**Objective 3:** 16M QA brain → automated bug evidence/reporting pipeline → benchmark bug detection → **complete autonomous QA reporting system**

[Objective 1](OBJECTIVE1.md) · [Objective 2](OBJECTIVE2.md) · [Objective 3](OBJECTIVE3.md)
