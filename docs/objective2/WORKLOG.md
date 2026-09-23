# Objective 2 — working log and handoff

Read this first after any context compaction or session restart. It holds the
user's standing brief for Objective 2, every finding so far, what has been
changed, and exactly where to resume. Keep it current as work progresses.

## FINAL STATE - Objective 2 is CLOSED (2026-09-21). Read this block first.

Development and training are finished. Nothing below this block is a to-do; the
older sections are the evidence trail and some of their resume commands are
**superseded**.

**Final approved artifacts (frozen, read-only, hashes in `artifacts.json`):**

| | path | SHA-256 | fact |
|---|---|---|---|
| **Final brain (THE MAIN BRAIN)** | `glitch_hunter_main_brain.zip` (project root) | `d906d09e11002915c6de39ea83afc3670f02ba48bd769e21c3f3a17b0cc67899` | exactly 16,000,000 steps |
| Its coverage state | `glitch_hunter_main_brain_coverage.npz` (project root) | `aa384707881754508fbc5f5d4c9af385720cb39186a5b33054b5ba73daedaec4` | stamped `model_timesteps` 16,000,000; 3,166,235 / 3,757,990 testable px = **84.2534%**; mask fingerprint `21eab893...`; config hash `2aa92606...` |
| Retention evidence | `evaluation/results/glitch_hunter_qa_16000000_steps_d906d09e.json` | `be13f1e04da2f5c54c0b7906d6c69ddb6bf62714414ebfdd9d045763b2ae57b5` | 500-episode official protocol |

**Renamed 2026-09-24 (same bytes, same hashes):** the pair was called
`checkpoints_qa/glitch_hunter_qa_16000000_steps.zip` (+ `_coverage.npz`) until
the clean-up that day, which gave it one obvious name at the project root and
removed the superseded copies (the 16M milestone pair, the old root QA pair,
`backup_6M/`, `checkpoints/`, the archived runs, `logs/`). Older sections
below use the old names.

A byte-identical backup copy of the pair, the evidence (retention result, 6M
baseline, coverage audit trail, raw reward telemetry, the campaign's training
log) and the record `FINAL_OBJECTIVE2.json` live in
`checkpoints_qa/final_objective2_16000000/`. Every file there is read-only.

**Final retention (official 500-episode protocol, seeds and settings read from the
frozen 6M baseline; 12 workers, 51.7 min):** completion **283/500 = 56.6%** (95% CI
52.2-60.9%) against 46.8% for the 6M baseline; mean progress **0.750** against
0.710; the greedy run completes the level in 434 steps; verdict **HEALTHY**.
Endings: 283 level-complete, 126 deaths (goomba 92, koopa 19, pit 15), 91
time-outs. The 6M baseline is unchanged.

**Coverage is 84.25% of the testable mask. It is NOT 100%, and no document or
report may say so.** 591,755 testable px were not visited.

**Why training was stopped here (the user's decision, and the evidence for it):**

* Returns had become very small. Measured against the new denominator: 6.5M
  72.49%, 8.5M 79.46%, 10.5M 81.08%, 12.5M 81.58%, 14.5M 83.23%, 16.0M 84.25%.
  The last 0.8M steps (about 1.2 h, 15.2M -> 16.0M) added 10,182 px (+0.27 pt):
  about 4,060 in the first ~0.6M steps, then ~6,040 in one breakthrough near
  15.98M, with long stretches of 0 new px between. Gains come mostly from rare
  multi-jump breakthroughs, and the stagnation detector warned repeatedly.
  85% would need about 28,000 more px from the 16M state; the user's estimate
  from the earlier rate was 10-12 h for 2-3 points.
* What is left is mostly open air. Of the 591,755 remaining px, 503,804 (85.1%)
  are at y < 300 and 392,502 (66.3%) at y < 200 (y = 0 is the top of the world).
  Coverage by altitude: y 300-400 95.03%, y 400-500 95.94%, y 500-600 93.72%,
  y 200-300 85.51%, y 100-200 71.97%, y 0-100 56.12%. Within 60 px of any
  solid (ground, pipes, stairs, bricks) coverage is **92.14%** (101,125 px
  remaining); farther from any solid it is 80.15%. Only 87,951 remaining px are
  at y >= 300.
* The user's judgement is that Mario would practically never be in the
  remaining high-air space during real play, so extra coverage there has little
  QA value. The data are consistent with that. **It is a judgement, not a proof:
  nothing shows that no bug exists in the uncovered space.**
* Continuing was also not free of risk: every extra step adds drift pressure on
  the completion skill, which is only ever measured by the 500-episode test.

**How to read "accepted practical ceiling" honestly.** 84.25% is the accepted
*practical stopping point*, not a proven upper limit. Of the uncovered px,
415,634 are reachable even under the strict arc model (an arc dies at a wall) and
another 176,121 only under the generous one; the strict-model bound is 95.31%,
the generous bound 100%. The true reachable ceiling lies between those and is
unknown.

**Integrity checks run at closure (all passed):** the model zip tests clean and
carries 16,000,000 steps; the coverage file is stamped 16,000,000 and the
current mask fingerprint; an independent recount of the visited bitmap against
the mask gives exactly 3,166,235 / 3,757,990 and the stored totals; the 15.2M
coverage is a subset of the 16M coverage (0 px lost, +10,182 gained);
`train_agent.py --dry-run-resume --resume-from <the pair>` reports "retention:
HEALTHY", paired at 16,000,000, integrity OK; `tools/verify_artifacts.py` 40
artifacts at that time (37 after the cleanup below), 0 changed (the 6M brain and every backup, the baseline, the masks and
mask inputs are byte-identical); `checkpoints_qa/pre_main_6032768` is still the
healthy 6M-era master (`7d2f3e37...`, matches its own 500-episode result).

**An inconsistency was found and fixed during closure.** When the run reached
the safety cap it wrote the repo-root `glitch_hunter_qa.zip` one rollout later,
at **16,002,816** steps, never evaluated. Automatic resume picks the highest step
count, so a bare `train_agent.py` would have resumed that unevaluated file, not
the approved brain (dry-run showed exactly that). Its coverage is the same
visited set as the approved pair, so nothing was lost, but the weights differ.
That pair was archived (later deleted in the cleanup below; its hashes stay in
`FINAL_OBJECTIVE2.json`) and the root pair was then **replaced by
byte-identical copies of the approved pair**. The root files are ordinary
working files: any future training run overwrites them, which is why the frozen
copies exist. Never treat the root pair as authoritative; check its hash against
the table above.

**Cleanup done after closure (2026-09-21, ~690 MB).** Deleted: all regenerable
caches (`.mypy_cache`, `.pytest_cache`, `.ruff_cache`, every `__pycache__`); the 23 unevaluated intermediate milestones 6.8M-15.6M with their
coverage files and flags; `candidate_anchorB_6196608_mask_v3/`;
`pre_mask_v4_15200000/`; `archive_post_cap_root_16002816/`; the old-mask
`coverage_bootstrap_6000000_mask_v3.npz`; four regenerable `coverage_audits/`
files (the hand-made `arc_reach_map_14.8M.png` was kept); `tools/recheck_bootstrap.py`
(a finished one-time migration); and `docs/CAMPAIGN_LAUNCH.md`. So any older
path named below (`candidate_anchorB_6196608*`, `pre_mask_v4_15200000/`, a
6.8M-15.6M milestone) **no longer exists** - the evidence tables and results
remain; the files do not. Kept on purpose: the frozen final folder and the 16.0M
milestone pair, `pre_main_6032768`, `pre_unlimited_6360000`, the 13 experiment
`archive_*_run` folders, all 6M brain copies, `logs/console.log` and the root pair.

**Second cleanup (2026-09-23, 108.7 MB, user-approved).** Also deleted: the
6.4M REGRESSED checkpoint (+ coverage + flag; its verdict and numbers stay in
`evaluation/results/glitch_hunter_qa_6400000_steps_cfaa3b48.json` and above),
`candidate_retreat_6196608/` (the campaign's reproducible start is
`pre_unlimited_6360000/`; its HEALTHY result stays in `evaluation/results/`),
the live `checkpoints_qa/coverage_audit_trail.jsonl`,
`checkpoints_qa/reward_telemetry.jsonl` and `logs/train.log` (each
byte-identical to its hash-protected copy in
`final_objective2_16000000/evidence/`), all caches, and one synthetic
Objective-3 test incident. Verified afterwards: 37 artifacts / 0 changed;
resume still selects the approved 16,000,000-step pair.

**Starting Objective 3 from this exact state.** Use the frozen pair explicitly,
never the automatic choice:

```
venv_gpu\Scripts\python.exe train_agent.py --dry-run-resume --resume-from glitch_hunter_main_brain.zip
```

(a dry run that trains and writes nothing). A launch of any QA campaign is still
refused unless `--safety-cap-timesteps N` or `--unrestricted` is given, so a bare
`train_agent.py` cannot silently continue Objective 2. Do NOT relaunch the old
"Resume from 15.2M" command below: it would re-run finished work from an older
state. Objective 3's own training must write to its own checkpoint locations; it
must not overwrite `glitch_hunter_main_brain*` or the
frozen folder. If it trains on the same Level-1 coverage campaign it is a new
campaign from this state, and its result is judged by the same 500-episode
retention test against the 6M baseline before anything is accepted.

**Known limitations at closure (nothing hidden):**

1. Coverage is 84.25%, not 100%; the true reachable ceiling is unknown (95.31%
   strict bound, 100% generous bound). The uncovered space is not evidence of
   "no bugs there".
2. Retention is one 500-episode measurement of one checkpoint. Its CI is 52.2-60.9%;
   run-to-run variance was measured at several points earlier in this log.
   Completion is 56.6%, not near 100%: 25% of episodes end in a death and 18%
   in a time-out. Walls were reduced, not eliminated (see the handoff report).
3. The denominator was changed mid-campaign (4,002,095 -> 3,757,990, real-arc
   envelope). Percentages before and after are not comparable unless
   re-expressed against the same denominator, as done above. The migration was
   lossless (0 px lost), but the arc model is a model: 8,274 px of real play at
   the very top of the world (x 6000-6200) are kept as "observed" although the
   arc envelope denies them, and their cause was never identified.
4. 21,473 px were flagged "anomalous" at the freeze (17,004 when this run was
   resumed; it rose during the last breakthrough). They are classifier output,
   never triaged one by one.
5. The stagnation detector is detect-only by design (`STAGNATION_ESCALATE =
   False`). The watchdog raised many "brain collapse" alerts during the
   campaign; they were treated as benign because completion retention is judged
   separately by the 500-episode test, which passed.
6. `checkpoints_qa/`, `evaluation/results/` and the frozen folder are git-ignored
   local files. Git tracks only `artifacts.json` (the hashes) and these docs; a
   copy of the frozen folder on another disk is worth making, and is the
   owner's to arrange.

## The brief (user's requirements, condensed from the 2026-09-19 prompt)

Build a master Level 1-1 QA agent that explores until all genuinely testable
spatial coverage is exhausted and verified, while retaining the ability to
finish the level. Hard audit → fix → controlled experiment → measure → refine,
until either (A) the architecture is genuinely ready for the next
**user-launched** main campaign, or (B) a precisely identified fundamental
blocker remains. Do not start Objective 3.

1. Hard-audit the whole pipeline; look for hidden assumptions, stale state,
   exploits, contradictions, unreachable targets, train/eval mismatch, etc.
2. Scientific trial and error: hypothesis → experiment → measurement →
   decision. Keep failed experiments documented so they are not repeated.
3. Derive all numeric values (reward scales, LR, KL, thresholds, jump
   guidance, eval sample size) from measurement. Never ask the user for them.
4. Coverage: world-space, only testable pixels count, persistent across
   episodes/deaths/restarts, never un-covered, one global campaign across
   workers, integrity-checked. Do not change the denominator merely to make
   completion easier; do not chase pixels demonstrably outside testable space.
   The definition must be defensible from engine behaviour.
5. Solve remaining-space targeting, not just novelty reward.
6. Fix vertical exploration and jumping from engine physics (collider, jump
   dynamics, apex, launch positions) — no scripted walkthroughs.
7. Systematic, not random exploration; transit through old ground is legit.
8. Reward serves the objective; no secondary reward may substitute for
   discovery (interaction, locomotion, oscillation, standing, known routes).
9. Preserve completion ability; newest checkpoint ≠ best; accept checkpoints on
   exploration + PPO health + completion retention + behaviour.
10. EXPLORE → COMPLETE driven by evidence, never elapsed time alone.
11. Resets only on completion, death, or demonstrated useless stuck/loop.
    Never time-based. Distinguish transit / hard exploration / stagnation.
12. PPO stability is part of the objective; fix optimisation, not reward hacks.
13. Resume safety: never silently resume older / mismatched / corrupt / known
    bad checkpoints; explicit rollback; validate timestep, fingerprint, hash,
    retention.
14. Generic anomaly discovery; separate real physics, expected states,
    classifier limits, and genuinely suspicious behaviour; preserve evidence.
15. The real engine is ground truth; reconcile models with measurements.
16. Excellent terminal observability: clear startup summary and periodic live
    status; raw telemetry kept.
17. Dashboard/window UX (Stop pauses without minimising, close pauses, Start
    reopens centred, windows never affect observations). Headless eval where
    proven equivalent.
18. Integration/regression tests for system-level invariants.
19. Iterate until the whole system aligns; never claim readiness from code
    alone.
20. Final handoff report: 17 numbered sections (flaws, root causes, failed
    approaches, architecture, changes, targeting, jumping, anti-farming,
    retention, PPO, coverage integrity, resume, anomaly validation, validation
    results, limitations, exact manual command, readiness verdict) plus all
    test/lint/integrity results. Mark anything unproven as unproven.

Standing rules: never start the main/unrestricted campaign (the user launches
it); announce bounded experiments before launching (they open 8 windows);
never touch the D: drive; do not commit unless asked; keep the original 6M
artifacts and the 6.03M healthy pair byte-identical.

## Key paths and hashes

* Healthy seed pair (never modify): `checkpoints_qa/pre_main_6032768/`
  zip `7d2f3e37…`, coverage `2e4379c9…`, 2,225,509 covered. The root
  `glitch_hunter_qa.zip` / `_coverage.npz` must be restored to these bytes
  after every experiment (training overwrites the root on save).
* Regressed evidence (keep): `checkpoints_qa/glitch_hunter_qa_6400000_steps.zip` `cfaa3b48…`.
* Candidates from experiments: `checkpoints_qa/candidate_*`.
* Archived per-run telemetry: `checkpoints_qa/archive_*`. Before a run, move the
  live `reward_telemetry.jsonl` + `coverage_audit_trail.jsonl` into an archive
  folder and restore the copies from `pre_main_6032768/`.
* Scratchpad analysis tools (session temp dir): ppo_health.py,
  telemetry_audit.py, behaviour_probe.py, paired_eval.py (headless paired
  subset of the unchanged retention protocol), jump_physics.py, runup.py,
  well_escape.py, well_trace.py, flagpole_footprint.py.

## Findings (with evidence)

1. **Completion is extinguished by discounting.** GAMMA 0.99 → horizon 100
   agent steps; a completing episode takes a median 451. The +5 flag is worth
   0.054 at the first action; novelty outweighs it ~146:1 discounted. EXPLORE
   pays 0 for moving right. Undiscounted, finishing pays most (+27 vs −4 death).
   So the skill decays through lack of reinforcement, not interference.
2. **Farming:** the 6.4M policy was 66.6% Jump + 21.3% Left+Jump, 10.8%
   rightward, median max_x 858 — hitting ? blocks at x≈776–1,172 (interaction)
   and direction-agnostic locomotion. FIXED: secondary-income gate (below).
3. **Tall-pipe skill gap:** recurring timeouts at max_x 1,943 / 2,415 are Mario
   pressed flush against 172 px obstacles. Standing jump = 166 px (fails),
   running jump (|x_vel| > 4.5) = 183 px. Needs a run-up; policy never backs off.
4. **Staircase well (x≈5,904–5,988) is NOT a soft-lock** (initial physics claim
   refuted): escapable via an engine quirk, ~1% of random attempts.
5. **Engine quirk (anomaly evidence):** acceleration state leaks across the
   ground→air transition. Turnaround sets x_accel 0.35 and sprint sets
   max_x_vel 800; `jumping()` resets neither, so mid-air acceleration runs at
   0.35/frame uncapped (observed x_vel 7.65 > walk cap 6.0), allowing a
   running-jump rise with no run-up. Reproduce: well_trace.py (seed 20260920,
   random attempt 61, start x 5955).
6. `RUN_ACCEL = 20` never applies from a standstill (walk and sprint accelerate
   identically, 0.15/frame, 29 frames / 70 px to reach 4.5).
7. **Denominator — flagpole trigger (the one correction still standing).**
   Checkpoint '11' = rect x 8504, y 5, 6×600: full level height. Touching it at
   any height starts the scripted slide + walk to the castle door (checkpoint
   '12', x 8775, mario.kill()). Reachability B/C model solids only, so they
   flood past it. Real-play coverage past the pole is FROZEN across bootstrap,
   6.03M and 6.4M. ~169–172k of 200,832 adopted-testable px at x ≥ 8504 look
   unreachable → the objective 4,013,723 would be unattainable (~95.8% ceiling).
   Reachable past the trigger = the full-height pole column (x≈8504–8539) +
   the ground-level walk corridor to the door. The first engine replay only
   grabbed low and missed 6,829 real-play pole-column px, so it was extended
   with high grabs from the staircase top (flagpole_footprint.py, 2nd run);
   its acceptance check is: every real-play past-trigger px must be reproduced.
   Second replay (672 approaches incl. high staircase grabs) still missed
   3,392 real-play px, all in the pole column x 8504–8539. RESOLVED by the
   mechanism: keep free reachability in the GRAB BAND x 8504..8564 (trigger
   right 8510 + big collider 40 + one frame 14) at every height, and beyond
   it only the recorded CORRIDOR, column-filled from its top edge down.
   Acceptance: 0 of 22,542 real-play past-trigger px fall outside it (bootstrap,
   6.03M, 6.4M, rehearsal B). Removes 147,060 px → predicted denominator
   3,866,663. A 2-replay corridor was REJECTED (missed 3,871 px: the approach,
   not just Mario's size, changes the scripted path). The pole also stands on
   a 43 px base block (x 8488–8528), so a ground walk-in stops at x 8458.
   IMPLEMENTED (reachability.flag_trigger_rect / apply_flag_trigger /
   beyond_flag_region, new ANOMALOUS class CLS_BEYOND_FLAG, corridor stored in
   the bundle, builder records it with 672 parallel replays,
   build_testable REFUSES without a corridor). Old mask archived at
   `exploration_data/archive_pre_flag_trigger/`. Rebuild RUNNING.
   `tools/migrate_coverage.py` re-stamps coverage files losslessly (refuses to
   overwrite, refuses if covered_testable would drop, verifies load_verified).
8. Run-up-aware reachability (183 only with ≥70 px ground) was built and
   WITHDRAWN: finding 5 shows speed can be built in the air, so 183-everywhere
   is justified. Its "cluster 1" (1,709 px, x 7,422–7,547) stays testable.
9. **Frontier potential is inert:** nearest unexplored cell is a median 38 px
   away (cap 1,200). 100% of standable anchors have uncovered space in their
   jump envelope (median 35,070 px). Targeting is not the bottleneck; arc
   diversity + completion retention are. A planner was NOT built for that reason.
10. **Coverage is a union**, so it rises even while the median episode rots — a
    lagging indicator of policy health. Telemetry had no max_x (added).
11. **KL early stops** on the first ~3 updates after a resume are a transient.
12. **Critic hypothesis refuted:** a frozen-actor warm-up fitted the critic to
    EV 0.65–0.83 with the actor bit-identical; resuming from it the policy still
    drifted (EV up to 0.906 while drifting).
13. **Eval noise:** two runs of the SAME config scored 27% and 38% on the same
    100 seeds (±~9 pt CI at n=100). Pooled 65/200 = 32.5% vs healthy 43.8%
    (500 ep). Decisive comparisons need the full 500-episode protocol.
14. Headless evaluation is proven identical to windowed: 40/40 episodes
    byte-identical (paired_eval.py on the healthy 6.03M).
15. StagnationCallback was a one-way ratchet (novelty ×1.25→2.5, ent_coef
    ×1.15→0.06), ent_coef saved into checkpoints; fired on stuck-env stalls.

## Changes made (all since committed)

* `rewards/qa.py`, `exploration/config.py`: secondary-income gate — interaction
  AND locomotion scaled by `QA_SECONDARY_DROUGHT_SCALE` (0.1) on the shared
  drought condition, hardened to 0 once `DROUGHT_EPISODE_CAP` is spent.
  Replay: zero-yield net-positive 38/129 → 0/129, productive−zero gap 24.9 →
  26.0. Coupled budget (secondary ≤ k·novelty) tried and REJECTED.
* `NORMAL_LR = 2.5e-5` (was 1e-4: 8/9 updates early-stopped).
* Health-aware resume + `--resume-from` / `--dry-run-resume`
  (`training/checkpoints.retention_verdicts`); REGRESSED never auto-resumed.
* `ValueWarmupCallback(freeze_actor=…, ev_release=…)` — off by default
  (`VF_WARMUP_FREEZE_ACTOR = False`); tested, not the cause of drift.
* Telemetry: `max_x`, `max_x_at_transition`, `rehearsal` per episode.
* **Completion rehearsal:** `COMPLETION_REHEARSAL_PERIOD` (default 0) — every
  Nth episode per worker starts in COMPLETE (dense progress reward), reason
  `R_rehearsal`, credit 1.0. Passed EXPLICITLY via `make_env(...,
  rehearsal_period=)` and `train_agent.py --rehearsal-period N`. (A launcher
  that set config in the parent silently did nothing in the spawned workers:
  0/173 episodes rehearsed. Never override worker-read config in a parent.)
* `STAGNATION_ESCALATE = False`: detect + warn only. QA resume pins
  `ent_coef` to `ENT_COEF_BASE`.
* `tests/test_completion_eval.py` walks sub-folders of checkpoints_qa.
* `tools/evaluate_completion.py` is HEADLESS by default (`--windowed` to watch);
  headless proven identical to windowed, 40/40 episodes.
* New tests for all of the above; full suite was green (481 passed, 3 skipped)
  before the latest rehearsal/stagnation changes — rerun before handoff.

## Experiments (all from the healthy 6.03M pair, cap 6,196,608 unless noted)

| run | config | outcome |
|---|---|---|
| run 2 | gate, lr 2.5e-5 | paired 100 ep: 27% (healthy 47%) — REGRESSED |
| replicate | same as run 2 (rehearsal launcher bug) | paired 100 ep: 38% — WARNING |
| frozen critic | actor frozen, critic lr 1e-3 | EV 0.32→0.83, actor bit-identical |
| warm-start | resume from critic-warmed, actor free | drifted anyway (ep len 297→1,840) |
| **rehearsal B** | gate + `--rehearsal-period 4` (verified: 45/162 episodes rehearsed) | training looked healthy (explore completion held 40–52%, coverage 2.62M) BUT official 500-ep retention: **30.2% (CI 26.3–34.4) REGRESSED** — indistinguishable from no rehearsal (pooled 32.5%). Rehearsal at period 4 REFUTED as a fix. |

**KL predicts retention (r = −0.962, 6 checkpoints).** Mean KL(healthy‖candidate)
on 9,175 states from the healthy policy's own retention-protocol trajectories:
6.03M 0.000→43.8%, 6M 0.117→46.8%, replicate 0.253→38.0%, run2 0.323→27.0%,
rehearsal B 0.392→30.2%, 6.4M 0.750→6.0%. Fit over drift products:
retention ≈ 46.7 − 51.5·KL → WARNING at KL≈0.12, REGRESSED at KL≈0.24. Design
chosen: anchor consolidation — after each PPO update, pull the policy back
toward the frozen healthy 6.03M on a fixed anchor-state set (seeds disjoint
from the eval) whenever anchor KL exceeds a budget of 0.10.

IMPLEMENTED: `tools/build_anchor_set.py` → `exploration_data/anchor_states.npz`
(6,007 states, seeds 1000–1023, reference sha checked); `AnchorConsolidationCallback`
(separate Adam, actor params only, never the value head or the frozen reference,
no-op under budget); `train_agent.py --anchor-kl [KL]`. Anchor-set KL also tracks
retention (r = −0.924) but on a HIGHER scale (6M baseline 0.146 vs 0.117), so the
budget was refitted on the anchor scale: retention ≈ 47.3 − 42.0·KL → WARNING at
0.164, budget **0.13**.

**Experiment "anchor A"** (gate + `--anchor-kl`, no rehearsal, cap 6,196,608),
candidate `checkpoints_qa/candidate_anchorA_6196608/` (sha 7946b682):
the constraint behaved exactly as designed — update 1 drove anchor KL 0 → 0.204
(one update crosses the WARNING line!), consolidation pulled it to 0.007;
updates 2,3,5–9 stayed under budget and were left alone; update 4 hit 0.141 and
was pulled to 0.016. Saved model KL **0.1156** (under budget) → predicted
retention 42.4%. PPO health best of all runs (3/10 early stops vs 7/10; EV
0.36–0.70). Official 500-ep retention: **40.2% (CI 36.0–44.6), WARNING** — 0.4
points (two episodes) below HEALTHY, up from rehearsal B's 30.2% REGRESSED.
Mean progress 0.700 vs baseline 0.710 (rehearsal B: 0.610); timeouts 48 vs 90;
**greedy policy completes again** (373 steps, max_x 8,751) where rehearsal B's
greedy jammed at 1,364 and 6.4M's at 828. Predicted 42.4%, actual 40.2% — the
fit is sound but ~2 points optimistic. Exploration 219,727 new testable px.

**Experiment "anchor B"** (same, budget 0.08), candidate
`checkpoints_qa/candidate_anchorB_6196608/` (sha 57b7788b): saved KL **0.0441**,
exploration **280,129** new px — BETTER on both axes than anchor A, so the
tighter budget did not cost throughput (n=1 each; run-to-run variance is
large). Predicted retention 44.7%. Official 500-ep retention: **49.2%
(CI 44.8–53.6), HEALTHY** — better than the 6M baseline (46.8%, +2.4) and the
healthy seed (43.8%); mean progress 0.730 vs 0.710; greedy completes; 246/246
completions inside the original 401-unit clock. First candidate that improves
on the seed at BOTH completion and exploration. Replicate pending (run-to-run
variance was measured at 11 points, so one HEALTHY result is not yet proof).

**Experiment "anchor C"** (replicate of B, sha 38fd89c9): **43.8% completion
(CI 39.5–48.2), WARNING** — equal to the healthy seed on completion and far
above the 40.46% line, but failing the PROGRESS criterion at 0.670 vs 0.6777
(by 0.008); greedy died at x 3,269. So at budget 0.08 the catastrophic
regression is gone (both replicates ≥ the seed's 43.8%, vs 27–30% before) but
one run in two is marginal on progress. Mean progress is exactly what the
run-up fix should raise, since clearing the 172 px walls is what lifts max_x.

**Mask v3 — BOTH FORMS, built and validated.** Method B 4,097,097 → 4,252,679;
Method C → 4,169,305; flag trigger −167,210; **adopted 4,002,095** (fingerprint
ded5cd19…). Near the original 4,013,723 but for two correct and nearly
cancelling reasons. Acceptance: across 9 coverage states no covered pixel was
lost, covered counts ROSE where big Mario had been misfiled (run 2 +16,037),
and the worst anomalous count fell **25,356 → 2**. Seed and bootstrap
re-migrated (`*_mask_v3`), config/artifacts/test pins updated, 23/23 verified,
resume dry-run clean at 55.6086%.

**RUN-UP (§6) implemented.** `reachability.barrier_tops/runup_rise/runup_zones/
load_runup_zones` derive from the solid mask, with gaps shorter than Mario
treated as wall (the tall pipe at x 1975 spans y 366–535 over a floor at 538 —
a 2 px gap had made a 172 px pipe look like floating scenery). A zone is where
the NEXT barrier is >166 px (standing) and ≤183 px (running): exactly 5 zones,
536 columns, containing all three measured timeout points and excluding short
pipes, staircases, floating bricks and the 344 px steps no jump clears. New
`runup` reward channel pays `QA_RUNUP_REWARD` at take-off when |x_vel| > 4.5
inside a zone, ONCE PER ZONE PER EPISODE (unfarmable; deliberately NOT
drought-gated, since a Mario stuck at a wall is by definition in drought).
6 tests incl. 50 take-offs paying once. Experiment "runup" (mask v3 + gate +
--anchor-kl 0.08 + run-up): RUNNING. Metric to watch: mean progress / max_x.

Two callback bugs found and fixed by its own tests: (1) the FINAL update was
saved unconsolidated (no rollout follows it) → `_on_training_end` consolidates;
(2) consolidation can DIVERGE (a rig measured 1.42 → 22.55) → the actor is
snapshotted and restored if KL ends worse, so it can only help or do nothing.

Observability (§16): banner now also prints worker count, the bootstrap vs
QA-discovered split (`lc.provenance`) and the anomalous-px count.
Anomaly recorded: `docs/objective2/ANOMALY_acceleration_leak.md` + runnable reproductions
in `docs/objective2/evidence/` (excluded from lint, kept exactly as recorded).

**Mask v3 — BOTH OF MARIO'S FORMS (pre-existing bug, found 2026-09-20).**
Methods A/B/C and the noncoverage classifier all ran on the SMALL collider
(30×40) only, while coverage records the collider at its real per-form size.
Big Mario is 40×80, so from the same footing his head is 40 px higher than any
small-Mario anchor: ordinary big-Mario play fell outside the mask and was
classified ANOMALOUS. Measured on the run-2 candidate: 25,356 anomalous px, of
which the big envelope explains ALL (16,037 `unreachable_altitude` at y 100–160
and 9,319 `impossible_sky` at y −80…−20, the same cause through the sky
ceiling). Five of seven candidate coverage states were affected (anchor B 2,786;
rehearsal B 6,676; critic-warmed 1,704). Method B small 4,097,097 → big
4,242,649. NOT caused by the flag correction (zero `beyond_flag_trigger` px).
The agent had COVERED those pixels — the engine demonstrating what the model
denied (§15). FIX: `build_testable` unions A/B/C over both forms, and
`classify_noncoverage` takes the most permissive ceiling across forms and the
widest collider for the pit test. Rebuild RUNNING; migration + re-validation
to follow. Old (small-only) mask archived at
`exploration_data/archive_mask_v2_small_only/`.

**Mask v2 done:** rebuilt (3,866,663; fingerprint 4d133203…9847); 6.03M and
bootstrap coverage migrated (numerators unchanged); root coverage = the migrated
copy; `config.BOOTSTRAP_COVERAGE` → `coverage_bootstrap_6000000_mask_v2.npz`;
artifacts.json 23/23 OK; test pins updated.

**Lesson:** training-time completion overstates eval completion by 12+ points
even for the healthy policy (≈56% in training vs 43.8% eval). Never accept a
checkpoint on training statistics; only the official 500-episode protocol.
Four cause-targeted fixes (gate, critic, LR, rehearsal) all failed to stop the
drift. Next: bound the drift directly (KL to the frozen healthy policy), with
the strength calibrated by measuring whether KL-from-healthy predicts
retention loss across the existing checkpoints.

## Candidate results (all from the 6.03M seed, 164k steps, mask v3 denominator)

| candidate | covered | new | cov% | retention (500 ep) |
|---|---|---|---|---|
| 6.03M seed | 2,225,509 | – | 55.61% | 43.8% prog 0.690 HEALTHY |
| anchorA (KL 0.13) | 2,445,236 | 219,727 | 61.10% | 40.2% prog 0.700 WARNING |
| **anchorB (KL 0.08)** | 2,508,424 | 282,915 | 62.68% | **49.2% prog 0.730 HEALTHY** |
| anchorC (KL 0.08, replicate) | 2,528,906 | 303,397 | 63.19% | 43.8% prog 0.670 WARNING |
| gated only (run 2) | 2,606,181 | 380,672 | 65.12% | 27.0% (n=100) |
| rehearsal B | 2,630,975 | 405,466 | 65.74% | 30.2% prog 0.610 REGRESSED |
| runup (mask v3 + KL 0.08 + run-up) | 2,514,251 | 288,742 | 62.82% | 44.2% prog 0.680 HEALTHY |
| **retreat (run-up + retreat shaping)** | 2,584,903 | **359,394** | **64.59%** | **48.0% prog 0.690 HEALTHY** |

**The retreat candidate wins on both axes at once**, which no earlier candidate
did: the most new pixels of any constrained run (359,394) *and* a healthy
retention verdict (48.0%, 95% CI 43.6-52.4; castle door 240, flagpole 240).
Deaths: goomba 137, koopa 16, pit 44; timeouts 63. Against the 6M reference it
is +1.2% completion and -0.020 progress. Its completion is statistically
indistinguishable from anchorB's 49.2% (the CIs overlap almost entirely), so it
is chosen on its 76,479-pixel exploration margin, not on a completion
difference that the sample size cannot resolve.

Retention thresholds (derived, not prescribed): completion WARNING <0.4046,
REGRESSED <0.3412; progress WARNING <0.6777, REGRESSED <0.6406. For scale, the
**unconstrained** 6.4M run scores 6.0% / prog 0.280 — that is the completion
extinction this whole workstream exists to prevent.

The `runup` progress figure (0.680) clears the WARNING line by 0.0023, i.e. by
roughly one episode in 500. Treat it as "passed, but not distinguishable from
the warning band" rather than as a comfortable pass.

**A tooling mistake worth remembering:** for much of this session output was
piped through `grep -viE "^pygame|^hello|warn"`. The `warn` term silently
deleted every line containing "WARNING" — including whole table rows whose
verdict was WARNING, which is how anchorA and anchorC appeared to vanish from
a summary. Filter only on `^pygame|^hello`; never filter on a word that can
appear in a result.

## Where to resume

### HISTORICAL (2026-09-21, superseded by the FINAL STATE block at the top): unlimited campaign paused at 15.36M; denominator is now 3,757,990

> **Do not use the resume command in this subsection.** The campaign was resumed
> from 15.2M, run to 16.0M, evaluated and closed. The final pair is in the FINAL
> STATE block. The section is kept as the record of how the denominator changed.

**What happened.** The user launched the unrestricted campaign from
`pre_unlimited_6360000` and let it run ~9M steps: coverage 66.84% -> 78.86%
(15.2M milestone: 3,156,053 px), level completion stayed roughly 40-60% of
episodes, the anchor never warned, no crash, anomalous px constant at 15,360.
Growth was two big breakthroughs (+55,745 px near 12.97M, +24,138 px near 14.79M)
between long plateaus; the last ran from ~14.88M past 15.36M with ~0 new px.
Policy was not collapsed (entropy_loss -1.5, explained variance 0.87-0.90), so
the plateau was not a training fault. `STAGNATION_ESCALATE` stays False - its own
comment records that escalating never helped.

**Why it plateaued: the denominator over-counted.** Methods B/C bound a jump by
a RECTANGLE (rise 183 AND reach 480 together); no arc does both. Rebuilt with
228 real engine arcs (`tools/collect_jump_arcs.py`, 600 take-offs; max rise 183):

| model | reachable px | of covered play it explains |
|---|---|---|
| old rectangle mask | 4,002,095 | - |
| real arcs, walls slide the arc (generous) | 3,749,716 | 99.74% |
| real arcs, an arc dies at a wall (strict) | 3,457,007 | 96.05% |

Of the 846,634 px then uncovered: <= 602,529 reachable, >= 244,105 not; all of
the unreachable part is high in the sky (y < 300). Everything below y 300 that
was still uncovered was reachable. So 100% was unattainable, and the plateau was
the hard multi-jump upper band.

**Change made (since committed).** `TESTABLE_TOTAL` 4,002,095 -> **3,757,990**,
`TESTABLE_FINGERPRINT` `21eab893d63a2578...`. New code: `reachability.method_arcs`
/ `apply_arc_envelope` / `load_jump_arcs` / `load_observed_reach`, wired into
`build_testable(arcs=, observed=)`; `tools/collect_jump_arcs.py`;
`tools/build_reachability.py --tighten --coverage NPZ...` (equivalent to a full
rebuild, a few minutes; archives the old mask). Inputs: `exploration_data/
jump_arcs.npz`, `observed_reach.npz` (8,274 px of real play the arcs deny -
mostly the very top of the world at x 6000-6200 - kept testable so coverage is
lossless; cause not identified, death-jump or enemy bounce are only hypotheses).
The 244,105 removed px are classified CONNECTIVITY_GAP (model gap, never a
glitch); the taxonomy changed on exactly those px and nowhere else (checked).
Old mask archived at `exploration_data/archive_mask_v3_rectangle/`. Bootstrap
re-stamped to `coverage_bootstrap_6000000_mask_v4.npz`. New tests:
`tests/test_arc_reach.py` (9). `artifacts.json` updated. Still an UPPER bound
(the generous model): the strict model's ceiling, computed as covered plus
strictly-reachable remaining, is 95.31% of the new mask (an earlier note here
said ~92%; that was wrong).

**The campaign was paused by the user at 12:57 (step 15,363,224). Its pause-time
save, `glitch_hunter_qa.zip` at the repo root, is TRUNCATED (788,854 bytes,
BadZipFile) - do not resume from it.** Resume from the 15.2M milestone, migrated
to the new mask at `checkpoints_qa/pre_mask_v4_15200000/` (loses ~163k steps and
~900 px of coverage, both re-earnable). Verified by `--dry-run-resume`: 15,200,000
steps, 3,156,053 / 3,757,990 (83.9825%), remaining 601,937, integrity OK.

```
venv_gpu\Scripts\python.exe train_agent.py --resume-from checkpoints_qa\pre_mask_v4_15200000\glitch_hunter_qa.zip --anchor-kl --unrestricted
```

(add `| Tee-Object -FilePath logs\console.log` in PowerShell to keep SB3's table
visible to a monitor; it is written as UTF-16LE). Every older coverage file is
now stamped with the previous mask and will be refused - migrate with
`tools/migrate_coverage.py` before using one.

### 2026-09-21: campaign stopped at 16,000,000 steps (user's chosen stop point)

Resumed from `pre_mask_v4_15200000` with `--anchor-kl --safety-cap-timesteps
16000000` (about 1.2 h, ~185 steps/s); stopped cleanly at the cap. Exact milestone:
`checkpoints_qa/glitch_hunter_qa_16000000_steps.zip` (+ `_coverage.npz`), coverage
3,166,235 / 3,757,990 = 84.2534%. At the cap the run also wrote the repo-root
`glitch_hunter_qa.zip` at 16,002,816 steps (one rollout later); see the FINAL
STATE block for how that was archived and the root pair realigned to the
approved 16,000,000-step pair (the truncated pause-time file is gone).

Official 500-episode retention test on that milestone (12 workers, 51.7 min):
**56.6% completion (95% CI 52.2-60.9%) vs 46.8% at 6M, mean progress 0.750 vs 0.710,
greedy run completes in 434 steps - VERDICT HEALTHY.** Result file:
`evaluation/results/glitch_hunter_qa_16000000_steps_d906d09e.json`.

### Earlier state (2026-09-20)

Done since the last revision of this section: rehearsal B was evaluated at the
full 500 episodes and **refuted** (30.2% / 0.610, REGRESSED) — `COMPLETION_
REHEARSAL_PERIOD` stays 0; the flag trigger was derived from the engine and
folded into `build_testable`, giving the mask v3 denominator 4,002,095 with
fingerprint `ded5cd19…`; the 6.03M coverage was migrated (copy, never edited in
place); the evaluator now runs headless by default (40/40 episodes identical to
windowed); the acceleration leak is written up in
`docs/objective2/ANOMALY_acceleration_leak.md`; ruff, mypy and artifact verification all
pass.

Also done: the retreat candidate returned **48.0% / prog 0.690, HEALTHY** and is
the chosen campaign start; the full suite passed (532 tests, 3 skipped, 0
failures); `ANCHOR_KL_TARGET` was lowered 0.13 -> 0.08 on the measured evidence
(0.13 was the one setting that produced a WARNING, and the tighter budget also
explored more); anchorB was pre-migrated to mask v3 at
`checkpoints_qa/candidate_anchorB_6196608_mask_v3` as a ready fallback; and
`docs/objective2/HANDOFF.md` holds the 17-section report.

**A live coverage table was added** (`training/callbacks.py`,
`CoverageStatsCallback._print_table_row`) — a column-aligned, human-readable
Step/Covered/Testable/Coverage%/Remaining/New(sess)/New per 10k table printed
every `every` steps (header repeats every 10 rows), routed through `log.info`
rather than raw `print` so it respects the console handler's pending-progress-
line coordination and lands in the run log too. Covered by two new tests.

**First real campaign milestone, run with the user's explicit go-ahead**
(2026-09-20, ~16:14-16:28): `--resume-from
checkpoints_qa/candidate_retreat_6196608/glitch_hunter_qa.zip --anchor-kl
--safety-cap-timesteps 6360000`, 163,392 steps. Stopped cleanly at the cap
(actual final step 6,360,448 - PPO's rollout granularity, not a bug). Coverage
64.59% -> 66.84% (+89,993 px), zero anomalous px throughout, anchor KL never
needed a pullback (stayed inside the 0.08 budget the whole run). Two
`WATCHDOG ALERT` lines fired (steps 6,256,560 and 6,307,800) but both times
coverage kept climbing and level completions stayed strong in the same window
(10/15 and 9/12) - checked and judged benign, not stopped. Preserved as
`checkpoints_qa/pre_unlimited_6360000/` and retention-tested at the full 500
episodes: **43.6% [39.3%, 48.0%], progress 0.690, HEALTHY** (sha `32465cfe`).
Nominally below the retreat checkpoint's own 48.0%, but the CIs overlap
heavily (43.6-48.0 overlap region) - consistent with the run-to-run noise
already measured between anchorB/anchorC (49.2% vs 43.8%), not a regression.
No rollback triggered.

**A real, unrelated bug was found via GitHub CI and fixed**: `tools/
migrate_coverage.py` used `os.path.relpath(src, ROOT)` for a provenance field,
which raises `ValueError` on Windows when the two paths are on different
drive letters - exactly what happens on the GitHub Windows runner (checkout on
`D:`, pytest's `tmp_path` on `C:`). Never surfaced locally (one drive here).
Fixed with a try/except falling back to `os.path.abspath`; added a test that
monkeypatches `os.path.relpath` to simulate the exact cross-drive failure.
Verified locally (ruff, mypy, the 6-test file). **Not yet committed or
pushed** - the user was asked and hasn't answered yet.

**Objective 2 is complete and awaiting the user's launch.** Nothing is
committed. The campaign has NOT been started.

The open items are now the campaign's, not this workstream's:

1. The user launches the capped validation milestone in handoff section 16
   (`--resume-from checkpoints_qa\candidate_retreat_6196608\glitch_hunter_qa.zip
   --anchor-kl --safety-cap-timesteps 6360000`). Verified by dry run: resolves
   to 6,196,608 paired with its own 2,584,903-px map, integrity OK.
2. Review that milestone against the candidate table above before deciding on
   `--unrestricted`. Stopping rule: one bad milestone -> roll back and
   continue; two in a row -> stop, because that is no longer variance.

Still genuinely unresolved, and to be stated plainly in the handoff rather than
smoothed over: the tall-pipe/wall failure mode is reduced but not eliminated
(~11% of episodes still end against a wall); exploration under the KL
constraint is ~25% slower than unconstrained, which is a deliberate trade, not
a bug; and no constrained run has been observed past ~164k steps, so long-run
behaviour is extrapolated from milestone checks, not measured.
