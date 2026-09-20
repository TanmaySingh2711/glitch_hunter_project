# Objective 2 — working log and handoff

Read this first after any context compaction or session restart. It holds the
user's standing brief for Objective 2, every finding so far, what has been
changed, and exactly where to resume. Keep it current as work progresses.

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

## Changes made (uncommitted)

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
Anomaly recorded: `docs/ANOMALY_acceleration_leak.md` + runnable reproductions
in `docs/evidence/` (excluded from lint, kept exactly as recorded).

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

Done since the last revision of this section: rehearsal B was evaluated at the
full 500 episodes and **refuted** (30.2% / 0.610, REGRESSED) — `COMPLETION_
REHEARSAL_PERIOD` stays 0; the flag trigger was derived from the engine and
folded into `build_testable`, giving the mask v3 denominator 4,002,095 with
fingerprint `ded5cd19…`; the 6.03M coverage was migrated (copy, never edited in
place); the evaluator now runs headless by default (40/40 episodes identical to
windowed); the acceleration leak is written up in
`docs/ANOMALY_acceleration_leak.md`; ruff, mypy and artifact verification all
pass.

Also done: the retreat candidate returned **48.0% / prog 0.690, HEALTHY** and is
the chosen campaign start; the full suite passed (532 tests, 3 skipped, 0
failures); `ANCHOR_KL_TARGET` was lowered 0.13 -> 0.08 on the measured evidence
(0.13 was the one setting that produced a WARNING, and the tighter budget also
explored more); anchorB was pre-migrated to mask v3 at
`checkpoints_qa/candidate_anchorB_6196608_mask_v3` as a ready fallback; and
`docs/OBJECTIVE2_HANDOFF.md` holds the 17-section report.

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
