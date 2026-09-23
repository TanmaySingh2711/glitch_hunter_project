# Objective 2 — final handoff report

Scope: the Level 1-1 QA explorer. Objective 3 was not started.

Everything below is measured unless a sentence says otherwise. Where a claim is
an extrapolation rather than an observation it is labelled **UNPROVEN**. (When
this report was written the changes were uncommitted; they have since been
committed.)

> **OBJECTIVE 2 IS CLOSED (2026-09-21) - this report describes the state
> *before* the campaign; the outcome is in
> [WORKLOG.md](WORKLOG.md), "FINAL STATE".** The
> unrestricted campaign was run to exactly **16,000,000 steps** and stopped by
> the user at **84.25%** practical spatial coverage (3,166,235 / 3,757,990
> testable px - **not** 100%). Final brain:
> `glitch_hunter_main_brain.zip` (SHA-256 `d906d09e...`; called
> `checkpoints_qa/glitch_hunter_qa_16000000_steps.zip` until 2026-09-24),
> paired with `glitch_hunter_main_brain_coverage.npz`
> (`aa384707...`), frozen with its evidence in
> `checkpoints_qa/final_objective2_16000000/`. Official 500-episode retention:
> **56.6% completion (CI 52.2-60.9%) vs 46.8% baseline, progress 0.750 vs 0.710,
> greedy run completes, verdict HEALTHY.** This resolves the two things this
> report marked open: the "no constrained run observed past ~164k steps" risk
> (§15.3, §17) - a 16M-step campaign under the anchor kept completion - and the
> "coverage can be driven to exhaustion" question (§15.5) - it was not driven to
> exhaustion: the remaining 591,755 px are 85% high air (y < 300) and the gain
> per hour had become very small, so the user stopped it. Whether a longer run
> could reach more remains unproven. Sections 15-17 below are kept as written.

The working log with the full experiment history is
[WORKLOG.md](WORKLOG.md). This report is the summary; the
worklog is the evidence.

> **Superseded number (2026-09-21).** Every `4,002,095` in this report is the rectangle-envelope denominator. It was replaced by **3,757,990** after the real-arc envelope showed 244,105 of those pixels can never be reached by any jump (see the worklog, "Where to resume" -> CURRENT). Percentages below are historical; the same coverage reads about 5 points higher against the new denominator.

---

## 1. Flaws found in the audit

| # | Flaw | How it showed up |
|---|---|---|
| 1 | **Completion extinction** | A QA run from 6.03M to 6.4M steps dropped completion from 43.8% to **6.0%** and progress from 0.69 to 0.28. The agent traded the entire level-finishing skill for exploration reward. |
| 2 | **Denominator was wrong twice** | The "testable" pixel count included space Mario cannot occupy, and excluded space he can. Coverage% was therefore not comparable between runs. |
| 3 | **Anomaly classifier was mostly noise** | 25,356 flagged "anomalies", nearly all of them ordinary physics or expected states. |
| 4 | **Secondary reward could substitute for discovery** | Interaction/locomotion shaping stayed payable forever, so an episode that discovered nothing could still be profitable. |
| 5 | **Vertical exploration was broken** | Mario could not clear the tall pipes/walls. He arrived at them at zero speed and jumped straight up, which is 17 px short. |
| 6 | **Train/eval mismatch** | Evaluation opened real windows; window presence had never been proven not to affect observations. |
| 7 | **Silent resume hazards** | Nothing stopped a launch from resuming an older checkpoint, or pairing a checkpoint with a coverage file recorded against a different mask. |
| 8 | **Time-based resets** | Episodes could end because a clock ran out, not because anything was demonstrated about the state. |
| 9 | **A launcher bug that invalidated an experiment** | A config value set in the parent process was silently re-read from disk by the 8 spawned workers, so a feature reported as "on" was off in every worker. |

## 2. Root causes

Flaws 1, 4, 5 and 8 are not independent bugs; three of them share one root.

**Flaw 1 — the discount horizon, not the reward weights.** With γ=0.99 the
agent's planning horizon is 1/(1−γ) = **100 agent steps**, half-life 69 steps.
Finishing the level takes far longer than that. So completion is not a goal the
value function can see from most states — it is beyond the horizon — whereas
exploration reward is immediate. No re-weighting of the reward fixes this,
because the problem is that one term is visible to the optimiser and the other
is not. This is why the first three attempted fixes (all reward-side) failed;
see section 3.

**Flaw 4 — a gate with no exhaustion condition.** Secondary reward was
throttled during a discovery drought but never cut off, so its integral over a
long episode was unbounded.

**Flaw 5 — a physics fact nobody had measured.** Measured in the real engine:
a standing jump rises **166 px**, a running jump **183 px**, and the barriers
in question are **172 px**. The rise only increases once |x_vel| > **4.5**,
which takes **29 frames / 70 px** of run-up. So the skill Mario was missing was
not *jumping* — it was **backing up to get a run-up**, a behaviour that gets no
gradient from any forward-progress reward.

**Flaw 9 — process boundaries.** `SubprocVecEnv` workers re-import config from
disk; parent-process mutation does not cross the boundary.

## 3. Approaches that were tried and failed

Kept here so they are not re-attempted.

| Approach | Result | Why it failed |
|---|---|---|
| **Re-weighting exploration vs completion reward** | No effect on extinction | Wrong lever — the cause is the discount horizon (§2), not the weights. |
| **Completion rehearsal** (every Nth episode starts in COMPLETE) | **Refuted.** 30.2% / prog 0.610 — REGRESSED, worse than doing nothing | Rehearsal episodes teach the end of the level from a state distribution the agent never actually reaches on its own. `COMPLETION_REHEARSAL_PERIOD` is now 0. |
| **Rehearsal, first attempt** | Void — 0/173 episodes actually rehearsed | The launcher bug (flaw 9). A whole 164k-step experiment was wasted. Every later run now verifies from telemetry that the mechanism engaged before the result is read. |
| **Anchor KL budget 0.13** | 40.2% — WARNING | The budget was set from a *predicted* regression fit with a margin thinner than run-to-run variance. Refuted in favour of 0.08. |
| **Run-up reward alone** (pay for taking off fast) | Wall failures unchanged, 15.4% vs 12–16% baseline | It paid for a behaviour the agent already had when it happened to arrive running, and gave no gradient for the missing one (backing up). Led directly to the retreat shaping that did work. |
| **Stagnation escalation** (intervene when stuck) | Disabled, left detect-only | Could not be shown to help, and it is an intervention in the very behaviour being measured. `STAGNATION_ESCALATE = False`. |

## 4. Architecture as it now stands

PPO (SB3 2.9.0, torch 2.5.1+cu121, Python 3.12), 8 `SubprocVecEnv` workers
against the real game engine. One **global** campaign coverage map shared
across workers, persisted with every checkpoint.

The piece that makes Objective 2 work is **anchor consolidation**. A frozen
copy of the last known-healthy policy is kept as a fixed reference. After each
PPO update, KL(reference ‖ policy) is measured on a fixed set of anchor states
and, if it exceeds the budget, the policy is pulled back toward the reference
(capped at 64 steps per update, lr 1e-4). Two properties matter:

- The reference is **fixed**, not a moving average of recent policies. Drift
  therefore cannot compound across milestones — each milestone is pulled back
  toward the same healthy point, not toward its predecessor.
- Consolidation **snapshots and restores**: if a consolidation step increases
  KL rather than reducing it, the snapshot is restored. It can only help or do
  nothing. (This guard was added after a test rig diverged 1.42 → 22.55.)

Anchor states come from seeds 1000–1023, deliberately **disjoint** from the
retention protocol's seeds, so the metric being optimised is never the metric
being reported.

## 5. Changes made

(Uncommitted when written; since committed.) Principal files:

- `exploration/config.py` — new constants, each with its measurement rationale
  in a comment: `ANCHOR_*` (consolidation), `JUMP_RISE_STANDING=166`,
  `RUNUP_THRESHOLD_VEL=4.5`, `RUNUP_RUN_PX=70`, `QA_RUNUP_REWARD`,
  `QA_RETREAT_REWARD`, `QA_SECONDARY_DROUGHT_SCALE=0.1`, `NORMAL_LR=2.5e-5`,
  `TESTABLE_TOTAL=4002095`, `COMPLETION_REHEARSAL_PERIOD=0`,
  `STAGNATION_ESCALATE=False`.
- `exploration/reachability.py` — flag-trigger geometry, `CLS_BEYOND_FLAG`,
  barrier/run-up zone derivation, both Mario forms unioned with per-form spawn.
- `rewards/qa.py` — secondary-reward exhaustion, run-up and retreat shaping,
  new telemetry exports.
- `training/callbacks.py` — `AnchorConsolidationCallback`, value warm-up,
  detect-only `StagnationCallback`.
- `train_agent.py` — `--anchor-kl`, `--resume-from`, `--dry-run-resume`,
  `--rehearsal-period`; richer startup banner.
- New tools: `tools/build_anchor_set.py`, `tools/migrate_coverage.py`;
  `tools/evaluate_completion.py` now headless by default.
- New docs: this report, the worklog, `docs/objective2/ANOMALY_acceleration_leak.md`,
  `docs/objective2/evidence/`.

## 6. Remaining-space targeting

Coverage is a 1 px world-space lattice, persistent across episodes, deaths and
restarts, and **monotone** — a pixel once covered is never un-covered. The
testable mask is built by unioning three independent reachability methods over
**both** Mario forms (small and big, each with its own spawn transform, since
the two have different colliders). Method C (connectivity BFS) is the adopted
primary.

Targeting is toward *remaining* space rather than novelty alone: run-up zones
and barrier tops are derived from the mask so the agent is shaped toward the
specific places that are reachable but unvisited, not merely toward anything
new.

## 7. Jumping and vertical exploration

Derived from the engine, not scripted. No obstacle-specific walkthrough exists
anywhere in the code — the brief forbade that and it was not done.

Measured facts: standing rise 166 px; running rise 183 px; the rise increases
only above |x_vel| 4.5; reaching that takes 29 frames / 70 px of ground.
Barriers of interest are 172 px — i.e. **between** the two rise heights, which
is exactly why this failed silently for so long.

The fix is two-part shaping, both derived from those numbers:

1. **Run-up reward** — pay for taking off with speed (refuted alone, §3).
2. **Retreat shaping** — pay, once per barrier zone per episode, proportional to
   how far back toward `RUNUP_RUN_PX` (70) the agent moves from its own
   forward-most point in that zone. This is the part that supplies the missing
   gradient.

Retreat was also checked for *possibility* before being rewarded: the level
hard-clamps Mario to `viewport.x + 5` and the camera is one-way, so if the
camera had caught up there would be no room to back into and the whole approach
would have been impossible rather than merely unlearned. There is room.

Result: retreat/run-up reward was paid in **62%** of episodes (mean 0.74, max
3.00), and wall-endings fell to **10.8%** (23/213) against a 12–17% baseline.
Reduced, **not eliminated** — see §15.

## 8. Anti-farming

The failure mode was an episode that discovers nothing yet still profits.

Secondary reward (interaction, locomotion) is now **gated** during a discovery
drought and, past a per-episode cap, **exhausted** — it returns exactly 0.0
rather than a reduced trickle. The gate deliberately excludes coherent transit,
because crossing old ground to reach new ground is legitimate and must not be
punished.

Retreat reward is itself farmable in principle (walk backwards repeatedly), so
it is paid **once per zone per episode**, against the episode's forward-most
point in that zone, and capped at `RUNUP_RUN_PX`. Oscillating cannot earn it
twice.

Verified: **0 of 129** zero-discovery episodes were profitable.

## 9. Completion retention

Protocol: 500 episodes, seeds 20260911+, bare engine, sampled (not greedy)
actions, 1600 engine time units, headless. Thresholds were derived from the
baseline's own variance, not chosen: completion WARNING <0.4046, REGRESSED
<0.3412; progress WARNING <0.6777, REGRESSED <0.6406.

| Checkpoint | Completion (95% CI) | Progress | Verdict |
|---|---|---|---|
| 6.03M healthy seed | 0.438 [0.395, 0.482] | 0.690 | baseline |
| **Unconstrained 6.4M** | **0.060** [0.042, 0.084] | **0.280** | **the extinction this work exists to prevent** |
| anchorA (KL 0.13) | 0.402 [0.360, 0.446] | 0.700 | WARNING |
| anchorB (KL 0.08) | 0.492 [0.448, 0.536] | 0.730 | HEALTHY |
| anchorC (KL 0.08, replicate) | 0.438 [0.395, 0.482] | 0.670 | WARNING (progress) |
| runup (KL 0.08 + run-up) | 0.442 [0.399, 0.486] | 0.680 | HEALTHY |
| rehearsal B | 0.302 [0.263, 0.344] | 0.610 | REGRESSED |
| **retreat (KL 0.08 + retreat)** | **0.480** [0.436, 0.524] | **0.690** | **HEALTHY — chosen** |

anchorB and anchorC are the **same configuration** and scored 0.492 vs 0.438.
That spread is the honest measure of run-to-run variance here, and it is why no
single run is treated as proof (§15).

**Why retreat is the campaign start.** It is the only candidate that wins on
both axes at once: the most new pixels of any constrained run (359,394, vs
282,915 for anchorB) *and* a healthy retention verdict. Its completion (0.480)
is **not** distinguishable from anchorB's (0.492) — the confidence intervals
overlap almost entirely — so the choice rests on the 76,479-pixel exploration
margin, which is real, rather than on a completion difference the sample size
cannot resolve. Detail: castle door 240, flagpole 240; deaths goomba 137,
koopa 16, pit 44; timeouts 63; 239 of 240 completions finish inside the
original 401-unit clock.

## 10. PPO health

- Learning rate reduced to **2.5e-5** (`NORMAL_LR`). The inherited 1e-4 came
  from the legacy completion phase and is too high for a QA resume.
- Entropy coefficient is pinned to `ENT_COEF_BASE` on QA resume, so a resumed
  run does not silently inherit a decayed schedule position.
- Value-function warm-up available with optional actor freeze and an
  explained-variance release condition.
- Anchor KL per update on the winning configuration stayed controlled: 0.311 on
  the first update (the largest seen, immediately pulled back), then 0.053,
  0.087, 0.024, 0.036, 0.048, 0.073, 0.080, 0.043.

No reward-side hack was used to paper over an optimisation problem.

## 11. Coverage integrity

Denominator: **4,002,095** testable pixels, mask fingerprint `ded5cd19a477a8ed`.

It was corrected **twice**, and in both directions — once removing space Mario
cannot occupy, once adding space he can. It was never changed to make any
number look better, which is checkable: the corrections *lowered* every
candidate's coverage percentage (e.g. anchorB 64.80% → 62.68%).

The flag trigger is derived from the engine, not assumed: `Checkpoint(8504,
'11', 5, 6)` with default height 600, giving a 6×600 rect spanning y5–605.
Space beyond it is classified `CLS_BEYOND_FLAG` (anomalous), not counted as
testable.

Every coverage file stores a `config_hash` over format, grid, world,
frontier cell, fingerprint and total. A checkpoint cannot be paired with a
coverage file from a different mask without the launcher refusing.

When the mask changed, coverage files were **migrated, never edited in place**:
`tools/migrate_coverage.py` writes a copy with the same bitmap, a recounted
`covered_testable` and a re-stamped fingerprint. Verified on anchorB:
2,505,638/3,866,663 → 2,508,424/4,002,095, `load_verified OK`.

## 12. Resume safety

- `--dry-run-resume` runs the real selection and the real verification and
  exits without training, because the failure being guarded against is silent:
  resuming an older master looks exactly like a normal launch.
- Checkpoint and coverage are verified as a **pair** (timestep, fingerprint,
  total, hash). A mismatch aborts before anything is trained or written.
- Naming `--resume-from` explicitly is treated as a decision: a REGRESSED
  verdict is reported but not vetoed.
- The original 6M artifacts and the 6.03M healthy pair are byte-identical to
  their starting state; hashes are recorded in the worklog and re-verified in
  §14.

## 13. Anomaly validation

The classifier went from **25,356** flagged anomalies to **2** — not by
loosening it, but by correctly modelling four categories that were previously
conflated: real physics, expected engine states, classifier limitations, and
genuinely suspicious behaviour.

One genuine finding is preserved with its evidence in
`docs/objective2/ANOMALY_acceleration_leak.md`: `MAX_RUN_SPEED` is 800 and `RUN_ACCEL` is
20, but from a standstill walking and sprinting accelerate **identically**
(0.15/frame) — the run acceleration path is effectively dead. This is a real
engine behaviour, reported rather than worked around, and it is *why* the
run-up distance is 70 px rather than something shorter.

Raw evidence scripts are kept in `docs/objective2/evidence/` (jump physics, run-up, well
escape, well trace) and excluded from lint.

## 14. Validation results

- **ruff**: All checks passed.
- **mypy**: Success, no issues in 41 source files.
- **artifact integrity**: 23 artifacts in the manifest, **0 changed**.
- **protected hashes** — unchanged, as required:
  - `mario_brain_checkpoint.zip` `690d5702…`
  - `checkpoints_qa/pre_main_6032768/glitch_hunter_qa.zip` `7d2f3e37…`
  - `checkpoints_qa/pre_main_6032768/glitch_hunter_qa_coverage.npz` `2e4379c9…`
- **test suite**: **532 tests, 0 failures** (3 skipped), exit 0.
- **re-check after late edits**: the `ANCHOR_KL_TARGET` change and the new
  migrated-anchorB directory both landed while that suite was running, so the
  100 tests that could plausibly be affected (resume/stage selection, training
  callbacks, coverage migration, completion eval) were re-run afterwards —
  0 failures — and artifact verification was repeated: still 0 changed.
- **launch command dry-run verified** (§16), which confirmed three things
  worth stating separately, because each is a silent-failure mode that would
  otherwise only show up mid-campaign:
  - the REGRESSED 6.4M checkpoint is **automatically refused** with a warning
    and the launcher falls back to the healthy one;
  - automatic selection picks the 6.03M seed, **not** the retreat candidate —
    which is why `--resume-from` is mandatory in §16 rather than optional;
  - `--resume-from` pairs the candidate with **its own** coverage file
    (2,584,903 / 4,002,095, integrity OK), so the campaign does not silently
    start from the seed's map and discard 359,394 earned pixels.
- **headless equivalence**: 40/40 episodes byte-identical windowed vs headless,
  which is what licenses headless evaluation.
- Tests added this objective: run-up/retreat (10), anchor consolidation (6),
  coverage migration (5), rehearsal (9), flag trigger (7), secondary gate (3+).

## 15. Limitations and unproven claims

Stated plainly rather than smoothed over.

1. **Walls are reduced, not solved.** ~10.8% of episodes still end against a
   barrier. The retreat skill is learned but not reliable.
2. **Exploration under the KL constraint is ~25% slower** than unconstrained.
   This is a deliberate trade for keeping completion, not a defect — but it is
   a real cost and it means the campaign will take longer.
3. **UNPROVEN: long-run behaviour.** No constrained run has been observed past
   ~164k steps. Everything about how this behaves over a multi-million-step
   campaign is *extrapolation from milestone checks*, not measurement. This is
   the single largest open risk.
4. **Run-to-run variance is significant** (0.492 vs 0.438 on identical config).
   Any single milestone result should not be over-read.
5. **UNPROVEN: that coverage can be driven to exhaustion.** The best measured
   coverage is ~64.6%. Nothing here demonstrates the remaining ~35% is
   *reachable in practice*, only that it is reachable in principle per the mask.
6. The anomaly classifier is validated against known cases; novel anomaly
   classes are by definition untested.

## 16. Exact manual command to start the campaign

> **Historical.** These commands started the capped validation. Do not use them
> now: the campaign has been run and closed. Start Objective 3 only from the
> frozen pair named in the worklog's FINAL STATE block.

**I have not run this and will not — the campaign is yours to launch.** It opens
8 game windows.

Verify the resume selection first (trains nothing, writes nothing):

```
cd C:\Users\tanma\Desktop\glitch_hunter_project
venv_gpu\Scripts\python.exe train_agent.py --dry-run-resume
```

Then the **controlled validation run** — capped, not unrestricted:

```
venv_gpu\Scripts\python.exe train_agent.py ^
  --resume-from checkpoints_qa\candidate_retreat_6196608\glitch_hunter_qa.zip ^
  --anchor-kl ^
  --safety-cap-timesteps 6360000
```

Both commands above were **dry-run verified** (§14): the selection resolves to
6,196,608 paired with its own 2,584,903-pixel coverage map, integrity OK.
`--resume-from` is **required** — without it the launcher picks the 6.03M seed,
because candidate directories are deliberately not in the automatic pool.

`--anchor-kl` with no value uses `ANCHOR_KL_TARGET`, which is now **0.08**
(it was 0.13; 0.13 was refuted, §3). The cap gives ~164k steps — one milestone,
the same size as every experiment behind this report — so the first campaign
milestone is directly comparable to the evidence here.

**Only after reviewing that milestone** does the unrestricted campaign make
sense, by replacing the cap with `--unrestricted`.

**Stopping rule, from the measured variance:** one bad milestone → roll back to
the last good checkpoint and continue; the fixed anchor means one bad milestone
does not drag the next one down. **Two bad milestones in a row → stop and
review**, because at that point it is not variance and rolling back again will
not address it.

## 17. Readiness verdict

The brief asked for one of two outcomes: (A) genuinely ready for the
user-launched campaign, or (B) a precisely identified fundamental blocker.

**The answer is (A), with one scope limit stated explicitly: ready for a
capped validation milestone, not yet demonstrated for an unrestricted
multi-million-step campaign.**

What justifies (A):

- The original blocker is solved and measured. Completion extinction took the
  agent to 6.0%; the chosen checkpoint sits at **48.0%**, above the 43.8%
  baseline it started from, while adding **359,394** new pixels.
- The mechanism is understood, not merely observed to work — the cause was the
  discount horizon (§2), and the fix targets that cause.
- The safety properties needed for an unattended run exist and are tested:
  consolidation can only help or do nothing; the anchor is fixed so drift
  cannot compound; resume verifies checkpoint/coverage as a pair and refuses
  mismatches; coverage is monotone and its denominator is defensible from
  engine behaviour.
- Lint, types and artifact integrity are clean, and the protected artifacts are
  byte-identical.

What prevents an unqualified (A), and why it is a scope limit rather than a
blocker:

- **No constrained run has been observed past ~164k steps** (§15.3). This is
  not a known defect — it is an absence of evidence, and the only way to get
  that evidence is to run the campaign. That is precisely why §16 recommends a
  capped validation milestone first rather than `--unrestricted`.
- Wall failures (~10.8%) and coverage exhaustion (~64.6%, ceiling unknown) are
  open questions, but neither threatens the integrity of the run; both are
  measurable from campaign telemetry as it proceeds.

There is **no candidate for (B)**. Nothing found in this objective looks like a
fundamental blocker — every failure traced to a cause that was either fixed or
bounded and documented.

**Recommendation:** launch the capped validation run in §16, review that one
milestone against the evidence in §9, and only then decide on the unrestricted
campaign. Apply the stopping rule: one bad milestone → roll back and continue;
two in a row → stop and review.

**I have not launched it, and will not without being asked.**
