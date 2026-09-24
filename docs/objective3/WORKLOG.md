# Objective 3 — working log

Read this first after any context compaction or session restart. It holds the
user's brief for Objective 3 (condensed), the design decisions with their
evidence, what has been built, and where work stands. Keep it current.

## The brief (condensed from the user's 2026-09-23 prompt)

Turn the QA system into an automated glitch-evidence and bug-reporting
pipeline:

    detected QA event -> preserve evidence -> collect context -> durable incident
    -> Markdown + PDF report -> dashboard "Testing Stopped - Bug Found"
    -> user reviews / downloads -> everything preserved

Before that: two game variants, `mario_clean` (canonical baseline, never gets
bugs) and `mario_bugged` (future bugs go ONLY here, and only when the user
names them). Right now BOTH must be clean and behave identically. Do not
inject, invent or ask about bugs yet.

Required pieces (numbered as in the brief): 1 Objective-2 final state is a
read-only input; 2 one canonical, versioned incident schema; 3 an
event-driven detector -> reporting boundary that keeps detector semantics;
4 freeze the trigger state before execution destroys it; 5 the exact trigger
frame, lossless, no off-by-one; 6 supporting technical context; 7 Markdown
report (facts vs interpretation, no invented root cause); 8 real PDF report;
9 deterministic collision-free storage; 10 dashboard Bug Found state from
backend truth; 11 incident summary in the dashboard; 12 safe open/download
(validate ids, no arbitrary paths); 13 safe pause, user-controlled resume,
resume never overwrites evidence; 14 many incidents; 15 evidence-based
duplicate control with occurrence counts; 16 hard tests; 17 real end-to-end
proof (synthetic event, clearly labelled); 18 documentation; 19 GIF context +
exact trigger frame, buffering must not change behaviour; 20 trajectory +
honest reproduction status; 21 severity / detector confidence / reproduction
kept distinct, fingerprint, occurrence count; 22 immutable self-contained
bundle with integrity hashes, versioned re-renders; 23 rendering failure never
loses the incident, visible failures, no recursive incident loops;
24 incident history in the dashboard.

Invariants: the final Objective-2 brain, its coverage and all historical
checkpoints stay untouched; no retraining, no coverage campaign; observing
must not change game behaviour; evidence must match the real trigger; no
fabricated facts; synthetic events always labelled synthetic and never
presented as real bugs; Objective 3 testable without retraining.

Standing rules from earlier sessions still apply: never start training
without an explicit request, ask before commits (and never add a Claude
co-author trailer), never touch the D: drive, derive numbers from evidence.

Final report to the user: 26 numbered sections plus full-suite / ruff / mypy /
artifact results, ending with exactly "OBJECTIVE 3 CLOSED — READY FOR
DELIBERATE BUG INJECTION" or "OBJECTIVE 3 NOT READY — <reason>".

## Key facts found while inspecting (2026-09-23)

* The only detector is `CustomMarioEnv._detect_glitches`, run once per
  engine SUBSTEP: below_world, above_world, speed, score_drop, coin_drop,
  each reported at most once per episode. Its message reaches the dashboard
  via `info['glitch_alert']` held for a TTL of 4 substeps (SkipObservation
  keeps only the last substep's info).
* The dashboard's stream frame is rendered AFTER the whole 4-substep agent
  step, i.e. 0-3 frames after a detection. A screenshot taken there would be
  off by up to 3 frames - the trigger frame must be captured inside the
  substep that detected it.
* The engine is deterministic: game time is the env's `fake_time`; the only
  wall-clock read (`tools.Control.update`) is never called by the env. So an
  episode is a pure function of the per-substep action sequence (+ the QA
  clock holds), which is what makes replay-based reproduction possible.
* The game is imported as the top-level package `data` from the game
  directory, so one process can host only one game variant (sys.modules).
* `tests/test_hud_timer.py` pins LEGACY_NOOP_600_SHA256, the hash of 600 raw
  engine frames holding NOOP - a direct behavioural fingerprint of the game.
* Dependencies added: fpdf2 2.8.8 (PDF) and Pillow 12.3.0 (GIF; fpdf2 needs
  it anyway). cv2 5.0 is already present for PNG.

## Design decisions (with the evidence behind each)

* Evidence context window = 2 x (longest run-up 45 frames + longest real jump
  arc 77 frames, from exploration_data/jump_arcs.npz) = 244 substeps = 61
  agent steps (~4 s): long enough to hold the whole manoeuvre that led to the
  trigger and the approach before it.

* Two variants = two directories, `mario_clean/` (git mv of mario_clone,
  history kept) and `mario_bugged/` (byte copy). The upstream package name
  `data` is kept, so ONE process hosts ONE variant; `claim_game_variant`
  refuses a second. Clean tree pinned: CLEAN_GAME_TREE_SHA256 = b1f6b184...
  = mario_clone's tree hash measured before the rename (48 files). Bugged may
  differ only in files a bug in mario_bugged/INJECTED_BUGS.json names (empty).
* Evidence is OPT-IN on the env (`enable_evidence`): off, step/reset are
  unchanged (pinned: identical obs/info/rewards on vs off). On: a per-frame
  trace ring (244), the episode's per-frame action log + clock top-ups, and a
  frozen `Detection` (reporting/events.py) at the substep a detector fires -
  including the full-res frame copied from the display surface right then.
* Detector semantics unchanged: same 5 checks, same once-per-episode latch,
  same TTL delivery of info['glitch_alert']; `report()` gained metrics and an
  optional key. Extra detectors (only `SyntheticProbe`, marked synthetic) go
  through the same `report()`.
* Pipeline (reporting/pipeline.py): capture() runs on the game thread and
  writes raw evidence (trigger.png, trajectory.json, context_frames.zip,
  incident.json LAST) before returning; a worker thread then renders
  context.gif, runs the replay, renders report.md and report.pdf, with
  RENDER_MAX_ATTEMPTS=3 per artifact, then finalizes (manifest read-only).
* Store (reporting/store.py): `incidents/INC-<utc>-<6 hex>/`, id claimed by
  exclusive mkdir, atomic writes with a bounded rename retry (Windows holds),
  never overwrite, read-only files, versioned re-renders (report.v2.md),
  append-only `occurrences.jsonl`, `_incomplete/` for crash leftovers.
* De-dup (reporting/analysis.py): same game tree + detector + kind +
  synthetic flag AND colliders within DEDUP_RADIUS (54 x 91 px = largest
  collider + one frame of max motion). Otherwise a new incident.
* Severity per kind (none is critical); detector confidence from evidence
  (invariant vs threshold margin, downgrades); reproduction separate.
* Reproduction (reporting/reproduce.py): replays the recorded per-frame
  actions + clock top-ups from reset in a separate headless process;
  compares every trace frame, the detector on the trigger frame, and the
  trigger pixels. Statuses: reproduced / reproduced_state_only /
  not_reproduced / diverged / not_possible / timeout / error.
* Dashboard: plays the frozen Objective-2 brain (hash-checked against
  FINAL_OBJECTIVE2.json), evidence on; a NEW incident pauses the game thread
  before the next step (`bug_found`, pause_reason); only Start/Reset clear
  it. Routes: /api/status, /api/incidents, /api/incidents/<id>,
  /incidents/<id>/<allow-listed file>, /incidents/<id>/bundle.zip.
  `python app.py [--game mario_bugged] [--synthetic-probe X ...]`.

## Progress log

* 2026-09-23: variants created and verified (12 tests: pin, no undeclared
  difference, per-process isolation, and a subprocess fingerprint of each
  variant - NOOP-600 frames = LEGACY_NOOP_600_SHA256, geometry, 725-frame
  scripted physics/collision run incl. a death, rendering - all identical).
* reporting/ package written: events, variants, schema, store, analysis,
  provenance, evidence, render, reproduce, pipeline. Dashboard backend,
  service, app routes, frontend (banner + history + links) wired.
* Tests added: test_game_variants (12), test_incident_capture (10),
  test_incident_pipeline (54), test_incident_reproduce (10),
  test_dashboard_incidents (27), test_objective2_protection (3). All pass.
* tools/incidents.py (list/show/verify/rerender/reproduce/recover) and
  tools/validate_incident_pipeline.py (real E2E) written.
* FIRST REAL E2E (scratchpad store): 36/36 checks in 31.6 s - approved brain
  on the clean game, two synthetic probes: stop before the next step, raw
  evidence on disk at the stop, reports rendered, replay "reproduced" (QA
  mode, with clock holds), routes serve/deny correctly, resume -> second
  NEW incident, later episode -> occurrence (count 2, no stop), Objective-2
  files byte-identical.

* WINDOWED E2E (`--windowed`, a real game window, replay still headless):
  36/36 in 29.7 s. The windowed trigger frame and the headless replay's frame
  are pixel-identical, so "reproduced" holds across that difference too.
* Replay cost measured: 1.48 ms per engine frame (3 x 3,000 frames), ~0.65 s
  warm start; recorded in config beside the timeout it justifies.

## Hard-audit findings (all fixed, all pinned by tests)

1. Stream frame trails the trigger by 0-3 frames -> capture in the substep.
2. DETECTOR FALSE POSITIVE: on a frame the engine could not be read,
   UNKNOWN_STATE_INFO's placeholder zeros were compared with the last real
   score/coins - reproduced as "Coin total went backwards (3 -> 0)". Fixed in
   `_detect_glitches`: skip score/coin checks on such frames and keep the last
   real baseline. glitch_alert feeds only the dashboard log, never the reward,
   so Objective 2 is unaffected. Test: test_glitch_detection.
3. Id / file-name regexes accepted non-ASCII digits (`\d`) and a trailing
   newline (`$`) -> ASCII classes + fullmatch.
4. Windows cannot rename over a file another handle holds open (a browser
   downloading manifest.json) -> bounded rename retry in the store.
5. A capture that fails outright left no trace -> capture_failures.jsonl.
6. Placeholder readings got only a one-level confidence downgrade -> low.
7. `tools/incidents.py list` would have run recovery (moving files) -> the
   read-only commands construct the pipeline with recover=False.
8. The report said "Detected (UTC)" for what is the capture instant ->
   "Captured (UTC)" + "Engine time (exact)".

9. The slow E2E test ran the tool in-process; its final Reset closed the
   shared pygame display and 118 later tests errored -> it now runs the tool
   in its own process, as it is used for real.
10. Replaying a real incident happens in a child process, so coverage never
    saw reproduce.py (29%, total 89% < the 90% floor) -> the same verdicts are
    also asserted by calling replay() in-process (89%, total 90.52%). The floor
    was not lowered.
11. A browser reconnecting after a bug stop rewrote pause_reason to "connect";
    a bundle deleted by hand broke /api/incidents; detail() 404'd on an
    incident another process wrote -> all three fixed and tested.

## Status (2026-09-23)

DONE. `tools/check.py --full`: lint, mypy (Windows + Linux), 685 passed /
18 skipped (the checkout-dependent skips that were there before) with
coverage 90.52% >= 90, artifacts 37 / 0 changed. uv.lock refreshed with
`uv lock` (added fpdf2, fonttools, defusedxml; nothing else changed;
`uv lock --check` clean). Nothing committed - the user decides.

## 2026-09-24: the six benchmark bugs injected into mario_bugged

The owner specified six bugs and left the locations to the implementer.
mario_clean untouched (pin holds); the final brain was only loaded for inference.

**Method.** The approved brain was run on the clean game as the dashboard
plays it (greedy) and with sampled actions (40 episodes, seeds 5000-5039),
recording Mario and every enemy per frame. Key fact: greedy play on this
deterministic engine is ONE fixed route (clean: level complete in 434 steps,
every episode; the real dashboard loop confirmed the same on the bugged game
over 6 consecutive episodes). A bug that stalls or kills early hides every
later bug from the dashboard, so each bug was placed where that route - with
the earlier bugs already in place - naturally touches such geometry.

**Final placement (dashboard route order):** open-sky-jump (take-off at
x 250-480) -> pipe-clip (pipe 4) -> invisible-wall (x 4412) -> ceiling-clip
(brick at 5058) -> stair-clip (step4/step5 top blocks) -> false-goomba-hit
(goomba14, kills Mario 33 px above it). The greedy episode ends there, every
time.

**Redesigns (each measured, then fixed):**
1. Wall at x 4220 / 3126 etc.: the brain never jumps when blocked (it holds
   right forever) and most spots stalled or killed the run; x 3126 worked but
   stood inside a bush (background decoration) - not "visually empty". Final
   x 4412: no bush (decorations end 4323, start 4611), nothing overhead.
2. Stair: sinking one tile made the greedy route drop into the pyramid gap and
   stall (the gap is a clean-game trap too, 2/40 sampled). Two tiles (86 px):
   Mario climbs out; no stall on any route tried; clean-game gap trap gone.
3. Sky jump: x2.2 alone peaked at y 91 (the brain taps jump); the defect also
   skips the early-release cut -> y -294 on a tap or a hold (detector: -200).
4. Ceiling: a full one-way brick let Mario slip past its side while falling;
   final: no collision only while moving up.
5. Pipe: left rim only (21 px) instead of half-width, for a visible fall-in.
6. Goomba: sideways-only margin on goomba15 never fired on the route; final:
   goomba14 with a 36 px margin on every side.

**Validation (final code).**
* tests/test_injected_bugs.py: each bug normal on clean, defective on bugged,
  repeatable; 8 control spots identical on both games.
* Real dashboard loop, 6 episodes: all six bugs every episode; identical route.
* Objective 3 on the real dashboard stack, no probe: stopped on "Mario far
  above the level" (above_world, not synthetic), GIF/MD/PDF rendered, replay
  reproduced, bundle verifies, provenance = mario_bugged + the six ids; 900
  steps later still one incident (seen 3x), no second stop. The other five are
  not detected by any current detector (detectors for them are feasible:
  collider vs DRAWN solids, blocked with nothing drawn, hit with a gap).
* 40 sampled episodes (same seeds as clean): triggered sky 27, wall 21, pipe
  16, stair 9, ceiling 8, Goomba 4 (all six in one episode: 1). Of episodes
  reaching each spot: pipe 16/27, wall 21/22, ceiling 8/13, stair 9/9, Goomba
  4/8. Oracles fire 0 times on the 40 clean episodes. Cost to the agent:
  completion 21/40 -> 4/40; stalls at pipe 4's (unchanged) left face 3 -> 9
  (sky-jump landings leave no run-up), 2 at the invisible wall, 0 in the
  pyramid gap (was 2); Koopa deaths 1 -> 5 (the wall turns the Koopa back).
* Integrity: every changed line block carries its bug's marker; the whole diff
  is pinned in the manifest; geometry differs only in the declared colliders;
  the scripted run is identical for 338 frames and first differs inside the
  open-sky-jump zone.

## 2026-09-24: generic detectors for all six benchmark bugs

Owner's brief: detect the remaining five injected bugs - and, added
mid-task, the open-sky jump too - with GENERIC invariants (no injected x/y),
no clean-game false positives, same Detection -> incident -> stop -> evidence
-> report -> replay flow, above_world untouched, the brain untouched.

**Design.** One module, `reporting/collision_invariants.py`, four rules run
by the env while evidence is on (§4a of README): solid penetration
(clip_into_step / _pipe / _ground / _block), collision with nothing
(invisible_collision), hit without contact (hit_without_contact), impossible
jump (impossible_jump). The reference for static solids is the level DESIGN,
`reporting/level1_design.json` (37 rects: 4 ground, 6 pipes, 27 steps),
generated from mario_clean by tools/build_level_design.py and tested every
run to equal mario_clean's colliders and to be drawn (0 sky pixels inside,
inset by the tolerance). Bricks, ? boxes and enemies are read live (sprites
are drawn at their rects). The running game's colliders are never the
reference; rule 2 uses them only as proof that a collision happened.

**False positives found while building (each fixed at its cause, each pinned
as a regression test; no tolerance was loosened):**
1. Side stop at pipe 4 in clean play: level1 resolves x before y, so the pipe
   stopped Mario 3 px lower than where the frame ended -> the side probe spans
   both heights.
2. Standing on the last pixel of pipe 1: my foot probe trimmed 1 px per side
   -> full width, as the engine's own test.
3. 16 "head bump with nothing above" in 100 clean episodes: big Mario smashes
   the brick in the same frame -> judge against both frames' blocks.
4. 28 "stopped with nothing beside" at ~1 px/frame: with sprint held and no
   direction the engine zeroes x_vel by itself (RUN_ACCEL 20), no collision ->
   rule 2 now requires a collider actually touching Mario there, and only
   then asks whether it is drawn.
5. One sky jump gave two impossible_jump incidents (speed, then height) ->
   one report per episode for the rule.

**Results (final code).**
* Clean game, approved brain: 101 episodes (greedy + 100 sampled, seeds
  7000-7099): 0 detections of any new kind. Clean jump envelope: 10.5
  px/frame, 183 px (limits 12.5, 258). Real dashboard stack, 1300 steps: 0
  stops, 0 incidents. tools/validate_incident_pipeline.py: see gate run.
* Bugged game, scripted scenarios (tests/test_injected_bugs.py): each bug
  caught by exactly its own kind; every clean scenario and every control spot
  silent.
* Bugged game, 40 sampled episodes vs the geometric oracles: sky 27/27,
  wall 21/21, Goomba 4/4, stair 8/9, ceiling 7/8, pipe 11/16; 0 detections
  without the bug. Every miss was a 1-6 px graze on one axis (inside the 6 px
  tolerance, which exists because e.g. a pipe body is drawn narrower than its
  lip).
* Bugged game, real dashboard stack, 1300 steps: stopped on impossible_jump,
  above_world, clip_into_pipe, invisible_collision, clip_into_block,
  clip_into_step (x2: both columns), hit_without_contact - every incident
  reproduced, GIF/MD/PDF rendered, bundle verified; later episodes only
  counted (3-6 sightings each), no new stops.
* Cost: +0.16 ms per engine frame, evidence on only.
