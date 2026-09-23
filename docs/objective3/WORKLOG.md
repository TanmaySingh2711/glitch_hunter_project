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
