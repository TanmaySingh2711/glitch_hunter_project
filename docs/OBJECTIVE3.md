# Objective 3 — Automated Bug Reporting System

When the QA agent sees the game break a rule, the moment is frozen, saved as
evidence, reported (GIF, Markdown, PDF), replayed, and shown on the dashboard.

## Start

1. Used the final **16M QA brain** and Objective-2 artifacts as protected read-only inputs.
2. Created two game variants: `mario_clean` and `mario_bugged`.
3. Created a standard bug/incident data model.
4. Connected anomaly detection to an automated reporting pipeline.

## During

5. Added exact trigger-state preservation.
6. Added exact trigger-frame screenshots and context GIFs.
7. Added trajectory evidence and replay/reproduction.
8. Added severity, confidence, duplicate detection, and occurrence tracking.
9. Created self-contained evidence bundles for every incident.
10. Added automatic Markdown and PDF reports.
11. Added dashboard support for `Testing Stopped – Bug Found`, incident details, screenshot/GIF viewing, report download, and incident history.
12. Added safe pause/resume, multi-incident handling, retries, and failure recovery.
13. Performed hard audits and fixed false positives, stale-frame issues, path/security issues, reconnect problems, file-lock problems, and other reporting flaws.
14. Completed synthetic E2E validation and real browser validation.

## End

15. Injected six benchmark bugs into `mario_bugged`: stair clipping, pipe clipping, ceiling clipping, invisible wall, false Goomba hit, abnormal sky jump. On 2026-10-03, at the owner's request, three were moved or reshaped so the dashboard's drawn routes meet them and still finish: the pipe clip, the invisible wall, and the false Goomba hit, which became the stomp from too far (bug 5).
16. Added generic detectors for all six bug types.
17. Verified **0 false alarms on the clean game**.
18. Verified that all six bugs could be detected, reproduced, and automatically reported.
19. Proved the complete workflow: **Bug detected → Testing pauses → Screenshot/GIF captured → Replay → Markdown/PDF generated → Bug Tracker updated → Report downloadable**
20. Final tests and integrity checks passed, while Objective-2 protected artifacts remained unchanged.

**Flow:** 16M QA brain → Automated detection/evidence/reporting pipeline → Dashboard + reports → 6 benchmark bugs detected → **Fully automated QA reporting system**

## How it works

`detector fires → Detection frozen (exact frame, trace, state, action log) → incident saved → testing pauses → GIF/MD/PDF rendered → replayed in a separate process → Bug Tracker updated`

* **Evidence bundle** (`generated/incidents/INC-…/`): `incident.json`, `trigger.png`,
  `trajectory.json`, context frames, `context.gif`, `report.md`, `report.pdf`,
  `reproduction.json`, `manifest.json` (hashes). Written atomically, read-only, never overwritten.
* **Where to fix it** (`reporting/fix_hint.py`): every report, and the dashboard's
  *Bug found* card, names the lines in the game's own code that cause the bug
  and the exact edit for each (delete these lines / change this line to ...).
  It is inferred from the incident's record (the drawn level, the colliders in
  view, the detector's measurements) and the game's source, read with `ast`;
  the list of injected bugs is never read. Labelled a lead, not a proven cause.
  Proved on the six benchmark bugs (`tests/test_fix_hint.py`): every suspect
  line is the injected code, and with all eleven suggested edits applied to a
  copy of the bugged game, every bug's scenario plays exactly like the clean
  game.
* **Replay verdicts:** reproduced / reproduced_state_only / not_reproduced / not_possible.
* **Duplicates:** same game, detector, kind and site → the same incident, count raised.
* **Dashboard:** every detection pauses testing; *Resume testing* continues; the Bug
  Tracker shows this session's bugs (Reset, or refreshing the page, empties it;
  nothing is deleted), and Bug History lists every incident and run report on disk.
  Only there can saved evidence be deleted: a Delete button on each row and a
  Clear history button on each list, both asked first. A delete removes the
  whole bundle and its later sightings, so the same bug found again is saved
  as a new incident. Three things are kept: the bug or run Live Testing is
  stopped on, a report still being written, and anything asked for by another
  website (`tests/test_history_delete.py`).
  Pick "Clean game" or "Bugged game" on the Live Testing tab. The five-step
  list in Live status shows each automatic stage as it completes (bug
  detected, evidence and replay, report ready), from the incident's own
  manifest; a *Bug found* card shows the bug and its files.
* **Clean-run report:** on the clean game, testing also stops when a run ends.
  A run that reaches the castle gets a run report in `generated/run_reports/RUN-…/`
  (`reporting/run_report.py`): `run.json`, `final.png`, `finish.gif`,
  `report.md`, `report.pdf` - "No Bugs Found" when no detector fired. It
  covers that run and those 13 detectors only; it does not prove the game
  bug-free.

## The six benchmark bugs (`games/mario_bugged/INJECTED_BUGS.json`)

| # | Bug | Where | Caught by (generic rule) |
|---|---|---|---|
| 1 | Stair clipping | top blocks of the first staircase's two columns | `clip_into_step` |
| 2 | Pipe clipping | pipe 4: its collider starts 32 px below its drawn top | `clip_into_pipe` |
| 3 | Ceiling clipping | lone brick at x 5058: no collision while rising | `clip_into_block` |
| 4 | Invisible wall | undrawn 1-tile collider at x 3360 | `invisible_collision` |
| 5 | Stomp from too far | goomba14's stomp box reaches 36 px above its head | `stomp_without_contact` |
| 6 | Abnormal sky jump | jump taken at x 250–480: 2.2× take-off | `impossible_jump` (+ `above_world`) |

Every changed game line carries `INJECTED BUG <id>`; the whole clean→bugged
diff is pinned by hash; `mario_clean` stays byte-identical to its pin.

## Detectors

* **Engine invariants** (from the start): below the world, above the world,
  impossible speed, score or coins going backwards.
* **Collision and jump-physics invariants** (`reporting/collision_invariants.py`),
  checked everywhere against what is **drawn** (`reporting/level1_design.json`
  + live bricks/boxes/enemies), never against the game's own colliders:
  solid penetration > 6 px · collision with an undrawn collider · hit with no
  contact (≤ 2 px) · stomp with no contact (≤ 2 px; added 2026-10-03 for the far
  stomp) · rise faster than 12.5 px/frame or higher than 258 px.

## Validation

| Check | Result |
|---|---|
| Clean game, 101 brain episodes + dashboard runs | **0 false alarms** |
| Clean game, 80 more drawn runs with the stomp check (2026-10-03) | **0 false alarms** |
| Bugged game, one seeded dashboard run (`tests/test_bug_incidents.py`) | all 6 bugs stop testing, each **reproduced**, GIF/MD/PDF done |
| Bugged game, 40 drawn dashboard runs (2026-10-03) | 30 reached the castle; 39 met at least two bugs, 4 met all six |
| Bugged game, 40 sampled episodes | caught every clip deeper than 6 px; never fired without the bug |
| Pipeline validation tool | 37/37 checks |
| Full quality gate | lint, types, 730 tests, coverage 90.64%, Objective-2 artifacts unchanged |

## Problems found and fixed

* Audit: frame taken too late, placeholder readings reported as bugs, file
  locks on Windows, lost captures, reconnect state, path traversal → all fixed and tested.
* Detectors: 5 clean-game false alarms while building (x-before-y collision
  order, ledge edge, smashed bricks, engine speed reset, double jump report) → fixed at the cause, pinned as tests.
* Bug placement: several first designs trapped the brain (e.g. a wall it never
  jumps over) → moved/redesigned until the dashboard route meets all six.

## Limitations

* Clips shallower than 6 px are deliberately not reported.
* The designed-solids file is for Level 1-1 only (`tools/build_level_design.py`).
* Every dashboard run takes a new route (moves drawn from the brain's own
  policy at temperature 0.5), on both games. On the clean game 77% of 140
  measured runs reached the castle, each by a different route. On the bugged
  game 30 of 40 measured runs reached the castle (31 of 40 on the clean game,
  same seeds), 39 of 40 met at least two of the six bugs and 4 met all six; a
  run is not guaranteed to meet every bug. A run stuck in a trap (the pit
  between the two pyramids, pipe 4) is ended after 200 steps without progress.
* "Where to fix it" is a lead, not a proven cause: it reads Level 1-1's level
  and player code by function name (`level1.py`, `mario.py`).
* The sky-jump landing leaves more agents stuck at pipe 4 in sampled play.

## Complete project flow

**Objective 1:** Game → RL setup → PPO training → **6M autonomous brain**

**Objective 2:** 6M autonomous brain → QA exploration + glitch-hunting system → training + audits/recovery → **16M final QA brain**

**Objective 3:** 16M QA brain → automated bug evidence/reporting pipeline → benchmark bug detection → **complete autonomous QA reporting system**

[Objective 1](OBJECTIVE1.md) · [Objective 2](OBJECTIVE2.md) · [Objective 3](OBJECTIVE3.md)
