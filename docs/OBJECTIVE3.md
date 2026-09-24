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

15. Injected six benchmark bugs into `mario_bugged`: stair clipping, pipe clipping, ceiling clipping, invisible wall, false Goomba hit, abnormal sky jump.
16. Added generic detectors for all six bug types.
17. Verified **0 false alarms on the clean game**.
18. Verified that all six bugs could be detected, reproduced, and automatically reported.
19. Proved the complete workflow: **Bug detected → Testing pauses → Screenshot/GIF captured → Replay → Markdown/PDF generated → Bug Tracker updated → Report downloadable**
20. Final tests and integrity checks passed, while Objective-2 protected artifacts remained unchanged.

**Flow:** 16M QA brain → Automated detection/evidence/reporting pipeline → Dashboard + reports → 6 benchmark bugs detected → **Fully automated QA reporting system**

## How it works

`detector fires → Detection frozen (exact frame, trace, state, action log) → incident saved → testing pauses → GIF/MD/PDF rendered → replayed in a separate process → Bug Tracker updated`

* **Evidence bundle** (`incidents/INC-…/`): `incident.json`, `trigger.png`,
  `trajectory.json`, context frames, `context.gif`, `report.md`, `report.pdf`,
  `reproduction.json`, `manifest.json` (hashes). Written atomically, read-only, never overwritten.
* **Replay verdicts:** reproduced / reproduced_state_only / not_reproduced / not_possible.
* **Duplicates:** same game, detector, kind and site → the same incident, count raised.
* **Dashboard:** every detection pauses testing; START TESTING resumes; the Bug
  Tracker shows this session's bugs (Reset empties it; nothing is deleted).
  Pick "Mario Game (Cleaned)" or "Mario Game (Bugged)" from the list.

## The six benchmark bugs (`mario_bugged/INJECTED_BUGS.json`)

| # | Bug | Where | Caught by (generic rule) |
|---|---|---|---|
| 1 | Stair clipping | top blocks of the first staircase's two columns | `clip_into_step` |
| 2 | Pipe clipping | pipe 4: only its left rim is solid | `clip_into_pipe` |
| 3 | Ceiling clipping | lone brick at x 5058: no collision while rising | `clip_into_block` |
| 4 | Invisible wall | undrawn 2-tile collider at x 4412 | `invisible_collision` |
| 5 | False Goomba hit | goomba14's hurt box 36 px too big | `hit_without_contact` |
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
  contact (≤ 2 px) · rise faster than 12.5 px/frame or higher than 258 px.

## Validation

| Check | Result |
|---|---|
| Clean game, 101 brain episodes + dashboard runs | **0 false alarms** |
| Bugged game, dashboard route | all 6 bugs stop testing, each **reproduced**, GIF/MD/PDF done |
| Bugged game, 40 sampled episodes | caught every clip deeper than 6 px; never fired without the bug |
| Pipeline validation tool | 36/36 checks |
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
* The dashboard plays greedily: one fixed route per game; on the bugged game
  the false Goomba hit ends every greedy episode.
* The sky-jump landing leaves more agents stuck at pipe 4 in sampled play.

## Complete project flow

**Objective 1:** Game → RL setup → PPO training → **6M autonomous brain**

**Objective 2:** 6M autonomous brain → QA exploration + glitch-hunting system → training + audits/recovery → **16M final QA brain**

**Objective 3:** 16M QA brain → automated bug evidence/reporting pipeline → benchmark bug detection → **complete autonomous QA reporting system**

[Objective 1](OBJECTIVE1.md) · [Objective 2](OBJECTIVE2.md) · [Objective 3](OBJECTIVE3.md)
