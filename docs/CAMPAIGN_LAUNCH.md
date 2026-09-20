# Campaign launch — corrected spec and monitoring plan

## The banner you pasted did not match this project's real state

Three numbers in it were wrong, checked against the actual files on disk and
the code that reads them:

| Field | You pasted | Actually is | Why it matters |
|---|---|---|---|
| Resume Brain | 6,400,000 steps | **6,196,608 steps** (`candidate_retreat_6196608`) | 6,400,000 is the checkpoint that **regressed** (6.0% completion retention). The launcher auto-refuses it and falls back to the healthy seed unless told otherwise — verified live by dry run. |
| Testable/Coverable Pixels | 4,013,723 | **4,002,095** | 4,013,723 is the denominator from *before* the flag-trigger correction (docs/OBJECTIVE2_HANDOFF.md §11). Training against the old number would silently mis-score coverage. |
| Safety Cap | NONE (unrestricted) | **6,360,000** (a ~164k-step validation milestone) | Section 17 of the handoff report's readiness verdict is explicit: ready for a capped milestone, not yet proven unrestricted. Nothing in this project auto-stops an unrestricted run — see "What actually happens if something goes wrong" below. |

Also: "Running milestone audit..." isn't a step this codebase performs — no
such function exists in `train_agent.py` or `training/callbacks.py`. If you
saw that line somewhere, it wasn't from this project's current code.

`World Raster Pixels: 5,452,200` was correct as pasted (the level's fixed
9087×600 area — informational only, never used as a denominator).

## The command to run

```
cd C:\Users\tanma\Desktop\glitch_hunter_project
venv_gpu\Scripts\python.exe train_agent.py ^
  --resume-from checkpoints_qa\candidate_retreat_6196608\glitch_hunter_qa.zip ^
  --anchor-kl ^
  --safety-cap-timesteps 6360000
```

This is step 1 of the real campaign, not a toy experiment — but it is
deliberately capped. Per the handoff's readiness verdict, the reason is not a
known defect: no constrained run has been observed past ~164k steps, and the
only way to get that evidence is to run one. Once this milestone is reviewed
against the table in handoff §9, the next command drops the cap.

## What the corrected startup banner will actually read

```
=============================================
GLITCH HUNTER — QA EXPLORATION TRAINING
=============================================

Resume Brain:              6,196,608 steps
Training Goal:              SPATIAL COVERAGE COMPLETION
Fixed Final Step Target:    NONE

World Raster Pixels:       5,452,200
Testable/Coverable Pixels: 4,002,095
Already Covered:           2,584,903
Coverage:                  64.59%
Remaining Testable:        1,417,192

Bootstrap Known:           1,872,441
QA Discovered:                712,462
Anomalous Pixels:                   0

Reward Mode:                QA Exploration
Lifecycle:                  EXPLORE -> COMPLETE
Workers:                    8

Checkpoint Every:          400,000 steps
Next scheduled checkpoint: 6,400,000  (beyond this run's cap -
                            the final save happens at the safety cap instead)

Safety Cap:                6,360,000  (~164k-step validation milestone)
Success Condition:         4,002,095 / 4,002,095
=============================================
```

## What "monitoring" means here, concretely

Nothing in this codebase auto-stops a run on a soft problem. Checked directly
in `training/callbacks.py`: the watchdog that detects reward collapse only
**logs a warning and continues** ("Continuing training (entropy should help
recovery)"); `StagnationCallback` is detect-only by design. The hard stops
that do exist — resume verification, the artifact-hash check, the safety-cap
itself — are already in the launch command above. Everything past that is on
me, as you asked.

I'm watching `logs/train.log` (the same file every run already writes to, no
matter which terminal starts the process) for:

- **Hard failures** — `Traceback`, `Error`, `CRITICAL`, the process dying, an
  artifact hash coming back `CHANGED`.
- **`WATCHDOG ALERT`** — logged but not acted on by the code; I will act on it
  if it's not a brief, recovering blip like the one in the retreat run's own
  log (peak 63.3 -> avg -0.1 at step 6,157,312, which self-recovered).
- **Coverage moving the wrong way** — the `[COVERAGE]` line's covered-pixel
  count should only ever increase (coverage is monotone by design); a drop
  means something is corrupting the map, not exploring it.
- **The `[LIFECYCLE]` line's death/level_complete balance collapsing** —
  e.g. level_complete going to zero for an extended stretch, which is what
  completion extinction (handoff §2) looks like while it's happening, not
  after.
- **The run ending before its cap** for any reason other than the safety cap
  itself.

If I see something I judge wrong, I will stop the run directly (I can do this
regardless of which terminal launched it, the same way the 8 worker windows
are visible to process tools), tell you exactly what I saw and why, and then
resume from the **last checkpoint the retention/artifact checks still call
healthy** — never from the failing step itself. That is the same rollback
rule from the handoff: one bad milestone, roll back and continue; two in a
row, stop for a real review rather than rolling back again.

I am not launching this. Run the command above yourself; I'll be watching the
log as soon as it starts.
