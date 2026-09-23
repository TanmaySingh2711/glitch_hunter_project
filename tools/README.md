# tools/

Command-line scripts run by hand, each with one job. Every one is read-only
towards checkpoints unless its row says otherwise, prints what it did, and
refuses (with the reason) rather than guessing when an input is missing.
`python tools/<name>.py --help` shows the options.

## Everyday

| Tool | What it does | Writes |
|---|---|---|
| `check.py` | Every quality gate CI runs: ruff, mypy, tests, artifact hashes. `--full` adds the slow tests and the coverage floor. | nothing |
| `verify_artifacts.py` | Re-hashes the protected artifacts (the 6M brain, masks, baseline) against `artifacts.json`. | nothing (`--record` rewrites the manifest) |
| `benchmark_step.py` | Milliseconds per agent step for the bare engine, the legacy wrapper and the QA wrapper; `--profile` shows where the time goes. | nothing |
| `evaluate_completion.py` | Can a checkpoint still finish Level 1-1? Plays the frozen protocol and compares with the 6M baseline: HEALTHY / WARNING / REGRESSED. | `evaluation/results/` |
| `remaining_coverage_map.py` | Where the uncovered testable pixels are, as a picture and a JSON region list. | `coverage_audits/` |
| `verify_level1.py` | After Level 1 is fully covered: integrity, policy health and completion retention of the snapshot. Decides whether it is the final Level-1 brain. | one read-only `*_verification.json` beside the snapshot; the retention result in `evaluation/results/` |

## Incidents (Objective 3)

| Tool | What it does | Writes |
|---|---|---|
| `incidents.py` | The incident store from the command line: `list`, `show`, `verify` (re-hash every bundle against its manifest), `rerender` (new versioned reports, with a reason), `reproduce` (replay again), `recover` (finish what an interrupted run left). See docs/objective3/README.md. | nothing for list/show/verify; a NEW versioned file + manifest history for rerender/reproduce |
| `validate_incident_pipeline.py` | End-to-end proof on the real dashboard stack: the approved brain on the clean game, two SYNTHETIC probes, every stage asserted (stop, evidence, reports, replay, downloads, resume, duplicates, Objective-2 untouched). `--windowed` uses a real game window. ~30 s. | its own store, `incidents/_validation/<time>/` (ignored by the dashboard's history) |

## QA campaign set-up (run once, in this order)

| Tool | What it does | Writes |
|---|---|---|
| `collect_jump_arcs.py` | Records the engine's REAL jump arcs from flat ground (600 take-off sequences, ~90 s): the trajectories the reachability model uses instead of a rise x reach rectangle no jump can fill. Deterministic. | `exploration_data/jump_arcs.npz` |
| `build_reachability.py` | Builds the testable-pixel mask (the coverage denominator) from the live level geometry by three reconciled methods plus the real-arc envelope (~15 min); `--dry-run` reports without writing. `--tighten --coverage NPZ...` applies the arc envelope to the EXISTING mask in a few minutes (equivalent to a rebuild; archives the old mask, keeps real play the arcs deny). Re-run only if the level layout changes. | `exploration_data/reachable_mask.npz`, `observed_reach.npz`, the adopted totals into `exploration/config.py` |
| `build_anchor_set.py` | Records the ANCHOR states - what the healthy policy sees when it plays Level 1-1 - from seeds disjoint from the retention protocol's. `AnchorConsolidationCallback` measures KL against these to hold completion retention during training (see config ANCHOR CONSOLIDATION). | `exploration_data/anchor_states.npz` |
| `migrate_coverage.py` | Re-stamps a campaign coverage file to the CURRENT testable mask after a mask rebuild, losslessly: the visited bitmap is copied byte-for-byte and only the four mask-dependent fields are recounted. Refuses to overwrite, refuses if coverage would go down, and verifies the result loads. | a new coverage `.npz` (never the source) |
| `bootstrap_coverage.py` | Replays the 6M brain under its own reward to seed the coverage map with its habitual routes (~3 min). | `exploration_data/coverage_bootstrap_6000000.npz` |
| `calibrate_reward.py` | Solves `NOVELTY_WEIGHT` against the legacy reward's scale (~8 min). `--dry-run` reports without editing. | `NOVELTY_WEIGHT` in `exploration/config.py` |

## Research record (kept so the recorded numbers can be reproduced)

| Tool | What it measured | Writes |
|---|---|---|
| `calibrate_phase_reward.py` | Phase 4B: every QA reward channel, per phase and per step, on real trajectories and scripted controllers; `--report` re-analyses a run without replaying it. The PHASE-GATED REWARD values in `exploration/config.py` come from it. | `calibration_runs/*.pkl` |

## Conventions

Every tool starts with the same three lines (see `common/cli.py`): the project
root onto `sys.path`, then `prepare_tool()` - UTF-8 stdout, headless SDL for
the ones that replay the game, and the project's logging. A tool's printed
report is its product; everything the library code logs appears alongside it.
