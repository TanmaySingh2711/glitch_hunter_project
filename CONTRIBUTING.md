# Contributing

## Set up

Follow the README's Installation section, then add the development tools:

```bash
pip install pytest==9.1.1 pytest-cov==7.1.0 ruff==0.16.6 mypy==2.3.1 pre-commit==4.6.2
pre-commit install          # optional: lint + type-check on every commit
```

(`uv sync --active` installs the same versions from `pyproject.toml`'s `dev` group.)

## Before every commit

```bash
python tools/check.py        # ruff, mypy, fast tests, artifact hashes (~5 min)
python tools/check.py --full # before a merge or a release: + slow tests, 90% coverage floor (~30 min)
```

CI (`.github/workflows/ci.yml`) runs the same gates on every push, on Linux
and Windows.

## Code conventions

* **Types.** Everything outside `tests/` passes `mypy --strict` (config in
  `pyproject.toml`). New code is annotated; `Any` only at an untyped library
  boundary (SB3 internals, the vendored engine).
* **Logging, not print.** Library modules do
  `log = logging.getLogger(__name__)`; entry points call
  `common.logging_setup.configure_logging()`. `print()` is for the tools'
  reports only (ruff enforces this).
* **Config is evidence.** Every tunable lives in `exploration/config.py` with
  the measurement it came from beside it. Changing a number means stating the
  new evidence in the same place.
* **Comments say why.** The code says what. A comment that restates the next
  line is noise; one that records the incident, measurement or constraint
  behind it is the most valuable line in the file.
* **Never touch `mario_clean/`.** It is vendored third-party code (see
  `THIRD_PARTY_NOTICES.md`) and the pinned baseline game; read what you need
  from outside it. Deliberate QA bugs go only in `mario_bugged/`: mark every
  changed block of lines with `# INJECTED BUG <id>`, declare the bug in
  `INJECTED_BUGS.json`, and re-pin its `diff_sha256` - the tests reject
  anything else (see `mario_bugged/VARIANT.md`). Add bugs only when the
  project owner specifies them.
* **Detectors are generic.** A rule in `reporting/collision_invariants.py`
  (or `custom_mario_env._detect_glitches`) must hold everywhere in the level
  and must not name the place of any injected bug. It is judged against what
  is drawn (`reporting/level1_design.json`, regenerated only with
  `tools/build_level_design.py`), never against the game's own colliders. A
  new rule must stay silent on the clean game; every false positive found
  gets fixed at its cause and pinned as a regression test in
  `tests/test_collision_invariants.py`.
* **The legacy reward is frozen.** `rewards/legacy.py` is what the 6M brain
  was trained under; `tests/test_reward_wrapper.py` pins it to the bit.

## Tests

* Put a test beside the behaviour it pins, and name it for the behaviour
  (`test_a_very_long_drought_in_coherent_transit_cannot_reset`), not the
  function.
* Mark anything that takes more than a few seconds `@pytest.mark.slow`.
* A test that needs a git-ignored artifact (`exploration_data/`,
  `checkpoints_qa/`, `glitch_hunter_main_brain.zip`) must `pytest.skip` with the reason when it is
  absent, so CI and fresh clones stay green.

## Commits

One logical change per commit, with a message that says what changed and
why:

```
lifecycle: require span-level progress before a safety reset

The LOOP arm could fire on elapsed steps alone during coherent transit.
It now needs SAFETY_STUCK_WINDOWS non-transit, zero-yield windows and
no progress across them. Tests: test_episode_lifecycle.py (10 new).
```

Prefix with the area touched (`lifecycle:`, `rewards:`, `dashboard:`,
`reporting:`, `detectors:`, `variants:`, `training:`, `tools:`, `docs:`,
`ci:`). Never commit generated artifacts - the `.gitignore` lists them and
says why. `mario_brain_checkpoint.zip` (the 6M fallback brain) is the one
deliberate exception; the README's images in `assets/` are source, not
artifacts.

## Artifacts

Training outputs, masks and baselines are protected state. If a change
legitimately alters one (a rebuilt mask, a new baseline), re-record the
manifest in the same commit and say so:

```bash
python tools/verify_artifacts.py --record
```

## Releasing the final brain

The final 16M brain and its three companion files are never committed; they
ship as one GitHub Release asset that `tools/final_brain.py install`
downloads and verifies. Only if the approved brain itself ever changes (a new
closure record and new hashes in `artifacts.json`), rebuild and publish the
bundle:

```bash
python tools/final_brain.py bundle dist/glitch_hunter_final_brain_16M.zip
gh release create <tag> dist/glitch_hunter_final_brain_16M.zip --title "Glitch Hunter <tag>"
```

and update `RELEASE_TAG` in `tools/final_brain.py` to the new tag.
