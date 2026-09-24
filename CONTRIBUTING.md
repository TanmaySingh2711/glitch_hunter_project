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
python tools/check.py        # ruff, mypy, fast tests, artifact hashes (~1-2 min)
python tools/check.py --full # before a merge or a training launch: + slow tests, coverage floor
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
  from outside it. Deliberate QA bugs go only in `mario_bugged/`, each
  declared in its `INJECTED_BUGS.json` (see `mario_bugged/VARIANT.md`).
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
`training:`, `tools:`, `docs:`, `ci:`). Never commit generated artifacts - the
`.gitignore` lists them and says why; `mario_brain_checkpoint.zip` is the one
deliberate exception.

## Artifacts

Training outputs, masks and baselines are protected state. If a change
legitimately alters one (a rebuilt mask, a new baseline), re-record the
manifest in the same commit and say so:

```bash
python tools/verify_artifacts.py --record
```
