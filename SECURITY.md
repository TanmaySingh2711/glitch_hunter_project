# Security

Glitch Hunter is a single-user research project that runs on your own
machine. This page states what it exposes, what it trusts, and how that is
kept narrow.

## The dashboard server

* **Loopback only by default.** `app.py` binds `127.0.0.1`. Listening on
  another interface needs an explicit `GLITCH_HUNTER_HOST`, and the server
  logs a warning when it does (`app.bind_address`, pinned by
  `tests/test_concurrency.py`).
* **No authentication.** Anyone who can reach the port can start, stop and
  reset the agent and watch the stream. That is acceptable on loopback and on
  a network you trust, and only there.
* **Same-origin Socket.IO.** Flask-SocketIO's default CORS policy is left in
  place, so a page on another origin cannot drive the dashboard from your
  browser.
* **No debug server.** `debug=False` always - the Werkzeug debugger is a
  remote code execution console by design.
* **Offline.** The Socket.IO client is vendored (`static/vendor/`); the page
  loads nothing from the internet.
* **Incident files are served from an allow-list, not from paths.** The
  report and evidence routes (`/incidents/<id>/<file>`, `/api/incidents/...`)
  accept only a well-formed id of an existing incident (ASCII, whole-string
  match) and a file name from a fixed list, and the resolved path must still
  lie inside that incident's folder (`reporting/store.IncidentStore.artifact_path`).
  Anything else - `../`, encoded separators, absolute paths, unknown files -
  is a 404; `tests/test_dashboard_incidents.py` tries a dozen such requests.
  Files are sent with `X-Content-Type-Options: nosniff`, and the page puts
  incident text into the DOM with `textContent`, never as HTML.
* **Run reports are served the same way.** `/runs/<id>/<file>` accepts only a
  well-formed id of an existing clean-run report and a file name from a fixed
  list (`reporting/run_report.RunReports.path`); `tests/test_run_report.py`
  checks the refusals.
* **A replay runs only what the bundle names from a closed list.** Incident
  reproduction starts a local Python process (`reporting/reproduce.py`) that
  reads the bundle; extra detectors are rebuilt only from a fixed registry
  (`reporting.events.detector_from_spec`), never from code in the bundle.

## What run_dashboard.bat changes on this computer

`app.py --desktop` (what `run_dashboard.bat` starts) keeps the dashboard at
full speed on battery (`desktop.py`). It needs no administrator rights and
changes only two things. Its own process opts out of Windows' power
throttling. While the dashboard runs, the Windows power mode is set to *Best
performance*. Your own power mode is put back when the dashboard stops (Ctrl+C,
or closing its window). Windows only lets a normal user change the mode of
the power source in use, so a battery mode changed while on battery is put
back the next time the laptop is on battery, by a small background process
that then exits. Until then the original values are kept in
`.dashboard_power.json` in the project folder.

## Files it loads

* **Checkpoints are code.** Stable-Baselines3 checkpoints are zip files whose
  parameters are loaded with pickle-based deserialisation. Loading a
  checkpoint runs whatever it contains. Only load checkpoints you (or this
  repository) produced; `tools/verify_artifacts.py` confirms the protected
  ones are still the bytes that were recorded in `artifacts.json`.
* **The final 16M brain is verified twice before it is used.** It is not in
  git; `tools/final_brain.py install` downloads it over HTTPS from this
  repository's GitHub Release, checks all four files against the SHA-256
  values tracked in `artifacts.json`, and writes nothing unless every one
  matches. It accepts only the four expected file names (no other path can
  be written) and never overwrites a different file. The dashboard then loads
  `glitch_hunter_main_brain.zip` only if its SHA-256 matches the closure
  record `FINAL_OBJECTIVE2.json`; otherwise it logs a warning and uses the
  latest local QA checkpoint, or the 6M brain.
* **Everything else is data.** Coverage files and masks are `.npz` archives
  read with `allow_pickle=False`, and every field is validated (format
  version, grid geometry, mask fingerprint, the bitmap's own counts) before
  use. Records are JSON.
* `tools/calibrate_phase_reward.py --report` reads the pickle files that same
  tool wrote; never point it at pickles from anywhere else.

## Secrets

None. The project has no credentials, tokens or network services to
authenticate against, and nothing in the repository should ever contain one.
`ruff`'s bandit rules (`S`) run in CI.

## Dependencies

All runtime dependencies are pinned exactly (`pyproject.toml`,
`requirements.txt`, `uv.lock`). To audit them for known vulnerabilities:

```bash
pip install pip-audit
pip-audit -r requirements.txt      # set PYTHONUTF8=1 first on Windows
```

Last audit (2026-09-25, pip-audit): **no known vulnerabilities** in
`requirements.txt` (which includes `stable-baselines3`, `fpdf2` and
`pillow`) or in `torch==2.14.0`. pip-audit checks the version number on
PyPI; the `+cu126` GPU wheels come from PyTorch's own index.

**torch upgrade (2026-09-25).** The project used to pin torch 2.5.1 (CUDA
12.1), which has 19 published advisories - among them two ways to run code
while loading a checkpoint (CVE-2025-32434, CVE-2026-24747) and
memory-corruption or crash bugs in specific operators. It now pins 2.14.0
(CUDA 12.6), which has none. Before the switch, both brains (16M and 6M) were
run for 1,500 steps on each game under both versions: every move and every
position was identical.

Keep the virtual environment's own `pip` and `setuptools` current
(`python -m pip install -U pip setuptools`); older versions carry
advisories of their own, unrelated to this project's code.

## Reporting a problem

Open an issue on the repository, or contact the maintainer directly for
anything that should not be public until it is fixed.
