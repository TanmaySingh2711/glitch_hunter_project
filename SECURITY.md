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

## Files it loads

* **Checkpoints are code.** Stable-Baselines3 checkpoints are zip files whose
  parameters are loaded with pickle-based deserialisation. Loading a
  checkpoint runs whatever it contains. Only load checkpoints you (or this
  repository) produced; `tools/verify_artifacts.py` confirms the protected
  ones are still the bytes that were recorded in `artifacts.json`.
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

Last audit (2026-09-11): no known vulnerabilities in `requirements.txt`.

**Known exception - torch 2.5.1.** It is pinned because no CUDA 12.1 build
of anything newer exists (README, "Why the pip install is 3 steps"), and
2.5.1 predates the fix for CVE-2025-32434 (a `torch.load(weights_only=True)`
bypass, fixed in 2.6.0). The exposure is the one already stated under
"Checkpoints are code": it only matters when loading an untrusted
checkpoint, which this project never does. Upgrade torch as soon as a CUDA
build that fits the install allows it. pip-audit cannot check the `+cu121`
wheels itself, since they are not on PyPI.

Keep the virtual environment's own `pip` and `setuptools` current
(`python -m pip install -U pip setuptools`); older versions carry
advisories of their own, unrelated to this project's code.

## Reporting a problem

Open an issue on the repository, or contact the maintainer directly for
anything that should not be public until it is fixed.
