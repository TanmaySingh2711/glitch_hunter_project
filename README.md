# Glitch Hunter Project

[![CI](https://github.com/TanmaySingh2711/glitch_hunter_project/actions/workflows/ci.yml/badge.svg)](https://github.com/TanmaySingh2711/glitch_hunter_project/actions/workflows/ci.yml)

A trained AI plays a Super Mario Bros clone, and you watch it live in your browser.

---

## Setup (copy-paste these, in order)

**1. Get Python 3.12** from [python.org](https://www.python.org/downloads/) if you don't have it. (Not 3.13, not 3.14 — must be 3.12.)

**2. Open a terminal in this folder and create a virtual environment:**
```bash
py -3.12 -m venv venv_gpu
venv_gpu\Scripts\activate
```
*(Mac/Linux: `python3.12 -m venv venv_gpu` then `source venv_gpu/bin/activate`)*

**3. Install everything.**

*Easiest — one command, nothing to get in the right order:*
```bash
pip install uv
uv sync
```

*Or with plain pip, in this exact order* (don't skip ahead or combine these):
```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install stable-baselines3==2.9.0 --no-deps
pip install -r requirements.txt
```
No NVIDIA GPU? Use this instead for the first line — everything still works, just training would be slow (watching the AI play is unaffected either way):
```bash
pip install torch==2.5.1
```

**4. Run it:**
```bash
python app.py
```

**5. Open your browser** to **http://localhost:5000** and click **"START TESTING"**.

That's it. A game window will pop up and the same footage streams live to your browser.

---

## Using the dashboard

- **Start Testing** — opens the game window, centred on the screen you're using, and starts streaming it to the browser. After a pause it carries on from the same moment.
- **Stop Testing** — pauses the game (both the pop-up window and the browser view freeze on the same frame). The window stays where it is; minimise, move or close it yourself whenever you like.
- **The game window's X** — closes the window and pauses testing. Nothing is lost: **Start Testing** reopens it and resumes.
- **Reset Dashboard** — ends the session and closes the window; the next start is a fresh one. Refreshing the page only pauses.
- **BUG TRACKER** (red panel) — the incidents found in this session, newest
  first. **Reset Dashboard** empties it, like the log (nothing is
  deleted: every incident stays saved in `incidents/`, listed by
  `python tools/incidents.py list`). It stays empty unless the game actually breaks a rule it's
  supposed to follow: Mario alive below the floor, far above the level,
  moving at an impossible speed, the score/coin counter running backwards,
  Mario ending up inside ground, a pipe, a step or a block, being stopped by
  something that isn't drawn, being hurt by an enemy he never touched, or
  jumping faster or higher than any real jump.
  An empty panel is the normal, healthy result. Each entry links to its
  **PDF report, Markdown report, exact trigger frame, GIF** and the whole
  evidence bundle.
- **Testing Stopped - Bug Found** — when a new incident is caught, testing
  stops by itself (the evidence is already saved) and a red banner over the
  video shows what happened. It stays until you press **Start Testing**,
  which carries on from the same moment. Every detection stops testing - a
  later sighting of the same bug too (it is counted on the same entry, not
  added as a new one).
- **LOG TERMINAL** (green panel) — every action the AI takes, with its reward.
- **Select Game Environment** — which game the AI tests: **Mario Game
  (Cleaned)**, the clean baseline, or **Mario Game (Bugged)**, the copy for
  deliberate bugs. Changing it resets the dashboard and loads that game;
  press **Start Testing** to play it. (`python app.py --game mario_bugged`
  starts on the bugged one.)

How incidents are captured, reported, de-duplicated and replayed:
[`docs/objective3/README.md`](docs/objective3/README.md).

Works fully offline — nothing is loaded from the internet.

The server listens on **your machine only**. If you want to open the dashboard
from another device (a phone on the same Wi-Fi, say), opt in explicitly:

```bash
set GLITCH_HUNTER_HOST=0.0.0.0     &:: Windows
python app.py
```
Only do that on a network you trust — there is no password on the dashboard,
so anyone who can reach it can drive the agent and watch the stream.

If you ever see the AI acting completely random instead of playing well, check the terminal for a `[WARNING] ... running an UNTRAINED policy` line — it means `mario_brain_checkpoint.zip` (the trained brain) is missing from the folder.

---

## Project Structure

```
app.py                  the web server (Flask + WebSockets) that streams the game to your browser
dashboard_service.py    the one thread that owns the game window (Start / Stop / Reset / its X)
dashboard_backend.py    loads the approved brain and turns each step into a frame, a log line and
                        any incident it caught
game_window.py          where the window appears, and the Windows calls pygame lacks
custom_mario_env.py     the Mario game as a Gymnasium environment, plus the agent's 84x84 view
agent_logic.py          GlitchHunterWrapper: coverage recording, episode lifecycle, reward
rewards/                the two objectives - qa.py (exploration) and legacy.py (the 6M brain's)
train_agent.py          trains the AI (takes hours - most people never need to run it)
training/               training building blocks: callbacks, checkpoint helpers, value-head reset
exploration/            world-pixel coverage, the testable-pixel mask, the EXPLORE -> COMPLETE
                        lifecycle, Level-1 completion, and config.py - every tunable, with its evidence
evaluation/             can a checkpoint still finish the level? (completion retention, verification)
reporting/              bug incidents: evidence capture, the incident store, GIF/Markdown/PDF
                        reports, replay-based reproduction, the two game variants
common/                 logging, atomic file writes, the tools' shared start-up
tools/                  command-line scripts, one job each - see tools/README.md
tests/                  the automated checks (see "Running the tests")
docs/ARCHITECTURE.md    how the pieces fit together, and the invariants that hold across them
docs/PERFORMANCE.md     where the time and memory go per worker, and how to re-measure
docs/objective2/WORKLOG.md   the QA exploration phase: final state, every finding, why it stopped
docs/objective2/HANDOFF.md   the 17-section report on that phase (written before the campaign ran)
docs/objective3/README.md      bug evidence and reports: how an anomaly becomes a reviewable incident
incidents/              (created at run time, not in git) one folder of evidence per incident
mario_clean/            the actual Super Mario Bros game (Python/Pygame). Not written by us - see Credits
mario_bugged/           a copy of it with six deliberate test bugs - see mario_bugged/VARIANT.md
static/, templates/     the dashboard page; static/vendor/socket.io.min.js is kept locally so the
                        dashboard works with no internet connection
mario_brain_checkpoint.zip   the trained AI's "brain" - needed for the AI to play well
artifacts.json          SHA-256 of every protected artifact (python tools/verify_artifacts.py)
```

Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md). Security notes:
[`SECURITY.md`](SECURITY.md).

---

## The QA exploration phase (from 6M steps onward)

> **This phase is finished (2026-09-21).** The campaign ran to **16,000,000
> steps** and was stopped there deliberately, at **84.25%** of the testable
> pixels (3,166,235 / 3,757,990) — **not** 100%. The remaining 591,755 pixels
> are mostly high air that Mario has no reason to occupy, and coverage had
> slowed to a fraction of a point per hour. The final brain still finishes the
> level: **56.6%** completion over the 500-episode protocol, against 46.8% for
> the 6M baseline it started from.
>
> Final brain: **`glitch_hunter_main_brain.zip`** (with
> `glitch_hunter_main_brain_coverage.npz`), read-only at the project root; its
> closure record, evidence and a backup copy are in
> `checkpoints_qa/final_objective2_16000000/`. Full reasoning, hashes and
> limitations: [`docs/objective2/WORKLOG.md`](docs/objective2/WORKLOG.md)
> ("FINAL STATE"). The rest of this section describes how the phase works and
> still applies.

The first 6,000,000 steps trained the agent to **complete the level**. From
there it was retrained to **explore the world**, which is a different
objective, not a refinement of the old one:

> previously explored territory = **transit space**
> new global world-space coverage = **reward space**

The agent may cross old ground freely to reach the frontier, but old pixels
never pay again.

### The switch

Everything is behind one constant, `REWARD_MODE` in `exploration/config.py`:

```python
REWARD_MODE = "qa_exploration"      # or "legacy_completion"
```

`train_agent.py` reads it and derives everything else - which reward runs,
which checkpoint files are written, which milestones apply. The two can never
drift apart.

**The 6M brain is never written in QA mode.** `mario_brain_checkpoint.zip`
and `checkpoints/` are read once to seed the QA phase and never touched
again; QA training writes only `glitch_hunter_qa.zip` and `checkpoints_qa/`.
Git is its backup (`git checkout -- mario_brain_checkpoint.zip`).

### Running it, in order

```bash
python tools/build_reachability.py     # once - the coverage denominator
python tools/bootstrap_coverage.py     # ~3 min - seed the map from the 6M brain
python tools/calibrate_reward.py       # ~8 min - solve NOVELTY_WEIGHT
python train_agent.py --safety-cap-timesteps 6020000   # the controlled +20k validation
# ...review it, then the campaign itself:
python train_agent.py --unrestricted                   # open-ended; or --safety-cap-timesteps N
python tools/evaluate_completion.py <checkpoint>       # can it still finish the level?
```

The first three are one-off. Each refuses to run if the previous one has not,
rather than silently proceeding with a wrong denominator or an empty map.
A QA launch with no safety cap is refused unless `--unrestricted` is given,
so an unbounded campaign never starts by accident.

`--unrestricted` ends only when every testable pixel is covered, which the
2026 campaign showed does not happen in practice — it was stopped at a chosen
step count with `--safety-cap-timesteps`, and a checkpoint is accepted on the
500-episode retention result, never on training statistics.
`tools/verify_level1.py` applies only if coverage ever does reach 100%.
`python tools/remaining_coverage_map.py <coverage.npz>` draws what is left of
the level at any point.

### Going back to the old behaviour

Set `REWARD_MODE = "legacy_completion"` and re-run `train_agent.py`. It will
resume `mario_brain_checkpoint.zip` under the exact reward that produced it -
that code path is unchanged and is pinned by `tests/test_reward_wrapper.py`,
which names its mode explicitly so it does not follow the config default.
Nothing needs to be deleted or restored.

### What "coverage" means here

A world pixel counts as explored the first time Mario's **collider rect** has
occupied or swept across it, in world coordinates, at his real per-form size
(30x40 small, 40x80 big). Not screen space, not the visible frame, not the
background art. The bitmap persists across episode resets, process restarts
and all 8 training workers - the workers share one bitmap through
`multiprocessing.shared_memory`, so the same opening stretch of the level is
credited once in total, not once per worker.

The denominator is the **testable** mask: every pixel some *physically
reachable* placement of the collider covers. Reachable, not merely
unobstructed - it excludes sky above the world, everything below the death
plane, solid interiors, altitude no jump can attain, and anywhere that cannot
be reached from the spawn point through a chain of real moves.

| quantity | value | role |
|---|---|---|
| `world_raster_px` | 5,452,200 | the 9087x600 level. Informational **only** - never a denominator |
| `testable_coverable_px` | 3,757,990 | **the denominator** (`config.TESTABLE_TOTAL`) |
| `covered_testable_px` | — | the numerator |
| `noncoverage_px` | — | recorded, but outside the mask. Never coverage |

`coverage % = covered_testable / testable_coverable`. Nothing else.

It is built by three independent methods that must reconcile - geometric
openness, a jump envelope from the measured physics, and a connectivity BFS
from spawn - and the strictest is adopted, then narrowed once more by the
**real-arc envelope**: 228 jump trajectories recorded from the engine itself
(`tools/collect_jump_arcs.py`), because the three methods bound a jump by a
rectangle (full rise *and* full reach at once) that no real arc can fill.
That last step removed 244,105 pixels no jump can ever reach.
`tools/build_reachability.py` prints the full reconciliation and refuses to
write a mask if the methods disagree.
The derivation, including the resolution-convergence study behind it, is in
`exploration/config.py`.

**Leaving the mask is not a bug.** Ordinary play does it constantly: jump arcs
rise above the top of the world, pit deaths carry Mario below the floor, and
the engine's own collision resolution lets him sink up to 5 px into a block.
Every out-of-mask pixel is classified against the level geometry, and only the
genuinely impossible ones reach the glitch report - over the entire 40-episode
bootstrap, that count is zero.

## Bug reports (Objective 3)

When the detector sees the game break a rule, the dashboard stops and the
moment becomes an **incident** in `incidents/<id>/`: the exact trigger frame
(lossless, full resolution), a GIF of the seconds before it, the per-frame
trajectory, the episode's complete input log, a Markdown and a PDF report
that keep what was measured apart from what is inferred, and a replay of the
episode in a separate process that says honestly whether it happens again.
Repeat sightings of the same bug are counted, not re-reported.

```bash
python app.py --game mario_bugged        # test the variant with the six deliberate bugs
python tools/incidents.py list           # every incident, from the command line
python tools/incidents.py verify         # re-hash every evidence file
python tools/validate_incident_pipeline.py   # prove the whole pipeline end to end
```

There are two copies of the game: `mario_clean/` (the untouched baseline,
pinned by hash) and `mario_bugged/` (six deliberate benchmark bugs, declared in
`mario_bugged/INJECTED_BUGS.json`).
`--synthetic-probe X` adds a fake, clearly labelled "bug" at world x X, only
to exercise the pipeline. Full design, schema and limitations:
[`docs/objective3/README.md`](docs/objective3/README.md).

---

## Credits & Licensing

The game in `mario_clean/` (and its copy `mario_bugged/`) was written by **Justin Meister**
([Mario-Level-1](https://github.com/justinmeister/Mario-Level-1)) — not by us. It has
**no open-source license**, and its author states it is "intended for non-commercial
educational purposes." The artwork and sounds are Nintendo's property.

**So: learn from this, don't sell it.** Our own code — everything outside
`mario_clean/`, `mario_bugged/` and `static/vendor/` (the Socket.IO client, MIT-licensed by its
own authors) — is MIT-licensed.

Full details in [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

---

## Technical Reference

Everything below this line is background/troubleshooting detail — not needed for normal setup or use.

### Why the pip install is 3 steps, not just `pip install -r requirements.txt`

(`uv sync` avoids all of this — it's encoded in `pyproject.toml` already. This
section explains the manual pip path.)

`stable-baselines3==2.9.0` requires `torch>=2.8`, but no CUDA-enabled build of torch newer than 2.5.1 exists (checked directly against PyTorch's own package index). Installing stable-baselines3 normally would silently replace your GPU-enabled torch with a non-GPU one to satisfy that requirement — no error, just much slower training with no explanation why. `--no-deps` stops it from touching torch at all.

### Why Python 3.12 specifically

PyTorch's GPU-enabled (CUDA) builds don't exist yet for Python 3.14 (checked directly against `https://download.pytorch.org/whl/cu121/` — wheels exist for 3.10–3.13, not 3.14). Many systems default to whatever the newest installed Python is, so even if 3.12 is also on your machine, you may need to request it explicitly with `py -3.12` as shown above.

### GPU requirement

Only **training** meaningfully needs a GPU. **Watching the AI play does not** — loading the trained brain and picking one action per frame is cheap, and it automatically runs on CPU if no GPU is found. The game itself is plain 2D rendering, not GPU-accelerated.

### Checkpoints

| Path | What it is |
|---|---|
| `mario_brain_checkpoint.zip` | **The trained brain.** Training resumes from this if present, and the dashboard always plays this one. |
| `checkpoints/mario_brain_checkpoint_{N}_steps.zip` | Snapshots taken every 400k training steps. Never overwritten. |
| `checkpoints/.milestone_saved_*.flag` | Bookkeeping for the snapshots above. **Do not delete** — they're 0 bytes but deleting them causes a resumed training run to overwrite every snapshot with the current model, destroying the training history. |

To roll back to an earlier snapshot:
```bash
cp checkpoints/mario_brain_checkpoint_5200000_steps.zip mario_brain_checkpoint.zip
```

### Running the tests

```bash
pip install pytest==9.1.1 pytest-cov==7.1.0 ruff==0.16.6 mypy==2.3.1   # or: uv sync (dev group)
python tools/check.py          # lint, type check, fast tests, artifact hashes (~2 min)
python tools/check.py --full   # everything, including the slow tests (~15 min) and coverage
```

No GPU needed. The fast tests cover the environment contract, the reward
rules of both modes, the episode lifecycle and safety reset, coverage and its
persistence, the dashboard's window control, and the bug detector (including
that it stays quiet during normal play). The `slow` ones replay the real
engine for thousands of steps, rebuild the 3,757,990-pixel mask, and load the
real 6M brain. Tests that need a git-ignored file (`exploration_data/`,
`checkpoints_qa/`, the main brain) skip themselves when it is missing.

The same gates run on every push in CI (`.github/workflows/ci.yml`), and
`pre-commit install` runs lint and type checks on every commit.

### Why it only listens on localhost

`app.py` binds to `127.0.0.1`. It used to bind to `0.0.0.0`, which listens on
every network interface — so on shared Wi-Fi, anyone who guessed the address
could open the dashboard, start and stop the agent, and watch the video feed.
Nothing pointed this out: the README only ever said to visit `localhost`, so
the exposure was accidental rather than intended. Override with
`GLITCH_HUNTER_HOST` (see above) when you actually want it, and `GLITCH_HUNTER_PORT`
if 5000 is taken.

### Checking the server is healthy

`http://localhost:5000/healthz` returns a small JSON snapshot:

```json
{"status":"ok","testing":false,"threads_total":2,"frame_loops":0}
```

`frame_loops` should never exceed **1**. It exists because an early version
of the threading rewrite spawned a new frame-loop thread on every Start
click; they piled up and the server stopped responding. If you ever see this
above 1, that bug is back.

### Why the server no longer uses eventlet

`app.py` runs Flask-SocketIO in **threading** mode. eventlet is unmaintained,
and it ran everything on a single thread that only switched at I/O points it
had patched — so one blocking GPU call froze the whole server, including the
heartbeat that tells your browser the connection is alive. That caused a real
bug here: the first "Start Testing" click could close the game window it had
just opened. Real threads don't have that failure mode.

**Don't `pip install eventlet` again.** Flask-SocketIO picks its async mode
from whatever is installed, so merely having it present would switch the
server back without any warning.

### Upgrading dependencies later

The pinned versions above were tested together and don't need touching just because time has passed — PyPI doesn't delete old package versions, so these keep installing indefinitely. The one real deadline: **Python 3.12 loses security support on 2028-10-31** ([official schedule](https://devguide.python.org/versions/)). Before then, upgrade in a *new* venv first, re-check for the same kind of torch-version conflict described above, and smoke-test (load the checkpoint, step the environment a few times) before trusting it.
