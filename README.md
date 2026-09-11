# Glitch Hunter Project

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
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install stable-baselines3==2.9.0 --no-deps
pip install -r requirements.txt
```
No NVIDIA GPU? Use this instead for the first line — everything still works, just training would be slow (watching the AI play is unaffected either way):
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
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
- **BUG TRACKER** (red panel) — stays empty unless the game actually breaks a
  rule it's supposed to follow: Mario alive below the floor, moving at an
  impossible speed, or the score/coin counter running backwards. An empty
  panel is the normal, healthy result — it only speaks up for real problems.
- **LOG TERMINAL** (green panel) — every action the AI takes, with its reward.

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

- `app.py` — the web server (Flask + WebSockets) that streams the game to your browser.
- `agent_logic.py` — how the AI is rewarded during training, plus the code that runs it live for the dashboard.
- `custom_mario_env.py` — connects the Mario game to the AI training library.
- `train_agent.py` — trains the AI from scratch (takes hours — most people will never need to run this).
- `mario_clone/` — the actual Super Mario Bros game (Python/Pygame). Not written by us — see Credits below.
- `static/` & `templates/` — the dashboard's look (HTML/CSS/JS). Includes
  `static/vendor/socket.io.min.js`, kept locally so the dashboard works with
  no internet connection.
- `mario_brain_checkpoint.zip` — the trained AI's "brain". Needed for the AI to play well; see above if it's missing.
- `checkpoints/` — snapshots from training, not needed just to watch the AI play.
- `tests/` — fast automated checks (`python -m pytest`, ~4 seconds).
- `pyproject.toml` — the one-command `uv sync` install, plus linter settings.

---

## The QA exploration phase (from 6M steps onward)

The first 6,000,000 steps trained the agent to **complete the level**. From
here it is being retrained to **explore the world**, which is a different
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
`backup_6M/` holds a verified byte-identical copy as a second line of
defence.

### Running it, in order

```bash
python tools/build_reachability.py     # once - the coverage denominator
python tools/bootstrap_coverage.py     # ~3 min - seed the map from the 6M brain
python tools/calibrate_reward.py       # ~8 min - solve NOVELTY_WEIGHT
python train_agent.py --safety-cap-timesteps 6020000   # the controlled +20k validation
# ...review it, then the campaign itself, which runs until Level 1 is fully covered:
python train_agent.py --unrestricted
python tools/verify_level1.py          # after completion: is it the final Level-1 brain?
```

The first three are one-off. Each refuses to run if the previous one has not,
rather than silently proceeding with a wrong denominator or an empty map.
A QA launch with no safety cap is refused unless `--unrestricted` is given,
so an unbounded campaign never starts by accident.
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
| `testable_coverable_px` | 4,013,723 | **the denominator** |
| `covered_testable_px` | — | the numerator |
| `noncoverage_px` | — | recorded, but outside the mask. Never coverage |

`coverage % = covered_testable / testable_coverable`. Nothing else.

It is built by three independent methods that must reconcile - geometric
openness, a jump envelope from the measured physics, and a connectivity BFS
from spawn - and the strictest is adopted. `tools/build_reachability.py`
prints the full reconciliation and refuses to write a mask if they disagree.
The derivation, including the resolution-convergence study behind it, is in
`exploration/config.py`.

**Leaving the mask is not a bug.** Ordinary play does it constantly: jump arcs
rise above the top of the world, pit deaths carry Mario below the floor, and
the engine's own collision resolution lets him sink up to 5 px into a block.
Every out-of-mask pixel is classified against the level geometry, and only the
genuinely impossible ones reach the glitch report - over the entire 40-episode
bootstrap, that count is zero.

## Credits & Licensing

The game in `mario_clone/` was written by **Justin Meister**
([Mario-Level-1](https://github.com/justinmeister/Mario-Level-1)) — not by us. It has
**no open-source license**, and its author states it is "intended for non-commercial
educational purposes." The artwork and sounds are Nintendo's property.

**So: learn from this, don't sell it.** Our own code (`app.py`, `agent_logic.py`,
`custom_mario_env.py`, `train_agent.py`, the dashboard) is MIT-licensed.

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
pip install pytest
python -m pytest
```
About 4 seconds, no GPU and no trained model needed. Worth running after any
change to `custom_mario_env.py` or `agent_logic.py` — it covers the
environment contract, the reward-shaping rules, and the bug detector
(including that it stays quiet during normal play).

Linting, if you want it: `pip install ruff` then `python -m ruff check .`

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
