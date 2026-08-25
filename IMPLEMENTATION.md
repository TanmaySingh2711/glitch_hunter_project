# Implementation Plan

Full-project correctness and cleanup pass. Each item below states the
problem, what was done, and how it was actually verified rather than
assumed.

---

## 1. The BUG TRACKER panel could never display anything

**Problem.** `agent_logic.py` reads `info['glitch_alert']` to build its
`BUG FOUND` log line, and `main.js` routes that line into the bug panel —
but **nothing in the project ever set that key**. The feature the project is
named after was wired to a value that was never written, so the panel sat
empty forever.

**What was built.** `CustomMarioEnv._detect_glitches()` now checks five
invariants the engine is supposed to hold and writes `info['glitch_alert']`
when one breaks:

| Check | Why it is a genuine bug |
|---|---|
| Alive below the death plane (`y > SCREEN_HEIGHT`) | `level1.check_for_mario_death()` kills Mario the instant he passes it, so seeing him alive there means that check failed. `in_castle`/`flag_get` excluded — the engine deliberately exempts the end-of-level walk. |
| `y < -200` | Clipped far above the level ceiling. |
| `abs(x_vel) > 25` | Speed the engine's acceleration should never reach. |
| Score decreased while alive | Score is monotonic within an episode. |
| Coin total decreased while alive | Same. |

**Thresholds came from measurement, not guesswork.** 4,000 steps across 22
episodes of the real trained policy produced `|x_vel|` max **13.2**, `y_pos`
range **-29..498**, and **zero** score or coin decreases. Every numeric limit
sits roughly 2x outside that observed envelope.

**Two design details that matter:**

- *Per-episode de-duplication.* A stuck out-of-bounds Mario would otherwise
  emit the same alert every frame and bury the panel. Each kind reports at
  most once per episode.
- *Delivery through the skip wrapper.* `MaxAndSkipObservation(skip=4)` calls
  `step()` four times per agent decision and returns **only the last**
  substep's `info` — the other three are discarded. An alert raised on
  substep 1–3 would vanish before anything could show it. Alerts are
  therefore held for `GLITCH_ALERT_TTL` substeps and re-attached each time.
  The TTL equals the skip value on purpose: long enough to always reach a
  surviving `info`, short enough to expire before the *next* one, so each
  glitch surfaces exactly once.

**Verification.**
- *No false positives:* 3,000 agent steps / 16 episodes of real trained
  play → **0 alerts**.
- *Real violations do get through:* injecting a non-self-repairing invariant
  break (score 5000 → 100 while alive) was delivered to the agent-visible
  `info` **exactly once**.
- An earlier delivery test teleported Mario below the death plane and saw
  no alert. That was the *test* being wrong, not the detector — the engine's
  own death check fires on the same update, so nothing was actually broken.
  Worth recording: the below-the-floor check is a guard for a case the
  engine currently handles correctly, and only speaks up if that handling
  ever breaks.

---

## 2. Window staggering had never worked

**Problem.** `custom_mario_env.py` sets `SDL_VIDEO_WINDOW_POS` and deletes
`SDL_VIDEO_CENTERED` so that 8 parallel training windows land at different
spots instead of stacking. But `setup.py` — imported immediately afterwards —
began with `os.environ['SDL_VIDEO_CENTERED'] = '1'`, silently putting it
back before `set_mode()` ran.

**Verified directly:** with both variables set, a window requested at
`(137,241)` was actually created at `(360,101)`. Centering won every time.

**Fix.** Removed that line from `setup.py`, with a comment recording the
measurement so it does not get reinstated.

---

## 3. Game depended on the current working directory at runtime

**Problem.** `setup.py` built resource paths relatively
(`"resources/music/..."`). `pg.mixer.music.load()` stores those paths and
re-opens them *during play*, so `step()` had to wrap every frame in
`getcwd` + `chdir(mario_clone)` + `chdir(back)` — and the game was
unloadable from any other directory.

**Fix.** `setup.py` now derives an absolute `_RESOURCES` path from its own
`__file__`, so the game is location-independent. The per-frame `chdir` in
`step()` and the one in `reset()` were removed.

**Verification.** Measured `env.step()` before **1.576 ms** → after
**1.511 ms** (**+4.1%**); this runs 4x per agent decision across 8 training
envs. Confirmed `os.getcwd()` is unchanged after stepping, and the full
training wrapper stack still builds and runs.

---

## 4. Emoji in a log line could crash the server on Windows

**Problem.** This console's stdout is `cp1252`. Printing the `🚨` used in
the bug-alert lines raises `UnicodeEncodeError` — verified directly. It
happened not to fire only because that string was sent over the socket
rather than printed; the new bug tracker makes printing it plausible.

**Fix.** `sys.stdout.reconfigure(..., encoding='utf-8')` in `app.py`.

---

## 5. Training crashed for anyone following requirements.txt

**Problem.** `train_agent.py` set `tensorboard_log="./logs/"`, and SB3 hard-
asserts `"tensorboard is not installed"`. `tensorboard` was **not** in
`requirements.txt` — it only worked locally because it had been installed ad
hoc. A friend following the README would crash on their first training run.

**Fix.** Tensorboard is now auto-detected: present → curves are logged;
absent → training runs normally with an informational note. This keeps a
large dependency optional for people who only want to watch the agent play,
instead of making it mandatory or leaving a crash in place.

---

## 6. Smaller correctness items

- **Generator teardown.** `agent_gen = None` left the old generator
  suspended at its `yield` until GC finalized it. `discard_agent_gen()` now
  calls `close()` explicitly, tearing the frame loop down at the moment the
  user asked for it.
- **Dead code removed.** Unused `numpy`/`cv2` imports, the never-called
  `render()` method, and the `short_episode_count` / `total_episodes`
  counters (incremented, never read — plus the docstring line that claimed
  they fed the alerts).
- **Stale API claims.** `metadata = {"render_modes": ...}` and the
  `render_mode` parameter advertised a Gymnasium render protocol this env
  does not implement; both removed.
- **Unused assets deleted** (~2.9 MB): verified every resource lookup uses a
  literal string key, and traced the single dynamic one (`music_dict[key]`)
  through all its call sites.
- `pygame` import hoisted out of `stop_all_music()`; three f-strings with no
  placeholders de-f-ed.

---

## Licensing

The vendored `mario_clone/` has **no upstream license** — its source repo
([justinmeister/Mario-Level-1](https://github.com/justinmeister/Mario-Level-1))
ships no LICENSE file and states only "intended for non-commercial
educational purposes." A blanket permissive license over this repository
would therefore have been a false claim. `LICENSE` is scoped explicitly to
original work; `THIRD_PARTY_NOTICES.md` records the vendored game and the
Nintendo-owned artwork.
