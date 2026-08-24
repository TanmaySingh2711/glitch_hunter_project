# Implementation Plan

## Problem

The dashboard's live view and pop-up window ran well under the 60 FPS the
game logic itself is authored for — as low as 10-15 FPS on battery, ~25-30
FPS even plugged in. Profiling `env.step()` (the actual bottleneck, ~82% of
total per-decision time) found the observation pipeline dominates:
`pg.surfarray.array3d()` on the full 800x600 window (a numpy copy+transpose
of the whole surface) followed by a separate `cv2.resize()` down to
256x240 — together ~76% of every real game tick, and this runs up to 4x
per agent decision under `MaxAndSkipObservation`'s frame-skipping.

## Options considered

- Reduce `MaxAndSkipObservation`'s skip value: would proportionally cut the
  number of real ticks per decision, but changes the effective time-per-
  decision the checkpoint was *trained* on — not safe to change without
  retraining.
- Reduce native `SCREEN_SIZE` in the mario_clone engine: would shrink the
  cost at its root, but sprite/camera/physics code throughout mario_clone
  assumes 800x600 coordinates — too invasive and out of scope.
- Downscale via `pg.transform.scale()` (SDL/C-level) *before* converting to
  a numpy array, instead of capturing the full-resolution array first and
  downscaling with `cv2.resize()` after. Chosen: verified 11.9x faster for
  this exact resize (0.47ms vs 5.6ms, measured over 200 calls), and —
  checked directly, not assumed — produces **byte-for-byte identical**
  output to the old `cv2.INTER_NEAREST` path at this scale ratio, so it
  changes nothing about what the trained model actually sees.

Separately, the stream sent to the browser was full 800x600 JPEG at
default quality (~95) through base64 encoding, though the dashboard only
displays it scaled into a much smaller panel. Switched to: downscale to
480x360 before encoding, JPEG quality 80, and raw binary WebSocket frames
instead of base64 (socket.io handles binary payloads natively) — measured
3.4x faster at the encode stage alone. This wasn't the dominant cost
(the observation pipeline was), but it stacks with the fix above for free.

## What was built

- `custom_mario_env.py`: new `_fast_obs()` replaces the
  `render()` + `cv2.resize()` pattern in both `step()` and `reset()`.
- `agent_logic.py`: streamed frames resize to 480x360 (`STREAM_SIZE`) at
  JPEG quality 80 (`STREAM_JPEG_QUALITY`) and are sent as raw bytes instead
  of a base64 string. Added `[STREAM FPS]` server-side logging (printed
  every ~2s) so the real achieved rate is directly readable from the
  terminal rather than guessed from how smooth the browser looks.
- `static/js/main.js`: `video_frame` handler now builds a `Blob`/object URL
  from the binary frame instead of a `data:` URI, with the previous URL
  revoked each time to avoid leaking browser memory.
- Window lifecycle (`open_window()` / `close_window()` /
  `_bring_to_front()` in `custom_mario_env.py`, wired through
  `open_agent_window()` / `close_agent_window()` in `agent_logic.py` and
  the `app.py` socket handlers): the pop-up window now opens and comes to
  the foreground on every "Start Testing" click, stays open-but-frozen on
  "Stop Testing" (nothing closes it - the stream and window both simply
  stop being updated, together), and closes on "Reset Dashboard" or a page
  refresh/disconnect.

## Verification

- `_fast_obs()` benchmarked in isolation: 11.92x faster, and its output
  checked pixel-by-pixel against the old path - 100% identical, including
  after simulating the downstream grayscale+84x84 resize the model
  actually receives (see `IMPLEMENTATION.md` git history / session log for
  the exact scripts and numbers if needed again later).
- Full per-decision breakdown profiled stage by stage (predict / step /
  render / encode) before and after, isolating exactly which stage
  improved and by how much, rather than trusting a single before/after
  total.
- Window open -> close -> reopen -> step 5 frames -> close, run live
  end-to-end with real pixel output produced at each stage.
