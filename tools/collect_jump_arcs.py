"""Record REAL jump arcs from the engine: exploration_data/jump_arcs.npz.

    python tools/collect_jump_arcs.py

Method A/B/C bound a jump by a RECTANGLE - up to JUMP_RISE_PX and out to
JUMP_REACH_PX at the same time. No arc does both: a jump that has risen 183 px
is at its apex and can only be a few hundred px from where it started at that
height, and it is falling back from there. The rectangle over-counts, and
measured against real play it over-counts by about 250,000 px of the
denominator (see reachability.apply_arc_envelope).

This records the truth instead. From flat ground it drives the bare engine
(one frame per step, no reward wrapper) through a structured family of
take-offs - stand / walk / sprint run-ups in both directions, every jump
action, the jump key held for 3..90 frames, three things to do after release -
and keeps each collider trajectory, frame by frame, relative to the take-off.
Each trajectory is then extended with the engine's terminal fall
(y_vel = MAX_Y_VEL, x_vel kept: the engine has no air drag) so a launch from a
platform keeps falling past the height of the ground it was measured on.

The output is small (a few hundred paths) and deterministic: the engine has
no randomness in this setting, and the paths are de-duplicated and sorted.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.cli import prepare_tool

ROOT = prepare_tool()

import numpy as np

from common.fileio import atomic_write
from exploration import config

NOOP, WR, WRJ, RR, RRJ, J, WL, _CROUCH, RL, LJ = range(10)

# Where each family starts. Both have hundreds of px of flat ground on the
# side they run toward, before the first pipe.
RIGHT = ("R", 250, WR, RR, (WRJ, RRJ, J), WL)
LEFT = ("L", 1300, WL, RL, (LJ, J), WR)
RUNUP_FRAMES = (20, 45)
HOLD_FRAMES = (3, 8, 14, 20, 26, 34, 50, 90)
SETTLE_FRAMES = 30
MAX_AIR_FRAMES = 140
EXT_FRAMES = 70                      # terminal-fall extension appended to every arc
TERMINAL_DY = 11                     # constants.MAX_Y_VEL


def sequences() -> list[tuple[str, int, int, int, int, int, int]]:
    """(direction, start_x, run-up frames, run-up action, jump action, hold, after)."""
    out: list[tuple[str, int, int, int, int, int, int]] = []
    for direction, start, walk, run, jumps, opposite in (RIGHT, LEFT):
        runups = [(0, NOOP)] + [(n, a) for n in RUNUP_FRAMES for a in (walk, run)]
        for n, a in runups:
            for jump in jumps:
                for hold in HOLD_FRAMES:
                    out.extend((direction, start, n, a, jump, hold, after)
                               for after in (run, NOOP, opposite))
    return out


def trial(env: Any, start_x: int, runup_n: int, runup_a: int, jump_a: int, hold: int,
          after_a: int) -> list[tuple[int, int, float]] | None:
    """Frame-by-frame (dx, dy_down, x_vel) from take-off to landing, or None."""
    env.reset()
    env.game.state.mario.rect.x = start_x
    info: dict = {}
    for _ in range(SETTLE_FRAMES):
        _o, _r, _t, _tr, info = env.step(NOOP)
    for _ in range(runup_n):
        _o, _r, _t, _tr, info = env.step(runup_a)
    if not info["on_ground"]:
        return None
    x0, y0 = info["mario_rect"][0], info["mario_rect"][1]
    pts: list[tuple[int, int, float]] = []
    airborne = False
    for f in range(MAX_AIR_FRAMES):
        _o, _r, terminated, _tr, info = env.step(jump_a if f < hold else after_a)
        rect = info["mario_rect"]
        pts.append((rect[0] - x0, rect[1] - y0, float(info["x_vel"])))
        airborne = airborne or not info["on_ground"]
        if terminated or (airborne and info["on_ground"]):
            break
    return pts if airborne else None


def extend(pts: list[tuple[int, int, float]]) -> tuple[tuple[int, int], ...]:
    seq = [(0, 0)] + [(dy, dx) for dx, dy, _v in pts]
    vx = round(pts[-1][2])
    for _ in range(EXT_FRAMES):
        seq.append((seq[-1][0] + TERMINAL_DY, seq[-1][1] + vx))
    return tuple(seq)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument("--out", default=config.JUMP_ARCS_PATH)
    args = ap.parse_args(argv)
    from custom_mario_env import CustomMarioEnv

    env = CustomMarioEnv()
    seqs = sequences()
    print(f"  {len(seqs)} take-off sequences from flat ground")
    paths: set[tuple[tuple[int, int], ...]] = set()
    t0 = time.time()
    try:
        for i, s in enumerate(seqs):
            pts = trial(env, *s[1:])
            if pts:
                paths.add(extend(pts))
            if i % 100 == 0:
                print(f"  {i:>4}/{len(seqs)}   {len(paths)} distinct arcs   "
                      f"{time.time() - t0:.0f}s")
    finally:
        env.close_window()

    ordered = sorted(paths)
    depth = max(len(p) for p in ordered)
    arr = np.zeros((len(ordered), depth, 2), dtype=np.int16)
    for i, p in enumerate(ordered):
        row = np.array(p, dtype=np.int16)
        arr[i, :len(row)] = row
        arr[i, len(row):] = row[-1]              # a finished arc holds its last position
    rise = int(-arr[:, :, 0].min())
    print(f"  {len(ordered)} distinct arcs, {depth} frames, max rise {rise} px "
          f"(config.JUMP_RISE_PX = {config.JUMP_RISE_PX})")
    if rise > config.JUMP_RISE_PX:
        raise SystemExit(f"an arc rose {rise} px, above the {config.JUMP_RISE_PX} px the "
                         f"rectangle model was allowed - one of the two is wrong")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with atomic_write(args.out) as fh:
        np.savez_compressed(fh, paths=arr, n_sequences=np.int64(len(seqs)),
                            ext_frames=np.int64(EXT_FRAMES), max_rise=np.int64(rise))
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
