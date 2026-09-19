"""The TESTABLE-pixel mask: what fraction of the world Mario can actually reach.

═══════════════════════════════════════════════════════════════════════════
WHY THIS FILE WAS REWRITTEN — the previous denominator was impossible
═══════════════════════════════════════════════════════════════════════════
An earlier version computed a "coverable" set of 7,606,986 px and used it as
the coverage denominator. That number is larger than the entire world:

    world raster  9087 x 600 = 5,452,200
    claimed coverable        = 7,606,986      (+2,154,786 impossible px)

Measured cause, exactly: the anchor filter allowed `y >= GRID_Y0` (-320)
instead of `y >= 0`, so 2,907,840 px of SKY ABOVE THE WORLD were counted as
coverable. It also applied no jump-height constraint at all, so it counted
altitude Mario cannot reach even where the sky is inside the world.

A coverage percentage computed against that number is meaningless. It is
retired. Nothing in this file may use it.

═══════════════════════════════════════════════════════════════════════════
THE THREE METHODS
═══════════════════════════════════════════════════════════════════════════
All three work in WORLD coordinates (origin 0,0, size 9087x600). The camera
is never involved: `Level1` keeps every collider and sprite in world space
and subtracts `viewport` only at blit time.

  METHOD A - geometric openness.        INFORMATIONAL ONLY.
      "Could a 30x40 box sit here without touching a solid?" No gravity, no
      jump limit. Over-counts badly. Kept only so the reconciliation report
      can show how much the physics constraints actually remove.

  METHOD B - jump envelope.
      Adds two physical facts: an anchor must be inside the world, and it
      must be no higher than JUMP_RISE_PX above the highest STANDABLE anchor
      within JUMP_REACH_PX horizontally. Removes unreachable altitude.

  METHOD C - connectivity BFS.          ADOPTED when it agrees with B.
      Everything in B, plus: the anchor must be reachable from the spawn
      point through a chain of real moves. Falls propagate down through free
      space; jumps propagate up to JUMP_RISE_PX and out to JUMP_REACH_PX but
      are BLOCKED BY GEOMETRY as they propagate, so a platform sealed behind
      a wall is not counted merely because it is the right height.

  Required ordering:  C <= B <= A.  Violation means a bug, not a result.

═══════════════════════════════════════════════════════════════════════════
DEFINITION OF "TESTABLE"
═══════════════════════════════════════════════════════════════════════════
A world pixel is TESTABLE if some physically reachable placement of Mario's
collider covers it. Not "a box fits here" - reachable. Consequences:

  * y < 0 (sky above the world)      -> NOT testable. Recorded as anomalous.
  * y >= 600 (below the death plane) -> NOT testable. Recorded as anomalous.
  * interior of a solid              -> NOT testable. Recorded as anomalous.
  * unreachable altitude             -> NOT testable.

Anything Mario's collider occupies OUTSIDE this mask is not coverage. It is
`noncoverage_px`, and it is NOT automatically a bug: ordinary play leaves the
testable set constantly, via jump arcs above the world, pit deaths below it,
and the engine's own 1-5 px collision resolution. Each such pixel is
classified once, at build time, and only the genuinely impossible classes
reach the glitch tracker. See CLASS_NAMES below.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from common.fileio import atomic_write

from . import config

Mask = np.ndarray                  # 2-D bool array, (rows, cols)
Stats = dict[str, Any]


# ══════════════════════════════════════════════════════════════════════════
# WINDOW HELPERS
# Boolean "is any source within this offset range" along one axis, in O(n)
# via a cumulative sum. Used for morphological dilation, the sliding-window
# minimum of the ceiling, and the anchor -> pixel expansion.
# ══════════════════════════════════════════════════════════════════════════
def _window_any(mask: Mask, lo: int, hi: int, axis: int) -> Mask:
    """out[i] = any(mask[i+lo : i+hi+1]) along `axis`, clipped at the edges."""
    n = mask.shape[axis]
    pad_lo, pad_hi = max(0, -lo), max(0, hi)
    pad = [(0, 0)] * mask.ndim
    pad[axis] = (pad_lo, pad_hi)
    m = np.pad(mask.astype(np.int32), pad)
    c = np.cumsum(m, axis=axis)
    zshape = list(c.shape)
    zshape[axis] = 1
    c = np.concatenate([np.zeros(zshape, c.dtype), c], axis=axis)
    start = np.arange(n) + lo + pad_lo
    end = np.arange(n) + hi + 1 + pad_lo
    lo_c = np.take(c, np.clip(start, 0, c.shape[axis] - 1), axis=axis)
    hi_c = np.take(c, np.clip(end, 0, c.shape[axis] - 1), axis=axis)
    return (hi_c - lo_c) > 0


def _summed_area(mask: Mask) -> np.ndarray:
    """Integral image with a zero-padded first row/column."""
    sat = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    sat[1:, 1:] = np.cumsum(np.cumsum(mask, axis=0, dtype=np.int64),
                            axis=1, dtype=np.int64)
    return sat


def mask_fingerprint(mask: Mask) -> str:
    """SHA-256 of the packed mask. Identifies WHICH denominator a coverage
    file was recorded against, so a state built on the retired 7,606,986
    mask cannot silently load as if it were compatible."""
    return hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()


# ══════════════════════════════════════════════════════════════════════════
# GEOMETRY
# ══════════════════════════════════════════════════════════════════════════
def rasterize_solids(level_state: Any, *, x0: int = 0, y0: int = 0,
                     w: int | None = None, h: int | None = None) -> tuple[Mask, int]:
    """Burns every collider group into a mask, in WORLD coordinates.

    `level_state` is a live `Level1`. The groups are read from outside rather
    than by modifying mario_clone/, which stays read-only.
    """
    w = config.LEVEL_W if w is None else w
    h = config.LEVEL_H if h is None else h
    solid = np.zeros((h, w), dtype=bool)
    groups = ('ground_group', 'pipe_group', 'step_group',
              'brick_group', 'coin_box_group')
    n_rects = 0
    for name in groups:
        group = getattr(level_state, name, None)
        if group is None:
            continue
        for sprite in group:
            r = sprite.rect
            n_rects += 1
            gx0, gy0 = r.x - x0, r.y - y0
            gx1, gy1 = gx0 + r.w, gy0 + r.h
            gx0, gy0 = max(0, gx0), max(0, gy0)
            gx1, gy1 = min(w, gx1), min(h, gy1)
            if gx1 > gx0 and gy1 > gy0:
                solid[gy0:gy1, gx0:gx1] = True
    return solid, n_rects


def anchor_grids(solid: Mask, mario_w: int | None = None,
                 mario_h: int | None = None) -> tuple[Mask, Mask]:
    """Valid and standable anchor sets, in WORLD coordinates.

    An ANCHOR is a top-left placement of Mario's collider. It is VALID when
    the whole mario_w x mario_h box is inside the world and free of solids.
    It is STANDABLE when the row immediately beneath the box contains solid
    ground under Mario's footprint - i.e. he can rest there.

    Anchor index (ay, ax) maps to world pixel (ay, ax) directly; the grids
    are (LEVEL_H - mario_h + 1) x (LEVEL_W - mario_w + 1).
    """
    mario_w = config.MARIO_SMALL_W if mario_w is None else mario_w
    mario_h = config.MARIO_SMALL_H if mario_h is None else mario_h
    h, w = solid.shape

    sat = _summed_area(solid)
    ah, aw = h - mario_h + 1, w - mario_w + 1
    occupied = (sat[mario_h:mario_h + ah, mario_w:mario_w + aw]
                - sat[0:ah, mario_w:mario_w + aw]
                - sat[mario_h:mario_h + ah, 0:aw]
                + sat[0:ah, 0:aw])
    valid = (occupied == 0)

    # Standable: any solid pixel in the row directly below the footprint.
    # Row index ay + mario_h. The bottom anchor row has no row beneath it
    # inside the world, so it can never be standable.
    below = _window_any(solid, 0, mario_w - 1, axis=1)[:, :aw]
    standable = np.zeros_like(valid)
    standable[:ah - 1] = valid[:ah - 1] & below[mario_h:mario_h + ah - 1]
    return valid, standable


def anchors_to_pixels(anchors: Mask, mario_w: int | None = None,
                      mario_h: int | None = None) -> Mask:
    """Every world pixel covered by some anchor in the set."""
    mario_w = config.MARIO_SMALL_W if mario_w is None else mario_w
    mario_h = config.MARIO_SMALL_H if mario_h is None else mario_h
    ah, aw = anchors.shape[0], anchors.shape[1]
    full = np.zeros((config.LEVEL_H, config.LEVEL_W), dtype=bool)
    full[:ah, :aw] = anchors
    out = _window_any(full, -(mario_h - 1), 0, axis=0)
    out = _window_any(out, -(mario_w - 1), 0, axis=1)
    return out


# ══════════════════════════════════════════════════════════════════════════
# METHOD A — geometric openness. INFORMATIONAL ONLY.
# ══════════════════════════════════════════════════════════════════════════
def method_a(solid: Mask, mario_w: int | None = None,
             mario_h: int | None = None) -> tuple[Mask, Stats]:
    """"A box fits here." No gravity, no jump limit. Over-counts."""
    valid, _standable = anchor_grids(solid, mario_w, mario_h)
    px = anchors_to_pixels(valid, mario_w, mario_h)
    return px, {'anchors': int(valid.sum()), 'px': int(px.sum())}


# ══════════════════════════════════════════════════════════════════════════
# METHOD B — jump envelope.
# ══════════════════════════════════════════════════════════════════════════
NO_CEILING = np.int32(1 << 20)      # sentinel: no standable ground in reach


def column_ceiling(standable: Mask) -> np.ndarray:
    """Per anchor-column, the highest anchor y a jump can reach. UNCLAMPED.

    Deliberately not clipped at y = 0. Mario's collider legitimately rises
    ABOVE the top of the world during a jump - the engine does not stop it -
    so the honest ceiling is negative wherever a high platform sits within
    jumping distance. Clamping it here would make every normal jump arc look
    like an out-of-world anomaly, which is exactly the false signal the
    noncoverage classifier exists to avoid.

    Columns with no standable ground within JUMP_REACH_PX get NO_CEILING, so
    nothing above them is ever considered reachable.
    """
    ah = standable.shape[0]
    ys = np.arange(ah, dtype=np.int32)[:, None]
    col_top = np.where(standable.any(axis=0),
                       np.min(np.where(standable, ys, NO_CEILING), axis=0),
                       NO_CEILING).astype(np.int32)
    r = config.JUMP_REACH_PX
    padded = np.pad(col_top, (r, r), constant_values=NO_CEILING)
    win = np.lib.stride_tricks.sliding_window_view(padded, 2 * r + 1)
    return win.min(axis=1) - config.JUMP_RISE_PX


def method_b(solid: Mask, mario_w: int | None = None,
             mario_h: int | None = None) -> tuple[Mask, Stats]:
    """Adds the reachable-altitude constraint to Method A.

    Per world column, find the highest STANDABLE anchor within
    JUMP_REACH_PX horizontally. Nothing more than JUMP_RISE_PX above it can
    be reached by a jump from anywhere nearby, so every anchor above that
    ceiling is discarded.

    This is a genuine upper bound and does NOT check connectivity: a ledge
    at a legal height still counts even if it is walled off. Method C adds
    that.
    """
    valid, standable = anchor_grids(solid, mario_w, mario_h)
    ah = valid.shape[0]
    ceiling = column_ceiling(standable)
    reachable = valid & (np.arange(ah, dtype=np.int32)[:, None] >= ceiling[None, :])
    px = anchors_to_pixels(reachable, mario_w, mario_h)
    return px, {
        'valid_anchors': int(valid.sum()),
        'standable_anchors': int(standable.sum()),
        'anchors': int(reachable.sum()),
        'px': int(px.sum()),
    }


# ══════════════════════════════════════════════════════════════════════════
# METHOD C — connectivity BFS. The adopted method.
# ══════════════════════════════════════════════════════════════════════════
def _shift(mask: Mask, dy: int, dx: int) -> Mask:
    out = np.zeros_like(mask)
    ys = slice(max(0, dy), mask.shape[0] + min(0, dy))
    xs = slice(max(0, dx), mask.shape[1] + min(0, dx))
    ys_src = slice(max(0, -dy), mask.shape[0] + min(0, -dy))
    xs_src = slice(max(0, -dx), mask.shape[1] + min(0, -dx))
    out[ys, xs] = mask[ys_src, xs_src]
    return out


def _propagate(seed: Mask, passable: Mask, dy: int, dx: int, steps: int) -> Mask:
    """Flood `seed` in one direction through `passable`, up to `steps` cells.

    Stepwise rather than a single dilation ON PURPOSE: propagating one cell
    at a time and re-intersecting with `passable` means geometry BLOCKS the
    spread. A one-shot rectangular dilation would happily jump straight
    through a wall, which is exactly the looseness Method C exists to remove.
    """
    cur = seed.copy()
    for _ in range(steps):
        nxt = cur | (_shift(cur, dy, dx) & passable)
        if np.array_equal(nxt, cur):
            break
        cur = nxt
    return cur


def method_c(solid: Mask, spawn_xy: tuple[int, int], mario_w: int | None = None,
             mario_h: int | None = None, coarse: int | None = None,
             max_iters: int = 60) -> tuple[Mask, Stats]:
    """Everything in Method B, plus reachability from the spawn point.

    Runs on a `coarse`-px anchor lattice; a coarse cell is open if ANY fine
    anchor inside it is valid, which keeps the result a SUPERSET of the
    fine-grained truth rather than eroding it. `coarse=1` is no lattice at
    all - the exact answer - and is what config.REACHABILITY_LATTICE selects
    by default. Coarser values exist for the convergence study that justified
    that choice (see the table in config.py); they cost less and, after the
    Method-B trim, were measured to agree to within 116 px.

    Fixed-point loop, from the spawn anchor:
      1. FALL   - propagate down through free space (gravity is free).
      2. JUMP   - from cells that are STANDABLE and already reached, rise up
                  to JUMP_RISE_PX and spread out to JUMP_REACH_PX, blocked by
                  geometry at every step. Both orderings (up-then-out and
                  out-then-up) are unioned, because a real arc does both at
                  once and neither ordering alone covers the envelope.
    Repeats until nothing new is reached.
    """
    mario_w = config.MARIO_SMALL_W if mario_w is None else mario_w
    mario_h = config.MARIO_SMALL_H if mario_h is None else mario_h
    coarse = config.REACHABILITY_LATTICE if coarse is None else coarse
    valid, standable = anchor_grids(solid, mario_w, mario_h)
    ah, aw = valid.shape

    ch, cw = -(-ah // coarse), -(-aw // coarse)
    pad_v = np.zeros((ch * coarse, cw * coarse), dtype=bool)
    pad_v[:ah, :aw] = valid
    pad_s = np.zeros_like(pad_v)
    pad_s[:ah, :aw] = standable
    c_valid = pad_v.reshape(ch, coarse, cw, coarse).any(axis=(1, 3))
    c_stand = pad_s.reshape(ch, coarse, cw, coarse).any(axis=(1, 3))

    reach = np.zeros_like(c_valid)
    sx, sy = spawn_xy
    reach[min(ch - 1, sy // coarse), min(cw - 1, sx // coarse)] = True
    if not c_valid[min(ch - 1, sy // coarse), min(cw - 1, sx // coarse)]:
        raise ReachabilityMismatch(
            f"spawn anchor {spawn_xy} is not a valid anchor - the collider "
            f"size or the level geometry has changed")

    rise = -(-config.JUMP_RISE_PX // coarse)
    span = -(-config.JUMP_REACH_PX // coarse)

    for _ in range(max_iters):
        before = int(reach.sum())

        # 1. gravity
        reach = _propagate(reach, c_valid, 1, 0, ch)

        # 2. jump, from standable ground already reached
        launch = reach & c_stand
        if launch.any():
            up = _propagate(launch, c_valid, -1, 0, rise)
            out = _propagate(up, c_valid, 0, -1, span)
            out = _propagate(out, c_valid, 0, 1, span)

            side = _propagate(launch, c_valid, 0, -1, span)
            side = _propagate(side, c_valid, 0, 1, span)
            side = _propagate(side, c_valid, -1, 0, rise)

            reach |= (out | side)

        if int(reach.sum()) == before:
            break

    # Upsample back to the fine anchor lattice and re-apply exact validity.
    fine = np.repeat(np.repeat(reach, coarse, axis=0), coarse, axis=1)
    fine = fine[:ah, :aw] & valid
    px = anchors_to_pixels(fine, mario_w, mario_h)
    return px, {
        'valid_anchors': int(valid.sum()),
        'standable_anchors': int(standable.sum()),
        'anchors': int(fine.sum()),
        'px': int(px.sum()),
        'coarse': coarse,
    }


# ══════════════════════════════════════════════════════════════════════════
# NONCOVERAGE CLASSIFICATION
#
# "Outside the testable mask" is NOT the same as "a bug". Measured against
# the 40-episode bootstrap of ordinary 6M-policy play, every single one of
# the 9,687 out-of-mask pixels was explained by documented, normal engine
# behaviour:
#
#   above the world, y in [-29, -1]   ordinary jump arcs; the engine does not
#                                     stop the collider at y = 0
#   below the world, y in [600, 646]  pit deaths; level1.py:1319 kills Mario
#                                     at rect.y > 600, and 100% of these were
#                                     in columns that are genuinely bottomless
#   inside a solid, depth 1..5        the engine's own collision resolution,
#                                     never observed at 6+ (PENETRATION_TOL)
#
# Treating those as glitch evidence would make the QA report cry wolf during
# completely normal play, which is worse than not reporting at all. So each
# out-of-world pixel is classified ONCE, at mask-build time, against the same
# geometry the denominator came from, and only the genuinely impossible
# classes reach the glitch system.
#
# A fourth outcome is kept honest and separate: CONNECTIVITY_GAP means Mario
# reached somewhere Method B allows but Method C's BFS did not find. That is
# a shortfall in MY reachability model, not a bug in the game, and it must
# not be reported as either coverage or a glitch.
#
# ─── THREE MORE EXPLANATIONS, ADDED FROM MEASURED FALSE POSITIVES ───
# The 6.02M short validation reported 599 anomalous pixels, all of class
# deep_penetration, where the 40-episode bootstrap had reported 0. Both
# clusters turned out to be defects in THIS classifier, not in the game, and
# they have different causes - so they get different classes rather than one
# widened tolerance. (Raising PENETRATION_TOL would have buried both without
# explaining either, and would have blunted the detector everywhere.)
#
# A third cause surfaced immediately afterwards, in the controlled
# COMPLETE-phase validation: 1,079 FLOOR_CLIP pixels in a single episode
# (9,951 over a longer run) from ordinary pit deaths at x 3708-3744. See the
# pit rule below - the lesson is the same one all three teach, that a static
# model of a live engine goes wrong in the direction of crying wolf, so an
# unexplained anomaly is worth measuring before it is worth believing.
#
#   MUTABLE_SOLID   579 px at x 5070-5094, y 371-401: the deep interior of
#                   brick20 (a 43x43 Brick at 5058,365). The class map is
#                   built ONCE from the level's opening geometry, but the
#                   geometry is not constant - level1.py:776 calls
#                   brick.kill() when big Mario hits a contents-less brick
#                   from below, and the block is gone for the rest of the
#                   episode. 29 of the 31 bricks are destructible this way,
#                   and every brick and coin box also RISES ~18 px while
#                   BUMPED, freeing the bottom of its own footprint. Standing
#                   where a destroyed block used to be is not a clip.
#
#   SWEEP_ARTIFACT  20 px at x 7748-7751, y 458-462, in pipe6's top-right
#                   corner - geometry that never moves or dies. This one is
#                   not about the game at all: coverage.record() marks the
#                   BOUNDING BOX of the previous and current collider rects,
#                   which for a diagonal move past a convex corner contains
#                   pixels NEITHER rect occupied. Measured: prev=(7747,412)
#                   and cur=(7758,423), dx=+11 dy=+11 (inside the engine's
#                   14/12 single-frame envelope), both placements entirely
#                   collision-free, union box x 7747-7787 y 412-462 - which
#                   covers the whole cluster. The collider never entered the
#                   pipe; the evidence is simply coarser than a pixel.
#
# That is why SWEEP_ARTIFACT is a MODEL GAP and not an EXPECTED class: it
# describes a limit of how coverage is RECORDED, not something the engine
# did. Keeping it separate means the report still says out loud that the
# anomaly detector cannot resolve finer than the recorded box.
#
# Both are applied ONLY to pixels this classifier would otherwise call
# DEEP_PENETRATION, so no other class can move. Together they account for
# 64,106 of 556,128 deep pixels (11.5%) - the remaining 492,022 stay
# anomalous, so the detector is corrected, not disarmed.
# ══════════════════════════════════════════════════════════════════════════
CLS_TESTABLE = 0            # inside the mask; ordinary coverage

CLS_JUMP_ARC = 1            # EXPECTED: above the world within jump reach
CLS_PIT_FALL = 2            # EXPECTED: below the world in a bottomless column
CLS_COLLISION_TOL = 3       # EXPECTED: engine collision-resolution overlap

CLS_CONNECTIVITY_GAP = 4    # MODEL GAP: legal altitude, BFS missed it

CLS_IMPOSSIBLE_SKY = 5      # ANOMALOUS: higher than any jump can reach
CLS_FLOOR_CLIP = 6          # ANOMALOUS: below the world through solid floor
CLS_DEEP_PENETRATION = 7    # ANOMALOUS: inside a solid past the tolerance
CLS_OUTSIDE_LEVEL = 8       # ANOMALOUS: beyond the level horizontally
CLS_UNREACHABLE_ALT = 9     # ANOMALOUS: in-world altitude no jump reaches

CLS_MUTABLE_SOLID = 10      # EXPECTED: solid at build time, removable in play
CLS_SWEEP_ARTIFACT = 11     # MODEL GAP: only the recorded box reaches it

CLASS_NAMES = {
    CLS_TESTABLE: 'testable',
    CLS_JUMP_ARC: 'jump_arc',
    CLS_PIT_FALL: 'pit_fall',
    CLS_COLLISION_TOL: 'collision_tolerance',
    CLS_CONNECTIVITY_GAP: 'connectivity_gap',
    CLS_IMPOSSIBLE_SKY: 'impossible_sky',
    CLS_FLOOR_CLIP: 'floor_clip',
    CLS_DEEP_PENETRATION: 'deep_penetration',
    CLS_OUTSIDE_LEVEL: 'outside_level',
    CLS_UNREACHABLE_ALT: 'unreachable_altitude',
    CLS_MUTABLE_SOLID: 'mutable_solid',
    CLS_SWEEP_ARTIFACT: 'sweep_artifact',
}
EXPECTED_CLASSES = (CLS_JUMP_ARC, CLS_PIT_FALL, CLS_COLLISION_TOL,
                    CLS_MUTABLE_SOLID)
MODEL_GAP_CLASSES = (CLS_CONNECTIVITY_GAP, CLS_SWEEP_ARTIFACT)
ANOMALOUS_CLASSES = (CLS_IMPOSSIBLE_SKY, CLS_FLOOR_CLIP, CLS_DEEP_PENETRATION,
                     CLS_OUTSIDE_LEVEL, CLS_UNREACHABLE_ALT)


def _dilate(mask: Mask, radius: int) -> Mask:
    """Chessboard dilation by `radius`, via two 1-D window passes."""
    out = _window_any(mask, -radius, radius, axis=0)
    return _window_any(out, -radius, radius, axis=1)


def mutable_solid_mask(level_state: Any) -> Mask:
    """Solid at build time, but which the ENGINE itself can clear in play.

    The class map is a single static snapshot; the level is not. Two engine
    behaviours move solid geometry after the mask is built, and a collider
    standing in the space they vacate is doing nothing wrong:

      destroyed  level1.adjust_mario_for_y_brick_collisions calls
                 brick.kill() when big Mario hits a brick from below and the
                 brick has no contents (level1.py:776). The sprite leaves
                 brick_group permanently, so its ENTIRE footprint is free for
                 the rest of the episode. Bricks holding coins or a star are
                 never killed, so only `contents is None` qualifies.

      bumped     Brick.bumped() and the coin boxes run rect.y += y_vel from
                 y_vel = -6 under gravity 1.2, so the sprite sits up to
                 BUMP_RISE_PX higher than its rest position for those frames.
                 The bottom of its resting footprint is genuinely empty then.

    Returns a WORLD-sized mask of exactly those footprints - no collider
    geometry is applied here. It does not need to be: classify_noncoverage
    intersects this with the DEEP_PENETRATION set before using it, so the
    only pixels it can ever excuse are ones already established to be buried
    more than PENETRATION_TOL inside a solid.
    """
    out = np.zeros((config.LEVEL_H, config.LEVEL_W), dtype=bool)

    def burn(top: int, bottom: int, left: int, right: int) -> None:
        t, b = max(0, top), min(config.LEVEL_H, bottom)
        left_, right_ = max(0, left), min(config.LEVEL_W, right)
        if b > t and right_ > left_:
            out[t:b, left_:right_] = True

    for sprite in getattr(level_state, 'brick_group', ()):
        r = sprite.rect
        if getattr(sprite, 'contents', None) is None:
            burn(r.top, r.bottom, r.left, r.right)       # killable outright
        burn(r.bottom - config.BUMP_RISE_PX, r.bottom, r.left, r.right)
    for sprite in getattr(level_state, 'coin_box_group', ()):
        r = sprite.rect
        burn(r.bottom - config.BUMP_RISE_PX, r.bottom, r.left, r.right)
    return out


def sweep_coverable(solid: Mask, forms: Sequence[tuple[int, int]] | None = None,
                    max_dx: int | None = None,
                    max_dy: int | None = None) -> Mask:
    """Every pixel a LEGAL recorded sweep box can contain.

    SpatialCoverage.record() does not mark the path the collider traced; it
    marks the axis-aligned BOUNDING BOX of the previous and current rects
    (coverage.py:record). For a diagonal move past a convex corner that box
    contains pixels neither rect ever occupied, and the classifier then reads
    them as if the collider had been there.

    So this computes the honest resolution limit of that evidence: the union,
    over every pair of collision-free placements one engine frame apart, of
    the box record() would mark for them. A pixel inside this set is not
    proof that anything reached it, and must not be reported as a glitch.

    Build-time only and deliberately exhaustive - it is a few minutes over
    the (2*max_dx+1) * (2*max_dy+1) displacement envelope per form, run once
    per mask, rather than an approximation nobody could later audit.
    """
    forms = ([(config.MARIO_SMALL_W, config.MARIO_SMALL_H),
              (config.MARIO_BIG_W, config.MARIO_BIG_H)]
             if forms is None else list(forms))
    max_dx = config.MAX_FRAME_DX if max_dx is None else max_dx
    max_dy = config.MAX_FRAME_DY if max_dy is None else max_dy
    ah, aw = solid.shape
    out = np.zeros(solid.shape, dtype=bool)

    for w, h in forms:
        # An ANCHOR is legal when the whole w x h collider is in-world and
        # free - the same definition anchor_grids() uses for validity. The
        # window runs FORWARD from the anchor (it covers rows ay..ay+h-1),
        # which is the opposite direction to anchors_to_pixels().
        occupied = _window_any(_window_any(solid, 0, h - 1, axis=0),
                               0, w - 1, axis=1)
        anchors = ~occupied
        anchors[max(0, ah - h + 1):, :] = False
        anchors[:, max(0, aw - w + 1):] = False

        for dx in range(-max_dx, max_dx + 1):
            for dy in range(-max_dy, max_dy + 1):
                # d and -d describe the SAME set of ordered pairs seen from
                # the other end, and produce the same boxes: pairs_-d is
                # pairs_d translated by d, and the extra translation by
                # min(0, -d) lands both on the same origin, with the same
                # (w+|dx|) x (h+|dy|) size. So half the envelope is redundant
                # and skipping it halves the build with no change in result.
                if dy < 0 or (dy == 0 and dx < 0):
                    continue
                # Anchors that are free AND whose partner one frame away is
                # free too - only those pairs can actually occur.
                pairs = anchors & _shift(anchors, -dy, -dx)
                if not pairs.any():
                    continue
                # record() marks [min(ax,bx), max(ax,bx)+w) x the same in y,
                # i.e. a (w+|dx|) x (h+|dy|) box whose top-left corner is the
                # anchor offset by the negative part of the displacement.
                origin = _shift(pairs, min(0, dy), min(0, dx))
                box = _window_any(origin, -(h + abs(dy) - 1), 0, axis=0)
                box = _window_any(box, -(w + abs(dx) - 1), 0, axis=1)
                out |= box
    return out


def classify_noncoverage(solid_world: Mask, testable_world: Mask, b_world: Mask,
                         mario_w: int | None = None, mario_h: int | None = None,
                         mutable_world: Mask | None = None,
                         sweep_world: Mask | None = None) -> np.ndarray:
    """Assigns every PADDED-GRID pixel a class code. Built once, with the mask.

    Computed at build time rather than per-query so the taxonomy is versioned
    and fingerprinted alongside the denominator it belongs to - a coverage
    state cannot be re-interpreted under a different set of rules later.

    `mutable_world` (mutable_solid_mask) and `sweep_world` (sweep_coverable)
    are optional because both need inputs this function does not have - the
    live level state and a few minutes of work respectively. Omitting either
    only means its explanation is not applied, and the pixels it would have
    accounted for stay DEEP_PENETRATION; nothing else changes.
    """
    mario_w = config.MARIO_SMALL_W if mario_w is None else mario_w
    mario_h = config.MARIO_SMALL_H if mario_h is None else mario_h
    _valid, standable = anchor_grids(solid_world, mario_w, mario_h)

    cls = np.full((config.GRID_H, config.GRID_W), CLS_OUTSIDE_LEVEL, np.uint8)
    ys = np.arange(config.GRID_H) + config.GRID_Y0
    xs = np.arange(config.GRID_W) + config.GRID_X0
    in_x = (xs >= 0) & (xs < config.LEVEL_W)

    # ── above the world: legal iff within the per-column jump ceiling ──
    # The anchor ceiling is indexed by anchor column; a PIXEL column x is
    # covered by anchors x-mario_w+1 .. x, so take the lowest ceiling of any
    # anchor that could put a collider pixel there.
    ceil_anchor = column_ceiling(standable)
    per_col = np.full(config.LEVEL_W, NO_CEILING, np.int32)
    per_col[:ceil_anchor.shape[0]] = ceil_anchor
    ceil_px = np.lib.stride_tricks.sliding_window_view(
        np.pad(per_col, (mario_w - 1, 0), constant_values=NO_CEILING),
        mario_w).min(axis=1).astype(np.int64)

    full_ceil = np.full(config.GRID_W, NO_CEILING, np.int64)
    full_ceil[in_x] = ceil_px
    sky = (ys < 0)[:, None] & in_x[None, :]
    reachable_sky = sky & (ys[:, None] >= full_ceil[None, :])
    cls[sky] = CLS_IMPOSSIBLE_SKY
    cls[reachable_sky] = CLS_JUMP_ARC

    # ── below the world: legal iff the column is genuinely bottomless ──
    # level1.py kills Mario at rect.y > LEVEL_H, so the collider legitimately
    # occupies a band below the floor for the frame or two before the check
    # fires. Falling below through a column that HAS a floor is a clip.
    #
    # "Bottomless" asks about the FLOOR, not about the whole column. The
    # earlier rule was `~solid.any(axis=0)` - no solid anywhere from sky to
    # floor - and that is not the same question. The pit at x 3683-3773 has
    # brick6..brick13 floating at y 193-235 far above it and no ground at all
    # beneath, so ordinary pit deaths there were classified FLOOR_CLIP, which
    # is ANOMALOUS: the controlled COMPLETE-phase validation produced 1,079
    # such pixels in a single episode (x 3708-3744, y 600-646, Mario in
    # state=fall), and a longer run 9,951. Floating geometry high overhead
    # cannot catch a falling collider, so it must not make a pit look solid.
    #
    # The floor is what the bottom of the column holds: if a big Mario's full
    # height of it is empty there is nothing left to land on. That is a
    # strict widening - every column the old rule called bottomless still is
    # (verified: 367 -> 458 columns, 0 lost) - and it needs no level-specific
    # constant.
    pit_col = ~solid_world[config.LEVEL_H - config.MARIO_BIG_H:, :].any(axis=0)
    pit_near = _window_any(pit_col[None, :], -(mario_w - 1), mario_w - 1,
                           axis=1)[0]
    full_pit = np.zeros(config.GRID_W, bool)
    full_pit[in_x] = pit_near
    depth_limit = config.LEVEL_H + config.MAX_FRAME_DY + config.MARIO_BIG_H
    below = (ys >= config.LEVEL_H)[:, None] & in_x[None, :]
    cls[below] = CLS_FLOOR_CLIP
    cls[below & full_pit[None, :] & (ys < depth_limit)[:, None]] = CLS_PIT_FALL

    # ── inside the world ──
    wy0, wx0 = -config.GRID_Y0, -config.GRID_X0
    sl = (slice(wy0, wy0 + config.LEVEL_H), slice(wx0, wx0 + config.LEVEL_W))
    tol = config.PENETRATION_TOL
    near_solid = _dilate(solid_world, tol)

    # Assigned in order of increasing precedence: each line overwrites the
    # previous where they overlap, and the most benign explanation that fits
    # is the one that survives. Anything still bearing the initial value at
    # the end is a pixel nothing normal accounts for.
    w = np.full((config.LEVEL_H, config.LEVEL_W), CLS_UNREACHABLE_ALT, np.uint8)
    # Everything a tolerated overlap explains: the solid's own outer shell,
    # plus the free nooks that only open up while the collider is sunk into
    # it. (Measured: a 5x4 pocket under a pipe's lower-left corner at
    # x=1202..1206, unreachable by any fully-free 30x40 placement, but swept
    # by the collider during a perfectly ordinary 1-5 px wall overlap.)
    w[near_solid] = CLS_COLLISION_TOL
    # Legal altitude but the BFS never got there: my model, not a game bug.
    # This OUTRANKS the overlap explanation on purpose. Most of the level is
    # within 6 px of something solid, so letting collision tolerance absorb
    # these would hide the one signal that says the denominator is too small.
    w[b_world & ~testable_world] = CLS_CONNECTIVITY_GAP
    # Deeper than the engine's resolution ever goes - PENETRATION_TOL is 6
    # and the observed maximum over the whole bootstrap was 5.
    deep = _erode_tol(solid_world, tol)
    w[deep] = CLS_DEEP_PENETRATION
    # ── the two measured false-positive causes, applied ONLY to `deep` ──
    # Scoped to the deep set on purpose: these explanations answer "why is
    # the collider recorded inside a solid", and restricting them there makes
    # it provable that no other class can move when they are switched on.
    # Least specific first, so the most concrete explanation survives.
    if sweep_world is not None:
        w[deep & sweep_world] = CLS_SWEEP_ARTIFACT
    if mutable_world is not None:
        w[deep & mutable_world] = CLS_MUTABLE_SOLID
    w[testable_world] = CLS_TESTABLE
    cls[sl] = w
    return cls


def _erode_tol(solid: Mask, tol: int) -> Mask:
    """Solid pixels deeper than `tol` from any free pixel (chessboard)."""
    deep: Mask = ~_dilate(~solid, tol) & solid
    return deep


# ══════════════════════════════════════════════════════════════════════════
# RECONCILIATION
# ══════════════════════════════════════════════════════════════════════════
class ReachabilityMismatch(RuntimeError):
    """A mask file does not match the current grid, format, or geometry."""


class ReconciliationError(RuntimeError):
    """The three methods disagree in a way that means a bug, not a result."""


def build_testable(level_state: Any, spawn_xy: tuple[int, int],
                   tolerance: float = 0.05) -> tuple[Mask, Mask, np.ndarray, Stats]:
    """Runs all three methods, reconciles them, and returns the adopted mask.

    Adoption rule, from the brief and enforced here:
      * C <= B <= A must hold, or something is wrong with the code.
      * If C and B agree within `tolerance`, adopt C - it is the tighter and
        more physically meaningful of the two.
      * If they diverge by more, raise rather than silently pick one.
    """
    solid, n_rects = rasterize_solids(level_state)
    a_px, a = method_a(solid)
    b_px, b = method_b(solid)
    c_raw, c = method_c(solid, spawn_xy)

    # ─── LATTICE TRIM — measured across three resolutions, not assumed ───
    # A coarse Method-C cell is open if ANY fine anchor inside it is open,
    # which rounds OUTWARD, so C can sit a few pixels above B's exact
    # per-column ceiling and chained jumps compound it. The convergence study
    # measured that overshoot directly at three lattice pitches:
    #
    #     4 px   overshoot 11,257    trimmed 4,013,839
    #     2 px   overshoot  5,821    trimmed 4,013,839
    #     1 px   overshoot      0    trimmed 4,013,723   <- adopted, exact
    #
    # It halves with the pitch and vanishes at 1 px, which is what a pure
    # discretisation artifact does. At the adopted lattice C is a strict
    # subset of B and this trim is a no-op that costs one AND - kept as an
    # assertion in code form, since B is the proven ALTITUDE bound and C's
    # contribution is CONNECTIVITY. If a coarser lattice is ever selected the
    # trim starts doing real work again, and the size check below still
    # distinguishes rounding from a genuine bug.
    overshoot = int((c_raw & ~b_px).sum())
    c_px = c_raw & b_px
    c['px'] = int(c_px.sum())
    c['lattice_overshoot_px'] = overshoot

    world = config.LEVEL_W * config.LEVEL_H
    problems = []
    if not (c['px'] <= b['px'] <= a['px']):
        problems.append(f"ordering C({c['px']:,}) <= B({b['px']:,}) "
                        f"<= A({a['px']:,}) violated")
    for name, px in (('A', a['px']), ('B', b['px']), ('C', c['px'])):
        if px > world:
            problems.append(f"method {name} = {px:,} exceeds the world "
                            f"raster {world:,}")
    if (c_px & ~b_px).any():
        problems.append("Method C still exceeds B after the lattice trim")
    if overshoot > 0.01 * b['px']:
        problems.append(
            f"Method C overshot B by {overshoot:,} px "
            f"({100.0 * overshoot / b['px']:.2f}%), far more than lattice "
            f"rounding explains - investigate before adopting")
    if (a_px & solid).any() or (b_px & solid).any() or (c_px & solid).any():
        problems.append("a method marked solid interior as testable")
    if problems:
        raise ReconciliationError("; ".join(problems))

    delta = abs(b['px'] - c['px']) / max(1, b['px'])
    if delta <= tolerance:
        adopted, method = c_px, 'C'
    else:
        raise ReconciliationError(
            f"Methods B ({b['px']:,}) and C ({c['px']:,}) differ by "
            f"{100 * delta:.2f}%, above the {100 * tolerance:.0f}% tolerance. "
            f"Investigate before adopting either.")

    class_map = classify_noncoverage(
        solid, adopted, b_px,
        mutable_world=mutable_solid_mask(level_state),
        sweep_world=sweep_coverable(solid))

    stats = {
        'world_raster_px': world,
        'solid_px': int(solid.sum()),
        'n_solid_rects': n_rects,
        'method_a_px': a['px'], 'method_b_px': b['px'], 'method_c_px': c['px'],
        'method_a_anchors': a['anchors'],
        'method_b_anchors': b['anchors'], 'method_c_anchors': c['anchors'],
        'valid_anchors': b['valid_anchors'],
        'standable_anchors': b['standable_anchors'],
        'adopted_method': method,
        'testable_total': int(adopted.sum()),
        'bc_delta_pct': 100.0 * delta,
        'lattice': c['coarse'],
        'c_lattice_overshoot_px': overshoot,
        'connectivity_excluded_px': b['px'] - c['px'],
        'spawn_xy': list(spawn_xy),
    }
    return solid, adopted, class_map, stats


# ══════════════════════════════════════════════════════════════════════════
# PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════
def _embed_in_grid(world_mask: Mask) -> Mask:
    """Place a world-sized mask into the padded coverage grid.

    The coverage bitmap is padded (see config.GRID_*) so out-of-world
    positions can be RECORDED without clamping - clamping would fabricate
    coverage at the boundary. The testable mask is False everywhere in that
    padding by construction, which is what makes `visited & ~testable` a
    clean anomaly signal.
    """
    out = np.zeros((config.GRID_H, config.GRID_W), dtype=bool)
    y0, x0 = -config.GRID_Y0, -config.GRID_X0
    out[y0:y0 + config.LEVEL_H, x0:x0 + config.LEVEL_W] = world_mask
    return out


def save_masks(path: str, solid_world: Mask, testable_world: Mask,
               class_map: np.ndarray, stats: Stats) -> None:
    """Writes the mask bundle atomically.

    Atomic because a half-written mask sitting next to a good model
    checkpoint would silently change the coverage denominator - a failure
    that looks like a result.
    """
    solid = _embed_in_grid(solid_world)
    testable = _embed_in_grid(testable_world)
    with atomic_write(path) as fh:
        _write_mask_bundle(fh, solid, testable, class_map, stats)


def _write_mask_bundle(fh: Any, solid: Mask, testable: Mask,
                       class_map: np.ndarray, stats: Stats) -> None:
    np.savez_compressed(
        fh,
        solid_packed=np.packbits(solid),
        testable_packed=np.packbits(testable),
        # Why the taxonomy is stored rather than recomputed: it is derived
        # from the SAME geometry as the denominator, so shipping them
        # together means a coverage state can never be re-interpreted under
        # rules that have quietly drifted from the mask it was scored against.
        class_map=class_map,
        lattice=np.int64(stats['lattice']),
        c_lattice_overshoot_px=np.int64(stats['c_lattice_overshoot_px']),
        grid_geom=np.array([config.GRID_X0, config.GRID_Y0,
                            config.GRID_W, config.GRID_H], dtype=np.int64),
        world_dims=np.array([config.LEVEL_W, config.LEVEL_H], dtype=np.int64),
        testable_total=np.int64(stats['testable_total']),
        world_raster_px=np.int64(stats['world_raster_px']),
        solid_px=np.int64(stats['solid_px']),
        n_solid_rects=np.int64(stats['n_solid_rects']),
        method_a_px=np.int64(stats['method_a_px']),
        method_b_px=np.int64(stats['method_b_px']),
        method_c_px=np.int64(stats['method_c_px']),
        valid_anchors=np.int64(stats['valid_anchors']),
        standable_anchors=np.int64(stats['standable_anchors']),
        adopted_method=np.str_(stats['adopted_method']),
        bc_delta_pct=np.float64(stats['bc_delta_pct']),
        testable_fingerprint=np.str_(mask_fingerprint(testable)),
        format_version=np.int64(config.COVERAGE_FORMAT_VERSION),
    )


def load_masks(path: str | None = None) -> tuple[Mask, Mask, Stats]:
    """Loads the mask bundle, refusing anything that does not match config.

    Refuses rather than adapts: a mask built for a different grid or a
    different denominator would silently produce a wrong coverage
    percentage, which is worse than a crash because it looks like a result.
    """
    path = config.REACHABLE_MASK_PATH if path is None else path
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run `python tools/build_reachability.py` first."
        )
    # Closed on the way out: an open .npz holds a Windows file lock on the very
    # path tools/build_reachability.py atomically replaces.
    with np.load(path, allow_pickle=False) as d:
        return _parse_mask_bundle(d, path)


def _parse_mask_bundle(d: Any, path: str) -> tuple[Mask, Mask, Stats]:
    """load_masks() on an open NpzFile: every check, then the two masks."""
    if 'testable_packed' not in d:
        raise ReachabilityMismatch(
            f"{path} predates the testable-mask correction (it stores the "
            f"retired 'coverable' set, whose total exceeded the world "
            f"raster). Rebuild with tools/build_reachability.py.")

    found = int(d['format_version'])
    if found != config.COVERAGE_FORMAT_VERSION:
        raise ReachabilityMismatch(
            f"{path}: format_version {found}, "
            f"expected {config.COVERAGE_FORMAT_VERSION}")

    geom = tuple(int(v) for v in d['grid_geom'])
    expected = (config.GRID_X0, config.GRID_Y0, config.GRID_W, config.GRID_H)
    if geom != expected:
        raise ReachabilityMismatch(
            f"{path}: grid_geom {geom}, expected {expected}")

    n = config.GRID_W * config.GRID_H
    solid = np.unpackbits(d['solid_packed'])[:n].reshape(
        config.GRID_H, config.GRID_W).astype(bool)
    testable = np.unpackbits(d['testable_packed'])[:n].reshape(
        config.GRID_H, config.GRID_W).astype(bool)

    total = int(d['testable_total'])
    if total != int(testable.sum()):
        raise ReachabilityMismatch(
            f"{path}: stored testable_total {total:,} does not match the "
            f"mask itself ({int(testable.sum()):,})")
    if total > int(d['world_raster_px']):
        raise ReachabilityMismatch(
            f"{path}: testable_total {total:,} exceeds the world raster "
            f"{int(d['world_raster_px']):,} - impossible")

    meta = {k: (int(d[k]) if d[k].dtype.kind in 'iu' else d[k].item())
            for k in ('testable_total', 'world_raster_px', 'solid_px',
                      'n_solid_rects', 'method_a_px', 'method_b_px',
                      'method_c_px', 'valid_anchors', 'standable_anchors',
                      'adopted_method', 'bc_delta_pct', 'lattice',
                      'c_lattice_overshoot_px')}
    meta['testable_fingerprint'] = str(d['testable_fingerprint'])
    return solid, testable, meta


_CLASS_MAP_CACHE: np.ndarray | None = None


def load_class_map(path: str | None = None) -> np.ndarray:
    """The noncoverage taxonomy grid. Cached; loaded only when reporting.

    Deliberately NOT held by SpatialCoverage: it is 9.8 MB and every
    SubprocVecEnv worker carries a coverage channel, but only the parent ever
    renders a breakdown. Keeping it lazy keeps 8 workers x 9.8 MB out of
    memory for a table nobody asks them for.
    """
    global _CLASS_MAP_CACHE
    if _CLASS_MAP_CACHE is None:
        path = config.REACHABLE_MASK_PATH if path is None else path
        with np.load(path, allow_pickle=False) as d:
            if 'class_map' not in d:
                raise ReachabilityMismatch(
                    f"{path} has no noncoverage class map. Rebuild with "
                    f"tools/build_reachability.py.")
            cm = d['class_map']
        if cm.shape != (config.GRID_H, config.GRID_W):
            raise ReachabilityMismatch(
                f"class_map shape {cm.shape}, expected "
                f"{(config.GRID_H, config.GRID_W)}")
        _CLASS_MAP_CACHE = cm
    return _CLASS_MAP_CACHE
