"""Persistent world-space spatial coverage.

DEFINITION
    A world pixel (x, y) is EXPLORED the first time Mario's live collider rect
    has occupied it, or swept across it, at the end of any game frame.

Recorded on every substep - all four per agent decision, because
GlitchHunterWrapper sits below MaxAndSkipObservation(skip=4) - from the live
`mario.rect`, in world coordinates, using the current form's real dimensions
(30x40 small, 40x80 big). Nothing else counts: not HUD, not background art,
not anything camera-relative. Only where the physical collider has been.

WHY THE SWEEP IS EXACT
    Measured maximum per-frame displacement is |dx| <= 14, |dy| <= 12. The
    minimum collider dimension is 30 wide by 40 tall. Both displacements are
    strictly smaller, so consecutive rects ALWAYS overlap on both axes and
    their swept region is exactly their AABB union. There are no tunnelling
    gaps to fill, and the whole thing costs one numpy slice.

COST
    Measured 0.0073 ms for the bitmap write (0.0114 ms with visit counts)
    against a ~1.8 ms env.step(). About 0.6% overhead, which is why coverage
    is tracked at literal 1x1 world-pixel resolution rather than coarsened.
    Coarse cells appear only in the frontier index, never in coverage itself.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import uuid
from typing import Protocol, runtime_checkable

import numpy as np

from . import config

# ── shared control slots ──────────────────────────────────────────────────
# A tiny float64 block that lives beside the coverage bitmap. Workers are
# separate PROCESSES, so a plain module-level `config.NOVELTY_WEIGHT_MULT`
# set by the trainer would only ever change the parent's copy - every worker
# would keep the value it was forked/spawned with, and StagnationCallback
# would appear to work while doing nothing at all. Routing it through shared
# memory is the only way the escalation actually reaches the policies
# generating the rollouts.
CTRL_NOVELTY_MULT = 0
CTRL_SLOTS = 8


def make_shm_names(tag=None):
    """Unique names for one training run's shared blocks.

    The NAME is the only thing that survives cloudpickling into a
    SubprocVecEnv worker, so it - not the SharedMemory object - is what gets
    passed into make_env().
    """
    tag = tag or uuid.uuid4().hex[:12]
    return {
        'visited': f"gh_vis_{tag}",
        'counts': f"gh_cnt_{tag}",
        'control': f"gh_ctl_{tag}",
    }


class CoverageCheckpointMismatch(RuntimeError):
    """A coverage file does not belong to the model checkpoint beside it."""


class CoverageFormatMismatch(RuntimeError):
    """A coverage file was written by an incompatible version or grid."""


@runtime_checkable
class CoverageChannel(Protocol):
    """One measurable dimension of "how much of the thing have we explored".

    Only SpatialCoverage implements this today. The interface exists so a
    future channel - e.g. InteractionCoverage keyed on
    (x//B, y//B, vx_bucket, vy_bucket, on_ground, contact_flags) - can drop in
    without touching the wrapper, the checkpoint format (which stores a dict
    of per-channel state dicts) or the trainer.

    This is why record() takes a dict rather than two rects: SpatialCoverage
    uses only the rects today, but the signature already carries the velocity
    and contact information a state-space channel would need.
    """

    name: str

    def record(self, obs: dict) -> int: ...
    def covered_testable(self) -> int: ...
    def remaining(self) -> int | None: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, d: dict) -> None: ...


def _config_hash() -> str:
    """Fingerprints the tunables that would change what a number MEANS.

    Deliberately not every constant - reward weights can be retuned without
    invalidating a coverage file. Geometry, format, and the identity of the
    TESTABLE MASK itself are included: a state recorded against a different
    denominator describes a different quantity, and must not load as if it
    were comparable.
    """
    payload = {
        'format_version': config.COVERAGE_FORMAT_VERSION,
        'grid': [config.GRID_X0, config.GRID_Y0, config.GRID_W, config.GRID_H],
        'world': [config.LEVEL_W, config.LEVEL_H],
        'frontier_cell': config.FRONTIER_CELL,
        'testable_fingerprint': config.TESTABLE_FINGERPRINT,
        'testable_total': config.TESTABLE_TOTAL,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()


class SpatialCoverage:
    """The world-pixel coverage bitmap, optionally backed by shared memory.

    Two backing modes, one API:

    * Local  - plain numpy arrays. Used by the dashboard and the tests.
    * Shared - numpy views onto multiprocessing.shared_memory blocks created
               by the parent before SubprocVecEnv forks. Workers attach BY
               NAME (the only thing that survives cloudpickling) and must
               close but never unlink; only the parent unlinks.

    Call sites never branch on which is in use.

    The bitmap is stored on the PADDED grid (config.GRID_*), deliberately
    larger than the world, so an out-of-bounds glitch can be recorded where
    it actually happened instead of being clamped to the edge. The testable
    mask is False throughout that padding, which is what makes
    `visited & ~testable` a clean NONCOVERAGE signal rather than coverage.
    Most of that signal is normal behaviour, not glitches - see
    noncoverage_breakdown().
    """

    name = "spatial"

    def __init__(self, testable_mask=None, shm_names=None, grid=None,
                 create_shared=False):
        self.x0, self.y0, self.w, self.h = grid or (
            config.GRID_X0, config.GRID_Y0, config.GRID_W, config.GRID_H)

        self._shm = []
        self._owns_shm = create_shared
        self.control = None
        self.shm_names = dict(shm_names) if shm_names else None

        if shm_names:
            self._attach_shared(shm_names, create=create_shared)
        else:
            self.visited = np.zeros((self.h, self.w), dtype=np.uint8)
            self.visit_count = (
                np.zeros((self.h, self.w), dtype=np.uint16)
                if config.TRACK_VISIT_COUNTS else None)

        self.testable = testable_mask
        self.testable_total = (int(testable_mask.sum())
                               if testable_mask is not None else None)

        # Per-episode state
        self.prev_rect = None
        self.episode_new = 0
        self.steps_since_new_pixel = 0
        self.oob_events = 0

        # Frontier snapshot. Held stationary between refreshes so the
        # potential function below is piecewise policy-invariant.
        self._frontier_cells = np.zeros((0, 2), dtype=np.int64)
        self._steps_since_frontier = 10 ** 9
        self._remaining_cached = None
        # Bumped on every real refresh. The reward wrapper watches this and
        # SKIPS the shaping delta on the step a refresh lands: Phi changed
        # because the map moved, not because the agent did, and paying for
        # that would be paying for someone else's discovery.
        self.frontier_version = 0

        # Ring buffer of recent per-episode new-pixel counts, driving the
        # adaptive target.
        self.episode_new_history = []

        self._lock = None
        if config.STRICT_COVERAGE_LOCK:
            import multiprocessing
            self._lock = multiprocessing.Manager().Lock()

    # ── shared-memory plumbing ────────────────────────────────────────────
    def _attach_shared(self, names, create=False):
        from multiprocessing import shared_memory

        n_px = self.w * self.h
        vis_name = names['visited']
        self.visited = self._map(shared_memory, vis_name, n_px,
                                 np.uint8, create, (self.h, self.w))
        if config.TRACK_VISIT_COUNTS and names.get('counts'):
            self.visit_count = self._map(shared_memory, names['counts'],
                                         n_px * 2, np.uint16, create,
                                         (self.h, self.w))
        else:
            self.visit_count = None

        if names.get('control'):
            self.control = self._map(shared_memory, names['control'],
                                     CTRL_SLOTS * 8, np.float64, create,
                                     (CTRL_SLOTS,))
            if create:
                self.control[CTRL_NOVELTY_MULT] = config.NOVELTY_WEIGHT_MULT

    def _map(self, shared_memory, name, nbytes, dtype, create, shape):
        if create:
            try:
                shm = shared_memory.SharedMemory(name=name, create=True,
                                                 size=nbytes)
            except FileExistsError:
                shm = shared_memory.SharedMemory(name=name)
        else:
            shm = shared_memory.SharedMemory(name=name)
        self._shm.append(shm)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        if create:
            arr[:] = 0
        return arr

    # ── shared control ────────────────────────────────────────────────────
    @property
    def novelty_mult(self) -> float:
        """The live novelty multiplier, read from shared memory if present.

        Falls back to the static config value for local (test / dashboard)
        instances, so call sites never branch on the backing mode.
        """
        if self.control is not None:
            return float(self.control[CTRL_NOVELTY_MULT])
        return float(config.NOVELTY_WEIGHT_MULT)

    def set_novelty_mult(self, value: float) -> None:
        """PARENT ONLY. Escalates the novelty multiplier for every worker."""
        value = float(min(config.NOVELTY_MULT_MAX, max(1.0, value)))
        if self.control is not None:
            self.control[CTRL_NOVELTY_MULT] = value
        else:
            config.NOVELTY_WEIGHT_MULT = value

    def close(self):
        """Detach from shared memory. Workers call this; they never unlink."""
        for shm in self._shm:
            try:
                shm.close()
            except Exception:
                pass

    def unlink(self):
        """Destroy the shared blocks. PARENT ONLY, at shutdown.

        Unlinking from a worker would pull the grid out from under every other
        worker, so this is deliberately separate from close().
        """
        for shm in self._shm:
            try:
                shm.unlink()
            except Exception:
                pass

    # ── recording ─────────────────────────────────────────────────────────
    def begin_episode(self):
        """Resets per-episode state, keeping the persistent bitmap.

        prev_rect is cleared so the spawn teleport is NOT swept. Without this
        the first recorded sweep spans from wherever Mario died to the spawn
        point, painting a huge false corridor across the level. (An early
        measurement of "max dy = 40" was exactly this artefact.)
        """
        if self.episode_new or self.episode_new_history:
            self.episode_new_history.append(float(self.episode_new))
            del self.episode_new_history[:-config.TARGET_HISTORY_LEN]
        self.prev_rect = None
        self.episode_new = 0
        self.steps_since_new_pixel = 0

    def record(self, obs: dict) -> int:
        """Marks the swept region and returns how many pixels were new.

        `obs` carries prev/cur rects plus velocity and contact state. Only the
        rects are used today - see CoverageChannel for why the rest is in the
        signature.

        The count returned is RAW new pixels, including any outside the
        testable mask. Novelty reward is paid on testable pixels only - see
        record_testable() - so a clipping glitch cannot be farmed for reward.
        """
        cur = obs['cur_rect']
        prev = obs.get('prev_rect') or self.prev_rect

        cx0, cy0, cw, ch = cur
        if prev is None:
            sx0, sy0 = cx0, cy0
            sx1, sy1 = cx0 + cw, cy0 + ch
        else:
            px0, py0, pw, ph = prev
            sx0, sy0 = min(px0, cx0), min(py0, cy0)
            sx1, sy1 = max(px0 + pw, cx0 + cw), max(py0 + ph, cy0 + ch)

        self.prev_rect = (cx0, cy0, cw, ch)

        gx0, gy0 = sx0 - self.x0, sy0 - self.y0
        gx1, gy1 = sx1 - self.x0, sy1 - self.y0

        # Out-of-grid is NOT clamped - clamping would fabricate coverage at
        # the boundary. Being outside the padded grid is itself a glitch
        # signal, so it is counted and reported instead.
        if gx0 < 0 or gy0 < 0 or gx1 > self.w or gy1 > self.h:
            self.oob_events += 1
            gx0, gy0 = max(0, gx0), max(0, gy0)
            gx1, gy1 = min(self.w, gx1), min(self.h, gy1)
            if gx1 <= gx0 or gy1 <= gy0:
                self.steps_since_new_pixel += 1
                return 0

        if self._lock is not None:
            with self._lock:
                n_new = self._mark(gy0, gy1, gx0, gx1)
        else:
            n_new = self._mark(gy0, gy1, gx0, gx1)

        self.episode_new += n_new
        if n_new:
            self.steps_since_new_pixel = 0
        else:
            self.steps_since_new_pixel += 1
        self._steps_since_frontier += 1
        if n_new:
            self._remaining_cached = None
        return n_new

    def _mark(self, gy0, gy1, gx0, gx1):
        """Marks the swept box; returns newly-claimed TESTABLE pixels.

        The bitmap records everything the collider touched, because that is
        the evidence. The RETURN value - which drives novelty reward - counts
        only pixels inside the testable mask, so reaching sky or the inside
        of a solid can never be farmed for reward. Those land in
        noncoverage_px instead.
        """
        sub = self.visited[gy0:gy1, gx0:gx1]
        fresh = (sub == 0)
        if self.testable is not None:
            n_new = int(np.count_nonzero(
                fresh & self.testable[gy0:gy1, gx0:gx1]))
        else:
            n_new = int(np.count_nonzero(fresh))
        if fresh.any():
            sub |= 1
        if self.visit_count is not None:
            # Saturating, and deliberately non-atomic: visit counts are a
            # heatmap diagnostic and are NEVER a reward input, so a lost
            # increment under cross-worker contention costs nothing.
            cnt = self.visit_count[gy0:gy1, gx0:gx1]
            np.minimum(cnt, 65534, out=cnt)
            cnt += 1
        return n_new

    # ── metrics ───────────────────────────────────────────────────────────
    def total_unique(self) -> int:
        """Every world pixel the collider has occupied, testable or not.

        Diagnostic only. This is NOT the coverage numerator: it includes
        noncoverage pixels (sky, below the death plane, solid interiors), and
        counting those as coverage is exactly the mistake that produced a
        denominator larger than the world it described. Use
        covered_testable().
        """
        return int(np.count_nonzero(self.visited))

    # ── the separated quantities (world / testable / covered / anomalous) ──
    def covered_testable(self) -> int:
        """THE coverage numerator: visited pixels inside the testable mask."""
        if self.testable is None:
            return 0
        return int(np.count_nonzero(self.visited & self.testable))

    def noncoverage_px(self) -> int:
        """Visited pixels OUTSIDE the testable mask. Never coverage.

        This is the umbrella total, and MOST OF IT IS NORMAL. Mario's
        collider leaves the testable set constantly during ordinary play: a
        jump arc rises above the top of the world, a pit death carries him
        below the floor, and the engine's own collision resolution lets him
        sink a few pixels into a block before pushing him back out.

        None of that is a bug, so none of it may be reported as one. Use
        anomalous_px() for the genuinely impossible subset, and
        noncoverage_breakdown() for the full split.
        """
        if self.testable is None:
            return 0
        return int(np.count_nonzero(self.visited & ~self.testable))

    def _class_counts(self) -> dict:
        """Visited-pixel counts per noncoverage class. Lazy: loads the map."""
        from . import reachability
        cm = reachability.load_class_map()
        vis = self.visited.astype(bool)
        counts = np.bincount(cm[vis].ravel(),
                             minlength=max(reachability.CLASS_NAMES) + 1)
        return {reachability.CLASS_NAMES[i]: int(counts[i])
                for i in reachability.CLASS_NAMES}

    def anomalous_px(self) -> int:
        """ONLY the genuinely impossible noncoverage. Feeds the glitch system.

        Deliberately narrower than noncoverage_px(). Over the entire 40-episode
        bootstrap of ordinary play this is 0, and that is the point: a detector
        that fires on normal jumping, normal pit deaths and the engine's
        documented 1-5 px collision tolerance would be reporting the game's
        own rules as defects, and nobody would trust the ones that matter.
        """
        if self.testable is None:
            return 0
        from . import reachability
        counts = self._class_counts()
        return sum(counts[reachability.CLASS_NAMES[c]]
                   for c in reachability.ANOMALOUS_CLASSES)

    def expected_noncoverage_px(self) -> int:
        """Noncoverage explained by documented, normal engine behaviour."""
        if self.testable is None:
            return 0
        from . import reachability
        counts = self._class_counts()
        return sum(counts[reachability.CLASS_NAMES[c]]
                   for c in reachability.EXPECTED_CLASSES)

    def model_gap_px(self) -> int:
        """Reached, at a legal altitude, but Method C's BFS never found it.

        Not a game bug and not coverage: a shortfall in MY reachability
        model. Reported on its own so it can be acted on honestly - a
        non-zero value here means the denominator is slightly too small and
        the BFS needs revisiting, not that Mario did something wrong.
        """
        if self.testable is None:
            return 0
        from . import reachability
        counts = self._class_counts()
        return sum(counts[reachability.CLASS_NAMES[c]]
                   for c in reachability.MODEL_GAP_CLASSES)

    def remaining(self) -> int | None:
        """Testable pixels not yet covered. Exactly total - covered."""
        if self.testable is None:
            return None
        if self._remaining_cached is None:
            self._remaining_cached = self.testable_total - self.covered_testable()
        return self._remaining_cached

    def coverage_pct(self) -> float:
        """covered_testable / testable_total * 100. The ONLY coverage figure.

        Never uses the world raster (5,452,200), never uses the padded grid
        (9,830,400), never counts noncoverage pixels.
        """
        if not self.testable_total:
            return 0.0
        return 100.0 * self.covered_testable() / self.testable_total

    def noncoverage_breakdown(self) -> dict:
        """The full split of out-of-mask pixels by cause.

        Three groups, and the distinction between them is the whole point:

          expected   proven-normal engine behaviour. NOT glitch evidence, and
                     must never be reported as such.
          model_gap  reached at a legal altitude the BFS did not find. A
                     shortfall in the reachability model, not in the game.
          anomalous  genuinely impossible. This, and only this, is what the
                     glitch system should act on.
        """
        if self.testable is None:
            return {}
        from . import reachability as R
        counts = self._class_counts()
        group = {g: {R.CLASS_NAMES[c]: counts[R.CLASS_NAMES[c]] for c in cs}
                 for g, cs in (('expected', R.EXPECTED_CLASSES),
                               ('model_gap', R.MODEL_GAP_CLASSES),
                               ('anomalous', R.ANOMALOUS_CLASSES))}
        out = {'total': self.noncoverage_px(), **group}
        for g in ('expected', 'model_gap', 'anomalous'):
            out[g + '_total'] = sum(group[g].values())
        return out

    def assert_consistent(self) -> None:
        """Hard invariants. Cheap enough to run at every checkpoint.

        These are the exact failures that let an impossible denominator ship
        last time, so they are checked rather than trusted.
        """
        if self.testable is None:
            return
        covered = self.covered_testable()
        if covered > self.testable_total:
            raise AssertionError(
                f"covered_testable {covered:,} exceeds testable_total "
                f"{self.testable_total:,} - impossible")
        if self.remaining() != self.testable_total - covered:
            raise AssertionError("remaining != testable_total - covered")
        pct = self.coverage_pct()
        if not (0.0 <= pct <= 100.0):
            raise AssertionError(f"coverage_pct {pct} outside [0, 100]")
        if self.testable_total > config.WORLD_RASTER_PX:
            raise AssertionError(
                f"testable_total {self.testable_total:,} exceeds the world "
                f"raster {config.WORLD_RASTER_PX:,} - impossible")
        # Coverage and noncoverage must PARTITION the recorded bitmap: every
        # visited pixel is one or the other, never both, never neither.
        # (The previous form of this check was `visited & testable &
        # ~testable`, which is empty for any input and therefore proved
        # nothing.)
        noncov = self.noncoverage_px()
        total = self.total_unique()
        if covered + noncov != total:
            raise AssertionError(
                f"covered_testable {covered:,} + noncoverage {noncov:,} != "
                f"total recorded {total:,} - the two do not partition the "
                f"bitmap")

    # ── frontier ──────────────────────────────────────────────────────────
    def refresh_frontier(self, viewport_x=0, force=False):
        """Rebuilds the coarse frontier snapshot.

        Cells are 40x40 and a cell qualifies if it holds any reachable pixel
        that is still unvisited.

        Cells whose right edge lies left of viewport.x are DROPPED. The camera
        is one-way - update_viewport() only ever increases viewport.x, and
        Mario is hard-clamped to viewport.x + 5 - so anything behind it is
        physically unreachable this episode. Pointing the agent at it would be
        pointing at something impossible.
        """
        if not force and self._steps_since_frontier < config.FRONTIER_REFRESH_STEPS:
            return
        self._steps_since_frontier = 0
        self._remaining_cached = None
        self.frontier_version += 1
        self._frontier_cells = self._frontier_centres(viewport_x)

    def _frontier_centres(self, viewport_x):
        """World-space centres of unexplored testable cells ahead of the camera.

        Computed fresh from the live bitmap every call. refresh_frontier()
        stores the result as the PBRS snapshot; frontier_query() uses it
        directly and stores nothing.
        """
        empty = np.zeros((0, 2), dtype=np.int64)
        if self.testable is None:
            return empty
        cell = config.FRONTIER_CELL
        ch, cw = self.h // cell, self.w // cell
        unexplored = self.testable[:ch * cell, :cw * cell] & (
            self.visited[:ch * cell, :cw * cell] == 0)
        blocks = unexplored.reshape(ch, cell, cw, cell).any(axis=(1, 3))

        cy, cx = np.nonzero(blocks)
        if cy.size == 0:
            return empty
        world_cx = cx * cell + self.x0 + cell // 2
        world_cy = cy * cell + self.y0 + cell // 2
        ahead = (cx * cell + cell + self.x0) >= viewport_x
        return np.stack([world_cx[ahead], world_cy[ahead]],
                        axis=1).astype(np.int64)

    def frontier_query(self, viewport_x, points):
        """(cells ahead of the camera, nearest-cell distance per point).

        READ-ONLY, and deliberately independent of the PBRS snapshot. That
        snapshot refreshes every FRONTIER_REFRESH_STEPS substeps and can be
        left over from the PREVIOUS episode, built at a camera position far to
        the right of where the new episode starts - so it may report nothing
        ahead when the level is in fact full of unexplored space. The
        lifecycle's "nothing left to find" and "closing on the frontier"
        judgements would both be wrong on it. Refreshing the snapshot instead
        would change what the frontier-shaping REWARD sees, so this computes
        its own view and leaves the reward's untouched.

        Distances are inf when nothing remains ahead.
        """
        cells = self._frontier_centres(viewport_x)
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if cells.shape[0] == 0:
            return 0, np.full(pts.shape[0], np.inf)
        d = cells[None, :, :].astype(np.float64) - pts[:, None, :]
        return int(cells.shape[0]), np.sqrt((d ** 2).sum(axis=2)).min(axis=1)

    def phi(self, x, y) -> float:
        """Potential for frontier shaping: -min(dist, cap) / cap.

        Ng/Harada/Russell potential-based shaping. Returns 0 when no reachable
        frontier remains ahead, so the term vanishes rather than misleads.

        TWO HONEST CAVEATS, because "cannot be farmed" would be too strong:

        1. PBRS is policy-invariant under the DISCOUNTED objective PPO
           optimises. The UNDISCOUNTED sum around a closed loop is
           (gamma - 1) * sum(Phi), which with gamma < 1 and Phi <= 0 is a
           small POSITIVE residue - so pacing back and forth does pay
           something. Worst case per substep is
           FRONTIER_WEIGHT * (1 - gamma) * 1.0 = 0.001, against a drought
           ceiling of 0.02 per substep for failing to find new pixels: a 20x
           margin. Both numbers were rescaled together after the first
           calibration run (see config.FRONTIER_WEIGHT for why). The residue
           is dominated, not absent.
        2. A global potential that other workers mutate is not strictly
           policy-invariant either. Holding the frontier snapshot stationary
           between refreshes restores the guarantee piecewise.
        """
        if self._frontier_cells.shape[0] == 0:
            return 0.0
        d = self._frontier_cells - np.array([x, y], dtype=np.int64)
        dist = float(np.sqrt((d.astype(np.float64) ** 2).sum(axis=1)).min())
        return -min(dist, config.FRONTIER_DIST_CAP) / config.FRONTIER_DIST_CAP

    # ── adaptive target ───────────────────────────────────────────────────
    def target_informed(self) -> bool:
        """Is the target measured from this worker's own history yet?

        Before TARGET_MIN_HISTORY episodes, episode_target() returns the bare
        TARGET_FLOOR, which says nothing about how much this policy finds on
        this map.
        """
        return len(self.episode_new_history) >= config.TARGET_MIN_HISTORY

    def episode_target(self) -> int:
        """New-pixel target for the current episode.

        Tracks the median of recent episodes, so it follows the natural decay
        curve (measured: 72352 -> 10955 -> 16576 -> 11972 -> 1787 -> 48) and
        is clamped by what actually remains, so it can never demand more space
        than exists.

        This exerts PRESSURE. It is not a guarantee - RL cannot promise a
        fixed number of new pixels per episode.
        """
        hist = self.episode_new_history
        baseline = (float(np.median(hist[-config.TARGET_HISTORY_LEN:]))
                    if self.target_informed() else float(config.TARGET_FLOOR))
        remaining = self.remaining()
        ceiling = (max(config.TARGET_FLOOR,
                       int(config.TARGET_REMAIN_FRAC * remaining))
                   if remaining is not None else None)
        target = config.TARGET_BETA * baseline
        target = max(target, config.TARGET_FLOOR)
        if ceiling is not None:
            target = min(target, ceiling)
        return int(target)

    def novelty_shape(self, n_new: int) -> float:
        """The UNWEIGHTED, capped novelty term: min(CAP, sqrt(n_new/N_REF)).

        Separated from novelty_reward() so tools/calibrate_reward.py can
        accumulate the shape of an episode's novelty independently of the
        weight, then solve for the weight in closed form instead of doing a
        search over repeated 40-episode rollouts.
        """
        if n_new <= 0:
            return 0.0
        return min(config.NOVELTY_CAP, math.sqrt(n_new / config.N_REF))

    def novelty_reward(self, n_new: int) -> float:
        """sqrt-scaled, bounded novelty. Zero for a revisit, never negative."""
        if n_new <= 0:
            return config.REVISIT_REWARD
        return (config.NOVELTY_WEIGHT * self.novelty_mult) * self.novelty_shape(n_new)

    # ── persistence ───────────────────────────────────────────────────────
    def state_dict(self, model_timesteps=0) -> dict:
        d = {
            'visited_packed': np.packbits(self.visited),
            'total_covered': np.int64(self.total_unique()),
            'testable_total': np.int64(self.testable_total or 0),
            'testable_fingerprint': np.str_(config.TESTABLE_FINGERPRINT or ''),
            'oob_events': np.int64(self.oob_events),
            'episode_new_history': np.asarray(self.episode_new_history,
                                              dtype=np.float64),
            'grid_geom': np.array([self.x0, self.y0, self.w, self.h],
                                  dtype=np.int64),
            'world_dims': np.array([config.LEVEL_W, config.LEVEL_H],
                                   dtype=np.int64),
            'format_version': np.int64(config.COVERAGE_FORMAT_VERSION),
            'model_timesteps': np.int64(model_timesteps),
            'config_hash': np.str_(_config_hash()),
            'saved_at': np.str_(datetime.datetime.now().isoformat()),
        }
        if self.visit_count is not None:
            d['visit_counts'] = self.visit_count
        return d

    def save(self, path, model_timesteps=0):
        """Writes atomically.

        A half-written coverage file sitting next to a good model zip is the
        worst failure mode available here - it would look valid and silently
        misreport how much of the world has been explored.
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp = path + '.tmp'
        np.savez_compressed(tmp, **self.state_dict(model_timesteps))
        os.replace(tmp + '.npz', path)

    def load_state_dict(self, d: dict) -> None:
        found = int(d['format_version'])
        if found != config.COVERAGE_FORMAT_VERSION:
            raise CoverageFormatMismatch(
                f"coverage format_version {found}, expected "
                f"{config.COVERAGE_FORMAT_VERSION}")
        geom = tuple(int(v) for v in d['grid_geom'])
        expected = (self.x0, self.y0, self.w, self.h)
        if geom != expected:
            raise CoverageFormatMismatch(
                f"coverage grid_geom {geom}, expected {expected}")

        # ─── THE DENOMINATOR MUST MATCH ───
        # A state recorded against a different testable mask describes a
        # different quantity. Loading one silently would reproduce exactly
        # the failure this version exists to fix: a percentage computed
        # against a denominator that does not describe the same world.
        saved_fp = str(d['testable_fingerprint']) if 'testable_fingerprint' in d else ''
        if saved_fp and config.TESTABLE_FINGERPRINT and                 saved_fp != config.TESTABLE_FINGERPRINT:
            raise CoverageFormatMismatch(
                f"coverage was recorded against testable mask "
                f"{saved_fp[:16]}... but the current mask is "
                f"{config.TESTABLE_FINGERPRINT[:16]}.... Its percentages "
                f"are not comparable. Rebuild the coverage or the mask.")

        n = self.w * self.h
        bits = np.unpackbits(d['visited_packed'])[:n]
        self.visited[:] = bits.reshape(self.h, self.w)
        if self.visit_count is not None and 'visit_counts' in d:
            self.visit_count[:] = d['visit_counts']
        self.oob_events = int(d['oob_events'])
        self.episode_new_history = [float(v) for v in d['episode_new_history']]
        self._remaining_cached = None

    def load(self, path, model=None, allow_mismatch=False):
        """Loads coverage, refusing anything that does not match the model.

        A coverage file paired with the wrong checkpoint silently corrupts
        every downstream number - the agent would be re-rewarded for space it
        already explored, or starved of reward for space it has not. Refusing
        is the only safe default.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        d = np.load(path, allow_pickle=False)
        self.load_state_dict(d)

        if model is not None:
            saved = int(d['model_timesteps'])
            actual = int(getattr(model, 'num_timesteps', saved))
            if saved != actual and not allow_mismatch:
                raise CoverageCheckpointMismatch(
                    f"coverage was saved at model_timesteps={saved:,} but the "
                    f"model is at num_timesteps={actual:,}.\n"
                    f"  coverage file : {path}\n"
                    f"  model         : {getattr(model, '_gh_path', '<in memory>')}\n"
                    f"Pass --allow-coverage-mismatch to override.")
        return self


# ── construction helpers ──────────────────────────────────────────────────
def load_testable():
    """The TESTABLE mask from exploration_data/, or None if not built yet.

    Returns None rather than raising so the dashboard and the reward wrapper
    keep working on a fresh clone where nobody has run
    tools/build_reachability.py. Without it the frontier and the % metrics go
    quiet; novelty and the drought pressure still work, because those only
    need the `visited` bitmap.
    """
    from . import reachability
    try:
        _solid, testable, _meta = reachability.load_masks()
    except (FileNotFoundError, reachability.ReachabilityMismatch):
        return None
    return testable


def open_shared(shm_names, testable_mask=None, create=False):
    """Attach (or, in the parent, create) a shared-memory SpatialCoverage."""
    return SpatialCoverage(testable_mask=testable_mask,
                           shm_names=shm_names, create_shared=create)


def create_shared(testable_mask=None, tag=None):
    """PARENT ONLY: allocate the shared blocks before SubprocVecEnv starts.

    Returns (coverage, shm_names). Hand `shm_names` to the workers - it is
    the only part that survives cloudpickling - and keep `coverage` in the
    parent, both as the authoritative reader for metrics and as the owner
    that unlink()s at shutdown.

    On Windows the blocks are refcounted by the OS and vanish when the last
    handle closes, so the parent MUST outlive every worker. It does: the
    parent holds these until the finally block in train_agent.py.
    """
    names = make_shm_names(tag)
    cov = SpatialCoverage(testable_mask=testable_mask, shm_names=names,
                          create_shared=True)
    return cov, names
