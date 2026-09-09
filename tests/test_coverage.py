"""SpatialCoverage: recording, persistence, frontier, adaptive target.

These run entirely on the coverage structure, with synthetic rects rather than
a live game, so they are fast and deterministic. The tests that drive the real
environment live in test_coverage_env.py.
"""

import numpy as np
import pytest

from exploration import config
from exploration.coverage import (CoverageCheckpointMismatch,
                                  CoverageFormatMismatch, SpatialCoverage)


def rect(x, y, w=30, h=40):
    return (x, y, w, h)


@pytest.fixture
def cov():
    """Fresh coverage with a permissive reachable mask."""
    mask = np.ones((config.GRID_H, config.GRID_W), dtype=bool)
    c = SpatialCoverage(testable_mask=mask)
    c.begin_episode()
    return c


# ── Test 1 ────────────────────────────────────────────────────────────────
def test_new_pixels_raise_coverage(cov):
    """Moving into virgin space increases coverage by exactly the swept area."""
    n1 = cov.record({'cur_rect': rect(100, 100)})
    assert n1 == 30 * 40, "first placement should claim the whole collider"
    assert cov.total_unique() == 30 * 40

    # Step 10px right: the sweep is the AABB union, so it adds a 10x40 strip.
    n2 = cov.record({'cur_rect': rect(110, 100)})
    assert n2 == 10 * 40
    assert cov.total_unique() == 40 * 40


def test_sweep_is_contiguous_no_gaps(cov):
    """Max displacement is smaller than the collider, so no pixel is skipped."""
    cov.record({'cur_rect': rect(200, 200)})
    # Worst-case legal single-frame move.
    cov.record({'cur_rect': rect(200 + config.MAX_FRAME_DX,
                                 200 + config.MAX_FRAME_DY)})
    region = cov.visited[200 - cov.y0:200 + 40 + config.MAX_FRAME_DY - cov.y0,
                         200 - cov.x0:200 + 30 + config.MAX_FRAME_DX - cov.x0]
    # The union of two overlapping rects is their bounding box, fully filled.
    assert region.all(), "swept AABB left holes - the sweep is not contiguous"


# ── Test 2 ────────────────────────────────────────────────────────────────
def test_revisits_do_not_inflate_unique_count(cov):
    """Replaying an identical trajectory must not change unique coverage."""
    path = [rect(300 + i * 5, 300) for i in range(10)]
    for r in path:
        cov.record({'cur_rect': r})
    after_first = cov.total_unique()

    cov.begin_episode()
    for r in path:
        cov.record({'cur_rect': r})
    assert cov.total_unique() == after_first


# ── Test 3 (novelty component) ────────────────────────────────────────────
def test_revisits_pay_no_novelty_reward(cov):
    """A second pass over the same ground pays exactly zero - not negative."""
    n_new = cov.record({'cur_rect': rect(400, 400)})
    assert cov.novelty_reward(n_new) > 0

    cov.begin_episode()
    n_again = cov.record({'cur_rect': rect(400, 400)})
    assert n_again == 0
    assert cov.novelty_reward(n_again) == 0.0, "traversal must be free"


def test_novelty_is_bounded_and_monotone(cov):
    """Bounded above, and never decreasing in new-pixel count."""
    import itertools
    vals = [cov.novelty_reward(n) for n in (1, 100, 560, 2288, 100000)]
    assert all(b >= a for a, b in itertools.pairwise(vals)), "not monotone"
    ceiling = config.NOVELTY_WEIGHT * config.NOVELTY_WEIGHT_MULT * config.NOVELTY_CAP
    assert max(vals) <= ceiling + 1e-9


# ── Test 5 ────────────────────────────────────────────────────────────────
def test_reset_preserves_coverage(cov):
    """Episode boundaries must not clear the persistent bitmap."""
    cov.record({'cur_rect': rect(500, 300)})
    before = cov.total_unique()
    cov.begin_episode()
    assert cov.total_unique() == before
    assert cov.episode_new == 0, "per-episode counter should reset"


def test_begin_episode_clears_prev_rect(cov):
    """The spawn teleport must never be swept.

    Without this the first sweep of a new episode spans from wherever Mario
    died to the spawn point, painting a false corridor across the level.
    """
    cov.record({'cur_rect': rect(8000, 400)})
    cov.begin_episode()
    assert cov.prev_rect is None
    n = cov.record({'cur_rect': rect(100, 400)})
    assert n == 30 * 40, "spawn should claim only the collider, not a corridor"


# ── Test 7 ────────────────────────────────────────────────────────────────
def test_save_load_round_trip(tmp_path, cov):
    """Exact round-trip of the bitmap and the authoritative total."""
    for i in range(20):
        cov.record({'cur_rect': rect(600 + i * 7, 250)})
    total = cov.total_unique()
    raw = int(cov.visited.sum())
    path = str(tmp_path / "coverage_test.npz")
    cov.save(path, model_timesteps=6_000_000)

    mask = np.ones((config.GRID_H, config.GRID_W), dtype=bool)
    fresh = SpatialCoverage(testable_mask=mask)
    fresh.load(path)
    assert fresh.total_unique() == total
    assert int(fresh.visited.sum()) == raw


def test_save_is_atomic_no_tmp_left(tmp_path, cov):
    cov.record({'cur_rect': rect(700, 250)})
    path = str(tmp_path / "cov.npz")
    cov.save(path, model_timesteps=1)
    assert not (tmp_path / "cov.npz.tmp.npz").exists()
    assert not (tmp_path / "cov.npz.tmp").exists()


# ── Test 14 ───────────────────────────────────────────────────────────────
def test_checkpoint_mismatch_is_detected(tmp_path, cov):
    """Pairing coverage with the wrong model must raise, naming both numbers."""
    path = str(tmp_path / "coverage_5600000.npz")
    cov.record({'cur_rect': rect(800, 250)})
    cov.save(path, model_timesteps=5_600_000)

    class FakeModel:
        num_timesteps = 6_000_000

    fresh = SpatialCoverage(
        testable_mask=np.ones((config.GRID_H, config.GRID_W), dtype=bool))
    with pytest.raises(CoverageCheckpointMismatch) as exc:
        fresh.load(path, model=FakeModel())
    msg = str(exc.value)
    assert "5,600,000" in msg and "6,000,000" in msg


def test_format_mismatch_is_detected(tmp_path, cov):
    path = str(tmp_path / "cov.npz")
    cov.save(path, model_timesteps=1)
    d = dict(np.load(path, allow_pickle=False))
    d['format_version'] = np.int64(config.COVERAGE_FORMAT_VERSION + 99)
    np.savez_compressed(str(tmp_path / "bad.npz"), **d)

    fresh = SpatialCoverage(
        testable_mask=np.ones((config.GRID_H, config.GRID_W), dtype=bool))
    with pytest.raises(CoverageFormatMismatch):
        fresh.load(str(tmp_path / "bad.npz"))


# ── out-of-grid handling ──────────────────────────────────────────────────
def test_out_of_grid_is_counted_not_clamped(cov):
    """Positions outside the padded grid are a glitch signal, not coverage.

    Clamping them inward would fabricate coverage at the boundary.
    """
    before = cov.total_unique()
    cov.record({'cur_rect': rect(config.GRID_X0 - 5000, 0)})
    assert cov.oob_events == 1
    assert cov.total_unique() == before, "clamped position fabricated coverage"


# ── frontier ──────────────────────────────────────────────────────────────
def test_frontier_excludes_cells_behind_the_camera():
    """The camera is one-way, so frontier behind viewport.x is unreachable.

    update_viewport() only ever increases viewport.x, and Mario is hard-clamped
    to viewport.x + 5. Offering a frontier target behind that is offering
    something physically impossible this episode.
    """
    mask = np.zeros((config.GRID_H, config.GRID_W), dtype=bool)
    # Two unexplored patches: one far left, one far right.
    mask[500:540, 300:340] = True      # world x ~ 44
    mask[500:540, 6000:6040] = True    # world x ~ 5744
    c = SpatialCoverage(testable_mask=mask)

    c.refresh_frontier(viewport_x=0, force=True)
    xs_all = c._frontier_cells[:, 0]
    assert xs_all.min() < 1000 and xs_all.max() > 5000

    c.refresh_frontier(viewport_x=5000, force=True)
    xs_ahead = c._frontier_cells[:, 0]
    assert xs_ahead.size > 0
    assert (xs_ahead + config.FRONTIER_CELL >= 5000).all(), \
        "frontier behind the one-way camera was offered as a target"


def test_phi_is_zero_when_no_frontier_remains():
    mask = np.zeros((config.GRID_H, config.GRID_W), dtype=bool)
    c = SpatialCoverage(testable_mask=mask)
    c.refresh_frontier(viewport_x=0, force=True)
    assert c.phi(100, 100) == 0.0


def test_frontier_potential_residue_is_negligible():
    """Potential-based shaping must not be worth farming.

    Note what this does NOT assert. PBRS is policy-invariant under the
    DISCOUNTED objective; the undiscounted sum around a closed loop is
    (gamma - 1) * sum(Phi), which with gamma < 1 and Phi <= 0 is a small
    POSITIVE residue. An earlier version of this test asserted that sum was
    <= 0, which is simply the wrong invariant.

    What matters in practice is that the residue is dominated by the penalty
    for not exploring, so pacing is never a better strategy than exploring.
    """
    mask = np.zeros((config.GRID_H, config.GRID_W), dtype=bool)
    mask[500:540, 6000:6040] = True
    c = SpatialCoverage(testable_mask=mask)
    c.refresh_frontier(viewport_x=0, force=True)

    a, b = (1000, 400), (1400, 400)
    residue = ((config.GAMMA * c.phi(*b) - c.phi(*a))
               + (config.GAMMA * c.phi(*a) - c.phi(*b)))

    # Bounded by the theoretical worst case: both endpoints at max distance.
    worst = abs((config.GAMMA - 1.0) * -2.0)
    assert 0 <= residue <= worst + 1e-9

    # Compare per SUBSTEP, which is the quantity that actually competes: the
    # drought ceiling is a per-substep rate, so a per-cycle residue is not
    # comparable to it without dividing by the cycle length.
    per_substep = config.FRONTIER_WEIGHT * (1.0 - config.GAMMA) * 1.0
    assert per_substep < config.DROUGHT_MAX / 5, (
        f"pacing pays up to {per_substep:.5f}/substep against a "
        f"{config.DROUGHT_MAX}/substep drought ceiling - too close to farmable")


# ── adaptive target ───────────────────────────────────────────────────────
def test_target_falls_back_to_floor_without_history(cov):
    assert cov.episode_target() == config.TARGET_FLOOR


def test_target_tracks_recent_median(cov):
    """The target follows the agent's own decay curve rather than a fixed goal.

    Measured decay for a run-right policy was
    72352 -> 10955 -> 16576 -> 11972 -> 1787 -> 48, so any fixed target becomes
    impossible within ~5 episodes.
    """
    cov.episode_new_history = [72352, 10955, 16576, 11972, 1787]
    t = cov.episode_target()
    assert t == int(np.median(cov.episode_new_history))

    cov.episode_new_history = [1787, 48, 60, 55, 40]
    assert cov.episode_target() == config.TARGET_FLOOR


def test_target_never_exceeds_what_remains():
    """The target is clamped by remaining space, so it cannot demand the
    impossible near the end of a campaign."""
    mask = np.zeros((config.GRID_H, config.GRID_W), dtype=bool)
    mask[0:100, 0:100] = True          # only 10,000 reachable px
    c = SpatialCoverage(testable_mask=mask)
    c.episode_new_history = [500000] * 6
    assert c.episode_target() <= max(
        config.TARGET_FLOOR, int(config.TARGET_REMAIN_FRAC * c.remaining()))
