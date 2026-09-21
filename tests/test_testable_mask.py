"""The coverage denominator: it must describe the world, not exceed it.

These exist because a denominator of 7,606,986 shipped once. It was larger
than the entire 9087x600 world (5,452,200) because it counted 2,907,840 px of
sky above the world and applied no jump-height limit at all. Every coverage
percentage computed against it was meaningless, and nothing in the suite at
the time could tell.

The failure mode is what makes these tests worth having: a wrong denominator
does not raise, does not look wrong in a log, and produces percentages that
are perfectly plausible right up until someone checks them against the size
of the level.
"""

import numpy as np
import pytest

from exploration import config, reachability
from exploration.coverage import SpatialCoverage

RETIRED_DENOMINATOR = 7_606_986
PADDED_GRID = 9600 * 1024


@pytest.fixture(scope="module")
def masks():
    """The built mask bundle. Git-ignored (tools/build_reachability.py makes
    it), so a fresh clone or CI skips these; the slow clean-rebuild test below
    still checks the denominator from the live geometry there."""
    try:
        solid, testable, meta = reachability.load_masks()
    except FileNotFoundError:
        pytest.skip(f"{config.REACHABLE_MASK_PATH} is not in this checkout")
    return solid, testable, meta


# ── Test P: the denominator is testable space, not a raster ───────────────
def test_denominator_is_not_a_raster(masks):
    _solid, testable, _meta = masks
    total = int(testable.sum())
    assert total == config.TESTABLE_TOTAL
    assert total != config.WORLD_RASTER_PX, "world raster used as denominator"
    assert total != PADDED_GRID, "padded grid used as denominator"
    assert total < config.WORLD_RASTER_PX, (
        f"testable {total:,} >= world raster {config.WORLD_RASTER_PX:,} - a "
        f"denominator cannot be as large as the world it describes")


def test_the_retired_denominator_is_gone():
    """7,606,986 must never come back, in any form."""
    assert config.TESTABLE_TOTAL != RETIRED_DENOMINATOR
    assert not hasattr(config, 'EXPECTED_COVERABLE_PX'), (
        "EXPECTED_COVERABLE_PX held the impossible denominator and was "
        "retired; something has reintroduced it")


def test_testable_mask_excludes_sky_pit_and_solids(masks):
    """The four exclusions the definition requires."""
    solid, testable, _meta = masks
    ys = np.arange(config.GRID_H) + config.GRID_Y0
    xs = np.arange(config.GRID_W) + config.GRID_X0

    assert not (testable & (ys < 0)[:, None]).any(), "sky counted as testable"
    assert not (testable & (ys >= config.LEVEL_H)[:, None]).any(), \
        "below the death plane counted as testable"
    assert not (testable & ((xs < 0) | (xs >= config.LEVEL_W))[None, :]).any(), \
        "outside the level horizontally counted as testable"
    assert not (testable & solid).any(), "solid interior counted as testable"


# ── Test Q: the arithmetic is exact ───────────────────────────────────────
def test_coverage_arithmetic_is_exact(masks):
    _solid, testable, _meta = masks
    cov = SpatialCoverage(testable_mask=testable)
    cov.begin_episode()
    for i in range(60):
        cov.record({'cur_rect': (400 + i * 11, 480, 30, 40)})

    covered = cov.covered_testable()
    assert covered == int(np.count_nonzero(cov.visited & testable))
    assert covered + cov.remaining() == config.TESTABLE_TOTAL
    assert 0.0 <= cov.coverage_pct() <= 100.0
    assert cov.coverage_pct() == pytest.approx(
        100.0 * covered / config.TESTABLE_TOTAL)
    cov.assert_consistent()


def test_noncoverage_pixels_are_never_coverage(masks):
    """Item 7: reaching outside the mask is never progress."""
    _solid, testable, _meta = masks
    cov = SpatialCoverage(testable_mask=testable)
    cov.begin_episode()

    before_cov, before_non = cov.covered_testable(), cov.noncoverage_px()
    # Sky: well above the world, inside the padded grid so it is recorded.
    cov.record({'cur_rect': (3000, -200, 30, 40)})
    assert cov.covered_testable() == before_cov, \
        "a pixel in the sky increased QA coverage"
    assert cov.noncoverage_px() > before_non, \
        "an out-of-world pixel was not recorded as noncoverage"

    b = cov.noncoverage_breakdown()
    assert b['total'] == cov.noncoverage_px()
    assert (b['expected_total'] + b['model_gap_total']
            + b['anomalous_total']) == b['total'], \
        "the three noncoverage groups do not partition the total"
    cov.assert_consistent()


# ── Hardening: normal behaviour must not be reported as a glitch ──────────
def test_normal_engine_behaviour_is_not_called_a_glitch(masks):
    """A jump arc, a pit death and a tolerated overlap are NOT bugs.

    This is the regression guard for the accounting that previously reported
    all 9,687 bootstrap out-of-mask pixels as anomalies, when every one of
    them was documented, normal behaviour of this engine.
    """
    from exploration import reachability as reach
    _solid, testable, _meta = masks
    cov = SpatialCoverage(testable_mask=testable)
    cov.begin_episode()
    cov.record({'cur_rect': (3000, -20, 30, 40)})     # ordinary jump arc
    b = cov.noncoverage_breakdown()

    assert b['expected_total'] > 0, "a normal jump arc produced no expected px"
    assert b['anomalous_total'] == 0, (
        f"a normal jump arc was reported as {b['anomalous_total']} anomalous "
        f"px - this is exactly the false glitch signal the taxonomy exists "
        f"to prevent")
    assert b['expected']['jump_arc'] > 0
    # And the class map must agree with the mask about what is testable.
    cm = reach.load_class_map()
    assert not (testable & (cm != reach.CLS_TESTABLE)).any()
    assert not (~testable & (cm == reach.CLS_TESTABLE)).any()


def test_genuinely_impossible_pixels_are_still_anomalous(masks):
    """The taxonomy must not have made the detector unable to fire."""
    _solid, testable, _meta = masks
    cov = SpatialCoverage(testable_mask=testable)
    cov.begin_episode()
    # 300 px above the world: far higher than any jump envelope allows.
    cov.record({'cur_rect': (3000, -300, 30, 40)})
    b = cov.noncoverage_breakdown()
    assert b['anomalous_total'] > 0, \
        "an impossible altitude was not flagged as anomalous"
    assert b['anomalous']['impossible_sky'] > 0
    assert cov.anomalous_px() == b['anomalous_total']


def test_novelty_is_not_paid_for_anomalous_pixels(masks):
    """A clipping glitch must not be farmable for exploration reward."""
    _solid, testable, _meta = masks
    cov = SpatialCoverage(testable_mask=testable)
    cov.begin_episode()
    n_new = cov.record({'cur_rect': (3000, -200, 30, 40)})
    assert n_new == 0, "novelty was paid for pixels outside the world"
    assert cov.novelty_reward(n_new) == 0.0


def test_consistency_assertion_catches_an_impossible_state(masks):
    _solid, testable, _meta = masks
    cov = SpatialCoverage(testable_mask=testable)
    cov.testable_total = 10          # deliberately impossible
    cov.visited[:] = 1
    cov.invalidate_remaining()
    with pytest.raises(AssertionError):
        cov.assert_consistent()


def test_enemies_never_move_the_static_denominator(env):
    """The denominator is built from the level's STATIC colliders only.
    Enemies spawn, walk, die and leave shells; none of it may add or remove a
    single solid pixel, or the same world would have a different 100%."""
    import pygame as pg
    env.reset()
    state = env.game.state
    solid0, n0 = reachability.rasterize_solids(state)
    for _ in range(400):                          # run right until enemies are out
        if env.step(3)[2]:
            break
    assert len(state.enemy_group) > 0, "no enemy spawned - the test proves nothing"
    solid1, n1 = reachability.rasterize_solids(state)

    ghost = pg.sprite.Sprite()                    # an enemy parked in open air
    ghost.rect = pg.Rect(1000, 300, 40, 40)
    for name in ('enemy_group', 'shell_group', 'sprites_about_to_die_group'):
        getattr(state, name).add(ghost)
    try:
        solid2, n2 = reachability.rasterize_solids(state)
    finally:
        for name in ('enemy_group', 'shell_group', 'sprites_about_to_die_group'):
            getattr(state, name).remove(ghost)
    assert n0 == n1 == n2 == config.EXPECTED_SOLID_RECTS
    assert np.array_equal(solid0, solid1) and np.array_equal(solid0, solid2)
    assert int(solid0.sum()) == config.EXPECTED_SOLID_PX


# ── Test R: reproducible from a clean rebuild ─────────────────────────────
@pytest.mark.slow
def test_methods_reproduce_and_reconcile(env):
    """Rebuild from live geometry: same total, and B/C agree within 5%.

    Uses the session env fixture rather than constructing a second
    CustomMarioEnv - the Pygame window is created once per process.
    """
    env.reset()
    mario = env.game.state.mario
    # The recorded flag corridor is an INPUT to the denominator - it is
    # measured in the engine, not derived from geometry - so a reproducibility
    # check feeds back the one stored in the bundle. Whether the replay itself
    # reproduces that corridor is a separate claim, and the builder's own
    # acceptance check (every past-trigger px real play covered must be inside
    # it) is what establishes it.
    n = config.GRID_H * config.GRID_W
    with np.load(config.REACHABLE_MASK_PATH, allow_pickle=False) as d:
        corridor_g = np.unpackbits(d['flag_corridor_packed'])[:n].reshape(
            config.GRID_H, config.GRID_W).astype(bool)
    wy0, wx0 = -config.GRID_Y0, -config.GRID_X0
    corridor = corridor_g[wy0:wy0 + config.LEVEL_H, wx0:wx0 + config.LEVEL_W]
    # The real-arc envelope is an INPUT too: measured jump arcs, plus the real
    # play those arcs deny. A rebuild without them gives the retired
    # rectangle-envelope denominator.
    import os
    if not os.path.exists(config.JUMP_ARCS_PATH):
        pytest.skip(f"{config.JUMP_ARCS_PATH} is not in this checkout")
    _solid, testable, _cls, stats = reachability.build_testable(
        env.game.state, (mario.rect.x, mario.rect.y), flag_corridor=corridor,
        arcs=reachability.load_jump_arcs(), observed=reachability.load_observed_reach())

    assert stats['testable_total'] == config.TESTABLE_TOTAL, (
        "a clean rebuild produced a different denominator - it is not "
        "reproducible")
    assert reachability.mask_fingerprint(
        reachability._embed_in_grid(testable)) == config.TESTABLE_FINGERPRINT

    # Required ordering, and the agreement that justifies adopting C.
    assert stats['method_c_px'] <= stats['method_b_px'] <= stats['method_a_px']
    assert stats['bc_delta_pct'] <= 5.0, (
        f"Methods B and C differ by {stats['bc_delta_pct']:.2f}% - too far "
        f"apart to adopt either")
    assert stats['adopted_method'] == 'C'
    # The flag-trigger and real-arc corrections ran, and the arithmetic closes exactly.
    assert stats['flag_trigger'] is not None and stats['flag_removed_px'] > 0
    assert stats['arcs_removed_px'] > 0
    assert (stats['method_c_trimmed_px'] - stats['flag_removed_px']
            - stats['arcs_removed_px'] == stats['testable_total'])
    assert stats['solid_px'] == config.EXPECTED_SOLID_PX
    assert stats['n_solid_rects'] == config.EXPECTED_SOLID_RECTS
    for key in ('method_a_px', 'method_b_px', 'method_c_px'):
        assert stats[key] <= config.WORLD_RASTER_PX

    # At the adopted 1 px lattice there is no discretisation left: Method C
    # is a strict subset of Method B, so the trim removes nothing.
    assert stats['lattice'] == 1
    assert stats['c_lattice_overshoot_px'] == 0, (
        f"Method C exceeded Method B by {stats['c_lattice_overshoot_px']} px "
        f"at a 1 px lattice, where it should be a subset by construction")


@pytest.mark.slow
def test_lattice_does_not_change_the_denominator(masks):
    """Resolution convergence: the answer is the level's, not the lattice's.

    Runs the SAME connectivity method at 4 px and 2 px against the stored
    geometry and checks both land on the adopted total after the Method-B
    trim. The raw overshoot must shrink with the pitch - that is what
    distinguishes a discretisation artifact from a modelling error.
    """
    solid_g, _testable, _meta = masks
    wy0, wx0 = -config.GRID_Y0, -config.GRID_X0
    solid = solid_g[wy0:wy0 + config.LEVEL_H, wx0:wx0 + config.LEVEL_W]
    # Unioned over BOTH of Mario's forms, exactly as build_testable does -
    # the stored method_c_px is that union, and a small-only rebuild here
    # would be comparing two different quantities. The spawn ANCHOR is the
    # collider's top-left, so a taller form stands at a higher anchor.
    forms = ((config.MARIO_SMALL_W, config.MARIO_SMALL_H),
             (config.MARIO_BIG_W, config.MARIO_BIG_H))
    base_h = forms[0][1]
    b_px = np.logical_or.reduce(
        [reachability.method_b(solid, mw, mh)[0] for mw, mh in forms])

    trimmed, overshoot = {}, {}
    for pitch in (4, 2):
        raw = np.logical_or.reduce(
            [reachability.method_c(solid, (110, 498 + base_h - mh), mw, mh,
                                   coarse=pitch)[0] for mw, mh in forms])
        trimmed[pitch] = int((raw & b_px).sum())
        overshoot[pitch] = int((raw & ~b_px).sum())

    assert overshoot[2] < overshoot[4], (
        "the Method C overshoot did not shrink with a finer lattice, so it is "
        "not a discretisation artifact")
    # Against METHOD C's own total, not TESTABLE_TOTAL: the flag-trigger
    # correction is applied AFTER Method C and removes a fixed set of pixels
    # that has nothing to do with the lattice, so including it here would
    # compare two different quantities.
    method_c = int(_meta['method_c_px'])
    for pitch, got in trimmed.items():
        assert abs(got - method_c) <= 200, (
            f"a {pitch} px lattice gives {got:,} against Method C's "
            f"{method_c:,} - the denominator depends on the lattice, not "
            f"just the level")


# ── Item 12: a stale denominator must not load silently ───────────────────
def test_mask_file_carries_a_fingerprint(masks):
    _solid, _testable, meta = masks
    assert meta['testable_fingerprint'] == config.TESTABLE_FINGERPRINT
    assert meta['adopted_method'] == 'C'
    assert meta['testable_total'] == config.TESTABLE_TOTAL
    assert meta['method_b_px'] >= meta['method_c_px']
    assert meta['lattice'] == config.REACHABILITY_LATTICE


def test_mask_file_records_the_arc_envelope_and_keeps_observed_play(masks):
    _solid, testable, meta = masks
    assert meta['arcs_removed_px'] > 0, "the real-arc envelope removed nothing"
    assert meta['observed_px'] > 0
    observed = reachability.load_observed_reach()
    if observed is None:
        pytest.skip("observed_reach.npz is not in this checkout")
    # Real play the arcs deny is never dropped: coverage may not go down.
    assert int(observed.sum()) == meta['observed_px']
    assert (reachability._embed_in_grid(observed) & ~testable).sum() == 0


def test_coverage_state_from_a_different_denominator_is_refused(tmp_path, masks):
    """The guard that makes the version bump meaningful."""
    from exploration.coverage import CoverageFormatMismatch
    _solid, testable, _meta = masks
    cov = SpatialCoverage(testable_mask=testable)
    cov.begin_episode()
    cov.record({'cur_rect': (500, 480, 30, 40)})
    path = str(tmp_path / "cov.npz")
    cov.save(path, model_timesteps=6_000_000)

    d = dict(np.load(path, allow_pickle=False))
    d['testable_fingerprint'] = np.str_("deadbeef" * 8)
    np.savez_compressed(str(tmp_path / "stale.npz"), **d)

    fresh = SpatialCoverage(testable_mask=testable)
    with pytest.raises(CoverageFormatMismatch, match="testable mask"):
        fresh.load(str(tmp_path / "stale.npz"))


def test_bootstrap_rescored_against_the_adopted_mask(masks):
    """The shipped bootstrap must be consistent with the current denominator."""
    import os
    if not os.path.exists(config.BOOTSTRAP_COVERAGE):
        pytest.skip("bootstrap not generated")
    _solid, testable, _meta = masks
    cov = SpatialCoverage(testable_mask=testable)
    cov.load(config.BOOTSTRAP_COVERAGE)
    cov.assert_consistent()
    assert cov.covered_testable() <= config.TESTABLE_TOTAL
    assert 0.0 < cov.coverage_pct() < 100.0


# ── Test: the tracked docs quote the denominator that is actually in force ──
def test_the_docs_quote_the_current_denominator():
    """README.md and docs/ARCHITECTURE.md both print the denominator as a
    literal. Three mask generations shipped (4,013,723 -> 4,002,095 ->
    3,757,990) and the docs kept the first one through all of them: a reader
    checking a coverage percentage against the README would have been out by
    6.8% with nothing in the suite to say so.

    Prose about a *superseded* number is fine and expected - the worklog
    explains each change - so this only asks that the current denominator
    appears, and that the retired ones are never presented as current.
    """
    import os
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    current = f"{config.TESTABLE_TOTAL:,}"
    retired = ("4,013,723", "4,002,095", "3,866,663", f"{RETIRED_DENOMINATOR:,}")
    for rel in ("README.md", os.path.join("docs", "ARCHITECTURE.md")):
        with open(os.path.join(root, rel), encoding="utf-8") as fh:
            text = fh.read()
        assert current in text, (
            f"{rel} never mentions the live denominator {current}; it was "
            f"changed in exploration/config.py without updating the docs")
        for old in retired:
            for line in text.splitlines():
                if old not in line:
                    continue
                assert re.search(r"superseded|retired|was\b|previous|earlier|before|old",
                                 line, re.IGNORECASE), (
                    f"{rel} presents the retired denominator {old} as current:\n  {line.strip()}")
