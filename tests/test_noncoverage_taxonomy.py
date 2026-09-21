"""The two false-positive causes the 6.02M short validation exposed.

That run reported 599 anomalous pixels, all `deep_penetration`, where the
40-episode bootstrap had reported 0. Neither cluster was a game bug:

  579 px  x 5070-5094, y 371-401   the deep interior of brick20, a 43x43
                                   Brick at (5058, 365) that big Mario
                                   destroys outright (level1.py:776)
   20 px  x 7748-7751, y 458-462   pipe6's top-right corner - geometry that
                                   never moves, reached only by the BOUNDING
                                   BOX record() marks between two frames

The tempting fix was to raise PENETRATION_TOL until both went quiet. These
tests exist because that would have been the wrong one: it explains neither
cluster, and it blunts the detector everywhere. Each cause gets its own
class, each class is proven here against the real engine, and the detector
is proven to still fire on things that are genuinely impossible.
"""

import numpy as np
import pytest

from exploration import config
from exploration import reachability as reach
from exploration.coverage import SpatialCoverage

BRICK20_CORE = (5070, 5094, 371, 401)      # x0, x1, y0, y1  (world, inclusive)
PIPE6_CORNER = (7748, 7751, 458, 462)


@pytest.fixture(scope="module")
def level_state():
    env = pytest.importorskip("custom_mario_env").CustomMarioEnv()
    env.reset()
    yield env.game.state
    env.close()


@pytest.fixture(scope="module")
def class_map():
    try:
        return reach.load_class_map()
    except (FileNotFoundError, reach.ReachabilityMismatch):
        pytest.skip(f"{config.REACHABLE_MASK_PATH} is not in this checkout")


def _classes_at(cm, box):
    x0, x1, y0, y1 = box
    sub = cm[y0 - config.GRID_Y0:y1 - config.GRID_Y0 + 1,
             x0 - config.GRID_X0:x1 - config.GRID_X0 + 1]
    return {reach.CLASS_NAMES[int(v)] for v in np.unique(sub)}


# ── CAUSE 1: the level's geometry is not constant ────────────────────────
def test_the_engine_really_does_destroy_brick20(level_state):
    """The premise of CLS_MUTABLE_SOLID, driven through the engine's own path.

    If this ever fails, the mutable-solid class is excusing pixels for a
    reason that no longer exists and should be removed, not kept.
    """
    brick = next((b for b in level_state.brick_group
                  if b.rect.x == 5058 and b.rect.y == 365), None)
    assert brick is not None, "brick20 is not where the taxonomy says it is"
    assert brick.contents is None, \
        "brick20 gained contents - only contents-less bricks can be killed"

    before, _ = reach.rasterize_solids(level_state)
    x0, x1, y0, y1 = BRICK20_CORE
    assert before[y0:y1 + 1, x0:x1 + 1].all(), \
        "the cluster is not solid in the opening geometry"

    # level1.adjust_mario_for_y_brick_collisions: big Mario, rising, from below
    level_state.mario.big = True
    level_state.mario.rect.width = config.MARIO_BIG_W
    level_state.mario.rect.height = config.MARIO_BIG_H
    level_state.mario.rect.x = brick.rect.centerx - config.MARIO_BIG_W // 2
    level_state.mario.rect.y = brick.rect.bottom + 1
    level_state.mario.y_vel = -7
    level_state.adjust_mario_for_y_brick_collisions(brick)

    assert not brick.alive(), "the engine did not remove the brick"
    after, _ = reach.rasterize_solids(level_state)
    assert not after[y0:y1 + 1, x0:x1 + 1].any(), \
        "the brick died but its footprint is still solid"
    assert int((before & ~after).sum()) == brick.rect.w * brick.rect.h


def test_only_contents_less_bricks_count_as_destructible(level_state):
    """A brick holding coins or a star is never killed, so it is not mutable."""
    mutable = reach.mutable_solid_mask(level_state)
    kept = [b for b in level_state.brick_group if b.contents is not None]
    assert kept, "the level has no contents-bearing bricks to check"
    for b in kept:
        # Its upper body must NOT be excused; only the bump band at its foot.
        top_band = mutable[b.rect.top:b.rect.bottom - config.BUMP_RISE_PX,
                           b.rect.left:b.rect.right]
        assert not top_band.any(), (
            f"the brick at {b.rect.topleft} holds {b.contents!r} and can "
            f"never be destroyed, but its interior was excused anyway")


def test_the_bump_band_matches_the_engines_own_physics():
    """BUMP_RISE_PX is integrated from Brick.bumped(), not picked."""
    y, vel, peak = 0.0, -6.0, 0.0
    for _ in range(100):
        y += vel
        vel += 1.2
        peak = min(peak, y)
        if y >= 5:
            break
    assert int(abs(np.floor(peak))) == config.BUMP_RISE_PX, (
        f"BUMP_RISE_PX is {config.BUMP_RISE_PX} but Brick.bumped() rises "
        f"{abs(np.floor(peak))} px")


# ── CAUSE 2: the recorded evidence is a box, not a path ──────────────────
def test_record_marks_the_bounding_box_not_the_path():
    """The premise of CLS_SWEEP_ARTIFACT, pinned on the recorder itself."""
    cov = SpatialCoverage()
    cov.begin_episode()
    cov.record({'cur_rect': (1000, 100, 10, 10)})
    cov.record({'cur_rect': (1010, 110, 10, 10)})
    g_y, g_x = -config.GRID_Y0, -config.GRID_X0
    # The corner at (1000..1009, 110..119) is in NEITHER rect, but a bounding
    # box spans it - which is exactly how a pixel nothing touched is recorded.
    assert cov.visited[g_y + 110, g_x + 1000], (
        "record() no longer marks the bounding box - CLS_SWEEP_ARTIFACT's "
        "whole premise has changed and the class needs revisiting")


def test_pipe6_corner_is_reachable_by_a_legal_sweep_but_not_by_a_collider():
    """Two free placements one frame apart cover the cluster; no rect does.

    prev=(7747, 412) -> cur=(7758, 423) is dx=+11 dy=+11, inside the engine's
    14/12 envelope, and both placements are entirely collision-free.
    """
    import custom_mario_env
    env = custom_mario_env.CustomMarioEnv()
    env.reset()
    try:
        solid, _ = reach.rasterize_solids(env.game.state)
    finally:
        env.close()

    w, h = config.MARIO_SMALL_W, config.MARIO_SMALL_H
    prev, cur = (7747, 412), (7758, 423)
    for x, y in (prev, cur):
        assert not solid[y:y + h, x:x + w].any(), \
            f"placement {(x, y)} is not collision-free"
    assert abs(cur[0] - prev[0]) <= config.MAX_FRAME_DX
    assert abs(cur[1] - prev[1]) <= config.MAX_FRAME_DY

    x0, x1, y0, y1 = PIPE6_CORNER
    bx0, by0 = min(prev[0], cur[0]), min(prev[1], cur[1])
    bx1, by1 = max(prev[0], cur[0]) + w, max(prev[1], cur[1]) + h
    assert bx0 <= x0 and bx1 > x1 and by0 <= y0 and by1 > y1, \
        "the union box no longer covers the observed cluster"
    # ... and the collider itself never enters the pipe on either frame.
    for x, y in (prev, cur):
        assert not (x <= x1 and x + w > x0 and y <= y1 and y + h > y0), \
            f"placement {(x, y)} overlaps the cluster - this would be a real clip"


def test_sweep_coverable_matches_a_brute_force_enumeration():
    """The vectorised build against the definition, pair by pair.

    sweep_coverable() skips half the displacement envelope on the argument
    that d and -d describe the same pairs from the other end. That identity
    is worth a fraction of a second to check rather than to trust: the brute
    force here enumerates every ordered pair, both signs included.
    """
    rng = np.random.default_rng(4)
    solid = np.zeros((40, 60), bool)
    solid[20:32, 10:26] = True
    solid[8:14, 40:55] = True
    solid[rng.integers(0, 40, 12), rng.integers(0, 60, 12)] = True
    w, h, mdx, mdy = 6, 8, 3, 2

    fast = reach.sweep_coverable(solid, forms=[(w, h)], max_dx=mdx, max_dy=mdy)

    def is_free(x, y):
        return (x >= 0 and y >= 0 and x + w <= solid.shape[1]
                and y + h <= solid.shape[0] and not solid[y:y + h, x:x + w].any())

    brute = np.zeros_like(solid)
    for ay in range(solid.shape[0]):
        for ax in range(solid.shape[1]):
            if not is_free(ax, ay):
                continue
            for dx in range(-mdx, mdx + 1):
                for dy in range(-mdy, mdy + 1):
                    bx, by = ax + dx, ay + dy
                    if not is_free(bx, by):
                        continue
                    brute[min(ay, by):max(ay, by) + h,
                          min(ax, bx):max(ax, bx) + w] = True

    assert np.array_equal(fast, brute), (
        f"{int((fast ^ brute).sum())} pixels differ from the brute-force "
        f"enumeration - the half-envelope symmetry does not hold")


def test_sweep_coverable_never_reaches_past_the_frame_envelope():
    """A solid deeper than one frame's travel must stay unreachable."""
    solid = np.zeros((200, 200), bool)
    solid[50:150, 50:150] = True            # a 100x100 block
    sweep = reach.sweep_coverable(solid, forms=[(10, 10)], max_dx=4, max_dy=4)
    # The middle of the block is far past (collider + displacement) from any
    # free anchor, so no legal box can contain it.
    assert not sweep[95:105, 95:105].any(), \
        "a sweep box reached the core of a block it cannot get near"
    assert sweep[40:50, 40:50].all(), "free space next to the block is coverable"


# ── CAUSE 3: "bottomless" is a question about the FLOOR ──────────────────
PIT_UNDER_BRICKS = (3708, 3744, 600, 646)     # x0, x1, y0, y1


def test_a_pit_with_bricks_far_above_it_is_still_a_pit(level_state):
    """The premise of the widened pit rule, measured on the real level.

    x 3683-3773 has brick6..brick13 floating at y 193-235 and NO ground at
    all beneath them. `~solid.any(axis=0)` sees the bricks and calls the
    column solid, so an ordinary pit death there was classed FLOOR_CLIP -
    an anomaly. The controlled validation produced 1,079 such pixels in one
    episode with Mario in state=fall.
    """
    solid, _ = reach.rasterize_solids(level_state)
    x0, x1, _y0, _y1 = PIT_UNDER_BRICKS
    lower = solid[500:config.LEVEL_H, x0:x1 + 1]
    assert not lower.any(), "this column is not actually a pit any more"
    overhead = solid[:300, x0:x1 + 1]
    assert overhead.any(), "the floating bricks that caused the bug are gone"

    old_rule = ~solid.any(axis=0)
    assert not old_rule[x0:x1 + 1].any(), (
        "the retired whole-column rule would now call this a pit, so this "
        "test no longer pins the bug it was written for")
    new_rule = ~solid[config.LEVEL_H - config.MARIO_BIG_H:, :].any(axis=0)
    assert new_rule[x0:x1 + 1].all(), \
        "the floor-band rule does not recognise the pit under the bricks"


def test_the_pit_rule_only_ever_widens(level_state):
    """Every column the old rule called bottomless still is."""
    solid, _ = reach.rasterize_solids(level_state)
    old_rule = ~solid.any(axis=0)
    new_rule = ~solid[config.LEVEL_H - config.MARIO_BIG_H:, :].any(axis=0)
    assert not (old_rule & ~new_rule).any(), (
        "the floor-band rule lost a column the whole-column rule accepted - "
        "it is supposed to be a strict widening")
    assert int(new_rule.sum()) == 458 and int(old_rule.sum()) == 367


def test_falling_into_that_pit_is_not_an_anomaly(class_map):
    x0, x1, y0, y1 = PIT_UNDER_BRICKS
    assert _classes_at(class_map, (x0, x1, y0, y1)) == {'pit_fall'}


def test_falling_through_a_real_floor_is_still_a_clip(class_map):
    """The widening must not have made every below-world pixel innocent."""
    # x 1000 is ordinary ground: a collider below the world there passed
    # through a floor, which is exactly what FLOOR_CLIP is for.
    assert _classes_at(class_map, (1000, 1020, 605, 620)) == {'floor_clip'}
    total = int((class_map == reach.CLS_FLOOR_CLIP).sum())
    assert total > 800_000, (
        f"only {total:,} floor_clip px remain - the pit rule has widened far "
        f"past the 8,372 px it was measured to move")


# ── The corrected map: right answers, and still a working detector ───────
def test_both_observed_clusters_are_explained_and_neither_is_anomalous(class_map):
    assert _classes_at(class_map, BRICK20_CORE) == {'mutable_solid'}
    assert _classes_at(class_map, PIPE6_CORNER) == {'sweep_artifact'}


def test_the_short_validation_coverage_now_reports_zero_anomalies(class_map):
    """The end-to-end claim, against the real 6,032,768-step coverage file.

    The PROTECTED copy, not the root file: every training run overwrites the
    root pair when it saves, so a test reading it would score whatever the
    last experiment happened to leave there. The bitmap is identical.
    """
    import os
    path = os.path.join("checkpoints_qa", "pre_main_6032768", "glitch_hunter_qa_coverage.npz")
    if not os.path.exists(path):
        pytest.skip("the short-validation coverage file is not in this checkout")
    from exploration.coverage import load_testable
    cov = SpatialCoverage(testable_mask=load_testable())
    with np.load(path, allow_pickle=False) as d:
        cov.visited[:] = np.unpackbits(
            d['visited_packed'])[:config.GRID_H * config.GRID_W].reshape(
                config.GRID_H, config.GRID_W)
    b = cov.noncoverage_breakdown()
    assert cov.anomalous_px() == 0, (
        f"still {cov.anomalous_px()} anomalous px: {b['anomalous']}")
    assert b['expected']['mutable_solid'] == 579
    assert b['model_gap']['sweep_artifact'] == 20
    # The coverage numerator itself must not have moved - not by the taxonomy
    # corrections, and not by the flag-trigger correction either, which only
    # removed pixels no trajectory had ever covered.
    assert cov.covered_testable() == 2_225_509
    # 4,013,723 before two corrections that nearly cancel: +155,582 px for
    # big Mario's collider, -167,210 px past the flag trigger - giving 4,002,095;
    # then -244,105 px no measured jump arc reaches (real play the arcs deny is
    # kept, which is why the numerator above did not move).
    assert config.TESTABLE_TOTAL == 3_757_990


def test_the_detector_still_fires_on_a_real_deep_clip(class_map):
    """The correction must not have disarmed deep_penetration."""
    deep = int((class_map == reach.CLS_DEEP_PENETRATION).sum())
    assert deep > 400_000, (
        f"only {deep:,} px are still deep_penetration - the new classes have "
        f"absorbed far more than the measured 44,933 and the detector is no "
        f"longer meaningfully armed")

    ys, _xs = np.nonzero(class_map == reach.CLS_DEEP_PENETRATION)
    assert len(ys), "no deep_penetration pixels left to test against"


def test_penetration_tol_was_not_quietly_widened():
    """The fix that was explicitly not wanted."""
    assert config.PENETRATION_TOL == 6, (
        "PENETRATION_TOL moved. The 599 false positives sat 7-22 px deep, so "
        "widening the tolerance would have hidden them instead of explaining "
        "them - and would have stopped the detector seeing real clips of the "
        "same depth anywhere else in the level.")


def test_the_taxonomy_groups_stay_a_partition(class_map):
    all_classes = set(reach.CLASS_NAMES)
    grouped = (set(reach.EXPECTED_CLASSES) | set(reach.MODEL_GAP_CLASSES)
               | set(reach.ANOMALOUS_CLASSES) | {reach.CLS_TESTABLE})
    assert grouped == all_classes, "a class belongs to no group, or to two"
    assert len(reach.EXPECTED_CLASSES) + len(reach.MODEL_GAP_CLASSES) \
        + len(reach.ANOMALOUS_CLASSES) + 1 == len(all_classes)
    assert reach.CLS_MUTABLE_SOLID in reach.EXPECTED_CLASSES
    assert reach.CLS_SWEEP_ARTIFACT in reach.MODEL_GAP_CLASSES, (
        "sweep_artifact describes a limit of how coverage is RECORDED, not "
        "something the engine did, so it must not be filed as normal engine "
        "behaviour")
    assert reach.CLS_DEEP_PENETRATION in reach.ANOMALOUS_CLASSES


def test_the_bootstrap_baseline_is_unchanged_by_the_correction(class_map):
    """The 6M bootstrap read 0 anomalies before the change; it still must."""
    import os
    path = config.BOOTSTRAP_COVERAGE
    if not os.path.exists(path):
        pytest.skip("the bootstrap coverage file is not in this checkout")
    from exploration.coverage import load_testable
    cov = SpatialCoverage(testable_mask=load_testable())
    with np.load(path, allow_pickle=False) as d:
        cov.visited[:] = np.unpackbits(
            d['visited_packed'])[:config.GRID_H * config.GRID_W].reshape(
                config.GRID_H, config.GRID_W)
    b = cov.noncoverage_breakdown()
    assert cov.anomalous_px() == 0
    # None of the bootstrap's pixels moved into the new classes.
    assert b['expected'].get('mutable_solid', 0) == 0
    assert b['model_gap'].get('sweep_artifact', 0) == 0
    assert b['expected']['jump_arc'] == 7_578
    assert b['expected']['pit_fall'] == 1_410
    assert b['expected']['collision_tolerance'] == 699


# ══════════════════════════════════════════════════════════════════════════
# THE FLAG TRIGGER
#
# Checkpoint '11' is a 6 x 600 rect at x 8504, y 5 - the whole level height.
# Touching it at any height hands Mario to the scripted flag sequence, so
# past it he can occupy only the grab band (free reachability, unchanged)
# and the recorded slide-and-walk corridor. Reachability Methods B and C
# model solids only, and flooded 147,060 unreachable px into the old
# denominator.
# ══════════════════════════════════════════════════════════════════════════
TRIGGER = (8504, 5, 6, 600)


def _world(fill=False):
    m = np.zeros((config.LEVEL_H, config.LEVEL_W), dtype=bool)
    if fill:
        m[:] = True
    return m


def test_the_flag_trigger_is_read_from_the_level_itself(level_state):
    """Not hard-coded: a layout change must move or remove it, loudly."""
    rect = reach.flag_trigger_rect(level_state)
    assert rect == TRIGGER, (
        f"checkpoint '11' is at {rect}; the flag-trigger correction and the "
        f"denominator were derived for {TRIGGER}")
    assert rect[1] <= 5 and rect[1] + rect[3] >= config.LEVEL_H, \
        "the trigger no longer spans the full level height, so it can be bypassed"


def test_nothing_before_the_trigger_is_touched():
    testable = _world(fill=True)
    corrected, removed = reach.apply_flag_trigger(testable, TRIGGER, _world())
    assert corrected[:, :TRIGGER[0]].all()
    assert not removed[:, :TRIGGER[0]].any()


def test_the_grab_band_keeps_free_reachability_at_every_height():
    """The grab can happen at any height free movement reaches, so the band
    is never thinned - that is what makes the rule a superset."""
    testable = _world(fill=True)
    corrected, _ = reach.apply_flag_trigger(testable, TRIGGER, _world())
    band_right = reach.flag_grab_band_right(TRIGGER)
    assert band_right == TRIGGER[0] + TRIGGER[2] + config.MARIO_BIG_W + config.MAX_FRAME_DX
    assert corrected[:, TRIGGER[0]:band_right].all()
    assert not corrected[:, band_right:].any(), \
        "with no recorded corridor nothing past the band may stay testable"


def test_beyond_the_band_only_the_filled_corridor_survives():
    testable = _world(fill=True)
    corridor = _world()
    corridor[458, 8700] = True          # one recorded collider pixel...
    corridor[520, 8700] = True
    corrected, removed = reach.apply_flag_trigger(testable, TRIGGER, corridor)
    col = corrected[:, 8700]
    assert col[458:].all(), "the column is not filled down from the recorded top edge"
    assert not col[:458].any(), "the fill reached ABOVE where a collider was recorded"
    assert not corrected[:, 8701].any(), "a column with no recording was kept"
    assert int(corrected.sum() + removed.sum()) == int(testable.sum())


def test_the_correction_only_ever_removes_pixels():
    """It never invents testable space the geometry did not already allow."""
    rng = np.random.default_rng(0)
    testable = rng.random((config.LEVEL_H, config.LEVEL_W)) < 0.5
    corridor = rng.random((config.LEVEL_H, config.LEVEL_W)) < 0.01
    corrected, removed = reach.apply_flag_trigger(testable, TRIGGER, corridor)
    assert not (corrected & ~testable).any()
    assert not (corrected & removed).any()
    assert np.array_equal(corrected | removed, testable)


def test_a_rebuild_without_the_corridor_is_refused(level_state):
    """Silently rebuilding the pre-correction mask would put back 147,060
    unreachable px, and an exact-equality success rule could never fire."""
    with pytest.raises(reach.ReconciliationError, match="flag corridor"):
        reach.build_testable(level_state, (110, 498))


def test_beyond_the_flag_is_anomalous_not_a_model_gap():
    """Coverage there means Mario passed the flagpole without the script."""
    assert reach.CLS_BEYOND_FLAG in reach.ANOMALOUS_CLASSES
    assert reach.CLS_BEYOND_FLAG not in reach.MODEL_GAP_CLASSES
