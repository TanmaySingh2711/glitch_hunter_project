"""The real-arc envelope: reachability from measured jumps instead of a rectangle.

The rectangle (rise JUMP_RISE_PX AND reach JUMP_REACH_PX at once) lets Mario
land on ledges no jump touches. These tests build a tiny level where the two
models disagree on purpose, so the difference is a property of the code and not
of the level data.
"""
import numpy as np
import pytest

from exploration import config
from exploration import reachability as reach

H, W = config.LEVEL_H, config.LEVEL_W
GROUND_TOP = 538
SPAWN = (110, GROUND_TOP - config.MARIO_SMALL_H)        # the collider's top-left, on the island


def synthetic_arcs(speeds=(0, 4, 8), fall_frames=20) -> np.ndarray:
    """Parabolic jumps under the engine's rising gravity (0.31), then a fall."""
    paths = []
    for v in speeds:
        t = np.arange(0, 69)
        rise = 10.5 * t - 0.155 * t * t                  # 178 px apex at t = 34
        dy = list(-np.round(rise).astype(int))
        dx = [int(v * k) for k in t]
        for _k in range(1, fall_frames + 1):
            dy.append(dy[-1] + 11)
            dx.append(dx[-1] + v)
        paths.append(np.stack([dy, dx], axis=1))
    return np.array(paths, dtype=np.int32)


def island_world(rise: int, x0: int) -> np.ndarray:
    """A 200 px island over a pit, and ONE 120 px ledge `rise` px up starting at x0.

    A single ledge on purpose: a second one would be a stepping stone, and the
    chain island -> ledge -> ledge is reachable by both models.
    """
    solid = np.zeros((H, W), dtype=bool)
    solid[GROUND_TOP:, 0:200] = True                     # island
    solid[GROUND_TOP - rise:GROUND_TOP - rise + 12, x0:x0 + 120] = True
    return solid


def test_fall_closure_falls_to_the_first_blocked_cell():
    valid = np.ones((10, 5), dtype=bool)
    valid[6, 2] = False
    seed = np.zeros_like(valid)
    seed[1, 2] = True
    got = reach.fall_closure(seed, valid)
    assert got[1:6, 2].all() and not got[6:, 2].any() and not got[0, 2]
    assert got.sum() == 5


def test_walk_closure_covers_a_supported_run_and_one_cell_past_each_end():
    valid = np.ones((4, 12), dtype=bool)
    standable = np.zeros_like(valid)
    standable[2, 3:9] = True                             # a ledge, cells 3..8
    seed = np.zeros_like(valid)
    seed[2, 5] = True
    got = reach.walk_closure(seed, valid, standable)
    assert got[2, 3:9].all()
    assert got[2, 2] and got[2, 9]                       # steps off either end (then falls)
    assert not got[2, 1] and not got[2, 10]
    assert got.sum() == 8


def test_an_arc_cannot_land_where_the_rectangle_says_it_can():
    # A ledge 170 px up and 440 px out: the rectangle (rise <= 183, reach <= 480)
    # allows it, but a jump 170 px up is at its apex, a couple of hundred px along.
    solid = island_world(rise=170, x0=640)
    px, stats = reach.method_arcs(solid, SPAWN, synthetic_arcs())
    rect_px, _s = reach.method_c(solid, SPAWN, coarse=4)
    top = GROUND_TOP - 170
    # Standing on the ledge occupies the pixel row just above its top face.
    assert rect_px[top - 1, 700], "the rectangle model should reach the ledge"
    assert not px[top - 1, 700], "no measured arc gets 170 px up 440 px out"
    assert stats['launch_anchors'] > 0 and stats['rounds'] >= 1


def test_the_arc_envelope_only_ever_removes_pixels():
    solid = island_world(rise=170, x0=640)
    arcs = synthetic_arcs()
    testable, _s = reach.method_c(solid, SPAWN, coarse=4)
    tightened, arc_px, stats = reach.apply_arc_envelope(testable, solid, SPAWN, arcs)
    assert not (tightened & ~testable).any(), "the filter added a pixel"
    assert (tightened == (testable & arc_px)).all()
    assert stats['arcs_removed_px'] == int((testable & ~tightened).sum()) > 0


def test_pixels_real_play_demonstrated_survive_the_filter():
    solid = island_world(rise=170, x0=640)
    arcs = synthetic_arcs()
    testable, _s = reach.method_c(solid, SPAWN, coarse=4)
    far = np.zeros_like(testable)
    far[GROUND_TOP - 171:GROUND_TOP - 170, 690:710] = True       # observed by the engine
    far &= testable
    assert far.any()
    tightened, _arc_px, stats = reach.apply_arc_envelope(testable, solid, SPAWN, arcs, far)
    assert tightened[far].all()
    assert stats['observed_px'] == int(far.sum())


def test_a_ledge_a_real_jump_lands_on_is_reached():
    """Positive control: 100 px up, 230 px out is an ordinary landing."""
    solid = island_world(rise=100, x0=430)
    px, _s = reach.method_arcs(solid, SPAWN, synthetic_arcs())
    assert px[GROUND_TOP - 101, 480]


def test_load_jump_arcs_refuses_an_arc_that_does_not_start_at_take_off(tmp_path):
    bad = synthetic_arcs()
    bad[0, 0] = (3, 4)
    path = tmp_path / "arcs.npz"
    np.savez_compressed(path, paths=bad.astype(np.int16))
    with pytest.raises(reach.ReachabilityMismatch, match="take-off"):
        reach.load_jump_arcs(str(path))


def test_load_jump_arcs_refuses_the_wrong_shape(tmp_path):
    path = tmp_path / "arcs.npz"
    np.savez_compressed(path, paths=np.zeros((5, 7), dtype=np.int16))
    with pytest.raises(reach.ReachabilityMismatch, match="shape"):
        reach.load_jump_arcs(str(path))


def test_observed_reach_round_trips(tmp_path):
    mask = np.zeros((H, W), dtype=bool)
    mask[100:110, 6000:6100] = True
    path = str(tmp_path / "observed.npz")
    reach.save_observed_reach(path, mask, ["a.npz", "b.npz"])
    assert np.array_equal(reach.load_observed_reach(path), mask)
    assert reach.load_observed_reach(str(tmp_path / "absent.npz")) is None
