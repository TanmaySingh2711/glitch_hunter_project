"""Test 11 and friends: coverage recorded from the REAL game is world-space.

test_coverage.py drives SpatialCoverage with synthetic rects. That proves the
data structure is correct but proves nothing about the coordinates being fed
into it, which is where the single most damaging possible mistake lives:

    If mario.rect were screen-space, every episode would re-explore the same
    800px-wide strip forever. Coverage would look busy, novelty would keep
    paying out, and the agent would never be rewarded for reaching anywhere
    new. It would be a silent, permanent failure that no unit test on the
    bitmap alone could see.

So these tests drive the actual environment and check the coordinates.
"""

import numpy as np

from exploration import config
from exploration.coverage import SpatialCoverage


def _run(env, action=3, n=400):
    """Sprint right, collecting the world rect and camera each substep."""
    env.reset()
    frames = []
    for _ in range(n):
        _obs, _r, terminated, _t, info = env.step(action)
        frames.append((info.get('mario_rect'), info.get('viewport_x', 0)))
        if terminated:
            env.reset()
    return frames


def test_info_exposes_world_rect_and_viewport(fresh):
    _, _, _, _, info = fresh.step(1)
    assert 'mario_rect' in info and 'viewport_x' in info
    rect = info['mario_rect']
    assert rect is not None and len(rect) == 4
    assert all(isinstance(v, int) for v in rect)
    assert rect[2] > 0 and rect[3] > 0


def test_collider_dimensions_match_config(fresh):
    _, _, _, _, info = fresh.step(1)
    _x, _y, w, h = info['mario_rect']
    if info['status'] == 'small':
        assert (w, h) == (config.MARIO_SMALL_W, config.MARIO_SMALL_H), (
            "small-Mario collider is not what exploration/config.py assumes - "
            "the reachability denominator was built from those dimensions")


# ── Test 11 ───────────────────────────────────────────────────────────────
def test_rect_is_world_space_not_screen_space(env):
    """The camera must have moved, and the rect must NOT have moved with it."""
    frames = _run(env)
    rects = [r for r, _v in frames if r]
    cams = [v for r, v in frames if r]
    assert max(cams) > 0, "camera never scrolled - the test proves nothing"
    assert max(x for x, _y, _w, _h in rects) > 800, (
        "mario.rect.x never exceeded the 800px window width, so it may be "
        "screen-space - coverage would be confined to one screen forever")

    # Screen offset stays inside the window while world x runs well past it.
    offsets = [x - v for (x, _y, _w, _h), v in zip(rects, cams, strict=True)]
    assert min(offsets) >= -config.MARIO_BIG_W
    assert max(offsets) <= 800 + config.MARIO_BIG_W


def test_camera_scrolling_alone_paints_no_coverage(env):
    """A stationary Mario under a moving camera must claim nothing.

    This is the direct phantom-coverage check: if any substep where the world
    rect is unchanged still reports new pixels, something camera-relative has
    leaked into the recording path.
    """
    frames = _run(env)
    cov = SpatialCoverage()
    cov.begin_episode()
    prev = None
    for rect, cam in frames:
        if rect is None:
            continue
        n_new = cov.record({'cur_rect': rect, 'viewport_x': cam})
        if prev is not None and rect == prev:
            assert n_new == 0, (
                f"camera motion painted {n_new} phantom pixels at rect={rect}")
        prev = rect


def test_coverage_extends_beyond_one_screen(env):
    """World-space coverage must reach columns a screen-space one cannot."""
    frames = _run(env)
    cov = SpatialCoverage()
    cov.begin_episode()
    for rect, cam in frames:
        if rect:
            cov.record({'cur_rect': rect, 'viewport_x': cam})

    cols = np.nonzero(cov.visited.any(axis=0))[0] + cov.x0
    assert cols.max() > 800, "coverage never left the first screen"

    # The same trajectory recorded in SCREEN space - what the bug would look
    # like - is confined to the window and covers dramatically less ground.
    screen_cov = SpatialCoverage()
    screen_cov.begin_episode()
    for rect, cam in frames:
        if rect:
            x, y, w, h = rect
            screen_cov.record({'cur_rect': (x - cam, y, w, h)})
    screen_cols = np.nonzero(screen_cov.visited.any(axis=0))[0] + screen_cov.x0
    assert screen_cols.max() <= 800 + config.MARIO_BIG_W
    assert cov.total_unique() > screen_cov.total_unique(), (
        "world-space and screen-space recording are indistinguishable here, "
        "so this test cannot tell the two apart")


def test_reset_does_not_sweep_the_spawn_teleport(env):
    """Coverage survives reset, and the respawn jump is not painted."""
    frames = _run(env, n=120)
    cov = SpatialCoverage()
    cov.begin_episode()
    for rect, cam in frames:
        if rect:
            cov.record({'cur_rect': rect, 'viewport_x': cam})
    before = cov.total_unique()
    assert before > 0

    env.reset()
    cov.begin_episode()
    assert cov.total_unique() == before, "reset cleared persistent coverage"

    _obs, _r, _term, _t, info = env.step(1)
    n_new = cov.record({'cur_rect': info['mario_rect']})
    # A swept teleport would paint a corridor thousands of pixels wide.
    assert n_new <= config.MARIO_BIG_W * config.MARIO_BIG_H, (
        f"the spawn teleport was swept as a corridor ({n_new} px)")
