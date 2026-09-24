"""The collision invariants (reporting/collision_invariants.py) and the level
design they measure against (reporting/level_design.py).

Unit tests drive the rules with plain stand-in sprites, including the two
false positives the clean-game validation found while the detectors were
being built (both are pinned here as regressions). The end-to-end proof -
each rule firing on its injected bug in mario_bugged and staying silent on
the clean game at the same spots - is in tests/test_injected_bugs.py.
"""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pygame as pg
import pytest

from reporting import collision_invariants as ci
from reporting import level_design

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUND = ("ground", (0, 538, 2000, 60))
PIPE = ("pipe", (600, 452, 83, 86))


def sprite(x, y, w, h, **kw):
    return SimpleNamespace(rect=pg.Rect(x, y, w, h), **kw)


def level(mario, bricks=(), boxes=(), enemies=(), shells=()):
    return SimpleNamespace(mario=mario, brick_group=list(bricks), coin_box_group=list(boxes),
                           enemy_group=list(enemies), shell_group=list(shells))


def mario(x, y, w=30, h=40, x_vel=0.0, y_vel=0.0, state="walk", dead=False, cause=None, **kw):
    return sprite(x, y, w, h, x_vel=x_vel, y_vel=y_vel, state=state, dead=dead,
                  death_cause=cause, **kw)


def run(inv, *frames):
    """Feed frames (levels) in order; returns every hit of the last one."""
    hits = []
    for lv in frames:
        hits = inv.check(lv)
    return hits


def kinds(hits):
    return [h[0] for h in hits]


@pytest.fixture
def inv():
    return ci.CollisionInvariants([GROUND, PIPE])


# ── helpers ──────────────────────────────────────────────────────────────────
def test_geometry_helpers():
    assert ci.overlap((0, 0, 10, 10), (5, 5, 10, 10)) == (5, 5)
    assert ci.overlap((0, 0, 10, 10), (10, 0, 10, 10)) == (0, 0)          # flush is not inside
    assert ci.gap((0, 0, 10, 10), (13, 0, 5, 5)) == 3
    assert ci.gap((0, 0, 10, 10), (5, 5, 5, 5)) == 0
    assert ci.union((0, 0, 10, 10), (20, 5, 10, 10)) == (0, 0, 30, 15)


# ── rule 1: solid penetration ────────────────────────────────────────────────
def test_standing_flush_on_a_solid_is_not_penetration(inv):
    assert run(inv, level(mario(610, 412)), level(mario(612, 412))) == []


def test_sinking_into_a_designed_solid_is_reported_with_its_kind_and_side(inv):
    hits = run(inv, level(mario(610, 400, y_vel=5, state="fall")),
               level(mario(610, 440, y_vel=5, state="fall")))
    assert kinds(hits) == ["clip_into_pipe"]
    metrics = hits[0][2]
    assert metrics["entered_from"] == "top" and metrics["depth_y"] == 28


def test_penetration_within_the_tolerance_is_not_reported(inv):
    assert run(inv, level(mario(610, 412 + ci.PENETRATION_TOL))) == []


def test_a_live_brick_entered_from_below_is_a_ceiling_clip(inv):
    brick = sprite(300, 365, 43, 43)
    hits = run(inv, level(mario(305, 420, y_vel=-8, state="jump"), bricks=[brick]),
               level(mario(305, 390, y_vel=-7, state="jump"), bricks=[brick]))
    assert kinds(hits) == ["clip_into_block"]
    assert hits[0][2]["entered_from"] == "bottom" and "ceiling clip" in hits[0][1]


def test_nothing_is_judged_while_the_engine_suspends_collision(inv):
    for state in ("death jump", "flag pole", "small to big", "walking to castle"):
        assert run(inv, level(mario(610, 440, state=state))) == []
    assert run(inv, level(mario(610, 440, dead=True))) == []
    assert run(inv, level(mario(610, 440, in_transition_state=True))) == []


# ── rule 2: collision with nothing ──────────────────────────────────────────
# The running game's colliders (ground_step_pipe_group) are passed separately:
# they are only proof that a collision happened. STRAY is one that is not drawn.
STRAY = (330, 452, 40, 86)


def lvl(m, colliders=(GROUND[1], PIPE[1]), **kw):
    lv = level(m, **kw)
    lv.ground_step_pipe_group = [sprite(*r) for r in colliders]
    return lv


def test_being_stopped_by_a_drawn_solid_is_fine(inv):
    assert run(inv, lvl(mario(560, 498, x_vel=6)), lvl(mario(570, 498, x_vel=0))) == []


def test_being_stopped_by_an_undrawn_collider_is_reported(inv):
    cols = (GROUND[1], PIPE[1], STRAY)
    hits = run(inv, lvl(mario(290, 498, x_vel=6), colliders=cols),
               lvl(mario(300, 498, x_vel=0), colliders=cols))
    assert kinds(hits) == ["invisible_collision"] and hits[0][2]["contact"] == "side"
    assert hits[0][2]["undrawn_contact_rect"][0] == 330


def test_regression_a_stop_with_no_collider_touching_is_not_a_collision(inv):
    """Clean game, in front of pipes 3 and 4: with sprint held and no direction
    the engine zeroes x_vel in one frame by itself (RUN_ACCEL = 20). No
    collider touches him, so nothing collided."""
    assert run(inv, lvl(mario(300, 498, x_vel=-1.4)), lvl(mario(300, 488, x_vel=0, y_vel=-10,
                                                                  state="fall"))) == []


def test_regression_side_contact_is_judged_at_the_height_before_the_vertical_move(inv):
    """Clean game, pipe 4: the pipe stopped Mario while his feet were still 3 px
    below its top; the same frame's upward move then lifted them to exactly its
    top. level1 resolves x before y, so the probe spans both heights."""
    hits = run(inv, lvl(mario(560, 415, x_vel=6, y_vel=-3.06, state="jump")),
               lvl(mario(570, 412, x_vel=0, y_vel=-2.75, state="jump")))
    assert hits == []


def test_standing_on_an_undrawn_collider_is_reported_and_on_a_solid_is_not(inv):
    assert run(inv, lvl(mario(300, 498)), lvl(mario(300, 498))) == []
    cols = (GROUND[1], STRAY)
    hits = run(inv, lvl(mario(335, 400, y_vel=5, state="fall"), colliders=cols),
               lvl(mario(335, 412, state="walk"), colliders=cols))
    assert kinds(hits) == ["invisible_collision"] and hits[0][2]["contact"] == "support"


def test_regression_standing_on_the_last_pixel_of_a_ledge_is_supported(inv):
    """Clean-game shape: Mario 1 px over pipe 1's right edge (the engine's own
    support test uses his full width)."""
    assert run(inv, lvl(mario(682, 412)), lvl(mario(682, 412))) == []


def test_a_head_bump_needs_something_drawn_above(inv):
    brick = sprite(300, 365, 43, 43)
    assert run(inv, lvl(mario(305, 420, y_vel=-3, state="jump"), bricks=[brick]),
               lvl(mario(305, 408, y_vel=7, state="fall"), bricks=[brick])) == []
    cols = (GROUND[1], (300, 365, 43, 43))                  # a collider where no block is drawn
    hits = run(inv, lvl(mario(305, 420, y_vel=-3, state="jump"), colliders=cols),
               lvl(mario(305, 408, y_vel=7, state="fall"), colliders=cols))
    assert kinds(hits) == ["invisible_collision"] and hits[0][2]["contact"] == "ceiling"


def test_regression_a_brick_smashed_by_the_bump_still_counts_as_drawn(inv):
    """Clean game: big Mario's head bump kills the brick in the same frame, so
    it is gone when the frame ends - but it was there when he hit it."""
    brick = sprite(300, 365, 43, 43)
    assert run(inv, lvl(mario(305, 420, h=80, y_vel=-8, state="jump"), bricks=[brick]),
               lvl(mario(305, 408, h=80, y_vel=7, state="fall"))) == []


# ── rule 4: impossible jump ─────────────────────────────────────────────────
def test_ordinary_jumps_and_bounces_are_legal(inv):
    frames = [lvl(mario(300, 498))] + [lvl(mario(300, 498 - 10 * i, y_vel=v, state="jump"))
                                       for i, v in enumerate((-10.5, -10.19, -7.0), 1)]
    assert run(inv, *frames) == []


def test_a_take_off_faster_than_the_fastest_jump_is_reported(inv):
    hits = run(inv, lvl(mario(300, 498)), lvl(mario(300, 476, y_vel=-22, state="jump")))
    assert kinds(hits) == ["impossible_jump"] and hits[0][2]["rule"] == "rise_speed"
    assert hits[0][2]["rise_speed"] == 22 and hits[0][2]["limit"] == ci.FASTEST_JUMP_SPEED


def test_climbing_higher_than_any_legal_jump_is_reported(inv):
    """Too high even at legal speeds (e.g. weakened gravity)."""
    hits = run(inv, lvl(mario(300, 498)), lvl(mario(300, 498 - 300, y_vel=-2, state="jump")))
    assert kinds(hits) == ["impossible_jump"] and hits[0][2]["rule"] == "rise_height"
    assert hits[0][2]["climb_px"] == 300 > ci.MAX_JUMP_RISE


def test_the_jump_limits_come_from_the_engine_constants():
    assert ci.FASTEST_JUMP_SPEED == 12.5 and ci.JUMP_GRAVITY == 0.31
    assert round(ci.MAX_JUMP_RISE) == 258


# ── rule 3: hit without contact ─────────────────────────────────────────────
def test_a_death_by_a_touching_enemy_is_fine(inv):
    g = sprite(335, 498, 40, 40)
    assert run(inv, level(mario(300, 498), enemies=[g]),
               level(mario(305, 498, dead=True, cause="goomba", state="death jump"), enemies=[g])) == []


def test_a_death_with_a_gap_is_reported(inv):
    g = sprite(370, 498, 40, 40)
    hits = run(inv, level(mario(300, 498), enemies=[g]),
               level(mario(302, 498, dead=True, cause="goomba", state="death jump"), enemies=[g]))
    assert kinds(hits) == ["hit_without_contact"] and hits[0][2]["gap_px"] == 38


def test_contact_anywhere_in_the_frame_counts(inv):
    """Enemies move after Mario in the same frame: a goomba that touched him
    and then walked 2 px away is still a touch (swept boxes)."""
    g = sprite(332, 498, 40, 40)
    lv0 = level(mario(300, 498), enemies=[g])
    inv.check(lv0)
    g.rect.x = 335
    assert inv.check(level(mario(300, 498, dead=True, cause="goomba", state="death jump"),
                           enemies=[g])) == []


def test_a_shrink_with_a_gap_is_reported_and_other_deaths_are_not_judged(inv):
    g = sprite(380, 458, 40, 40)
    hits = run(inv, level(mario(300, 418, h=80, state="walk"), enemies=[g]),
               level(mario(300, 418, h=80, state="big to small"), enemies=[g]))
    assert kinds(hits) == ["hit_without_contact"] and hits[0][2]["event"] == "shrink"
    assert run(ci.CollisionInvariants([GROUND]), level(mario(300, 560)),
               level(mario(300, 600, dead=True, cause="pit", state="death jump"))) == []


# ── the design reference ─────────────────────────────────────────────────────
def test_the_design_is_exactly_mario_clean(env):
    """env is the clean game (conftest): its live static colliders ARE the file."""
    env.reset()
    live = level_design.design_from_level(env.game.state)
    design = [{"solid": k, "x": r[0], "y": r[1], "w": r[2], "h": r[3]}
              for k, r in level_design.load_design()]
    assert sorted(design, key=lambda s: (s["solid"], s["x"], s["y"], s["w"], s["h"])) == live
    assert len(live) == 37


def test_every_designed_solid_is_drawn(env):
    """No sky colour anywhere a collider must be before it is reported: inside
    every designed solid, inset by the detector's tolerance (a pipe's body is
    drawn a few px narrower than its lip). The file describes what the player
    sees, not an invisible collider."""
    env.reset()
    background = env.game.state.background
    sky = background.get_at((5, 5))
    inset = ci.PENETRATION_TOL + 1
    for solid, (x, y, w, h) in level_design.load_design():
        y1 = min(y + h, background.get_height())
        samples = [(sx, sy) for sx in range(x + inset, x + w - inset, 3)
                   for sy in range(y + inset, y1 - inset, 3)]
        sky_hits = sum(1 for p in samples if background.get_at(p) == sky)
        assert samples and sky_hits == 0, (solid, x, y, sky_hits, len(samples))


def test_the_design_file_is_current():
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", PYTHONUTF8="1")
    result = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "build_level_design.py"),
                             "--check"], cwd=ROOT, env=env, capture_output=True, text=True,
                            timeout=300, check=False)
    assert result.returncode == 0, result.stdout + result.stderr[-2000:]


def test_a_design_file_in_another_format_is_refused(tmp_path):
    bad = tmp_path / "d.json"
    bad.write_text(json.dumps({"format": 99, "solids": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        level_design.load_design(str(bad))
