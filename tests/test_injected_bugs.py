"""Each of the six benchmark bugs in mario_bugged, proved against mario_clean.

tests/bug_probe.py plays the same controlled scenarios on both games (one
fresh process each) and measures Mario against the level as it is DRAWN. For
every bug this shows: the clean game behaves normally at that spot, the bugged
game shows the defect, it is reproducible (every scenario runs twice with an
identical trace), and a matching control spot the bug does not touch is
identical on both games. Where an Objective-3 detector can see the result, the
test says so; where none can, that is asserted too, so a future detector that
starts catching one shows up as a deliberate test change, not a surprise.
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _probe(variant):
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", PYTHONUTF8="1")
    result = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "bug_probe.py"), variant],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
                            check=False)
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def runs():
    return {"clean": _probe("mario_clean"), "bugged": _probe("mario_bugged")}


def scenario(runs, name):
    clean, bugged = runs["clean"][name], runs["bugged"][name]
    assert clean["repeatable"] and bugged["repeatable"], f"{name} is not deterministic"
    return clean, bugged


def test_every_scenario_is_repeatable(runs):
    for side in ("clean", "bugged"):
        flaky = [k for k, v in runs[side].items() if k != "variant" and not v["repeatable"]]
        assert not flaky, f"{side}: {flaky}"


def test_every_control_spot_is_identical_on_both_games(runs):
    controls = [k for k in runs["clean"] if k.startswith("control_")]
    assert len(controls) == 8
    for name in controls:
        assert runs["clean"][name] == runs["bugged"][name], name


# ── 1. stair-clip ────────────────────────────────────────────────────────────
def test_stair_clip(runs):
    for name in ("land_on_step4", "land_on_step5"):
        clean, bugged = scenario(runs, name)
        assert clean["rest_bottom"] == clean["drawn_top"] == 366 and clean["sunk_px"] == 0
        assert bugged["rest_bottom"] == 452 and bugged["sunk_px"] == 86
        assert bugged["overlap_with_block"] == [30, 40]             # Mario wholly inside the blocks
    clean, bugged = scenario(runs, "walk_step3_into_step4")
    assert clean["max_right_edge"] == clean["drawn_face_x"]         # blocked by the face
    assert bugged["max_right_edge"] > bugged["drawn_face_x"] + 40   # walked straight through it
    assert bugged["max_overlap_with_block"] == [30, 40]


# ── 2. pipe-clip ─────────────────────────────────────────────────────────────
def test_pipe_clip(runs):
    clean, bugged = scenario(runs, "pipe4_walk_top")
    assert clean["max_depth_inside_pipe_px"] == 0
    assert bugged["max_depth_inside_pipe_px"] == 40                 # his full height inside the pipe
    clean, bugged = scenario(runs, "pipe4_walk_in_from_right")
    assert clean["min_left_x"] == clean["pipe_right_edge"] == 2528
    assert bugged["min_left_x"] == 2466                             # 62 px into the pipe


# ── 3. ceiling-clip ──────────────────────────────────────────────────────────
def test_ceiling_clip(runs):
    clean, bugged = scenario(runs, "jump_under_brick20")
    assert clean["highest_top_y"] == clean["brick_bottom"] == 408   # head bump
    assert clean["max_overlap_with_brick"] == [0, 0]
    assert bugged["highest_top_y"] < 365                            # above the brick's top
    assert bugged["max_overlap_with_brick"] == [30, 40]


# ── 4. invisible-wall ────────────────────────────────────────────────────────
def test_invisible_wall(runs):
    clean, bugged = scenario(runs, "walk_across_wall_spot")
    assert not clean["stopped_at_spot"] and clean["max_right_edge"] > 4412 + 100
    assert bugged["stopped_at_spot"] and bugged["max_right_edge"] == 4412
    # Nothing is drawn there: the pixels are the clean game's own.
    assert bugged["drawn_pixels_sha256"] == clean["drawn_pixels_sha256"]


# ── 5. false-goomba-hit ──────────────────────────────────────────────────────
def test_false_goomba_hit(runs):
    clean, bugged = scenario(runs, "goomba14_walks_into_mario")
    assert clean["end"] == ["death", "goomba"] and clean["gap_at_hit_px"] == 0
    assert bugged["end"] == ["death", "goomba"] and 30 <= bugged["gap_at_hit_px"] <= 36


# ── 6. open-sky-jump ─────────────────────────────────────────────────────────
def test_open_sky_jump(runs):
    for name in ("tap_jump_at_x300", "held_jump_at_x300"):
        clean, bugged = scenario(runs, name)
        assert clean["highest_top_y"] > 300                         # an ordinary jump
        assert bugged["highest_top_y"] < -200                       # far above the screen
    assert runs["clean"]["held_jump_at_x300"]["highest_top_y"] == \
        runs["clean"]["control_held_jump_at_x600"]["highest_top_y"]


# ── Objective 3: every bug is caught by its generic detector ────────────────
# (reporting/collision_invariants.py; above_world is the pre-existing engine
# invariant and stays as it was). Exact sets: the right detector fires, and no
# other one does.
EXPECTED_DETECTIONS = {
    "land_on_step4": ["clip_into_step"],
    "land_on_step5": ["clip_into_step"],
    "walk_step3_into_step4": ["clip_into_step"],
    "pipe4_walk_top": ["clip_into_pipe"],
    "pipe4_walk_in_from_right": ["clip_into_pipe"],
    "jump_under_brick20": ["clip_into_block"],
    "walk_across_wall_spot": ["invisible_collision"],
    "goomba14_walks_into_mario": ["hit_without_contact"],
    "tap_jump_at_x300": ["above_world", "impossible_jump"],
    "held_jump_at_x300": ["above_world", "impossible_jump"],
}


def test_each_bug_is_detected_by_its_own_detector(runs):
    for name, kinds in EXPECTED_DETECTIONS.items():
        assert runs["bugged"][name]["detections"] == kinds, name


def test_no_detector_fires_anywhere_on_the_clean_game(runs):
    fired = {k: v["detections"] for k, v in runs["clean"].items() if k != "variant" and v["detections"]}
    assert fired == {}


def test_no_detector_fires_at_any_control_spot_of_the_bugged_game(runs):
    controls = {k: v["detections"] for k, v in runs["bugged"].items() if k.startswith("control_")}
    assert controls and all(v == [] for v in controls.values()), controls
