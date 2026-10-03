"""reporting/fix_hint.py: "Where to fix it" leads to the code that causes a bug.

The six benchmark bugs are the check. tests/fix_hint_incidents.json holds the
eight real incidents the dashboard recorded for them (trimmed to what the
lead reads: the detector's measurements and the geometry near Mario). For
each, the line the dashboard shows first, and every "suspect" line, must fall
inside a block of code that differs from the clean game and carries that
bug's INJECTED BUG marker - so the lead lands on the injected code without
ever reading INJECTED_BUGS.json. Then the rules on hand-made geometry, and on
the clean game, where every lead must point at real, unchanged code.
"""
import copy
import difflib
import json
import os

import pytest

from exploration import config
from reporting import fix_hint as fh
from reporting import render
from reporting.variants import BUG_MARKER, game_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "tests", "fix_hint_incidents.json"), encoding="utf-8") as _fh:
    INCIDENTS = json.load(_fh)
BUG_OF_KIND = {"impossible_jump": "open-sky-jump", "above_world": "open-sky-jump",
               "clip_into_pipe": "pipe-clip", "invisible_collision": "invisible-wall",
               "clip_into_block": "ceiling-clip", "clip_into_step": "stair-clip",
               "stomp_without_contact": "far-stomp"}


def _changed_blocks(rel):
    """[(first, last, {bug ids})] of the bugged file's lines that differ from
    the clean game's."""
    def lines(variant):
        with open(os.path.join(game_dir(variant), rel), encoding="utf-8") as fh:
            return fh.read().splitlines()
    a, b = lines(config.CLEAN_GAME_VARIANT), lines(config.BUGGED_GAME_VARIANT)
    out = []
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag != "equal" and j2 > j1:
            out.append((j1 + 1, j2, {m for line in b[j1:j2] for m in BUG_MARKER.findall(line)}))
    return out


def _bugs_at(place):
    """The bugs whose changed code this place quotes: the markers on its own
    lines, or - for a changed line with no marker of its own - its block's."""
    rel = place["file"].split(f"{config.BUGGED_GAME_VARIANT}/", 1)[1]
    with open(os.path.join(ROOT, *place["file"].split("/")), encoding="utf-8") as fh_:
        lines = fh_.read().splitlines()
    out = set()
    for first, last, ids in _changed_blocks(rel):
        a, b = max(first, place["line"]), min(last, place["end_line"])
        if a <= b:
            own = {m for i in range(a, b + 1) for m in BUG_MARKER.findall(lines[i - 1])}
            out |= own or ids
    return out


@pytest.mark.parametrize("incident", INCIDENTS, ids=lambda r: f"{r['fingerprint']['kind']}-{r['incident_id'][-6:]}")
def test_the_lead_lands_on_the_injected_code(incident):
    bug = BUG_OF_KIND[incident["fingerprint"]["kind"]]
    hint = fh.fix_hint(incident)
    assert hint and hint["places"], "no lead"
    suspects = [p for p in hint["places"] if p["suspect"]]
    assert suspects
    for p in suspects:
        assert _bugs_at(p) == {bug}, f"{p['file']}:{p['line']} is not {bug}'s code"
    # What the dashboard's bug card shows first is one of them.
    head = fh.headline(hint)
    first = suspects[0]
    assert head["where"].startswith(f"{first['file']} line {first['line']}")


def test_every_changed_block_of_the_ceiling_clip_is_named():
    """Fixing a bug whole means undoing it everywhere it acts: the brick's
    flag and both collision checks that honour it."""
    record = next(r for r in INCIDENTS if r["fingerprint"]["kind"] == "clip_into_block")
    touched = {(p["line"], p["end_line"]) for p in fh.fix_hint(record)["places"] if p["suspect"]}
    blocks = [(f, last) for f, last, ids in _changed_blocks("data/states/level1.py") if "ceiling-clip" in ids]
    assert len(blocks) == 3
    for first, last in blocks:
        assert any(a <= last and b >= first for a, b in touched), (first, last)


def test_a_short_collider_gets_the_drawn_size_as_the_fix():
    record = next(r for r in INCIDENTS if r["fingerprint"]["kind"] == "clip_into_pipe")
    hint = fh.fix_hint(record)
    assert hint["diagnosis"] == "collider_smaller_than_drawing"
    assert "width 83, height 170" in hint["suggestion"]
    assert hint["places"][0]["code"] == ["pipe4 = collider.Collider(2445, 398, 83, 138)"]


def test_a_stray_collider_is_found_where_it_is_built_and_where_it_joins_the_level():
    record = next(r for r in INCIDENTS if r["fingerprint"]["kind"] == "invisible_collision")
    places = fh.fix_hint(record)["places"]
    assert [p["code"] for p in places] == [["stray_step = collider.Collider(3360, 495, 40, 43)"],
                                           ["stray_step)  # INJECTED BUG invisible-wall"]]


# ── the rules, on hand-made geometry against the CLEAN game ──────────────────
def _clip(solid_rect, colliders, solid="pipe", entered="top"):
    x, y = solid_rect[0] + 10, solid_rect[1] - 20
    return {"incident_id": "INC-test", "synthetic": False, "fingerprint": {"kind": f"clip_into_{solid}"},
            "detector": {"metrics": {"solid": solid, "solid_rect": list(solid_rect),
                                     "mario_rect": [x, y + 10, 30, 40], "depth_x": 20,
                                     "depth_y": 10, "entered_from": entered}},
            "provenance": {"game": {"variant": config.CLEAN_GAME_VARIANT}},
            "geometry": [{"group": solid, "x": c[0], "y": c[1], "w": c[2], "h": c[3]} for c in colliders]}


def test_a_drawing_with_no_collider_points_at_the_setup_function():
    hint = fh.fix_hint(_clip((1973, 366, 83, 170), []))
    assert hint["diagnosis"] == "missing_collider"
    assert hint["places"][0]["function"] == "Level1.setup_pipes"
    assert "x 1973, y 366, 83 x 170 px" in hint["suggestion"]


def test_a_full_size_collider_points_at_the_collision_checks_vertical_first():
    hint = fh.fix_hint(_clip((1973, 366, 83, 170), [(1973, 366, 83, 170)]))
    assert hint["diagnosis"] == "collision_check"
    assert [p["function"] for p in hint["places"]] == ["Level1.check_mario_y_collisions",
                                                        "Level1.check_mario_x_collisions"]
    assert all(not p["suspect"] for p in hint["places"])      # the clean game overrides nothing
    side = fh.fix_hint(_clip((1973, 366, 83, 170), [(1973, 366, 83, 170)], entered="left side"))
    assert side["places"][0]["function"] == "Level1.check_mario_x_collisions"


def test_on_the_clean_game_every_quoted_line_is_real_code():
    for record in INCIDENTS:
        clean = copy.deepcopy(record)
        clean["provenance"]["game"]["variant"] = config.CLEAN_GAME_VARIANT
        hint = fh.fix_hint(clean)
        for p in (hint or {}).get("places", []):
            path = os.path.join(ROOT, *p["file"].split("/"))
            with open(path, encoding="utf-8") as fh_:
                lines = fh_.read().splitlines()
            assert p["file"].startswith(f"{config.GAMES_DIR}/{config.CLEAN_GAME_VARIANT}/")
            assert p["code"] and p["code"][0].strip() == lines[p["line"] - 1].strip()
            assert "INJECTED" not in " ".join(p["code"])


def test_nothing_to_fix_for_a_synthetic_event_or_an_unknown_game():
    assert fh.fix_hint({"synthetic": True}) is None
    record = copy.deepcopy(INCIDENTS[0])
    record["provenance"]["game"]["variant"] = "not_a_game"
    assert fh.fix_hint(record) is None
    assert fh.headline(None) is None


@pytest.mark.parametrize(("kind", "metrics", "functions"), [
    ("below_world", {"y": 700}, {"Level1.check_for_mario_death"}),
    ("speed", {"x_vel": 20}, {"Mario.setup_forces", "Mario.walking", "Mario.walking_to_castle"}),
    ("score_drop", {"score": 90}, set()),        # the clean level only ever adds to the score
])
def test_the_engine_rule_findings_point_at_the_rule(kind, metrics, functions):
    record = {"synthetic": False, "fingerprint": {"kind": kind}, "detector": {"metrics": metrics},
              "provenance": {"game": {"variant": config.CLEAN_GAME_VARIANT}}, "geometry": []}
    hint = fh.fix_hint(record)
    assert hint["summary"]
    assert {p["function"] for p in hint["places"]} == functions


def test_the_report_carries_the_lead_and_labels_it_inferred():
    record = next(r for r in INCIDENTS if r["fingerprint"]["kind"] == "clip_into_step")
    lines, places = render.fix_lines(record)
    assert lines[0].startswith("This stair step's collider is smaller than its drawing")
    assert lines[-1] == fh.BASIS and "not a proven root cause" in fh.BASIS
    assert render.place_title(places[0]).startswith("games/mario_bugged/data/states/level1.py line ")
    assert render.fix_lines({"synthetic": True}) == ([], [])


# ── the proof: apply every suggested fix, and each bug is gone ─────────────
BUG_SCENARIOS = {
    "stair-clip": ("land_on_step4", "land_on_step5", "walk_step3_into_step4"),
    "pipe-clip": ("pipe4_walk_top",),
    "ceiling-clip": ("jump_under_brick20",),
    "invisible-wall": ("walk_across_wall_spot",),
    "far-stomp": ("drop_onto_goomba14",),
    "open-sky-jump": ("tap_jump_at_x300", "held_jump_at_x300"),
}


def _suggested_edits():
    """{file: {(first, last): fix}} - every suspect's fix, over all incidents."""
    edits = {}
    for record in INCIDENTS:
        for place in fh.fix_hint(record)["places"]:
            if place["suspect"] and place["fix"]:
                fix = place["fix"]
                edits.setdefault(place["file"], {})[(fix["first"], fix["last"])] = fix
    return edits


def _patched_copy(tmp_path):
    """A copy of the bugged game with every suggested fix applied, bottom-up."""
    import shutil
    root = tmp_path / "patched"
    shutil.copytree(game_dir(config.BUGGED_GAME_VARIANT), root / config.GAMES_DIR / config.BUGGED_GAME_VARIANT,
                    ignore=shutil.ignore_patterns("__pycache__"))
    for rel, fixes in _suggested_edits().items():
        path = root / rel
        lines = path.read_text(encoding="utf-8").splitlines()
        for (first, last), fix in sorted(fixes.items(), reverse=True):
            lines[first - 1:last] = fix["code"] if fix["do"] == "replace" else []
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def _probe(script, *args):
    import subprocess
    import sys
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", PYTHONUTF8="1")
    done = subprocess.run([sys.executable, os.path.join(ROOT, "tests", script), *args], cwd=ROOT,
                          env=env, capture_output=True, text=True, timeout=600, check=False)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_every_suggested_fix_is_a_concrete_edit():
    edits = _suggested_edits()
    assert set(edits) == {"games/mario_bugged/data/states/level1.py",
                          "games/mario_bugged/data/components/mario.py"}
    # 2 take-offs, 1 pipe, 2 stair columns, 2 for the wall (built, listed),
    # 3 for the brick (its flag, both collision checks), 1 stomp check.
    assert sum(len(v) for v in edits.values()) == 11
    for record in INCIDENTS:
        assert fh.headline(fh.fix_hint(record))["fixes"], record["fingerprint"]["kind"]


def test_applying_the_suggested_fixes_makes_every_bug_spot_behave_like_the_clean_game(tmp_path):
    """Each benchmark bug's scenarios, played on the bugged game with the
    suggested fixes applied, are identical to the clean game's; unpatched,
    they are not (tests/test_injected_bugs.py)."""
    clean = _probe("bug_probe.py", config.CLEAN_GAME_VARIANT)
    fixed = _probe("fix_probe.py", str(_patched_copy(tmp_path)))
    for bug, names in BUG_SCENARIOS.items():
        for name in names:
            assert fixed[name] == clean[name], f"{bug}: {name} still differs after the fix"
    controls = [k for k in clean if k.startswith("control_")]
    assert controls and all(fixed[k] == clean[k] for k in controls)
