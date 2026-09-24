"""The two game variants: mario_clean is the untouched baseline, mario_bugged
carries the six declared benchmark bugs (mario_bugged/INJECTED_BUGS.json) and
nothing else.

Three kinds of proof, strongest last:
  * content  - clean matches its pin; every difference in bugged is accounted
               for by a declared bug at file level, at line level (each
               changed block carries that bug's marker) and as a whole (the
               diff hashes to the pinned, validated one)
  * loading  - the env runs the variant it was asked for, from that
               directory, and refuses to mix two variants in one process
  * behaviour - each variant, in its own fresh process, plays the same
               scripted run: the level geometry differs only in the declared
               colliders, and play is identical until Mario first enters a
               declared bug's zone; the clean game still draws the
               pre-Objective-3 NOOP frames. What each bug DOES is proved in
               tests/test_injected_bugs.py.
"""
import json
import os
import subprocess
import sys

import pytest

from exploration import config
from reporting import variants

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── content ──────────────────────────────────────────────────────────────────
def test_the_clean_game_is_exactly_the_pinned_baseline():
    digest = variants.game_tree_sha256(variants.game_dir(config.CLEAN_GAME_VARIANT))
    assert digest == config.CLEAN_GAME_TREE_SHA256, (
        "mario_clean/ changed. It is the trustworthy baseline and must never be "
        "edited - deliberate bugs go in mario_bugged/ (see mario_bugged/VARIANT.md)")


# The six bugs the project owner specified on 2026-09-24, in the brief's order.
SPECIFIED_BUGS = ["stair-clip", "pipe-clip", "ceiling-clip", "invisible-wall",
                  "false-goomba-hit", "open-sky-jump"]
REQUIRED_FIELDS = ("id", "number", "name", "summary", "subsystem", "files", "location",
                   "trigger", "expected_clean", "injected_behaviour", "implementation",
                   "reproducibility", "objective3_detection", "zone")


def test_exactly_the_six_specified_bugs_are_declared():
    bugs = variants.injected_bugs(config.BUGGED_GAME_VARIANT)
    assert [b["id"] for b in bugs] == SPECIFIED_BUGS, (
        "the declared bugs changed; update this test only when the project owner "
        "has specified that change")
    assert [b["number"] for b in bugs] == list(range(1, 7))
    for bug in bugs:
        missing = [f for f in REQUIRED_FIELDS if not bug.get(f)]
        assert not missing, f"{bug['id']} lacks {missing}"


def test_every_changed_line_is_attributed_to_a_declared_bug():
    assert variants.unattributed_changes(config.BUGGED_GAME_VARIANT) == []


def test_the_bug_diff_is_exactly_the_validated_one():
    """Changing any injected bug - or adding anything - changes this hash.
    Re-run tests/test_injected_bugs.py and the reachability check, then re-pin."""
    assert variants.bug_diff_sha256(config.BUGGED_GAME_VARIANT) == \
        variants.declared_diff_sha256(config.BUGGED_GAME_VARIANT)


def test_the_bugs_live_only_in_the_files_they_declare():
    clean = variants.game_tree_files(variants.game_dir(config.CLEAN_GAME_VARIANT))
    bugged = variants.game_tree_files(variants.game_dir(config.BUGGED_GAME_VARIANT))
    changed = {f for f in set(clean) | set(bugged) if clean.get(f) != bugged.get(f)}
    declared = {f for b in variants.injected_bugs(config.BUGGED_GAME_VARIANT) for f in b["files"]}
    assert changed == declared == {"data/states/level1.py", "data/components/mario.py"}


def test_every_difference_in_the_bugged_game_must_be_declared():
    """The rule that will hold once bugs exist: an undeclared difference is
    either an accident or an undocumented bug, and fails here."""
    assert variants.undeclared_differences(config.BUGGED_GAME_VARIANT) == []


def test_an_undeclared_edit_is_caught(tmp_path, monkeypatch):
    """Negative control on a scratch copy: edit one game file and the check
    names it; declare it and the check accepts it."""
    import shutil
    for v in config.GAME_VARIANTS:
        shutil.copytree(variants.game_dir(v), tmp_path / v,
                        ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(variants, "PROJECT_ROOT", str(tmp_path))
    target = tmp_path / config.BUGGED_GAME_VARIANT / "data" / "constants.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    assert variants.undeclared_differences(config.BUGGED_GAME_VARIANT) == ["data/constants.py"]

    manifest = tmp_path / config.BUGGED_GAME_VARIANT / variants.BUG_MANIFEST_NAME
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    doc["bugs"].append({"id": "demo", "files": ["data/constants.py"]})
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    assert variants.undeclared_differences(config.BUGGED_GAME_VARIANT) == []
    ident = variants.variant_identity(config.BUGGED_GAME_VARIANT)
    assert ident["declared_injected_bugs"][-1] == "demo"
    assert not ident["matches_pinned_clean_tree"]
    # ...but declaring the FILE is not enough: the added line carries no
    # marker, so it is still unaccounted for, and the pinned diff has moved.
    assert [c.split(":")[0] for c in variants.unattributed_changes(config.BUGGED_GAME_VARIANT)] \
        == ["data/constants.py"]
    assert variants.bug_diff_sha256(config.BUGGED_GAME_VARIANT) != \
        variants.declared_diff_sha256(config.BUGGED_GAME_VARIANT)
    target.write_text(target.read_text(encoding="utf-8").replace("# edited", "# INJECTED BUG demo"),
                      encoding="utf-8")
    assert variants.unattributed_changes(config.BUGGED_GAME_VARIANT) == []


def test_an_unmarked_edit_inside_a_declared_file_is_caught(tmp_path, monkeypatch):
    """The line-level rule: level1.py is declared, yet one more unmarked
    change in it is reported - and so is a marker naming an undeclared bug."""
    import shutil
    for v in config.GAME_VARIANTS:
        shutil.copytree(variants.game_dir(v), tmp_path / v,
                        ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(variants, "PROJECT_ROOT", str(tmp_path))
    level = tmp_path / config.BUGGED_GAME_VARIANT / "data" / "states" / "level1.py"
    text = level.read_text(encoding="utf-8")
    edited = text.replace("pipe6 = collider.Collider(7675, 452, 83, 82)",
                          "pipe6 = collider.Collider(7675, 452, 60, 82)")
    assert edited != text
    level.write_text(edited, encoding="utf-8")
    assert variants.undeclared_differences(config.BUGGED_GAME_VARIANT) == []      # file is declared
    found = variants.unattributed_changes(config.BUGGED_GAME_VARIANT)
    assert len(found) == 1 and found[0].startswith("data/states/level1.py:")
    level.write_text(edited.replace("(7675, 452, 60, 82)", "(7675, 452, 60, 82)  # INJECTED BUG nope"),
                     encoding="utf-8")
    assert len(variants.unattributed_changes(config.BUGGED_GAME_VARIANT)) == 1


def test_the_tree_hash_ignores_line_endings_and_bytecode(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root, eol in ((a, b"\n"), (b, b"\r\n")):
        (root / "data" / "__pycache__").mkdir(parents=True)
        (root / "resources").mkdir()
        (root / "data" / "m.py").write_bytes(b"x = 1" + eol + b"y = 2" + eol)
        (root / "resources" / "i.png").write_bytes(b"\x89PNG\r\n")
    (b / "data" / "__pycache__" / "m.cpython-312.pyc").write_bytes(b"junk")
    assert variants.game_tree_sha256(str(a)) == variants.game_tree_sha256(str(b))
    (b / "resources" / "i.png").write_bytes(b"\x89PNG\n")       # binaries stay byte-exact
    assert variants.game_tree_sha256(str(a)) != variants.game_tree_sha256(str(b))


def test_a_malformed_bug_manifest_is_an_error_not_no_bugs(tmp_path, monkeypatch):
    (tmp_path / config.BUGGED_GAME_VARIANT).mkdir()
    monkeypatch.setattr(variants, "PROJECT_ROOT", str(tmp_path))
    (tmp_path / config.BUGGED_GAME_VARIANT / variants.BUG_MANIFEST_NAME).write_text(
        '{"bugs": {"oops": 1}}', encoding="utf-8")
    with pytest.raises(ValueError):
        variants.injected_bugs(config.BUGGED_GAME_VARIANT)


def test_only_registered_variants_exist():
    with pytest.raises(ValueError):
        variants.game_dir("../mario_clean")
    assert config.DEFAULT_GAME_VARIANT == config.CLEAN_GAME_VARIANT


# ── loading ──────────────────────────────────────────────────────────────────
def test_the_env_runs_the_clean_game_by_default(env):
    import data
    assert env.game_variant == config.CLEAN_GAME_VARIANT
    assert os.path.normcase(os.path.dirname(os.path.dirname(data.__file__))) == \
        os.path.normcase(os.path.join(ROOT, config.CLEAN_GAME_VARIANT))


def test_a_process_refuses_to_mix_variants(env):
    from custom_mario_env import claim_game_variant
    assert claim_game_variant(config.CLEAN_GAME_VARIANT)          # the one already loaded
    with pytest.raises(RuntimeError, match=r"One game variant at a time"):
        claim_game_variant(config.BUGGED_GAME_VARIANT)
    with pytest.raises(ValueError):
        claim_game_variant("mario_clone")


# ── behaviour ────────────────────────────────────────────────────────────────
def _probe(variant, switch_from=None):
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", PYTHONUTF8="1")
    result = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "variant_probe.py"),
                             variant, *([switch_from] if switch_from else [])], cwd=ROOT, env=env, capture_output=True, text=True,
                            timeout=300, check=False)
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def fingerprints():
    return {v: _probe(v) for v in config.GAME_VARIANTS}


def test_each_variant_loads_from_its_own_directory(fingerprints):
    for v, fp in fingerprints.items():
        assert fp["loaded_from"] == v


def test_the_clean_game_still_draws_the_pre_objective_3_frames(fingerprints):
    from test_hud_timer import LEGACY_NOOP_600_SHA256
    assert fingerprints[config.CLEAN_GAME_VARIANT]["noop600"] == LEGACY_NOOP_600_SHA256


def test_a_switched_game_is_exactly_the_freshly_loaded_game(fingerprints):
    """The dashboard's game switch: load one variant, release it, load the
    other in the SAME process. The result must be the other variant, byte for
    byte what a fresh process plays - nothing of the first one survives."""
    clean, bugged = config.CLEAN_GAME_VARIANT, config.BUGGED_GAME_VARIANT
    for target, source in ((bugged, clean), (clean, bugged)):
        switched = _probe(target, switch_from=source)
        assert switched["loaded_from"] == target
        assert switched["switched_from"] == source
        for key in ("noop600", "geometry", "scripted_state", "scripted_frames",
                    "scripted_episode_ends"):
            assert switched[key] == fingerprints[target][key], f"{key}: {source} -> {target}"


# The declared bugs' collider changes, and nothing else, in level geometry.
EXPECTED_GEOMETRY_CHANGES = {
    "removed": {("pipe_group", 2445, 366, 83, 170),        # pipe-clip
                ("step_group", 5874, 366, 40, 176),        # stair-clip (step4)
                ("step_group", 6001, 366, 40, 176)},       # stair-clip (step5)
    "added": {("pipe_group", 2445, 366, 21, 170),
              ("step_group", 5874, 452, 40, 90),
              ("step_group", 6001, 452, 40, 90),
              ("step_group", 4412, 452, 40, 86)},          # invisible-wall
}


def test_the_level_geometry_differs_only_in_the_declared_colliders(fingerprints):
    clean, bugged = ({tuple(r) for r in fingerprints[v]["geometry_rects"]}
                     for v in config.GAME_VARIANTS)
    assert clean - bugged == EXPECTED_GEOMETRY_CHANGES["removed"]
    assert bugged - clean == EXPECTED_GEOMETRY_CHANGES["added"]


def test_outside_the_declared_bugs_both_variants_behave_identically(fingerprints):
    """Same NOOP frames at the start, and the same scripted run frame for
    frame until the two games first differ - which must happen inside a
    declared bug's zone (today: a jump taken inside the open-sky-jump zone)."""
    clean, bugged = (fingerprints[v] for v in config.GAME_VARIANTS)
    assert clean["noop600"] == bugged["noop600"]
    a, b = clean["scripted_per_substep"], bugged["scripted_per_substep"]
    first = next((i for i in range(len(a)) if a[i] != b[i]), len(a))
    assert first > 100, f"the games diverged after only {first} frames"
    assert a[:first] == b[:first]
    rect = a[first - 1][0]                       # where Mario was when they parted
    zones = {bug["id"]: bug["zone"]["x"] for bug in variants.injected_bugs(config.BUGGED_GAME_VARIANT)}
    inside = [bid for bid, (lo, hi) in zones.items() if lo <= rect[0] + rect[2] and rect[0] <= hi]
    assert inside, f"the games first differ at x {rect[0]}, outside every declared bug zone"
    frames_before = [f for f in clean["scripted_per_frame"] if f[0] < first]
    assert frames_before and frames_before == bugged["scripted_per_frame"][:len(frames_before)]
