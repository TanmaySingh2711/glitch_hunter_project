"""The two game variants: mario_clean is the untouched baseline, mario_bugged
is where deliberate bugs will go - and today there are none.

Three kinds of proof, strongest last:
  * content  - clean matches its pin; bugged differs from clean nowhere a
               declared bug does not name (and none is declared)
  * loading  - the env runs the variant it was asked for, from that
               directory, and refuses to mix two variants in one process
  * behaviour - each variant, in its own fresh process, plays the same
               scripted run: observations, physics, collision, timing,
               rendering and level geometry all fingerprint identically, and
               the clean game still draws the pre-Objective-3 NOOP frames
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


def test_no_bug_has_been_injected_yet():
    assert variants.injected_bugs(config.BUGGED_GAME_VARIANT) == [], (
        "a bug was declared in mario_bugged/INJECTED_BUGS.json; update this test "
        "only when the project owner has specified that bug")
    assert variants.game_tree_files(variants.game_dir(config.BUGGED_GAME_VARIANT)) == \
        variants.game_tree_files(variants.game_dir(config.CLEAN_GAME_VARIANT))


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
    manifest.write_text(json.dumps({"bugs": [{"id": "demo", "files": ["data/constants.py"]}]}),
                        encoding="utf-8")
    assert variants.undeclared_differences(config.BUGGED_GAME_VARIANT) == []
    ident = variants.variant_identity(config.BUGGED_GAME_VARIANT)
    assert ident["declared_injected_bugs"] == ["demo"]
    assert not ident["matches_pinned_clean_tree"]


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
    with pytest.raises(RuntimeError, match=r"One game variant per process"):
        claim_game_variant(config.BUGGED_GAME_VARIANT)
    with pytest.raises(ValueError):
        claim_game_variant("mario_clone")


# ── behaviour ────────────────────────────────────────────────────────────────
def _probe(variant):
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", PYTHONUTF8="1")
    result = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "variant_probe.py"),
                             variant], cwd=ROOT, env=env, capture_output=True, text=True,
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


def test_both_variants_behave_identically(fingerprints):
    clean, bugged = (fingerprints[v] for v in config.GAME_VARIANTS)
    for key in ("noop600", "geometry", "scripted_state", "scripted_frames",
                "scripted_episode_ends"):
        assert clean[key] == bugged[key], f"{key} differs between the two games"
