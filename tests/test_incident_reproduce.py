"""Reproduction must be honest in both directions: a real replay of a real
incident says "reproduced", and every way the evidence can fail to support
that - a tampered log, a changed game, a detector that stays quiet, a state
that drifts - says so instead.

Each case runs the real replay process on a real bundle captured from the
game (reporting/reproduce.py), so these are slower than the rest of the
suite: roughly a second per replay.
"""
import json
import os
import shutil
import stat

import pytest

from exploration import config
from reporting.events import SyntheticProbe
from reporting.pipeline import IncidentPipeline, SessionRecorder
from reporting.provenance import session_provenance
from reporting.reproduce import run_reproduction
from reporting.store import IncidentStore


def _capture_real(env, store, probe_x=700, mutate=None):
    """Plays the real game to a trigger and captures it (no rendering)."""
    env.enable_evidence()
    if probe_x is not None:
        env.add_detector(SyntheticProbe(probe_x))
    env.reset()
    rec = SessionRecorder()
    rec.begin_episode(env.episode_index)
    prov = session_provenance(None, "legacy_completion", config.CLEAN_GAME_VARIANT)
    pipe = IncidentPipeline(store, reproduce=False, start_worker=False)
    for i in range(400):
        if mutate is not None:
            mutate(env, i)
        for _ in range(config.SUBSTEPS_PER_AGENT_STEP):
            env.step(3)
        rec.step_taken(3)
        found = env.drain_detections()
        if found:
            out = pipe.capture(found[0], rec.context("legacy_completion", prov))
            assert out.status == "new", out.error
            return store.bundle_dir(out.incident_id)
    pytest.fail("no detection")


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, env_module):
    store = IncidentStore(str(tmp_path_factory.mktemp("incidents")))
    return _capture_real(env_module, store)


@pytest.fixture(scope="module")
def env_module(request):
    # The session env, borrowed for this module's own capture.
    env = request.getfixturevalue("env")
    yield env
    env.disable_evidence()


def _editable_copy(bundle, tmp_path):
    dst = tmp_path / os.path.basename(bundle)
    shutil.copytree(bundle, dst)
    for name in os.listdir(dst):
        os.chmod(dst / name, stat.S_IREAD | stat.S_IWRITE)
    return dst


def _edit_json(path, fn):
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    fn(doc)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)


def test_a_real_incident_reproduces_exactly(bundle):
    result = run_reproduction(bundle)
    assert result["status"] == "reproduced", result
    assert result["state_match"] and result["frame_match"]
    assert result["first_divergence"] is None
    assert result["compared_frames"] >= 1
    assert result["elapsed_s"] > 0


def test_a_tampered_action_log_is_not_replayed(bundle, tmp_path):
    copy = _editable_copy(bundle, tmp_path)
    def tamper(doc):
        doc["episode_actions"]["data"] = doc["episode_actions"]["data"][:-4] + "AAAA"
    _edit_json(copy / "trajectory.json", tamper)
    result = run_reproduction(str(copy))
    assert result["status"] == "not_possible" and "SHA-256" in result["detail"]


def test_a_recorded_state_the_replay_does_not_reach_is_divergence(bundle, tmp_path):
    copy = _editable_copy(bundle, tmp_path)
    target = {}
    def tamper(doc):
        entry = doc["trace"][len(doc["trace"]) // 2]
        entry["x"] += 7
        target["substep"] = entry["substep"]
    _edit_json(copy / "trajectory.json", tamper)
    result = run_reproduction(str(copy))
    assert result["status"] == "diverged"
    assert result["first_divergence"]["substep"] == target["substep"]
    assert result["first_divergence"]["field"] == "x"


def test_a_changed_game_is_not_replayed(bundle, tmp_path):
    copy = _editable_copy(bundle, tmp_path)
    _edit_json(copy / "incident.json",
               lambda d: d["provenance"]["game"].update(tree_sha256="0" * 64))
    result = run_reproduction(str(copy))
    assert result["status"] == "not_possible" and "changed" in result["detail"]


def test_an_episode_recorded_without_its_start_is_not_replayed(bundle, tmp_path):
    copy = _editable_copy(bundle, tmp_path)
    _edit_json(copy / "incident.json",
               lambda d: d["reproduction_inputs"].update(replayable=False, why_not="test"))
    assert run_reproduction(str(copy))["status"] == "not_possible"


def test_a_quiet_detector_on_an_identical_state_is_not_reproduced(bundle, tmp_path):
    copy = _editable_copy(bundle, tmp_path)
    _edit_json(copy / "incident.json", lambda d: d["detector"].update(id="engine_invariants/speed"))
    result = run_reproduction(str(copy))
    assert result["status"] == "not_reproduced"
    assert result["state_match"] is True


def test_different_pixels_on_an_identical_state_are_said_to_differ(bundle, tmp_path):
    copy = _editable_copy(bundle, tmp_path)
    _edit_json(copy / "incident.json",
               lambda d: d["evidence"]["trigger_frame"].update(pixel_sha256="0" * 64))
    result = run_reproduction(str(copy))
    assert result["status"] == "reproduced_state_only" and result["frame_match"] is False


def test_a_replay_that_runs_out_of_time_says_so(bundle):
    result = run_reproduction(bundle, timeout_s=0.01)
    assert result["status"] == "timeout"


def test_a_missing_bundle_is_an_error_not_a_crash(tmp_path):
    assert run_reproduction(str(tmp_path / "nothing"))["status"] == "error"


def test_an_anomaly_caused_outside_the_recorded_inputs_is_not_claimed_reproduced(
        env_module, tmp_path):
    """A score drop injected by editing engine memory is not in the action
    log, so a replay cannot recreate it - and must not pretend to."""
    def inject(env, i):
        if i == 5:
            env.game.state.game_info['score'] = 5000
        if i == 6:
            env.game.state.game_info['score'] = 100
    env_module.disable_evidence()
    bundle = _capture_real(env_module, IncidentStore(str(tmp_path / "inc")), probe_x=None,
                           mutate=inject)
    with open(os.path.join(bundle, "incident.json"), encoding="utf-8") as fh:
        assert json.load(fh)["detector"]["id"] == "engine_invariants/score_drop"
    result = run_reproduction(bundle)
    assert result["status"] in ("diverged", "not_reproduced"), result
    assert result["status"] != "reproduced"


# ── the same verdicts from the replay logic run in this process ─────────────
# (run_reproduction above runs it in a child process, which is how it is used;
#  these call replay() directly so each branch is also measured here)
def _tamper_log(doc):
    doc["episode_actions"]["data"] = doc["episode_actions"]["data"][:-4] + "AAAA"


def _tamper_trace(doc):
    doc["trace"][len(doc["trace"]) // 2]["x"] += 7


@pytest.mark.parametrize(("file", "edit", "expected"), [
    (None, None, "reproduced"),
    ("trajectory.json", _tamper_log, "not_possible"),
    ("trajectory.json", _tamper_trace, "diverged"),
    ("incident.json", lambda d: d["provenance"]["game"].update(tree_sha256="0" * 64),
     "not_possible"),
    ("incident.json", lambda d: d["reproduction_inputs"].update(replayable=False), "not_possible"),
    ("incident.json", lambda d: d["detector"].update(id="engine_invariants/speed"),
     "not_reproduced"),
    ("incident.json", lambda d: d["evidence"]["trigger_frame"].update(pixel_sha256="0" * 64),
     "reproduced_state_only"),
    ("incident.json", lambda d: d["reproduction_inputs"].update(
        extra_detectors=[{"type": "arbitrary_code"}]), "not_possible"),
    ("trajectory.json", lambda d: d["episode_actions"].update(substeps=1), "not_possible"),
])
def test_the_replay_verdicts_in_process(bundle, tmp_path, file, edit, expected):
    from reporting.reproduce import replay
    target = bundle
    if file is not None:
        target = _editable_copy(bundle, tmp_path)
        _edit_json(target / file, edit)
    assert replay(str(target))["status"] == expected


def test_the_replay_cli_writes_its_verdict(bundle, tmp_path):
    from reporting.reproduce import main
    out = tmp_path / "verdict.json"
    assert main([bundle, str(out)]) == 0
    with open(out, encoding="utf-8") as fh:
        assert json.load(fh)["status"] == "reproduced"
    assert main([]) == 2
