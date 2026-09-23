"""The engine half of Objective 3, on the real game.

  * OBSERVING CHANGES NOTHING: an episode played with evidence on is
    frame-for-frame and info-for-info the episode played with it off.
  * THE TRIGGER FRAME IS THE TRIGGER'S: the Detection holds exactly the frame
    the engine drew on the substep the detector fired - not the one before,
    not the one after, and not the dashboard's later stream frame.
  * THE RECORD DESCRIBES ONE EPISODE: the trace and the action log end at the
    trigger and never reach back past the reset.
"""
import hashlib

import numpy as np
import pygame as pg
import pytest

from custom_mario_env import SkipObservation
from exploration import config
from reporting.events import SyntheticProbe

SCRIPT = [3] * 60 + [4] * 20 + [3] * 40 + [5] * 10 + [0] * 10 + [8] * 15 + [3] * 60


def _surface():
    return np.ascontiguousarray(pg.surfarray.array3d(pg.display.get_surface()).transpose(1, 0, 2))


def _run(env, script):
    env.reset()
    obs_hash, infos = hashlib.sha256(), []
    for a in script:
        obs, r, done, trunc, info = env.step(a)
        obs_hash.update(obs.tobytes())
        infos.append((sorted(info), repr(sorted(info.items())), r, done, trunc))
        if done:
            env.reset()
    return obs_hash.hexdigest(), infos


def test_evidence_changes_neither_observations_nor_info(env):
    off = _run(env, SCRIPT)
    env.enable_evidence()
    env.add_detector(SyntheticProbe(10_000))           # a detector that never fires here
    on = _run(env, SCRIPT)
    assert on[0] == off[0], "observations differ with evidence on"
    assert on[1] == off[1], "info dicts, rewards or episode ends differ with evidence on"


def test_evidence_is_off_by_default_and_costs_nothing(fresh):
    assert fresh.evidence_enabled is False
    for _ in range(40):
        fresh.step(3)
    assert fresh.drain_detections() == [] and fresh.last_trace_entry() is None


def test_the_trigger_frame_is_exactly_the_frame_the_detector_fired_on(env):
    env.enable_evidence()
    env.add_detector(SyntheticProbe(700))
    env.reset()
    frames = []
    for _ in range(400):
        env.step(3)
        frames.append(_surface())
        found = env.drain_detections()
        if found:
            break
    else:
        pytest.fail("the probe never fired")
    det = found[0]
    i = len(frames) - 1
    assert det.episode_substep == i
    assert np.array_equal(det.frame, frames[i]), "not the frame drawn on the trigger substep"
    assert not np.array_equal(det.frame, frames[i - 1]), "the frame before would also pass"
    env.step(3)
    assert not np.array_equal(det.frame, _surface()), "the frame after would also pass"
    assert det.frame.shape == (600, 800, 3)


def test_the_dashboard_stream_frame_is_not_the_trigger_frame(env):
    """At the agent-step level the stream frame is drawn after all 4 substeps,
    so it trails the trigger by 3 - k frames. The Detection must not."""
    env.enable_evidence()
    env.add_detector(SyntheticProbe(700))
    skip = SkipObservation(env, skip=config.SUBSTEPS_PER_AGENT_STEP)
    skip.reset()
    for _ in range(200):
        skip.step(3)
        found = env.drain_detections()
        if found:
            break
    det = found[0]
    k = det.episode_substep % config.SUBSTEPS_PER_AGENT_STEP
    stream = _surface()
    if k < config.SUBSTEPS_PER_AGENT_STEP - 1:
        assert not np.array_equal(det.frame, stream)
    else:
        assert np.array_equal(det.frame, stream)       # trigger on the last substep: same frame


def test_the_detection_describes_the_trigger_substep(env):
    env.enable_evidence()
    env.add_detector(SyntheticProbe(800))      # before the first goomba (x ~890) can end it
    env.reset()
    info = None
    for _ in range(400):
        *_, info = env.step(3)
        found = env.drain_detections()
        if found:
            break
    det = found[0]
    assert det.kind == "synthetic_probe" and det.synthetic is True
    assert det.detector == "synthetic_probe/x>=800"
    assert list(det.state["mario_rect"]) == list(info["mario_rect"])
    assert det.state["mario_rect"][0] + det.state["mario_rect"][2] >= 800
    assert det.trace[-1]["substep"] == det.episode_substep
    assert det.trace[-1]["x"] == info["mario_rect"][0]
    assert len(det.actions) == det.episode_substep + 1 and det.actions_complete
    assert det.episode_index == env.episode_index
    assert det.engine_time_ms == pytest.approx((det.episode_substep + 1) * 1000.0 / 60.0)
    vx0 = info["viewport_x"]
    assert det.geometry and all(g["x"] + g["w"] >= vx0 and g["x"] <= vx0 + 800
                                for g in det.geometry)
    assert info["glitch_alert"].startswith("[SYNTHETIC TEST EVENT")


def test_the_context_window_is_bounded_and_never_crosses_a_reset(env):
    env.enable_evidence(context_substeps=50)
    env.add_detector(SyntheticProbe(800))
    env.reset()
    for _ in range(30):
        env.step(3)
    env.reset()                                        # a new episode starts the record over
    for _ in range(600):
        env.step(3)
        found = env.drain_detections()
        if found:
            break
    det = found[0]
    assert len(det.trace) == 50
    assert [t["substep"] for t in det.trace] == list(range(det.episode_substep - 49,
                                                           det.episode_substep + 1))
    assert det.trace[0]["substep"] > 0 or det.episode_substep < 50


def test_the_builtin_detector_keeps_its_once_per_episode_semantics(fresh):
    fresh.enable_evidence()
    fresh.reset()
    for _ in range(3):
        fresh.step(1)
    fresh.game.state.game_info['score'] = 5000
    fresh.step(1)
    fresh.game.state.game_info['score'] = 100
    fresh.step(1)
    fresh.game.state.game_info['score'] = 5000
    fresh.step(1)
    fresh.game.state.game_info['score'] = 10                   # a second drop, same episode
    for _ in range(4):
        fresh.step(1)
    found = fresh.drain_detections()
    assert [d.kind for d in found] == ["score_drop"]
    assert found[0].metrics == {"previous_score": 5000, "score": 100}
    assert found[0].detector == "engine_invariants/score_drop"
    assert found[0].synthetic is False


def test_evidence_switched_on_mid_episode_is_marked_not_replayable(fresh):
    for _ in range(20):
        fresh.step(3)
    fresh.enable_evidence()
    fresh.add_detector(SyntheticProbe(0))                 # fires on the next frame
    fresh.step(3)
    det = fresh.drain_detections()[0]
    assert det.actions_complete is False
    fresh.reset()
    fresh.step(3)
    assert fresh.drain_detections()[0].actions_complete is True


def test_an_undrained_backlog_is_capped_and_counted(fresh):
    fresh.enable_evidence()
    for x in range(fresh.MAX_PENDING_DETECTIONS + 5):
        fresh.add_detector(SyntheticProbe(-1000 - x))     # all fire on the first frame
    fresh.reset()
    fresh.step(0)
    assert len(fresh.drain_detections()) == fresh.MAX_PENDING_DETECTIONS
    assert fresh.dropped_detections == 5


def test_qa_clock_top_ups_are_recorded_for_the_replay(env):
    from agent_logic import GlitchHunterWrapper
    from exploration.coverage import SpatialCoverage
    w = GlitchHunterWrapper(env, reward_mode="qa_exploration", coverage=SpatialCoverage(
        testable_mask=np.ones((config.GRID_H, config.GRID_W), dtype=bool)))
    assert env.holds_engine_clock is True
    env.enable_evidence()
    env.add_detector(SyntheticProbe(0))
    w.reset()
    hud = env.game.state.overhead_info_display
    hud.time = 1                                          # the next substep needs a top-up
    w.step(0)
    det = env.drain_detections()[0]
    assert det.clock_holds == ((0, config.QA_EPISODE_TIME_UNITS),)
    assert det.holds_engine_clock is True
    assert det.episode_time_units == config.QA_EPISODE_TIME_UNITS
