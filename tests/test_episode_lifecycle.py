"""The episode lifecycle: the engine timer, EXPLORE -> COMPLETE, and endings.

Two kinds of test live here, on purpose:

  * The transition and safety-reset LOGIC is driven with synthetic positions
    through EpisodeLifecycle directly. "Walk in a perfectly straight line for
    2,000 agent steps finding nothing" is then an exact statement rather than
    something the real level may or may not allow.
  * The TIMER is tested on the real engine, because the whole point of the
    fix is what the real engine does. The two that have to play out a full
    episode are marked slow.

Letters in brackets are the brief's test IDs (section 9).
"""

import inspect
import math

import numpy as np
import pytest

from agent_logic import GlitchHunterWrapper
from exploration import config
from exploration.coverage import SpatialCoverage
from exploration.lifecycle import (EndReason, EpisodeLifecycle, EpisodePhase,
                                   Transition, classify_end)

SPS = config.SUBSTEPS_PER_AGENT_STEP
WINDOW = config.LIFECYCLE_WINDOW * SPS          # substeps per window


class FakeCoverage:
    """Just the surface the lifecycle reads, fully controlled.

    `testable=None` switches the frontier half of the transit test off, so a
    test that is about straightness is not also silently about the frontier.
    """

    def __init__(self, target=10_000, frontier=None):
        self.episode_new = 0
        self._target = target
        self.testable = None if frontier is None else True
        self._frontier = frontier          # fn(viewport_x, points) -> (n, d)

    def episode_target(self):
        return self._target

    def frontier_query(self, viewport_x, points):
        return self._frontier(viewport_x, points)


def _info(x, y=498, **kw):
    return {'mario_rect': (int(x), int(y), 30, 40), 'viewport_x': 0, **kw}


def _walk_straight(lc, substeps, x0=100.0, speed=2.0, n_new=0):
    """A coherent traverse: net displacement == path length."""
    x = x0
    for _ in range(substeps):
        x += speed
        lc.observe(_info(x), n_new)
    return x


def _pace(lc, substeps, x=400, amplitude=20, n_new=0):
    """Back and forth in a tiny box - the shape of genuine stuck."""
    for i in range(substeps):
        lc.observe(_info(x + (amplitude if (i // 30) % 2 else 0)), n_new)


# ══════════════════════════════════════════════════════════════════════════
# THE ENGINE TIMER — root cause and fix
# ══════════════════════════════════════════════════════════════════════════
def test_limit_hierarchy_is_consistent():
    """Exactly one authoritative clock; every other limit knows its place.

    The bug this pins: TimeLimit was 4000 while the engine timed out at 2,451,
    and the safety floor was 5,000 - two limits that could never fire, and
    nothing noticed.
    """
    qa_cap = config.QA_EPISODE_CAP_AGENT_STEPS_MEASURED
    legacy_cap = config.ENGINE_CAP_AGENT_STEPS_MEASURED
    # TimeLimit is a backstop: above the timer, so it never pre-empts it.
    assert qa_cap < config.QA_EPISODE_MAX_STEPS
    assert legacy_cap < config.LEGACY_EPISODE_MAX_STEPS
    # The safety floor is REACHABLE within a QA episode...
    assert qa_cap > config.SAFETY_MIN_EPISODE_STEPS, (
        f"safety floor {config.SAFETY_MIN_EPISODE_STEPS} is at or above the "
        f"{qa_cap}-step QA cap - it is dead code")
    # ...and was not at the old cap, which is why the timer had to move.
    assert legacy_cap < config.SAFETY_MIN_EPISODE_STEPS
    # T4 is a backstop inside the episode, per the brief: 0.75 x the cap.
    assert int(0.75 * qa_cap) == config.MAX_EXPLORE_STEPS
    assert config.QA_EPISODE_TIME_UNITS > config.ENGINE_TIME_UNITS_DEFAULT


def test_retired_drought_limit_is_gone():
    assert not hasattr(config, 'DROUGHT_HARD_LIMIT'), (
        "DROUGHT_HARD_LIMIT ended episodes on a bare drought and was retired; "
        "something has reintroduced it")


def test_make_env_timelimit_follows_the_mode():
    """TimeLimit is chosen per mode, in the parent, from config."""
    import train_agent
    for mode, want in (("qa_exploration", config.QA_EPISODE_MAX_STEPS),
                       ("legacy_completion", config.LEGACY_EPISODE_MAX_STEPS)):
        init = train_agent.make_env(0, reward_mode=mode)
        got = inspect.getclosurevars(init).nonlocals
        assert got['max_steps'] == want, f"{mode}: TimeLimit {got['max_steps']}"
        assert got['skip'] == config.SUBSTEPS_PER_AGENT_STEP


def test_qa_mode_sets_the_engine_timer(fresh):
    w = GlitchHunterWrapper(fresh, reward_mode="qa_exploration",
                            coverage=SpatialCoverage(
                                testable_mask=np.ones((config.GRID_H, config.GRID_W), bool)))
    w.reset()
    info = w.step(0)[4]
    assert info['time_left'] == config.QA_EPISODE_TIME_UNITS
    assert fresh.end_on_level_complete is True


def test_legacy_mode_keeps_the_engine_timer_even_after_qa(fresh):
    """The leak guard. One env is shared across the whole test session (and
    could be re-wrapped in the dashboard): a QA wrapper must not leave its
    budget behind for a legacy wrapper to inherit."""
    GlitchHunterWrapper(fresh, reward_mode="qa_exploration",
                        coverage=SpatialCoverage(
                            testable_mask=np.ones((config.GRID_H, config.GRID_W), bool)))
    w = GlitchHunterWrapper(fresh, reward_mode="legacy_completion",
                            attach_coverage=False)
    w.reset()
    info = w.step(0)[4]
    assert info['time_left'] == config.ENGINE_TIME_UNITS_DEFAULT
    assert fresh.end_on_level_complete is False
    assert w.lifecycle is None
    assert 'episode_phase' not in info, "legacy mode grew a lifecycle"


def _hold_noop_until_done(env, units):
    env.episode_time_units = units
    env.end_on_level_complete = False
    env.reset()
    sub = 0
    while True:
        _o, _r, done, _t, info = env.step(0)
        sub += 1
        if done:
            return sub, info


@pytest.mark.slow
def test_measured_timer_caps(env):
    """The root cause, reproduced. 401 units x ~24.45 substeps = 9806 =
    2,451 agent steps; 1600 units = 39,126 = 9,781. If either number moves,
    the engine's clock changed and every limit above it must be re-derived."""
    try:
        sub, info = _hold_noop_until_done(env, None)
        assert sub // SPS == config.ENGINE_CAP_AGENT_STEPS_MEASURED
        assert info['death_cause'] == 'timeout'
        sub, info = _hold_noop_until_done(env, config.QA_EPISODE_TIME_UNITS)
        assert sub // SPS == config.QA_EPISODE_CAP_AGENT_STEPS_MEASURED
        assert info['death_cause'] == 'timeout'
    finally:
        env.episode_time_units = None


def _to_castle(env, wrapper=None):
    """Teleport next to the flagpole and hop onto its base block."""
    (wrapper or env).reset()
    env.game.state.mario.rect.x = 8400
    sub, castle, info = 0, None, {}
    step = (wrapper or env).step
    while sub < 3000:
        _o, _r, done, _t, info = step(2)
        sub += 1
        if castle is None and info.get('flag_get'):
            castle = sub
        if done:
            return castle, sub, info
    raise AssertionError("never finished the level")


def test_qa_episode_ends_at_the_castle_door(fresh):
    """No dead countdown after completion. At the door the engine burns the
    remaining budget one unit per frame; at 1600 units that was 429 agent
    steps in which no action mattered."""
    w = GlitchHunterWrapper(fresh, reward_mode="qa_exploration",
                            coverage=SpatialCoverage(
                                testable_mask=np.ones((config.GRID_H, config.GRID_W), bool)))
    castle, done_at, info = _to_castle(fresh, w)
    assert done_at == castle, f"{done_at - castle} substeps ran after the door"
    assert info['episode_end_reason'] == EndReason.LEVEL_COMPLETE
    assert info['time_left'] > 0


def test_legacy_episode_keeps_the_victory_sequence(fresh):
    """Legacy is what the 6M brain was trained in: it keeps the full tail."""
    fresh.episode_time_units = None
    fresh.end_on_level_complete = False
    castle, done_at, _info = _to_castle(fresh)
    assert done_at - castle > 100


# ══════════════════════════════════════════════════════════════════════════
# EXPLORE -> COMPLETE
# ══════════════════════════════════════════════════════════════════════════
def test_episode_starts_in_explore():                               # [A]
    lc = EpisodeLifecycle(FakeCoverage())
    assert lc.phase is EpisodePhase.EXPLORE
    lc.observe(_info(120), 50)
    assert lc.phase is EpisodePhase.EXPLORE
    lc.phase = EpisodePhase.COMPLETE
    lc.begin_episode()
    assert lc.phase is EpisodePhase.EXPLORE, "a new episode began in COMPLETE"


def test_first_step_is_explore_on_the_real_map(fresh):              # [A]
    """On the map the QA run actually starts from, not a blank one."""
    import os
    if not os.path.exists(config.BOOTSTRAP_COVERAGE):
        pytest.skip("bootstrap not generated")
    from exploration.coverage import load_testable
    cov = SpatialCoverage(testable_mask=load_testable())
    cov.load(config.BOOTSTRAP_COVERAGE)
    w = GlitchHunterWrapper(fresh, reward_mode="qa_exploration", coverage=cov)
    w.reset()
    info = w.step(0)[4]
    assert info['episode_phase'] == EpisodePhase.EXPLORE.value
    assert info['phase_transition_reason'] is None


def test_meeting_the_target_moves_to_complete():                    # [E]
    cov = FakeCoverage(target=1000)
    lc = EpisodeLifecycle(cov)
    cov.episode_new = 999
    lc.observe(_info(200), 5)
    assert lc.phase is EpisodePhase.EXPLORE
    cov.episode_new = 1000
    lc.observe(_info(202), 1)
    assert lc.phase is EpisodePhase.COMPLETE
    assert lc.transition_reason == Transition.TARGET_MET


def test_transition_is_one_way():
    cov = FakeCoverage(target=10)
    lc = EpisodeLifecycle(cov)
    cov.episode_new = 10
    lc.observe(_info(200), 10)
    assert lc.phase is EpisodePhase.COMPLETE
    step = lc.transition_step
    cov.episode_new = 0
    _walk_straight(lc, WINDOW * 3, n_new=500)
    assert lc.phase is EpisodePhase.COMPLETE
    assert lc.transition_step == step, "the transition was re-recorded"


def test_sustained_exhaustion_moves_to_complete():                  # [F]
    lc = EpisodeLifecycle(FakeCoverage())
    _pace(lc, WINDOW * (config.YIELD_WINDOWS - 1), n_new=0)
    assert lc.phase is EpisodePhase.EXPLORE, "T2 fired a window early"
    _pace(lc, WINDOW, n_new=0)
    assert lc.phase is EpisodePhase.COMPLETE
    assert lc.transition_reason == Transition.YIELD_EXHAUSTED


def test_transit_drought_does_not_transition():                     # [G]
    """Crossing covered ground finds nothing - and is exactly the job."""
    lc = EpisodeLifecycle(FakeCoverage())
    _walk_straight(lc, WINDOW * config.YIELD_WINDOWS * 2, n_new=0)
    assert lc.phase is EpisodePhase.EXPLORE
    assert lc.last_window['in_transit'] is True
    assert lc.last_window['straightness'] > 0.8
    assert lc.consecutive_stuck == 0


def test_closing_on_the_frontier_counts_as_transit():               # [G]
    """A zig-zag has low straightness, but if it is closing on unexplored
    space it is transit. The frontier gain is the stronger signal."""
    goal = np.array([50_000.0, 498.0])

    def frontier(_vx, pts):
        pts = np.asarray(pts, float)
        return 7, np.hypot(*(pts - goal).T)
    lc = EpisodeLifecycle(FakeCoverage(frontier=frontier))
    x = 400.0
    for i in range(WINDOW * config.YIELD_WINDOWS * 2):
        x += 3.0 if (i // 40) % 3 else -2.5        # net progress, lots of doubling back
        lc.observe(_info(x, y=498 + 30 * ((i // 20) % 2)), 0)
    assert lc.last_window['straightness'] < config.TRANSIT_STRAIGHTNESS
    assert lc.last_window['frontier_gain'] > config.TRANSIT_FRONTIER_GAIN_PX
    assert lc.phase is EpisodePhase.EXPLORE


def test_nothing_left_ahead_moves_to_complete():                    # T3
    lc = EpisodeLifecycle(FakeCoverage(frontier=lambda _vx, pts: (0, np.full(len(pts), np.inf))))
    _walk_straight(lc, WINDOW, n_new=0)
    assert lc.phase is EpisodePhase.COMPLETE
    assert lc.transition_reason == Transition.NOTHING_AHEAD


def test_explore_backstop_fires_only_at_its_limit():                # T4
    lc = EpisodeLifecycle(FakeCoverage())
    # Coherent transit the whole way, so nothing but T4 can fire.
    _walk_straight(lc, config.MAX_EXPLORE_STEPS * SPS - 1, speed=0.3, n_new=0)
    assert lc.phase is EpisodePhase.EXPLORE
    lc.observe(_info(9000), 0)
    assert lc.phase is EpisodePhase.COMPLETE
    assert lc.transition_reason == Transition.EXPLORE_BACKSTOP
    assert lc.transition_step == config.MAX_EXPLORE_STEPS


def test_phase_is_in_info_on_every_substep(qa_env):
    """MaxAndSkipObservation keeps only the last of four info dicts, so the
    phase has to be on all of them or it is lost three times in four."""
    from gymnasium.wrappers import MaxAndSkipObservation
    w, drive = qa_env
    skip = MaxAndSkipObservation(w, skip=SPS)
    drive(x=500)
    for _ in range(5):
        info = skip.step(0)[4]
        assert info['episode_phase'] in ('explore', 'complete')
        assert 'phase_transition_reason' in info


def test_phase_does_not_yet_affect_reward(qa_env):
    """Reward-neutral, for now, by construction. The same substeps return the
    same reward whether the episode is in EXPLORE or COMPLETE.

    Phase-gated reward is a LATER phase and will change this deliberately -
    when it does, this test should be replaced, not loosened."""
    w, drive = qa_env
    drive(x=800)
    w.lifecycle.phase = EpisodePhase.EXPLORE
    a = [drive(x=800)[1] for _ in range(20)]
    w.lifecycle.phase = EpisodePhase.COMPLETE
    b = [drive(x=800)[1] for _ in range(20)]
    assert a == b


def test_reaching_complete_does_not_end_the_episode(qa_env):
    w, drive = qa_env
    # A blank map: the first substep finds a whole collider's worth of new
    # pixels (1,200), above the 500-px floor target.
    assert w.lifecycle.target == config.TARGET_FLOOR
    _obs, _r, done, _t, info = drive(x=900)
    assert info['episode_phase'] == 'complete'
    assert info['phase_transition_reason'] == Transition.TARGET_MET
    assert not done, "the phase transition terminated the episode"


# ══════════════════════════════════════════════════════════════════════════
# SAFETY RESET — needs all three
# ══════════════════════════════════════════════════════════════════════════
def _stuck_for(lc, substeps):
    _pace(lc, substeps, n_new=0)


def test_safety_reset_needs_all_three():                            # [L]
    floor = config.SAFETY_MIN_EPISODE_STEPS * SPS

    # Stuck and in drought, but before the floor.
    lc = EpisodeLifecycle(FakeCoverage())
    _stuck_for(lc, floor - SPS)
    assert lc.consecutive_stuck >= config.SAFETY_STUCK_WINDOWS
    assert lc.drought_agent_steps() > config.SAFETY_DROUGHT_STEPS
    assert not lc.safety_reset_due(), "fired below the floor"

    # Past the floor and stuck, but a recent discovery.
    lc = EpisodeLifecycle(FakeCoverage())
    _stuck_for(lc, floor)
    lc.observe(_info(400), config.SAFETY_MEANINGFUL_NEW_PX)
    # +1: the window containing the discovery is not a stuck window.
    _stuck_for(lc, WINDOW * (config.SAFETY_STUCK_WINDOWS + 1))
    assert lc.agent_steps > config.SAFETY_MIN_EPISODE_STEPS
    assert lc.consecutive_stuck >= config.SAFETY_STUCK_WINDOWS
    assert lc.drought_agent_steps() <= config.SAFETY_DROUGHT_STEPS
    assert not lc.safety_reset_due(), "fired without a long drought"

    # All three.
    lc = EpisodeLifecycle(FakeCoverage())
    _stuck_for(lc, floor + SPS)
    assert lc.safety_reset_due()


def test_drought_alone_cannot_reset():                              # [L]
    """Past the floor, in a 1,500+ step drought - but moving coherently, so
    no window is stuck and the reset cannot fire."""
    lc = EpisodeLifecycle(FakeCoverage())
    _walk_straight(lc, (config.SAFETY_MIN_EPISODE_STEPS + 1000) * SPS,
                   speed=0.3, n_new=0)
    assert lc.drought_agent_steps() > config.SAFETY_DROUGHT_STEPS
    assert lc.consecutive_stuck == 0
    assert not lc.safety_reset_due()


def test_step_count_alone_cannot_reset():                           # [M]
    lc = EpisodeLifecycle(FakeCoverage())
    for i in range((config.SAFETY_MIN_EPISODE_STEPS + 2000) * SPS):
        lc.observe(_info(100 + 0.3 * i), 0)
        assert not lc.safety_reset_due(), f"fired at agent step {lc.agent_steps}"


def test_thresholds_are_configurable(monkeypatch):                 # [N]
    monkeypatch.setattr(config, 'SAFETY_MIN_EPISODE_STEPS', 100)
    monkeypatch.setattr(config, 'SAFETY_DROUGHT_STEPS', 50)
    monkeypatch.setattr(config, 'SAFETY_STUCK_WINDOWS', 1)
    lc = EpisodeLifecycle(FakeCoverage())
    _stuck_for(lc, WINDOW + 4)
    assert lc.safety_reset_due(), "config change did not change behaviour"

    monkeypatch.setattr(config, 'YIELD_WINDOWS', 1)
    lc = EpisodeLifecycle(FakeCoverage())
    _pace(lc, WINDOW)
    assert lc.transition_reason == Transition.YIELD_EXHAUSTED


def test_safety_reset_ends_the_episode_through_the_wrapper(qa_env, monkeypatch):
    """Wired end to end: fires, charges the existing terminal term, and is
    reported as its own ending - not as a death."""
    monkeypatch.setattr(config, 'SAFETY_MIN_EPISODE_STEPS', 10)
    monkeypatch.setattr(config, 'SAFETY_DROUGHT_STEPS', 5)
    monkeypatch.setattr(config, 'SAFETY_STUCK_WINDOWS', 1)
    w, drive = qa_env
    drive(x=600)
    done, info, n = False, {}, 0
    while not done and n < 5 * WINDOW:
        _obs, _r, done, _t, info = w.step(0)
        n += 1
    assert done, "the safety reset never ended the episode"
    assert info['episode_end_reason'] == EndReason.SAFETY_RESET


@pytest.mark.slow
def test_safety_floor_is_reachable_in_the_real_engine(fresh):
    """The claim this phase exists to make true. Mario stands still at spawn
    in the REAL engine: before the fix the engine timed him out at 2,451
    agent steps; now the safety reset ends it just past the 5,000 floor,
    with engine time still left on the clock."""
    w = GlitchHunterWrapper(fresh, reward_mode="qa_exploration",
                            coverage=SpatialCoverage(
                                testable_mask=np.ones((config.GRID_H, config.GRID_W), bool)))
    w.reset()
    done, info = False, {}
    while not done:
        _obs, _r, done, _t, info = w.step(0)
    steps = info['lifecycle_agent_steps']
    assert info['episode_end_reason'] == EndReason.SAFETY_RESET
    assert config.SAFETY_MIN_EPISODE_STEPS < steps <= config.SAFETY_MIN_EPISODE_STEPS + 1
    assert info['time_left'] > 0, "the engine timer got there first"


# ══════════════════════════════════════════════════════════════════════════
# HOW EPISODES END — a taxonomy, not a single "done"
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("info,safety,want", [
    ({'flag_get': True}, False, EndReason.LEVEL_COMPLETE),
    ({'death_cause': 'timeout', 'is_dead': True}, False, EndReason.TIMEOUT),
    ({'death_cause': 'goomba', 'is_dead': True}, False, EndReason.DEATH),
    ({'death_cause': 'pit', 'is_dead': True}, False, EndReason.DEATH),
    ({}, True, EndReason.SAFETY_RESET),
    ({}, False, EndReason.ENGINE_DONE),
    # An engine ending outranks the reset on a shared substep.
    ({'death_cause': 'pit', 'is_dead': True}, True, EndReason.DEATH),
])
def test_end_reasons(info, safety, want):
    assert classify_end(info, safety) == want


def test_timeout_is_not_reported_as_death():
    """The engine reports a timeout as a death (death_cause 'timeout'). The
    lifecycle keeps them apart - running out of time is not dying to the
    level, and the two need different fixes."""
    assert classify_end({'death_cause': 'timeout', 'is_dead': True}, False) \
        != EndReason.DEATH


# ══════════════════════════════════════════════════════════════════════════
@pytest.fixture
def qa_env(env, patch_step):
    """QA wrapper over the shared env, driven with synthetic infos."""
    cov = SpatialCoverage(
        testable_mask=np.ones((config.GRID_H, config.GRID_W), dtype=bool))
    w = GlitchHunterWrapper(env, reward_mode="qa_exploration", coverage=cov)
    w.reset()
    base = {
        'x_pos': 500, 'y_pos': 498, 'x_vel': 0.0, 'on_ground': True,
        'status': 'small', 'flag_get': False, 'powerup_active_count': 0,
        'nearest_powerup_dx': None, 'death_cause': None, 'is_dead': False,
        'score': 0, 'coins': 0, 'viewport_x': 0,
    }

    obs = np.zeros(env.observation_space.shape, dtype=np.uint8)

    def drive(x):
        info = dict(base, x_pos=x, mario_rect=(int(x), 498, 30, 40))
        patch_step(lambda a, _i=info: (obs, 0.0, False, False, dict(_i)))
        return w.step(0)
    yield w, drive
    # Leave the shared env as a fresh wrapper would find it.
    env.episode_time_units = None
    env.end_on_level_complete = False


def test_math_helpers_sanity():
    """straightness of a straight line is 1; of a round trip is ~0."""
    lc = EpisodeLifecycle(FakeCoverage())
    _walk_straight(lc, WINDOW)
    assert math.isclose(lc.last_window['straightness'], 1.0, rel_tol=1e-6)
    lc = EpisodeLifecycle(FakeCoverage())
    for i in range(WINDOW):
        lc.observe(_info(400 + (i if i < WINDOW // 2 else WINDOW - i)), 0)
    assert lc.last_window['straightness'] < 0.01
