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

import hashlib
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

    def __init__(self, target=10_000, frontier=None, informed=True):
        self.episode_new = 0
        self._target = target
        self._informed = informed
        self.testable = None if frontier is None else True
        self._frontier = frontier          # fn(viewport_x, points) -> (n, d)

    def episode_target(self):
        return self._target

    def target_informed(self):
        return self._informed

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
    """TimeLimit is chosen per mode, in the parent, from config - and QA
    training has none: a step count alone may not end a QA episode."""
    import train_agent
    for mode, want in (("qa_exploration", None),
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
    offset = env.game.state.overhead_info_display.display_time_offset
    sub, frames = 0, []
    while True:
        o, _r, done, _t, info = env.step(0)
        sub += 1
        frames.append(hashlib.sha1(o.tobytes()).digest())
        # The TIME box and the clock the timeout reads, on every substep:
        # the clock itself in legacy; in QA the legacy-equivalent clock,
        # holding at 1 until the real clock is out (Phase 4D).
        t = info['time_left']
        want = t if offset == 0 or t <= 0 else max(1, t - offset)
        assert info['hud_time'] == want, f"substep {sub}"
        if done:
            return sub, info, frames


@pytest.mark.slow
def test_measured_timer_caps(env):
    """The root cause, reproduced. 401 units x ~24.45 substeps = 9806 =
    2,451 agent steps; 1600 units = 39,126 = 9,781. If either number moves,
    the engine's clock changed and every limit above it must be re-derived.

    The QA run is also the full-length check of the TIME box (Phase 4D): it
    never lets the timeout drift off the real clock, and for every substep
    legacy is alive, the QA frame IS the legacy frame, byte for byte."""
    try:
        legacy_sub, info, legacy_frames = _hold_noop_until_done(env, None)
        assert legacy_sub // SPS == config.ENGINE_CAP_AGENT_STEPS_MEASURED
        assert info['death_cause'] == 'timeout'
        assert info['hud_time'] == 0
        sub, info, qa_frames = _hold_noop_until_done(env, config.QA_EPISODE_TIME_UNITS)
        assert sub // SPS == config.QA_EPISODE_CAP_AGENT_STEPS_MEASURED
        assert info['death_cause'] == 'timeout'
        assert info['time_left'] == 0 and info['hud_time'] == 0
        alive = legacy_sub - 1          # legacy's last frame is its death
        first_diff = next((i for i in range(alive)
                           if qa_frames[i] != legacy_frames[i]), None)
        assert first_diff is None, f"QA frame differs from legacy at substep {first_diff + 1}"
    finally:
        env.episode_time_units = None


# ══════════════════════════════════════════════════════════════════════════
# THE RESPAWN RULE — time alone never ends a QA episode
# ══════════════════════════════════════════════════════════════════════════
def _qa_wrapper(env):
    return GlitchHunterWrapper(env, reward_mode="qa_exploration",
                               coverage=SpatialCoverage(
                                   testable_mask=np.ones((config.GRID_H, config.GRID_W), bool)))


def _hold_until(step, substeps):
    """NOOP for up to `substeps`; returns (done, info, substeps taken)."""
    info = {}
    for n in range(1, substeps + 1):
        _o, _r, done, _t, info = step(0)
        if done:
            return True, info, n
    return False, info, substeps


def test_qa_clock_running_out_does_not_end_the_episode(fresh):
    """The engine is one unit from timing out. In QA mode it must not: the
    clock is topped up, the TIME box keeps drawing 001, and Mario lives."""
    w = _qa_wrapper(fresh)
    w.reset()
    fresh.game.state.overhead_info_display.time = 2
    done, info, _n = _hold_until(w.step, 200)          # ~8 clock units
    assert not done, f"QA episode ended: {info.get('death_cause')}"
    assert info['death_cause'] is None
    assert info['clock_extensions'] >= 1
    assert info['time_left'] > 1
    assert info['hud_time'] == 1, "the TIME box moved off the legacy-equivalent 001"


def test_legacy_clock_still_times_out(fresh):
    """Legacy keeps the timeout the 6M brain was trained with."""
    w = GlitchHunterWrapper(fresh, reward_mode="legacy_completion", attach_coverage=False)
    w.reset()
    fresh.game.state.overhead_info_display.time = 2
    done, info, _n = _hold_until(w.step, 200)
    assert done and info['death_cause'] == 'timeout'


def test_the_bare_engine_still_times_out_on_the_qa_budget(fresh):
    """The completion evaluator plays the bare engine on the QA budget; its
    frozen protocol counts timeouts, so the hold must not reach it."""
    fresh.episode_time_units = config.QA_EPISODE_TIME_UNITS
    fresh.reset()
    fresh.game.state.overhead_info_display.time = 2
    done, info, _n = _hold_until(fresh.step, 200)
    assert done and info['death_cause'] == 'timeout'
    assert info['hud_time'] == 0


def test_the_hold_never_touches_the_castle_countdown(fresh):
    """At the castle door the engine converts the remaining time to score;
    topping the clock up there would pay out invented time."""
    fresh.episode_time_units = config.QA_EPISODE_TIME_UNITS
    fresh.reset()
    state = fresh.game.state
    state.overhead_info_display.time = 1
    state.mario.in_castle = True
    assert fresh.hold_clock() == 0
    state.mario.in_castle = False
    fresh.episode_time_units = None
    assert fresh.hold_clock() == 0, "held a clock with no QA budget (no display offset)"


@pytest.mark.slow
def test_productive_exploration_outlives_the_old_qa_clock(fresh, monkeypatch):
    """The conflict this rule resolves, on the real engine. Before it, every
    QA episode died with death_cause 'timeout' at agent step 9,781 however
    productive it was. Mario here keeps 'discovering' (the coverage reports a
    new pixel every substep) and must still be alive well past that point."""
    w = _qa_wrapper(fresh)
    monkeypatch.setattr(w.coverage, 'record', lambda obs: 1)
    w.reset()
    past = (config.QA_EPISODE_CAP_AGENT_STEPS_MEASURED + 250) * SPS
    done, info, n = _hold_until(w.step, past)
    assert not done, f"ended at substep {n}: {info.get('episode_end_reason')}"
    assert info['lifecycle_agent_steps'] > config.QA_EPISODE_CAP_AGENT_STEPS_MEASURED
    assert info['clock_extensions'] >= 1
    assert info['hud_time'] == 1


OLD_CLOCK = config.QA_EPISODE_CAP_AGENT_STEPS_MEASURED      # 9,781: a diagnostic only


def _wander(lc, substeps, n_new_every=None):
    """A genuine repeated local loop: roams a 400 x 100 px area, doubling back
    constantly - never a small box (so never STUCK), never straight enough to
    read as transit, never gaining ground. Continues from lc.substeps."""
    for _ in range(substeps):
        i = lc.substeps
        x = 400 + abs((i % 200) - 100) * 4
        y = 398 + abs((i % 130) - 65) * 100 // 65
        n_new = 1 if n_new_every and i % n_new_every == n_new_every - 1 else 0
        lc.observe(_info(x, y), n_new)


def _back_and_forth_transit(lc, windows):
    """Each window: walk 400 px one way, then stand. Every window reads as
    coherent transit (straightness 1.0), and finds nothing."""
    for k in range(windows):
        sign = 1 if k % 2 == 0 else -1
        x = 400 if sign > 0 else 800
        for i in range(WINDOW):
            lc.observe(_info(x + sign * min(i, 400)), 0)


def test_the_rule_has_no_elapsed_time_arm():                                   # [1]
    """No step count - 9,781 or any other - is a reset threshold any more."""
    import exploration.lifecycle as lifecycle
    from exploration.lifecycle import SafetyReason
    assert not hasattr(config, 'SAFETY_NO_DISCOVERY_STEPS')
    assert not hasattr(config, 'SAFETY_LOOP_MIN_EPISODE_STEPS')
    assert {v for k, v in vars(SafetyReason).items() if k.isupper()} == {
        'stuck', 'unproductive_loop'}
    lcls = lifecycle.EpisodeLifecycle
    src = "".join(inspect.getsource(f) for f in (
        lcls.safety_evidence, lcls._span_progressed, lcls._span_straightness))
    for name in ('QA_EPISODE_CAP_AGENT_STEPS_MEASURED', 'MAX_EXPLORE_STEPS',
                 'QA_EPISODE_MAX_STEPS', 'QA_EPISODE_TIME_UNITS'):
        assert name not in src, f"the safety reset reads {name}"


def test_elapsed_steps_and_a_drought_together_cannot_reset():                 # [1]
    """A slow, straight walk finding nothing for 3x the old engine lifetime:
    the longest episode and the longest drought, and no stagnation at all."""
    lc = EpisodeLifecycle(FakeCoverage())
    for i in range(3 * OLD_CLOCK * SPS):
        lc.observe(_info(100 + 0.05 * i), 0)
        if i % WINDOW == 0:
            assert not lc.safety_reset_due(), f"reset at agent step {lc.agent_steps}"
    assert lc.agent_steps >= 3 * OLD_CLOCK and lc.drought_agent_steps() >= 3 * OLD_CLOCK
    assert not lc.safety_reset_due()


def test_a_very_long_drought_in_coherent_transit_cannot_reset():             # [2]
    """Window after window of transit - even back and forth, the pattern the
    removed no-discovery arm used to cut off at 9,781 - never resets."""
    lc = EpisodeLifecycle(FakeCoverage())
    for _ in range(3 * OLD_CLOCK // config.LIFECYCLE_WINDOW + 1):
        _back_and_forth_transit(lc, 1)
        assert lc.last_window['in_transit']
        assert not lc.safety_reset_due(), f"reset in transit at {lc.agent_steps}"
    assert lc.drought_agent_steps() > 3 * OLD_CLOCK
    assert lc.consecutive_unproductive == 0


def test_a_slow_zigzag_that_gains_ground_is_not_a_loop():                    # [3]
    """Every window doubles back (none reads as transit) and nothing new is
    found - but Mario is still getting further right. That is progress. On
    flat ground his box has zero AREA, so this also pins the gap that used to
    let the STUCK arm read a wide, advancing zigzag as stuck."""
    lc = EpisodeLifecycle(FakeCoverage())
    x = 100.0
    for i in range(3 * OLD_CLOCK * SPS):
        x += 3 if i % 420 < 240 else -3                # +720, -540: net +180 a cycle
        lc.observe(_info(x), 0)
        if i % WINDOW == WINDOW - 1:
            assert not lc.safety_reset_due(), f"reset at agent step {lc.agent_steps}"
    assert not lc.last_window['in_transit']
    assert lc.consecutive_unproductive >= config.SAFETY_STUCK_WINDOWS
    assert not lc.safety_reset_due(), "reset Mario while he was still gaining ground"


def test_a_genuine_repeated_local_loop_resets():                               # [4]
    """Past the floor and the drought, four stagnant non-transit windows that
    together go nowhere: that, and nothing about the step count, resets."""
    from exploration.lifecycle import SafetyReason
    lc = EpisodeLifecycle(FakeCoverage())
    fired_at = None
    for _ in range(OLD_CLOCK):
        _wander(lc, SPS)
        if lc.safety_reset_due():
            fired_at = lc.agent_steps
            break
    assert lc.consecutive_stuck == 0, "the wander is not the small-box stuck shape"
    assert lc.safety_evidence() == SafetyReason.LOOP
    assert config.SAFETY_MIN_EPISODE_STEPS < fired_at < OLD_CLOCK, (
        "the loop was only caught at the old engine lifetime")

    # The same loop with a new pixel every 1,000 steps is productive: never.
    lc = EpisodeLifecycle(FakeCoverage())
    for _ in range(3 * OLD_CLOCK):
        _wander(lc, SPS, n_new_every=1_000 * SPS)
        assert not lc.safety_reset_due(), f"reset a productive episode at {lc.agent_steps}"


def test_the_loop_reset_ends_the_episode_through_the_wrapper(qa_env, monkeypatch):  # [4]
    """Wired end to end, and reported with the evidence it fired on."""
    monkeypatch.setattr(config, 'SAFETY_MIN_EPISODE_STEPS', 10)
    monkeypatch.setattr(config, 'SAFETY_DROUGHT_STEPS', 5)
    monkeypatch.setattr(config, 'SAFETY_STUCK_WINDOWS', 1)
    _w, drive = qa_env
    done, info, i = False, {}, 0
    while not done and i < 5 * WINDOW:
        _o, _r, done, _t, info = drive(x=400 + abs((i % 200) - 100) * 4, y=398 + (i % 130))
        i += 1
    assert done, "the loop reset never ended the episode"
    assert info['episode_end_reason'] == EndReason.SAFETY_RESET
    assert info['safety_reset_reason'] == 'unproductive_loop'


def test_long_travel_over_old_ground_never_ends_the_episode(env, patch_step):  # [1][2][3]
    """Through the real QA wrapper, on a map where every pixel is already
    covered: Mario walks the level back and forth for well over the old
    engine lifetime, finding nothing at all. No timeout, no TimeLimit, no
    reset - just a long episode."""
    cov = SpatialCoverage(testable_mask=np.ones((config.GRID_H, config.GRID_W), bool))
    cov.visited[:] = 1                                       # all old territory
    w = GlitchHunterWrapper(env, reward_mode="qa_exploration", coverage=cov)
    w.reset()
    obs = np.zeros(env.observation_space.shape, dtype=np.uint8)
    base = {'y_pos': 498, 'x_vel': 3.0, 'on_ground': True, 'status': 'small',
            'flag_get': False, 'powerup_active_count': 0, 'nearest_powerup_dx': None,
            'death_cause': None, 'is_dead': False, 'score': 0, 'coins': 0, 'viewport_x': 0}
    info, n = {}, 0
    for n in range(1, (OLD_CLOCK + 600) * SPS + 1):
        lap = (n // 2400) % 2                                   # 7,200 px each way
        x = 200 + (n % 2400) * 3 if lap == 0 else 7400 - (n % 2400) * 3
        step_info = dict(base, x_pos=x, mario_rect=(x, 498, 30, 40))
        patch_step(lambda a, _i=step_info: (obs, 0.0, False, False, dict(_i)))
        _o, _r, done, _t, info = w.step(0)
        assert not done, f"ended at agent step {n // SPS}: {info.get('episode_end_reason')}"
    assert info['lifecycle_agent_steps'] > OLD_CLOCK
    assert w.lifecycle.drought_agent_steps() > OLD_CLOCK
    assert info['coverage_episode_new'] == 0


@pytest.mark.slow
def test_a_real_pacing_loop_is_reset_on_evidence(fresh):                       # [4][5]
    """The real engine, a map already fully covered, Mario pacing right and
    left near spawn: a genuine useless loop. It ends on stagnation evidence,
    at the first moment the evidence holds - well before the old engine
    lifetime, which plays no part."""
    cov = SpatialCoverage(testable_mask=np.ones((config.GRID_H, config.GRID_W), bool))
    cov.visited[:] = 1
    w = GlitchHunterWrapper(fresh, reward_mode="qa_exploration", coverage=cov)
    w.reset()
    done, info, n = False, {}, 0
    while not done and n < (OLD_CLOCK + 100) * SPS:
        _o, _r, done, _t, info = w.step(1 if (n // (8 * SPS)) % 2 == 0 else 6)
        n += 1
    assert done, "a pacing loop was never reset"
    assert info['episode_end_reason'] == EndReason.SAFETY_RESET
    assert info['safety_reset_reason'] in ('stuck', 'unproductive_loop')
    assert config.SAFETY_MIN_EPISODE_STEPS < info['lifecycle_agent_steps'] < OLD_CLOCK


def test_qa_death_still_ends_the_run(fresh):                                   # [6]
    w = _qa_wrapper(fresh)
    w.reset()
    for _ in range(10):
        w.step(0)
    mario = fresh.game.state.mario
    mario.death_cause = 'goomba'
    mario.start_death_jump(fresh.game.state.game_info)
    _o, _r, done, _t, info = w.step(0)
    assert done and info['is_dead']
    assert info['episode_end_reason'] == EndReason.DEATH
    assert info['safety_reset_reason'] is None


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
    """Target alone is not enough (Phase 4C): COMPLETE also needs the yield
    to have genuinely dried up - see test_target_alone_does_not_transition
    and test_declining_yield_after_target_moves_to_complete for each half."""
    cov = FakeCoverage(target=1000)
    lc = EpisodeLifecycle(cov)
    cov.episode_new = 999
    lc.observe(_info(200), 5)
    assert lc.phase is EpisodePhase.EXPLORE
    cov.episode_new = 1000
    lc.observe(_info(202), 1)
    assert lc.phase is EpisodePhase.EXPLORE, "fired the instant the count crossed the target"
    # Still finding things: healthy yield keeps it in EXPLORE even though the
    # target has been met the whole time.
    for i in range(20):
        cov.episode_new += 10
        lc.observe(_info(210 + i), 10)
    assert lc.phase is EpisodePhase.EXPLORE
    # Now the yield genuinely dries up.
    _pace(lc, (config.T1_DECLINE_DROUGHT_STEPS + 1) * SPS, n_new=0)
    assert lc.phase is EpisodePhase.COMPLETE
    assert lc.transition_reason == Transition.TARGET_MET


def test_target_alone_does_not_transition():                        # [1]
    """Hitting the adaptive target does not itself trigger COMPLETE while
    novelty yield remains healthy - required test 1."""
    cov = FakeCoverage(target=500)
    lc = EpisodeLifecycle(cov)
    for i in range(50):
        cov.episode_new += 20
        lc.observe(_info(200 + i), 20)
    assert cov.episode_new >= lc.target
    assert lc.phase is EpisodePhase.EXPLORE


def test_declining_yield_after_target_moves_to_complete():          # [2]
    """Target achieved AND sustained novelty decline CAN trigger COMPLETE -
    required test 2."""
    cov = FakeCoverage(target=500)
    lc = EpisodeLifecycle(cov)
    cov.episode_new = 500
    lc.observe(_info(200), 500)
    assert lc.phase is EpisodePhase.EXPLORE
    _pace(lc, (config.T1_DECLINE_DROUGHT_STEPS + 1) * SPS, n_new=0)
    assert lc.phase is EpisodePhase.COMPLETE
    assert lc.transition_reason == Transition.TARGET_MET


def test_short_novelty_dips_do_not_transition():                    # [3]
    """A brief pause well under the decline threshold does not fire T1 -
    required test 3. A find right at the edge resets the drought clock, the
    same way it resets the reward's own drought pressure."""
    cov = FakeCoverage(target=500)
    lc = EpisodeLifecycle(cov)
    cov.episode_new = 500
    lc.observe(_info(200), 500)
    _pace(lc, (config.T1_DECLINE_DROUGHT_STEPS - 1) * SPS, n_new=0)
    assert lc.phase is EpisodePhase.EXPLORE, "a short dip fired T1 early"
    cov.episode_new += 1
    lc.observe(_info(500), 1)             # resets the drought clock
    assert lc.phase is EpisodePhase.EXPLORE
    _pace(lc, (config.T1_DECLINE_DROUGHT_STEPS - 1) * SPS, n_new=0)
    assert lc.phase is EpisodePhase.EXPLORE, "the reset was not honoured"


def test_an_uninformed_target_cannot_fire_T1():
    """A worker's first episodes have no history, so the target is the bare
    floor. Meeting it is not evidence of anything and must not end EXPLORE -
    on a virgin map it used to on the first substep (Phase 4B)."""
    cov = FakeCoverage(target=config.TARGET_FLOOR, informed=False)
    lc = EpisodeLifecycle(cov)
    cov.episode_new = 50 * config.TARGET_FLOOR
    lc.observe(_info(200), 5000)
    assert lc.phase is EpisodePhase.EXPLORE
    # The other criteria still work: sustained exhaustion moves it on (one
    # extra window, since the first one holds the 5,000 px found above).
    _pace(lc, WINDOW * (config.YIELD_WINDOWS + 1), n_new=0)
    assert lc.transition_reason == Transition.YIELD_EXHAUSTED


def test_real_coverage_informs_the_target_after_enough_history():
    cov = SpatialCoverage(
        testable_mask=np.ones((config.GRID_H, config.GRID_W), dtype=bool))
    cov.episode_new_history = [8000.0] * (config.TARGET_MIN_HISTORY - 1)
    assert not EpisodeLifecycle(cov).target_informed
    cov.episode_new_history.append(8000.0)
    lc = EpisodeLifecycle(cov)
    assert lc.target_informed and lc.target == 8000


def test_transition_is_one_way():                                   # [6]
    cov = FakeCoverage(target=10)
    lc = EpisodeLifecycle(cov)
    cov.episode_new = 10
    lc.observe(_info(200), 10)
    assert lc.phase is EpisodePhase.EXPLORE, "target alone should not fire it yet"
    _pace(lc, (config.T1_DECLINE_DROUGHT_STEPS + 1) * SPS, n_new=0)
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


def test_transit_with_target_met_does_not_transition():             # [4]
    """Legitimate transit toward remaining unexplored space does not force
    COMPLETE even once the target is met and the drought has run well past
    the decline threshold - required test 4.

    The exemption needs a CLOSED window's worth of evidence, the same as
    everywhere else in this file it is used (in_coherent_transit is False,
    "no evidence", until the first window closes) - so the crossing has to
    run a full LIFECYCLE_WINDOW, not merely past T1_DECLINE_DROUGHT_STEPS.
    """
    goal = np.array([50_000.0, 498.0])

    def frontier(_vx, pts):
        pts = np.asarray(pts, float)
        return 7, np.hypot(*(pts - goal).T)
    cov = FakeCoverage(target=10, frontier=frontier)
    lc = EpisodeLifecycle(cov)
    cov.episode_new = 10
    lc.observe(_info(200), 10)
    assert lc.phase is EpisodePhase.EXPLORE
    # A straight crossing of already-covered ground toward the frontier,
    # long enough to close a window and to push the drought well past the
    # decline threshold: zero new pixels the whole way.
    _walk_straight(lc, WINDOW + (config.T1_DECLINE_DROUGHT_STEPS + 1) * SPS, n_new=0)
    assert lc.last_window['in_transit'] is True
    assert lc.drought_agent_steps() > config.T1_DECLINE_DROUGHT_STEPS
    assert lc.phase is EpisodePhase.EXPLORE, "coherent transit was read as exhaustion"


def test_elapsed_steps_alone_cannot_trigger_T1():                   # [8]
    """No fixed elapsed-step threshold, on its own, can fire T1 - required
    test 8. Finding new pixels every window (so T2 never sees an exhausted
    streak) but never reaching an unreachably high target: the episode runs
    all the way to the T4 backstop, and the transition is T4, never T1."""
    cov = FakeCoverage(target=10 ** 9)
    lc = EpisodeLifecycle(cov)
    x = 0.0
    for _ in range(config.MAX_EXPLORE_STEPS * SPS):
        x += 3.0
        cov.episode_new += 1            # keeps every window well above YIELD_FLOOR
        lc.observe(_info(x), 1)
        if lc.phase is EpisodePhase.COMPLETE:
            break
    assert lc.phase is EpisodePhase.COMPLETE
    assert lc.transition_reason == Transition.EXPLORE_BACKSTOP


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


def test_reaching_complete_does_not_end_the_episode(qa_env):
    w, drive = qa_env
    # A blank map: the first substep finds a whole collider's worth of new
    # pixels (1,200), above an informed 1,000-px target.
    w.coverage.episode_new_history = [1000.0] * config.TARGET_MIN_HISTORY
    w.lifecycle.begin_episode()
    assert w.lifecycle.target == 1000 and w.lifecycle.target_informed
    _obs, _r, done, _t, info = drive(x=900)
    assert info['episode_phase'] == 'explore', "target alone fired it before any drought"
    assert w.coverage.episode_new >= w.lifecycle.target
    # Standing still finds nothing further - the yield genuinely dries up.
    for _ in range((config.T1_DECLINE_DROUGHT_STEPS + 1) * SPS):
        _obs, _r, done, _t, info = drive(x=900)
        if done:
            break
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
    assert info['safety_reset_reason'] == 'stuck'                            # [5]
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

    def drive(x, y=498):
        info = dict(base, x_pos=x, y_pos=y, mario_rect=(int(x), int(y), 30, 40))
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
