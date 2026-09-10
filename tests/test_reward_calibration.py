"""Phase 4B: the calibrated phase-gated reward, and the books it keeps.

The constants were calibrated on real 6M trajectories with
tools/calibrate_phase_reward.py (inference only). These tests pin what that
calibration has to keep true, stated against the same references it used:

  * the measured mean EXPLORE novelty of one 6M episode from the bootstrap
    map (config.CALIB_EXPLORE_EPISODE_NOVELTY_MEAN), which bounds a whole
    full-credit COMPLETE run - the policy cannot see the phase, so one
    completion must not be worth more than one exploration episode;
  * the per-substep orderings inside COMPLETE, where progress has to beat
    idling, local novelty and dying, and the flag has to stay a finish line
    rather than a jackpot.

Numbers in brackets are the Phase 4B brief's test list.
"""

import numpy as np
import pytest
from test_phase_reward import BASE, EMPTY, FULL, LEGACY_GOLDEN, Rig, _legacy_trajectory, _rush

from agent_logic import QA_CHANNELS, GlitchHunterWrapper
from exploration import config
from exploration.coverage import SpatialCoverage
from exploration.lifecycle import EpisodePhase, Transition

SPS = config.SUBSTEPS_PER_AGENT_STEP
WINDOW = config.LIFECYCLE_WINDOW * SPS
BUDGET = config.COMPLETE_PROGRESS_PER_PX * config.LEVEL_COMPLETE_SPAN_PX
COMPLETE_DRIVEN = ('progress', 'flag', 'time')


@pytest.fixture
def empty(env, patch_step):
    return Rig(env, patch_step, EMPTY)


@pytest.fixture
def full(env, patch_step):
    return Rig(env, patch_step, FULL)


def _driven(rig, phase, keys):
    return sum(rig.w.ep_channels[phase][k] for k in keys)


def _pace_until_done(rig, limit):
    """Back and forth in a 20 px box - the shape of genuine stuck."""
    total, i = 0.0, 0
    while i < limit:
        _o, r, done, _t, info = rig.drive(700 + (20 if (i // 30) % 2 else 0))
        total += r
        i += 1
        if done:
            return total, r, info
    raise AssertionError("the safety reset never fired")


# ══════════════════════════════════════════════════════════════════════════
# The books: every substep's channels sum to its reward
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("phase", ['explore', 'complete'])
def test_channels_sum_to_the_reward_on_every_substep(full, phase):
    if phase == 'complete':
        full.complete()
    full.r(3000)
    infos = [{}, {'x_vel': 6.0}, {'on_ground': False, 'x_vel': 6.0},
             {'score': 400, 'coins': 1}, {'status': 'tall'},
             {'flag_get': True, 'status': 'fireball', 'score': 90_000}]
    for i, extra in enumerate(infos * 3):
        r = full.r(3000 + 9 * i, y=498 - (i % 5) * 7, **extra)
        ch = full.w.last_channels
        assert set(ch) == set(QA_CHANNELS)
        assert sum(ch.values()) == pytest.approx(r, abs=1e-12)
    assert full.w.ep_reward_total == pytest.approx(
        sum(sum(c.values()) for c in full.w.ep_channels.values()), abs=1e-9)
    other = 'explore' if phase == 'complete' else 'complete'
    assert set(full.w.ep_channels[other].values()) == {0.0}


def test_final_info_carries_a_snapshot_of_the_episode_books(empty):
    empty.r(1000)
    empty.complete()
    info = empty.drive(1006, done=True)[4]
    snap = info['qa_channels']
    assert snap['complete']['progress'] == pytest.approx(
        config.COMPLETE_PROGRESS_PER_PX * 6)
    empty.reset()
    empty.r(1000)
    assert snap['complete']['progress'] > 0, "reset() emptied the last snapshot"


# ══════════════════════════════════════════════════════════════════════════
# [1] EXPLORE is free of progress reward
# ══════════════════════════════════════════════════════════════════════════
def test_explore_pays_no_progress_across_the_whole_level(empty):
    empty.w.lifecycle.completion_credit = 1.0          # even with credit set
    _rush(empty)
    for phase in ('explore', 'complete'):
        assert empty.w.ep_channels[phase]['progress'] == 0.0
        assert empty.w.ep_channels[phase]['time'] == 0.0


# ══════════════════════════════════════════════════════════════════════════
# [2] COMPLETE progress beats COMPLETE idling
# ══════════════════════════════════════════════════════════════════════════
def test_complete_progress_beats_idling(empty):
    n = 1600

    def run(step):
        empty.reset()
        empty.r(2000)
        empty.complete()
        return sum(empty.r(2000 + step * (i + 1)) for i in range(n))
    walking, idling = run(config.WALK_PX_PER_SUBSTEP), run(0)
    assert idling == pytest.approx(-config.COMPLETE_TIME_PENALTY_EPISODE_CAP)
    assert walking > 0 > idling
    # Net positive on EVERY walking substep, not just in total: urgency
    # never outweighs the step it is hurrying.
    assert (config.WALK_PX_PER_SUBSTEP * config.COMPLETE_PROGRESS_PER_PX
            > 4 * config.COMPLETE_TIME_PENALTY)


# ══════════════════════════════════════════════════════════════════════════
# [3] COMPLETE does not overpower exploration at campaign scale
# ══════════════════════════════════════════════════════════════════════════
def test_a_full_completion_is_worth_less_than_one_exploration_episode(empty):
    """The whole of COMPLETE's pay - a full-credit run from the spawn to the
    castle - through the real wrapper, against the measured mean EXPLORE
    novelty of one 6M episode from the bootstrap map."""
    empty.r(110)
    empty.complete()
    _rush(empty, speed=config.WALK_PX_PER_SUBSTEP)
    paid = _driven(empty, 'complete', ('progress', 'flag'))
    assert paid == pytest.approx(BUDGET + config.COMPLETE_FLAG_REWARD, rel=0.01)
    assert paid <= config.CALIB_EXPLORE_EPISODE_NOVELTY_MEAN
    # ... and it is not "many" exploration episodes either.
    assert paid <= 2.5 * config.CALIB_EXPLORE_EPISODE_NOVELTY_MEDIAN


def test_one_realistic_exploration_episode_outearns_a_full_completion(full):
    """A mean-sized EXPLORE episode's worth of genuine discovery, driven as
    real new ground, out-pays the complete COMPLETE payout."""
    full.r(1000, y=560)
    x, y, earned = 1000, 560, 0.0
    while config.NOVELTY_WEIGHT * full.w.ep_novelty_shape < (
            config.CALIB_EXPLORE_EPISODE_NOVELTY_MEAN):
        x += 7
        if x > 8000:
            x, y = 1000, y - 40
        earned += full.r(x, y)
    assert earned >= BUDGET + config.COMPLETE_FLAG_REWARD


# ══════════════════════════════════════════════════════════════════════════
# [4] COMPLETE novelty cannot dominate forward progress
# ══════════════════════════════════════════════════════════════════════════
def test_complete_novelty_is_under_half_of_progress_on_virgin_ground(full):
    full.r(5000)
    full.complete()
    for i in range(1, 40):
        full.r(5000 + config.WALK_PX_PER_SUBSTEP * i)
        ch = full.w.last_channels
        assert ch['novelty'] > 0, "coverage stopped paying anything"
        assert 2 * ch['novelty'] <= ch['progress'] + 1e-12


def test_a_detour_onto_virgin_ground_pays_less_than_heading_on(full):
    """Same number of substeps: climb straight up onto never-seen pixels, or
    walk toward the castle over ground covered earlier."""
    n = 60
    route = [6000 + config.WALK_PX_PER_SUBSTEP * i for i in range(n + 1)]
    for x in route:                                   # cover the route
        full.r(x)
    full.reset()
    full.r(route[0])
    full.complete()
    heading_on = sum(full.r(x) for x in route[1:])
    assert full.w.ep_channels['complete']['novelty'] == 0.0

    full.reset()
    full.r(route[0])
    full.complete()
    detour = sum(full.r(route[0], y=498 - 4 * i) for i in range(1, n + 1))
    assert full.w.ep_channels['complete']['novelty'] > 0
    assert heading_on > 2 * detour


# ══════════════════════════════════════════════════════════════════════════
# [5] Death cannot profitably avoid the COMPLETE time cost
# ══════════════════════════════════════════════════════════════════════════
def _die(rig, x):
    return rig.r(x, is_dead=True, death_cause='goomba',
                 env_reward=-config.ENGINE_DEATH_PENALTY, done=True)


def test_dying_saves_at_most_half_a_death(empty):
    """Stuck in COMPLETE, nowhere to go: die now, or wait out the time cost
    and end at the timer (which the engine also charges as a death)."""
    empty.r(3000)
    empty.complete()
    die_now = _die(empty, 3000)

    empty.reset()
    empty.r(3000)
    empty.complete()
    wait = sum(empty.r(3000) for _ in range(4000)) + _die(empty, 3000)

    saved = die_now - wait
    # The whole cap, less the one time tick the dying substep itself pays.
    cap = config.COMPLETE_TIME_PENALTY_EPISODE_CAP
    assert cap - config.COMPLETE_TIME_PENALTY - 1e-9 <= saved <= cap + 1e-9
    assert saved <= 0.5 * config.ENGINE_DEATH_PENALTY
    assert die_now < 0, "dying was reward-positive"


@pytest.mark.parametrize("start", [1000, 5000, 8200])
def test_continuing_toward_the_castle_always_beats_dying(empty, start):
    def run(finish):
        empty.reset()
        empty.r(start)
        empty.complete()
        total, x = 0.0, start
        while x < 8700:
            x = min(x + config.WALK_PX_PER_SUBSTEP, 8700)
            total += empty.r(x)
        if finish:
            return total + empty.r(8700, flag_get=True, done=True)
        return total + _die(empty, 8700)

    empty.reset()
    empty.r(start)
    empty.complete()
    die_now = _die(empty, start)
    assert run(True) > die_now + config.ENGINE_DEATH_PENALTY
    assert run(True) > run(False)


# ══════════════════════════════════════════════════════════════════════════
# [6] The safety reset cannot be reward-positive
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("phase", ['explore', 'complete'])
def test_a_real_safety_reset_costs(empty, phase):
    """Genuinely stuck for 5,000+ agent steps until the reset fires."""
    empty.r(700)
    if phase == 'complete':
        empty.complete()
    total, last, info = _pace_until_done(empty, 12_000 * SPS)
    assert info['episode_end_reason'] == 'safety_reset'
    assert last < 0
    assert total < 0
    assert empty.w.ep_channels[phase]['safety_reset'] == config.DROUGHT_TERMINAL_PENALTY


def test_the_reset_substep_stays_negative_even_on_a_good_substep(empty, monkeypatch):
    """Worst case: it fires on a substep that also sprints forward in
    COMPLETE and lands a clean running jump."""
    empty.r(3000, on_ground=False, x_vel=6.0)
    empty.w.was_on_ground = True
    empty.r(3014, on_ground=False, x_vel=6.0)
    empty.complete()
    monkeypatch.setattr(empty.w.lifecycle, 'safety_reset_due', lambda: True)
    r = empty.r(3100, x_vel=6.0)
    assert empty.w.last_channels['safety_reset'] == config.DROUGHT_TERMINAL_PENALTY
    assert r < 0


# ══════════════════════════════════════════════════════════════════════════
# [7] Early / low-credit transitions cannot be farmed
# ══════════════════════════════════════════════════════════════════════════
def test_the_floor_target_cannot_hand_out_a_transition(env, patch_step):
    """A worker's first episodes: 30,000 new px against the 500 px floor
    target, and EXPLORE continues - every one of them paid at full rate."""
    cov = SpatialCoverage(testable_mask=FULL)
    w = GlitchHunterWrapper(env, reward_mode="qa_exploration", coverage=cov)
    w.reset()
    assert not w.lifecycle.target_informed
    obs = np.zeros(env.observation_space.shape, dtype=np.uint8)
    for i in range(100):
        info = dict(BASE, x_pos=900 + 7 * i, mario_rect=(900 + 7 * i, 498, 30, 40))
        patch_step(lambda a, _i=info: (obs, 0.0, False, False, dict(_i)))
        info = w.step(0)[4]
    assert cov.episode_new > 50 * config.TARGET_FLOOR
    assert info['episode_phase'] == 'explore'
    assert w.ep_channels['complete']['novelty'] == 0.0


def test_idling_into_T2_earns_no_completion(env, patch_step):
    """Real transitions: stand still off the frontier until T2 fires, then
    run for the castle. Credit 0 - no progress, and a flag worth exactly the
    EXPLORE flag - so the trip through COMPLETE bought nothing."""
    rig = Rig(env, patch_step, FULL)
    for i in range(400):                       # an earlier episode covered this spot
        rig.r(700 + (20 if (i // 30) % 2 else 0))
    del rig.w.lifecycle._check_transition      # real transitions from here on
    rig.w.reset()
    while not rig.w.lifecycle.is_complete:
        rig.r(700 + (20 if (rig.w.ep_substeps // 30) % 2 else 0))
    assert rig.cov.episode_new == 0
    assert rig.w.lifecycle.transition_reason == Transition.YIELD_EXHAUSTED
    assert rig.w.lifecycle.completion_credit == 0.0
    x = 700
    while x < 8700:
        x = min(x + 14, 8700)
        rig.r(x)
    rig.r(8700, flag_get=True, done=True)
    c = rig.w.ep_channels['complete']
    assert c['progress'] == 0.0
    assert c['flag'] == config.EXPLORE_FLAG_REWARD


def test_low_credit_scales_the_whole_payout(empty):
    def payout(credit):
        empty.reset()
        empty.r(110)
        empty.complete(credit=credit, reason=Transition.EXPLORE_BACKSTOP)
        _rush(empty)
        return _driven(empty, 'complete', ('progress', 'flag'))
    none, half, full_credit = payout(0.0), payout(0.5), payout(1.0)
    assert none == pytest.approx(config.EXPLORE_FLAG_REWARD)
    assert none < half < full_credit
    assert half == pytest.approx((none + full_credit) / 2, rel=1e-6)


# ══════════════════════════════════════════════════════════════════════════
# [8] Legitimate target-achieved completion is positively rewarded
# ══════════════════════════════════════════════════════════════════════════
def test_meeting_an_informed_target_then_finishing_pays(env, patch_step):
    cov = SpatialCoverage(testable_mask=FULL)
    cov.episode_new_history = [2000.0] * config.TARGET_MIN_HISTORY
    w = GlitchHunterWrapper(env, reward_mode="qa_exploration", coverage=cov)
    w.reset()
    obs = np.zeros(env.observation_space.shape, dtype=np.uint8)

    def step(x, y=498, **kw):
        info = dict(BASE, x_pos=x, mario_rect=(x, y, 30, 40), **kw)
        patch_step(lambda a, _i=info: (obs, 0.0, kw.get('flag_get', False),
                                       False, dict(_i)))
        return w.step(0)[4]

    x = 150
    while not w.lifecycle.is_complete:
        step(x, y=560 - (x % 3) * 30)
        x += 3
    assert w.lifecycle.transition_reason == Transition.TARGET_MET
    assert w.lifecycle.completion_credit == 1.0
    while x < 8700:
        x = min(x + config.WALK_PX_PER_SUBSTEP, 8700)
        step(x)
    step(8700, flag_get=True)
    c = w.ep_channels['complete']
    assert c['flag'] == config.COMPLETE_FLAG_REWARD
    assert _driven_w(w, 'complete') > 0.5 * (BUDGET + config.COMPLETE_FLAG_REWARD)
    assert sum(c.values()) > 0


def _driven_w(w, phase):
    return sum(w.ep_channels[phase][k] for k in COMPLETE_DRIVEN)


# ══════════════════════════════════════════════════════════════════════════
# [9] The flag is useful but not dominant
# ══════════════════════════════════════════════════════════════════════════
def test_flag_is_useful_but_not_a_jackpot():
    assert config.EXPLORE_FLAG_REWARD <= config.COMPLETE_FLAG_REWARD
    # Useful: finishing is worth more than not finishing, and much more than
    # dying at the pole.
    assert config.COMPLETE_FLAG_REWARD > 0
    assert config.COMPLETE_FLAG_REWARD + config.ENGINE_DEATH_PENALTY >= 10
    # Not dominant: the route is worth more than the finish line, and the
    # finish line is a fraction of one exploration episode.
    assert config.COMPLETE_FLAG_REWARD < BUDGET
    assert config.COMPLETE_FLAG_REWARD < 0.5 * config.CALIB_EXPLORE_EPISODE_NOVELTY_MEAN
    assert config.COMPLETE_FLAG_REWARD < 500.0 / 50


def test_the_flag_substep_never_clips(full):
    """The worst the flag substep can carry: full credit, a sprint onto new
    ground, a clean running jump landing, and a powerup on the same frame."""
    full.r(8600, on_ground=False, x_vel=6.0)
    full.w.was_on_ground = True
    full.r(8614, y=420, on_ground=False, x_vel=6.0)
    full.complete(credit=1.0)
    clips = full.w.qa_clip_events
    r = full.r(8628, y=400, x_vel=6.0, flag_get=True, status='tall')
    assert full.w.qa_clip_events == clips
    assert r < config.QA_REWARD_CLIP
    assert full.w.last_channels['clip'] == 0.0


# ══════════════════════════════════════════════════════════════════════════
# [10] Clipping stays exceptional
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("phase", ['explore', 'complete'])
def test_a_busy_full_level_run_never_clips(full, phase):
    """Spawn to castle on virgin ground with running jumps, coins, stomps, a
    powerup and a shrink along the way. Calibration measured 0 clips in
    ~1,000,000 real substeps; this is the synthetic worst of the same."""
    full.r(110)
    if phase == 'complete':
        full.complete()
    x, score, coins = 110, 0, 0
    for i in range(1, 700):
        x += 12
        airborne = (i % 40) >= 30
        if i % 50 == 0:
            score, coins = score + 200, coins + 1
        if i % 97 == 0:
            score += 100
        full.r(min(x, 8700), y=420 if airborne else 498, on_ground=not airborne,
               x_vel=6.0, score=score, coins=coins,
               status='tall' if 300 <= i < 500 else 'small')
    full.r(8700, flag_get=True, done=True, score=score, coins=coins)
    assert full.w.qa_clip_events == 0


def _farm_clean_jumps(rig, n_jumps):
    """Running jumps back and forth on the spawn screen, forever."""
    x, d = 400, 1
    for _ in range(n_jumps):
        for _ in range(6):                                  # run up
            x += 6 * d
            rig.r(x, x_vel=6.0 * d)
        for _ in range(12):                                 # airborne, > 60 px
            x += 7 * d
            rig.r(x, y=420, on_ground=False, x_vel=6.0 * d)
        rig.r(x, x_vel=6.0 * d)                             # land
        d = -d


@pytest.mark.parametrize("phase", ['explore', 'complete'])
def test_a_locomotion_farm_stops_paying_at_the_cap(empty, phase):
    """Phase 4B measured a real in-place farm at 35.7 per episode, 4.4x the
    natural maximum. Direction-agnostic jumps stay rewarded - up to the cap."""
    empty.r(400)
    if phase == 'complete':
        empty.complete()
    _farm_clean_jumps(empty, 200)
    paid = empty.w.ep_channels[phase]['locomotion']
    assert paid == pytest.approx(config.QA_LOCOMOTION_EPISODE_CAP)
    before = empty.w.ep_reward_total
    _farm_clean_jumps(empty, 20)
    assert empty.w.ep_channels[phase]['locomotion'] == paid
    assert empty.w.ep_reward_total <= before, "the farm kept paying past the cap"


def test_ordinary_locomotion_is_untouched_by_the_cap(empty, monkeypatch):
    """Below the cap every payment is unchanged to the bit: the same jumps
    with the cap and with no cap at all."""
    def rewards():
        empty.reset()
        empty.r(400)
        out = []
        orig = empty.r
        empty.r = lambda *a, **k: out.append(orig(*a, **k)) or out[-1]
        try:
            _farm_clean_jumps(empty, 20)
        finally:
            empty.r = orig
        return out, empty.w.ep_channels['explore']['locomotion']
    capped, paid = rewards()
    monkeypatch.setattr(config, 'QA_LOCOMOTION_EPISODE_CAP', float('inf'))
    uncapped, _ = rewards()
    assert 0 < paid < config.QA_LOCOMOTION_EPISODE_CAP
    assert capped == uncapped
    # The measured natural maximum sits under the cap.
    assert config.QA_LOCOMOTION_EPISODE_CAP > 8.14


# ══════════════════════════════════════════════════════════════════════════
# [11] Long productive EXPLORE episodes remain viable
# ══════════════════════════════════════════════════════════════════════════
def test_productive_exploration_to_the_full_qa_cap_is_not_taxed(full):
    """The whole 9,781-step QA episode finding new ground, then the timer
    ends it. Nothing that is not novelty may add up to a meaningful cost -
    so running long is never worse than stopping early."""
    total, subs, x, y, d = 0.0, 0, 20, 560, 1
    cap = config.QA_EPISODE_CAP_AGENT_STEPS_MEASURED * SPS
    while subs < cap - 1:
        x += 7 * d
        if not 20 <= x <= 9000:
            x = min(max(x, 20), 9000)
            y -= 10
            d = -d
        total += full.r(x, y)
        subs += 1
    total += full.r(x, y, is_dead=True, death_cause='timeout',
                    env_reward=-config.ENGINE_DEATH_PENALTY, done=True)
    novelty = full.w.ep_channels['explore']['novelty']
    e = full.w.ep_channels['explore']
    assert e['time'] == 0.0 and e['drought'] == 0.0 and e['progress'] == 0.0
    # The one fixed cost is the timer's own -5, the same as any other end.
    assert total - novelty >= -config.ENGINE_DEATH_PENALTY - 1.0
    assert total > 0.9 * novelty


# ══════════════════════════════════════════════════════════════════════════
# [12] legacy_completion is untouched - even by extreme phase constants
# ══════════════════════════════════════════════════════════════════════════
def test_legacy_ignores_every_phase_constant(env, patch_step, monkeypatch):
    """The golden trajectory (tests/test_phase_reward.py), replayed with every
    phase-gated constant set to something absurd. Legacy must not move."""
    for name, value in [('COMPLETE_PROGRESS_PER_PX', 99.0),
                        ('COMPLETE_FLAG_REWARD', 999.0),
                        ('EXPLORE_FLAG_REWARD', 999.0),
                        ('COMPLETE_TIME_PENALTY', 9.0),
                        ('COMPLETE_TIME_PENALTY_EPISODE_CAP', 99.0),
                        ('COMPLETE_NOVELTY_MULT', 1.0),
                        ('TARGET_MIN_HISTORY', 0)]:
        monkeypatch.setattr(config, name, value)
    w = GlitchHunterWrapper(env, reward_mode="legacy_completion",
                            attach_coverage=False)
    w.reset()
    obs = np.zeros(env.observation_space.shape, dtype=np.uint8)
    got = []
    for info in _legacy_trajectory():
        patch_step(lambda a, _i=info: (obs, -5.0 if _i['is_dead'] else 0.0,
                                       _i['is_dead'], False, dict(_i)))
        got.append(w.step(0)[1])
    assert got == pytest.approx(LEGACY_GOLDEN, abs=1e-9)


# ══════════════════════════════════════════════════════════════════════════
# The 4A values are rejected now, each for the reason calibration found
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("name,value,why", [
    ("COMPLETE_PROGRESS_PER_PX", 0.03, "more than the measured mean exploration"),
    ("COMPLETE_FLAG_REWARD", 10.0, "clip"),
    ("COMPLETE_TIME_PENALTY", 0.005, "quarter of walking progress"),
    ("COMPLETE_TIME_PENALTY_EPISODE_CAP", 4.0, "half the death penalty"),
    ("COMPLETE_NOVELTY_MULT", 0.1, "2x margin"),
])
def test_the_provisional_4A_values_fail_the_calibrated_balance(monkeypatch, name,
                                                               value, why):
    monkeypatch.setattr(config, name, value)
    with pytest.raises(ValueError, match=why):
        config.assert_phase_reward_balance()


def test_the_phase_is_still_invisible_to_the_policy(env):
    """4B calibrates around phase invisibility; it does not remove it. If
    this ever fails, the leakage analysis in config no longer applies."""
    from gymnasium.wrappers import MaxAndSkipObservation
    w = GlitchHunterWrapper(env, reward_mode="qa_exploration",
                            coverage=SpatialCoverage(testable_mask=EMPTY))
    obs, _ = MaxAndSkipObservation(w, skip=SPS).reset()
    assert obs.shape == env.observation_space.shape
    assert w.lifecycle.phase is EpisodePhase.EXPLORE
