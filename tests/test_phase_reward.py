"""Phase-gated reward: EXPLORE pays for coverage, COMPLETE pays for finishing.

Driven with synthetic info dicts through the real GlitchHunterWrapper, so
"walk exactly these pixels in exactly this phase" is an exact statement.
Transitions are switched off in the fixture and the phase is set by each
test, because the thing under test here is what each phase PAYS - when the
phase changes is tests/test_episode_lifecycle.py's business.

Two masks, on purpose:
  * EMPTY - nothing is testable, so novelty and the frontier potential are
    both identically zero. Every remaining term is exact, which is what lets
    these tests assert equalities instead of inequalities.
  * FULL - everything is testable, for the tests that are ABOUT novelty.
"""

import inspect

import numpy as np
import pytest

from agent_logic import GlitchHunterWrapper
from exploration import config
from exploration.coverage import SpatialCoverage
from exploration.lifecycle import (EpisodeLifecycle, EpisodePhase,
                                   Transition)

EMPTY = np.zeros((config.GRID_H, config.GRID_W), dtype=bool)
FULL = np.ones((config.GRID_H, config.GRID_W), dtype=bool)

BASE = {
    'x_pos': 500, 'y_pos': 498, 'x_vel': 0.0, 'on_ground': True,
    'status': 'small', 'flag_get': False, 'powerup_active_count': 0,
    'nearest_powerup_dx': None, 'death_cause': None, 'is_dead': False,
    'score': 0, 'coins': 0, 'viewport_x': 0,
}


class Rig:
    """A QA wrapper, its coverage, and a way to drive and to set the phase."""

    def __init__(self, env, patch_step, mask):
        self.env, self.patch_step = env, patch_step
        self.cov = SpatialCoverage(testable_mask=mask)
        self.w = GlitchHunterWrapper(env, reward_mode="qa_exploration",
                                     coverage=self.cov)
        self.w.lifecycle._check_transition = lambda n_new: None
        self.obs = np.zeros(env.observation_space.shape, dtype=np.uint8)
        self.w.reset()

    def drive(self, x, y=498, done=False, env_reward=0.0, **kw):
        info = dict(BASE, x_pos=int(x), y_pos=int(y),
                    mario_rect=(int(x), int(y), 30, 40), **kw)
        self.patch_step(lambda a, _i=info: (self.obs, env_reward, done, False,
                                            dict(_i)))
        return self.w.step(0)

    def r(self, *a, **kw):
        return self.drive(*a, **kw)[1]

    def complete(self, credit=1.0, reason=Transition.TARGET_MET):
        lc = self.w.lifecycle
        lc.phase = EpisodePhase.COMPLETE
        lc.transition_reason = reason
        lc.completion_credit = credit

    def reset(self):
        self.w.reset()


@pytest.fixture
def empty(env, patch_step):
    return Rig(env, patch_step, EMPTY)


@pytest.fixture
def full(env, patch_step):
    return Rig(env, patch_step, FULL)


# ══════════════════════════════════════════════════════════════════════════
# 1. The phase changes the reward
# ══════════════════════════════════════════════════════════════════════════
def test_same_state_and_action_pay_differently_by_phase(empty):
    """Identical substeps: a walk right across ground that pays no novelty.
    EXPLORE pays exactly nothing for it. COMPLETE pays forward progress,
    less its time cost - exactly."""
    path = [2000 + config.WALK_PX_PER_SUBSTEP * i for i in range(40)]

    empty.r(path[0])
    explore = [empty.r(x) for x in path[1:]]

    empty.reset()
    empty.r(path[0])
    empty.complete()
    complete = [empty.r(x) for x in path[1:]]

    step = config.COMPLETE_PROGRESS_PER_PX * config.WALK_PX_PER_SUBSTEP
    assert explore == [0.0] * 39, "EXPLORE paid for walking right"
    assert complete == pytest.approx([step - config.COMPLETE_TIME_PENALTY] * 39)


# ══════════════════════════════════════════════════════════════════════════
# 2-3. Novelty
# ══════════════════════════════════════════════════════════════════════════
def test_explore_novelty_is_the_dominant_positive_term(full):
    """On new ground in EXPLORE, novelty is essentially the whole reward -
    and by config, it out-pays every other positive EXPLORE term."""
    full.r(3000)
    total = sum(full.r(3000 + 7 * i, x_vel=6.0) for i in range(1, 60))
    novelty = config.NOVELTY_WEIGHT * full.w.ep_novelty_shape
    assert novelty > 0
    assert novelty >= 0.95 * total, (
        f"novelty was {novelty:.2f} of {total:.2f} earned on new ground")

    one_stride = full.cov.novelty_reward(int(config.N_REF))
    assert one_stride > config.QA_CLEAN_JUMP_REWARD * 2    # danger-zone doubled
    assert one_stride > 2.0 * config.QA_MOMENTUM_SCALE * 0.01   # sprint, max
    assert 4 * one_stride > config.FRONTIER_WEIGHT              # whole potential


def test_old_pixels_pay_zero_novelty_in_either_phase(full):
    path = [4000 + 7 * i for i in range(40)]
    for x in path:
        full.r(x)                               # cover them
    for phase in ('explore', 'complete'):
        full.reset()
        if phase == 'complete':
            full.complete()
        before = full.w.ep_novelty_shape
        for x in path:
            info = full.drive(x)[4]
            assert info['coverage_new_px'] == 0
        assert full.w.ep_novelty_shape == before, f"{phase}: old pixels paid"


def test_complete_novelty_is_a_reduced_tie_breaker(full, monkeypatch):
    """Reduced, not zeroed, and not escalated by the stagnation multiplier."""
    full.complete()
    # Local-mode coverage reads the multiplier from config; monkeypatch
    # guarantees it is restored even if an assertion below fails.
    monkeypatch.setattr(config, 'NOVELTY_WEIGHT_MULT', config.NOVELTY_MULT_MAX)
    full.r(5000)
    r = full.r(5007)
    shape = full.cov.novelty_shape(full.w.last_n_new)
    expected = (config.COMPLETE_NOVELTY_MULT * config.NOVELTY_WEIGHT * shape
                + config.COMPLETE_PROGRESS_PER_PX * 7
                - config.COMPLETE_TIME_PENALTY)
    assert shape > 0
    assert r == pytest.approx(expected, abs=1e-9)


# ══════════════════════════════════════════════════════════════════════════
# 4-6. EXPLORE: no rush, no time tax, no progress
# ══════════════════════════════════════════════════════════════════════════
def _rush(rig, x0=110, x1=8700, speed=14):
    total, x = rig.r(x0), x0
    while x < x1:
        x += speed
        total += rig.r(min(x, x1))
    return total + rig.r(x1, flag_get=True, done=True)


def test_explore_does_not_reward_rushing_to_the_finish(empty):
    """A flat-out sprint from spawn to the castle, exploring nothing, in
    EXPLORE: the flag is cancelled by the shortfall it triggers, so the whole
    rush is worth nothing - and the same run in COMPLETE is worth a lot."""
    explore_rush = _rush(empty)
    assert explore_rush <= 0.0, f"rushing in EXPLORE paid {explore_rush:.2f}"
    assert empty.w.ep_progress_paid == 0.0

    empty.reset()
    empty.r(110)
    empty.complete()
    complete_rush = _rush(empty)
    assert complete_rush > explore_rush + 100


def test_long_productive_exploration_is_not_taxed_by_time(full):
    """5,000+ agent steps finding new ground every substep. Everything that
    is not novelty must net out to almost nothing - under the old flat
    0.005/substep it would have been -100 here, growing with every step."""
    total, substeps, x, y, d = 0.0, 0, 20, 560, 1
    while substeps < 5_200 * config.SUBSTEPS_PER_AGENT_STEP:
        x += 7 * d
        if not 20 <= x <= 9000:            # snake to the next row up
            x = min(max(x, 20), 9000)
            for _ in range(4):
                y -= 10
                total += full.r(x, y)
                substeps += 1
            d = -d
        total += full.r(x, y)
        substeps += 1
    assert full.w.lifecycle.agent_steps > config.SAFETY_MIN_EPISODE_STEPS
    assert full.cov.steps_since_new_pixel < config.DROUGHT_GRACE
    residual = total - config.NOVELTY_WEIGHT * full.w.ep_novelty_shape
    assert residual > -1.0, (
        f"non-novelty terms cost {residual:.2f} over {substeps:,} productive "
        f"substeps")
    assert full.w.ep_drought_paid == 0.0


def test_a_productive_substep_deep_into_an_episode_costs_nothing(empty):
    """Agent step 7,500: no time tax, however long the episode has run."""
    empty.r(1000)
    empty.w.lifecycle.substeps = 7_500 * config.SUBSTEPS_PER_AGENT_STEP
    empty.cov.steps_since_new_pixel = 0          # just discovered something
    assert empty.r(1000) == 0.0


def test_completion_progress_cannot_be_earned_in_explore(empty):
    """Not even with completion credit sitting at 1.0."""
    empty.w.lifecycle.completion_credit = 1.0
    x = 110
    for _ in range(600):
        x += 14
        info = empty.drive(x)[4]
    assert empty.w.lifecycle.phase is EpisodePhase.EXPLORE
    assert empty.w.ep_progress_paid == 0.0
    assert info['qa_progress_paid'] == 0.0


def test_transit_over_old_ground_waives_the_drought(empty):
    """Old territory stays transit space: once a window reads as coherent
    transit, crossing ground that pays nothing also costs nothing."""
    window = config.LIFECYCLE_WINDOW * config.SUBSTEPS_PER_AGENT_STEP
    x = 110
    for _ in range(window):
        x += 3
        empty.r(x)
    assert empty.w.lifecycle.in_coherent_transit
    paid = empty.w.ep_drought_paid
    for _ in range(window * 2):
        x += 3
        empty.r(x)
    assert empty.w.ep_drought_paid == paid, "coherent transit was charged drought"


def test_standing_still_is_still_charged_drought(empty):
    """The exemption is for transit, not for doing nothing."""
    window = config.LIFECYCLE_WINDOW * config.SUBSTEPS_PER_AGENT_STEP
    for _ in range(window * 2):
        empty.r(700)
    assert not empty.w.lifecycle.in_coherent_transit
    assert empty.w.ep_drought_paid > 0


# ══════════════════════════════════════════════════════════════════════════
# 7-8. COMPLETE: a real incentive, and the flag
# ══════════════════════════════════════════════════════════════════════════
def test_complete_rewards_heading_for_the_finish(empty):
    empty.r(6000)
    empty.complete()
    right = empty.r(6006)
    still = empty.r(6006)
    left = empty.r(6000)
    assert right > still, "moving toward the finish paid no more than idling"
    assert right > left
    assert right > 0


def test_complete_finishing_beats_dying_and_pays_usefully(empty):
    """From mid-level: finish vs die at the same point."""
    def run(finish):
        empty.reset()
        empty.r(4000)
        empty.complete()
        total, x = 0.0, 4000
        while x < 8700:
            x += 6
            total += empty.r(min(x, 8700))
        if finish:
            return total + empty.r(8700, flag_get=True, done=True)
        return total + empty.r(8700, is_dead=True, death_cause='goomba',
                               env_reward=-5.0, done=True)
    finished, died = run(True), run(False)
    assert finished > died + config.ENGINE_DEATH_PENALTY
    assert finished > 50, f"finishing from mid-level paid only {finished:.1f}"


@pytest.mark.parametrize("phase,credit,flag", [
    ('explore', 1.0, config.EXPLORE_FLAG_REWARD),
    ('complete', 1.0, config.COMPLETE_FLAG_REWARD),
    ('complete', 0.5, (config.EXPLORE_FLAG_REWARD + config.COMPLETE_FLAG_REWARD) / 2),
    ('complete', 0.0, config.EXPLORE_FLAG_REWARD),
])
def test_flag_reward_is_phase_gated(empty, phase, credit, flag):
    empty.r(8600)
    empty.w.lifecycle.completion_credit = credit
    if phase == 'complete':
        empty.complete(credit)
    r = empty.r(8600, flag_get=True)
    time = config.COMPLETE_TIME_PENALTY if phase == 'complete' else 0.0
    assert r == pytest.approx(flag - time, abs=1e-9)


# ══════════════════════════════════════════════════════════════════════════
# 9-10. Coverage in COMPLETE; one way
# ══════════════════════════════════════════════════════════════════════════
def test_coverage_is_still_recorded_in_complete(full):
    full.r(6500)
    full.complete()
    covered = full.cov.covered_testable()
    for i in range(1, 30):
        full.r(6500 + 7 * i, y=420)
    assert full.cov.covered_testable() > covered + 29 * 7 * 30
    assert full.cov.episode_new > 0
    assert full.w.ep_novelty_shape > 0


def test_complete_stays_complete_for_the_episode(env, patch_step):
    """Real transitions this time: the blank map meets the target on the
    first substep, and then a long unproductive stretch - which in EXPLORE
    would ramp the drought - never brings EXPLORE's terms back."""
    cov = SpatialCoverage(testable_mask=FULL)
    w = GlitchHunterWrapper(env, reward_mode="qa_exploration", coverage=cov)
    w.reset()
    obs = np.zeros(env.observation_space.shape, dtype=np.uint8)
    info = dict(BASE, mario_rect=(900, 498, 30, 40), x_pos=900)
    patch_step(lambda a: (obs, 0.0, False, False, dict(info)))
    phases = [w.step(0)[4]['episode_phase'] for _ in range(3_000)]
    assert phases[0] == 'complete'
    assert set(phases) == {'complete'}, "the episode fell back to EXPLORE"
    assert w.ep_drought_paid == 0.0
    w.reset()
    assert w.lifecycle.phase is EpisodePhase.EXPLORE, "a new episode began in COMPLETE"


def test_frontier_shaping_closes_at_the_transition(full):
    """Paid once, bounded by FRONTIER_WEIGHT, then never again."""
    for i in range(10):
        full.r(7000, y=498 - i)
    phi_t = full.w.last_frontier_phi
    full.complete()
    first = full.r(7000, y=489)
    second = full.r(7000, y=489)
    closure = first + config.COMPLETE_TIME_PENALTY
    assert closure == pytest.approx(-config.FRONTIER_WEIGHT * phi_t, abs=1e-9)
    assert 0.0 <= closure <= config.FRONTIER_WEIGHT
    assert second == pytest.approx(-config.COMPLETE_TIME_PENALTY, abs=1e-9)


# ══════════════════════════════════════════════════════════════════════════
# Anti-rush: completion is worth what exploration earned
# ══════════════════════════════════════════════════════════════════════════
class _Cov:
    testable = None
    episode_new = 0

    def episode_target(self):
        return 10_000


@pytest.mark.parametrize("reason,new,credit", [
    (Transition.TARGET_MET, 12_000, 1.0),
    (Transition.NOTHING_AHEAD, 0, 1.0),       # nothing left to find: fully earned
    (Transition.YIELD_EXHAUSTED, 0, 0.0),     # idled into it: nothing earned
    (Transition.YIELD_EXHAUSTED, 5_000, 0.5),
    (Transition.EXPLORE_BACKSTOP, 2_500, 0.25),
])
def test_completion_credit(reason, new, credit):
    cov = _Cov()
    lc = EpisodeLifecycle(cov)
    cov.episode_new = new
    assert lc._credit_for(reason) == pytest.approx(credit)


def test_idling_into_T2_then_sprinting_does_not_pay(empty):         # [H]
    """The speedrunner with a wait bolted on: sit off-frontier until T2
    fires, then sprint for the flag. With no exploration done the credit is
    0, so the sprint earns no progress and the flag no more than EXPLORE's -
    which the shortfall cancels. Exploring first is what the payout needs."""
    empty.r(110)
    empty.complete(credit=0.0, reason=Transition.YIELD_EXHAUSTED)
    idle_then_rush = _rush(empty)
    assert idle_then_rush <= 0.0, f"idling into T2 then rushing paid {idle_then_rush:.2f}"

    empty.reset()
    empty.r(110)
    empty.complete(credit=1.0, reason=Transition.TARGET_MET)
    earned = _rush(empty)
    assert earned > idle_then_rush + 100


# ══════════════════════════════════════════════════════════════════════════
# 11. Legacy is untouched - value for value
# ══════════════════════════════════════════════════════════════════════════
def _legacy_trajectory():
    """Tiles, milestones, running jumps, backing up, a coin, a stomp, a
    powerup reveal and pickup, a shrink, and a death. Recorded against the
    Phase 3 code, before any phase-gated change existed."""
    base = {'x_pos': 110, 'y_pos': 498, 'x_vel': 0.0, 'on_ground': True,
            'status': 'small', 'flag_get': False, 'powerup_active_count': 0,
            'nearest_powerup_dx': None, 'death_cause': None, 'is_dead': False,
            'score': 0, 'coins': 0, 'viewport_x': 0,
            'mario_rect': (110, 498, 30, 40)}
    out = []
    x, score, coins = 110, 0, 0
    for i in range(160):
        on_ground = not (40 <= i % 60 < 52)
        if i < 120:
            x += 7 if i % 9 else -3
        if i == 70:
            score, coins = score + 200, coins + 1
        if i == 90:
            score += 1000
        info = dict(base, x_pos=x, x_vel=6.5 if i % 9 else -3.0,
                    on_ground=on_ground, score=score, coins=coins,
                    mario_rect=(x, 498 if on_ground else 420, 30, 40),
                    status='tall' if 95 <= i < 130 else 'small',
                    powerup_active_count=1 if 93 <= i < 95 else 0,
                    nearest_powerup_dx=40 if 88 <= i < 95 else None)
        if i == 159:
            info.update(death_cause='goomba', is_dead=True)
        out.append(info)
    return out


LEGACY_GOLDEN = [
    1.38, 0.080333333333, 1.080666666667, 0.081, 0.081333333333,
    0.081666666667, 0.082, 0.082333333333, 1.082666666667, -0.02,
    0.080333333333, 0.080666666667, 0.081, 0.081333333333, 0.081666666667,
    1.082, 0.082333333333, 0.082666666667, -0.02, 0.080333333333,
    0.080666666667, 0.081, 1.081333333333, 0.081666666667, 0.082,
    0.082333333333, 0.082666666667, -0.02, 0.080333333333, 1.080666666667,
    0.081, 0.081333333333, 0.081666666667, 0.082, 0.082333333333,
    1.082666666667, -0.02, 0.080333333333, 0.080666666667, 0.081, 0.08,
    0.08, 1.08, 0.08, 0.08, -0.02, 0.08, 0.08, 0.08, 26.08, 0.08, 0.08,
    3.080333333333, 0.080666666667, -0.02, 0.080333333333, 0.080666666667,
    1.081, 0.081333333333, 0.081666666667, 0.082, 0.082333333333,
    1.082666666667, -0.02, 0.080333333333, 0.080666666667, 0.081,
    0.081333333333, 0.081666666667, 1.082, 3.082333333333, 0.082666666667,
    -0.02, 0.080333333333, 0.080666666667, 0.081, 0.081333333333,
    1.081666666667, 0.082, 0.082333333333, 0.082666666667, -0.02,
    0.080333333333, 0.080666666667, 1.081, 0.081333333333, 0.081666666667,
    0.082, 0.075733333333, 1.082733333333, 149.980066666667, 0.0804,
    0.080733333333, 8.081066666667, 0.0814, 20.083666666667, 0.082,
    1.082333333333, 0.082666666667, -0.02, 0.08, 0.08, 0.08, 0.08, 1.08,
    0.08, 0.08, 0.08, -0.02, 0.08, 0.08, 1.08, 3.080333333333,
    0.080666666667, 0.081, 0.081333333333, 0.081666666667, -0.02,
    26.080333333333, 0.080666666667, -0.019, -0.018666666667,
    -0.018333333333, -0.018, -0.017666666667, -0.017333333333, -0.02,
    -0.019666666667, -0.019333333333, -0.019, -10.018666666667,
    -0.018333333333, -0.018, -0.017666666667, -0.017333333333, -0.02,
    -0.019666666667, -0.019333333333, -0.019, -0.018666666667,
    -0.018333333333, -0.018, -0.017666666667, -0.017333333333, -0.02,
    -0.019666666667, -0.019333333333, -0.019, -0.018666666667,
    -0.018333333333, -0.018, -0.017666666667, -0.017333333333, -0.02,
    -0.019666666667, -0.019333333333, -0.019, -0.018666666667,
    -0.018333333333, -5.018,
]


def test_legacy_reward_is_value_for_value_unchanged(env, patch_step):
    w = GlitchHunterWrapper(env, reward_mode="legacy_completion",
                            attach_coverage=False)
    w.reset()
    obs = np.zeros(env.observation_space.shape, dtype=np.uint8)
    got, dones = [], []
    for info in _legacy_trajectory():
        patch_step(lambda a, _i=info: (obs, -5.0 if _i['is_dead'] else 0.0,
                                       _i['is_dead'], False, dict(_i)))
        _o, r, d, _t, _info = w.step(0)
        got.append(r)
        dones.append(d)
    assert got == pytest.approx(LEGACY_GOLDEN, abs=1e-9)
    assert dones.index(True) == 159
    assert w.lifecycle is None


# ══════════════════════════════════════════════════════════════════════════
# 12. Bounds
# ══════════════════════════════════════════════════════════════════════════
def test_phase_reward_balance_holds():
    config.assert_phase_reward_balance()


@pytest.mark.parametrize("name,value", [
    ("COMPLETE_FLAG_REWARD", 500.0),          # the legacy flag, restored
    ("EXPLORE_TIME_PENALTY", 0.005),          # the old flat time tax
    ("COMPLETE_NOVELTY_MULT", 1.0),           # novelty fighting the finish
    ("COMPLETE_TIME_PENALTY_EPISODE_CAP", 5.0),
    ("EXPLORE_FLAG_REWARD", 50.0),
])
def test_phase_reward_balance_rejects_bad_retunes(monkeypatch, name, value):
    monkeypatch.setattr(config, name, value)
    with pytest.raises(ValueError, match="phase reward balance"):
        config.assert_phase_reward_balance()


def test_death_penalty_constant_matches_the_env():
    from custom_mario_env import CustomMarioEnv
    assert f"reward = -{config.ENGINE_DEATH_PENALTY}" in inspect.getsource(
        CustomMarioEnv.step)


@pytest.mark.parametrize("phase", ['explore', 'complete'])
def test_no_term_explodes_past_the_safety_bounds(full, phase):
    full.r(7000)
    if phase == 'complete':
        full.complete()
    extremes = [
        {'score': 99_999, 'coins': 999},
        {'flag_get': True, 'status': 'fireball'},
        {'powerup_active_count': 50},
    ]
    for e in extremes:
        assert abs(full.r(7010, **e)) <= config.QA_REWARD_CLIP + 1e-9
    # A teleport is a glitch, not progress: it moves the baseline, pays one
    # frame's worth at most.
    before = full.w.ep_progress_paid
    full.r(9000)
    assert full.w.ep_progress_paid - before <= (
        config.COMPLETE_PROGRESS_PER_PX * config.MAX_FRAME_DX + 1e-9)


def test_complete_time_and_progress_are_bounded_per_episode(empty):
    empty.r(110)
    empty.complete()
    x = 110
    for _ in range(3_000):
        x = min(x + 14, config.LEVEL_W)
        empty.r(x)
    assert empty.w.ep_complete_time_paid == pytest.approx(
        config.COMPLETE_TIME_PENALTY_EPISODE_CAP)
    assert empty.w.ep_progress_paid <= (
        config.COMPLETE_PROGRESS_PER_PX * config.LEVEL_W + 1e-9)
