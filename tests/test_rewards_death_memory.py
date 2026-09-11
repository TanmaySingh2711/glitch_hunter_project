"""ADAPTIVE DEATH MEMORY (rewards/shared.py) and the shaping it switches on.

Ten deaths in a row by the same cause in the same 200 px column mark that
column as a danger zone for fifteen episodes; inside one, each mode adjusts
exactly the terms the zone's cause calls for. The memory is cross-episode by
design, so these drive several episodes through one wrapper.
"""
import pytest

from agent_logic import GlitchHunterWrapper
from exploration import config
from rewards.shared import (
    DANGER_ZONE_BOOST_EPISODES,
    DEATH_STREAK_TRIGGER,
    RewardState,
    danger_bucket,
    powerup_potential,
)

BASE = {'x_pos': 500, 'y_pos': 400, 'x_vel': 0.0, 'on_ground': True, 'score': 0,
        'coins': 0, 'status': 'small', 'flag_get': False, 'powerup_active_count': 0,
        'nearest_powerup_dx': None, 'death_cause': None, 'is_dead': False,
        'mario_rect': (500, 400, 30, 40), 'viewport_x': 0}


def _memory():
    m = RewardState()
    m._init_death_memory()
    return m


def _die(m, bucket, cause):
    m._track_death_memory(True, {'death_cause': cause}, bucket)


# ── the memory itself ──────────────────────────────────────────────────────
def test_ten_same_deaths_in_a_row_mark_a_danger_zone():
    m = _memory()
    for _ in range(DEATH_STREAK_TRIGGER - 1):
        _die(m, 3, 'pit')
    assert m.danger_zones == {}
    _die(m, 3, 'pit')
    assert m.danger_zones == {3: {'cause': 'pit', 'episodes_left': DANGER_ZONE_BOOST_EPISODES}}
    assert m.death_streaks[(3, 'pit')] == 0, "the streak re-arms after firing"


def test_a_different_death_breaks_the_streak():
    m = _memory()
    for _ in range(DEATH_STREAK_TRIGGER - 1):
        _die(m, 3, 'pit')
    _die(m, 3, 'goomba')
    _die(m, 3, 'pit')
    assert m.danger_zones == {}
    assert m.death_streaks[(3, 'pit')] == 1


def test_endings_that_are_not_deaths_are_not_remembered():
    m = _memory()
    for _ in range(2 * DEATH_STREAK_TRIGGER):
        m._track_death_memory(True, {'death_cause': None}, 3)       # safety reset
        m._track_death_memory(False, {'death_cause': 'pit'}, 3)     # not the last substep
    assert m.death_streaks == {} and m.danger_zones == {}


def test_a_zone_expires_on_schedule():
    m = _memory()
    m.danger_zones = {3: {'cause': 'pit', 'episodes_left': DANGER_ZONE_BOOST_EPISODES}}
    for _ in range(DANGER_ZONE_BOOST_EPISODES - 1):
        m._decay_danger_zones()
    assert m.danger_zones[3]['episodes_left'] == 1
    m._decay_danger_zones()
    assert m.danger_zones == {}


def test_the_memory_survives_episode_resets_through_the_wrapper(env, patch_step):
    w = GlitchHunterWrapper(env, reward_mode="legacy_completion")
    for _ in range(DEATH_STREAK_TRIGGER):
        w.reset()
        patch_step(lambda a: (None, -5.0, True, False,
                              dict(BASE, x_pos=650, death_cause='goomba', is_dead=True)))
        w.step(0)
    assert w.danger_zones[danger_bucket(650)]['cause'] == 'goomba'
    w.reset()
    assert w.danger_zones[danger_bucket(650)]['episodes_left'] == DANGER_ZONE_BOOST_EPISODES - 1


def test_helpers():
    assert danger_bucket(199.9) == 0 and danger_bucket(200) == 1
    assert powerup_potential(None) == 0.0
    assert powerup_potential(-150) == pytest.approx(-0.5)
    assert powerup_potential(10_000) == -1.0


# ── what a zone changes ─────────────────────────────────────────────────────
@pytest.fixture
def rig(env, patch_step):
    def make(mode, zone_cause=None):
        w = GlitchHunterWrapper(env, reward_mode=mode)
        if mode == "qa_exploration":
            w.lifecycle.auto_transition = False
        w.reset()
        if zone_cause:
            w.danger_zones[danger_bucket(BASE['x_pos'])] = {
                'cause': zone_cause, 'episodes_left': 5}

        def step(**info):
            patch_step(lambda a, i=dict(BASE, **info): (None, 0.0, False, False, dict(i)))
            return w.step(0)[1]
        return w, step
    return make


def _sprint(step, n=10):
    return [step(x_vel=6.0) for _ in range(n)]


def test_a_pit_zone_doubles_legacy_momentum(rig):
    _w, plain = rig("legacy_completion")
    base = _sprint(plain)
    _w, zoned = rig("legacy_completion", 'pit')
    boosted = _sprint(zoned)
    # Same trajectory; only the momentum term differs, and it doubles.
    extra = [b - a for a, b in zip(base, boosted, strict=True)]
    frames = range(1, 11)
    assert extra == pytest.approx([0.01 * f / 30.0 for f in frames])


def test_a_timeout_zone_halves_the_legacy_time_penalty(rig):
    _w, plain = rig("legacy_completion")
    _w, zoned = rig("legacy_completion", 'timeout')
    plain(), zoned()                               # first-step tile/altitude bonuses
    assert zoned() - plain() == pytest.approx(0.01)


def test_an_enemy_zone_nudges_a_jump_in_both_modes(rig):
    _w, plain = rig("legacy_completion")
    _w, zoned = rig("legacy_completion", 'goomba')
    plain(), zoned()
    assert zoned(on_ground=False) - plain(on_ground=False) == pytest.approx(0.05)

    w_plain, qa_plain = rig("qa_exploration")
    w_zoned, qa_zoned = rig("qa_exploration", 'koopa')
    qa_plain(), qa_zoned()
    diff = qa_zoned(on_ground=False) - qa_plain(on_ground=False)
    assert diff == pytest.approx(0.05 * config.QA_POWERUP_SCALE)
    assert w_zoned.ep_locomotion_paid > w_plain.ep_locomotion_paid


def test_a_pit_zone_gives_the_legacy_stuck_detector_more_patience(rig):
    w_plain, plain = rig("legacy_completion")
    w_zoned, zoned = rig("legacy_completion", 'pit')
    for _ in range(200):                           # standing still: spread 0
        plain()
        zoned()
    assert w_plain.stuck_counter == w_zoned.stuck_counter > 40
    # Past 40 checks the plain run is charged; the zoned one waits for 100.
    assert plain() < zoned()
