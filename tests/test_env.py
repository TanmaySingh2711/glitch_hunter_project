"""Core environment contract, plus regression guards for bugs already fixed."""
import os

import numpy as np


def test_action_space_is_ten(env):
    # Must stay in sync with ACTION_NAMES in agent_logic.py and the info
    # modal in templates/index.html.
    assert env.action_space.n == 10


def test_reset_returns_obs_and_info(fresh):
    obs, info = fresh.reset()
    assert obs.shape == (240, 256, 3)
    assert obs.dtype == np.uint8
    assert isinstance(info, dict)


def test_step_returns_gymnasium_five_tuple(fresh):
    result = fresh.step(1)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (240, 256, 3)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert truncated is False
    assert isinstance(info, dict)


def test_info_exposes_keys_the_reward_wrapper_depends_on(fresh):
    _, _, _, _, info = fresh.step(1)
    # GlitchHunterWrapper reads every one of these; a missing key silently
    # changes shaping via its .get() defaults rather than raising.
    for key in ('x_pos', 'y_pos', 'x_vel', 'on_ground', 'score', 'coins',
                'status', 'flag_get', 'powerup_active_count',
                'nearest_powerup_dx', 'death_cause', 'is_dead'):
        assert key in info, f"info is missing {key!r}"


def test_walking_right_actually_moves_mario(fresh):
    start = fresh.step(1)[4]['x_pos']
    for _ in range(30):
        _, _, _, _, info = fresh.step(1)
    assert info['x_pos'] > start, "Mario did not move right"


def test_step_does_not_change_working_directory(fresh):
    """Regression guard.

    step() used to chdir into the game directory and back on every single frame,
    because the game's resource paths were relative. setup.py now builds them
    absolutely, and the chdir was removed from the hot path - if anyone
    reintroduces a chdir that fails to restore, this catches it.
    """
    before = os.getcwd()
    for _ in range(5):
        fresh.step(1)
    assert os.getcwd() == before


def test_window_can_close_and_reopen(env):
    """The dashboard closes the window on Reset/disconnect and reopens it on
    the next Start. This used to crash with 'display Surface quit' because a
    stale surface reference survived in level1.py."""
    import pygame as pg
    env.close_window()
    assert pg.display.get_surface() is None
    env.open_window()
    assert pg.display.get_surface() is not None
    env.reset()
    obs, _, _, _, _ = env.step(1)
    assert obs.shape == (240, 256, 3)


def test_the_lazy_skip_matches_maxandskip_frame_for_frame(env):
    """SkipObservation renders only the two frames the max-pool keeps. Same
    engine, same actions - including an episode that ends mid-skip - must give
    the plain MaxAndSkipObservation's observations, rewards and endings
    exactly, and the engine must be left rendering normally afterwards."""
    from gymnasium.wrappers import MaxAndSkipObservation

    from custom_mario_env import SkipObservation
    rng = np.random.default_rng(7)
    actions = [int(a) for a in rng.choice([1, 2, 3, 4, 6], size=150)]

    def trace(wrapper_cls):
        w = wrapper_cls(env, skip=4)
        w.reset()
        out = []
        for a in actions:
            obs, r, term, trunc, _info = w.step(a)
            out.append((obs.copy(), r, term, trunc))
            if term or trunc:
                w.reset()
        return out

    plain, lazy = trace(MaxAndSkipObservation), trace(SkipObservation)
    for i, (p, q) in enumerate(zip(plain, lazy, strict=True)):
        assert np.array_equal(p[0], q[0]), f"frame differs at agent step {i}"
        assert p[1:] == q[1:], f"reward/ending differs at agent step {i}"
    assert env.render_observation is True
    assert env.step(1)[0].any(), "a bare step after the skip returned a blank frame"


def test_the_skip_leaves_rendering_on_even_when_a_step_raises(env, patch_step):
    from custom_mario_env import SkipObservation

    def boom(_a):
        raise RuntimeError("engine fault")
    w = SkipObservation(env, skip=4)
    patch_step(boom)
    try:
        w.step(1)
    except RuntimeError:
        pass
    assert env.render_observation is True
