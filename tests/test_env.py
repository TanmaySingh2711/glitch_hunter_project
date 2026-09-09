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

    step() used to chdir into mario_clone/ and back on every single frame,
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
