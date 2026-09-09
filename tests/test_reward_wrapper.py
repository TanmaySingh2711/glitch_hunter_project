"""GlitchHunterWrapper shaping rules - LEGACY COMPLETION MODE.

These pin the intent behind the reward design, which is easy to break
silently: a shaping change shows up as "the agent got worse after 2M steps",
not as an exception.

Every test here now names its mode EXPLICITLY rather than relying on the
config default. That is the point of the file: this is the reward the 6M
checkpoint was actually trained under, and it has to keep behaving exactly
this way both as the calibration baseline and as the fallback if the QA
retrofit has to be backed out. If config.REWARD_MODE flips, these must not
follow it.

QA-mode behaviour is pinned separately, in test_reward_qa.py.
"""
from agent_logic import GlitchHunterWrapper


def test_full_training_wrapper_stack_builds_and_steps(env):
    """The exact stack train_agent.py uses. Observation shape feeds straight
    into the CNN policy, so a change here silently invalidates the
    checkpoint."""
    from gymnasium.wrappers import (MaxAndSkipObservation, GrayscaleObservation,
                                    ResizeObservation, FrameStackObservation,
                                    TimeLimit)
    e = GlitchHunterWrapper(env, reward_mode="legacy_completion")
    e = MaxAndSkipObservation(e, skip=4)
    e = GrayscaleObservation(e, keep_dim=False)
    e = ResizeObservation(e, (84, 84))
    e = FrameStackObservation(e, 4)
    e = TimeLimit(e, max_episode_steps=4000)

    obs, _ = e.reset()
    assert obs.shape == (4, 84, 84), "policy input shape changed"
    obs, reward, _term, _trunc, _info = e.step(1)
    assert obs.shape == (4, 84, 84)
    assert isinstance(reward, float)


def test_exploring_new_ground_is_rewarded(env):
    """Rule 1: reward thorough exploration. Moving into previously unvisited
    tiles should pay out."""
    w = GlitchHunterWrapper(env, reward_mode="legacy_completion")
    w.reset()
    total = 0.0
    for _ in range(40):
        _, r, term, _, _ = w.step(3)   # run right
        total += r
        if term:
            break
    assert total > 0, "moving into new territory earned nothing"


def test_standing_still_is_not_more_attractive_than_moving(env):
    """Rule 2: idling must never beat making progress, or the agent learns to
    camp. Compares like for like over the same number of steps."""
    w = GlitchHunterWrapper(env, reward_mode="legacy_completion")

    w.reset()
    idle = 0.0
    for _ in range(40):
        _, r, term, _, _ = w.step(0)   # stand still
        idle += r
        if term:
            break

    w.reset()
    moving = 0.0
    for _ in range(40):
        _, r, term, _, _ = w.step(3)   # run right
        moving += r
        if term:
            break

    assert moving > idle, (
        f"idling ({idle:.2f}) scored >= moving ({moving:.2f})")


def test_coins_are_not_double_counted(env, patch_step):
    """The engine awards BOTH coin_total +1 and score +200 for a single coin.
    The wrapper subtracts the coin-attributable part out of the score delta so
    one coin cannot pay twice - that bug made camping under a brick more
    attractive than risking a pit.

    Drives the wrapper directly with a controlled info dict, so this asserts
    the wrapper's real arithmetic rather than restating the formula.
    """
    w = GlitchHunterWrapper(env, reward_mode="legacy_completion")
    w.reset()

    base = {'x_pos': 500, 'y_pos': 400, 'x_vel': 0.0, 'on_ground': True,
            'status': 'small', 'flag_get': False, 'powerup_active_count': 0,
            'nearest_powerup_dx': None, 'death_cause': None, 'is_dead': False}

    def fake_step(_action, info):
        patch_step(lambda a: (None, 0.0, False, False, dict(base, **info)))
        return w.step(0)

    # Settle position-based shaping so both runs below start level.
    fake_step(0, {'score': 0, 'coins': 0})
    w.last_score, w.last_coins, w.max_x_reached = 0, 0, 10_000

    # One coin: score +200 AND coins +1, as the engine reports it.
    coin_reward = fake_step(0, {'score': 200, 'coins': 1})[1]

    # Same score gain with NO coin (e.g. stomping an enemy) must pay MORE,
    # because none of it is discounted as coin double-counting.
    w.last_score, w.last_coins = 0, 0
    enemy_reward = fake_step(0, {'score': 200, 'coins': 0})[1]

    assert enemy_reward > coin_reward, (
        f"coin path ({coin_reward:.2f}) should be discounted below the "
        f"non-coin path ({enemy_reward:.2f}); double-counting has returned")
