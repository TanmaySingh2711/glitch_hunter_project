"""Guards on the BUG TRACKER detector.

Two things can go wrong with a bug detector and both are bad:
  1. it cries wolf during normal play, burying the panel in noise
  2. it stays silent when something real breaks
There is a test for each. There is also a test for the delivery mechanism,
which is subtle enough to be worth pinning down explicitly.
"""
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    MaxAndSkipObservation,
    ResizeObservation,
)


def test_quiet_during_normal_play(fresh):
    """No alerts from ordinary movement.

    Thresholds were set from a measured envelope of real trained play
    (|x_vel| max 13.2, y_pos -29..498, zero score/coin decreases). This is a
    smaller, model-free version of that check: plain forward movement must
    not trip anything.
    """
    alerts = []
    for i in range(200):
        _, _, term, _, info = fresh.step(4 if i % 3 else 1)
        if info.get('glitch_alert'):
            alerts.append(info['glitch_alert'])
        if term:
            fresh.reset()
    assert alerts == [], f"false positives during normal play: {alerts}"


def test_detects_score_running_backwards(fresh):
    """Score is monotonic within an episode; a decrease is a real accounting
    bug. Chosen for this test precisely because the engine will NOT quietly
    repair it, unlike e.g. teleporting Mario below the death plane (which
    level1's own check_for_mario_death corrects on the same update)."""
    for _ in range(3):
        fresh.step(1)
    fresh.game.state.game_info['score'] = 5000
    fresh.step(1)

    fresh.game.state.game_info['score'] = 100
    alert = None
    for _ in range(4):
        _, _, _, _, info = fresh.step(1)
        if info.get('glitch_alert'):
            alert = info['glitch_alert']
            break
    assert alert is not None, "score regression was not detected"
    assert "backwards" in alert


def test_same_glitch_is_not_re_detected_within_an_episode(fresh):
    """A fault must be DETECTED once per episode, not re-detected forever.

    Note what this does and does not assert. At this raw-env level the alert
    is deliberately visible for a short burst - GLITCH_ALERT_TTL substeps -
    because that TTL is what carries it through MaxAndSkipObservation, which
    discards 3 of every 4 info dicts (see the wrapper test below, which pins
    the "exactly once" guarantee at the level the agent and dashboard
    actually see). What matters here is that the burst is BOUNDED and never
    returns: without per-episode de-duplication a persistent fault would
    alert on every frame forever and flood the panel.
    """
    from custom_mario_env import GLITCH_ALERT_TTL

    for _ in range(3):
        fresh.step(1)
    fresh.game.state.game_info['score'] = 5000
    fresh.step(1)
    fresh.game.state.game_info['score'] = 100

    seen = []
    for i in range(40):
        _, _, term, _, info = fresh.step(1)
        if info.get('glitch_alert'):
            seen.append(i)
        if term:
            break

    assert seen, "score regression was never detected"
    # One detection plus its TTL re-attachments, and nothing after.
    assert len(seen) <= GLITCH_ALERT_TTL + 1, (
        f"alert repeated {len(seen)}x - de-duplication is not working")
    # The burst must be contiguous and then stop for good.
    assert seen == list(range(seen[0], seen[0] + len(seen))), (
        f"alert reappeared after stopping: {seen}")


def test_alert_survives_the_frame_skip_wrapper(env):
    """The delivery guarantee.

    MaxAndSkipObservation(skip=4) calls step() four times per agent decision
    and returns ONLY the last substep's info - the other three are thrown
    away. An alert raised on substep 1-3 would vanish without the TTL that
    re-attaches it. This test fails if that mechanism is removed.
    """
    wrapped = MaxAndSkipObservation(env, skip=4)
    wrapped = GrayscaleObservation(wrapped, keep_dim=False)
    wrapped = ResizeObservation(wrapped, (84, 84))
    wrapped = FrameStackObservation(wrapped, 4)

    wrapped.reset()
    for _ in range(4):
        wrapped.step(1)
    env.game.state.game_info['score'] = 5000
    wrapped.step(1)

    env.game.state.game_info['score'] = 100
    delivered = []
    for _ in range(8):
        _, _, term, trunc, info = wrapped.step(1)
        if info.get('glitch_alert'):
            delivered.append(info['glitch_alert'])
        if term or trunc:
            break
    assert len(delivered) == 1, (
        f"alert should reach the agent exactly once, got {len(delivered)}")
