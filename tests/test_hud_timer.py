"""The QA clock runs long; the TIME box keeps showing the legacy clock.

The engine keeps ONE clock, OverheadInfo.time, and every rule reads it: the
timeout, the time-to-score countdown at the castle, the hurry-up music. QA
mode sets it to 1,600 units. What the TIME box DRAWS leaves the 1,199 extra
units out: it shows the clock a legacy episode would show at the same point
(401 counting down), holds at 001 once legacy would have timed out, and
reads 000 on exactly the frame the real clock runs out. So until legacy's
own timeout, a QA frame is a legacy frame, pixel for pixel.

These run on the real engine: the claims are about what the real engine
draws and when it really times out. The full-length versions - the whole
39,126-substep QA episode with the box checked on every substep, and its
frames matched to legacy's for legacy's entire lifetime - are in
test_measured_timer_caps (test_episode_lifecycle.py).
"""
import hashlib

import numpy as np

from agent_logic import GlitchHunterWrapper
from exploration import config
from exploration.coverage import SpatialCoverage

OFFSET = config.QA_EPISODE_TIME_UNITS - config.ENGINE_TIME_UNITS_DEFAULT

# sha256 of the 600 raw frames the UNMODIFIED engine drew in legacy mode,
# holding NOOP from reset. Recorded before this change, and identical under
# a real window and SDL's dummy driver.
LEGACY_NOOP_600_SHA256 = (
    "f7f3408f65786a0e266bd4fdc457a350c58d0d769d0c4a3e8bdc5c7c3adeecc9")


def shown(t, offset=OFFSET):
    """What the TIME box should read for clock value t."""
    return t if offset == 0 or t <= 0 else max(1, t - offset)


def _qa(env):
    return GlitchHunterWrapper(
        env, reward_mode="qa_exploration",
        coverage=SpatialCoverage(
            testable_mask=np.ones((config.GRID_H, config.GRID_W), dtype=bool)))


def _hud(env):
    return env.game.state.overhead_info_display


def drawn(env):
    """The digits actually in the TIME box, decoded from its glyph sprites -
    what gets blitted, not what any attribute claims."""
    hud = _hud(env)
    glyph = {id(img): ch for ch, img in hud.image_dict.items()}
    return ''.join(glyph[id(c.image)] for c in hud.count_down_images)


def _noop_600_sha256(step):
    h, infos = hashlib.sha256(), []
    for _ in range(600):
        o, _r, done, _t, info = step(0)
        h.update(o.tobytes())
        infos.append(info)
        assert not done
    return h.hexdigest(), infos


def test_qa_draws_the_legacy_clock_not_the_extension(fresh):
    w = _qa(fresh)
    w.reset()
    info = w.step(0)[4]
    assert info['time_left'] == config.QA_EPISODE_TIME_UNITS, "the clock itself moved"
    assert _hud(fresh).display_time_offset == OFFSET
    assert info['hud_time'] == config.ENGINE_TIME_UNITS_DEFAULT
    assert drawn(fresh) == '401'


def test_qa_frames_are_the_legacy_frames(fresh):
    """Before this change these frames carried a 4-digit '1600' where the 6M
    brain had only ever seen three digits. Now a QA episode draws the SAME
    600 frames, byte for byte, that the unmodified engine drew in legacy -
    while its own clock is 1,199 units ahead of what it shows."""
    w = _qa(fresh)
    w.reset()
    digest, infos = _noop_600_sha256(w.step)
    assert digest == LEGACY_NOOP_600_SHA256
    for info in infos:
        assert info['time_left'] - info['hud_time'] == OFFSET


def test_hud_counts_down_then_holds_at_001(fresh):
    """Jumps the clock to 3 units above the offset instead of playing 1,197
    units down to it: the box reads 3, 2, 1 like legacy's last units, then
    holds 001 while the clock keeps running - the same number in info and in
    the glyphs on every substep."""
    fresh.episode_time_units = config.QA_EPISODE_TIME_UNITS
    fresh.reset()
    _hud(fresh).time = OFFSET + 3
    clock = []
    for _ in range(8 * 25):
        info = fresh.step(0)[4]
        t = info['time_left']
        assert info['hud_time'] == shown(t)
        assert drawn(fresh) == f"{shown(t):03d}"
        clock.append(t)
    assert clock[0] == OFFSET + 3 and clock[-1] < OFFSET - 3, "never reached the hold"


def test_timeout_follows_the_clock_not_the_box(fresh):
    """A small offset makes the hold short enough to play out: the box runs
    3, 2, 1 and then holds 001 while the clock still has 4 units. The timeout
    must wait for the clock, and the box must not reach 000 a frame early."""
    fresh.reset()
    hud = _hud(fresh)
    hud.display_time_offset, hud.time = 4, 7
    sub = 0
    while True:
        _o, _r, done, _t, info = fresh.step(0)
        sub += 1
        if done:
            break
        assert info['hud_time'] == shown(info['time_left'], 4)
        assert info['hud_time'] > 0 and drawn(fresh) != '000', "000 before the timeout"
    assert info['death_cause'] == 'timeout'
    assert info['time_left'] == 0 and info['hud_time'] == 0 and drawn(fresh) == '000'
    # ~24.45 substeps a unit: timing out on the box's 3 would be near 73.
    assert sub > 6 * 24, f"timed out after {sub} substeps - on the box, not the clock"


def test_legacy_draws_exactly_what_it_drew_before(fresh):
    """The leak guard and the regression pin in one. A QA wrapper configures
    the shared env first; a legacy wrapper then takes it over; and the legacy
    engine must draw the same 600 frames, byte for byte, as the unmodified
    engine did. (The base env is stepped directly: legacy's own stuck rule
    would end a NOOP hold early, and the reward wrapper never touches pixels.)"""
    qa = _qa(fresh)
    qa.reset()
    qa.step(0)
    GlitchHunterWrapper(fresh, reward_mode="legacy_completion", attach_coverage=False)
    fresh.reset()
    assert _hud(fresh).display_time_offset == 0
    digest, infos = _noop_600_sha256(fresh.step)
    assert digest == LEGACY_NOOP_600_SHA256
    for info in infos:
        assert info['hud_time'] == info['time_left']
    assert infos[-1]['time_left'] == 377


def test_the_hud_offset_changes_pixels_only(fresh):
    """The same QA episode twice: as shipped, and with the offset removed so
    the box draws '1600' as it did before. Rewards, every reward channel, the
    phase, the lifecycle and every info key but hud_time come out identical;
    only the frames differ."""
    script = [3] * 80 + [4] * 20 + [1] * 100 + [0] * 200

    def play(remove_offset):
        w = _qa(fresh)
        w.reset()
        if remove_offset:
            _hud(fresh).display_time_offset = 0
        frames, rest = [], []
        for a in script:
            o, r, done, _t, info = w.step(a)
            frames.append(o)
            rest.append((r, done, {k: v for k, v in info.items() if k != 'hud_time'}))
            if done:
                break
        return frames, rest, {p: dict(c) for p, c in w.ep_channels.items()}

    f_shipped, shipped, ch_shipped = play(remove_offset=False)
    f_before, before, ch_before = play(remove_offset=True)
    np.testing.assert_equal(shipped, before)
    assert ch_shipped == ch_before
    assert any(not np.array_equal(a, b) for a, b in zip(f_shipped, f_before, strict=True)), (
        "removing the offset never changed a pixel - this test is not testing it")
