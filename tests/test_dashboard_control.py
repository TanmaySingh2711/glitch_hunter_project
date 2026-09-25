"""The dashboard's game window: who owns it, and what the user controls.

Three layers, tested separately so the suite does not depend on a physical
display or window manager:

  * GameWindowService, the state machine, driven through a fake backend that
    records every call and the thread it ran on.
  * game_window.centered_origin, the placement arithmetic, as pure numbers.
  * CustomMarioEnv's window methods on the real session window, skipped where
    the platform cannot hide or minimise a window.
"""
import hashlib
import threading
import time

import numpy as np
import pygame as pg
import pytest

import dashboard_service as ds
import game_window


class FakeBackend:
    """Records (call, thread name); models the window as closed/hidden/open."""

    def __init__(self):
        self.calls = []
        self.window = 'closed'
        self.centered = 0
        self.close_requested = False
        self.sessions = 0
        self.step_delay = 0.0
        self.run_end_at = None           # step on which each run ends (clean game)
        self.run_end = {'end_reason': 'level_complete',
                        'report': {'run_id': 'RUN-1', 'headline': 'No Bugs Found'}}
        self.lock = threading.Lock()

    def _rec(self, name):
        with self.lock:
            self.calls.append((name, threading.current_thread().name))

    def preload(self):
        self._rec('preload')
        self.window = 'hidden'

    def open_window(self):
        self._rec('open_window')
        was, self.window = self.window, 'open'
        if was != 'open':
            self.centered += 1               # created or re-shown: centred
        return {'closed': 'created', 'hidden': 'shown'}.get(was, 'focused')

    def hide_window(self):
        self._rec('hide_window')
        self.window = 'hidden'

    def close_window(self):
        self._rec('close_window')
        self.window = 'closed'

    def poll_close_request(self):
        self._rec('poll')
        if self.close_requested:
            self.close_requested = False
            return True
        return False

    def new_session(self):
        self._rec('new_session')
        self.sessions += 1
        sid = self.sessions

        def run():
            n = 0
            while True:
                n += 1
                if self.step_delay:
                    time.sleep(self.step_delay)
                item = {'frame': b'', 'log': f"session {sid} step {n}"}
                if self.run_end_at and n % self.run_end_at == 0:
                    item['run_end'] = self.run_end
                yield item
        return run()

    def stop_audio(self):
        self._rec('stop_audio')

    def begin_incident_session(self):
        self._rec('begin_incident_session')

    def switch_game(self, variant):
        self._rec(f'switch_game:{variant}')
        if variant == 'broken':
            raise RuntimeError("cannot load broken")
        changed, self.variant = variant != getattr(self, 'variant', 'mario_clean'), variant
        if changed:
            self.window = 'hidden'           # the new game pre-loads hidden
        return changed

    def names(self):
        with self.lock:
            return [c for c, _t in self.calls if c != 'poll']


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


@pytest.fixture
def svc():
    backend, events = FakeBackend(), []
    s = ds.GameWindowService(backend, emit=lambda e, p: events.append((e, p)),
                             log=lambda *a: None)
    s.start(timeout=5)
    s.fake, s.events = backend, events
    yield s
    s.shutdown()


def last_log(s):
    logs = [p['log'] for e, p in s.events if e == 'agent_log']
    return logs[-1] if logs else None


def run_for(s, steps):
    start = s.steps
    assert wait_for(lambda: s.steps >= start + steps), "the game loop is not stepping"


# ══════════════════════════════════════════════════════════════════════════
# Ownership
# ══════════════════════════════════════════════════════════════════════════
def test_every_window_call_runs_on_the_one_game_thread(svc):
    """The whole fix. Clicks arrive from other threads; the window is only
    ever touched from the thread that created it (the pre-load's)."""
    def click(fn):
        t = threading.Thread(target=fn)
        t.start()
        t.join()

    click(svc.start_testing)
    run_for(svc, 3)
    svc.fake.close_requested = True
    assert wait_for(lambda: not svc.testing)
    for fn in (svc.start_testing, svc.stop_testing, svc.reset, svc.start_testing):
        click(fn)
    svc.wait_idle()
    threads = {t for _c, t in svc.fake.calls}
    assert threads == {ds.THREAD_NAME}, threads
    assert svc.fake.calls[0][0] == 'preload'


def test_rapid_clicks_never_start_a_second_frame_loop(svc):
    """The old server stacked 11 frame loops under a click storm."""
    def storm():
        for i in range(60):
            (svc.start_testing, svc.stop_testing, svc.reset)[i % 3]()
    threads = [threading.Thread(target=storm) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert svc.wait_idle()
    loops = [t for t in threading.enumerate() if t.name == ds.THREAD_NAME and t.is_alive()]
    assert len(loops) == 1 and svc.alive


def test_a_click_lands_between_steps(svc):
    """Nothing is held across the pacing wait, so a Stop is handled after at
    most the step already in progress."""
    svc.fake.step_delay = 0.05
    svc.start_testing()
    run_for(svc, 2)
    before = svc.steps
    svc.stop_testing()
    assert svc.wait_idle()
    assert not svc.testing and svc.steps <= before + 1


# ══════════════════════════════════════════════════════════════════════════
# Stop / X / Start / Reset / refresh
# ══════════════════════════════════════════════════════════════════════════
def test_stop_pauses_and_leaves_the_window_alone(svc):
    svc.start_testing()
    run_for(svc, 3)
    n_calls = len(svc.fake.names())
    svc.stop_testing()
    svc.wait_idle()
    paused_at = svc.steps
    time.sleep(0.2)
    assert not svc.testing and svc.steps == paused_at, "still stepping after Stop"
    assert svc.fake.window == 'open', "Stop hid, minimised or closed the window"
    assert set(svc.fake.names()[n_calls:]) <= {'stop_audio'}


def test_the_window_stays_responsive_while_paused(svc):
    """Paused, the game thread keeps pumping the window's events - which is
    what keeps it movable/minimisable and its X button live."""
    svc.start_testing()
    run_for(svc, 1)
    svc.stop_testing()
    svc.wait_idle()
    polls = sum(1 for c, _t in svc.fake.calls if c == 'poll')
    time.sleep(0.3)
    assert sum(1 for c, _t in svc.fake.calls if c == 'poll') >= polls + 3
    svc.fake.close_requested = True                  # X while paused
    assert wait_for(lambda: svc.fake.window == 'hidden')


def test_x_while_running_hides_pauses_and_keeps_the_session(svc):
    svc.start_testing()
    run_for(svc, 5)
    session = svc.session
    svc.fake.close_requested = True
    assert wait_for(lambda: not svc.testing)
    assert svc.fake.window == 'hidden'
    assert svc.session is session, "closing the window discarded the session"
    assert ('testing_paused', {'reason': 'window_closed'}) in svc.events
    assert 'close_window' not in svc.fake.names(), "X destroyed the window"


def test_start_after_x_reopens_centred_and_resumes_the_same_session(svc):
    svc.start_testing()
    run_for(svc, 4)
    svc.fake.close_requested = True
    assert wait_for(lambda: not svc.testing)
    svc.wait_idle()
    stopped_at = int(last_log(svc).rsplit(' ', 1)[1])
    centred = svc.fake.centered

    svc.start_testing()
    run_for(svc, 2)
    assert svc.fake.window == 'open'
    assert svc.fake.centered == centred + 1, "the re-shown window was not centred"
    assert svc.fake.sessions == 1, "Start restarted the campaign"
    logs = [p['log'] for e, p in svc.events if e == 'agent_log']
    assert f"session 1 step {stopped_at + 1}" in logs, "did not resume where it paused"


def test_start_on_an_open_window_does_not_recentre_it(svc):
    """Centred when created or re-shown - never pulled back once open."""
    svc.start_testing()
    run_for(svc, 1)
    assert svc.fake.centered == 1
    svc.stop_testing()
    svc.start_testing()
    svc.wait_idle()
    assert svc.fake.centered == 1


def test_reset_ends_the_session_and_the_next_start_is_fresh(svc):
    svc.start_testing()
    run_for(svc, 3)
    svc.reset()
    svc.wait_idle()
    assert svc.session is None and svc.fake.window == 'closed'
    assert 'begin_incident_session' in svc.fake.names()     # the Bug Tracker starts empty
    svc.start_testing()
    run_for(svc, 1)
    assert svc.fake.sessions == 2 and last_log(svc).startswith("session 2 step")


def test_a_slow_step_keeps_the_pace_instead_of_halving_it(svc):
    """A 30 ms step (a laptop on battery) still gets ~30 steps/s. The old
    rule - wait as long again as the step took - gave it 16."""
    svc.fake.step_delay = 0.030
    svc.start_testing()
    run_for(svc, 3)
    start, t0 = svc.steps, time.monotonic()
    time.sleep(1.5)
    rate = (svc.steps - start) / (time.monotonic() - t0)
    svc.stop_testing()
    assert 23 <= rate <= 32, f"{rate:.1f} steps/s"


def test_a_finished_clean_run_stops_testing_and_keeps_its_result(svc):
    svc.fake.run_end_at = 3
    svc.start_testing()
    assert wait_for(lambda: not svc.testing and svc.steps >= 3)
    svc.wait_idle()
    assert svc.steps == 3, "testing went on after the run ended"
    assert svc.pause_reason == 'level_complete'
    assert svc.status()['run_result'] == svc.fake.run_end
    assert ('testing_paused', {'reason': 'level_complete'}) in svc.events
    assert ('run_finished', svc.fake.run_end) in svc.events


def test_start_after_a_finished_run_plays_the_next_run(svc):
    svc.fake.run_end_at = 3
    svc.start_testing()
    assert wait_for(lambda: not svc.testing and svc.steps >= 3)
    svc.start_testing()
    run_for(svc, 1)
    assert svc.run_result is None and ('run_cleared', {}) in svc.events
    assert svc.fake.sessions == 1 and last_log(svc).startswith("session 1 step")


def test_reset_after_a_finished_run_clears_it_and_the_next_start_is_fresh(svc):
    svc.fake.run_end = {'end_reason': 'death', 'report': None}
    svc.fake.run_end_at = 2
    svc.start_testing()
    assert wait_for(lambda: not svc.testing and svc.steps >= 2)
    svc.wait_idle()
    assert svc.pause_reason == 'mario_died'
    svc.reset()
    svc.wait_idle()
    assert svc.run_result is None and svc.status()['run_result'] is None
    svc.start_testing()
    run_for(svc, 1)
    assert svc.fake.sessions == 2


def test_switching_the_game_resets_first_then_loads_the_other_game(svc):
    svc.start_testing()
    run_for(svc, 3)
    svc.switch_game('mario_bugged')
    svc.wait_idle()
    names = svc.fake.names()
    assert names.index('close_window') < names.index('switch_game:mario_bugged')
    assert not svc.testing and svc.session is None and svc.pause_reason == 'reset'
    assert ('game_switched', {'variant': 'mario_bugged', 'ok': True}) in svc.events
    svc.start_testing()
    run_for(svc, 1)
    assert svc.fake.sessions == 2 and last_log(svc).startswith("session 2 step")


def test_a_failed_game_switch_is_reported_not_raised(svc):
    svc.switch_game('broken')
    svc.wait_idle()
    assert svc.alive
    assert any(e == 'game_switched' and p['ok'] is False and 'broken' in p['error']
               for e, p in svc.events)


def test_a_browser_refresh_pauses_and_keeps_everything(svc):
    svc.start_testing()
    run_for(svc, 3)
    session = svc.session
    svc.client_disconnected()
    svc.client_connected()
    svc.wait_idle()
    assert not svc.testing
    assert svc.fake.window == 'open' and svc.session is session
    svc.start_testing()
    run_for(svc, 1)
    assert svc.fake.sessions == 1


def test_a_failing_step_pauses_and_tells_the_browser():
    class Broken(FakeBackend):
        def new_session(self):
            self.sessions += 1

            def run():
                yield {'frame': b'', 'log': 'ok'}
                raise RuntimeError("boom")
            return run()
    events, backend = [], Broken()
    s = ds.GameWindowService(backend, emit=lambda e, p: events.append((e, p)),
                             log=lambda *a: None)
    s.start(timeout=5)
    try:
        s.start_testing()
        assert wait_for(lambda: ('testing_paused', {'reason': 'error'}) in events)
        assert not s.testing and s.alive
        # The crashed session is dropped, so the next Start plays a fresh one
        # rather than reporting that the dead one has ended.
        assert s.session is None
        s.start_testing()
        assert wait_for(lambda: backend.sessions == 2)
        assert ('testing_paused', {'reason': 'session_ended'}) not in events
    finally:
        s.shutdown()


# ══════════════════════════════════════════════════════════════════════════
# Placement arithmetic
# ══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize(('area', 'size', 'want'), [
    ((0, 0, 1920, 1040), (816, 639), (552, 200)),           # laptop, taskbar below
    ((-1920, 0, 0, 1080), (816, 639), (-1368, 220)),        # a monitor left of the primary
    ((0, 40, 1366, 768), (816, 639), (275, 84)),            # taskbar on top
    ((0, 0, 800, 500), (816, 639), (0, 0)),                 # window bigger than the area
])
def test_centred_origin(area, size, want):
    assert game_window.centered_origin(area, size) == want


def test_placement_degrades_to_a_no_op_without_a_window(monkeypatch):
    monkeypatch.setattr(game_window, '_hwnd', lambda: None)
    assert game_window.center_on_current_display() is None
    assert game_window.hide() is False and game_window.show() is False
    assert game_window.is_minimized() is False and game_window.bring_to_front() is False


def test_every_new_window_is_asked_for_the_centre(monkeypatch):
    """No caller-left position may win over centring."""
    monkeypatch.setenv('SDL_VIDEO_WINDOW_POS', '50,50')
    monkeypatch.delenv('SDL_VIDEO_CENTERED', raising=False)
    game_window.request_centered_creation()
    import os
    assert 'SDL_VIDEO_WINDOW_POS' not in os.environ
    assert os.environ['SDL_VIDEO_CENTERED'] == '1'


# ══════════════════════════════════════════════════════════════════════════
# The real window (the shared session env)
# ══════════════════════════════════════════════════════════════════════════
def _can_hide(env):
    env.open_window()
    env.hide_window()
    ok = env.window_state() == 'hidden'
    env.open_window()
    return ok


def test_real_window_centres_on_create_and_reshow_only(env, monkeypatch):
    if not _can_hide(env):
        pytest.skip("this video driver cannot hide a window")
    centred = []
    monkeypatch.setattr(game_window, 'center_on_current_display',
                        lambda: centred.append(1))
    assert env.open_window() == 'focused' and centred == []
    env.hide_window()
    assert env.window_state() == 'hidden'
    assert env.open_window() == 'shown' and len(centred) == 1
    env.reset()
    for _ in range(30):
        env.step(1)
    x = env.game.state.mario.rect.x
    env.close_window()
    assert env.window_state() == 'closed'
    assert env.open_window() == 'created' and len(centred) == 2
    # The live game follows the recreated window: the episode continues from
    # where it was, without a reset.
    assert env.game.screen is pg.display.get_surface()
    info = env.step(1)[4]
    assert info['mario_rect'][0] >= x, "the episode restarted"


def test_start_puts_the_window_in_the_taskbar_minimised(env):
    """The dashboard's Start: a created or re-shown window goes to the
    taskbar, minimised and inactive; one already open is left alone."""
    import sys
    if not _can_hide(env):
        pytest.skip("this video driver cannot hide a window")
    win32 = sys.platform == 'win32' and bool(game_window._hwnd())
    try:
        env.hide_window()
        assert env.open_window(minimized=True) == 'shown'
        assert env.window_state() == 'open'
        if win32:
            assert game_window.is_minimized(), "the re-shown window is not minimised"
        assert env.open_window(minimized=True) == 'unchanged'
        env.close_window()
        assert env.open_window(minimized=True) == 'created'
        if win32:
            assert game_window.is_minimized(), "the new window is not minimised"
        assert env.game.screen is pg.display.get_surface()
    finally:
        env.open_window()                # restored for the tests after this one
    if win32:
        assert not game_window.is_minimized()


def test_the_sdl_window_wrapper_outlives_every_call(env):
    """pygame keeps a raw pointer to this wrapper inside the SDL window; one
    freed while the window lived crashed the dashboard (access violation on
    the next window event). The same live wrapper must be handed back."""
    env.open_window()
    first = game_window._sdl_window()
    if first is None:
        pytest.skip("no SDL window wrapper on this platform")
    assert game_window._sdl_window() is first
    assert any(w is first for _s, w in game_window._SDL_WINDOWS)


def test_the_x_button_reaches_poll_close_request(env):
    env.open_window()
    pg.event.post(pg.event.Event(pg.QUIT))
    assert env.poll_close_request() is True
    assert env.poll_close_request() is False, "the close request was not consumed"


def _play(env, actions):
    env.reset()
    out = []
    for a in actions:
        obs, _r, _d, _t, info = env.step(a)
        out.append((hashlib.sha1(obs.tobytes()).hexdigest(), info['mario_rect']))
    return out


def test_hiding_or_minimising_changes_nothing_the_agent_sees(env):
    """Closing (hiding) or minimising the game UI must not corrupt anything:
    the observation and the game state are identical, frame for frame."""
    if not _can_hide(env):
        pytest.skip("this video driver cannot hide a window")
    actions = list(np.random.default_rng(7).integers(0, 10, size=120))
    visible = _play(env, actions)
    try:
        env.hide_window()
        hidden = _play(env, actions)
        env.open_window()
        w = game_window._sdl_window()
        w.minimize()
        minimised = _play(env, actions)
    finally:
        env.open_window()
    assert hidden == visible, "hiding the window changed what the agent sees"
    assert minimised == visible, "minimising the window changed what the agent sees"


def test_a_real_window_lands_inside_the_cursor_monitors_work_area(env):
    """The Win32 half of the placement rule, on the live window."""
    import sys
    env.open_window()
    area = game_window.current_work_area()
    if sys.platform != 'win32' or area is None or not game_window._hwnd():
        pytest.skip("needs a real Win32 window and monitor")
    origin = game_window.center_on_current_display()
    assert origin is not None
    left, top, right, bottom = area
    assert left <= origin[0] <= right and top <= origin[1] <= bottom
    assert isinstance(game_window.bring_to_front(), bool)
    assert game_window.is_minimized() is False
