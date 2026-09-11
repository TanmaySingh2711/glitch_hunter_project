"""Guards on the thread-safety work.

app.py runs Flask-SocketIO in threading mode, so click handlers execute on a
different OS thread from the frame loop. Two real bugs came out of that and
both are pinned here:

  1. A handler could destroy the pygame window mid-step. A 20-second stress
     test produced 9 crashes ("video system not initialized"). env_lock now
     serialises env access.
  2. Every Start spawned a thread. Hammering it produced 11 concurrent frame
     loops, which starved each other and made the server stop answering HTTP
     entirely.

A third came later: the handler threads also OWNED the pygame window, and
Windows destroys a window with its creating thread. All three are now closed
by one design - a single game thread owns the window and is the only frame
loop (dashboard_service.py, tested in test_dashboard_control.py).

These run without a live server - they exercise the primitives directly.
"""
import threading

import pytest

import dashboard_backend


def test_env_lock_is_reentrant():
    """open_agent_window() takes env_lock and then calls into helpers that
    may take it again. A plain Lock would self-deadlock on the second
    acquire; it has to be an RLock."""
    assert isinstance(dashboard_backend.env_lock, type(threading.RLock()))
    with dashboard_backend.env_lock:
        acquired = dashboard_backend.env_lock.acquire(blocking=False)
        assert acquired, "env_lock must be reentrant"
        dashboard_backend.env_lock.release()


def test_close_window_waits_for_an_in_flight_step(env):
    """The core safety property: teardown cannot land in the middle of a
    step. Simulates the frame loop holding env_lock and asserts that
    close_agent_window() blocks until it is released, rather than yanking
    the display out from under it."""
    order = []
    holding = threading.Event()
    release = threading.Event()

    def fake_frame_loop():
        with dashboard_backend.env_lock:
            holding.set()
            order.append('step-start')
            release.wait(timeout=5)
            order.append('step-end')

    worker = threading.Thread(target=fake_frame_loop)
    worker.start()
    assert holding.wait(timeout=5), "worker never took the lock"

    closer_done = threading.Event()

    def closer():
        dashboard_backend.close_agent_window()
        order.append('closed')
        closer_done.set()

    closer_thread = threading.Thread(target=closer)
    closer_thread.start()

    # While the step holds the lock, the close MUST NOT have happened.
    assert not closer_done.wait(timeout=0.5), (
        "close_agent_window() ran during a step - env_lock is not protecting it")
    assert 'closed' not in order

    release.set()
    worker.join(timeout=5)
    assert closer_done.wait(timeout=5), "close never completed after release"
    closer_thread.join(timeout=5)

    # The step finished in full before teardown ran.
    assert order == ['step-start', 'step-end', 'closed'], order

    env.open_window()   # leave the shared fixture usable for other tests


def test_app_handlers_never_touch_the_window_themselves():
    """Every socket handler runs on its own short-lived thread, and a window
    created or driven from one of those dies with it (Windows destroys a
    window when its creating thread exits - measured). So app.py must only
    post commands to the game thread (dashboard_service.py); the frame loop,
    the pre-load and every window call live there. The behaviour - one frame
    loop, clicks landing between steps - is tested in test_dashboard_control.
    """
    import pathlib
    src = pathlib.Path(dashboard_backend.__file__).with_name('app.py').read_text(
        encoding='utf-8')
    code = chr(10).join(line for line in src.splitlines()
                        if not line.lstrip().startswith('#'))
    assert 'GameWindowService' in code
    for forbidden in ('open_agent_window', 'close_agent_window', 'next(',
                      'run_mario_agent', 'start_background_task', 'import pygame'):
        assert forbidden not in code, f"app.py drives the window directly: {forbidden}"


def test_server_binds_to_localhost_by_default():
    """Regression guard: app.py used to bind 0.0.0.0, exposing an
    unauthenticated dashboard to everyone on the local network. It must
    default to loopback, with the wider bind available only as an explicit
    opt-in via GLITCH_HUNTER_HOST. Checked on the function main() actually
    calls, not on the source text."""
    import app
    assert app.bind_address({}) == ('127.0.0.1', 5000)
    assert app.bind_address({'GLITCH_HUNTER_HOST': '0.0.0.0',
                             'GLITCH_HUNTER_PORT': '8080'}) == ('0.0.0.0', 8080)


def test_a_bad_port_is_refused_with_a_reason():
    import app
    for bad in ('http', '0', '70000'):
        with pytest.raises(SystemExit, match='GLITCH_HUNTER_PORT'):
            app.bind_address({'GLITCH_HUNTER_PORT': bad})
