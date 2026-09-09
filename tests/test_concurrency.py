"""Guards on the thread-safety work.

app.py runs Flask-SocketIO in threading mode, so click handlers execute on a
different OS thread from the frame loop. Two real bugs came out of that and
both are pinned here:

  1. A handler could destroy the pygame window mid-step. A 20-second stress
     test produced 9 crashes ("video system not initialized"). env_lock now
     serialises env access.
  2. Every Start spawned a thread. Hammering it produced 11 concurrent frame
     loops, which starved each other and made the server stop answering HTTP
     entirely. handle_start_testing() now retires the previous loop first.

These run without a live server - they exercise the primitives directly.
"""
import threading

import agent_logic


def test_env_lock_is_reentrant():
    """open_agent_window() takes env_lock and then calls into helpers that
    may take it again. A plain Lock would self-deadlock on the second
    acquire; it has to be an RLock."""
    assert isinstance(agent_logic.env_lock, type(threading.RLock()))
    with agent_logic.env_lock:
        acquired = agent_logic.env_lock.acquire(blocking=False)
        assert acquired, "env_lock must be reentrant"
        agent_logic.env_lock.release()


def test_close_window_waits_for_an_in_flight_step(env):
    """The core safety property: teardown cannot land in the middle of a
    step. Simulates the frame loop holding env_lock and asserts that
    close_agent_window() blocks until it is released, rather than yanking
    the display out from under it."""
    order = []
    holding = threading.Event()
    release = threading.Event()

    def fake_frame_loop():
        with agent_logic.env_lock:
            holding.set()
            order.append('step-start')
            release.wait(timeout=5)
            order.append('step-end')

    worker = threading.Thread(target=fake_frame_loop)
    worker.start()
    assert holding.wait(timeout=5), "worker never took the lock"

    closer_done = threading.Event()

    def closer():
        agent_logic.close_agent_window()
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


def test_single_frame_loop_invariant_is_enforced_in_source():
    """handle_start_testing() must retire the previous frame loop before
    starting a new one. Without this, rapid Start clicks stack OS threads
    until the server stops responding."""
    import pathlib
    src = pathlib.Path(agent_logic.__file__).with_name('app.py').read_text(
        encoding='utf-8')
    assert 'agent_thread' in src
    assert 'previous.join(' in src, (
        "the previous frame loop is no longer joined before a new one starts")
    assert 'THREAD_RETIRE_TIMEOUT' in src


def test_frame_loop_releases_lock_before_sleeping():
    """The loop must hold env_lock for one step only, never across the sleep
    - otherwise every Reset click waits a full frame period before it can
    tear anything down."""
    import pathlib
    src = pathlib.Path(agent_logic.__file__).with_name('app.py').read_text(
        encoding='utf-8')
    body = src[src.index('def background_agent_task'):]
    lines = body.splitlines()

    def find(needle):
        for i, line in enumerate(lines):
            if needle in line:
                return i
        raise AssertionError(f"{needle!r} not found in background_agent_task")

    def indent(i):
        return len(lines[i]) - len(lines[i].lstrip())

    next_line = find('next(agent_gen)')
    sleep_line = find('socketio.sleep(')

    # Anchor on the `with env_lock:` that actually wraps next() - the one
    # immediately above it. There is an earlier, unrelated env_lock block at
    # the top of the function guarding generator creation, and matching that
    # one instead would compare against the wrong indent entirely.
    lock_line = max(i for i, line in enumerate(lines[:next_line])
                    if 'with env_lock:' in line)

    assert lock_line < next_line < sleep_line, "next() must be inside the lock"
    # The sleep must sit at or outside the `with` statement's own indent -
    # i.e. it is NOT nested inside the locked block.
    assert indent(sleep_line) <= indent(lock_line), (
        "socketio.sleep() is nested inside `with env_lock:` - holding the "
        "lock across the sleep makes teardown block for a whole frame")


def test_server_binds_to_localhost_by_default():
    """Regression guard: app.py used to bind 0.0.0.0, exposing an
    unauthenticated dashboard to everyone on the local network. It must
    default to loopback, with the wider bind available only as an explicit
    opt-in via GLITCH_HUNTER_HOST.

    Comment lines are stripped before checking - app.py documents the old
    value in a comment, and matching that would make this fail for entirely
    the wrong reason.
    """
    import pathlib
    src = pathlib.Path(agent_logic.__file__).with_name('app.py').read_text(
        encoding='utf-8')
    code_lines = [line for line in src.splitlines()
                  if not line.lstrip().startswith('#')]
    code = chr(10).join(code_lines)

    assert "host='0.0.0.0'" not in code, "app.py binds all interfaces again"
    assert "GLITCH_HUNTER_HOST" in code, "the opt-in override is gone"
    assert "'127.0.0.1'" in code, "the loopback default is gone"
