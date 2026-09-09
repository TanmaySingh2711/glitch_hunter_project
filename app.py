import os
import sys
# Python fully buffers stdout when it isn't attached to an interactive
# terminal (e.g. redirected to a log file, or launched from another tool) -
# print() calls just sit in the buffer until it fills or the process exits.
# Reconfiguring to line-buffering here means every print (this file's own,
# plus everything it imports - agent_logic.py's [STREAM FPS] line, the
# startup pre-load messages below) shows up immediately, which matters for
# actually being able to watch this server's real-time behavior rather
# than only seeing output in one batch on exit.
#
# encoding='utf-8' is not cosmetic: on Windows this console defaults to
# cp1252, which cannot encode the emoji used in the bug-alert log lines -
# printing one raises UnicodeEncodeError and kills whatever was running.
# Verified directly: printing '\U0001f6a8' under cp1252 raises
# "'charmap' codec can't encode character". Forcing UTF-8 makes every log
# line printable regardless of the machine's console codepage.
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import threading
import traceback
import time
from flask import Flask, render_template
from flask_socketio import SocketIO
# agent_logic pulls in custom_mario_env, which sets SDL_AUDIODRIVER before
# pygame loads - so it has to be imported before pygame is used here.
from agent_logic import (run_mario_agent, open_agent_window,
                         close_agent_window, env_lock)
import pygame as pg

app = Flask(__name__)

# ─── async_mode='threading' (was 'eventlet') ───
# eventlet is unmaintained and Flask-SocketIO's own maintainer now
# recommends the threading mode as the default. It also removes a failure
# mode this project actually hit: eventlet runs every greenlet on ONE OS
# thread and only switches at I/O points it has monkey-patched, so a
# blocking CPU/GPU call (PPO.load(), or a slow env.step()) stalls the whole
# server - including the heartbeat that tells the browser the connection is
# alive. Real OS threads do not share that problem: the frame loop can sit
# on the GPU without stopping the server from answering anyone.
socketio = SocketIO(app, async_mode='threading')

# With real threads, the shared state below is genuinely concurrent - the
# frame loop reads it from its own thread while click handlers write to it
# from theirs. Under eventlet this was safe by accident (one thread, and
# switches only ever happened at explicit yield points). `task_epoch += 1`
# is a read-modify-write and is NOT atomic, so two clicks arriving together
# could lose an increment and leave a stale frame loop running. This lock
# makes each state transition indivisible.
state_lock = threading.RLock()

# Cache-busting: appended as ?v=... on static asset URLs (see index.html)
# so a browser that already cached an old style.css/main.js is forced to
# fetch the current one after every restart, instead of silently showing a
# stale page until the user thinks to hard-refresh.
ASSET_VERSION = str(int(time.time()))

test_running = False
agent_gen = None
task_epoch = 0

# The single live frame-loop thread, or None.
#
# ─── WHY THIS EXISTS ───
# Every "Start Testing" used to call start_background_task() unconditionally.
# Under eventlet that was harmless: greenlets are cheap and a superseded one
# died at its next yield. With real OS threads it is not - a stress test
# firing start/stop/reset from 4 clients produced ELEVEN consecutive
# START_TESTING handlers, each spawning a thread that then queued up on
# env_lock. The server pinned a core and stopped answering HTTP entirely.
#
# There must only ever be ONE frame loop. handle_start_testing() now retires
# the previous one (bump the epoch, then join it) before starting another.
agent_thread = None

# How long to wait for a superseded frame loop to notice and exit. It only
# has to finish the step it is on, which is single-digit milliseconds; the
# generous ceiling is purely so a pathological case degrades into "skip this
# click" instead of blocking the handler forever.
THREAD_RETIRE_TIMEOUT = 5.0

@app.route('/')
def index():
    return render_template('index.html', asset_version=ASSET_VERSION)


@app.route('/healthz')
def healthz():
    """Liveness + thread census.

    Exists because a stress test once wedged this server by spawning an
    unbounded number of frame-loop threads, and there was no way to see that
    happening from outside. `frame_loops` must never exceed 1; anything more
    means the single-frame-loop invariant in handle_start_testing() has
    regressed.
    """
    frame_loops = sum(
        1 for t in threading.enumerate()
        if t.is_alive() and getattr(t, "_target", None) is background_agent_task
    )
    return {
        "status": "ok",
        "testing": test_running,
        "threads_total": threading.active_count(),
        "frame_loops": frame_loops,
    }

def background_agent_task(epoch):
    global test_running, agent_gen
    # Creating the generator touches the env, so it belongs under the lock
    # too - otherwise a Reset arriving right now could null out agent_gen
    # between this check and the assignment.
    with env_lock:
        if agent_gen is None:
            agent_gen = run_mario_agent()

    target_frame_time = 1.0 / 60.0

    # ─── HALF SPEED ───
    # Each yielded item here is already 4 physics ticks of in-game time
    # (MaxAndSkipObservation(skip=4) in agent_logic.py repeats the chosen
    # action for 4 frames before this loop sees the next one). On a fast
    # plugged-in machine this loop was hitting the 60fps target_frame_time
    # ceiling above, meaning 4 game-ticks were being pushed through every
    # 1/60s of real time - about 4x real-time game speed, which read as
    # "too fast". On a slower/unplugged machine it instead ran flat-out,
    # compute-bound, at whatever rate env.step()+render+encode could
    # manage - also faster than intended relative to how much game-time
    # each of those steps carries.
    #
    # Fix: every iteration, sleep for the SAME length the iteration just
    # took (or target_frame_time, whichever is bigger) on top of its own
    # elapsed time - doubling the real-world time between steps. This
    # halves the effective game-speed relative to whatever this loop
    # would otherwise have achieved on THIS machine right now, whether
    # that baseline was the 60fps ceiling (plugged in) or a slower
    # compute-bound rate (on battery) - unlike a fixed lower fps target,
    # it adapts automatically to either case instead of only fixing one.

    # NOTE: this is a `while` loop driving next() by hand rather than a
    # `for item in agent_gen`. That is deliberate - it is the only way to
    # hold env_lock for exactly the duration of ONE step and release it
    # before sleeping. A `for` loop hides the next() call, so the lock would
    # have to wrap the whole body including the sleep, and every Reset click
    # would then block for a full frame period instead of a few milliseconds.
    try:
        last_t = time.time()
        while True:
            if not test_running or task_epoch != epoch:
                break

            # Advance the game under the lock, so a concurrent Reset or
            # disconnect cannot tear down the pygame window mid-step.
            with env_lock:
                # Re-checked INSIDE the lock: this thread may have been
                # waiting here while a handler superseded it, in which case
                # stepping the env now would resurrect a window the user
                # just closed.
                if not test_running or task_epoch != epoch:
                    break
                try:
                    item = next(agent_gen)
                except StopIteration:
                    break

            # Emitting and sleeping happen OUTSIDE the lock - neither needs
            # the env, and holding it across the sleep is what would make
            # teardown feel laggy.
            socketio.emit('video_frame', {'frame': item['frame']})
            if item['log']:
                socketio.emit('agent_log', {'log': item['log']})

            now = time.time()
            elapsed = now - last_t
            baseline_period = max(elapsed, target_frame_time)
            sleep_time = (2 * baseline_period) - elapsed
            socketio.sleep(max(sleep_time, 0))
            last_t = time.time()
    except Exception as e:
        print(f"Error in background task: {e}")
        traceback.print_exc()
    finally:
        if task_epoch == epoch:
            test_running = False

def stop_all_music():
    # Best-effort: the mixer may not be initialized at all (audio is forced
    # to the "dummy" driver in custom_mario_env.py), so a failure here must
    # never take down a click handler.
    try:
        if pg.mixer.get_init():
            pg.mixer.music.stop()
            pg.mixer.stop()
    except Exception:
        pass


def discard_agent_gen():
    """Drops the current agent generator and closes it explicitly.

    Assigning `agent_gen = None` alone leaves the old generator suspended at
    its `yield` until the garbage collector happens to finalize it. Calling
    close() first raises GeneratorExit at that yield point immediately, so
    the frame loop is torn down deterministically at the moment the user
    asked for it rather than at some arbitrary later time.
    """
    global agent_gen
    # env_lock guarantees the frame loop is NOT inside next(agent_gen) right
    # now. Calling close() on a generator that another thread is actively
    # executing raises "generator already executing"; the lock makes that
    # impossible rather than relying on luck.
    with env_lock:
        if agent_gen is not None:
            try:
                agent_gen.close()
            except Exception:
                pass
            agent_gen = None

@socketio.on('disconnect')
def handle_disconnect():
    # Covers a page refresh or a closed tab, not just an explicit Reset
    # click - the game window should not be left open with nobody watching.
    global test_running, task_epoch
    with state_lock:
        test_running = False
        task_epoch += 1
        stop_all_music()
        close_agent_window()

@socketio.on('connect')
def handle_connect():
    global test_running, task_epoch
    with state_lock:
        test_running = False
        task_epoch += 1
        discard_agent_gen()

@socketio.on('start_testing')
def handle_start_testing(data=None):
    # `data` is unused on purpose and must stay: Flask-SocketIO passes the
    # emitted payload positionally when the client sends one. The dashboard
    # currently emits 'start_testing' with no payload, but a zero-argument
    # handler would raise TypeError the moment anything did send one.
    global test_running, task_epoch
    print(">>> START_TESTING received!")
    # Opens (or refocuses) the game window BEFORE backgrounding the frame
    # loop, so the pop-up/focus happens immediately on click rather than
    # waiting for the first streamed frame. Runs synchronously in this
    # handler - on the very first-ever click this includes loading the
    # model, so a short pause here is expected and matches how it always
    # briefly paused before streaming its first frame.
    global agent_thread
    with state_lock:
        # Retire any previous frame loop FIRST. Bumping the epoch is the
        # signal for it to break out; test_running=False makes that
        # unconditional even if the epoch check races.
        previous = agent_thread
        test_running = False
        task_epoch += 1

    if previous is not None and previous.is_alive():
        # Joined OUTSIDE state_lock: the old loop may need to take env_lock
        # to finish its current step, and holding state_lock here would be
        # an unnecessary second dependency in that wait.
        previous.join(timeout=THREAD_RETIRE_TIMEOUT)
        if previous.is_alive():
            print("[WARN] previous frame loop did not exit; not starting "
                  "another (this would otherwise leak a thread).")
            return

    with state_lock:
        open_agent_window()
        test_running = True
        task_epoch += 1
        epoch = task_epoch
        agent_thread = socketio.start_background_task(
            background_agent_task, epoch)

@socketio.on('stop_testing')
def handle_stop_testing():
    # Deliberately does NOT touch the window. Pausing is just "stop calling
    # env.step()" - background_agent_task's loop exits on test_running
    # becoming False, so both the pop-up window (last painted frame stays
    # on screen) and the live stream (no more video_frame emits) freeze on
    # the same last frame together, with nothing extra to do here.
    global test_running, task_epoch
    with state_lock:
        test_running = False
        task_epoch += 1
        stop_all_music()

@socketio.on('reset_game')
def handle_reset_game():
    global test_running, task_epoch
    with state_lock:
        test_running = False
        task_epoch += 1
        discard_agent_gen()  # Force a fresh generator on the next start
        stop_all_music()
        close_agent_window()

if __name__ == '__main__':
    # ─── PRE-LOAD THE MODEL BEFORE ACCEPTING CONNECTIONS ───
    # PPO.load() onto the GPU is a real blocking call (1-3+ seconds of pure
    # CPU/GPU work, not I/O). Under the old eventlet mode that froze the
    # entire server, because every greenlet shared one OS thread: a client
    # already connected and waiting through the first-ever "Start Testing"
    # click would miss its heartbeat, decide the server was dead, and
    # reconnect - which fired the disconnect handler and closed the game
    # window that had *just* been opened, moments before run_mario_agent()
    # tried to use it. That produced "pygame.error: video system not
    # initialized" inside env.reset() in a live client test.
    #
    # Threading mode removes that specific failure (a blocking call in one
    # thread no longer stops the server answering others), but this pre-load
    # stays for a plainer reason: it moves a multi-second stall off the
    # user's first click and onto startup, where a printed message explains
    # it. open_agent_window() creates the env/model AND, as a side effect of
    # constructing CustomMarioEnv, a real OS window - close_agent_window()
    # hides it again until an actual "Start Testing" click.
    #
    # It also pins ALL pygame/SDL work to this main thread for the life of
    # the process: the window is created here, and every later open/close/
    # step call reuses it. That matters because SDL is not safe to drive
    # from arbitrary threads.
    print("Pre-loading the AI model (this can take a few seconds)...")
    open_agent_window()
    close_agent_window()
    print("Model loaded. Ready for connections.")

    # debug=False on purpose: the Werkzeug reloader spawns a SECOND python
    # process that also imports agent_logic and loads the model onto the GPU,
    # which wastes VRAM and leaves an orphan process behind on shutdown.
    # allow_unsafe_werkzeug=True is required in threading mode: without
    # eventlet's WSGI server, Flask-SocketIO falls back to Werkzeug's, which
    # refuses to start outside debug mode unless explicitly allowed. This is
    # a single-user dashboard on localhost, which is exactly the case that
    # flag exists for.
    # ─── BIND TO LOCALHOST ONLY BY DEFAULT ───
    # This used to be host='0.0.0.0', which listens on EVERY network
    # interface - anyone on the same Wi-Fi could open the dashboard, drive
    # the agent, and watch the stream, with no password in front of it. The
    # README only ever told you to visit localhost, so the exposure was
    # accidental rather than intended.
    #
    # 127.0.0.1 means "this machine only". To deliberately watch from
    # another device (a phone on the same network, say), opt in explicitly:
    #     set GLITCH_HUNTER_HOST=0.0.0.0     (Windows)
    #     GLITCH_HUNTER_HOST=0.0.0.0 python app.py   (Mac/Linux)
    # Only do that on a network you trust - there is still no auth.
    host = os.environ.get('GLITCH_HUNTER_HOST', '127.0.0.1')
    port = int(os.environ.get('GLITCH_HUNTER_PORT', '5000'))
    if host != '127.0.0.1':
        print(f"[WARNING] Listening on {host} - reachable by other devices "
              f"on this network, with no authentication.")
    print(f"Dashboard ready at http://localhost:{port}")

    socketio.run(app, host=host, port=port, debug=False,
                 allow_unsafe_werkzeug=True)
