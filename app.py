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

import eventlet
eventlet.monkey_patch()
import traceback
import time
from flask import Flask, render_template
from flask_socketio import SocketIO
# agent_logic pulls in custom_mario_env, which sets SDL_AUDIODRIVER before
# pygame loads - so it has to be imported before pygame is used here.
from agent_logic import run_mario_agent, open_agent_window, close_agent_window
import pygame as pg

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

# Cache-busting: appended as ?v=... on static asset URLs (see index.html)
# so a browser that already cached an old style.css/main.js is forced to
# fetch the current one after every restart, instead of silently showing a
# stale page until the user thinks to hard-refresh.
ASSET_VERSION = str(int(time.time()))

test_running = False
agent_gen = None

task_epoch = 0

@app.route('/')
def index():
    return render_template('index.html', asset_version=ASSET_VERSION)

def background_agent_task(epoch):
    global test_running, agent_gen
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

    try:
        last_t = time.time()
        for item in agent_gen:
            if not test_running or task_epoch != epoch:
                break
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
    test_running = False
    task_epoch += 1
    stop_all_music()
    close_agent_window()

@socketio.on('connect')
def handle_connect():
    global test_running, task_epoch
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
    open_agent_window()
    test_running = True
    task_epoch += 1
    socketio.start_background_task(background_agent_task, task_epoch)

@socketio.on('stop_testing')
def handle_stop_testing():
    # Deliberately does NOT touch the window. Pausing is just "stop calling
    # env.step()" - background_agent_task's loop exits on test_running
    # becoming False, so both the pop-up window (last painted frame stays
    # on screen) and the live stream (no more video_frame emits) freeze on
    # the same last frame together, with nothing extra to do here.
    global test_running, task_epoch
    test_running = False
    task_epoch += 1
    stop_all_music()

@socketio.on('reset_game')
def handle_reset_game():
    global test_running, task_epoch
    test_running = False
    task_epoch += 1
    discard_agent_gen()  # Force a fresh generator on the next start
    stop_all_music()
    close_agent_window()

if __name__ == '__main__':
    # ─── PRE-LOAD THE MODEL BEFORE ACCEPTING CONNECTIONS ───
    # PPO.load() onto the GPU is a real blocking call (1-3+ seconds of pure
    # CPU/GPU work, not I/O) - eventlet's cooperative scheduler can't service
    # ANY other socket activity while it runs, no matter which greenlet it's
    # called from. If a client is already connected and waiting when this
    # happens (e.g. the first-ever "Start Testing" click), the server misses
    # that client's ping/heartbeat for long enough that it looks dead - the
    # client disconnects and reconnects, which fires our disconnect handler
    # and closes the game window that was *just* opened, moments before
    # run_mario_agent() tries to use it. Confirmed exactly this way: a live
    # client test produced "pygame.error: video system not initialized"
    # inside env.reset(), immediately after open_agent_window() had already
    # run successfully.
    #
    # Fix: do the slow part here, once, before socketio.run() starts
    # accepting connections at all - so no live client is ever waiting
    # through it. open_agent_window() creates the env/model AND, as a side
    # effect of constructing CustomMarioEnv for the first time, a real OS
    # window - close_agent_window() immediately hides that until an actual
    # "Start Testing" click, matching the intended behavior (window only
    # appears on Start, not just because the server is running). Every
    # subsequent open (including the first real "Start Testing") is then
    # just the fast reopen/focus path, never the slow model load.
    print("Pre-loading the AI model (this can take a few seconds)...")
    open_agent_window()
    close_agent_window()
    print("Model loaded. Ready for connections.")

    # debug=False on purpose: the Werkzeug reloader spawns a SECOND python
    # process that also imports agent_logic and loads the model onto the GPU,
    # which wastes VRAM and leaves an orphan process behind on shutdown.
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
