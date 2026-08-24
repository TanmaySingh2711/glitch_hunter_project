import eventlet
eventlet.monkey_patch()
import traceback
import time
from flask import Flask, render_template
from flask_socketio import SocketIO
from agent_logic import run_mario_agent, open_agent_window, close_agent_window

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
            sleep_time = target_frame_time - elapsed
            if sleep_time > 0:
                socketio.sleep(sleep_time)
            else:
                socketio.sleep(0)
            last_t = time.time()
    except Exception as e:
        print(f"Error in background task: {e}")
        traceback.print_exc()
    finally:
        if task_epoch == epoch:
            test_running = False

def stop_all_music():
    try:
        import pygame as pg
        if pg.mixer.get_init():
            pg.mixer.music.stop()
            pg.mixer.stop()
    except Exception:
        pass

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
    global agent_gen, test_running, task_epoch
    test_running = False
    task_epoch += 1
    agent_gen = None

@socketio.on('start_testing')
def handle_start_testing(data=None):
    global test_running, task_epoch
    print(f">>> START_TESTING received!")
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
    global agent_gen, test_running, task_epoch
    test_running = False
    task_epoch += 1
    agent_gen = None  # Force a fresh generator on the next start
    stop_all_music()
    close_agent_window()

if __name__ == '__main__':
    # debug=False on purpose: the Werkzeug reloader spawns a SECOND python
    # process that also imports agent_logic and loads the model onto the GPU,
    # which wastes VRAM and leaves an orphan process behind on shutdown.
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
