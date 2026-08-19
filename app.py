import eventlet
eventlet.monkey_patch()
import traceback

from flask import Flask, render_template
from flask_socketio import SocketIO
from agent_logic import run_mario_agent

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

test_running = False
agent_gen = None

task_epoch = 0

current_env_type = None

@app.route('/')
def index():
    return render_template('index.html')

def background_agent_task(epoch, env_type="mario"):
    global test_running, agent_gen, current_env_type
    
    if agent_gen is None or current_env_type != env_type:
        agent_gen = run_mario_agent(env_type)
        current_env_type = env_type
        
    import time
    target_frame_time = 1.0 / 60.0
    
    try:
        last_t = time.time()
        for item in agent_gen:
            if not test_running or current_env_type != env_type or task_epoch != epoch:
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
    global test_running, task_epoch
    test_running = False
    task_epoch += 1
    stop_all_music()

@socketio.on('connect')
def handle_connect():
    global agent_gen, test_running, task_epoch
    test_running = False
    task_epoch += 1
    agent_gen = None

@socketio.on('start_testing')
def handle_start_testing(data=None):
    global test_running, task_epoch
    env_type = data.get('env', 'mario') if data else 'mario'
    print(f">>> START_TESTING received! env_type={env_type}")
    test_running = True
    task_epoch += 1
    socketio.start_background_task(background_agent_task, task_epoch, env_type)

@socketio.on('stop_testing')
def handle_stop_testing():
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

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
