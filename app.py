import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template
from flask_socketio import SocketIO
from agent_logic import run_mario_agent

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

test_running = False

@app.route('/')
def index():
    return render_template('index.html')

def background_agent_task():
    global test_running
    try:
        for item in run_mario_agent():
            if not test_running:
                break
            socketio.emit('video_frame', {'frame': item['frame']})
            socketio.emit('agent_log', {'log': f"Step {item['step']}: Action {item['action']} | Reward: {item['reward']:.2f}"})
            socketio.sleep(0) # Let eventlet flush
    except Exception as e:
        print(f"Error in background task: {e}")
    finally:
        test_running = False

@socketio.on('start_testing')
def handle_start_testing():
    global test_running
    if not test_running:
        test_running = True
        socketio.start_background_task(background_agent_task)

@socketio.on('stop_testing')
def handle_stop_testing():
    global test_running
    test_running = False

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
