# Glitch Hunter Project

This project is a web-based visualizer for a Super Mario Bros reinforcement learning agent. It uses Flask and WebSockets to stream the agent's gameplay frames and live logs to a browser interface in real-time.

## Current Progress (Objective 1)
- Set up a Flask backend with `Flask-SocketIO` for real-time bidirectional communication.
- Implemented a Super Mario Bros reinforcement learning agent using `gym_super_mario_bros` and `stable_baselines3` (PPO).
- Developed a web interface that connects to the backend and displays live video frames and agent logs.
- The agent logic runs in a background eventlet task to allow asynchronous frame streaming.

*Note: Objectives 2 and 3 are pending and will be documented here once completed.*

## Project Structure
- `app.py`: The main Flask application and WebSocket event handlers.
- `agent_logic.py`: Contains the logic for initializing the Mario environment, running the PPO agent, and yielding frames/logs.
- `requirements.txt`: Python dependencies.
- `static/` & `templates/`: Frontend assets and HTML templates for the web interface.

## How to Run

1. **Activate Virtual Environment** (if you are using one):
   ```bash
   venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python app.py
   ```

4. **View in Browser**:
   Open `http://127.0.0.1:5000` or `http://localhost:5000` in your web browser. Click "Start Testing" to begin streaming the agent.
