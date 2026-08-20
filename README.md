# Glitch Hunter Project

A web-based visualizer for a Super Mario Bros reinforcement learning agent. Uses Flask and WebSockets to stream the agent's gameplay frames and live logs to a browser interface in real-time.

The agent plays a Python/Pygame clone of Super Mario Bros 1-1, trained with PPO (Proximal Policy Optimization) via Stable Baselines3.

## Project Structure
- `app.py`: Flask backend with WebSocket event handlers for real-time streaming.
- `agent_logic.py`: RL agent wrapper (GlitchHunterWrapper) with curiosity rewards, stuck detection, and powerup bonuses.
- `custom_mario_env.py`: Gymnasium-compatible wrapper around the Pygame Mario clone.
- `train_agent.py`: Training script using 8 parallel environments with PPO (CnnPolicy).
- `mario_clone/`: The full Python/Pygame Super Mario Bros 1-1 clone (graphics, sounds, game logic).
- `static/` & `templates/`: Frontend dashboard (HTML/CSS/JS).

## How to Run

1. **Activate Virtual Environment**:
   ```bash
   venv_gpu\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Agent** (optional — takes several hours):
   ```bash
   python train_agent.py
   ```

4. **Run the Dashboard**:
   ```bash
   python app.py
   ```

5. **View in Browser**:
   Open `http://localhost:5000` and click "Start Testing" to watch the agent play.
