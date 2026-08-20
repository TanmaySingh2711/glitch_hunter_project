import os
import cv2
import gymnasium as gym

from stable_baselines3 import PPO
import numpy as np
from collections import deque
import base64
from custom_mario_env import CustomMarioEnv

ACTION_NAMES = {
    0: "Stand Still",
    1: "Walk Right",
    2: "Walk Right + Jump",
    3: "Run Right",
    4: "Run Right + Jump",
    5: "Jump",
    6: "Walk Left",
    7: "Crouch"
}

class GlitchHunterWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.visited_x = set()
        self.stuck_counter = 0
        self.last_x_pos = None
        self.x_history = deque(maxlen=30)
        self.total_steps = 0
        
        self.last_score = 0
        self.last_coins = 0
        self.last_status = 'small'

    def reset(self, **kwargs):
        # Clear state variables back to init state
        self.visited_x.clear()
        self.stuck_counter = 0
        self.last_x_pos = None
        self.x_history.clear()
        
        self.last_score = 0
        self.last_coins = 0
        self.last_status = 'small'
        return self.env.reset(**kwargs)

    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        
        self.total_steps += 1
        
        x_pos = info.get('x_pos', self.last_x_pos or 0)        # Curiosity bonus
        if x_pos not in self.visited_x:
            self.visited_x.add(x_pos)
            reward += 1.0

        # Stuck detection (robust version)
        self.x_history.append(x_pos)
        if len(self.x_history) == self.x_history.maxlen:
            spread = max(self.x_history) - min(self.x_history)
            if spread <= 40:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0

        # Stuck penalty
        if self.stuck_counter > 30:
            reward -= 1.0

        # --- CURIOSITY AND POWERUP REWARDS ---
        score = info.get('score', 0)
        coins = info.get('coins', 0)
        status = info.get('status', 'small')

        if score > self.last_score:
            reward += (score - self.last_score) * 0.05
        if coins > self.last_coins:
            reward += (coins - self.last_coins) * 2.0
            
        # Powerup bonus
        if status != self.last_status:
            if status in ['tall', 'fireball'] and self.last_status == 'small':
                reward += 20.0
            elif status == 'fireball' and self.last_status == 'tall':
                reward += 10.0
            elif status == 'small' and self.last_status in ['tall', 'fireball']:
                reward -= 20.0  # Penalty for taking damage
                
        self.last_score = score
        self.last_coins = coins
        self.last_status = status
        # -------------------------------------

        self.last_x_pos = x_pos
        
        return obs, float(reward), done, False, info

from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation, ResizeObservation, MaxAndSkipObservation

_global_env = None
_global_model = None

def run_mario_agent():
    global _global_env, _global_model
    
    if _global_env is None:
        
        _global_env = CustomMarioEnv()
        _global_env = GlitchHunterWrapper(_global_env)
        _global_env = MaxAndSkipObservation(_global_env, skip=4)
        _global_env = GrayscaleObservation(_global_env, keep_dim=False)
        _global_env = ResizeObservation(_global_env, (84, 84))
        _global_env = FrameStackObservation(_global_env, 4)
        
        # Initialize model
        model_path = "mario_brain_checkpoint.zip"
        if os.path.exists(model_path):
            _global_model = PPO.load(model_path, env=_global_env, device="auto")
        else:
            _global_model = PPO('CnnPolicy', _global_env, verbose=0)
        
    env = _global_env
    model = _global_model
    
    # Initial reset
    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, _ = reset_result
    else:
        obs = reset_result
    obs = obs.copy()

    step_count = 0
    while True:
        action, _states = model.predict(obs, deterministic=False)
        action_val = int(action.item()) if hasattr(action, 'item') else int(action)
        
        # Step environment
        step_result = env.step(action_val)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        
        obs = obs.copy()
        step_count += 1
        
        # Render frame
        try:
            frame = env.render(mode="rgb_array")
        except TypeError:
            frame = env.render()
            
        if frame is not None:
            # Convert RGB to BGR for cv2
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Encode BGR to JPEG
            _, buffer = cv2.imencode('.jpg', frame_bgr)
            b64_string = base64.b64encode(buffer).decode('utf-8')
        else:
            b64_string = ""
            

        log_message = None
        if info.get('glitch_alert'):
            log_message = '🚨 BUG FOUND: ' + info['glitch_alert']
        else:
            action_name = ACTION_NAMES.get(int(action_val), "Unknown")
            log_message = f"Step {step_count}: Action: {action_name} ({action_val}) | Reward: {float(reward):.2f}"
            
        yield {
            'frame': b64_string,
            'action': action_val,
            'step': step_count,
            'reward': float(reward),
            'log': log_message
        }
        
        if done:
            reset_result = env.reset()
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs, _ = reset_result
            else:
                obs = reset_result
            obs = obs.copy()
