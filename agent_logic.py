import time
import cv2
import gymnasium as gym

from stable_baselines3 import PPO
import numpy as np
from collections import deque
import base64
from custom_mario_env import CustomMarioEnv

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
        
        x_pos = info.get('x_pos', self.last_x_pos or 0)
        y_pos = info.get('y_pos', 0)



        # Curiosity bonus
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

class SpeedScalerWrapper(gym.Wrapper):
    """
    Decouples the fixed 60Hz server loop from the physics update rate using delta time.
    """
    def __init__(self, env, target_speed_multiplier=1.0, base_fps=60.0):
        super().__init__(env)
        self.target_fps = base_fps * target_speed_multiplier
        self.frame_accumulator = 0.0
        self.last_time = time.time()
        self.last_obs = None
        self.last_reward = 0.0
        self.last_done = False
        self.last_info = {}

    def reset(self, **kwargs):
        self.frame_accumulator = 0.0
        self.last_time = time.time()
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            self.last_obs = obs[0]
            self.last_info = obs[1]
        else:
            self.last_obs = obs
            self.last_info = {}
        self.last_reward = 0.0
        self.last_done = False
        return obs

    def step(self, action):
        now = time.time()
        elapsed = now - self.last_time
        self.last_time = now
        
        # Prevent spiral of death if inference hangs
        if elapsed > 0.1:
            elapsed = 0.1
            
        self.frame_accumulator += elapsed * self.target_fps
        updates_to_run = int(self.frame_accumulator)
        self.frame_accumulator -= updates_to_run

        reward_sum = 0.0
        
        if updates_to_run == 0:
            return self.last_obs, 0.0, self.last_done, False, self.last_info

        for _ in range(updates_to_run):
            if self.last_done:
                break
            step_result = self.env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            
            reward_sum += reward
            self.last_obs = obs
            self.last_info = info
            self.last_done = done

        return self.last_obs, float(reward_sum), self.last_done, False, self.last_info

from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation, ResizeObservation

_global_env = None
_global_model = None
_current_env_type = None

def run_mario_agent(env_type="mario"):
    global _global_env, _global_model, _current_env_type
    
    if _global_env is None or _current_env_type != env_type:
        _current_env_type = env_type
        
        if env_type == "mario_python":
            _global_env = CustomMarioEnv()
            _global_env = GlitchHunterWrapper(_global_env)
            _global_env = GrayscaleObservation(_global_env, keep_dim=False)
            _global_env = ResizeObservation(_global_env, (84, 84))
            _global_env = FrameStackObservation(_global_env, 4)
        else:
            import gym_super_mario_bros
            from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
            from nes_py.wrappers import JoypadSpace
            
            # Setup environment on first run (takes ~4.5s)
            try:
                _global_env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode="rgb_array", apply_api_compatibility=True)
            except Exception:
                try:
                    _global_env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode="rgb_array")
                except Exception:
                    _global_env = gym_super_mario_bros.make('SuperMarioBros-v0')
                
            _global_env = JoypadSpace(_global_env, SIMPLE_MOVEMENT)
            _global_env = GlitchHunterWrapper(_global_env)
            _global_env = SpeedScalerWrapper(_global_env, 0.50)
            _global_env = GrayscaleObservation(_global_env, keep_dim=False)
            _global_env = ResizeObservation(_global_env, (84, 84))
            _global_env = FrameStackObservation(_global_env, 4)
        
        # Initialize model
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
            
        ACTION_NAMES = {
            0: "Stand Still",
            1: "Walk Right",
            2: "Walk Right + Jump",
            3: "Run Right",
            4: "Run Right + Jump",
            5: "Jump",
            6: "Walk Left"
        }

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
