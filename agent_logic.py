import time
import cv2
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO

def run_mario_agent():
    # Try passing render_mode="rgb_array"
    try:
        env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode="rgb_array", apply_api_compatibility=True)
    except Exception:
        try:
            env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode="rgb_array")
        except Exception:
            env = gym_super_mario_bros.make('SuperMarioBros-v0')
            
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    
    # Note: Using an untrained model, so behavior will look random.
    # This is fine for demoing the pipeline as requested, not for actual skilled play.
    model = PPO('CnnPolicy', env, verbose=0)
    
    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, _ = reset_result
    else:
        obs = reset_result
    obs = obs.copy()

    step_count = 0
    
    while True:
        action, _ = model.predict(obs)
        action_val = int(action.item()) if hasattr(action, 'item') else int(action)
        
        step_result = env.step(action_val)
        
        # Handle 4-tuple or 5-tuple depending on gym version compatibility
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        
        obs = obs.copy()

        # Render frame
        try:
            frame = env.render()
        except TypeError:
            # Fallback if old API
            frame = env.render(mode="rgb_array")
            
        if frame is not None:
            # Convert RGB to BGR for cv2
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Encode BGR to JPEG
            _, buffer = cv2.imencode('.jpg', frame_bgr)
            import base64
            b64_string = base64.b64encode(buffer).decode('utf-8')
        else:
            b64_string = ""

        yield {
            'frame': b64_string,
            'action': int(action_val),
            'step': step_count,
            'reward': float(reward)
        }
        
        step_count += 1
        time.sleep(0.02)
        
        if done:
            reset_result = env.reset()
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs, _ = reset_result
            else:
                obs = reset_result
            obs = obs.copy()
