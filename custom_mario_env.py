import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

import pygame as pg

class CustomMarioEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        
        # Actions: 0: NOOP, 1: Right, 2: Right+Jump, 3: Right+Sprint, 4: Right+Sprint+Jump, 5: Jump, 6: Left
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=255, shape=(240, 256, 3), dtype=np.uint8)
        
        # Load the Pygame clone safely
        # Load the Pygame clone safely using absolute paths so SubprocVecEnv workers don't crash
        self.orig_cwd = os.getcwd()
        project_root = os.path.dirname(os.path.abspath(__file__))
        mario_clone_dir = os.path.join(project_root, 'mario_clone')
        os.chdir(mario_clone_dir)
        sys.path.insert(0, mario_clone_dir)
        
        try:
            from data import setup, tools, constants as c
            from data.states import level1
            
            self.tools_module = tools
            self.setup_module = setup
            self.c_module = c
            
            self.game = tools.Control(setup.ORIGINAL_CAPTION)
            state_dict = {
                          c.LEVEL1: level1.Level1()
            }
            self.game.setup_states(state_dict, c.LEVEL1)  # Skip straight to level 1!
            self.fake_time = 0.0
        finally:
            os.chdir(self.orig_cwd)
            sys.path.pop(0)
            self.action_space = spaces.Discrete(7)
            
    def step(self, action):
        pg.event.pump()
        keys = {
            pg.K_RIGHT: False,
            pg.K_LEFT: False,
            pg.K_a: False, # Jump
            pg.K_s: False, # Sprint
        }
        
        if action in [1, 2, 3, 4]:
            keys[pg.K_RIGHT] = True
        if action in [2, 4, 5]:
            keys[pg.K_a] = True
        if action in [3, 4]:
            keys[pg.K_s] = True
        if action == 6:
            keys[pg.K_LEFT] = True
            
        # Overwrite pygame's get_pressed logic
        class FakeKeys:
            def __getitem__(self, key):
                return keys.get(key, False)
        
        self.game.keys = FakeKeys()
        
        # Advance game time by exactly 1 frame (60 FPS = 16.666 ms)
        self.fake_time += (1000.0 / 60.0)
        self.game.current_time = self.fake_time
        
        # Update state with fake keys
        orig_cwd = os.getcwd()
        project_root = os.path.dirname(os.path.abspath(__file__))
        mario_clone_dir = os.path.join(project_root, 'mario_clone')
        os.chdir(mario_clone_dir)
        try:
            self.game.state.update(self.game.screen, self.game.keys, self.game.current_time)
        finally:
            os.chdir(orig_cwd)
        
        full_obs = self.render()
        obs = cv2.resize(full_obs, (256, 240), interpolation=cv2.INTER_NEAREST)
        
        reward = 0.0
        done = False
        info = {}
        
        try:
            mario = self.game.state.mario
            info['x_pos'] = mario.rect.x
            info['y_pos'] = mario.rect.y
            
            info['score'] = self.game.state.game_info.get('score', 0) if hasattr(self.game.state, 'game_info') else 0
            info['coins'] = self.game.state.game_info.get('coin total', 0) if hasattr(self.game.state, 'game_info') else 0
            
            if getattr(mario, 'fire', False):
                info['status'] = 'fireball'
            elif getattr(mario, 'big', False):
                info['status'] = 'tall'
            else:
                info['status'] = 'small'
            
            if mario.dead:
                reward = -15
            
            # RL agents don't need to watch the 3 second death animation, so we return done immediately if Mario is dead
            done = self.game.state.done or getattr(self.game.state.mario, 'dead', False) if hasattr(self.game.state, 'mario') else self.game.state.done
            
        except AttributeError:
            info['x_pos'] = 0
            info['y_pos'] = 0
            info['score'] = 0
            info['coins'] = 0
            info['status'] = 'small'
            done = self.game.state.done
            
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.fake_time = 0.0
        orig_cwd = os.getcwd()
        project_root = os.path.dirname(os.path.abspath(__file__))
        mario_clone_dir = os.path.join(project_root, 'mario_clone')
        os.chdir(mario_clone_dir)
        try:
            from data.states import level1
            
            # Completely destroy and recreate the game instance to guarantee no frozen state carries over
            self.game = self.tools_module.Control(self.setup_module.ORIGINAL_CAPTION)
            state_dict = {
                self.c_module.LEVEL1: level1.Level1()
            }
            self.game.setup_states(state_dict, self.c_module.LEVEL1)
            
            persist_data = {
                self.c_module.COIN_TOTAL: 0, 
                self.c_module.SCORE: 0, 
                self.c_module.LIVES: 3, 
                self.c_module.CURRENT_TIME: 0.0, 
                self.c_module.LEVEL_STATE: None, 
                self.c_module.CAMERA_START_X: 0, 
                self.c_module.MARIO_DEAD: False,
                self.c_module.TOP_SCORE: 0
            }
            self.game.state.startup(0.0, persist_data)
        finally:
            os.chdir(orig_cwd)
        full_obs = self.render()
        obs = cv2.resize(full_obs, (256, 240), interpolation=cv2.INTER_NEAREST)
        return obs, {}

    def render(self):
        # Convert Pygame surface to numpy array (H, W, C) for Gymnasium
        surface = pg.display.get_surface()
        if surface is None:
            return np.zeros((600, 800, 3), dtype=np.uint8)
        view = pg.surfarray.array3d(surface)
        view = view.transpose([1, 0, 2])
        return view
