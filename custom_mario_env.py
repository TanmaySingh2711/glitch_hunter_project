import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

# Force Pygame to run headlessly without popping up a window on the server
os.environ["SDL_VIDEODRIVER"] = "dummy"
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
        orig_cwd = os.getcwd()
        os.chdir(os.path.join(orig_cwd, 'mario_clone'))
        sys.path.insert(0, os.getcwd())
        
        try:
            from data import setup, tools, constants as c
            from data.states import main_menu, load_screen, level1
            
            self.tools_module = tools
            self.setup_module = setup
            self.c_module = c
            
            self.game = tools.Control(setup.ORIGINAL_CAPTION)
            state_dict = {c.MAIN_MENU: main_menu.Menu(),
                          c.LOAD_SCREEN: load_screen.LoadScreen(),
                          c.TIME_OUT: load_screen.TimeOut(),
                          c.GAME_OVER: load_screen.GameOver(),
                          c.LEVEL1: level1.Level1()}
            self.game.setup_states(state_dict, c.LEVEL1)  # Skip straight to level 1!
            self.fake_time = 0.0
        finally:
            os.chdir(orig_cwd)
            sys.path.pop(0)
            
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
        os.chdir(os.path.join(orig_cwd, 'mario_clone'))
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
            
            if mario.dead:
                reward = -15
            
            # RL agents don't need to watch the 3 second death animation, so we return done immediately if Mario is dead
            done = self.game.state.done or getattr(self.game.state.mario, 'dead', False) if hasattr(self.game.state, 'mario') else self.game.state.done
            
        except AttributeError:
            info['x_pos'] = 0
            info['y_pos'] = 0
            done = self.game.state.done
            
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.fake_time = 0.0
        orig_cwd = os.getcwd()
        os.chdir(os.path.join(orig_cwd, 'mario_clone'))
        try:
            from data.states import main_menu, load_screen, level1
            
            # Completely destroy and recreate the game instance to guarantee no frozen state carries over
            self.game = self.tools_module.Control(self.setup_module.ORIGINAL_CAPTION)
            state_dict = {
                self.c_module.MAIN_MENU: main_menu.Menu(),
                self.c_module.LOAD_SCREEN: load_screen.LoadScreen(),
                self.c_module.TIME_OUT: load_screen.TimeOut(),
                self.c_module.GAME_OVER: load_screen.GameOver(),
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
