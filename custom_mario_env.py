import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

import pygame as pg

# Disable audio to prevent sound spam during training
os.environ["SDL_AUDIODRIVER"] = "dummy"

class FakeKeys:
    def __init__(self):
        self.keys = {}
    def __getitem__(self, key):
        return self.keys.get(key, False)
    def update(self, new_keys):
        self.keys = new_keys

class CustomMarioEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        
        # ═══════════════════════════════════════════════════════════════════
        # ACTION SPACE (10 discrete actions)
        # 0: NOOP
        # 1: Right (walk)
        # 2: Right + Jump
        # 3: Right + Sprint (run)
        # 4: Right + Sprint + Jump   <- the "long jump" needed to clear pipes/pits
        # 5: Jump (vertical, in place)
        # 6: Left (walk)
        # 7: Crouch / Down
        # 8: Left + Sprint (run backward)  <- lets the agent retreat quickly to
        #    build a runway before sprinting back into a jump, instead of being
        #    limited to a slow walk-left that makes backing up too costly.
        # 9: Left + Jump (jump while retreating - dodging/escaping enemies
        #    approaching from the right, or backing off a ledge safely)
        #
        # Expanded from the original 8 actions specifically to give the agent
        # a way to execute "back up, then sprint+jump" momentum plays. This
        # is NOT backward-compatible with checkpoints trained on the old
        # 8-action space (the policy's action head shape changes) - see
        # migrate_checkpoint.py / IMPLEMENTATION.md for how to carry over
        # the learned visual features anyway.
        # ═══════════════════════════════════════════════════════════════════
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=255, shape=(240, 256, 3), dtype=np.uint8)
        
        self.fake_keys = FakeKeys()
        
        # Load the Pygame clone safely using absolute paths so SubprocVecEnv workers don't crash
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.mario_clone_dir = os.path.join(self.project_root, 'mario_clone')
        
        orig_cwd = os.getcwd()
        os.chdir(self.mario_clone_dir)
        sys.path.insert(0, self.mario_clone_dir)
        
        # Tile windows across the screen so they don't stack perfectly on top of each other
        # The window size is ~800x600. We'll stagger them diagonally.
        import random
        # Give them a random offset so the 8 windows spread out on the desktop
        x_pos = random.randint(50, 800)
        y_pos = random.randint(50, 400)
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x_pos},{y_pos}"
        if 'SDL_VIDEO_CENTERED' in os.environ:
            del os.environ['SDL_VIDEO_CENTERED']
            
        try:
            from data import setup, tools, constants as c
            from data.states import level1
            
            # Store mario clone directory
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
            os.chdir(orig_cwd)
            if self.mario_clone_dir in sys.path:
                sys.path.remove(self.mario_clone_dir)
            
    def step(self, action):
        pg.event.pump()
        keys = {
            pg.K_RIGHT: False,
            pg.K_LEFT: False,
            pg.K_a: False, # Jump
            pg.K_s: False, # Sprint
            pg.K_DOWN: False # Crouch
        }
        
        if action in [1, 2, 3, 4]:
            keys[pg.K_RIGHT] = True
        if action in [6, 8, 9]:
            keys[pg.K_LEFT] = True
        if action in [2, 4, 5, 9]:
            keys[pg.K_a] = True          # Jump
        if action in [3, 4, 8]:
            keys[pg.K_s] = True          # Sprint
        if action == 7:
            keys[pg.K_DOWN] = True
            
        self.fake_keys.update(keys)
        self.game.keys = self.fake_keys
        
        # ═══════════════════════════════════════════════════════════════════
        # PERMANENT FIX for BUG #1: Bypass the Pygame clone's anti-spam
        # jump lock. The clone requires the jump key to be released before
        # another jump can register. Since the RL agent holds keys down
        # continuously, we force allow_jump=True before every update so
        # the agent can jump every time it asks to.
        # ═══════════════════════════════════════════════════════════════════
        try:
            self.game.state.mario.allow_jump = True
        except AttributeError:
            pass
        
        # Advance game time by exactly 1 frame (60 FPS = 16.666 ms)
        self.fake_time += (1000.0 / 60.0)
        self.game.current_time = self.fake_time
        
        # Update state with fake keys
        orig_cwd = os.getcwd()
        os.chdir(self.mario_clone_dir)
        try:
            self.game.state.update(self.game.screen, self.game.keys, self.game.current_time)
            pg.display.update()  # Force OS to paint the window so it doesn't freeze black
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

            # Velocity + ground-contact info, needed by the reward wrapper to
            # detect and reward "momentum building" (sprinting before a jump)
            # and to tell a deliberate running jump apart from idle bouncing.
            c = self.c_module
            info['x_vel'] = float(getattr(mario, 'x_vel', 0.0))
            info['y_vel'] = float(getattr(mario, 'y_vel', 0.0))
            info['mario_state'] = getattr(mario, 'state', c.STAND)
            info['on_ground'] = info['mario_state'] in (c.STAND, c.WALK)
            info['facing_right'] = bool(getattr(mario, 'facing_right', True))

            info['score'] = self.game.state.game_info.get('score', 0) if hasattr(self.game.state, 'game_info') else 0
            info['coins'] = self.game.state.game_info.get('coin total', 0) if hasattr(self.game.state, 'game_info') else 0
            info['lives'] = self.game.state.persist.get(c.LIVES, 3) if hasattr(self.game.state, 'persist') else 3

            if getattr(mario, 'fire', False):
                info['status'] = 'fireball'
            elif getattr(mario, 'big', False):
                info['status'] = 'tall'
            else:
                info['status'] = 'small'

            # Level-complete flag: True the moment Mario touches the flagpole
            # area and starts the end-of-level sequence. Exposed so the
            # reward wrapper can grant a large one-time completion bonus
            # instead of waiting for the ~2s castle/fireworks animation to
            # fully finish before the episode is marked done.
            info['flag_get'] = bool(getattr(mario, 'in_castle', False))

            # ─── ACTIVE POWERUP INFO ───
            # Exposes whatever mushroom/fire flower/star/1-up is currently
            # revealed and moving around the level (level1.py's
            # self.powerup_group), so the reward wrapper can (a) tell the
            # instant a "?" block/brick is bumped and releases something
            # (powerup_active_count going up), and (b) nudge the agent
            # toward the nearest one while it's out, via nearest_powerup_dx
            # (signed pixel offset from Mario; positive = powerup is to the
            # right). Both become 0/None when nothing is currently active.
            powerup_group = getattr(self.game.state, 'powerup_group', None)
            if powerup_group is not None:
                info['powerup_active_count'] = len(powerup_group)
                nearest_dx = None
                nearest_type = None
                for p in powerup_group:
                    dx = p.rect.centerx - mario.rect.centerx
                    if nearest_dx is None or abs(dx) < abs(nearest_dx):
                        nearest_dx = dx
                        nearest_type = getattr(p, 'name', 'powerup')
                info['nearest_powerup_dx'] = nearest_dx
                info['nearest_powerup_type'] = nearest_type
            else:
                info['powerup_active_count'] = 0
                info['nearest_powerup_dx'] = None
                info['nearest_powerup_type'] = None

            # Small death penalty so the agent learns dying is undesirable,
            # but small enough that it won't cause "standing still" paralysis
            info['death_cause'] = None
            if mario.dead:
                reward = -5.0
                info['death_cause'] = getattr(mario, 'death_cause', None)
            
            # RL agents don't need to watch the 3 second death animation, so we return done immediately if Mario is dead
            done = self.game.state.done or getattr(self.game.state.mario, 'dead', False) if hasattr(self.game.state, 'mario') else self.game.state.done
            
        except AttributeError:
            info['x_pos'] = 0
            info['y_pos'] = 0
            info['x_vel'] = 0.0
            info['y_vel'] = 0.0
            info['mario_state'] = self.c_module.STAND
            info['on_ground'] = True
            info['facing_right'] = True
            info['score'] = 0
            info['coins'] = 0
            info['lives'] = 3
            info['status'] = 'small'
            info['flag_get'] = False
            info['powerup_active_count'] = 0
            info['nearest_powerup_dx'] = None
            info['nearest_powerup_type'] = None
            info['death_cause'] = None
            done = self.game.state.done
            
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.fake_time = 0.0
        orig_cwd = os.getcwd()
        os.chdir(self.mario_clone_dir)
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
        
        pg.display.update()
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
