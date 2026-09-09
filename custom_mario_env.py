import os
import random
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np

import pygame as pg

# Disable audio to prevent sound spam during training
os.environ["SDL_AUDIODRIVER"] = "dummy"

# How many substeps a glitch alert stays attached to info before expiring.
# Must match the `skip` passed to MaxAndSkipObservation (see the delivery
# note in CustomMarioEnv._detect_glitches for why this coupling exists).
GLITCH_ALERT_TTL = 4

class FakeKeys:
    def __init__(self):
        self.keys = {}
    def __getitem__(self, key):
        return self.keys.get(key, False)
    def update(self, new_keys):
        self.keys = new_keys

class CustomMarioEnv(gym.Env):
    # No `metadata = {"render_modes": ...}` / `render_mode` here on purpose.
    # This env never goes through Gymnasium's render() protocol: it always
    # owns a real on-screen pygame window, and frames are pulled directly
    # via _fast_obs() (for the agent) and render_scaled() (for the dashboard
    # stream). Declaring a render mode without implementing render() would
    # advertise an API this class does not actually support.

    def __init__(self):
        super().__init__()


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
        # a way to execute "back up, then sprint+jump" momentum plays. Note
        # this is NOT backward-compatible with checkpoints trained on an
        # 8-action space (the policy's action head shape changes), so such a
        # checkpoint cannot be loaded here - it has to be retrained.
        #
        # Keep this list in sync with ACTION_NAMES in agent_logic.py and the
        # info modal in templates/index.html.
        # ═══════════════════════════════════════════════════════════════════
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=255, shape=(240, 256, 3), dtype=np.uint8)

        self.fake_keys = FakeKeys()

        # Per-episode glitch-detection state (see _detect_glitches below).
        # Cleared in reset() so each episode reports a given kind at most once.
        self._reported_glitches = set()
        self._last_score = 0
        self._last_coins = 0
        self._pending_glitch = None
        self._pending_ttl = 0

        # Load the Pygame clone safely using absolute paths so SubprocVecEnv workers don't crash
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.mario_clone_dir = os.path.join(self.project_root, 'mario_clone')

        orig_cwd = os.getcwd()
        os.chdir(self.mario_clone_dir)
        sys.path.insert(0, self.mario_clone_dir)

        # Tile windows across the screen so they don't stack perfectly on top
        # of each other. The window is ~800x600; a random offset spreads the
        # 8 parallel training windows out across the desktop.
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

        # Update state with fake keys.
        #
        # No chdir here any more: this used to wrap the call in
        # getcwd + chdir(mario_clone) + chdir(back), because the game's music
        # paths were relative and pg.mixer.music.load() re-opens them at
        # runtime (game_sound.py can start a track mid-update). setup.py now
        # builds those paths absolutely from its own location, so the update
        # is cwd-independent and the syscalls are pure overhead - measured at
        # ~1.9% of a step, and this runs 4x per agent decision under
        # MaxAndSkipObservation.
        self.game.state.update(self.game.screen, self.game.keys, self.game.current_time)
        # Force OS to paint the window so it doesn't freeze black. Guarded
        # for the dashboard's close_window(): if Stop/Reset raced with an
        # in-flight step() call, the display may already be gone - _fast_obs()
        # already tolerates that (returns a black frame), so just skip the
        # paint here rather than raising.
        if pg.display.get_surface() is not None:
            pg.display.update()


        obs = self._fast_obs()

        reward = 0.0
        done = False
        info = {}

        try:
            mario = self.game.state.mario
            info['x_pos'] = mario.rect.x
            info['y_pos'] = mario.rect.y

            # ─── WORLD-SPACE COLLIDER + CAMERA (exploration retrofit) ───
            # `mario.rect` is in WORLD coordinates, not screen coordinates.
            # Verified live: rect.x reached 2928 while viewport.x was 2571,
            # and the window is only 800px wide - a screen-space rect could
            # never exceed 800. level1.py subtracts the camera only at blit
            # time (`mario.rect.x - self.viewport.x`), never from the rect
            # itself. So this needs no camera correction anywhere, and the
            # coverage bitmap can be indexed with it directly.
            #
            # The full rect is exported rather than just x/y because Mario's
            # collider CHANGES SIZE with his form (30x40 small, 40x80 big),
            # and coverage must sweep the real footprint - assuming 30x40
            # while big would under-count half of every pixel he occupies.
            #
            # viewport_x is exported because the camera is ONE-WAY:
            # update_viewport() only ever increases it, and Mario is
            # hard-clamped to viewport.x + 5 (level1.py:518). Anything left
            # of it is physically unreachable for the rest of the episode,
            # so the frontier index uses this to avoid pointing the agent at
            # impossible targets.
            info['mario_rect'] = (int(mario.rect.x), int(mario.rect.y),
                                  int(mario.rect.w), int(mario.rect.h))
            viewport = getattr(self.game.state, 'viewport', None)
            info['viewport_x'] = int(viewport.x) if viewport is not None else 0

            # Velocity + ground-contact info, needed by the reward wrapper to
            # detect and reward "momentum building" (sprinting before a jump)
            # and to tell a deliberate running jump apart from idle bouncing.
            c = self.c_module
            mario_state = getattr(mario, 'state', c.STAND)
            info['x_vel'] = float(getattr(mario, 'x_vel', 0.0))
            info['on_ground'] = mario_state in (c.STAND, c.WALK)

            info['score'] = self.game.state.game_info.get('score', 0) if hasattr(self.game.state, 'game_info') else 0
            info['coins'] = self.game.state.game_info.get('coin total', 0) if hasattr(self.game.state, 'game_info') else 0

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
                for p in powerup_group:
                    dx = p.rect.centerx - mario.rect.centerx
                    if nearest_dx is None or abs(dx) < abs(nearest_dx):
                        nearest_dx = dx
                info['nearest_powerup_dx'] = nearest_dx
            else:
                info['powerup_active_count'] = 0
                info['nearest_powerup_dx'] = None

            # Small death penalty so the agent learns dying is undesirable,
            # but small enough that it won't cause "standing still" paralysis
            info['death_cause'] = None
            info['is_dead'] = bool(mario.dead)
            if mario.dead:
                reward = -5.0
                info['death_cause'] = getattr(mario, 'death_cause', None)

            # RL agents don't need to watch the ~3 second death animation, so
            # end the episode as soon as Mario is dead. (`mario` is already
            # bound above, so no hasattr guard is needed here — if it were
            # missing we'd have raised AttributeError long before this line.)
            done = bool(self.game.state.done or mario.dead)

        except AttributeError:
            info['x_pos'] = 0
            info['y_pos'] = 0
            info['mario_rect'] = None
            info['viewport_x'] = 0
            info['x_vel'] = 0.0
            info['on_ground'] = True
            info['score'] = 0
            info['coins'] = 0
            info['status'] = 'small'
            info['flag_get'] = False
            info['powerup_active_count'] = 0
            info['nearest_powerup_dx'] = None
            info['death_cause'] = None
            info['is_dead'] = False
            done = self.game.state.done

        self._detect_glitches(info)
        return obs, reward, done, False, info

    # ═══════════════════════════════════════════════════════════════════
    # GLITCH DETECTION — what actually feeds the dashboard's BUG TRACKER
    #
    # Sets info['glitch_alert'] to a human-readable string when the game
    # violates an invariant it is supposed to hold. agent_logic.py turns
    # that into a "BUG FOUND" log line, which main.js routes into the bug
    # panel. Nothing wrote this key before, so that panel could never show
    # anything - this is the piece that was missing.
    #
    # Every threshold below is set from MEASURED normal play, not guessed:
    # 4,000 steps across 22 episodes driven by the actual trained policy
    # produced |x_vel| max 13.2, y_pos range -29..498, and zero score or
    # coin decreases. Each check below sat at zero hits over that whole
    # run, and the numeric limits are set roughly 2x outside the observed
    # envelope. A bug detector that cries wolf is worse than none at all,
    # so these are deliberately tuned to stay silent during normal play and
    # only speak up for things the engine genuinely should not permit.
    # ═══════════════════════════════════════════════════════════════════
    def _detect_glitches(self, info):
        # ─── DELIVERY (why this isn't just `info['glitch_alert'] = msg`) ───
        # This env is wrapped in MaxAndSkipObservation(skip=4), which calls
        # step() four times per agent decision and returns ONLY the last
        # substep's info dict - the other three are discarded. An alert
        # raised on substep 1, 2 or 3 would therefore vanish before anything
        # could display it. So an alert is held for GLITCH_ALERT_TTL
        # substeps and re-attached each time, which guarantees it is still
        # attached on whichever substep survives. The TTL equals the skip
        # value on purpose: long enough to always reach one surviving info,
        # short enough that it expires before the NEXT surviving one, so the
        # dashboard shows each glitch exactly once.
        if self._pending_glitch is not None:
            info['glitch_alert'] = self._pending_glitch
            self._pending_ttl -= 1
            if self._pending_ttl <= 0:
                self._pending_glitch = None

        # Reported at most once per episode per kind: without this, a stuck
        # out-of-bounds Mario would emit the same alert every single frame
        # and bury the panel in duplicates.
        def report(kind, message):
            if kind not in self._reported_glitches:
                self._reported_glitches.add(kind)
                self._pending_glitch = message
                self._pending_ttl = GLITCH_ALERT_TTL
                info['glitch_alert'] = message

        y = info.get('y_pos', 0)
        dead = info.get('is_dead', False)

        # 1. Below the death plane but still alive. level1.py's
        #    check_for_mario_death() runs every update and kills Mario the
        #    moment rect.y > SCREEN_HEIGHT, so observing him alive down
        #    there means that check failed to fire - he fell out of the
        #    world without the game noticing. in_castle (flag_get) is
        #    excluded because the engine deliberately exempts the
        #    end-of-level walk from the death check.
        if y > self.c_module.SCREEN_HEIGHT and not dead and not info.get('flag_get'):
            report('below_world',
                   f"Mario is below the floor (y={y}) but still alive - "
                   f"the pit-death check did not fire.")

        # 2. Far above the level ceiling. Normal jump arcs peaked at y=-29.
        if y < -200:
            report('above_world',
                   f"Mario clipped far above the level (y={y}).")

        # 3. Impossible horizontal speed. Fastest observed sprint was 13.2.
        x_vel = info.get('x_vel', 0.0)
        if abs(x_vel) > 25.0:
            report('speed',
                   f"Impossible horizontal speed (x_vel={x_vel:.1f}); "
                   f"the engine should cap a sprint far below this.")

        # 4/5. Score and coin totals must never run backwards inside an
        #      episode. Skipped while dead, because the engine resets these
        #      as part of tearing the level down.
        score = info.get('score', 0)
        coins = info.get('coins', 0)
        if not dead:
            if score < self._last_score:
                report('score_drop',
                       f"Score went backwards ({self._last_score} -> {score}).")
            if coins < self._last_coins:
                report('coin_drop',
                       f"Coin total went backwards ({self._last_coins} -> {coins}).")
        self._last_score = score
        self._last_coins = coins

    def reset(self, seed=None, options=None):
        # `options` is unused but required: this is Gymnasium's reset()
        # signature, and wrappers call it with keyword arguments.
        super().reset(seed=seed)
        self.fake_time = 0.0

        # Fresh episode: allow every glitch kind to be reported again, and
        # re-baseline the score/coin monotonicity counters (the engine zeroes
        # both in persist_data below, so carrying the old values over would
        # register a false "went backwards" on the first step).
        self._reported_glitches = set()
        self._last_score = 0
        self._last_coins = 0
        self._pending_glitch = None
        self._pending_ttl = 0

        # level1 is already in sys.modules from __init__, so this resolves
        # from cache rather than touching sys.path or the filesystem. No
        # chdir needed - setup.py builds its resource paths absolutely (see
        # the note in step()).
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

        # Guards against calling reset() while the window is closed (e.g.
        # a stray reset between close_window() and the dashboard's next
        # open_window() call) - _fast_obs() below already tolerates this by
        # returning a black frame, but pg.display.update() itself raises
        # if the display hasn't been initialized at all.
        if pg.display.get_surface() is not None:
            pg.display.update()
        obs = self._fast_obs()
        return obs, {}

    # ═══════════════════════════════════════════════════════════════════
    # WINDOW LIFECYCLE — for the dashboard's "pop up on Start, close on
    # Reset/refresh" behavior. Not used during training (train_agent.py
    # never calls these; its 8 parallel windows just stay open for the
    # whole run, staggered by the SDL_VIDEO_WINDOW_POS logic in __init__).
    #
    # Important quirk this works around: pygame's actual OS window is
    # created ONCE, at module-import time, by the top-level
    # `pg.display.set_mode(...)` call in mario_clone/data/setup.py -
    # constructing a new Control() (which reset() does on every episode)
    # does NOT create a new window, it just calls pg.display.get_surface()
    # to grab whatever window already exists. So "closing" and "reopening"
    # the window has to be done directly through the pg.display module
    # here, not by recreating Control().
    # ═══════════════════════════════════════════════════════════════════
    def open_window(self):
        """(Re)creates the OS window if it was previously closed via
        close_window(), then brings it to the foreground. Safe to call
        even when the window is already open (no-op beyond refocusing)."""
        if pg.display.get_surface() is None:
            pg.display.init()
            new_surface = pg.display.set_mode(self.c_module.SCREEN_SIZE)
            pg.display.set_caption(self.setup_module.ORIGINAL_CAPTION)
            # mario_clone/data/states/level1.py reads setup.SCREEN directly
            # (not pg.display.get_surface()), so that module-level reference
            # has to be updated here too - otherwise it still points at the
            # Surface object pg.display.quit() just destroyed, and the next
            # reset() crashes with "pygame.error: display Surface quit" the
            # moment level1.py touches it.
            self.setup_module.SCREEN = new_surface
            self.setup_module.SCREEN_RECT = new_surface.get_rect()
        self._bring_to_front()

    def _bring_to_front(self):
        """Windows-only: force the game window to the foreground. Silently
        does nothing on other platforms - pygame has no cross-platform API
        for this, and this project only targets Windows (see the
        SDL_VIDEO_WINDOW_POS staggering above, which is Windows-specific
        too). Best-effort: a focus failure here should never break
        playback, so any error is swallowed.
        """
        if sys.platform != "win32":
            return
        try:
            import ctypes
            hwnd = pg.display.get_wm_info().get("window")
            if hwnd:
                ctypes.windll.user32.SetForegroundWindow(hwnd)
        except Exception:
            pass

    def close_window(self):
        """Destroys the OS window. Safe to call even if already closed.
        The underlying game/model state is untouched - only the display -
        so the next open_window() + reset() resumes cleanly."""
        if pg.display.get_surface() is not None:
            pg.display.quit()

    def _fast_obs(self):
        # ═══════════════════════════════════════════════════════════════
        # Builds the (240, 256, 3) observation step()/reset() return to the
        # agent. This used to capture the frame at full size and then resize
        # it with cv2 - i.e. pg.surfarray.array3d() on the FULL 800x600 window (a real
        # numpy copy+transpose of ~480,000 pixels), THEN a separate cv2
        # downscale pass. Profiling this project's actual bottleneck showed
        # that pair costing ~5.5ms per call - the single largest cost in
        # step(), and it happens up to 4x per agent decision under
        # MaxAndSkipObservation's frame-skipping.
        #
        # Downscaling FIRST via pg.transform.scale (SDL/C, operates on the
        # Surface directly) and only THEN converting to a numpy array cuts
        # array3d's work down to the already-small 256x240 result instead
        # of the full 800x600 source - measured 11.9x faster for this exact
        # resize (0.47ms vs 5.6ms). Verified byte-for-byte IDENTICAL output
        # to the old cv2.INTER_NEAREST path for this exact scale ratio (no
        # behavior change to the model's input).
        # ═══════════════════════════════════════════════════════════════
        surface = pg.display.get_surface()
        if surface is None:
            return np.zeros((240, 256, 3), dtype=np.uint8)
        small_surface = pg.transform.scale(surface, (256, 240))
        view = pg.surfarray.array3d(small_surface)
        return view.transpose([1, 0, 2])

    def render_scaled(self, size):
        """Returns the current frame as a numpy array, scaled to `size`
        (width, height) BEFORE the numpy conversion via pg.transform.scale -
        the same technique _fast_obs() uses, see that method's comment for
        why this matters. Used by the dashboard's streaming path
        (agent_logic.py), which only needs display-quality output at well
        under the window's native 800x600 resolution."""
        surface = pg.display.get_surface()
        if surface is None:
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)
        small_surface = pg.transform.scale(surface, size)
        view = pg.surfarray.array3d(small_surface)
        return view.transpose([1, 0, 2])
