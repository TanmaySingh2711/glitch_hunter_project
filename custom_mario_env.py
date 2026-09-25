"""The Mario clone as a Gymnasium environment, plus the agent's view of it.

CustomMarioEnv drives one game variant (mario_clean by default, see
reporting/variants.py) one 60 fps frame per step() with synthetic key
presses, and reports what happened in the info dict: Mario's world-space
collider, velocity, score, powerups, death, the engine clock, and any GLITCH
it observed. wrap_observation() is the one observation chain every consumer
(training, the dashboard, the tools, the evaluator) puts on top of it.

With enable_evidence() (Objective 3 - off by default, so training and every
Objective-2 path run exactly as before), each detector verdict is also
frozen into a reporting.events.Detection at the substep it fired.
"""
from __future__ import annotations

import collections
import gc
import os
import sys
from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
import pygame as pg
from gymnasium import spaces
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    MaxAndSkipObservation,
    ResizeObservation,
)

import game_window
from exploration import config
from reporting.collision_invariants import CollisionInvariants
from reporting.events import Detection, ExtraDetector, json_safe
from reporting.level_design import load_design

# Disable audio to prevent sound spam during training
os.environ["SDL_AUDIODRIVER"] = "dummy"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _package_root(module: Any) -> str:
    """The game-variant directory a loaded `data` package came from."""
    return os.path.normcase(os.path.dirname(os.path.dirname(os.path.abspath(str(module.__file__)))))


def claim_game_variant(variant: str) -> str:
    """The directory of `variant`, after checking this process can run it.

    Both variants ship the game as the top-level package `data` (the upstream
    layout, deliberately untouched), and Python imports a package once per
    process. So a process hosts ONE variant at a time: asking for the other
    while one is loaded would silently run the first one's code while every
    record said otherwise - exactly the "wrong game selected" failure a QA
    report must never contain. That is refused here, loudly; the only way to
    the other variant is release_game_variant() first.
    """
    if variant not in config.GAME_VARIANTS:
        raise ValueError(f"unknown game variant {variant!r}; expected one of "
                         f"{', '.join(config.GAME_VARIANTS)}")
    game_dir = os.path.join(PROJECT_ROOT, variant)
    loaded = sys.modules.get('data')
    if loaded is not None and getattr(loaded, '__file__', None):
        if _package_root(loaded) != os.path.normcase(game_dir):
            raise RuntimeError(
                f"this process already runs the game from {_package_root(loaded)}; "
                f"it cannot also run {variant!r}. One game variant at a time - "
                f"release_game_variant() first, or start a new process.")
    return game_dir


def release_game_variant() -> None:
    """Unloads the game so this process can load the OTHER variant.

    Only for a caller that has already discarded every env built on the
    loaded variant (the dashboard's game switch). The window is destroyed and
    every `data` module is dropped, so the next CustomMarioEnv imports the
    requested variant from scratch - its code, its window, its resources.
    claim_game_variant then sees no loaded game, and nothing of the old
    variant can run: there is no mixing, just a clean second load.
    """
    if pg.display.get_init():
        pg.display.quit()
    for name in [n for n in sys.modules if n == 'data' or n.startswith('data.')]:
        del sys.modules[name]

# How many substeps a glitch alert stays attached to info before expiring.
# Must match the `skip` passed to MaxAndSkipObservation (see the delivery
# note in CustomMarioEnv._detect_glitches for why this coupling exists) -
# so it IS that skip, from the one place it is defined.
GLITCH_ALERT_TTL = config.SUBSTEPS_PER_AGENT_STEP

# ─── THE AGENT'S OBSERVATION ───
# Every agent decision spans SUBSTEPS_PER_AGENT_STEP engine frames
# (MaxAndSkipObservation, which also SUMS their rewards), seen as the last
# FRAME_STACK frames in grayscale at OBS_SIZE. The trained checkpoints'
# network input is exactly this shape, so it is defined once, here.
OBS_SIZE = (84, 84)
FRAME_STACK = 4


def wrap_observation(env: gym.Env[Any, Any],
                     skip: int = config.SUBSTEPS_PER_AGENT_STEP) -> gym.Env[Any, Any]:
    """Skip(skip) > Grayscale > Resize(84x84) > FrameStack(4) over `env`.

    `env` is whatever sits directly on the engine - GlitchHunterWrapper in
    training and the dashboard, the evaluator's read-only probe in
    evaluation/completion.py. Episode limits (TimeLimit) and SB3's Monitor
    are left to the caller, because they differ between those uses. The
    skip is SkipObservation (below): MaxAndSkipObservation's exact output,
    without rendering the frames it throws away.
    """
    wrapped: gym.Env[Any, Any] = SkipObservation(env, skip=skip)
    wrapped = GrayscaleObservation(wrapped, keep_dim=False)
    wrapped = ResizeObservation(wrapped, OBS_SIZE)
    return FrameStackObservation(wrapped, FRAME_STACK)


class FakeKeys:
    """Stands in for pygame.key.get_pressed(): the keys the agent is holding."""

    def __init__(self) -> None:
        self.keys: dict[int, bool] = {}

    def __getitem__(self, key: int) -> bool:
        return self.keys.get(key, False)

    def update(self, new_keys: dict[int, bool]) -> None:
        self.keys = new_keys


# Which keys each of the 10 discrete actions holds down (see ACTION SPACE).
_RIGHT_ACTIONS = frozenset((1, 2, 3, 4))
_LEFT_ACTIONS = frozenset((6, 8, 9))
_JUMP_ACTIONS = frozenset((2, 4, 5, 9))
_SPRINT_ACTIONS = frozenset((3, 4, 8))
_CROUCH_ACTION = 7

# The info dict step() reports when the engine has no Mario to read (see the
# AttributeError fallback in step()).
UNKNOWN_STATE_INFO: dict[str, Any] = {
    'x_pos': 0, 'y_pos': 0, 'mario_rect': None, 'viewport_x': 0, 'x_vel': 0.0,
    'on_ground': True, 'score': 0, 'coins': 0, 'status': 'small', 'flag_get': False,
    'powerup_active_count': 0, 'nearest_powerup_dx': None, 'death_cause': None,
    'is_dead': False, 'time_left': None, 'hud_time': None,
}

# Glitch thresholds, each ~2x outside the measured envelope of normal play
# (see GLITCH DETECTION below for the measurement).
ABOVE_WORLD_Y = -200
MAX_PLAUSIBLE_X_VEL = 25.0


class CustomMarioEnv(gym.Env[np.ndarray, int]):
    # No `metadata = {"render_modes": ...}` / `render_mode` here on purpose.
    # This env never goes through Gymnasium's render() protocol: it always
    # owns a real on-screen pygame window, and frames are pulled directly
    # via _fast_obs() (for the agent) and render_scaled() (for the dashboard
    # stream). Declaring a render mode without implementing render() would
    # advertise an API this class does not actually support.

    def __init__(self, game_variant: str | None = None) -> None:
        super().__init__()
        # Which game this env drives. The clean game unless a caller names
        # the other on purpose - training, the tools and the evaluator never do.
        self.game_variant = config.DEFAULT_GAME_VARIANT if game_variant is None else game_variant
        self.game_dir = claim_game_variant(self.game_variant)

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
        self._reported_glitches: set[str] = set()
        self._last_score = 0
        self._last_coins = 0
        self._pending_glitch: str | None = None
        self._pending_ttl = 0

        # ─── EVIDENCE (Objective 3) ───
        # Off unless enable_evidence() is called. While off, step() and reset()
        # do exactly what they did before Objective 3 existed: nothing below
        # is read, written or allocated.
        self.evidence_enabled = False
        self.episode_index = 0                   # resets so far
        self.holds_engine_clock = False          # set by the QA wrapper that tops the clock up
        self._extra_detectors: list[ExtraDetector] = []
        # The collision invariants (reporting/collision_invariants.py): on only
        # while evidence is, so training and evaluation run exactly as before.
        self._collision: CollisionInvariants | None = None
        self._trace: collections.deque[dict[str, Any]] = collections.deque(maxlen=1)
        self._episode_actions = bytearray()
        self._clock_holds: list[tuple[int, int]] = []
        self._actions_complete = False
        self._pending_detections: list[Detection] = []
        self.dropped_detections = 0

        # ─── EPISODE LIFECYCLE OVERRIDES (QA mode only) ───
        # Both default to "leave the engine exactly as it is", which is what
        # legacy mode and the 6M brain rely on. GlitchHunterWrapper sets them
        # in qa_exploration mode - gated on the wrapper's own mode rather than
        # a global, so a legacy wrapper in a QA-configured process still gets
        # the untouched 401-unit engine. See exploration/config.py EPISODE
        # LENGTH for the measurements behind both.
        #   episode_time_units      None -> the engine's own 401
        #   end_on_level_complete   end at the castle door, not after the
        #                           time-to-score countdown
        self.episode_time_units: int | None = None
        self.end_on_level_complete = False

        # Load the Pygame clone safely using absolute paths so SubprocVecEnv workers don't crash
        self.project_root = PROJECT_ROOT

        orig_cwd = os.getcwd()
        os.chdir(self.game_dir)
        sys.path.insert(0, self.game_dir)

        # The window is created by <variant>/data/setup.py the first time it
        # is imported in this process (see WINDOW LIFECYCLE below). Wherever
        # that happens - the dashboard, a training worker, a tool, a test - it
        # opens centred on the display the user is on (game_window.py). This
        # replaced a random offset per window, which put windows at arbitrary
        # places on the desktop.
        creates_window = 'data.setup' not in sys.modules
        game_window.request_centered_creation()

        try:
            import data
            from data import constants as c
            from data import setup, tools
            from data.states import level1

            # Belt and braces: the package that actually loaded must be the
            # requested variant's (a stray `data` earlier on sys.path would
            # otherwise win silently).
            if _package_root(data) != os.path.normcase(self.game_dir):
                raise RuntimeError(f"loaded the game from {_package_root(data)}, "
                                   f"not {self.game_dir}")

            # The clone's modules, kept so reset() never re-imports them.
            self.tools_module: Any = tools
            self.setup_module: Any = setup
            self.c_module: Any = c

            self.game: Any = tools.Control(setup.ORIGINAL_CAPTION)
            state_dict = {c.LEVEL1: level1.Level1()}
            self.game.setup_states(state_dict, c.LEVEL1)  # Skip straight to level 1!
            self.fake_time = 0.0
        finally:
            os.chdir(orig_cwd)
            if self.game_dir in sys.path:
                sys.path.remove(self.game_dir)
        if creates_window:
            game_window.center_on_current_display()
        # Set by hide_window(): the OS window exists but is off screen.
        self._window_hidden = False
        # False while SkipObservation is on a substep whose frame it will
        # discard: step() then skips the capture and returns a blank frame.
        # Always True outside a skip, so a bare step() is unchanged.
        self.render_observation = True
        self._blank_obs = np.zeros(self.observation_space.shape, dtype=np.uint8)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        pg.event.pump()
        action = int(action)
        self.fake_keys.update({
            pg.K_RIGHT: action in _RIGHT_ACTIONS,
            pg.K_LEFT: action in _LEFT_ACTIONS,
            pg.K_a: action in _JUMP_ACTIONS,           # Jump
            pg.K_s: action in _SPRINT_ACTIONS,         # Sprint
            pg.K_DOWN: action == _CROUCH_ACTION,       # Crouch
        })
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
        # getcwd + chdir(game dir) + chdir(back), because the game's music
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

        obs = self._fast_obs() if self.render_observation else self._blank_obs

        reward = 0.0
        done = False
        info: dict[str, Any] = {}

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

            # The engine's episode clock, exported so the lifecycle and its
            # tests can see the authoritative timer rather than infer it -
            # and, next to it, the number the TIME box is drawing, which in QA
            # mode is the legacy-equivalent clock (reset()).
            hud = self.game.state.overhead_info_display
            info['time_left'] = int(hud.time)
            info['hud_time'] = int(hud.display_time())

            # RL agents don't need to watch the ~3 second death animation, so
            # end the episode as soon as Mario is dead. (`mario` is already
            # bound above, so no hasattr guard is needed here — if it were
            # missing we'd have raised AttributeError long before this line.)
            done = bool(self.game.state.done or mario.dead)
            # Same reasoning for the victory sequence, in QA mode: after the
            # castle door every frame is a fixed countdown whose length is the
            # remaining time budget, not anything the agent did.
            if self.end_on_level_complete and info['flag_get']:
                done = True

        except AttributeError:
            # The level state is mid-teardown and Mario is not there to read.
            # Report "unknown" (mario_rect None - coverage records nothing),
            # never a guessed position.
            info.update(UNKNOWN_STATE_INFO)
            done = bool(self.game.state.done)

        if self.evidence_enabled:
            self._record_substep(action, info)
        self._detect_glitches(info)
        return obs, reward, done, False, info

    # ═══════════════════════════════════════════════════════════════════
    # GLITCH DETECTION — what actually feeds the dashboard's BUG TRACKER
    #
    # Sets info['glitch_alert'] to a human-readable string when the game
    # violates an invariant it is supposed to hold. dashboard_backend.py turns
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
    def _detect_glitches(self, info: dict[str, Any]) -> None:
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
        # and bury the panel in duplicates. With evidence on, the FIRST report
        # also freezes a Detection - on this substep, before the frame, the
        # trace or the state can move on (see reporting/events.py).
        def report(kind: str, message: str, metrics: dict[str, Any] | None = None,
                   detector: str | None = None, synthetic: bool = False,
                   key: str | None = None) -> None:
            key = key or kind
            if key in self._reported_glitches:
                return
            self._reported_glitches.add(key)
            self._pending_glitch = message
            self._pending_ttl = GLITCH_ALERT_TTL
            info['glitch_alert'] = message
            if self.evidence_enabled:
                self._freeze_detection(kind, message, detector or f"engine_invariants/{kind}",
                                       synthetic, metrics or {}, info)

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
                   f"the pit-death check did not fire.",
                   {'y': y, 'death_plane_y': self.c_module.SCREEN_HEIGHT,
                    'is_dead': False, 'flag_get': False})

        # 2. Far above the level ceiling. Normal jump arcs peaked at y=-29.
        if y < ABOVE_WORLD_Y:
            report('above_world',
                   f"Mario clipped far above the level (y={y}).",
                   {'y': y, 'threshold_y': ABOVE_WORLD_Y})

        # 3. Impossible horizontal speed. Fastest observed sprint was 13.2.
        x_vel = info.get('x_vel', 0.0)
        if abs(x_vel) > MAX_PLAUSIBLE_X_VEL:
            report('speed',
                   f"Impossible horizontal speed (x_vel={x_vel:.1f}); "
                   f"the engine should cap a sprint far below this.",
                   {'x_vel': x_vel, 'threshold_abs_x_vel': MAX_PLAUSIBLE_X_VEL})

        # 4/5. Score and coin totals must never run backwards inside an
        #      episode. Skipped while dead, because the engine resets these
        #      as part of tearing the level down.
        #
        #      Also skipped - and the baseline left alone - on a frame the
        #      engine could not be read (UNKNOWN_STATE_INFO, mario_rect None).
        #      Its score and coins are PLACEHOLDER zeros, not readings, and
        #      comparing them to the last real reading reported "Coin total
        #      went backwards (3 -> 0)" for a frame where nothing was measured
        #      at all (found in the Objective-3 audit; tests pin it). The other
        #      checks cannot fire on the placeholders (y 0, x_vel 0), and none of
        #      this reaches the reward: glitch_alert only feeds the dashboard.
        score = info.get('score', 0)
        coins = info.get('coins', 0)
        if info.get('mario_rect') is not None:
            if not dead:
                if score < self._last_score:
                    report('score_drop',
                           f"Score went backwards ({self._last_score} -> {score}).",
                           {'previous_score': self._last_score, 'score': score})
                if coins < self._last_coins:
                    report('coin_drop',
                           f"Coin total went backwards ({self._last_coins} -> {coins}).",
                           {'previous_coins': self._last_coins, 'coins': coins})
            self._last_score = score
            self._last_coins = coins

        # The collision invariants: Mario against what is DRAWN (evidence on only).
        # Each reports once per episode per key (per solid, per contact site).
        if self._collision is not None:
            for kind, message, metrics, detector, key in self._collision.check(self.game.state):
                report(kind, message, metrics, detector=detector, key=key)

        # Detectors added on top (today only the synthetic pipeline probe).
        # Same report-once rule, keyed per detector so two probes can both fire.
        for extra in self._extra_detectors:
            hit = extra.check(info)
            if hit is not None:
                message, metrics = hit
                report(extra.kind, message, metrics, detector=extra.detector_id,
                       synthetic=extra.synthetic, key=extra.detector_id)

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
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

        # A new episode starts a new evidence record. The trace and the action
        # log never carry frames from the episode before: an incident's context
        # and its replay must describe ONE episode. Pending detections are NOT
        # dropped - they are the previous episode's finished evidence, waiting
        # to be drained.
        self.episode_index += 1
        if self.evidence_enabled:
            self._trace.clear()
            self._episode_actions = bytearray()
            self._clock_holds = []
            self._actions_complete = True
            for extra in self._extra_detectors:
                extra.reset()
        if self._collision is not None:
            self._collision.reset()

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
        # Free the level just replaced NOW. Its sprites and groups reference
        # each other, so it is cyclic garbage that only a full collection
        # reclaims - and Python runs those rarely, so dead levels piled up.
        # Measured, one QA env over 30,000 agent steps: peak private memory
        # 688 -> 489 MB, mean 449 -> 331 MB, run time unchanged (one
        # collection per episode, and episodes are thousands of steps long).
        gc.collect()

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

        # The engine's HUD counter IS the episode clock (info.py), so a longer
        # QA episode is set there, on the one authoritative timer, rather than
        # emulated by a wrapper that would then disagree with the game about
        # when time runs out. Additive: None leaves the engine's 401 alone.
        #
        # The TIME box, though, leaves the extension out: it draws the clock a
        # legacy episode would show at this point (401 counting down), so up
        # to the substep legacy itself would time out, a QA frame is pixel for
        # pixel a legacy frame. Past that the box holds at 001 - a frame legacy
        # also drew - until the real clock runs out and it reads 000, on the
        # timeout frame. Drawing only: the clock, the timeout and the
        # time-to-score countdown all still run on the full budget. The
        # engine's own 401 is read off the fresh engine before it is replaced,
        # rather than copied here.
        if self.episode_time_units is not None:
            hud = self.game.state.overhead_info_display
            hud.display_time_offset = max(0, int(self.episode_time_units) - hud.time)
            hud.time = int(self.episode_time_units)

        # Guards against calling reset() while the window is closed (e.g.
        # a stray reset between close_window() and the dashboard's next
        # open_window() call) - _fast_obs() below already tolerates this by
        # returning a black frame, but pg.display.update() itself raises
        # if the display hasn't been initialized at all.
        if pg.display.get_surface() is not None:
            pg.display.update()
        obs = self._fast_obs()
        return obs, {}

    def hold_clock(self) -> int:
        """Keeps the engine clock from running out; returns the units added.

        QA mode only (GlitchHunterWrapper calls it before every substep - see
        config QA_TIMEOUT_ENDS_EPISODE): time alone may not end a QA episode.
        The clock loses at most one unit per engine update, so topping it up
        whenever it is down to 1 means level1.check_if_time_out() can never
        see 0. Anything else - the castle door, a death - ends the episode as
        before.

        The TIME box does not move: the same units are added to the display
        offset, so it keeps drawing the legacy-equivalent clock (001 by then,
        Phase 4D). Only while the level is being played - never during the
        castle's time-to-score countdown - and never without a QA budget,
        whose offset is what keeps the box still.
        """
        if not self.episode_time_units:
            return 0
        try:
            state = self.game.state
            hud, mario = state.overhead_info_display, state.mario
        except AttributeError:
            return 0
        if (hud.time > 1 or hud.state != self.c_module.LEVEL
                or mario.dead or mario.in_castle):
            return 0
        units = int(self.episode_time_units)
        hud.time += units
        hud.display_time_offset += units
        if self.evidence_enabled:
            # Before the substep it affects, so the index is that substep's.
            self._clock_holds.append((len(self._episode_actions), units))
        return units

    # ═══════════════════════════════════════════════════════════════════
    # EVIDENCE (Objective 3)
    #
    # Everything here only READS the engine. It never moves Mario, never
    # changes a timer and never touches the observation, so an episode played
    # with evidence on is frame-for-frame the episode played with it off
    # (tests/test_incident_capture.py pins that).
    # ═══════════════════════════════════════════════════════════════════
    def enable_evidence(self, context_substeps: int = config.EVIDENCE_CONTEXT_SUBSTEPS) -> None:
        """Start keeping the per-frame trace and the action log, and freezing
        a Detection whenever a detector fires. Takes effect fully from the
        next reset(): an episode already under way has no action log from its
        start, so its incidents are marked not replayable rather than guessed."""
        self.evidence_enabled = True
        if self._collision is None:
            self._collision = CollisionInvariants(load_design())
        self._trace = collections.deque(maxlen=max(1, int(context_substeps)))
        self._episode_actions = bytearray()
        self._clock_holds = []
        self._actions_complete = False

    def disable_evidence(self) -> None:
        """Back to the pre-Objective-3 behaviour: no trace, no log, no
        detections, no extra detectors."""
        self.evidence_enabled = False
        self._collision = None
        self.dropped_detections = 0
        self._extra_detectors.clear()
        self._pending_detections.clear()
        self._trace.clear()
        self._episode_actions = bytearray()
        self._clock_holds = []
        self._actions_complete = False

    def add_detector(self, detector: ExtraDetector) -> None:
        """Adds a detector on top of the built-in ones (the synthetic probe)."""
        self._extra_detectors.append(detector)

    @property
    def extra_detectors(self) -> tuple[ExtraDetector, ...]:
        return tuple(self._extra_detectors)

    def last_trace_entry(self) -> dict[str, Any] | None:
        """The per-frame record of the most recent substep (evidence on)."""
        return dict(self._trace[-1]) if self._trace else None

    def drain_detections(self) -> list[Detection]:
        """The Detections frozen since the last drain, oldest first."""
        out, self._pending_detections = self._pending_detections, []
        return out

    # Pending detections are drained after every agent step by whoever
    # enabled evidence. This cap only matters if nobody drains: a runaway list
    # must not grow for ever, and what was dropped is counted, not hidden.
    MAX_PENDING_DETECTIONS = 64

    def _record_substep(self, action: int, info: Mapping[str, Any]) -> None:
        self._episode_actions.append(action)
        mario = self._mario_extras()
        rect = info.get('mario_rect') or (None, None, None, None)
        self._trace.append({
            'substep': len(self._episode_actions) - 1,
            'engine_time_ms': round(self.fake_time, 3),
            'action': action,
            'x': rect[0], 'y': rect[1], 'w': rect[2], 'h': rect[3],
            'x_vel': info.get('x_vel'), 'y_vel': mario.get('y_vel'),
            'mario_state': mario.get('state'), 'on_ground': info.get('on_ground'),
            'status': info.get('status'), 'is_dead': info.get('is_dead'),
            'score': info.get('score'), 'coins': info.get('coins'),
            'time_left': info.get('time_left'), 'viewport_x': info.get('viewport_x'),
        })

    def _freeze_detection(self, kind: str, message: str, detector: str, synthetic: bool,
                          metrics: Mapping[str, Any], info: Mapping[str, Any]) -> None:
        if len(self._pending_detections) >= self.MAX_PENDING_DETECTIONS:
            self.dropped_detections += 1
            return
        state = {k: v for k, v in info.items() if k != 'glitch_alert'}
        self._pending_detections.append(Detection(
            kind=kind, message=message, detector=detector, synthetic=synthetic,
            metrics=json_safe(metrics), state=json_safe(state),
            mario=json_safe(self._mario_extras()),
            episode_index=self.episode_index,
            episode_substep=len(self._episode_actions) - 1,
            engine_time_ms=float(self.fake_time),
            frame=self._capture_frame(),
            trace=tuple(dict(t) for t in self._trace),
            geometry=self._visible_geometry(),
            actions=bytes(self._episode_actions),
            clock_holds=tuple(self._clock_holds),
            actions_complete=self._actions_complete,
            game_variant=self.game_variant,
            episode_time_units=self.episode_time_units,
            end_on_level_complete=bool(self.end_on_level_complete),
            holds_engine_clock=bool(self.holds_engine_clock),
            extra_detectors=tuple(d.spec() for d in self._extra_detectors),
        ))

    def _capture_frame(self) -> np.ndarray | None:
        """The frame this substep drew, full resolution, as its own copy.
        None when there is no window (then there is nothing to show, and the
        incident says so rather than substituting another frame)."""
        surface: pg.Surface | None = pg.display.get_surface()
        if surface is None:
            return None
        return np.ascontiguousarray(pg.surfarray.array3d(surface).transpose(1, 0, 2))

    def _mario_extras(self) -> dict[str, Any]:
        """Engine fields the info dict does not carry but a report needs."""
        try:
            mario = self.game.state.mario
        except AttributeError:
            return {}
        return {
            'y_vel': float(getattr(mario, 'y_vel', 0.0)),
            'state': str(getattr(mario, 'state', '')),
            'big': bool(getattr(mario, 'big', False)),
            'fire': bool(getattr(mario, 'fire', False)),
            'invincible': bool(getattr(mario, 'invincible', False)),
            'hurt_invincible': bool(getattr(mario, 'hurt_invincible', False)),
            'in_castle': bool(getattr(mario, 'in_castle', False)),
            'facing_right': bool(getattr(mario, 'facing_right', True)),
        }

    # The level's collider and sprite groups (level1.py), by what they hold.
    _GEOMETRY_GROUPS = (('ground', 'ground_group'), ('pipe', 'pipe_group'),
                        ('step', 'step_group'), ('brick', 'brick_group'),
                        ('coin_box', 'coin_box_group'), ('enemy', 'enemy_group'),
                        ('shell', 'shell_group'), ('powerup', 'powerup_group'))

    def _visible_geometry(self) -> tuple[dict[str, Any], ...]:
        """Every collider and sprite at least partly inside the camera's view
        at the trigger - i.e. exactly what the trigger frame shows - in world
        coordinates."""
        state = self.game.state
        viewport = getattr(state, 'viewport', None)
        if viewport is None:
            return ()
        x0, x1 = int(viewport.x), int(viewport.x) + int(viewport.w)
        found: list[dict[str, Any]] = []
        for label, attr in self._GEOMETRY_GROUPS:
            for sprite in getattr(state, attr, ()) or ():
                rect = getattr(sprite, 'rect', None)
                if rect is None or rect.right < x0 or rect.left > x1:
                    continue
                item: dict[str, Any] = {'group': label, 'x': int(rect.x), 'y': int(rect.y),
                                        'w': int(rect.w), 'h': int(rect.h)}
                name = getattr(sprite, 'name', None)
                if isinstance(name, str):
                    item['name'] = name
                sprite_state = getattr(sprite, 'state', None)
                if isinstance(sprite_state, str):
                    item['state'] = sprite_state
                found.append(item)
        return tuple(found)

    # ═══════════════════════════════════════════════════════════════════
    # WINDOW LIFECYCLE — for the dashboard's game window. Not used during
    # training (train_agent.py never calls these; its 8 parallel windows
    # stay open for the whole run).
    #
    # THREADING RULE (dashboard): every call below, and every step(), must
    # come from the ONE thread that created the window. Windows delivers a
    # window's messages - move, minimise, the X button - only to the thread
    # that created it, and DESTROYS the window when that thread exits. The
    # dashboard used to create it on a short-lived socket-handler thread: the
    # popup vanished as soon as the handler returned, and the X button never
    # reached anyone. dashboard_service.GameWindowService now owns it.
    #
    # Important quirk this works around: pygame's actual OS window is
    # created ONCE, at module-import time, by the top-level
    # `pg.display.set_mode(...)` call in <variant>/data/setup.py -
    # constructing a new Control() (which reset() does on every episode)
    # does NOT create a new window, it just calls pg.display.get_surface()
    # to grab whatever window already exists. So "closing" and "reopening"
    # the window has to be done directly through the pg.display module
    # here, not by recreating Control().
    #
    # Three states: OPEN, HIDDEN (hide_window: off screen, nothing destroyed)
    # and CLOSED (close_window: destroyed). open_window() centres the window
    # when it creates or re-shows it, and never re-centres one already open.
    # open_window(minimized=True) - the dashboard's Start - puts a created or
    # re-shown window in the taskbar instead, minimised and inactive; clicking
    # it there restores it at that centred position.
    # ═══════════════════════════════════════════════════════════════════
    def window_state(self) -> str:
        if pg.display.get_surface() is None:
            return 'closed'
        return 'hidden' if self._window_hidden else 'open'

    def open_window(self, minimized: bool = False) -> str:
        """Makes the game window visible and returns what it had to do:
        'created' (it was closed), 'shown' (it was hidden) or 'focused' (it
        was already open - only restored if minimised, never moved).

        minimized=True: a created or re-shown window goes to the taskbar,
        minimised and inactive, centred for when it is restored; one already
        open is left exactly as the user has it ('unchanged')."""
        state = self.window_state()
        if minimized and state == 'open':
            return 'unchanged'
        if state == 'closed':
            game_window.request_centered_creation()
            pg.display.init()
            new_surface = pg.display.set_mode(self.c_module.SCREEN_SIZE,
                                              pg.HIDDEN if minimized else 0)
            pg.display.set_caption(self.setup_module.ORIGINAL_CAPTION)
            # <variant>/data/states/level1.py reads setup.SCREEN directly
            # (not pg.display.get_surface()), so that module-level reference
            # has to be updated here too - otherwise it still points at the
            # Surface object pg.display.quit() just destroyed, and the next
            # reset() crashes with "pygame.error: display Surface quit" the
            # moment level1.py touches it. The live Control holds its own
            # reference too, which lets an episode continue without a reset.
            self.setup_module.SCREEN = new_surface
            self.setup_module.SCREEN_RECT = new_surface.get_rect()
            self.game.screen = new_surface
            game_window.center_on_current_display()
        elif state == 'hidden':
            if not minimized:
                game_window.show()
            game_window.center_on_current_display()
        else:
            game_window.show()          # restores it if the user minimised it
        self._window_hidden = False
        if minimized:
            game_window.show_minimized()
        else:
            game_window.bring_to_front()
        return {'closed': 'created', 'hidden': 'shown'}.get(state, 'focused')

    def hide_window(self) -> None:
        """Takes the window off screen WITHOUT destroying it: the game state,
        the frame on it and every Surface survive, so open_window() shows the
        episode exactly where it was."""
        if pg.display.get_surface() is not None and game_window.hide():
            self._window_hidden = True

    def close_window(self) -> None:
        """Destroys the OS window. Safe to call even if already closed.
        The underlying game/model state is untouched - only the display -
        so the next open_window() + reset() resumes cleanly."""
        if pg.display.get_surface() is not None:
            pg.display.quit()
        self._window_hidden = False

    def poll_close_request(self) -> bool:
        """Drains the window's event queue; True if the user clicked its X
        (or pressed Alt+F4). Nothing else in this project reads pygame
        events, so draining them all is safe - and keeps the queue from
        filling while the dashboard is paused."""
        if pg.display.get_surface() is None:
            return False
        closing = False
        for event in pg.event.get():
            if event.type == pg.QUIT or event.type == getattr(pg, 'WINDOWCLOSE', -1):
                closing = True
        return closing

    def _fast_obs(self) -> np.ndarray:
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
        surface: pg.Surface | None = pg.display.get_surface()
        if surface is None:
            return np.zeros((240, 256, 3), dtype=np.uint8)
        small_surface = pg.transform.scale(surface, (256, 240))
        view = pg.surfarray.array3d(small_surface)
        return view.transpose([1, 0, 2])

    def render_scaled(self, size: tuple[int, int]) -> np.ndarray:
        """Returns the current frame as a numpy array, scaled to `size`
        (width, height) BEFORE the numpy conversion via pg.transform.scale -
        the same technique _fast_obs() uses, see that method's comment for
        why this matters. Used by the dashboard's streaming path
        (dashboard_backend.py), which only needs display-quality output at well
        under the window's native 800x600 resolution."""
        surface: pg.Surface | None = pg.display.get_surface()
        if surface is None:
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)
        small_surface = pg.transform.scale(surface, size)
        view = pg.surfarray.array3d(small_surface)
        return view.transpose([1, 0, 2])


class SkipObservation(MaxAndSkipObservation[Any, Any, Any]):
    """MaxAndSkipObservation that only renders the frames it keeps.

    The parent repeats the action for `skip` engine frames and max-pools the
    LAST TWO observations; the first skip-2 are computed and thrown away.
    Capturing and downscaling a frame (_fast_obs, ~0.6 ms) is the largest
    first-party cost in a substep, so this tells the engine, through
    render_observation, which substeps' frames will actually be read.
    Measured on the QA training stack, headless, 3 x 1,500 agent steps:
    8.85 -> 7.68 ms per agent step: 13% less time, 15% more rollout
    throughput per worker (tools/benchmark_step.py).

    Identical output by construction: the frames that ARE rendered are
    rendered exactly as before, the max-pool and the reward sum are the
    parent's own arithmetic, and an episode ending on a skipped substep
    returns the same stale buffer the parent would have (it only ever writes
    the buffer on the last two substeps). tests/test_env.py pins it against
    the plain MaxAndSkipObservation, frame for frame.
    """

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        # Duck-typed: any engine exposing the flag opts in; anything else is
        # stepped exactly like MaxAndSkipObservation would.
        base: Any = self.env.unwrapped
        engine = base if hasattr(base, "render_observation") else None
        total_reward = 0.0
        terminated = truncated = False
        info: dict[str, Any] = {}
        try:
            for i in range(self._skip):
                if engine is not None:
                    engine.render_observation = i >= self._skip - 2
                obs, reward, terminated, truncated, info = self.env.step(action)
                if i == self._skip - 2:
                    self._obs_buffer[0] = obs
                if i == self._skip - 1:
                    self._obs_buffer[1] = obs
                total_reward += float(reward)
                if terminated or truncated:
                    break
        finally:
            if engine is not None:
                engine.render_observation = True
        return np.max(self._obs_buffer, axis=0), total_reward, terminated, truncated, info
