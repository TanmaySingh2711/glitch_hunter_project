"""The agent behind the dashboard: one env, one model, one frame stream.

Split out of agent_logic.py, which is now only the reward wrapper. This is
the RUNTIME half - it loads whichever brain is on disk, builds the same
observation chain training uses, and turns every agent step into a JPEG
frame plus a log line for the browser. dashboard_service.GameWindowService
drives it from its single game thread; app.py never touches it directly.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Generator
from typing import Any, cast

import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from agent_logic import ACTION_NAMES, GlitchHunterWrapper
from custom_mario_env import CustomMarioEnv, wrap_observation
from exploration import config
from exploration import coverage as coverage_mod

log = logging.getLogger(__name__)

# Must match LEGACY_CHECKPOINT_NAME in train_agent.py — the completion-phase
# master checkpoint that legacy training writes on finish and on Ctrl+C.
CHECKPOINT_NAME = "mario_brain_checkpoint"

_global_env: gym.Env[Any, Any] | None = None
_global_model: PPO | None = None

# ═══════════════════════════════════════════════════════════════════════
# ENV_LOCK — serialises every touch of the environment and its pygame window
#
# app.py runs Flask-SocketIO in threading mode, so a click handler ("Reset",
# or a browser disconnect) executes on a DIFFERENT OS thread from the frame
# loop. Without this lock, close_window() can destroy the SDL display while
# the frame loop is part-way through env.step() or render_scaled().
#
# That is not hypothetical. A stress test firing start/stop/reset at random
# intervals for 20 seconds produced 9 crashes in the frame loop:
#     pygame.error: video system not initialized
#     pygame.error: cannot convert without pygame.display initialized
# The old eventlet mode could not hit this - one thread, switching only at
# explicit yield points - so it arrived with the move to real threads.
#
# Rule: hold this for the duration of any single env operation, and NEVER
# across a sleep. Teardown then lands cleanly between two steps instead of
# in the middle of one.
#
# The dashboard itself no longer relies on it: every window and env call now
# runs on dashboard_service's single game thread (the only thread Windows
# lets drive the window), so there is nothing left to race. It stays as a
# guard for the module-level helpers, which remain callable from anywhere.
# ═══════════════════════════════════════════════════════════════════════
env_lock = threading.RLock()

# Streamed frames are resized down from the native 800x600 window before
# JPEG encoding. The dashboard displays them scaled into a much smaller
# panel anyway (CSS `object-fit: contain`), so sending full resolution only
# costs encode time and bandwidth for pixels nobody sees larger. 480x360
# keeps the exact 4:3 aspect ratio at 36% of the original pixel count.
STREAM_SIZE = (480, 360)
STREAM_JPEG_QUALITY = 80  # default cv2 quality (~95) costs real encode time
                          # for no visible difference at this display size
FPS_REPORT_EVERY_S = 2.0


def _base(env: gym.Env[Any, Any]) -> CustomMarioEnv:
    """The CustomMarioEnv at the bottom of the wrapper chain."""
    return cast(CustomMarioEnv, env.unwrapped)


def select_checkpoint() -> tuple[str, str]:
    """(model path, reward mode) for the brain the dashboard should show.

    ─── THE REWARD MODE FOLLOWS THE CHECKPOINT, NOT THE CONFIG ───
    The dashboard displays whatever brain is on disk, so it has to display
    that brain's OWN reward. Reading config.REWARD_MODE here would show QA
    rewards for a completion-phase policy - numbers that describe an
    objective this checkpoint was never trained on - and would additionally
    let the drought terminate episodes early during a demo, which looks
    exactly like a bug.

    The QA brain wins when it exists, because by then it is the current one.
    """
    qa_path = f"{config.CHECKPOINT_NAME_QA}.zip"
    if os.path.exists(qa_path):
        return qa_path, "qa_exploration"
    return f"{CHECKPOINT_NAME}.zip", "legacy_completion"


def _dashboard_coverage(mode: str) -> coverage_mod.SpatialCoverage | None:
    """In QA mode, coverage against the map that brain actually built.

    A fresh empty map would credit it with rediscovering the whole level
    every time the dashboard is opened.
    """
    if mode != "qa_exploration":
        return None
    coverage = coverage_mod.SpatialCoverage(testable_mask=coverage_mod.load_testable())
    paired = f"{config.CHECKPOINT_NAME_QA}_coverage.npz"
    if os.path.exists(paired):
        try:
            coverage.load(paired)
        except Exception as exc:      # never block playback on telemetry
            log.warning("could not load %s: %s", paired, exc)
    return coverage


def _ensure_global_env_and_model() -> tuple[gym.Env[Any, Any], PPO]:
    """Creates the global env + loads the model on first use, and returns
    both. Shared by run_mario_agent() and open_agent_window() so a window can
    be popped up (on the very first 'Start Testing' click) without
    duplicating this setup logic in two places."""
    global _global_env, _global_model
    if _global_env is not None and _global_model is not None:
        return _global_env, _global_model

    model_path, mode = select_checkpoint()
    env = wrap_observation(GlitchHunterWrapper(CustomMarioEnv(), reward_mode=mode,
                                               coverage=_dashboard_coverage(mode)))

    # Falls back to an untrained policy so the dashboard still runs (badly)
    # rather than crashing outright when no checkpoint is present.
    if os.path.exists(model_path):
        log.info("[DASHBOARD] %s (%s)", model_path, mode)
        model = PPO.load(model_path, env=env, device="auto")
    else:
        log.warning("%s not found — running an UNTRAINED policy.", model_path)
        model = PPO('CnnPolicy', env, verbose=0)
    _global_env, _global_model = env, model
    return env, model


def open_agent_window() -> str:
    """Ensures the env/model exist and the game window is visible and
    focused; returns what open_window() had to do. Called on every 'Start
    Testing' click (not just the first), so the window reliably comes to the
    front even if it's buried behind other windows from earlier."""
    with env_lock:
        env, _model = _ensure_global_env_and_model()
        return _base(env).open_window()


def close_agent_window() -> None:
    """Closes the game window if one exists. Called on dashboard reset. The
    model and env stay loaded in memory - only the OS window closes - so the
    next open_agent_window() call is fast, not a full reload."""
    with env_lock:
        if _global_env is not None:
            _base(_global_env).close_window()


class DashboardBackend:
    """The real work behind dashboard_service.GameWindowService.

    Every method runs on the service's one game thread - the thread that
    creates the window, and so the only one Windows lets drive it. env_lock
    is still taken, so the module-level helpers above stay safe to call.
    """

    def preload(self) -> None:
        """Loads the model and builds the env - which creates the window, on
        this thread - then hides the window until the first Start."""
        with env_lock:
            env, _model = _ensure_global_env_and_model()
            _base(env).hide_window()

    def open_window(self) -> str:
        return open_agent_window()

    def hide_window(self) -> None:
        with env_lock:
            if _global_env is not None:
                _base(_global_env).hide_window()

    def close_window(self) -> None:
        close_agent_window()

    def poll_close_request(self) -> bool:
        with env_lock:
            return _global_env is not None and _base(_global_env).poll_close_request()

    def new_session(self) -> Generator[dict[str, Any], None, None]:
        return run_mario_agent()

    def stop_audio(self) -> None:
        # Best-effort: the mixer may not be initialized at all (audio is
        # forced to the "dummy" driver in custom_mario_env.py).
        try:
            import pygame as pg
            if pg.mixer.get_init():
                pg.mixer.music.stop()
                pg.mixer.stop()
        except Exception:
            log.debug("could not stop audio", exc_info=True)


def encode_frame(frame: np.ndarray | None) -> bytes:
    """JPEG bytes for the browser, or b"" when there is no frame.

    BGR for cv2, then JPEG at a quality that doesn't waste CPU on precision
    nobody sees at this display size. Raw bytes, not base64:
    flask-socketio/python-socketio send `bytes` values as a native binary
    WebSocket frame automatically (the client's socket.io library reassembles
    it transparently). Base64 was costing ~33% more payload plus real
    encode/decode CPU time on both ends for no benefit.
    """
    if frame is None:
        return b""
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
    return buffer.tobytes() if ok else b""


def step_log_line(step: int, action: int, reward: float, info: dict[str, Any]) -> str:
    """The LOG TERMINAL line for one step - or the BUG TRACKER line, when the
    env raised a glitch alert on it. main.js routes on the '🚨 BUG FOUND: '
    marker and reads the step number from the 'Step N:' prefix; without that
    prefix every bug entry used to show as "Step ?"."""
    if info.get('glitch_alert'):
        return f"Step {step}: 🚨 BUG FOUND: {info['glitch_alert']}"
    action_name = ACTION_NAMES.get(action, "Unknown")
    return f"Step {step}: Action: {action_name} ({action}) | Reward: {reward:.2f}"


def _reset_obs(env: gym.Env[Any, Any]) -> np.ndarray:
    obs, _info = env.reset()
    return np.array(obs, copy=True)


def run_mario_agent() -> Generator[dict[str, Any], None, None]:
    """An endless session: one dict per agent step - the JPEG frame, the
    action, the step number, the reward and the log line - resetting the
    episode whenever it ends. The caller owns the pacing."""
    env, model = _ensure_global_env_and_model()
    obs = _reset_obs(env)

    step_count = 0
    fps_window_start = time.perf_counter()
    fps_window_frames = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        action_val = int(np.asarray(action).item())

        obs_raw, reward, terminated, truncated, info = env.step(action_val)
        done = terminated or truncated
        obs = np.array(obs_raw, copy=True)
        step_count += 1

        # Render already-downscaled to STREAM_SIZE via render_scaled() -
        # see custom_mario_env.py for why this avoids a full-resolution
        # array3d() call (the same technique _fast_obs() uses for the
        # model's observation, just at a different target size for display).
        frame_bytes = encode_frame(_base(env).render_scaled(STREAM_SIZE))

        # ─── SERVER-SIDE FPS INSTRUMENTATION ───
        # Logs the actual measured frame rate every ~2 seconds, so "is it
        # really running at 60fps" is something you can read from the
        # terminal instead of guessing from how smooth the browser looks.
        fps_window_frames += 1
        now = time.perf_counter()
        elapsed = now - fps_window_start
        if elapsed >= FPS_REPORT_EVERY_S:
            log.info("[STREAM FPS] %.1f", fps_window_frames / elapsed)
            fps_window_start = now
            fps_window_frames = 0

        yield {
            'frame': frame_bytes,
            'action': action_val,
            'step': step_count,
            'reward': float(reward),
            'log': step_log_line(step_count, action_val, float(reward), info),
        }

        if done:
            obs = _reset_obs(env)
