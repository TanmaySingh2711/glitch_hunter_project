"""The agent behind the dashboard: one env, one model, one frame stream.

Split out of agent_logic.py, which is now only the reward wrapper. This is
the RUNTIME half - it loads the approved brain, builds the same observation
chain training uses, and turns every agent step into a JPEG frame plus a log
line for the browser. dashboard_service.GameWindowService drives it from its
single game thread; app.py never touches it directly.

Objective 3: the env runs with evidence on, and every detector verdict it
freezes is handed to the incident pipeline (reporting/pipeline.py) straight
after the step that produced it - on this same thread, before the next step.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, replace
from typing import Any, cast

import cv2
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from agent_logic import ACTION_NAMES, GlitchHunterWrapper
from common.fileio import sha256_of
from custom_mario_env import PROJECT_ROOT, CustomMarioEnv, release_game_variant, wrap_observation
from exploration import config
from exploration import coverage as coverage_mod
from exploration.lifecycle import classify_end
from reporting import schema
from reporting.events import SyntheticProbe
from reporting.pipeline import IncidentPipeline, SessionRecorder
from reporting.provenance import objective2_record, session_provenance
from reporting.run_report import RunReports, build_run_record, new_run_id
from reporting.store import IncidentStore

log = logging.getLogger(__name__)

# Must match LEGACY_CHECKPOINT_NAME in train_agent.py — the completion-phase
# master checkpoint that legacy training writes on finish and on Ctrl+C.
CHECKPOINT_NAME = "mario_brain_checkpoint"


@dataclass(frozen=True)
class DashboardConfig:
    """How this dashboard runs. One game variant at a time: switching
    (switch_game_variant) unloads the old game completely before the new one
    loads (custom_mario_env.claim_game_variant / release_game_variant)."""
    game_variant: str = config.DEFAULT_GAME_VARIANT
    # SYNTHETIC pipeline probes (reporting.events.SyntheticProbe), by world
    # x. Empty in normal use: these fire on ordinary gameplay, only to prove
    # the reporting pipeline, and everything they produce is labelled so.
    synthetic_probes: tuple[int, ...] = ()
    incidents_dir: str = os.path.join(PROJECT_ROOT, config.INCIDENTS_DIR)
    run_reports_dir: str = os.path.join(PROJECT_ROOT, config.RUN_REPORTS_DIR)
    reproduce: bool = True


_config = DashboardConfig()
_global_env: gym.Env[Any, Any] | None = None
_global_model: PPO | None = None
_brain_path: str | None = None
_reward_mode = "legacy_completion"
_pipeline: IncidentPipeline | None = None
_run_reports: RunReports | None = None
_provenance: dict[str, Any] = {}

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


def _is_approved_brain(path: str) -> bool:
    """True when `path` is byte-for-byte the brain FINAL_OBJECTIVE2.json approved."""
    record = objective2_record()
    expected = (record or {}).get("brain", {}).get("sha256")
    return bool(expected) and sha256_of(path) == expected


def select_checkpoint() -> tuple[str, str]:
    """(model path, reward mode) for the brain the dashboard should show.

    ─── THE REWARD MODE FOLLOWS THE CHECKPOINT, NOT THE CONFIG ───
    The dashboard displays whatever brain is on disk, so it has to display
    that brain's OWN reward. Reading config.REWARD_MODE here would show QA
    rewards for a completion-phase policy - numbers that describe an
    objective this checkpoint was never trained on - and would additionally
    let the drought terminate episodes early during a demo, which looks
    exactly like a bug.

    ─── THE APPROVED BRAIN FIRST (Objective 3) ───
    The main brain (glitch_hunter_main_brain.zip, the approved Objective-2
    brain) is the authoritative QA source, and it is only used if its SHA-256
    still matches the closure record. The root
    glitch_hunter_qa.zip is an ordinary working file any training run
    overwrites, so it is only the fallback; then the 6M completion brain.
    """
    final = config.FINAL_BRAIN_PATH
    if os.path.exists(final):
        if _is_approved_brain(final):
            return final, "qa_exploration"
        log.warning("%s does not match the approved Objective-2 hash; not using it", final)
    qa_path = f"{config.CHECKPOINT_NAME_QA}.zip"
    if os.path.exists(qa_path):
        return qa_path, "qa_exploration"
    return f"{CHECKPOINT_NAME}.zip", "legacy_completion"


def _dashboard_coverage(mode: str, model_path: str | None = None
                        ) -> coverage_mod.SpatialCoverage | None:
    """In QA mode, coverage against the map that brain actually built.

    A fresh empty map would credit it with rediscovering the whole level
    every time the dashboard is opened. It is loaded into memory only: the
    dashboard never writes a coverage file, so the frozen pair stays frozen.
    """
    if mode != "qa_exploration":
        return None
    coverage = coverage_mod.SpatialCoverage(testable_mask=coverage_mod.load_testable())
    paired = (config.FINAL_COVERAGE_PATH if model_path == config.FINAL_BRAIN_PATH
              else f"{config.CHECKPOINT_NAME_QA}_coverage.npz")
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
    global _global_env, _global_model, _brain_path, _reward_mode
    if _global_env is not None and _global_model is not None:
        return _global_env, _global_model

    model_path, mode = select_checkpoint()
    base = CustomMarioEnv(game_variant=_config.game_variant)
    # Before the first reset, so every episode's action log starts at its reset.
    base.enable_evidence()
    for x in _config.synthetic_probes:
        base.add_detector(SyntheticProbe(x))
    env = wrap_observation(GlitchHunterWrapper(base, reward_mode=mode,
                                               coverage=_dashboard_coverage(mode, model_path)))

    # Falls back to an untrained policy so the dashboard still runs (badly)
    # rather than crashing outright when no checkpoint is present.
    if os.path.exists(model_path):
        log.info("[DASHBOARD] %s (%s) on %s", model_path, mode, _config.game_variant)
        model = PPO.load(model_path, env=env, device="auto")
        _brain_path = model_path
    else:
        log.warning("%s not found — running an UNTRAINED policy.", model_path)
        model = PPO('CnnPolicy', env, verbose=0)
        _brain_path = None
    _reward_mode = mode
    _global_env, _global_model = env, model
    return env, model


def configure(cfg: DashboardConfig) -> None:
    """Sets how this process runs. Only before the env exists: the game
    variant cannot change once a game has been imported."""
    global _config
    if _global_env is not None and cfg.game_variant != _config.game_variant:
        raise RuntimeError("the game variant is fixed once the env exists")
    _config = cfg


def switch_game_variant(variant: str) -> bool:
    """Makes `variant` the game under test; False if it already is.

    Everything built on the old game goes: the incident pipeline (its
    worker finishes what it was rendering first), the env and its window,
    the model loaded against that env, and the game's modules. The next
    preload() then builds all of it again on the new variant, so every
    incident records the game that actually produced it.
    """
    global _config, _global_env, _global_model, _pipeline, _provenance, _brain_path
    if variant not in config.GAME_VARIANTS:
        raise ValueError(f"unknown game variant {variant!r}")
    if variant == _config.game_variant:
        return False
    with env_lock:
        if _pipeline is not None:
            _pipeline.close()
        _pipeline, _provenance = None, {}
        if _global_env is not None:
            _base(_global_env).close_window()
            _global_env.close()
        _global_env = _global_model = None
        _brain_path = None
        release_game_variant()
        _config = replace(_config, game_variant=variant)
    log.info("[DASHBOARD] game under test is now %s", variant)
    return True


def ensure_pipeline(notify: Callable[[str, dict[str, Any]], Any] | None = None
                    ) -> IncidentPipeline:
    """The incident pipeline, created once (after the env and brain, whose
    identity every incident records)."""
    global _pipeline, _provenance, _run_reports
    if _run_reports is None or _run_reports.root != _config.run_reports_dir:
        _run_reports = RunReports(_config.run_reports_dir, notify=notify)
    if _pipeline is None:
        _provenance = session_provenance(_brain_path, _reward_mode, _config.game_variant)
        _pipeline = IncidentPipeline(IncidentStore(_config.incidents_dir), notify=notify,
                                     reproduce=_config.reproduce)
        if _pipeline.recovered.get("moved_incomplete") or _pipeline.recovered.get("requeued"):
            log.warning("[INCIDENTS] recovered after an interrupted run: %s", _pipeline.recovered)
        brain = _provenance["brain"]
        log.info("[INCIDENTS] %s | game %s (tree %s...) | brain %s (%s)",
                 _config.incidents_dir, _config.game_variant,
                 _provenance["game"]["tree_sha256"][:12], brain.get("path"),
                 "approved Objective-2 brain" if brain.get("approved_objective2_brain")
                 else "NOT the approved Objective-2 brain")
    return _pipeline


def describe() -> dict[str, Any]:
    """What the dashboard shows about this process: game, brain, test mode."""
    game = _provenance.get("game", {})
    brain = _provenance.get("brain", {})
    return {"game_variant": _config.game_variant,
            "game_tree_sha256": game.get("tree_sha256"),
            "game_is_clean_baseline": game.get("matches_pinned_clean_tree"),
            "brain_path": brain.get("path"),
            "brain_approved": brain.get("approved_objective2_brain"),
            "synthetic_probes": list(_config.synthetic_probes),
            "reward_mode": _reward_mode}


def open_agent_window() -> str:
    """Ensures the env/model exist and the game window is in the taskbar;
    returns what open_window() had to do. Called on every 'Start Testing'
    click: a closed or hidden window appears there minimised (the live view
    is on the dashboard; clicking the taskbar button shows the window,
    centred), and one the user already has open is left as it is."""
    with env_lock:
        env, _model = _ensure_global_env_and_model()
        return _base(env).open_window(minimized=True)


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

    def __init__(self, cfg: DashboardConfig | None = None) -> None:
        if cfg is not None:
            configure(cfg)
        # Set by app.py: how the incident pipeline's worker tells the browser
        # a report is ready. Optional - the pipeline works without it.
        self.notify: Callable[[str, dict[str, Any]], Any] | None = None

    def configure(self, cfg: DashboardConfig) -> None:
        configure(cfg)

    @property
    def pipeline(self) -> IncidentPipeline | None:
        return _pipeline

    @property
    def run_reports(self) -> RunReports | None:
        return _run_reports

    def describe(self) -> dict[str, Any]:
        return describe()

    def preload(self) -> None:
        """Loads the model and builds the env - which creates the window, on
        this thread - then hides the window until the first Start. Then the
        incident pipeline, which records that brain and game's identity."""
        with env_lock:
            env, _model = _ensure_global_env_and_model()
            _base(env).hide_window()
        ensure_pipeline(self.notify)

    def open_window(self) -> str:
        return open_agent_window()

    def switch_game(self, variant: str) -> bool:
        """switch_game_variant, then the same pre-load a fresh start does."""
        if not switch_game_variant(variant):
            return False
        self.preload()
        return True

    def hide_window(self) -> None:
        with env_lock:
            if _global_env is not None:
                _base(_global_env).hide_window()

    def close_window(self) -> None:
        close_agent_window()

    def poll_close_request(self) -> bool:
        with env_lock:
            return _global_env is not None and _base(_global_env).poll_close_request()

    def begin_incident_session(self) -> None:
        """Reset: the Bug Tracker's session list starts empty again."""
        if _pipeline is not None:
            _pipeline.begin_session()

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


def _capture_incidents(env: gym.Env[Any, Any], recorder: SessionRecorder) -> list[dict[str, Any]]:
    """Hands every Detection the step just froze to the pipeline, and says
    what became of each ('new', 'duplicate' or 'failed').

    Runs right after env.step() and BEFORE this step's stream frame joins the
    context ring: that frame was drawn after the trigger, so it must not be
    presented as leading up to it.
    """
    drain = getattr(_base(env), "drain_detections", None)
    if drain is None:
        return []
    detections = drain()
    if not detections:
        return []
    if _pipeline is None:
        log.error("%d detection(s) arrived before the incident pipeline existed; "
                  "they were not recorded", len(detections))
        return [{"status": "failed", "incident_id": None, "summary": None,
                 "error": "the incident pipeline was not running"}]
    out = []
    for det in detections:
        outcome = _pipeline.capture(det, recorder.context(_reward_mode, _provenance))
        out.append({"status": outcome.status, "incident_id": outcome.incident_id,
                    "summary": outcome.summary, "error": outcome.error,
                    "first_in_session": outcome.first_in_session})
    return out


def _finish_run(env: gym.Env[Any, Any], info: dict[str, Any], recorder: SessionRecorder,
                started: Any, furthest_x: int, bugs: list[dict[str, Any]]) -> dict[str, Any]:
    """How a clean-game run ended - and, when it reached the castle, its
    run report (reporting/run_report.py): the "no bugs found" counterpart of
    an incident. A report that cannot be written is logged, never raised: the
    run has ended either way."""
    end_reason = str(info.get('episode_end_reason')
                     or classify_end(info, bool(info.get('safety_reset_reason'))))
    out: dict[str, Any] = {'end_reason': end_reason, 'report': None}
    if end_reason != 'level_complete' or _run_reports is None:
        return out
    try:
        base = _base(env)
        record = build_run_record(
            run_id=new_run_id(schema.utc_now()), created=schema.utc_now(), started=started,
            session_id=recorder.session_id, env_episode_index=recorder.env_episode_index,
            agent_steps=recorder.episode_step, end_reason=end_reason, info=info,
            furthest_x=furthest_x, bugs=bugs, reward_mode=_reward_mode, provenance=_provenance)
        size = tuple(getattr(base.c_module, 'SCREEN_SIZE', (800, 600)))
        out['report'] = _run_reports.write(record, base.render_scaled((int(size[0]), int(size[1]))),
                                           recorder.recent_frames())
        log.info("[RUN] %s: %s after %d agent steps -> %s", record['run_id'],
                 out['report']['headline'], recorder.episode_step, _run_reports.root)
    except Exception:
        log.exception("could not write the run report")
    return out


def run_mario_agent() -> Generator[dict[str, Any], None, None]:
    """An endless session: one dict per agent step - the JPEG frame, the
    action, the step number, the reward, the log line and what became of any
    anomaly the step detected - resetting the episode whenever it ends. The
    caller owns the pacing.

    On the clean game the step that ends an episode also carries 'run_end'
    (why it ended, and the run report when Mario reached the castle), so the
    dashboard can stop there. The bugged game carries none and plays on."""
    env, model = _ensure_global_env_and_model()
    obs = _reset_obs(env)
    recorder = SessionRecorder()
    recorder.begin_episode(int(getattr(_base(env), "episode_index", 0)))
    run_started, furthest_x = schema.utc_now(), 0
    run_bugs: dict[str, dict[str, Any]] = {}

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
        recorder.step_taken(action_val)

        # Render already-downscaled to STREAM_SIZE via render_scaled() -
        # see custom_mario_env.py for why this avoids a full-resolution
        # array3d() call (the same technique _fast_obs() uses for the
        # model's observation, just at a different target size for display).
        frame_bytes = encode_frame(_base(env).render_scaled(STREAM_SIZE))
        incidents = _capture_incidents(env, recorder)
        # The GIF's context is these same stream frames: no second capture,
        # so the evidence costs the game loop nothing per step.
        recorder.push_frame(frame_bytes)
        for o in incidents:
            if o.get("status") in ("new", "duplicate") and o.get("summary"):
                run_bugs[str(o["incident_id"])] = o["summary"]
        rect = info.get('mario_rect')
        if rect:
            furthest_x = max(furthest_x, int(rect[0]))
        run_end = (_finish_run(env, info, recorder, run_started, furthest_x,
                               list(run_bugs.values()))
                   if done and _config.game_variant == config.CLEAN_GAME_VARIANT else None)

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

        item: dict[str, Any] = {
            'frame': frame_bytes,
            'action': action_val,
            'step': step_count,
            'reward': float(reward),
            'log': step_log_line(step_count, action_val, float(reward), info),
            'incidents': incidents,
        }
        if run_end is not None:
            item['run_end'] = run_end
        yield item

        if done:
            obs = _reset_obs(env)
            recorder.begin_episode(int(getattr(_base(env), "episode_index", 0)))
            run_started, furthest_x = schema.utc_now(), 0
            run_bugs = {}
