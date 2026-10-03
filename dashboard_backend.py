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
import sys
import threading
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, replace
from typing import Any, cast

import cv2
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from agent_logic import ACTION_NAMES, GlitchHunterWrapper
from common.fileio import sha256_of
from custom_mario_env import PROJECT_ROOT, CustomMarioEnv, release_game_variant, wrap_observation
from evaluation.completion import progress_of
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
    # Every run keeps the brain's top pick, so episodes repeat exactly. Only
    # for tools that need the same route twice (validate_incident_pipeline's
    # repeat sighting); the dashboard itself draws a new route every run.
    fixed_route: bool = False
    # Seeds the draw of each move, so a test can replay the same drawn routes
    # (tests/bug_incident_run.py). None in use: a new route every session.
    route_seed: int | None = None


_config = DashboardConfig()
_global_env: gym.Env[Any, Any] | None = None
_global_model: PPO | None = None
_brain_path: str | None = None
_reward_mode = "legacy_completion"
_pipeline: IncidentPipeline | None = None
_run_reports: RunReports | None = None
_provenance: dict[str, Any] = {}
# The coverage map the brain is shown against (QA brains only), for the
# dashboard's live panel: its reachable pixels covered when it was loaded,
# plus every new one since (banked at each reset, see _bank_coverage).
_coverage: coverage_mod.SpatialCoverage | None = None
_coverage_base = 0
_coverage_banked = 0

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
    global _coverage, _coverage_base, _coverage_banked
    if _global_env is not None and _global_model is not None:
        return _global_env, _global_model

    model_path, mode = select_checkpoint()
    base = CustomMarioEnv(game_variant=_config.game_variant)
    # Before the first reset, so every episode's action log starts at its reset.
    base.enable_evidence()
    for x in _config.synthetic_probes:
        base.add_detector(SyntheticProbe(x))
    _coverage = _dashboard_coverage(mode, model_path)
    # One full count, here at load; the live panel then adds what is new.
    _coverage_base = _coverage.covered_testable() if _coverage is not None else 0
    _coverage_banked = 0
    env = wrap_observation(GlitchHunterWrapper(base, reward_mode=mode, coverage=_coverage))

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
    global _config, _global_env, _global_model, _pipeline, _provenance, _brain_path, _coverage
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
        _coverage = None
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


def brain_facts() -> dict[str, Any]:
    """The loaded brain as the dashboard's panels show it: the file, whether
    it is the approved Objective-2 brain, how long it trained (from the file
    itself) and how many learned parameters it has (from the loaded model)."""
    brain = _provenance.get("brain", {})
    model = _global_model
    parameters = (sum(int(p.numel()) for p in model.policy.parameters())
                  if model is not None else None)
    return {"path": brain.get("path"), "approved": brain.get("approved_objective2_brain"),
            "num_timesteps": brain.get("num_timesteps"), "parameters": parameters,
            "reward_mode": _reward_mode, "actions": len(ACTION_NAMES)}


def coverage_now() -> dict[str, int] | None:
    """Reachable pixels covered now: the map the brain built in training plus
    everything new since the dashboard loaded it. None when no map is shown
    (the 6M brain, or no reachable-space mask installed)."""
    cov = _coverage
    if cov is None or cov.testable is None or cov.testable_total is None:
        return None
    new = _coverage_banked + int(cov.episode_new)
    return {"covered": _coverage_base + new, "total": cov.testable_total,
            "new_since_start": new}


def _bank_coverage() -> None:
    """Adds the ending episode's new reachable pixels to the running total,
    before a reset zeroes the coverage map's per-episode counter."""
    global _coverage_banked
    if _coverage is not None:
        _coverage_banked += int(_coverage.episode_new)


def telemetry(info: dict[str, Any], run: int, run_step: int, action: int,
              furthest_x: int) -> dict[str, Any]:
    """What the live panel shows for one step - every value read from this
    step, the loaded brain or the coverage map, never estimated."""
    rect = info.get('mario_rect')
    return {"run": run, "run_step": run_step, "action": ACTION_NAMES.get(action, "Unknown"),
            "x": int(rect[0]) if rect else None, "progress": round(progress_of(furthest_x), 4),
            "phase": info.get('episode_phase'), "coverage": coverage_now()}


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

    def brain_facts(self) -> dict[str, Any]:
        return brain_facts()

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
    _bank_coverage()
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


def choose_action(model: PPO, obs: np.ndarray, rng: np.random.Generator,
                  temperature: float = config.DASHBOARD_POLICY_TEMPERATURE) -> int:
    """The brain's next action: drawn from its own policy, not its top pick.

    The policy gives each of the 10 actions a probability. Always taking the
    top one (greedy) replays the same route every run; drawing from the
    probabilities - sharpened by `temperature` (< 1 favours the likelier
    actions) - makes each run a different route the brain itself learned.
    temperature 0 is the old greedy choice."""
    with torch.no_grad():
        tensor, _ = model.policy.obs_to_tensor(obs)
        dist: Any = model.policy.get_distribution(tensor).distribution
        logits = dist.logits[0].cpu().numpy().astype(np.float64)
    if temperature <= 0:
        return int(np.argmax(logits))
    z = logits / temperature
    p = np.exp(z - z.max())
    return int(rng.choice(len(p), p=p / p.sum()))


def route_temperature(run_number: int, fixed_runs: int) -> float:
    """The first `fixed_runs` runs of a session keep the brain's top pick (the
    fixed route); every later run draws a new one."""
    return 0.0 if run_number <= fixed_runs else config.DASHBOARD_POLICY_TEMPERATURE


def pit_trap_ahead(level: Any) -> bool:
    """Mario stands on a raised solid within config.EDGE_LOOKAHEAD px of its
    right edge, and the drop beyond is a pit he could never leave: a floor whose
    walls on both sides are higher than a standing jump rises
    (config.STANDING_JUMP_RISE) and whose width is too short to run up a
    faster jump (config.RUNUP_WIDTH). Judged from the level's own solids,
    the same everywhere - in Level 1-1 only the valley between the first
    staircase's two columns is such a pit."""
    mario = getattr(level, "mario", None)
    solids = list(getattr(level, "ground_step_pipe_group", None) or [])
    if mario is None or not solids:
        return False
    r = mario.rect
    if str(getattr(mario, "state", "")) not in ("walk", "standing") or float(getattr(mario, "y_vel", 0)) != 0:
        return False
    feet = r.bottom + 2
    edge = next((x for x in range(r.right, r.right + config.EDGE_LOOKAHEAD + 1, 4)
                 if not any(s.rect.collidepoint(x, feet) for s in solids)), None)
    if edge is None:
        return False                       # solid ground all the way ahead: no edge near
    ahead = edge
    under = [s.rect.top for s in solids if s.rect.left <= ahead < s.rect.right and s.rect.top > r.bottom]
    if not under:
        return False                       # a bottomless drop: not a pit to be stuck in
    floor = min(under)
    if floor - r.bottom <= config.STANDING_JUMP_RISE:
        return False                       # he could climb back out on the left
    return any(ahead < s.rect.left < ahead + config.RUNUP_WIDTH
               and s.rect.top < floor - config.STANDING_JUMP_RISE for s in solids)


def pit_traps(level: Any) -> list[tuple[int, int]]:
    """Every pit in the level Mario could never leave, as (left wall's right
    edge, right wall's left edge): two solids whose tops stand higher above
    the floor between them than a standing jump rises, closer together than
    a run-up needs. Worked out from the level's own solids."""
    solids = [s.rect for s in (getattr(level, "ground_step_pipe_group", None) or [])]
    traps = []
    for a in solids:
        for b in solids:
            gap = b.left - a.right
            if not config.MARIO_WIDTH <= gap < config.RUNUP_WIDTH:     # room to fall in
                continue
            floors = [f.top for f in solids if f.left <= a.right and f.right >= b.left and f.top > max(a.top, b.top)]
            if not floors:
                continue
            floor = min(floors)
            between = [c for c in solids if c.right > a.right and c.left < b.left and c.top < floor]
            if (not between and floor - a.top > config.STANDING_JUMP_RISE
                    and floor - b.top > config.STANDING_JUMP_RISE):
                traps.append((a.right, b.left))
    return sorted(set(traps))


def wall_ahead(level: Any) -> bool:
    """A solid stands right in front of Mario (within a few px of his toes),
    above his feet: a step or a pipe side to climb."""
    mario = getattr(level, "mario", None)
    solids = getattr(level, "ground_step_pipe_group", None) or []
    if mario is None:
        return False
    r = mario.rect
    return any(s.rect.collidepoint(r.right + 6, r.bottom - 10) for s in solids)


def near_pit_trap(x: int, traps: list[tuple[int, int]]) -> bool:
    """Within config.PIT_CAUTION_PX before a pit trap, or over it."""
    return any(left - config.PIT_CAUTION_PX <= x <= right for left, right in traps)


class StuckEscape:
    """Keeps a drawn route from getting stuck, so a run ends at the castle or
    in a death, never stuck. Two rules, both about the level as it stands,
    not about any one place:

      * At the edge of a pit he could never leave (pit_trap_ahead), Mario
        takes a full-height running jump across it. A standing jump rises
        about 161 px, a running one about 178, and the valley between the
        first staircase's two columns is walled 172 px high on both sides and
        too narrow to run in.
      * Stalled. After config.DASHBOARD_ESCAPE_AFTER steps without a new
        furthest x, attempts follow one another until he reaches new ground:
        a running jump held for a drawn number of steps (a hop up to full
        height), and every other attempt a run-up first - back off to the
        left, stop, run right and jump at speed (a 172 px pipe needs it).

    The lengths are drawn from the session's random source, so attempts
    differ. Otherwise the brain plays. Every move is logged like the
    brain's, so a replay repeats it exactly."""

    RUN_JUMP, RUN, BACK, STAND = 4, 3, 8, 0          # agent_logic.ACTION_NAMES

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.plan: list[int] = []
        self.attempts = 0
        self.active = False
        self.edge_jump = False

    def _draw(self, span: tuple[int, int]) -> int:
        return int(self.rng.integers(span[0], span[1] + 1))

    def _attempt(self) -> list[int]:
        jump = [self.RUN_JUMP] * self._draw(config.DASHBOARD_ESCAPE_HOLD)
        self.attempts += 1
        if self.attempts % 2 == 0:
            return ([self.BACK] * self._draw(config.DASHBOARD_ESCAPE_BACK)
                    + [self.STAND] * config.DASHBOARD_ESCAPE_PAUSE
                    + [self.RUN] * config.DASHBOARD_ESCAPE_RUN + jump
                    + [self.RUN] * config.DASHBOARD_ESCAPE_RUNUP)
        return jump + [self.RUN] * config.DASHBOARD_ESCAPE_RUNUP

    WALK, HOP = 1, 2                                  # Walk Right, Walk Right + Jump

    def climb(self, level: Any) -> int:
        """Near a pit trap: walk (no sprint, so no long flying jump), and hop
        up whatever stands in front, one step at a time."""
        mario = getattr(level, "mario", None)
        grounded = (mario is not None and str(getattr(mario, "state", "")) in ("walk", "standing")
                    and float(getattr(mario, "y_vel", 0)) == 0)
        if grounded and wall_ahead(level):
            self.plan = [self.HOP] * (config.CLIMB_HOP_HOLD - 1) + [self.WALK] * 2
            return self.HOP
        return self.WALK

    def next_action(self, since_progress: int, level: Any = None,
                    careful: bool = False) -> int | None:
        """The next move from here, or None while the brain should play."""
        if self.plan and (self.edge_jump or careful or since_progress >= config.DASHBOARD_ESCAPE_AFTER):
            return self.plan.pop(0)
        self.edge_jump = False
        if level is not None and pit_trap_ahead(level):
            self.edge_jump, self.active = True, True
            self.plan = [self.RUN_JUMP] * config.DASHBOARD_EDGE_JUMP_HOLD
            return self.plan.pop(0)
        if careful:
            self.active = True
            return self.climb(level)
        if since_progress < config.DASHBOARD_ESCAPE_AFTER:
            self.active, self.plan, self.attempts = False, [], 0
            return None
        self.active = True
        if not self.plan:
            self.plan = self._attempt()
        return self.plan.pop(0)


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
    run_number = 1                 # runs (episodes) played in this session
    # A fresh random source per session: each run takes a new route, except
    # the bugged game's first, which keeps the fixed route that meets all six
    # benchmark bugs (config.DASHBOARD_FIXED_ROUTE_RUNS).
    rng = np.random.default_rng(_config.route_seed)
    fixed_runs = (sys.maxsize if _config.fixed_route
                  else config.DASHBOARD_FIXED_ROUTE_RUNS.get(_config.game_variant, 0))
    temperature = route_temperature(run_number, fixed_runs)
    since_progress = 0             # agent steps since Mario last reached a new furthest x
    escape = StuckEscape(rng)
    traps = pit_traps(getattr(getattr(_base(env), 'game', None), 'state', None))
    model.policy.set_training_mode(False)

    step_count = 0
    fps_window_start = time.perf_counter()
    fps_window_frames = 0
    while True:
        level = getattr(getattr(_base(env), 'game', None), 'state', None)
        here = getattr(getattr(level, 'mario', None), 'rect', None)
        careful = here is not None and near_pit_trap(int(here.x), traps)
        forced = escape.next_action(since_progress, level, careful) if temperature > 0 else None
        action_val = forced if forced is not None else choose_action(model, obs, rng, temperature)

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
            since_progress = 0 if int(rect[0]) > furthest_x else since_progress + 1
            furthest_x = max(furthest_x, int(rect[0]))
        if not done and temperature > 0 and since_progress >= config.DASHBOARD_STUCK_STEPS:
            # A drawn route stuck in a trap: end the run now rather than when
            # the engine's own safety reset comes, minutes later.
            done, info = True, {**info, 'episode_end_reason': 'safety_reset'}
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
            'telemetry': telemetry(info, run_number, recorder.episode_step, action_val,
                                   furthest_x),
        }
        if run_end is not None:
            item['run_end'] = run_end
        yield item

        if done:
            obs = _reset_obs(env)
            recorder.begin_episode(int(getattr(_base(env), "episode_index", 0)))
            run_started, furthest_x, since_progress = schema.utc_now(), 0, 0
            run_bugs = {}
            run_number += 1
            temperature = route_temperature(run_number, fixed_runs)
