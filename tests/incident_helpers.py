"""Builders shared by the incident tests: a realistic Detection and capture
context without running the game, so the pipeline, the store and the
renderers can be exercised in milliseconds. The tests that need the REAL
engine (exact frames, replay) build their Detections from it instead.
"""
import cv2
import numpy as np

from exploration import config
from reporting.events import Detection
from reporting.schema import CaptureContext, ContextFrame

FAKE_TREE = "f" * 64


def frame(seed=0, h=600, w=800):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (92, 148, 252)                               # sky
    img[538:, :] = (200, 76, 12)                             # ground
    x = int(rng.integers(50, 700))
    img[498:538, x:x + 30] = (255, 0, 0)                     # "Mario"
    return img


def detection(kind="speed", x=1000, y=498, substep=203, episode=1, synthetic=False,
              metrics=None, frame_img=None, dead=False, rect=True, detector=None,
              trace_len=24):
    metrics = metrics if metrics is not None else {
        "speed": {"x_vel": 60.0, "threshold_abs_x_vel": 25.0},
        "above_world": {"y": -500, "threshold_y": -200},
        "below_world": {"y": 610, "death_plane_y": 600, "is_dead": False, "flag_get": False},
        "score_drop": {"previous_score": 500, "score": 100},
        "coin_drop": {"previous_coins": 3, "coins": 1},
        "synthetic_probe": {"collider_right_x": x + 30, "probe_x": x},
    }.get(kind, {})
    state = {"x_pos": x, "y_pos": y, "mario_rect": [x, y, 30, 40] if rect else None,
             "viewport_x": max(0, x - 300), "x_vel": float(metrics.get("x_vel", 6.0)),
             "on_ground": True, "score": 100, "coins": 1, "status": "small",
             "flag_get": False, "is_dead": dead, "time_left": 1500, "hud_time": 380}
    trace = tuple({"substep": substep - trace_len + 1 + i, "engine_time_ms": 0.0, "action": 3,
                   "x": x - (trace_len - 1 - i) * 10, "y": y, "w": 30, "h": 40,
                   "x_vel": 10.0, "y_vel": 0.0, "mario_state": "walk", "on_ground": True,
                   "status": "small", "is_dead": False, "score": 100, "coins": 1,
                   "time_left": 1500, "viewport_x": max(0, x - 300)}
                  for i in range(min(trace_len, substep + 1)))
    return Detection(
        kind=kind, message=f"test {kind} at x={x}",
        detector=detector or (f"synthetic_probe/x>={x}" if synthetic else f"engine_invariants/{kind}"),
        synthetic=synthetic, metrics=metrics, state=state,
        mario={"y_vel": 0.0, "state": "walk", "big": False, "fire": False, "invincible": False,
               "hurt_invincible": False, "in_castle": False, "facing_right": True},
        episode_index=episode, episode_substep=substep,
        engine_time_ms=substep * 1000.0 / 60.0,
        frame=frame(substep) if frame_img is None else frame_img,
        trace=trace,
        geometry=({"group": "ground", "x": 0, "y": 538, "w": 2953, "h": 60},
                  {"group": "enemy", "x": x + 120, "y": 498, "w": 40, "h": 40,
                   "name": "goomba", "state": "walk"}),
        actions=bytes([3]) * (substep + 1), clock_holds=(), actions_complete=True,
        game_variant=config.CLEAN_GAME_VARIANT, episode_time_units=None,
        end_on_level_complete=False, holds_engine_clock=False,
        extra_detectors=({"type": "synthetic_probe", "x": x},) if synthetic else ())


def jpeg(seed):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame(seed, 360, 480), cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


def provenance(tree=FAKE_TREE):
    return {"brain": {"path": "brain.zip", "present": True, "sha256": "b" * 64,
                      "num_timesteps": 16_000_000, "approved_objective2_brain": True},
            "game": {"variant": config.CLEAN_GAME_VARIANT, "tree_sha256": tree,
                     "matches_pinned_clean_tree": True, "declared_injected_bugs": []},
            "code": {"git_commit": "c" * 40, "uncommitted_changes": False},
            "runtime": {"python": "3.12", "platform": "test"}}


def context(det, frames=8, tree=FAKE_TREE, session_id="S-TEST"):
    agent_step = det.episode_substep // config.SUBSTEPS_PER_AGENT_STEP + 1
    ctx_frames = tuple(ContextFrame(agent_step - frames + i, agent_step - frames + i, jpeg(i))
                       for i in range(frames) if agent_step - frames + i >= 1)
    return CaptureContext(session_id=session_id, session_agent_step=agent_step,
                          episode_agent_step=agent_step, env_episode_index=det.episode_index,
                          agent_action=3, recent_agent_actions=(3,) * min(agent_step, 8),
                          context_frames=ctx_frames, reward_mode="qa_exploration",
                          provenance=provenance(tree))
