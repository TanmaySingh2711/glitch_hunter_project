"""Turns a Detection plus its capture context into the raw evidence files and
the canonical incident record. Pure functions: no disk, no threads - the
pipeline decides where the bytes go.
"""
from __future__ import annotations

import base64
import datetime
import hashlib
import io
import zipfile
from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from agent_logic import ACTION_NAMES
from exploration import config
from reporting import schema
from reporting.events import Detection, json_safe

TITLES = {
    "below_world": "Mario alive below the death plane",
    "above_world": "Mario far above the level",
    "speed": "Impossible horizontal speed",
    "score_drop": "Score ran backwards",
    "coin_drop": "Coin total ran backwards",
    "synthetic_probe": "SYNTHETIC pipeline test event (not a game bug)",
}
# A fixed timestamp inside the ZIP, so the same frames always give the same
# archive bytes (and the same SHA-256).
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_PNG_COMPRESSION = 3   # lossless either way; 3 is ~4x faster than the default 9


def encode_png(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_PNG_COMPRESSION, _PNG_COMPRESSION])
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


def decode_png(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not a PNG image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pixel_sha256(frame: np.ndarray) -> str:
    """Hash of the pixels themselves (shape + RGB bytes), independent of how
    they were compressed - what a replay compares against."""
    h = hashlib.sha256(f"{frame.shape[1]}x{frame.shape[0]}x{frame.shape[2]}:".encode())
    h.update(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    return h.hexdigest()


def context_zip(frames: tuple[schema.ContextFrame, ...]) -> bytes:
    """The stream frames before the trigger, uncompressed JPEGs + an index."""
    buf = io.BytesIO()
    index = []
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for i, f in enumerate(frames):
            name = f"frames/{i:04d}.jpg"
            info = zipfile.ZipInfo(name, date_time=_ZIP_EPOCH)
            zf.writestr(info, f.jpeg)
            index.append({"file": name, "session_agent_step": f.session_agent_step,
                          "episode_agent_step": f.episode_agent_step})
        zf.writestr(zipfile.ZipInfo("frames.json", date_time=_ZIP_EPOCH),
                    schema.dumps({"frames": index,
                                  "substeps_between_frames": config.SUBSTEPS_PER_AGENT_STEP}))
    return buf.getvalue()


def read_context_zip(data: bytes) -> list[tuple[dict[str, Any], np.ndarray]]:
    """(index entry, RGB frame) pairs, oldest first."""
    import json
    out = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        index = json.loads(zf.read("frames.json"))["frames"]
        for entry in index:
            img = cv2.imdecode(np.frombuffer(zf.read(entry["file"]), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                out.append((entry, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return out


def actions_digest(actions: bytes) -> str:
    return hashlib.sha256(actions).hexdigest()


def trajectory_doc(det: Detection, ctx: schema.CaptureContext, incident_id: str) -> dict[str, Any]:
    return {
        "schema": schema.TRAJECTORY_SCHEMA,
        "incident_id": incident_id,
        "episode_index": det.episode_index,
        "trigger_substep": det.episode_substep,
        "trace": [json_safe(t) for t in det.trace],
        "trace_note": ("One entry per engine frame, oldest first; the last entry IS the "
                       "trigger frame. At most config.EVIDENCE_CONTEXT_SUBSTEPS frames."),
        "episode_actions": {
            "encoding": "base64 of one byte per engine frame (action id 0-9), from reset",
            "complete_from_reset": det.actions_complete,
            "substeps": len(det.actions),
            "sha256": actions_digest(det.actions),
            "data": base64.b64encode(det.actions).decode("ascii"),
        },
        "clock_holds": [list(h) for h in det.clock_holds],
        "recent_agent_actions": [{"id": a, "name": ACTION_NAMES.get(a, "?")}
                                 for a in ctx.recent_agent_actions],
    }


def decode_actions(doc: Mapping[str, Any]) -> bytes:
    """The action log from a trajectory document, checked against its hash."""
    enc = doc["episode_actions"]
    raw = base64.b64decode(enc["data"])
    if actions_digest(raw) != enc["sha256"] or len(raw) != enc["substeps"]:
        raise ValueError("the recorded action log does not match its own SHA-256")
    return raw


def build_record(*, incident_id: str, created: datetime.datetime, det: Detection,
                 ctx: schema.CaptureContext, classification: Mapping[str, Any],
                 fingerprint: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    """The canonical incident record (reporting/schema.py)."""
    rect = det.state.get("mario_rect")
    viewport_x = det.state.get("viewport_x")
    location: dict[str, Any] = {"world_collider": None, "collider_center": None,
                                "screen": None, "viewport_x": viewport_x}
    if rect and len(rect) == 4:
        x, y, w, h = (int(v) for v in rect)
        location.update(world_collider={"x": x, "y": y, "w": w, "h": h},
                        collider_center={"x": x + w / 2.0, "y": y + h / 2.0})
        if isinstance(viewport_x, int):
            location["screen"] = {"x": x - viewport_x, "y": y,
                                  "note": "top-left of the collider in the 800x600 trigger frame"}

    notes: list[str] = []
    per_step = config.SUBSTEPS_PER_AGENT_STEP
    derived_step = det.episode_substep // per_step + 1
    if det.actions_complete:
        episode_agent_step = derived_step
        step_match = derived_step == ctx.episode_agent_step
        if not step_match:
            notes.append(f"The engine places the trigger in agent step {derived_step} of the "
                         f"episode; the caller counted {ctx.episode_agent_step}.")
    else:
        episode_agent_step = ctx.episode_agent_step
        step_match = True
        notes.append("Evidence recording began part-way through this episode: the action log "
                     "does not start at the reset, so the episode cannot be replayed.")
    episode_match = det.episode_index == ctx.env_episode_index
    if not episode_match:
        notes.append(f"The detection belongs to engine episode {det.episode_index} but the "
                     f"caller was on episode {ctx.env_episode_index}.")
    if det.frame is None:
        notes.append("No game window existed at the trigger, so there is no trigger frame.")

    kind = det.kind
    return {
        "schema": schema.INCIDENT_SCHEMA,
        "incident_id": incident_id,
        "created_utc": schema.iso_utc(created),
        "synthetic": det.synthetic,
        "summary": {
            "category": kind,
            "title": TITLES.get(kind, kind.replace("_", " ")),
            "description": det.message,
        },
        "detector": {
            "id": det.detector, "kind": kind, "message": det.message,
            "metrics": json_safe(det.metrics),
            "semantics": ("Evaluated on every engine frame after the game updates; reports each "
                          "kind at most once per episode (custom_mario_env._detect_glitches)."),
        },
        "classification": json_safe(classification),
        "fingerprint": json_safe(fingerprint),
        "location": location,
        "timing": {
            "session_id": ctx.session_id,
            "episode_index": det.episode_index,
            "episode_substep": det.episode_substep,
            "substep_in_agent_step": det.episode_substep % per_step,
            "episode_agent_step": episode_agent_step,
            "session_agent_step": ctx.session_agent_step,
            "engine_time_ms": round(det.engine_time_ms, 3),
            "captured_utc": schema.iso_utc(created),
        },
        "gameplay": {
            "agent_action": {"id": ctx.agent_action,
                             "name": ACTION_NAMES.get(ctx.agent_action, "?")},
            "substep_action": (det.actions[-1] if det.actions else None),
            "mario": json_safe(det.mario),
            "state": json_safe(det.state),
        },
        "geometry": [json_safe(g) for g in det.geometry],
        "provenance": json_safe({**ctx.provenance, "reward_mode": ctx.reward_mode}),
        "evidence": json_safe(evidence),
        "reproduction_inputs": {
            "replayable": bool(det.actions_complete),
            "why_not": (None if det.actions_complete else
                        "the action log does not start at the episode's reset"),
            "game_variant": det.game_variant,
            "episode_time_units": det.episode_time_units,
            "end_on_level_complete": det.end_on_level_complete,
            "holds_engine_clock": det.holds_engine_clock,
            "extra_detectors": [json_safe(s) for s in det.extra_detectors],
        },
        "consistency": {"episode_match": episode_match, "agent_step_match": step_match,
                        "notes": notes},
    }
