"""The canonical incident record: one schema for every consumer.

The pipeline writes it, the Markdown and PDF renderers read it, the replay
reads it, the dashboard reads it, the tests validate it. Nothing invents a
second representation.

    incident.json   IMMUTABLE FACTS, written once when the anomaly is captured
                    and made read-only. What was measured, where, when, by
                    which brain on which game, and which evidence files hold
                    the pixels and the trajectory.
    manifest.json   the bundle's mutable STATUS: which derived artifacts exist
                    (with hashes), how rendering and reproduction went, and
                    every change since. Kept separate so a later retry or
                    re-render can never rewrite a fact.

Serialisation is deterministic (sorted keys, fixed indent, UTF-8), so the same
record always produces the same bytes and the same SHA-256.
"""
from __future__ import annotations

import datetime
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

INCIDENT_SCHEMA = "glitch-hunter.incident/1"
MANIFEST_SCHEMA = "glitch-hunter.incident-manifest/1"
TRAJECTORY_SCHEMA = "glitch-hunter.incident-trajectory/1"
REPRODUCTION_SCHEMA = "glitch-hunter.incident-reproduction/1"
OCCURRENCE_SCHEMA = "glitch-hunter.incident-occurrence/1"

# INC-<UTC date>-<UTC time>-<6 random hex>. Sortable by capture time; the
# random part and an exclusive directory creation make it unique.
# ASCII digits only, and matched with fullmatch(): in Python `\d` also accepts
# other scripts' digits, and `$` accepts a trailing newline.
INCIDENT_ID_RE = re.compile(r"INC-[0-9]{8}-[0-9]{6}-[0-9a-f]{6}")

# Every file a bundle may hold. Reports re-rendered later get a version
# suffix (report.v2.pdf) instead of replacing the original.
RAW_ARTIFACTS = ("incident.json", "trigger.png", "context_frames.zip", "trajectory.json")
DERIVED_ARTIFACTS = ("context.gif", "report.md", "report.pdf", "reproduction.json")
_VERSION = r"(\.v(?:[2-9]|[1-9][0-9]+))?"
ARTIFACT_NAME_RE = re.compile(
    r"(incident\.json|trigger\.png|context_frames\.zip|trajectory\.json|manifest\.json"
    rf"|context{_VERSION}\.gif|report{_VERSION}\.(md|pdf)|reproduction{_VERSION}\.json)")

SEVERITIES = ("critical", "high", "medium", "low", "none", "unclassified")
CONFIDENCE_LEVELS = ("high", "medium", "low", "not_applicable")
REPRODUCTION_STATUSES = ("not_attempted", "pending", "reproduced", "reproduced_state_only",
                         "not_reproduced", "diverged", "not_possible", "timeout", "error")
RENDER_STATUSES = ("pending", "done", "failed", "skipped")


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def iso_utc(moment: datetime.datetime) -> str:
    """2026-09-23T10:15:30.123456Z - always UTC, always the same shape."""
    return moment.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def new_incident_id(moment: datetime.datetime, token: str) -> str:
    return f"INC-{moment.astimezone(datetime.UTC):%Y%m%d-%H%M%S}-{token}"


def dumps(record: Mapping[str, Any]) -> bytes:
    """The one serialisation every record, manifest and log line uses."""
    return (json.dumps(record, sort_keys=True, indent=1, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


@dataclass(frozen=True)
class ContextFrame:
    """One stream frame from before the trigger (the dashboard's own JPEG)."""
    session_agent_step: int
    episode_agent_step: int
    jpeg: bytes


@dataclass(frozen=True)
class CaptureContext:
    """What the caller of the env knows and the env does not: which session
    and agent step this is, what the agent chose, the frames it streamed,
    and the provenance of the whole session."""
    session_id: str
    session_agent_step: int              # agent steps this session, incl. the triggering one
    episode_agent_step: int              # agent steps this episode, incl. the triggering one
    env_episode_index: int               # the env's episode counter when the step was taken
    agent_action: int                    # the action of the triggering agent step
    recent_agent_actions: tuple[int, ...]    # oldest first, ending with agent_action
    context_frames: tuple[ContextFrame, ...]  # oldest first, all BEFORE the triggering step
    reward_mode: str
    provenance: Mapping[str, Any] = field(default_factory=dict)


# ── validation ──────────────────────────────────────────────────────────────
_REQUIRED: dict[str, type | tuple[type, ...]] = {
    "schema": str, "incident_id": str, "created_utc": str, "synthetic": bool,
    "summary": dict, "detector": dict, "classification": dict, "fingerprint": dict,
    "location": dict, "timing": dict, "gameplay": dict, "geometry": list,
    "provenance": dict, "evidence": dict, "reproduction_inputs": dict, "consistency": dict,
}
_NESTED: dict[str, dict[str, type | tuple[type, ...]]] = {
    "summary": {"category": str, "title": str, "description": str},
    "detector": {"id": str, "kind": str, "message": str, "metrics": dict},
    "classification": {"severity": str, "severity_rationale": str,
                       "detector_confidence": dict},
    "fingerprint": {"signature": str, "kind": str, "game_tree_sha256": str},
    "timing": {"session_id": str, "episode_index": int, "episode_substep": int,
               "episode_agent_step": int, "session_agent_step": int,
               "engine_time_ms": (int, float), "captured_utc": str},
    "provenance": {"brain": dict, "game": dict, "reward_mode": str},
    "reproduction_inputs": {"replayable": bool},
}


def validate_incident(record: Mapping[str, Any]) -> list[str]:
    """Every way `record` falls short of the schema; empty when valid."""
    errors: list[str] = []
    for key, typ in _REQUIRED.items():
        if key not in record:
            errors.append(f"missing {key}")
        elif not isinstance(record[key], typ) or (typ is int and isinstance(record[key], bool)):
            errors.append(f"{key} should be {getattr(typ, '__name__', typ)}")
    if errors:
        return errors
    if record["schema"] != INCIDENT_SCHEMA:
        errors.append(f"schema {record['schema']!r} is not {INCIDENT_SCHEMA!r}")
    if not INCIDENT_ID_RE.fullmatch(record["incident_id"]):
        errors.append(f"incident_id {record['incident_id']!r} is malformed")
    for section, fields in _NESTED.items():
        for key, typ in fields.items():
            value = record[section].get(key)
            if value is None or not isinstance(value, typ) or (
                    typ is int and isinstance(value, bool)):
                errors.append(f"{section}.{key} missing or not {getattr(typ, '__name__', typ)}")
    severity = record["classification"].get("severity")
    if severity not in SEVERITIES:
        errors.append(f"classification.severity {severity!r} is not one of {SEVERITIES}")
    level = record["classification"].get("detector_confidence", {}).get("level")
    if level not in CONFIDENCE_LEVELS:
        errors.append(f"detector_confidence.level {level!r} is not one of {CONFIDENCE_LEVELS}")
    if record["synthetic"] and level != "not_applicable":
        errors.append("a synthetic incident cannot carry a detector confidence")
    for name, meta in record["evidence"].items():
        if not isinstance(meta, dict) or not ARTIFACT_NAME_RE.fullmatch(str(meta.get("file", ""))):
            errors.append(f"evidence.{name} does not name a bundle file")
    return errors
