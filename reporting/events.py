"""The boundary between the detector and the incident pipeline.

A `Detection` is what the engine hands over, at the exact substep its
detector fired: the verdict, the numbers that tripped it, and an immutable
snapshot of everything needed to understand and replay the moment - the
frame on screen, the recent per-frame trace, the geometry in view, and the
episode's whole action log.

It is built INSIDE CustomMarioEnv.step(), before the substep returns,
because nothing later can rebuild it: the wrapper skips 4 engine frames per
agent decision and keeps only the last one's info, and the dashboard renders
its frame only after all 4. A snapshot taken there would be up to 3 frames
late. Everything here is copied, never referenced, so later frames cannot
reach back and change it.

This module stays dependency-light on purpose: the engine imports it.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detector verdict and the evidence frozen at the moment it fired."""

    kind: str                                   # 'below_world', 'speed', ... or 'synthetic_probe'
    message: str                                # the detector's own sentence (as shown in the log)
    detector: str                               # which check: 'engine_invariants/below_world', ...
    synthetic: bool                             # True: a pipeline-test event, never a game bug
    metrics: Mapping[str, Any]                  # the measured values that tripped it, and the limit
    state: Mapping[str, Any]                    # the env's info dict at the trigger substep
    mario: Mapping[str, Any]                    # engine fields info leaves out (y_vel, state, ...)
    episode_index: int                          # resets of this env so far (1 = first episode)
    episode_substep: int                        # 0-based engine frame within the episode
    engine_time_ms: float                       # the engine clock (deterministic, not wall time)
    frame: np.ndarray | None                    # the exact trigger frame, RGB, full resolution
    trace: tuple[Mapping[str, Any], ...]        # recent per-frame states, ending AT the trigger
    geometry: tuple[Mapping[str, Any], ...]     # colliders and sprites visible in the frame
    actions: bytes                              # every substep's action this episode, incl. trigger
    clock_holds: tuple[tuple[int, int], ...]    # (substep, units) where the QA clock was topped up
    actions_complete: bool                      # the log starts at the episode's reset
    game_variant: str
    episode_time_units: int | None              # engine config the replay must reproduce
    end_on_level_complete: bool
    holds_engine_clock: bool
    extra_detectors: tuple[Mapping[str, Any], ...] = field(default=())


class ExtraDetector(Protocol):
    """A detector added on top of the engine's built-in checks. The only one
    today is SyntheticProbe; `spec()` must be enough to rebuild it for a
    replay (reporting/reproduce.py), so it holds plain data only."""

    detector_id: str
    kind: str
    synthetic: bool

    def reset(self) -> None: ...
    def check(self, info: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None: ...
    def spec(self) -> dict[str, Any]: ...


class SyntheticProbe:
    """A deliberately FAKE anomaly, for exercising the reporting pipeline.

    Fires once per episode, the first frame Mario's collider reaches world
    x >= `x`. That is ordinary, correct gameplay - which is the point: it
    lets the whole pipeline (capture, pause, reports, dashboard, replay) be
    proven end to end without touching the game. Every incident it produces
    is marked synthetic and every report says, prominently, that it is a
    pipeline test and not a discovered bug.

    Deterministic from engine state alone, so a replay of the episode fires
    it again on the same frame - which is what lets reproduction itself be
    tested honestly.
    """

    kind = "synthetic_probe"
    synthetic = True

    def __init__(self, x: int) -> None:
        self.x = int(x)
        self.detector_id = f"synthetic_probe/x>={self.x}"

    def reset(self) -> None:
        """Nothing to clear: the engine's per-episode report-once rule
        already stops a second firing within an episode."""

    def check(self, info: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
        rect = info.get("mario_rect")
        if not rect or int(rect[0]) + int(rect[2]) < self.x:
            return None
        return ((f"[SYNTHETIC TEST EVENT - not a game bug] Mario's collider reached "
                 f"world x >= {self.x} (right edge {int(rect[0]) + int(rect[2])})."),
                {"collider_right_x": int(rect[0]) + int(rect[2]), "probe_x": self.x})

    def spec(self) -> dict[str, Any]:
        return {"type": "synthetic_probe", "x": self.x}


def detector_from_spec(spec: Mapping[str, Any]) -> ExtraDetector:
    """Rebuilds an extra detector from its spec. A closed list on purpose: a
    replay must never execute anything an incident file merely names."""
    if spec.get("type") == "synthetic_probe" and isinstance(spec.get("x"), int):
        return SyntheticProbe(int(spec["x"]))
    raise ValueError(f"unknown detector spec {dict(spec)!r}")


def json_safe(value: Any) -> Any:
    """Plain JSON types, recursively: tuples -> lists, numpy scalars ->
    Python numbers, non-finite floats -> None (JSON has no NaN), anything
    else -> its repr, so a snapshot can always be written."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, Sequence)) and not isinstance(value, (bytes, bytearray)):
        return [json_safe(v) for v in value]
    return repr(value)
