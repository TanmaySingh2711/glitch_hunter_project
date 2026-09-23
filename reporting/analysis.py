"""Severity, detector confidence and the duplicate fingerprint.

Three separate questions, kept separate on purpose:

  SEVERITY             if this anomaly is real, how much does it hurt play?
                       Fixed per kind, with the reason written down. Nothing
                       is "critical" today: that is reserved for a crash or a
                       corrupted game, which no current detector reports.
  DETECTOR CONFIDENCE  is the signal genuine engine state, not an artifact of
                       how it was read? Earned from evidence only: how far a
                       reading went past its threshold, whether the state was
                       read completely, whether the engine was mid-transition.
                       A detector firing is NOT, by itself, grounds for "high".
  REPRODUCTION         did a replay of the episode show it again? Measured
                       later, by reporting/reproduce.py, and reported
                       alongside - never folded into either number above.

Synthetic events get severity "none" and confidence "not_applicable": there
is no real anomaly to rank.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from custom_mario_env import ABOVE_WORLD_Y, MAX_PLAUSIBLE_X_VEL
from exploration import config
from reporting.events import Detection

# ─── SEVERITY: impact if real, per kind ───
SEVERITY: dict[str, tuple[str, str]] = {
    "below_world": ("high", ("Mario is alive below the death plane: the level's own death rule "
                             "failed, so the run cannot continue normally (an endless fall or a "
                             "soft-lock).")),
    "above_world": ("medium", ("Mario is far above the level's ceiling: he can pass over level "
                               "geometry and leave the bounds the level was designed for.")),
    "speed": ("medium", ("Horizontal speed far beyond the engine's own sprint cap: at this speed a "
                         "collider can pass through walls and enemies between two frames.")),
    "score_drop": ("low", ("The score ran backwards: bookkeeping is wrong, but play itself is "
                           "not blocked.")),
    "coin_drop": ("low", ("The coin total ran backwards: bookkeeping is wrong, but play itself "
                          "is not blocked.")),
    "synthetic_probe": ("none", "Synthetic pipeline-test event: no game behaviour is involved."),
}

# How each built-in detector decides. 'invariant': a rule the engine itself
# maintains was observed broken (compared exactly, before and after).
# 'threshold': a measured value crossed a limit set ~2x outside normal play.
_THRESHOLDS: dict[str, tuple[str, float, float]] = {
    # kind: (metric, threshold, edge of the measured normal-play envelope)
    "speed": ("abs_x_vel", MAX_PLAUSIBLE_X_VEL, config.NORMAL_PLAY_MAX_ABS_X_VEL),
    "above_world": ("height_above", -ABOVE_WORLD_Y, -config.NORMAL_PLAY_MIN_Y),
}
_INVARIANTS = frozenset(("below_world", "score_drop", "coin_drop"))

# The first two engine frames after a reset still settle the level (the
# camera and Mario are placed on them); readings there are the least
# trustworthy of an episode.
_SETTLING_SUBSTEPS = 2


def severity(kind: str) -> tuple[str, str]:
    return SEVERITY.get(kind, ("unclassified", f"No severity is defined for kind {kind!r}."))


def _reading(det: Detection) -> float | None:
    if det.kind == "speed":
        v = det.metrics.get("x_vel")
        return abs(float(v)) if isinstance(v, (int, float)) else None
    if det.kind == "above_world":
        y = det.metrics.get("y")
        return -float(y) if isinstance(y, (int, float)) else None
    return None


def detector_confidence(det: Detection) -> dict[str, Any]:
    """How much the evidence supports the signal being real engine state."""
    if det.synthetic:
        return {"level": "not_applicable", "basis": "synthetic", "margin": None,
                "reasons": ["Synthetic test event: there is no real anomaly to be confident about."],
                "downgrades": []}
    reasons: list[str] = []
    margin: float | None = None
    if det.kind in _INVARIANTS:
        basis, level = "invariant", 2
        reasons.append("An engine invariant was observed broken, comparing exact values "
                       f"(recorded: {dict(det.metrics)}).")
    elif det.kind in _THRESHOLDS:
        basis = "threshold"
        metric, limit, normal = _THRESHOLDS[det.kind]
        value = _reading(det)
        if value is None:
            level = 0
            reasons.append(f"The {metric} reading is missing.")
        else:
            margin = round((value - limit) / (limit - normal), 3)
            level = 2 if margin >= 1.0 else 1
            reasons.append(f"{metric} {value:g} vs threshold {limit:g}; normal play reaches "
                           f"{normal:g}. Margin past the threshold = {margin:g} x the gap between "
                           f"normal play and the threshold ("
                           + ("at least one full gap: high" if level == 2 else
                              "less than one gap: medium") + ").")
    else:
        basis, level = "unknown", 0
        reasons.append(f"No confidence rule exists for kind {det.kind!r}.")

    downgrades: list[str] = []
    if not det.state.get("mario_rect"):
        # Not a weaker measurement - no measurement: the engine fallback fills
        # the state with placeholders. Straight to low, whatever else holds.
        downgrades.append("Mario's state could not be read on this frame (engine fallback): "
                          "the values are placeholders, not a measurement.")
        level = 0
    if det.mario.get("in_castle") or det.state.get("flag_get"):
        downgrades.append("The end-of-level sequence was running; the engine suspends normal "
                          "rules during it.")
    if det.state.get("is_dead"):
        downgrades.append("Mario was dead on this frame; the level is being torn down.")
    if det.episode_substep < _SETTLING_SUBSTEPS:
        downgrades.append(f"Frame {det.episode_substep} of the episode: the level is still "
                          f"settling after the reset.")
    level = max(0, level - len(downgrades))
    return {"level": ("low", "medium", "high")[level], "basis": basis, "margin": margin,
            "reasons": reasons, "downgrades": downgrades}


def classify(det: Detection) -> dict[str, Any]:
    sev, why = severity(det.kind)
    return {"severity": sev, "severity_rationale": why,
            "detector_confidence": detector_confidence(det)}


# ─── DUPLICATES ───
def collider_center(det: Detection) -> tuple[float, float] | None:
    rect = det.state.get("mario_rect")
    if not rect or len(rect) != 4:
        return None
    x, y, w, h = (float(v) for v in rect)
    return (x + w / 2.0, y + h / 2.0)


def fingerprint(det: Detection, game_tree_sha256: str) -> dict[str, Any]:
    """The identity of an anomaly for de-duplication.

    `signature` = the category identity: game content, detector, kind, and
    synthetic-or-not. Two sightings can only be the same incident if their
    signatures are equal AND their sites are within DEDUP_RADIUS of each other
    (see config: they would overlap one frame apart). The signature alone is
    never enough - two real bugs can share a kind.
    """
    components = {"game_tree_sha256": game_tree_sha256, "detector": det.detector,
                  "kind": det.kind, "synthetic": det.synthetic}
    signature = hashlib.sha256("|".join(f"{k}={components[k]}" for k in sorted(components))
                               .encode("utf-8")).hexdigest()[:16]
    center = collider_center(det)
    return {"version": 1, "signature": signature, "kind": det.kind,
            "detector": det.detector, "synthetic": det.synthetic,
            "game_tree_sha256": game_tree_sha256,
            "site": None if center is None else {"x": round(center[0], 1),
                                                 "y": round(center[1], 1)},
            "radius": {"x": config.DEDUP_RADIUS_X, "y": config.DEDUP_RADIUS_Y}}


def same_incident(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """True when two fingerprints describe one incident (see fingerprint)."""
    if a.get("signature") != b.get("signature"):
        return False
    sa, sb = a.get("site"), b.get("site")
    if not sa or not sb:
        # Without a position there is no evidence two sightings are the same
        # thing, so they are not merged.
        return False
    return (abs(float(sa["x"]) - float(sb["x"])) <= config.DEDUP_RADIUS_X
            and abs(float(sa["y"]) - float(sb["y"])) <= config.DEDUP_RADIUS_Y)


def find_duplicate(fp: Mapping[str, Any],
                   known: Sequence[tuple[str, Mapping[str, Any]]]) -> str | None:
    """The id of the earliest known incident `fp` duplicates, if any."""
    for incident_id, other in known:
        if same_incident(fp, other):
            return incident_id
    return None
