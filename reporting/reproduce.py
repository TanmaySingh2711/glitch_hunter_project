"""Replay an incident's episode and report, honestly, whether it happened again.

WHY A REPLAY CAN WORK HERE. The engine is deterministic: game time is the
env's own frame counter, never the wall clock, and nothing in the game draws
random numbers. So an episode is a pure function of what the engine was fed
on each frame - the action, and (in QA mode) whether the clock was topped up.
The incident records exactly that, from the episode's reset to the trigger.

WHAT IS COMPARED. Every frame of the replay against the recorded per-frame
trace (collider, velocities, Mario's state, score, coins, clock, death), the
clock top-ups frame by frame, whether the SAME detector fires on the SAME
frame, and the trigger frame's pixels.

    reproduced             same state, same frame, same detector, identical pixels
    reproduced_state_only  as above, but the pixels differ
    not_reproduced         same state on every compared frame; the detector stayed quiet
    diverged               the state separated from the recording before the trigger
    not_possible           nothing honest to replay (no log from reset, game changed, ...)
    timeout / error        the replay process did not finish

ISOLATION. The replay runs in its own headless process: the live dashboard
owns the game window and this process's one game variant, and a replay must
never touch either. It only READS the bundle; the parent writes the result.

    python -m reporting.reproduce <bundle dir> <output json>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from typing import Any

from exploration import config
from reporting import schema

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The per-frame fields a replay must match exactly to count as the same state.
COMPARED_FIELDS = ("action", "x", "y", "w", "h", "x_vel", "y_vel", "mario_state", "on_ground",
                   "is_dead", "score", "coins", "time_left", "viewport_x")


def _load(bundle: str, name: str) -> dict[str, Any]:
    with open(os.path.join(bundle, name), encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def _result(status: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"status": status, "detail": detail, **extra}


def _first_difference(recorded: Mapping[str, Any], replayed: Mapping[str, Any]) -> dict[str, Any] | None:
    for key in COMPARED_FIELDS:
        a, b = recorded.get(key), replayed.get(key)
        if a != b:
            return {"field": key, "recorded": a, "replayed": b}
    return None


def replay(bundle: str) -> dict[str, Any]:
    """Runs IN the replay process. Returns the result document."""
    from custom_mario_env import CustomMarioEnv
    from reporting.events import detector_from_spec
    from reporting.evidence import decode_actions, pixel_sha256
    from reporting.variants import game_dir, game_tree_sha256

    record = _load(bundle, "incident.json")
    trajectory = _load(bundle, "trajectory.json")
    inputs = record["reproduction_inputs"]
    if not inputs.get("replayable"):
        return _result("not_possible", str(inputs.get("why_not") or "marked not replayable"))
    try:
        actions = decode_actions(trajectory)
    except (KeyError, ValueError) as exc:
        return _result("not_possible", f"the action log is unusable: {exc}")
    trigger = int(record["timing"]["episode_substep"])
    if len(actions) != trigger + 1:
        return _result("not_possible", f"the action log holds {len(actions)} frames but the "
                                       f"trigger is frame {trigger}")
    variant = inputs["game_variant"]
    recorded_tree = record["provenance"]["game"]["tree_sha256"]
    if game_tree_sha256(game_dir(variant)) != recorded_tree:
        return _result("not_possible", f"the {variant} game has changed since the incident "
                                       f"was captured, so a replay would test different code")
    try:
        detectors = [detector_from_spec(s) for s in inputs.get("extra_detectors", [])]
    except ValueError as exc:
        return _result("not_possible", str(exc))

    env = CustomMarioEnv(game_variant=variant)
    env.episode_time_units = inputs.get("episode_time_units")
    env.end_on_level_complete = bool(inputs.get("end_on_level_complete"))
    env.holds_engine_clock = bool(inputs.get("holds_engine_clock"))
    env.enable_evidence()
    for d in detectors:
        env.add_detector(d)
    env.reset()

    holds = {int(i): int(u) for i, u in trajectory.get("clock_holds", [])}
    recorded = {int(t["substep"]): t for t in trajectory.get("trace", [])}
    divergence: dict[str, Any] | None = None
    compared = 0
    for i in range(trigger + 1):
        if env.holds_engine_clock:
            units = env.hold_clock()
            if units != holds.get(i, 0) and divergence is None:
                divergence = {"substep": i, "field": "clock_top_up",
                              "recorded": holds.get(i, 0), "replayed": units}
        _obs, _r, done, _t, _info = env.step(actions[i])
        if i in recorded and divergence is None:
            compared += 1
            diff = _first_difference(recorded[i], env.last_trace_entry() or {})
            if diff:
                divergence = {"substep": i, **diff}
        if done and i < trigger:
            return _result("diverged", f"the replayed episode ended at frame {i}, before the "
                                       f"trigger frame {trigger}",
                           first_divergence=divergence or {"substep": i, "field": "episode_end"},
                           replayed_substeps=i + 1, compared_frames=compared)

    fired = env.drain_detections()
    detector_id = record["detector"]["id"]
    match = next((d for d in fired if d.detector == detector_id and d.episode_substep == trigger),
                 None)
    recorded_pixels = record.get("evidence", {}).get("trigger_frame", {}).get("pixel_sha256")
    frame_match = (match is not None and match.frame is not None and recorded_pixels is not None
                   and pixel_sha256(match.frame) == recorded_pixels)
    common = {"replayed_substeps": trigger + 1, "compared_frames": compared,
              "first_divergence": divergence, "state_match": divergence is None,
              "frame_match": frame_match,
              "detector_fired_at": [{"detector": d.detector, "frame": d.episode_substep}
                                    for d in fired]}
    if divergence is not None:
        return _result("diverged", f"the replay separated from the recording at frame "
                                   f"{divergence['substep']} ({divergence['field']})", **common)
    if match is None:
        return _result("not_reproduced", f"every compared frame matched, but {detector_id} did "
                                         f"not fire on frame {trigger}", **common)
    if frame_match:
        return _result("reproduced", f"{detector_id} fired again on frame {trigger} with the same "
                                     f"state and identical pixels", **common)
    return _result("reproduced_state_only", f"{detector_id} fired again on frame {trigger} with "
                                            f"the same state; the pixels differ", **common)


def timeout_for(record: Mapping[str, Any]) -> float:
    frames = int(record["timing"]["episode_substep"]) + 1
    return config.REPRODUCE_STARTUP_TIMEOUT_S + frames * config.REPRODUCE_PER_SUBSTEP_TIMEOUT_S


def run_reproduction(bundle: str, timeout_s: float | None = None) -> dict[str, Any]:
    """Runs the replay in a fresh headless process; never raises."""
    started = time.perf_counter()
    base = {"schema": schema.REPRODUCTION_SCHEMA,
            "incident_id": os.path.basename(os.path.normpath(bundle)),
            "attempted_utc": schema.iso_utc(schema.utc_now()),
            "method": "engine replay of the recorded per-frame actions from reset, "
                      "in a separate headless process"}
    try:
        record = _load(bundle, "incident.json")
        limit = timeout_s if timeout_s is not None else timeout_for(record)
    except (OSError, ValueError, KeyError) as exc:
        return {**base, **_result("error", f"cannot read the bundle: {exc}")}
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", PYTHONUTF8="1",
               OPENBLAS_NUM_THREADS="1")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "result.json")
        try:
            proc = subprocess.run([sys.executable, "-m", "reporting.reproduce", bundle, out],  # noqa: S603
                                  cwd=PROJECT_ROOT, env=env, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=limit, check=False)
        except subprocess.TimeoutExpired:
            return {**base, **_result("timeout", f"no result within {limit:.0f} s"),
                    "elapsed_s": round(time.perf_counter() - started, 2)}
        except OSError as exc:
            return {**base, **_result("error", f"could not start the replay: {exc}")}
        elapsed = round(time.perf_counter() - started, 2)
        if proc.returncode != 0 or not os.path.exists(out):
            return {**base, **_result("error", f"replay exited {proc.returncode}: "
                                               f"{proc.stderr.strip()[-800:]}"),
                    "elapsed_s": elapsed}
        with open(out, encoding="utf-8") as fh:
            result = json.load(fh)
    if result.get("status") not in schema.REPRODUCTION_STATUSES:
        return {**base, **_result("error", f"unknown replay status {result.get('status')!r}"),
                "elapsed_s": elapsed}
    return {**base, **result, "elapsed_s": elapsed}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        sys.stderr.write(__doc__ or "")
        return 2
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    bundle, out = args
    try:
        result = replay(bundle)
    except Exception as exc:          # reported as a result, never as a new incident
        result = _result("error", f"{type(exc).__name__}: {exc}")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
