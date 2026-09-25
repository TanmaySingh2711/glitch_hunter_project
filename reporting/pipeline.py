"""The incident pipeline: capture first, everything else after.

    capture()  - on the GAME THREAD, synchronously, before the next frame:
                 classify, de-duplicate, then write the raw evidence (trigger
                 frame, trajectory, context frames) and the canonical record.
                 When capture() returns "new", the incident is on disk.
    worker     - on its own thread, afterwards: GIF, replay, Markdown, PDF.
                 Each is retried a bounded number of times and its outcome
                 recorded in the bundle's manifest; a failure there can delay
                 or lose a REPORT, never the incident.

Nothing the worker does can raise into the game loop or create an incident:
incidents come only from engine detectors, so a broken PDF library cannot
turn into a "bug found".
"""
from __future__ import annotations

import collections
import datetime
import json
import logging
import os
import queue
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from exploration import config
from reporting import analysis, render, schema
from reporting.events import Detection, json_safe
from reporting.evidence import (
    build_record,
    context_zip,
    encode_png,
    pixel_sha256,
    trajectory_doc,
)
from reporting.reproduce import run_reproduction
from reporting.store import IncidentStore, StoreError

log = logging.getLogger(__name__)

Notify = Callable[[str, dict[str, Any]], Any]
PUBLIC_ARTIFACTS = ("trigger.png", "context.gif", "report.md", "report.pdf",
                    "reproduction.json", "incident.json", "trajectory.json",
                    "context_frames.zip", "manifest.json")


@dataclass(frozen=True)
class CaptureOutcome:
    status: str                         # 'new' | 'duplicate' | 'failed'
    incident_id: str | None
    summary: dict[str, Any] | None
    error: str | None = None
    # True the first time this incident is seen in the current session (a
    # new incident always; a known one after a Reset) - the dashboard stops.
    first_in_session: bool = False


# ═══════════════════════════════════════════════════════════════════════
# THE CALLER'S HALF OF THE EVIDENCE
# ═══════════════════════════════════════════════════════════════════════
class SessionRecorder:
    """What the caller of the env keeps for the evidence: session and step
    counters, the agent's recent actions, and the stream frames before a
    trigger. Everything is per EPISODE: context never spans a reset."""

    def __init__(self, session_id: str | None = None,
                 context_steps: int = config.EVIDENCE_CONTEXT_AGENT_STEPS) -> None:
        self.session_id = session_id or f"S-{schema.utc_now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        self.session_step = 0
        self.episode_step = 0
        self.env_episode_index = 0
        self._frames: collections.deque[schema.ContextFrame] = collections.deque(maxlen=context_steps)
        self._actions: collections.deque[int] = collections.deque(maxlen=context_steps)

    def begin_episode(self, env_episode_index: int) -> None:
        self.episode_step = 0
        self.env_episode_index = env_episode_index
        self._frames.clear()
        self._actions.clear()

    def step_taken(self, action: int) -> None:
        self.session_step += 1
        self.episode_step += 1
        self._actions.append(int(action))

    def context(self, reward_mode: str, provenance: Mapping[str, Any]) -> schema.CaptureContext:
        """The context of the step just taken. Its own stream frame is not
        in it yet: that frame was drawn after the trigger."""
        return schema.CaptureContext(
            session_id=self.session_id, session_agent_step=self.session_step,
            episode_agent_step=self.episode_step, env_episode_index=self.env_episode_index,
            agent_action=self._actions[-1] if self._actions else -1,
            recent_agent_actions=tuple(self._actions), context_frames=tuple(self._frames),
            reward_mode=reward_mode, provenance=provenance)

    def push_frame(self, jpeg: bytes) -> None:
        if jpeg:
            self._frames.append(schema.ContextFrame(self.session_step, self.episode_step, jpeg))

    def recent_frames(self) -> tuple[bytes, ...]:
        """The episode's last stream frames (JPEG), oldest first."""
        return tuple(f.jpeg for f in self._frames)


# ═══════════════════════════════════════════════════════════════════════
# THE PIPELINE
# ═══════════════════════════════════════════════════════════════════════
class IncidentPipeline:
    def __init__(self, store: IncidentStore, *, notify: Notify | None = None,
                 reproduce: bool = True, start_worker: bool = True, recover: bool = True,
                 clock: Callable[[], datetime.datetime] = schema.utc_now) -> None:
        self.store = store
        self.notify = notify
        self.reproduce = reproduce
        self._clock = clock
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._known: list[tuple[str, dict[str, Any]]] = []
        self._counts: collections.Counter[str] = collections.Counter()
        # The dashboard's view: incidents seen (new or again) since the last
        # begin_session() - the Bug Tracker shows these; the store keeps all.
        self._session: dict[str, int] = {}
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._busy = 0
        self._idle = threading.Condition(self._lock)
        # Renderers are attributes so a test can make one fail on purpose.
        self.render_gif = render.render_gif
        self.render_markdown = render.render_markdown
        self.render_pdf = render.render_pdf
        self.run_reproduction = run_reproduction
        self.recovered = self._load_existing(recover)
        self._worker: threading.Thread | None = None
        if start_worker:
            self._worker = threading.Thread(target=self._work, name="incident-reports",
                                            daemon=True)
            self._worker.start()

    # ── start-up ──────────────────────────────────────────────────────────
    def _load_existing(self, recover: bool) -> dict[str, list[str]]:
        """Reloads every bundle on disk: its fingerprint (so a new sighting of
        an old incident is recognised), its occurrence count, and - for a
        bundle an earlier run left unfinished - its place in the queue.
        recover=False only reads (tools/incidents.py list/show/verify)."""
        done = self.store.recover() if recover else {"moved_incomplete": [], "removed_temp": []}
        done["requeued"] = []
        done["manifest_rebuilt"] = []
        for incident_id in self.store.incident_ids():
            record = self.store.load_record(incident_id)
            self._records[incident_id] = record
            self._known.append((incident_id, record["fingerprint"]))
            bundle = self.store.bundle_dir(incident_id)
            if not recover:
                continue
            if not os.path.exists(os.path.join(bundle, "manifest.json")):
                self.store.write_manifest(bundle, self._initial_manifest(record, bundle))
                done["manifest_rebuilt"].append(incident_id)
            if not self.store.read_manifest(bundle).get("finalized"):
                self._enqueue(incident_id)
                done["requeued"].append(incident_id)
        for occ in self.store.occurrences():
            if occ.get("incident_id") in self._records:
                self._counts[occ["incident_id"]] += 1
        return done

    def _initial_manifest(self, record: Mapping[str, Any], bundle: str) -> dict[str, Any]:
        from common.fileio import sha256_of
        artifacts = {}
        for name in schema.RAW_ARTIFACTS:
            path = os.path.join(bundle, name)
            if os.path.isfile(path):
                artifacts[name] = {"file": name, "sha256": sha256_of(path),
                                   "bytes": os.path.getsize(path)}
        now = schema.iso_utc(self._clock())
        return {
            "schema": schema.MANIFEST_SCHEMA, "incident_id": record["incident_id"],
            "created_utc": record["created_utc"], "finalized": False,
            "artifacts": artifacts,
            "renders": {"context.gif": {"status": "pending", "attempts": 0},
                        "reproduction.json": {"status": "pending" if self.reproduce
                                              else "skipped", "attempts": 0},
                        "report.md": {"status": "pending", "attempts": 0},
                        "report.pdf": {"status": "pending", "attempts": 0}},
            "reproduction": {"status": "pending" if self.reproduce else "not_attempted"},
            "history": [{"utc": now, "event": "raw evidence captured"}],
        }

    # ── capture (game thread) ─────────────────────────────────────────────
    def capture(self, det: Detection, ctx: schema.CaptureContext) -> CaptureOutcome:
        created = self._clock()
        registered: str | None = None
        try:
            game_tree = str(ctx.provenance.get("game", {}).get("tree_sha256") or "unknown")
            classification = analysis.classify(det)
            fp = analysis.fingerprint(det, game_tree)
            with self._lock:
                dup = analysis.find_duplicate(fp, self._known)
                if dup is not None:
                    self.store.append_occurrence({
                        "schema": schema.OCCURRENCE_SCHEMA, "incident_id": dup,
                        "seen_utc": schema.iso_utc(created), "session_id": ctx.session_id,
                        "episode_index": det.episode_index, "episode_substep": det.episode_substep,
                        "session_agent_step": ctx.session_agent_step, "site": fp["site"],
                        "metrics": dict(det.metrics), "synthetic": det.synthetic})
                    self._counts[dup] += 1
                    first = dup not in self._session
                    self._session[dup] = self._session.get(dup, 0) + 1
                    return CaptureOutcome("duplicate", dup, self.summary(dup), first_in_session=first)
                incident_id, bundle = self.store.create_bundle(created)
                self._known.append((incident_id, fp))
                registered = incident_id

            evidence: dict[str, Any] = {}
            # Most important first: if the disk fills up half-way, the
            # trigger frame is what survives.
            if det.frame is not None:
                meta = self.store.write_artifact(bundle, "trigger.png", encode_png(det.frame))
                evidence["trigger_frame"] = {**meta, "width": int(det.frame.shape[1]),
                                             "height": int(det.frame.shape[0]),
                                             "pixel_sha256": pixel_sha256(det.frame),
                                             "episode_substep": det.episode_substep}
            meta = self.store.write_artifact(
                bundle, "trajectory.json", schema.dumps(trajectory_doc(det, ctx, incident_id)))
            evidence["trajectory"] = {**meta, "trace_frames": len(det.trace),
                                      "action_log_frames": len(det.actions)}
            if ctx.context_frames:
                meta = self.store.write_artifact(bundle, "context_frames.zip",
                                                 context_zip(ctx.context_frames))
                evidence["context_frames"] = {
                    **meta, "count": len(ctx.context_frames),
                    "first_episode_agent_step": ctx.context_frames[0].episode_agent_step,
                    "last_episode_agent_step": ctx.context_frames[-1].episode_agent_step}
            record = build_record(incident_id=incident_id, created=created, det=det, ctx=ctx,
                                  classification=classification, fingerprint=fp,
                                  evidence=evidence)
            errors = schema.validate_incident(record)
            if errors:
                raise ValueError(f"the incident record failed validation: {errors}")
            self.store.write_artifact(bundle, "incident.json", schema.dumps(record))
            with self._lock:
                self._records[incident_id] = record
                self.store.write_manifest(bundle, self._initial_manifest(record, bundle))
            self._enqueue(incident_id)
            with self._lock:
                self._session[incident_id] = self._session.get(incident_id, 0) + 1
            return CaptureOutcome("new", incident_id, self.summary(incident_id),
                                  first_in_session=True)
        except Exception as exc:
            log.exception("incident capture failed")
            if registered is not None:
                with self._lock:
                    self._known = [k for k in self._known if k[0] != registered]
            error = f"{type(exc).__name__}: {exc}"
            self._record_capture_failure(det, ctx, created, registered, error)
            return CaptureOutcome("failed", registered, None, error)

    def _record_capture_failure(self, det: Detection, ctx: schema.CaptureContext,
                                created: datetime.datetime, partial: str | None,
                                error: str) -> None:
        """Keeps the detector's own facts when the bundle could not be
        written. A full disk may defeat this too; that is then logged."""
        try:
            self.store.append_capture_failure(json_safe({
                "schema": "glitch-hunter.capture-failure/1", "failed_utc": schema.iso_utc(created),
                "error": error, "partial_bundle": partial, "kind": det.kind,
                "detector": det.detector, "message": det.message, "synthetic": det.synthetic,
                "metrics": det.metrics, "state": det.state, "episode_index": det.episode_index,
                "episode_substep": det.episode_substep, "session_id": ctx.session_id,
                "session_agent_step": ctx.session_agent_step, "game_variant": det.game_variant}))
        except Exception:
            log.exception("could not even record the failed capture")

    # ── the worker ────────────────────────────────────────────────────────
    def _enqueue(self, incident_id: str) -> None:
        with self._lock:
            self._busy += 1
        self._queue.put(incident_id)

    def _work(self) -> None:
        while True:
            incident_id = self._queue.get()
            if incident_id is None:
                return
            try:
                self.process(incident_id)
            except Exception:
                log.exception("rendering %s failed outside any step", incident_id)
            finally:
                with self._lock:
                    self._busy -= 1
                    self._idle.notify_all()

    def process(self, incident_id: str) -> None:
        """Renders whatever a bundle is still missing, then finalizes it.
        Safe to call again: finished steps are skipped."""
        bundle = self.store.bundle_dir(incident_id)
        record = self._records.get(incident_id) or self.store.load_record(incident_id)
        steps: list[tuple[str, Callable[[str, Mapping[str, Any], dict[str, Any]], bytes]]] = [
            ("context.gif", self._make_gif), ("reproduction.json", self._make_reproduction),
            ("report.md", self._make_markdown), ("report.pdf", self._make_pdf)]
        for name, make in steps:
            with self._lock:
                manifest = self.store.read_manifest(bundle)
            state = manifest["renders"].setdefault(name, {"status": "pending", "attempts": 0})
            if state["status"] in ("done", "skipped"):
                continue
            if state["status"] == "failed" and state["attempts"] >= config.RENDER_MAX_ATTEMPTS:
                continue
            self._run_step(bundle, record, name, make)
        with self._lock:
            manifest = self.store.read_manifest(bundle)
            manifest["finalized"] = True
            manifest["history"].append({"utc": schema.iso_utc(self._clock()),
                                        "event": "finalized: bundle is now read-only"})
            self.store.write_manifest(bundle, manifest)
        self._emit("incident_updated", self.summary(incident_id))

    def _run_step(self, bundle: str, record: Mapping[str, Any], name: str,
                  make: Callable[[str, Mapping[str, Any], dict[str, Any]], bytes]) -> None:
        for attempt in range(1, config.RENDER_MAX_ATTEMPTS + 1):
            with self._lock:
                manifest = self.store.read_manifest(bundle)
            state = manifest["renders"][name]
            try:
                path = os.path.join(bundle, name)
                if os.path.exists(path) and name not in manifest["artifacts"]:
                    # Written completely (atomic rename) by a run that died
                    # before recording it: adopt it rather than render twice.
                    from common.fileio import sha256_of
                    meta = {"file": name, "sha256": sha256_of(path),
                            "bytes": os.path.getsize(path)}
                    event = f"{name}: adopted, written by an interrupted earlier run"
                else:
                    data = make(bundle, record, manifest)
                    meta = self.store.write_artifact(bundle, name, data)
                    event = f"{name}: rendered"
                with self._lock:
                    manifest = self.store.read_manifest(bundle)
                    manifest["artifacts"][name] = meta
                    manifest["renders"][name] = {"status": "done", "attempts": attempt}
                    manifest["history"].append({"utc": schema.iso_utc(self._clock()),
                                                "event": event})
                    self.store.write_manifest(bundle, manifest)
                return
            except _Skip as skip:
                with self._lock:
                    manifest = self.store.read_manifest(bundle)
                    manifest["renders"][name] = {"status": "skipped", "attempts": attempt,
                                                 "reason": str(skip)}
                    self.store.write_manifest(bundle, manifest)
                return
            except Exception as exc:
                log.warning("%s for %s failed (attempt %d/%d): %s", name, record["incident_id"],
                            attempt, config.RENDER_MAX_ATTEMPTS, exc)
                with self._lock:
                    manifest = self.store.read_manifest(bundle)
                    state = {"status": "failed", "attempts": attempt,
                             "last_error": f"{type(exc).__name__}: {exc}"}
                    manifest["renders"][name] = state
                    manifest["history"].append({"utc": schema.iso_utc(self._clock()),
                                                "event": f"{name}: attempt {attempt} failed: "
                                                         f"{type(exc).__name__}: {exc}"})
                    self.store.write_manifest(bundle, manifest)
                if attempt < config.RENDER_MAX_ATTEMPTS:
                    time.sleep(config.RENDER_RETRY_DELAYS_S[min(attempt - 1,
                                                                 len(config.RENDER_RETRY_DELAYS_S) - 1)])

    # ── the four derived artifacts ───────────────────────────────────────
    @staticmethod
    def _read(bundle: str, name: str) -> bytes | None:
        path = os.path.join(bundle, name)
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as fh:
            return fh.read()

    def _json(self, bundle: str, name: str) -> dict[str, Any] | None:
        data = self._read(bundle, name)
        return None if data is None else json.loads(data)

    def _make_gif(self, bundle: str, record: Mapping[str, Any], _m: dict[str, Any]) -> bytes:
        ctx_zip, trig = self._read(bundle, "context_frames.zip"), self._read(bundle, "trigger.png")
        if ctx_zip is None and trig is None:
            raise _Skip("no frames were captured (no game window at the trigger)")
        return self.render_gif(record, ctx_zip, trig)

    def _make_reproduction(self, bundle: str, record: Mapping[str, Any],
                           _m: dict[str, Any]) -> bytes:
        if not self.reproduce:
            raise _Skip("reproduction is switched off for this pipeline")
        result = self.run_reproduction(bundle)
        with self._lock:
            manifest = self.store.read_manifest(bundle)
            manifest["reproduction"] = {"status": result.get("status", "error"),
                                        "detail": result.get("detail")}
            self.store.write_manifest(bundle, manifest)
        return schema.dumps(result)

    def _render_inputs(self, bundle: str, record: Mapping[str, Any]) -> tuple[
            dict[str, Any], int, dict[str, Any] | None, dict[str, Any] | None]:
        with self._lock:
            manifest = self.store.read_manifest(bundle)
            occurrences = 1 + self._counts[record["incident_id"]]
        return (manifest, occurrences, self._json(bundle, "trajectory.json"),
                self._json(bundle, "reproduction.json"))

    def _make_markdown(self, bundle: str, record: Mapping[str, Any],
                       _m: dict[str, Any]) -> bytes:
        manifest, occ, traj, rep = self._render_inputs(bundle, record)
        return self.render_markdown(record, manifest, occ, traj, rep,
                                    self._clock()).encode("utf-8")

    def _make_pdf(self, bundle: str, record: Mapping[str, Any], _m: dict[str, Any]) -> bytes:
        manifest, occ, traj, rep = self._render_inputs(bundle, record)
        return self.render_pdf(record, manifest, occ, traj, rep,
                               self._read(bundle, "trigger.png"),
                               self._read(bundle, "context_frames.zip"), self._clock())

    # ── queries (any thread) ──────────────────────────────────────────────
    def incident_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._records)

    def summary(self, incident_id: str) -> dict[str, Any]:
        """What the dashboard shows for one incident."""
        with self._lock:
            record = self._records.get(incident_id) or self.store.load_record(incident_id)
            bundle = self.store.bundle_dir(incident_id)
            try:
                manifest = self.store.read_manifest(bundle)
            except (OSError, ValueError):
                manifest = {"artifacts": {}, "renders": {}, "reproduction": {}}
            occurrences = 1 + self._counts[incident_id]
        arts = manifest.get("artifacts", {})
        latest = {}
        for base in ("context.gif", "report.md", "report.pdf", "reproduction.json"):
            stem, ext = os.path.splitext(base)
            versions = [n for n in arts if n == base or (n.startswith(stem + ".v") and n.endswith(ext))]
            if versions:
                latest[base] = max(versions, key=lambda n: (len(n), n))
        loc = record["location"].get("world_collider") or {}
        return {
            "incident_id": incident_id,
            "created_utc": record["created_utc"],
            "category": record["summary"]["category"],
            "title": record["summary"]["title"],
            "description": record["summary"]["description"],
            "synthetic": record["synthetic"],
            "severity": record["classification"]["severity"],
            "confidence": record["classification"]["detector_confidence"]["level"],
            "reproduction": manifest.get("reproduction", {}).get("status", "not_attempted"),
            "occurrences": occurrences,
            "location": {"x": loc.get("x"), "y": loc.get("y")},
            "episode_index": record["timing"]["episode_index"],
            "episode_agent_step": record["timing"]["episode_agent_step"],
            "session_agent_step": record["timing"]["session_agent_step"],
            "game_variant": record["provenance"].get("game", {}).get("variant"),
            "brain_approved": bool(record["provenance"].get("brain", {})
                                   .get("approved_objective2_brain")),
            "available": sorted(n for n in arts if n in PUBLIC_ARTIFACTS or n in latest.values()),
            "latest": latest,
            "renders": {k: v.get("status") for k, v in manifest.get("renders", {}).items()},
            "finalized": bool(manifest.get("finalized")),
        }

    def summaries(self) -> list[dict[str, Any]]:
        """Every incident, newest first. A bundle removed by hand while the
        dashboard runs is skipped (and logged), not allowed to break the list."""
        out = []
        for incident_id in reversed(self.incident_ids()):
            try:
                out.append(self.summary(incident_id))
            except (StoreError, OSError, ValueError, KeyError):
                log.warning("incident %s is no longer readable; left out of the list",
                            incident_id)
        return out

    def begin_session(self) -> None:
        """Starts a new dashboard session view (the dashboard's Reset). Nothing
        is deleted: every incident stays in the store and in summaries()."""
        with self._lock:
            self._session.clear()

    def session_summaries(self) -> list[dict[str, Any]]:
        """The incidents seen in this session, most recently first seen first,
        each with `seen_this_session`."""
        with self._lock:
            seen = list(self._session.items())
        out = []
        for incident_id, count in reversed(seen):
            try:
                out.append({**self.summary(incident_id), "seen_this_session": count})
            except (StoreError, OSError, ValueError, KeyError):
                log.warning("incident %s is no longer readable; left out of the list",
                            incident_id)
        return out

    def occurrences(self, incident_id: str) -> list[dict[str, Any]]:
        return [o for o in self.store.occurrences() if o.get("incident_id") == incident_id]

    def detail(self, incident_id: str) -> dict[str, Any]:
        summary = self.summary(incident_id)
        bundle = self.store.bundle_dir(incident_id)
        with self._lock:
            manifest = self.store.read_manifest(bundle)
        record = self._records.get(incident_id) or self.store.load_record(incident_id)
        return {"summary": summary, "record": record,
                "manifest": manifest, "reproduction": self._json(bundle, "reproduction.json"),
                "occurrences": self.occurrences(incident_id)}

    # ── lifecycle ─────────────────────────────────────────────────────────
    def wait_idle(self, timeout: float = 60.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._lock:
            while self._busy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(remaining)
        return True

    def close(self) -> None:
        if self._worker is not None:
            self._queue.put(None)
            self._worker.join(timeout=10)
            self._worker = None

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self.notify is None:
            return
        try:
            self.notify(event, payload)
        except Exception:                 # a closed browser must not break reporting
            log.debug("notify %s failed", event, exc_info=True)


class _Skip(Exception):
    """A derived artifact that cannot exist for this incident (not an error)."""


__all__ = ["CaptureOutcome", "IncidentPipeline", "SessionRecorder", "StoreError"]
