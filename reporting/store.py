"""Where incidents live on disk, and the rules that keep them trustworthy.

    incidents/
      INC-20260923-101530-4f2a9c/          one self-contained bundle per incident
        incident.json        immutable facts (reporting/schema.py)       read-only
        trigger.png          the exact trigger frame, lossless           read-only
        context_frames.zip   the stream frames before it (raw GIF input) read-only
        trajectory.json      per-frame trace + the episode's action log  read-only
        context.gif          derived: the moments before, then the trigger
        report.md            derived
        report.pdf           derived
        reproduction.json    derived: what a replay of the episode found
        manifest.json        status + SHA-256 of every file above; read-only once final
      occurrences.jsonl      append-only: later sightings of an existing incident
      capture_failures.jsonl append-only: detections whose bundle could not be written
      _incomplete/           bundles a crash interrupted before incident.json

RULES
* An id is claimed by creating its directory with os.mkdir, which fails if
  it exists - two captures, even in two processes, cannot share a bundle.
* Every file is written to a temp name, flushed and renamed into place, so a
  crash leaves the old state or the new one, never half a file.
* A written artifact is never replaced. Evidence and reports are made
  read-only as soon as they are written; a re-render gets a new name
  (report.v2.pdf), recorded in the manifest's history.
* incident.json is written LAST of the raw evidence. A bundle directory
  without it was interrupted mid-capture; recover() moves it to _incomplete/
  (kept, never deleted) so it can never be mistaken for a real incident.
* Paths are only ever built from a validated id and an allow-listed name,
  and must resolve inside the store - never from anything a browser sent.
"""
from __future__ import annotations

import datetime
import json
import os
import secrets
import stat
import threading
import time
from collections.abc import Iterator, Mapping
from typing import Any

from common.fileio import make_read_only, sha256_of
from reporting import schema

MANIFEST = "manifest.json"
OCCURRENCES = "occurrences.jsonl"
CAPTURE_FAILURES = "capture_failures.jsonl"
INCOMPLETE_DIR = "_incomplete"


class StoreError(Exception):
    """A request the store refuses (bad id, bad name, overwrite)."""


def _make_writable(path: str) -> None:
    os.chmod(path, stat.S_IREAD | stat.S_IWRITE)


# On Windows a file cannot be renamed over while another handle has it open -
# the dashboard serving manifest.json to a browser, say, or an indexer or
# antivirus scanning a file that was just written. Those holds last
# milliseconds, so the rename is retried briefly instead of failing the write.
_REPLACE_ATTEMPTS = 6
_REPLACE_BACKOFF_S = 0.05


def _write_atomically(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == _REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(_REPLACE_BACKOFF_S * (attempt + 1))
    except BaseException:
        if os.path.exists(tmp):
            _make_writable(tmp)
            os.remove(tmp)
        raise


class IncidentStore:
    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self._lock = threading.RLock()

    # ── identity and paths ────────────────────────────────────────────────
    def create_bundle(self, moment: datetime.datetime) -> tuple[str, str]:
        """Claims a fresh incident id (exclusive mkdir) and returns (id, dir)."""
        os.makedirs(self.root, exist_ok=True)
        for _ in range(32):
            incident_id = schema.new_incident_id(moment, secrets.token_hex(3))
            path = os.path.join(self.root, incident_id)
            try:
                os.mkdir(path)
            except FileExistsError:
                continue
            return incident_id, path
        raise StoreError("could not allocate a unique incident id")

    def bundle_dir(self, incident_id: str) -> str:
        if not isinstance(incident_id, str) or not schema.INCIDENT_ID_RE.fullmatch(incident_id):
            raise StoreError(f"not an incident id: {incident_id!r}")
        path = os.path.join(self.root, incident_id)
        if not os.path.isfile(os.path.join(path, "incident.json")):
            raise StoreError(f"no such incident: {incident_id}")
        return path

    def artifact_path(self, incident_id: str, name: str) -> str:
        """The path of an EXISTING artifact, or StoreError. The only way the
        dashboard turns a request into a file."""
        if not isinstance(name, str) or not schema.ARTIFACT_NAME_RE.fullmatch(name):
            raise StoreError(f"not a bundle file name: {name!r}")
        bundle = os.path.realpath(self.bundle_dir(incident_id))
        path = os.path.realpath(os.path.join(bundle, name))
        if os.path.commonpath([bundle, path]) != bundle or not os.path.isfile(path):
            raise StoreError(f"{incident_id} has no {name}")
        return path

    # ── writing ───────────────────────────────────────────────────────────
    def write_artifact(self, bundle: str, name: str, data: bytes,
                       read_only: bool = True) -> dict[str, Any]:
        """Writes a NEW file into a bundle, atomically; never replaces one."""
        if not schema.ARTIFACT_NAME_RE.fullmatch(name) or name == MANIFEST:
            raise StoreError(f"not an artifact name: {name!r}")
        path = os.path.join(bundle, name)
        with self._lock:
            if os.path.exists(path):
                raise StoreError(f"{name} already exists in {os.path.basename(bundle)}; "
                                 f"evidence is never overwritten")
            _write_atomically(path, data)
        if read_only:
            make_read_only(path)
        return {"file": name, "sha256": sha256_of(path), "bytes": len(data)}

    def next_version_name(self, bundle: str, name: str) -> str:
        """report.pdf -> report.v2.pdf -> report.v3.pdf ... (first unused)."""
        if not os.path.exists(os.path.join(bundle, name)):
            return name
        stem, ext = os.path.splitext(name)
        n = 2
        while os.path.exists(os.path.join(bundle, f"{stem}.v{n}{ext}")):
            n += 1
        return f"{stem}.v{n}{ext}"

    def read_manifest(self, bundle: str) -> dict[str, Any]:
        with open(os.path.join(bundle, MANIFEST), encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
        return data

    def write_manifest(self, bundle: str, manifest: Mapping[str, Any]) -> None:
        """Replaces manifest.json atomically. A FINALIZED manifest is
        read-only; changing it (a re-render) must go through unlock_manifest,
        which leaves a trace in its history."""
        path = os.path.join(bundle, MANIFEST)
        with self._lock:
            if os.path.exists(path) and not os.access(path, os.W_OK):
                raise StoreError(f"{os.path.basename(bundle)}'s manifest is finalized")
            _write_atomically(path, schema.dumps(manifest))
            if manifest.get("finalized"):
                make_read_only(path)

    def unlock_manifest(self, bundle: str) -> None:
        path = os.path.join(bundle, MANIFEST)
        if os.path.exists(path):
            _make_writable(path)

    # ── reading ───────────────────────────────────────────────────────────
    def incident_ids(self) -> list[str]:
        """Complete bundles, oldest first."""
        if not os.path.isdir(self.root):
            return []
        return sorted(name for name in os.listdir(self.root)
                      if schema.INCIDENT_ID_RE.fullmatch(name)
                      and os.path.isfile(os.path.join(self.root, name, "incident.json")))

    def load_record(self, incident_id: str) -> dict[str, Any]:
        with open(os.path.join(self.bundle_dir(incident_id), "incident.json"),
                  encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
        return data

    # ── occurrences ───────────────────────────────────────────────────────
    def _append_line(self, name: str, entry: Mapping[str, Any]) -> None:
        line = json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n"
        with self._lock:
            os.makedirs(self.root, exist_ok=True)
            with open(os.path.join(self.root, name), "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())

    def append_occurrence(self, occurrence: Mapping[str, Any]) -> None:
        """One line per later sighting. Append-only: sightings are added,
        never edited, and the bundle they point at is never touched."""
        self._append_line(OCCURRENCES, occurrence)

    def append_capture_failure(self, entry: Mapping[str, Any]) -> None:
        """The last-resort record of a detection whose bundle could not be
        written: one small line, so the fact that it happened survives."""
        self._append_line(CAPTURE_FAILURES, entry)

    def occurrences(self) -> Iterator[dict[str, Any]]:
        """Every recorded sighting, oldest first. A torn last line (the
        process died mid-append) is skipped, not fatal."""
        path = os.path.join(self.root, OCCURRENCES)
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    yield item

    # ── recovery ──────────────────────────────────────────────────────────
    def recover(self) -> dict[str, list[str]]:
        """Makes the store consistent after a crash, destroying nothing.

        * a bundle directory without incident.json was interrupted before its
          capture finished: moved to _incomplete/ (kept for inspection)
        * a leftover *.tmp inside a bundle is an interrupted atomic write
          whose target was never created or is complete: removed
        Returns what it did.
        """
        done: dict[str, list[str]] = {"moved_incomplete": [], "removed_temp": []}
        if not os.path.isdir(self.root):
            return done
        with self._lock:
            for name in sorted(os.listdir(self.root)):
                path = os.path.join(self.root, name)
                if not (schema.INCIDENT_ID_RE.fullmatch(name) and os.path.isdir(path)):
                    continue
                for leftover in os.listdir(path):
                    if leftover.endswith(".tmp"):
                        tmp = os.path.join(path, leftover)
                        _make_writable(tmp)
                        os.remove(tmp)
                        done["removed_temp"].append(f"{name}/{leftover}")
                if not os.path.isfile(os.path.join(path, "incident.json")):
                    target_dir = os.path.join(self.root, INCOMPLETE_DIR)
                    os.makedirs(target_dir, exist_ok=True)
                    os.replace(path, os.path.join(target_dir, name))
                    done["moved_incomplete"].append(name)
        return done

    def verify(self, incident_id: str) -> list[str]:
        """Re-hashes every artifact the manifest lists; returns the problems."""
        bundle = self.bundle_dir(incident_id)
        problems: list[str] = []
        manifest = self.read_manifest(bundle)
        for name, meta in manifest.get("artifacts", {}).items():
            path = os.path.join(bundle, name)
            if not os.path.isfile(path):
                problems.append(f"{name}: missing")
            elif sha256_of(path) != meta.get("sha256"):
                problems.append(f"{name}: SHA-256 differs from the manifest")
        return problems
