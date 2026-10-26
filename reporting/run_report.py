"""The report for a run that reached the end of the level.

An incident report (reporting/pipeline.py) says "a detector fired here". This
one is about a whole run, from the start of an episode to the castle. On the
clean baseline game it says no detector fired ("no bugs found"). On the
bugged game it is the run's final bug report: every bug the run recorded,
gathered in one report (each one keeps its own incident report too). Both
are written from measurements only - the run record is the canonical file,
and the Markdown and PDF are derived from it, like an incident's.

    run.json     the canonical record (RUN_SCHEMA)
    final.png    the last frame of the run, full resolution, lossless
    finish.gif   the run's last seconds, from the dashboard's stream frames
    report.md    human-readable, derived from run.json
    report.pdf   the same content as the Markdown

A "no bugs found" verdict is only ever about THIS run and THESE detectors: it
is evidence that the detectors stayed silent, not proof that the game has no
bugs. Every report says so.
"""
from __future__ import annotations

import datetime
import io
import json
import logging
import os
import re
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import cv2
import numpy as np
from fpdf.enums import XPos, YPos
from PIL import Image

from common.fileio import remove_tree, retire_dir
from reporting import schema
from reporting.evidence import TITLES, encode_png
from reporting.render import STREAM_SIZE, _heading, _kv_table, _ReportPDF, pdf_text

log = logging.getLogger(__name__)

RUN_SCHEMA = "glitch-hunter.run-report/1"
RUN_ID_RE = re.compile(r"RUN-[0-9]{8}-[0-9]{6}-[0-9a-f]{6}")
RAW_FILES = ("run.json", "final.png")
DERIVED_FILES = ("finish.gif", "report.md", "report.pdf")
FILES = RAW_FILES + DERIVED_FILES
_FRAME_MS = 4 * 1000.0 / 60.0     # one stream frame per agent step = 4 engine frames
_FINAL_HOLD_MS = 2000

OUTCOMES = {
    "level_complete": "Level complete",
    "death": "Mario died",
    "timeout": "Time ran out",
    "safety_reset": "Stopped: the agent was stuck",
    "engine_done": "Run ended",
    "time_limit": "Run ended (step limit)",
}

NO_BUGS_LIMITATION = (
    "\"No bugs found\" means no detector fired during this run. It is not proof that the game "
    "has no bugs: a bug the detectors do not look for, or one this run never reached, would "
    "not be reported.")
BUGS_LIMITATION = (
    "Only bugs a detector looks for, at places this run reached, are listed. Each one also has "
    "its own incident report, with its GIF, trace and reproduction; this report gathers the "
    "run's bugs in one place.")
LIMITATIONS = (
    "The run is one episode played by the trained agent. Another run may take a different path.",
    ("The GIF shows the dashboard's stream frames (480x360, JPEG, one every 4 engine frames); "
     "only final.png is full-resolution and lossless."),
)


def limitations(record: Mapping[str, Any]) -> list[str]:
    return [BUGS_LIMITATION if record["bugs"]["count"] else NO_BUGS_LIMITATION, *LIMITATIONS]


def new_run_id(moment: datetime.datetime) -> str:
    return f"RUN-{moment:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def detector_names() -> list[str]:
    """Every real detector the dashboard runs (the synthetic probe is not one)."""
    return [TITLES[k] for k in TITLES if k != "synthetic_probe"]


def build_run_record(*, run_id: str, created: datetime.datetime, started: datetime.datetime,
                     session_id: str, env_episode_index: int, agent_steps: int,
                     end_reason: str, info: Mapping[str, Any], furthest_x: int,
                     bugs: Sequence[Mapping[str, Any]], reward_mode: str,
                     provenance: Mapping[str, Any]) -> dict[str, Any]:
    """The canonical record of one finished run. `bugs` are the incident
    summaries recorded during it (new or seen again); none means no bug."""
    rect = info.get("mario_rect") or (None, None)
    return {
        "schema": RUN_SCHEMA,
        "run_id": run_id,
        "created_utc": schema.iso_utc(created),
        "started_utc": schema.iso_utc(started),
        "duration_s": round((created - started).total_seconds(), 1),
        "session_id": session_id,
        "env_episode_index": int(env_episode_index),
        "agent_steps": int(agent_steps),
        "engine_frames": int(agent_steps) * 4,
        "outcome": {"end_reason": end_reason,
                    "label": OUTCOMES.get(end_reason, end_reason),
                    "level_complete": end_reason == "level_complete"},
        "final_state": {"x": rect[0], "y": rect[1], "furthest_x": int(furthest_x),
                        "score": info.get("score"), "coins": info.get("coins"),
                        "time_left": info.get("hud_time", info.get("time_left")),
                        "power": info.get("status")},
        "bugs": {"count": len(bugs), "incidents": [bug_entry(b) for b in bugs]},
        "verdict": "no_bugs_found" if not bugs else "bugs_found",
        "detectors": detector_names(),
        "reward_mode": reward_mode,
        "provenance": dict(provenance),
    }


def bug_entry(bug: Mapping[str, Any]) -> dict[str, Any]:
    """What the run report keeps of one incident summary (pipeline.summary):
    enough to read every bug of the run in this one report."""
    loc = bug.get("location") or {}
    return {"incident_id": bug.get("incident_id"), "title": bug.get("title"),
            "category": bug.get("category"), "description": bug.get("description"),
            "severity": bug.get("severity"), "confidence": bug.get("confidence"),
            "location": {"x": loc.get("x"), "y": loc.get("y")},
            "episode_agent_step": bug.get("episode_agent_step"),
            "occurrences": bug.get("occurrences"), "reproduction": bug.get("reproduction"),
            "fix": bug.get("fix")}


def headline(record: Mapping[str, Any]) -> str:
    n = record["bugs"]["count"]
    if n == 0:
        return "No Bugs Found"
    return f"{n} Bug{'s' if n != 1 else ''} Found"


# ═══════════════════════════════════════════════════════════════════════
# SHARED CONTENT (Markdown and PDF read the same rows, so cannot disagree)
# ═══════════════════════════════════════════════════════════════════════
def overview_rows(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    p = record["provenance"]
    brain, game = p.get("brain", {}), p.get("game", {})
    fs = record["final_state"]
    brain_text = (f"{brain.get('path')} (SHA-256 {str(brain.get('sha256'))[:12]}...) - "
                  + ("the approved Objective-2 brain" if brain.get("approved_objective2_brain")
                     else "NOT the approved Objective-2 brain")) if brain.get("present") \
        else "no trained brain (untrained policy)"
    game_text = (f"{game.get('variant')} (game tree SHA-256 {str(game.get('tree_sha256'))[:12]}...)"
                 + (" - matches the pinned clean baseline" if game.get("matches_pinned_clean_tree")
                    else " - differs from the clean baseline"))
    bugs = record["bugs"]
    bug_text = "none" if bugs["count"] == 0 else "; ".join(
        f"{b['incident_id']} ({b['title']})" for b in bugs["incidents"])
    return [
        ("Verdict", headline(record)),
        ("Outcome", record["outcome"]["label"]),
        ("Run", record["run_id"]),
        ("Game", game_text),
        ("Brain", brain_text),
        ("Started (UTC)", record["started_utc"]),
        ("Finished (UTC)", record["created_utc"]),
        ("Duration", f"{record['duration_s']} s of wall-clock time"),
        ("Length", f"{record['agent_steps']:,} agent steps ({record['engine_frames']:,} engine frames)"),
        ("Furthest point", f"world x {fs['furthest_x']}"),
        ("Final position", f"world x {fs['x']}, y {fs['y']}"),
        ("Score / coins", f"{fs['score']} / {fs['coins']}"),
        ("Time left on the clock", str(fs["time_left"])),
        ("Bugs recorded in this run", bug_text),
        ("Session / episode", f"{record['session_id']} / episode {record['env_episode_index']}"),
    ]


def report_kind(record: Mapping[str, Any]) -> str:
    """A run with bugs gets the run's final bug report; one without, a run report."""
    return "Final bug report" if record["bugs"]["count"] else "Run report"


def verdict_line(record: Mapping[str, Any]) -> str:
    if record["bugs"]["count"] == 0:
        return (f"The agent played the clean game to the end ({record['outcome']['label'].lower()}) "
                f"and none of the {len(record['detectors'])} detectors fired.")
    return (f"The run ended ({record['outcome']['label'].lower()}) with "
            f"{record['bugs']['count']} bug(s) recorded. Every one is listed below; its own "
            f"incident report has the full evidence.")


def bug_rows(bug: Mapping[str, Any]) -> list[tuple[str, str]]:
    """One bug of the run, as rows (Markdown and PDF read the same ones)."""
    loc = bug.get("location") or {}
    rows = [("Incident", str(bug.get("incident_id"))),
            ("Category", str(bug.get("category"))),
            ("What happened", str(bug.get("description") or "-")),
            ("Severity / confidence", f"{bug.get('severity') or '-'} / {bug.get('confidence') or '-'}"),
            ("Where", f"world x {loc.get('x')}, y {loc.get('y')}"),
            ("When", f"agent step {bug.get('episode_agent_step')} of this run"),
            ("Times seen", str(bug.get("occurrences") or "-")),
            ("Reproduction", str(bug.get("reproduction") or "-").replace("_", " "))]
    rows += fix_rows(bug.get("fix"))
    return rows


def fix_rows(fix: Any) -> list[tuple[str, str]]:
    """The incident's fix hint (fix_hint.headline) as the dashboard shows it:
    the likely cause, then every concrete edit, or else the place and the
    suggestion."""
    if not fix:
        return []
    if not isinstance(fix, Mapping):
        return [("Fix hint", str(fix))]
    rows = []
    if fix.get("summary"):
        rows.append(("Likely cause", str(fix["summary"])))
    edits = fix.get("fixes") or []
    rows.extend(("Suggested edit", f"{edit.get('where')}: {edit.get('fix')}") for edit in edits)
    if not edits:
        if fix.get("where"):
            rows.append(("Where to fix", str(fix["where"])))
        if fix.get("suggestion"):
            rows.append(("Suggestion", str(fix["suggestion"])))
    return rows


def bug_heading(i: int, bug: Mapping[str, Any]) -> str:
    return f"Bug {i}: {bug.get('title')}"


# ═══════════════════════════════════════════════════════════════════════
# RENDERERS
# ═══════════════════════════════════════════════════════════════════════
def render_finish_gif(stream_jpegs: Sequence[bytes], final_png: bytes | None) -> bytes:
    """The run's last seconds at game speed, then the final frame, held."""
    frames: list[np.ndarray] = []
    durations: list[int] = []
    for jpeg in stream_jpegs:
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        durations.append(round(_FRAME_MS))
    if final_png:
        img = cv2.imdecode(np.frombuffer(final_png, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), STREAM_SIZE,
                                     interpolation=cv2.INTER_AREA))
            durations.append(_FINAL_HOLD_MS)
    if not frames:
        raise ValueError("no frames to animate")
    images = [Image.fromarray(f) for f in frames]
    buf = io.BytesIO()
    images[0].save(buf, format="GIF", save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=False, disposal=1)
    return buf.getvalue()


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(record: Mapping[str, Any], rendered_at: datetime.datetime) -> str:
    lines = [f"# Glitch Hunter {report_kind(record).lower()} - {headline(record)}", "",
             verdict_line(record), "", "| | |", "|---|---|"]
    lines += [f"| {_md_escape(k)} | {_md_escape(v)} |" for k, v in overview_rows(record)]
    incidents = record["bugs"]["incidents"]
    if incidents:
        lines += ["", f"## Bugs found in this run ({len(incidents)})"]
        for i, bug in enumerate(incidents, 1):
            lines += ["", f"### {_md_escape(bug_heading(i, bug))}", "", "| | |", "|---|---|"]
            lines += [f"| {_md_escape(k)} | {_md_escape(v)} |" for k, v in bug_rows(bug)]
    lines += ["", "## Final frame", "", "![final frame](final.png)", "",
              "The run's last seconds: [finish.gif](finish.gif)", "",
              f"## Detectors that watched this run ({len(record['detectors'])})", ""]
    lines += [f"- {d}" for d in record["detectors"]]
    lines += ["", "## Limitations", ""]
    lines += [f"- {line}" for line in limitations(record)]
    lines += ["", (f"_Rendered {rendered_at.astimezone(datetime.UTC):%Y-%m-%d %H:%M:%S} UTC "
                   f"from run.json ({record['schema']})._"), ""]
    return "\n".join(lines)


class _RunPDF(_ReportPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, pdf_text(f"Glitch Hunter - run {self.incident_id}"),
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.ln(2)


def _bug_block(pdf: Any, i: int, bug: Mapping[str, Any], frame: bytes | None) -> None:
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.multi_cell(0, 6, pdf_text(bug_heading(i, bug)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    _kv_table(pdf, bug_rows(bug))
    if frame:
        pdf.image(io.BytesIO(frame), w=min(pdf.epw, 75))
    pdf.ln(3)


def render_pdf(record: Mapping[str, Any], final_png: bytes | None,
               rendered_at: datetime.datetime,
               bug_frames: Mapping[str, bytes] | None = None) -> bytes:
    """`bug_frames`: each bug's trigger frame (PNG) by incident id, shown
    under that bug; a bug without one is listed without a picture."""
    pdf = _RunPDF(record["run_id"], False)
    pdf.set_title(f"Glitch Hunter run {record['run_id']}")
    pdf.set_creator("Glitch Hunter - reporting/run_report.py")
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 8, pdf_text(f"{report_kind(record)} - {headline(record)}"),
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, pdf_text(verdict_line(record)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    _kv_table(pdf, overview_rows(record))
    incidents = record["bugs"]["incidents"]
    if incidents:
        _heading(pdf, f"Bugs found in this run ({len(incidents)})")
        for i, bug in enumerate(incidents, 1):
            frame = (bug_frames or {}).get(str(bug.get("incident_id")))
            # One bug is never split over two pages: it is drawn once on a
            # dummy first, and starts a new page if it would not fit.
            with pdf.offset_rendering() as dummy:
                _bug_block(dummy, i, bug, frame)
            if dummy.page_break_triggered:
                pdf.add_page()
            _bug_block(pdf, i, bug, frame)
    if final_png:
        _heading(pdf, "Final frame", keep_with_next=125)     # the heading stays with its frame
        pdf.image(io.BytesIO(final_png), w=min(pdf.epw, 150))
    _heading(pdf, f"Detectors that watched this run ({len(record['detectors'])})")
    for d in record["detectors"]:
        pdf.multi_cell(0, 4.6, pdf_text(f"- {d}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    _heading(pdf, "Limitations")
    for line in limitations(record):
        pdf.multi_cell(0, 4.6, pdf_text(f"- {line}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.multi_cell(0, 4, pdf_text(
        f"Rendered {rendered_at.astimezone(datetime.UTC):%Y-%m-%d %H:%M:%S} UTC from run.json "
        f"({record['schema']}). The Markdown report carries the same content."),
        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    return bytes(pdf.output())


# ═══════════════════════════════════════════════════════════════════════
# STORE
# ═══════════════════════════════════════════════════════════════════════
class RunReportBusy(Exception):
    """A run report that cannot be deleted right now; the message says why."""


class RunReports:
    """One folder per run under `root`. write() puts the raw files on disk
    before it returns; the GIF, Markdown and PDF follow on a short-lived
    thread (the game is paused by then, but a Reset must not wait for them).
    `notify('run_report_updated', summary)` fires when they are done.

    `refresh`, when given, runs on that thread before anything is drawn and
    returns the record as it is by then - on the bugged game, with the replay
    verdicts that came in after the run ended - which replaces run.json."""

    def __init__(self, root: str, notify: Callable[[str, dict[str, Any]], Any] | None = None,
                 clock: Callable[[], datetime.datetime] = schema.utc_now) -> None:
        self.root = root
        self.notify = notify
        self._clock = clock
        self._renders: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def folder(self, run_id: str) -> str:
        if not RUN_ID_RE.fullmatch(run_id):
            raise KeyError(run_id)
        return os.path.join(self.root, run_id)

    def path(self, run_id: str, name: str) -> str:
        """The file's path, only for a real run and an allow-listed name."""
        if name not in FILES:
            raise KeyError(name)
        path = os.path.join(self.folder(run_id), name)
        if not os.path.isfile(path):
            raise KeyError(name)
        return path

    def write(self, record: Mapping[str, Any], final_frame: np.ndarray | None,
              stream_jpegs: Sequence[bytes], background: bool = True,
              bug_frames: Mapping[str, bytes] | None = None,
              refresh: Callable[[], Mapping[str, Any]] | None = None) -> dict[str, Any]:
        run_id = str(record["run_id"])
        folder = self.folder(run_id)
        os.makedirs(folder, exist_ok=True)
        self._write_record(record)
        final_png = encode_png(final_frame) if final_frame is not None else None
        if final_png:
            with open(os.path.join(folder, "final.png"), "wb") as fh:
                fh.write(final_png)
        with self._lock:
            self._renders[run_id] = dict.fromkeys(DERIVED_FILES, "pending")
        frames = list(stream_jpegs)
        if background:
            t = threading.Thread(target=self._render,
                                 args=(record, final_png, frames, bug_frames, refresh),
                                 name=f"run-report-{run_id}", daemon=True)
            self._threads.append(t)
            t.start()
        else:
            self._render(record, final_png, frames, bug_frames, refresh)
        return self.summary(record)

    def _write_record(self, record: Mapping[str, Any]) -> None:
        with open(os.path.join(self.folder(str(record["run_id"])), "run.json"), "wb") as fh:
            fh.write(json.dumps(record, indent=2, sort_keys=True).encode("utf-8"))

    def _render(self, record: Mapping[str, Any], final_png: bytes | None,
                frames: list[bytes], bug_frames: Mapping[str, bytes] | None = None,
                refresh: Callable[[], Mapping[str, Any]] | None = None) -> None:
        run_id = str(record["run_id"])
        folder = self.folder(run_id)
        if refresh is not None:
            try:
                fresh = refresh()
                if fresh.get("run_id") != run_id:
                    raise ValueError(f"refresh returned run {fresh.get('run_id')!r}")
                record = fresh
                self._write_record(record)
            except Exception:
                log.exception("run %s: could not refresh its record; reporting it as recorded",
                              run_id)
        now = self._clock()
        jobs: dict[str, Callable[[], bytes]] = {
            "finish.gif": lambda: render_finish_gif(frames, final_png),
            "report.md": lambda: render_markdown(record, now).encode("utf-8"),
            "report.pdf": lambda: render_pdf(record, final_png, now, bug_frames),
        }
        for name, job in jobs.items():
            try:
                data = job()
                with open(os.path.join(folder, name), "wb") as fh:
                    fh.write(data)
                status = "done"
            except Exception:
                log.exception("run %s: could not render %s", run_id, name)
                status = "failed"
            with self._lock:
                self._renders[run_id][name] = status
        if self.notify is not None:
            try:
                self.notify("run_report_updated", self.summary(record))
            except Exception:
                log.debug("run report notify failed", exc_info=True)

    def summary(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """What the dashboard shows: the verdict and the files ready so far."""
        run_id = str(record["run_id"])
        folder = self.folder(run_id)
        with self._lock:
            renders = dict(self._renders.get(run_id, {}))
        return {"run_id": run_id, "headline": headline(record),
                "verdict": record["verdict"], "outcome": record["outcome"]["label"],
                "created_utc": record["created_utc"], "agent_steps": record["agent_steps"],
                "bugs": record["bugs"]["count"],
                "bug_list": [{"incident_id": b.get("incident_id"), "title": b.get("title"),
                              "severity": b.get("severity")}
                             for b in record["bugs"]["incidents"]],
                "available": [n for n in FILES if os.path.isfile(os.path.join(folder, n))],
                "renders": renders}

    def summaries(self) -> list[dict[str, Any]]:
        """Every run report on disk, newest first (the dashboard's Bug History).
        A folder that is not a run report, or whose run.json cannot be read,
        is left out of the list, never allowed to break it."""
        try:
            names = sorted(os.listdir(self.root), reverse=True)
        except OSError:
            return []
        out = []
        for name in names:
            if not RUN_ID_RE.fullmatch(name):
                continue
            try:
                with open(os.path.join(self.root, name, "run.json"), encoding="utf-8") as fh:
                    record = json.load(fh)
                game = record.get("provenance", {}).get("game", {}).get("variant")
                out.append({**self.summary(record), "game_variant": game})
            except (OSError, ValueError, KeyError, TypeError):
                log.warning("run report %s is not readable; left out of the list", name)
        return out

    def delete(self, run_id: str) -> None:
        """Deletes one run report (Bug History's Delete and Clear history).
        KeyError when there is no such report; RunReportBusy while its GIF
        and reports are still being written, or one of its files is open."""
        folder = self.folder(run_id)
        if not os.path.isfile(os.path.join(folder, "run.json")):
            raise KeyError(run_id)
        with self._lock:
            if "pending" in self._renders.get(run_id, {}).values():
                raise RunReportBusy("Its report is still being written. Try again in a moment.")
            try:
                retired = retire_dir(folder)
            except OSError as exc:
                raise RunReportBusy("One of its files is open. Close it and try again.") from exc
            self._renders.pop(run_id, None)
        remove_tree(retired)

    def wait(self, timeout: float = 30.0) -> None:
        """Tests: until every render thread started so far has finished."""
        for t in list(self._threads):
            t.join(timeout)
