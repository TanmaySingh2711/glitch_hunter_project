"""The report for a clean-game run that reached the end of the level.

An incident report (reporting/pipeline.py) says "a detector fired here". This
one says the opposite about a whole run: the agent played the clean baseline
game from the start of an episode to the castle and no detector fired. Both
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

LIMITATIONS = (
    ("\"No bugs found\" means no detector fired during this run. It is not proof that the game "
     "has no bugs: a bug the detectors do not look for, or one this run never reached, would "
     "not be reported."),
    "The run is one episode played by the trained agent. Another run may take a different path.",
    ("The GIF shows the dashboard's stream frames (480x360, JPEG, one every 4 engine frames); "
     "only final.png is full-resolution and lossless."),
)


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
        "bugs": {"count": len(bugs),
                 "incidents": [{"incident_id": b.get("incident_id"), "title": b.get("title"),
                                "category": b.get("category")} for b in bugs]},
        "verdict": "no_bugs_found" if not bugs else "bugs_found",
        "detectors": detector_names(),
        "reward_mode": reward_mode,
        "provenance": dict(provenance),
    }


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


def verdict_line(record: Mapping[str, Any]) -> str:
    if record["bugs"]["count"] == 0:
        return (f"The agent played the clean game to the end ({record['outcome']['label'].lower()}) "
                f"and none of the {len(record['detectors'])} detectors fired.")
    return (f"The run ended ({record['outcome']['label'].lower()}) with "
            f"{record['bugs']['count']} bug(s) recorded; see their incident reports.")


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
    lines = [f"# Glitch Hunter run report - {headline(record)}", "",
             verdict_line(record), "", "| | |", "|---|---|"]
    lines += [f"| {_md_escape(k)} | {_md_escape(v)} |" for k, v in overview_rows(record)]
    lines += ["", "## Final frame", "", "![final frame](final.png)", "",
              "The run's last seconds: [finish.gif](finish.gif)", "",
              f"## Detectors that watched this run ({len(record['detectors'])})", ""]
    lines += [f"- {d}" for d in record["detectors"]]
    lines += ["", "## Limitations", ""]
    lines += [f"- {line}" for line in LIMITATIONS]
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


def render_pdf(record: Mapping[str, Any], final_png: bytes | None,
               rendered_at: datetime.datetime) -> bytes:
    pdf = _RunPDF(record["run_id"], False)
    pdf.set_title(f"Glitch Hunter run {record['run_id']}")
    pdf.set_creator("Glitch Hunter - reporting/run_report.py")
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 8, pdf_text(f"Run report - {headline(record)}"),
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, pdf_text(verdict_line(record)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    _kv_table(pdf, overview_rows(record))
    if final_png:
        _heading(pdf, "Final frame")
        pdf.image(io.BytesIO(final_png), w=min(pdf.epw, 150))
    _heading(pdf, f"Detectors that watched this run ({len(record['detectors'])})")
    for d in record["detectors"]:
        pdf.multi_cell(0, 4.6, pdf_text(f"- {d}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    _heading(pdf, "Limitations")
    for line in LIMITATIONS:
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
class RunReports:
    """One folder per run under `root`. write() puts the raw files on disk
    before it returns; the GIF, Markdown and PDF follow on a short-lived
    thread (the game is paused by then, but a Reset must not wait for them).
    `notify('run_report_updated', summary)` fires when they are done."""

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
              stream_jpegs: Sequence[bytes], background: bool = True) -> dict[str, Any]:
        run_id = str(record["run_id"])
        folder = self.folder(run_id)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "run.json"), "wb") as fh:
            fh.write(json.dumps(record, indent=2, sort_keys=True).encode("utf-8"))
        final_png = encode_png(final_frame) if final_frame is not None else None
        if final_png:
            with open(os.path.join(folder, "final.png"), "wb") as fh:
                fh.write(final_png)
        with self._lock:
            self._renders[run_id] = dict.fromkeys(DERIVED_FILES, "pending")
        frames = list(stream_jpegs)
        if background:
            t = threading.Thread(target=self._render, args=(record, final_png, frames),
                                 name=f"run-report-{run_id}", daemon=True)
            self._threads.append(t)
            t.start()
        else:
            self._render(record, final_png, frames)
        return self.summary(record)

    def _render(self, record: Mapping[str, Any], final_png: bytes | None,
                frames: list[bytes]) -> None:
        run_id = str(record["run_id"])
        folder = self.folder(run_id)
        now = self._clock()
        jobs: dict[str, Callable[[], bytes]] = {
            "finish.gif": lambda: render_finish_gif(frames, final_png),
            "report.md": lambda: render_markdown(record, now).encode("utf-8"),
            "report.pdf": lambda: render_pdf(record, final_png, now),
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
                "available": [n for n in FILES if os.path.isfile(os.path.join(folder, n))],
                "renders": renders}

    def wait(self, timeout: float = 30.0) -> None:
        """Tests: until every render thread started so far has finished."""
        for t in list(self._threads):
            t.join(timeout)
