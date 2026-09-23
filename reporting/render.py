"""The human-facing artifacts of an incident: GIF, Markdown, PDF.

All three are DERIVED. They read only the canonical record (incident.json),
the raw evidence files and the manifest's status, so each can be rebuilt at
any time and none of them can add a fact the capture did not record. Every
statement is either a measurement copied from the record or explicitly
labelled as interpretation.
"""
from __future__ import annotations

import datetime
import io
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np
from fpdf import FPDF
from fpdf.enums import TableCellFillMode, XPos, YPos
from fpdf.fonts import FontFace
from PIL import Image

from exploration import config
from reporting.evidence import decode_png, read_context_zip

STREAM_SIZE = (480, 360)          # the dashboard stream's frame size (dashboard_backend)
_FRAME_MS = 1000.0 / 60.0
TRACE_ROWS_IN_REPORT = 24         # six agent steps of per-frame detail before the trigger

REPRODUCTION_MEANING = {
    "reproduced": "A replay of the recorded episode reached the same state on the same frame, "
                  "the same detector fired there, and the frame is pixel-identical.",
    "reproduced_state_only": "The replay reached the same state on the same frame and the same "
                             "detector fired, but the rendered frame differs.",
    "not_reproduced": "The replay reached the same state on the same frame, but the detector "
                      "did not fire again.",
    "diverged": "The replay's game state separated from the recording before the trigger, so "
                "the moment could not be recreated.",
    "not_possible": "The episode could not be replayed.",
    "timeout": "The replay did not finish within its time budget.",
    "error": "The replay process failed.",
    "pending": "The replay has not finished yet.",
    "not_attempted": "No replay was attempted.",
}

SYNTHETIC_NOTICE_MD = ("> **SYNTHETIC TEST EVENT - NOT A GAME BUG.** Produced by the reporting "
                       "pipeline's own probe to prove capture, reports and the dashboard work. "
                       "Nothing about the game should be concluded from it.")

LIMITATIONS = (
    "The detector reports what it observed; it does not know WHY. No root cause is claimed.",
    ("The trigger frame is the exact engine frame the detector fired on. The game window, "
     "paused afterwards, can show a frame up to 3 engine frames later, because the agent's "
     "4-frame action completes before testing pauses."),
    ("Context frames are the dashboard's own stream frames: one every 4 engine frames, "
     "480x360, JPEG. Only the trigger frame is full-resolution and lossless."),
    ("Reproduction replays the recorded ENGINE inputs (every frame's action and clock top-up) "
     "from the episode's reset. It does not re-run the policy, and says nothing about how "
     "often the agent reaches this state."),
    ("Occurrence counts are a snapshot taken when this report was rendered; the dashboard "
     "shows the live count."),
)


def _utc(record: Mapping[str, Any]) -> str:
    return str(record.get("created_utc", "?"))


# ═══════════════════════════════════════════════════════════════════════
# GIF
# ═══════════════════════════════════════════════════════════════════════
def _label(img: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), thickness=-1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def render_gif(record: Mapping[str, Any], context_zip: bytes | None,
               trigger_png: bytes | None) -> bytes:
    """The moments before the trigger at real game speed, then the exact
    trigger frame, outlined in red and held so it can be read."""
    frames: list[np.ndarray] = []
    durations: list[int] = []
    trigger_substep = int(record["timing"]["episode_substep"])
    context = read_context_zip(context_zip) if context_zip else []
    for i, (entry, img) in enumerate(context):
        img = cv2.resize(img, STREAM_SIZE, interpolation=cv2.INTER_AREA) \
            if (img.shape[1], img.shape[0]) != STREAM_SIZE else img.copy()
        last_substep = int(entry["episode_agent_step"]) * config.SUBSTEPS_PER_AGENT_STEP - 1
        _label(img, f"{(last_substep - trigger_substep) * _FRAME_MS / 1000:+.2f} s  "
                    f"(agent step {entry['episode_agent_step']})", (255, 255, 255))
        frames.append(img)
        # Each frame shows until the next one: 4 engine frames apart, except
        # the last, which shows until the trigger frame itself.
        nxt = (int(context[i + 1][0]["episode_agent_step"]) * config.SUBSTEPS_PER_AGENT_STEP - 1
               if i + 1 < len(context) else trigger_substep)
        durations.append(max(10, round((nxt - last_substep) * _FRAME_MS)))
    if trigger_png:
        trig = cv2.resize(decode_png(trigger_png), STREAM_SIZE, interpolation=cv2.INTER_AREA)
        cv2.rectangle(trig, (0, 0), (STREAM_SIZE[0] - 1, STREAM_SIZE[1] - 1), (230, 30, 30), 6)
        _label(trig, f"TRIGGER  {record['summary']['category']}  engine frame {trigger_substep}",
               (255, 80, 80))
        frames.append(trig)
        durations.append(config.GIF_TRIGGER_HOLD_MS)
    if not frames:
        raise ValueError("no frames to animate: neither context frames nor a trigger frame")
    images = [Image.fromarray(f) for f in frames]
    buf = io.BytesIO()
    images[0].save(buf, format="GIF", save_all=True, append_images=images[1:],
                   duration=durations, loop=0, optimize=False, disposal=1)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════
# SHARED CONTENT (used by both Markdown and PDF, so they cannot disagree)
# ═══════════════════════════════════════════════════════════════════════
def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_fmt(v) for v in value) + ")"
    return str(value)


def overview_rows(record: Mapping[str, Any], manifest: Mapping[str, Any],
                  occurrences: int) -> list[tuple[str, str]]:
    c, t, p = record["classification"], record["timing"], record["provenance"]
    brain, game = p.get("brain", {}), p.get("game", {})
    rep = manifest.get("reproduction", {}).get("status", "not_attempted")
    loc = record["location"].get("world_collider") or {}
    steps = brain.get("num_timesteps")
    brain_text = (f"{brain.get('path')} (SHA-256 {str(brain.get('sha256'))[:12]}..., "
                  f"{f'{steps:,}' if isinstance(steps, int) else '?'} steps) - "
                  + ("the approved Objective-2 brain" if brain.get("approved_objective2_brain")
                     else "NOT the approved Objective-2 brain")) if brain.get("present") \
        else "no trained brain (untrained policy)"
    game_text = (f"{game.get('variant')} (game tree SHA-256 {str(game.get('tree_sha256'))[:12]}...)"
                 + (" - matches the pinned clean baseline" if game.get("matches_pinned_clean_tree")
                    else " - differs from the clean baseline")
                 + (f"; declared injected bugs: {', '.join(game['declared_injected_bugs'])}"
                    if game.get("declared_injected_bugs") else "; no declared injected bugs"))
    return [
        ("Incident", record["incident_id"]),
        ("Category", record["summary"]["category"]),
        ("Synthetic test event", "YES - pipeline test, not a game bug" if record["synthetic"]
         else "no"),
        ("Severity (impact if real)", c["severity"]),
        ("Detector confidence", c["detector_confidence"]["level"]),
        ("Reproduction", rep),
        ("Occurrences", f"{occurrences} (as of this report)"),
        ("Captured (UTC)", _utc(record)),
        ("Engine time (exact)", f"{t['engine_time_ms'] / 1000:.3f} s into the episode"),
        ("Where (world)", (f"collider x={loc.get('x')}, y={loc.get('y')}, "
                           f"{loc.get('w')}x{loc.get('h')}") if loc else "unknown"),
        ("When", (f"session {t['session_id']}, episode {t['episode_index']}, agent step "
                  f"{t['episode_agent_step']} (engine frame {t['episode_substep']}, "
                  f"{t['substep_in_agent_step'] + 1} of {config.SUBSTEPS_PER_AGENT_STEP})")),
        ("Game", game_text),
        ("Brain", brain_text),
        ("Reward mode", str(p.get("reward_mode"))),
    ]


def fact_rows(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    g, s = record["gameplay"], record["gameplay"]["state"]
    m = g.get("mario", {})
    rows = [("Detector", record["detector"]["id"]),
            ("Detector message", record["detector"]["message"])]
    rows += [(f"Measured: {k}", _fmt(v)) for k, v in sorted(record["detector"]["metrics"].items())]
    rows += [
        ("Collider (x, y, w, h)", _fmt(s.get("mario_rect"))),
        ("Velocity (x, y)", f"{_fmt(s.get('x_vel'))}, {_fmt(m.get('y_vel'))}"),
        ("Mario state / form", f"{m.get('state')} / {s.get('status')}"),
        ("On ground / dead", f"{s.get('on_ground')} / {s.get('is_dead')}"),
        ("Score / coins", f"{s.get('score')} / {s.get('coins')}"),
        ("Engine clock (time left)", _fmt(s.get("time_left"))),
        ("Camera (viewport x)", _fmt(s.get("viewport_x"))),
        ("Agent action (this step)", f"{g['agent_action']['id']} - {g['agent_action']['name']}"),
    ]
    return rows


def interpretation_lines(record: Mapping[str, Any]) -> list[str]:
    c = record["classification"]
    conf = c["detector_confidence"]
    if record["synthetic"]:
        return [("This incident was produced by the SYNTHETIC pipeline probe, which fires on "
                 "ordinary, correct gameplay (Mario reaching a chosen x position). It exists to "
                 "prove the reporting pipeline end to end. It is NOT a discovered game bug, and "
                 "nothing about the game should be concluded from it.")]
    lines = [f"Severity {c['severity']}: {c['severity_rationale']}"]
    lines += [f"Detector confidence {conf['level']} ({conf['basis']}): {r}" for r in conf["reasons"]]
    lines += [f"Confidence lowered: {d}" for d in conf.get("downgrades", [])]
    lines.append("Root cause: unknown. The system recorded what happened, not why.")
    return lines


def nearest_geometry(record: Mapping[str, Any], n: int = 8) -> list[Mapping[str, Any]]:
    center = record["location"].get("collider_center")
    items = list(record.get("geometry", []))
    if not center:
        return items[:n]

    def dist(g: Mapping[str, Any]) -> float:
        dx = max(g["x"] - center["x"], 0, center["x"] - (g["x"] + g["w"]))
        dy = max(g["y"] - center["y"], 0, center["y"] - (g["y"] + g["h"]))
        return float(dx * dx + dy * dy)
    return sorted(items, key=dist)[:n]


def trace_rows(trajectory: Mapping[str, Any] | None,
               limit: int = TRACE_ROWS_IN_REPORT) -> list[list[str]]:
    if not trajectory:
        return []
    return [[str(t.get("substep")), str(t.get("action")), _fmt(t.get("x")),
             _fmt(t.get("y")), _fmt(t.get("x_vel")), _fmt(t.get("y_vel")),
             str(t.get("mario_state")), str(t.get("on_ground"))]
            for t in list(trajectory.get("trace", []))[-limit:]]


TRACE_HEADER = ["frame", "action", "x", "y", "x_vel", "y_vel", "state", "on ground"]


def reproduction_lines(manifest: Mapping[str, Any],
                       reproduction: Mapping[str, Any] | None) -> list[str]:
    status = manifest.get("reproduction", {}).get("status", "not_attempted")
    lines = [f"Status: {status}. {REPRODUCTION_MEANING.get(status, '')}"]
    if reproduction:
        if reproduction.get("detail"):
            lines.append(f"Detail: {reproduction['detail']}")
        div = reproduction.get("first_divergence")
        if div:
            lines.append(f"First divergence: frame {div.get('substep')}, {div.get('field')} "
                         f"recorded {div.get('recorded')!r}, replayed {div.get('replayed')!r}")
        for key, label in (("state_match", "Same state on every compared frame"),
                           ("frame_match", "Trigger frame pixel-identical"),
                           ("compared_frames", "Frames compared"),
                           ("replayed_substeps", "Frames replayed"),
                           ("elapsed_s", "Replay time (s)")):
            if reproduction.get(key) is not None:
                lines.append(f"{label}: {_fmt(reproduction[key])}")
        fired = reproduction.get("detector_fired_at")
        if fired is not None:
            lines.append("Detectors that fired in the replay: " + (
                ", ".join(f"{f['detector']} at frame {f['frame']}" for f in fired) or "none"))
    return lines


def evidence_rows(manifest: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    rows = []
    for name, meta in sorted(manifest.get("artifacts", {}).items()):
        rows.append((name, f"{meta.get('bytes', 0):,} B", str(meta.get("sha256", ""))))
    for name, st in sorted(manifest.get("renders", {}).items()):
        if st.get("status") == "failed":
            rows.append((name, "FAILED", str(st.get("last_error", ""))[:120]))
    return rows


# ═══════════════════════════════════════════════════════════════════════
# MARKDOWN
# ═══════════════════════════════════════════════════════════════════════
def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(_md_escape(c) for c in r) + " |" for r in rows]
    return out


def render_markdown(record: Mapping[str, Any], manifest: Mapping[str, Any], occurrences: int,
                    trajectory: Mapping[str, Any] | None,
                    reproduction: Mapping[str, Any] | None,
                    rendered_at: datetime.datetime) -> str:
    out: list[str] = [f"# Incident {record['incident_id']} - {record['summary']['title']}", ""]
    if record["synthetic"]:
        out += [SYNTHETIC_NOTICE_MD, ""]
    out += [*_md_table(["", ""], overview_rows(record, manifest, occurrences)), ""]
    out += ["## What was observed (measured)", ""]
    out += [*_md_table(["Fact", "Value"], fact_rows(record)), ""]
    out += ["## Interpretation (inferred, not measured)", ""]
    out += [*(f"- {line}" for line in interpretation_lines(record)), ""]
    out += ["## Where", ""]
    loc = record["location"]
    if loc.get("screen"):
        out.append(f"- In the trigger frame (800x600), the collider's top-left is at screen "
                   f"({loc['screen']['x']}, {loc['screen']['y']}); the camera's left edge is "
                   f"at world x {loc.get('viewport_x')}.")
    counts: dict[str, int] = {}
    for g in record.get("geometry", []):
        counts[g["group"]] = counts.get(g["group"], 0) + 1
    visible = ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "nothing recorded"
    out.append(f"- Visible in the frame: {visible}.")
    near = nearest_geometry(record)
    if near:
        out += ["", "Nearest colliders and sprites to Mario (world coordinates):", ""]
        out += _md_table(["group", "x", "y", "w", "h", "name/state"],
                         [[g["group"], str(g["x"]), str(g["y"]), str(g["w"]), str(g["h"]),
                           f"{g.get('name', '')} {g.get('state', '')}".strip()] for g in near])
    out += ["", "## What led up to it", ""]
    rows = trace_rows(trajectory)
    if rows:
        out += [(f"The last {len(rows)} engine frames, oldest first; the final row is the "
                 "trigger frame:"), ""]
        out += _md_table(TRACE_HEADER, rows)
    if trajectory:
        acts = ", ".join(f"{a['id']} ({a['name']})"
                         for a in trajectory.get("recent_agent_actions", []))
        out += ["", (f"Agent actions, oldest first, ending with the triggering step: "
                     f"{acts or 'none'}.")]
    out += ["", "## Evidence", ""]
    arts = manifest.get("artifacts", {})
    if "trigger.png" in arts:
        out += ["Exact trigger frame (lossless, full resolution):", "",
                "![trigger frame](trigger.png)", ""]
    if "context.gif" in arts:
        out += ["The moments before, then the trigger (real game speed):", "",
                "![context](context.gif)", ""]
    out += _md_table(["file", "size", "SHA-256"], evidence_rows(manifest))
    out += ["", "## Reproduction", ""]
    out += [f"- {line}" for line in reproduction_lines(manifest, reproduction)]
    out += ["", "## Provenance", ""]
    p = record["provenance"]
    code, runtime, game = p.get("code", {}), p.get("runtime", {}), p.get("game", {})
    dirty = " (with uncommitted changes)" if code.get("uncommitted_changes") else ""
    out += [f"- Code: git {code.get('git_commit') or 'unavailable'}{dirty}",
            f"- Runtime: Python {runtime.get('python')}, {runtime.get('platform')}",
            f"- Game variant: {game.get('variant')}, tree {game.get('tree_sha256')}"]
    cons = record["consistency"]
    out += ["", "## Consistency checks", "",
            f"- Detection and caller agree on the episode: {cons['episode_match']}",
            f"- Detection and caller agree on the agent step: {cons['agent_step_match']}"]
    out += [f"- Note: {n}" for n in cons.get("notes", [])]
    out += ["", "## Limitations", "", *(f"- {x}" for x in LIMITATIONS)]
    out += ["", (f"_Rendered {rendered_at.astimezone(datetime.UTC):%Y-%m-%d %H:%M:%S} UTC from "
                 f"incident.json (schema {record['schema']}). Regenerating it never changes the "
                 "recorded facts._"), ""]
    return "\n".join(out)


# ═══════════════════════════════════════════════════════════════════════
# PDF
# ═══════════════════════════════════════════════════════════════════════
_PDF_MAP = str.maketrans({"\u2014": "-", "\u2013": "-", "\u2192": "->", "\u2190": "<-",
                          "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
                          "\u2026": "...", "\u00d7": "x", "\u2265": ">=", "\u2264": "<="})


def pdf_text(text: Any) -> str:
    """The PDF's core fonts are Latin-1; anything else becomes '?'."""
    return str(text).translate(_PDF_MAP).encode("latin-1", "replace").decode("latin-1")


class _ReportPDF(FPDF):
    def __init__(self, incident_id: str, synthetic: bool) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.incident_id = incident_id
        self.synthetic = synthetic
        self.set_auto_page_break(True, margin=16)
        self.set_title(f"Glitch Hunter incident {incident_id}")
        self.set_creator("Glitch Hunter - reporting/render.py")

    def header(self) -> None:
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, pdf_text(f"Glitch Hunter - incident {self.incident_id}"),
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if self.synthetic:
            self.set_text_color(200, 20, 20)
            self.cell(0, 5, "SYNTHETIC TEST EVENT - NOT A GAME BUG",
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, f"page {self.page_no()}/{{nb}}", align="C")


_ZEBRA = (244, 246, 248)
_HEAD = FontFace(emphasis="BOLD", fill_color=(222, 227, 234))


def _heading(pdf: FPDF, text: str, keep_with_next: float = 28) -> None:
    """A section title that never sits alone at the foot of a page."""
    if pdf.get_y() + keep_with_next > pdf.page_break_trigger:
        pdf.add_page()
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, pdf_text(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)


def _kv_table(pdf: FPDF, rows: Sequence[tuple[str, str]], first: float = 48) -> None:
    pdf.set_font("Helvetica", "", 8.5)
    with pdf.table(col_widths=(first, pdf.epw - first), first_row_as_headings=False,
                   line_height=4.6, text_align="LEFT", cell_fill_color=_ZEBRA,
                   cell_fill_mode=TableCellFillMode.ROWS) as table:
        for k, v in rows:
            row = table.row()
            row.cell(pdf_text(k))
            row.cell(pdf_text(v))


def _grid_table(pdf: FPDF, header: Sequence[str], rows: Sequence[Sequence[str]],
                font_size: float = 7.5, col_widths: Sequence[float] | None = None) -> None:
    pdf.set_font("Helvetica", "", font_size)
    with pdf.table(line_height=4, text_align="CENTER", col_widths=col_widths,
                   cell_fill_color=_ZEBRA, cell_fill_mode=TableCellFillMode.ROWS,
                   headings_style=_HEAD) as table:
        head = table.row()
        for h in header:
            head.cell(pdf_text(h))
        for r in rows:
            row = table.row()
            for c in r:
                row.cell(pdf_text(c))


def _filmstrip(context_zip: bytes | None, trigger_png: bytes | None, n: int = 4) -> bytes | None:
    """n context frames spread over the window, then the trigger - one PNG."""
    tiles: list[np.ndarray] = []
    context = read_context_zip(context_zip) if context_zip else []
    if context:
        picks = sorted({round(i * (len(context) - 1) / max(1, n - 1)) for i in range(n)})
        tiles = [cv2.resize(context[i][1], (240, 180), interpolation=cv2.INTER_AREA) for i in picks]
    if trigger_png:
        trig = cv2.resize(decode_png(trigger_png), (240, 180), interpolation=cv2.INTER_AREA)
        cv2.rectangle(trig, (0, 0), (239, 179), (230, 30, 30), 4)
        tiles.append(trig)
    if not tiles:
        return None
    strip = np.hstack([np.pad(t, ((0, 0), (0, 4), (0, 0)), constant_values=255) for t in tiles])
    ok, buf = cv2.imencode(".png", cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    return buf.tobytes() if ok else None


def render_pdf(record: Mapping[str, Any], manifest: Mapping[str, Any], occurrences: int,
               trajectory: Mapping[str, Any] | None, reproduction: Mapping[str, Any] | None,
               trigger_png: bytes | None, context_zip: bytes | None,
               rendered_at: datetime.datetime) -> bytes:
    pdf = _ReportPDF(record["incident_id"], bool(record["synthetic"]))
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 8, pdf_text(record["summary"]["title"]), new_x=XPos.LMARGIN,
                   new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, pdf_text(record["summary"]["description"]), new_x=XPos.LMARGIN,
                   new_y=YPos.NEXT)
    if record["synthetic"]:
        pdf.ln(1)
        pdf.set_fill_color(255, 225, 225)
        pdf.set_font("Helvetica", "B", 9)
        pdf.multi_cell(0, 5, "SYNTHETIC TEST EVENT - NOT A GAME BUG. Produced by the pipeline's "
                             "own probe on ordinary gameplay, to prove the reporting pipeline "
                             "works. Nothing about the game should be concluded from it.",
                       border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_fill_color(255, 255, 255)
    pdf.ln(2)
    _kv_table(pdf, overview_rows(record, manifest, occurrences))
    if trigger_png:
        _heading(pdf, "Exact trigger frame")
        pdf.image(io.BytesIO(trigger_png), w=min(pdf.epw, 150))
    pdf.add_page()
    _heading(pdf, "What was observed (measured)")
    _kv_table(pdf, fact_rows(record))
    _heading(pdf, "Interpretation (inferred, not measured)")
    for line in interpretation_lines(record):
        pdf.multi_cell(0, 4.6, pdf_text(f"- {line}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    strip = _filmstrip(context_zip, trigger_png)
    if strip:
        _heading(pdf, "Before the trigger (left to right), then the trigger (red)")
        pdf.image(io.BytesIO(strip), w=pdf.epw)
    rows = trace_rows(trajectory, 16)
    if rows:
        _heading(pdf, f"The last {len(rows)} engine frames (the last row is the trigger)")
        _grid_table(pdf, TRACE_HEADER, rows)
    near = nearest_geometry(record)
    if near:
        _heading(pdf, "Nearest colliders and sprites (world coordinates)")
        _grid_table(pdf, ["group", "x", "y", "w", "h", "name/state"],
                    [[g["group"], str(g["x"]), str(g["y"]), str(g["w"]), str(g["h"]),
                      f"{g.get('name', '')} {g.get('state', '')}".strip()] for g in near])
    _heading(pdf, "Reproduction")
    for line in reproduction_lines(manifest, reproduction):
        pdf.multi_cell(0, 4.6, pdf_text(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    _heading(pdf, "Evidence files")
    _grid_table(pdf, ["file", "size", "SHA-256"], evidence_rows(manifest), font_size=6.2,
                col_widths=(34, 20, pdf.epw - 54))
    _heading(pdf, "Consistency checks")
    cons = record["consistency"]
    agree = (f"Episode agrees: {cons['episode_match']}; agent step agrees: "
             f"{cons['agent_step_match']}")
    for line in [agree, *(f"Note: {n}" for n in cons.get("notes", []))]:
        pdf.multi_cell(0, 4.6, pdf_text(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    _heading(pdf, "Limitations")
    for line in LIMITATIONS:
        pdf.multi_cell(0, 4.6, pdf_text(f"- {line}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.multi_cell(0, 4, pdf_text(
        f"Rendered {rendered_at.astimezone(datetime.UTC):%Y-%m-%d %H:%M:%S} UTC from incident.json "
        f"({record['schema']}). The Markdown report carries the same content."),
        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    return bytes(pdf.output())
