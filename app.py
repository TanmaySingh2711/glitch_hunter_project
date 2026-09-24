"""The dashboard: a Flask-SocketIO server that streams the agent to a browser.

    python app.py                        then open http://localhost:5000
    python app.py --game mario_bugged    run the variant that carries deliberate bugs
    python app.py --synthetic-probe 1000 SYNTHETIC pipeline test: a fake "bug" whenever
                                         Mario reaches world x 1000 (labelled as such
                                         everywhere; never a real bug report)

Every socket handler only POSTS a command to the one game thread
(dashboard_service.GameWindowService); none of them touches the pygame
window or the env itself - see THE GAME WINDOW HAS ONE OWNER below.

Incident evidence (Objective 3) is served read-only under /api/incidents and
/incidents/<id>/<file>; see INCIDENT FILES below for how those requests are
kept inside the incident store.
"""
from __future__ import annotations

import io
import logging
import os
import sys

# Python fully buffers stdout when it isn't attached to an interactive
# terminal (e.g. redirected to a log file, or launched from another tool) -
# output just sits in the buffer until it fills or the process exits.
# Reconfiguring to line-buffering here means every line (this file's own,
# plus everything it imports - dashboard_backend's [STREAM FPS] line, the
# startup pre-load messages below) shows up immediately, which matters for
# actually being able to watch this server's real-time behavior rather
# than only seeing output in one batch on exit.
#
# encoding='utf-8' is not cosmetic: on Windows this console defaults to
# cp1252, which cannot encode the emoji used in the bug-alert log lines -
# printing one raises UnicodeEncodeError and kills whatever was running.
# Verified directly: printing '\U0001f6a8' under cp1252 raises
# "'charmap' codec can't encode character". Forcing UTF-8 makes every log
# line printable regardless of the machine's console codepage.
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

# One OpenBLAS thread, set before numpy loads: the dashboard does no linear
# algebra in numpy (the policy runs in torch), and numpy's OpenBLAS otherwise
# reserves a buffer per core at import - measured ~480 MB of private memory
# for nothing (see train_agent.limit_worker_blas_threads).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import threading
import time
import zipfile
from collections.abc import Mapping
from typing import Any

from flask import Flask, Response, abort, render_template, request, send_file
from flask_socketio import SocketIO

from common.logging_setup import configure_logging

# dashboard_backend pulls in custom_mario_env, which sets SDL_AUDIODRIVER
# before pygame loads - so it has to be imported before pygame is used anywhere.
from dashboard_backend import DashboardBackend, DashboardConfig
from dashboard_service import THREAD_NAME, GameWindowService
from exploration import config
from reporting.store import StoreError

log = logging.getLogger(__name__)

app = Flask(__name__)

# ─── async_mode='threading' (was 'eventlet') ───
# eventlet is unmaintained and Flask-SocketIO's own maintainer now
# recommends the threading mode as the default. It also removes a failure
# mode this project actually hit: eventlet runs every greenlet on ONE OS
# thread and only switches at I/O points it has monkey-patched, so a
# blocking CPU/GPU call (PPO.load(), or a slow env.step()) stalls the whole
# server - including the heartbeat that tells the browser the connection is
# alive. Real OS threads do not share that problem: the frame loop can sit
# on the GPU without stopping the server from answering anyone.
socketio = SocketIO(app, async_mode='threading')

# Cache-busting: appended as ?v=... on static asset URLs (see index.html)
# so a browser that already cached an old style.css/main.js is forced to
# fetch the current one after every restart, instead of silently showing a
# stale page until the user thinks to hard-refresh.
ASSET_VERSION = str(int(time.time()))

LOOPBACK = '127.0.0.1'
DEFAULT_PORT = 5000

# ─── THE GAME WINDOW HAS ONE OWNER ───
# In threading mode every socket event runs on its own short-lived thread.
# The handlers used to create and drive the pygame window themselves, and on
# Windows a window belongs to - and dies with - the thread that created it:
# the popup vanished right after "Start Testing" returned, and its X button
# was never seen. Every window/env operation now happens on ONE long-lived
# game thread (dashboard_service.py); the handlers below only post commands
# to it. That thread is also the single frame loop - there is no way to start
# a second one, however fast Start is clicked.
backend = DashboardBackend()
backend.notify = socketio.emit        # the report worker announces finished reports
service = GameWindowService(backend, emit=socketio.emit)


@app.route('/')
def index() -> str:
    return render_template('index.html', asset_version=ASSET_VERSION)


@app.route('/healthz')
def healthz() -> dict[str, Any]:
    """Liveness + thread census.

    Exists because a stress test once wedged this server by spawning an
    unbounded number of frame-loop threads, and there was no way to see that
    happening from outside. `frame_loops` must never exceed 1; anything more
    means the single-frame-loop invariant has regressed.
    """
    frame_loops = sum(1 for t in threading.enumerate()
                      if t.is_alive() and t.name == THREAD_NAME)
    return {
        "status": "ok",
        "testing": service.testing,
        "threads_total": threading.active_count(),
        "frame_loops": frame_loops,
    }


@app.route('/api/status')
def api_status() -> dict[str, Any]:
    """The backend's truth for the page: running, why it paused, the bug
    that stopped it, and which game and brain are running. A (re)loaded page
    renders from this, so a banner can never be a frontend-only invention."""
    return {**service.status(), **backend.describe()}


# ─── INCIDENT FILES ───
# A request names an incident id and a file name - never a path. The store
# accepts only a well-formed id of an EXISTING incident and a name on its
# allow-list, and checks the resolved file is still inside that incident's
# folder (reporting/store.IncidentStore.artifact_path). Everything else is a
# 404, so "../", absolute paths, symlinks and encoded tricks all end there.
_MIMETYPES = {".png": "image/png", ".gif": "image/gif", ".pdf": "application/pdf",
              ".json": "application/json", ".zip": "application/zip"}


def _pipeline_or_404() -> Any:
    pipeline = backend.pipeline
    if pipeline is None:
        abort(404)
    return pipeline


@app.route('/api/incidents')
def api_incidents() -> dict[str, Any]:
    # The Bug Tracker shows this session's incidents (since the dashboard
    # started or was last reset); ?all=1 lists every incident in the store.
    pipeline = backend.pipeline
    if pipeline is None:
        return {"incidents": [], "scope": "session"}
    if request.args.get('all') == '1':
        return {"incidents": pipeline.summaries(), "scope": "all"}
    return {"incidents": pipeline.session_summaries(), "scope": "session"}


@app.route('/api/incidents/<incident_id>')
def api_incident(incident_id: str) -> dict[str, Any]:
    pipeline = _pipeline_or_404()
    try:
        detail: dict[str, Any] = pipeline.detail(incident_id)
    except (StoreError, KeyError, OSError):
        abort(404)
    return detail


@app.route('/incidents/<incident_id>/bundle.zip')
def incident_bundle(incident_id: str) -> Response:
    """The whole evidence bundle as one download."""
    pipeline = _pipeline_or_404()
    try:
        folder = pipeline.store.bundle_dir(incident_id)
        listing = sorted(os.listdir(folder))
    except (StoreError, OSError):
        abort(404)
    paths = []
    for name in listing:                 # only real, allow-listed bundle files
        try:
            paths.append((name, pipeline.store.artifact_path(incident_id, name)))
        except StoreError:
            continue
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, path in paths:
            zf.write(path, arcname=f"{incident_id}/{name}")
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name=f"{incident_id}.zip", max_age=0)


@app.route('/incidents/<incident_id>/<name>')
def incident_file(incident_id: str, name: str) -> Response:
    """One evidence file, shown in the browser or (?download=1) saved."""
    pipeline = _pipeline_or_404()
    try:
        path = pipeline.store.artifact_path(incident_id, name)
    except StoreError:
        abort(404)
    ext = os.path.splitext(name)[1].lower()
    download = request.args.get('download') == '1'
    # Markdown opens as readable text; downloaded, it keeps its own type.
    # No charset here: Flask appends "; charset=utf-8" to every text/* type
    # itself, and naming it too sent the header with the charset twice.
    mimetype = ("text/markdown" if download else "text/plain") \
        if ext == ".md" else _MIMETYPES.get(ext, "application/octet-stream")
    response = send_file(path, mimetype=mimetype, as_attachment=download,
                         download_name=f"{incident_id}_{name}", max_age=0)
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@socketio.on('connect')
def handle_connect() -> None:
    # A (re)connecting client starts from "not running", and the dashboard
    # agrees; the session and the window are kept, so Start resumes.
    service.client_connected()


@socketio.on('disconnect')
def handle_disconnect() -> None:
    # A page refresh or a closed tab pauses testing. The game window stays
    # as the user left it - only the user closes it (its X, or Reset).
    service.client_disconnected()


@socketio.on('start_testing')
def handle_start_testing(data: object = None) -> None:
    # `data` is unused on purpose and must stay: Flask-SocketIO passes the
    # emitted payload positionally when the client sends one. The dashboard
    # currently emits 'start_testing' with no payload, but a zero-argument
    # handler would raise TypeError the moment anything did send one.
    service.start_testing()


@socketio.on('stop_testing')
def handle_stop_testing() -> None:
    # Pause only. The window is not minimised, hidden or closed: it keeps
    # showing the last frame, exactly like the paused stream.
    service.stop_testing()


@socketio.on('reset_game')
def handle_reset_game() -> None:
    # Ends the session (the next Start is a fresh one) and closes the window.
    service.reset()


@socketio.on('switch_game')
def handle_switch_game(data: object = None) -> None:
    # The "Select Game Environment" list. Only a known variant name gets
    # through; the game thread resets, unloads this game and loads that one,
    # then answers with 'game_switched'.
    variant = data.get('variant') if isinstance(data, dict) else None
    if variant in config.GAME_VARIANTS:
        service.switch_game(variant)


def bind_address(environ: Mapping[str, str] = os.environ) -> tuple[str, int]:
    """(host, port) to listen on.

    ─── LOCALHOST ONLY BY DEFAULT ───
    This used to be host='0.0.0.0', which listens on EVERY network
    interface - anyone on the same Wi-Fi could open the dashboard, drive
    the agent, and watch the stream, with no password in front of it. The
    README only ever told you to visit localhost, so the exposure was
    accidental rather than intended.

    127.0.0.1 means "this machine only". To deliberately watch from
    another device (a phone on the same network, say), opt in explicitly:
        set GLITCH_HUNTER_HOST=0.0.0.0     (Windows)
        GLITCH_HUNTER_HOST=0.0.0.0 python app.py   (Mac/Linux)
    Only do that on a network you trust - there is still no auth.
    """
    host = environ.get('GLITCH_HUNTER_HOST', LOOPBACK)
    raw_port = environ.get('GLITCH_HUNTER_PORT', str(DEFAULT_PORT))
    try:
        port = int(raw_port)
    except ValueError:
        raise SystemExit(f"GLITCH_HUNTER_PORT must be a number, not {raw_port!r}") from None
    if not 0 < port < 65536:
        raise SystemExit(f"GLITCH_HUNTER_PORT {port} is outside 1-65535")
    return host, port


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="The Glitch Hunter dashboard.")
    ap.add_argument("--game", choices=config.GAME_VARIANTS, default=config.DEFAULT_GAME_VARIANT,
                    help="which game variant to test (default: the clean baseline)")
    ap.add_argument("--synthetic-probe", type=int, action="append", default=[], metavar="X",
                    help="PIPELINE TEST ONLY: report a synthetic, clearly labelled 'bug' the "
                         "first time each episode Mario reaches world x >= X (repeatable)")
    ap.add_argument("--incidents-dir", default=None,
                    help=f"where incident bundles are written (default: {config.INCIDENTS_DIR}/)")
    ap.add_argument("--no-reproduce", action="store_true",
                    help="skip the replay-based reproduction of each incident")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args([] if argv is None else argv)
    backend.configure(DashboardConfig(
        game_variant=args.game, synthetic_probes=tuple(args.synthetic_probe),
        incidents_dir=args.incidents_dir or DashboardConfig().incidents_dir,
        reproduce=not args.no_reproduce))
    if args.synthetic_probe:
        log.warning("SYNTHETIC PIPELINE TEST MODE: probes at x = %s. Incidents they produce are "
                    "pipeline tests, not game bugs, and are labelled so.", args.synthetic_probe)
    host, port = bind_address()

    # ─── PRE-LOAD THE MODEL BEFORE ACCEPTING CONNECTIONS ───
    # PPO.load() onto the GPU is a real blocking call (1-3+ seconds of pure
    # CPU/GPU work), so it happens at startup, where a logged message
    # explains it, rather than on the user's first click. It runs ON THE GAME
    # THREAD: building the env creates the OS window, and the thread that
    # creates it is the only one that can ever drive it (see above). The
    # window is hidden until the first "Start Testing".
    log.info("Pre-loading the AI model (this can take a few seconds)...")
    service.start()
    log.info("Model loaded. Ready for connections.")

    if host != LOOPBACK:
        log.warning("Listening on %s - reachable by other devices on this network, "
                    "with no authentication.", host)
    log.info("Dashboard ready at http://localhost:%d", port)

    # debug=False on purpose: the Werkzeug reloader spawns a SECOND python
    # process that also imports the backend and loads the model onto the GPU,
    # which wastes VRAM and leaves an orphan process behind on shutdown.
    # allow_unsafe_werkzeug=True is required in threading mode: without
    # eventlet's WSGI server, Flask-SocketIO falls back to Werkzeug's, which
    # refuses to start outside debug mode unless explicitly allowed. This is
    # a single-user dashboard on localhost, which is exactly the case that
    # flag exists for.
    socketio.run(app, host=host, port=port, debug=False,
                 allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main(sys.argv[1:])
