"""The dashboard's game window, owned by exactly one thread.

WHY ONE THREAD. On Windows a window belongs to the thread that created it:
only that thread receives its messages (move, minimise, the X button), and
the window is DESTROYED when that thread exits. Flask-SocketIO runs every
event handler on its own short-lived thread, and the dashboard used to
create and drive the pygame window from them. Measured: a window created
inside a handler thread reported IsWindow() False the moment that thread
finished, while testing carried on invisibly; a window created on the main
thread and stepped from another went "Not Responding" within 6 s and never
saw a WM_CLOSE.

So a single long-lived GAME THREAD does every window and env operation - the
pre-load, each step, the idle event pump, hiding, showing, closing. Socket
handlers only post commands to it. That also gives the old invariants for
free: there is exactly one frame loop, and nothing is ever held across a
sleep, because the loop's only wait is on its own command queue.

WHAT THE USER CONTROLS
  Start   show the window (created or re-shown centred; an already-open one
          is only brought forward) and resume the SAME session.
  Stop    pause. The window stays exactly as it is - never minimised,
          hidden or closed.
  X       (the window's own close button, while running or paused) hide the
          window and pause. The session is kept; Start brings it back.
  Reset   end the session and close the window; the next Start is fresh.
  Game    (the "Select Game Environment" list) a Reset, then the other game
          variant is loaded in place of this one; the next Start plays it.
  A browser refresh / closed tab pauses, and changes nothing else.

Pausing or closing never touches the agent's state: the session is a
suspended generator, so nothing about the episode, the policy or its
telemetry is discarded, and the dashboard writes no checkpoint, coverage or
lifecycle file at all.

BUG FOUND (Objective 3). When a step reports a NEW incident, this thread
pauses before it takes another step - the evidence is already on disk by then
(reporting/pipeline.capture ran inside that step) - and remembers it as
`bug_found` until the user presses Start (resume) or Reset. Nothing resumes
by itself, not even when the reports finish rendering. A later sighting of an
incident already recorded is counted but does not pause.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Generator
from typing import Any, Protocol

THREAD_NAME = "game-window"
IDLE_PUMP_S = 0.05          # event-pump period while paused: keeps the window responsive
TARGET_FRAME_S = 1.0 / 60.0

_log = logging.getLogger(__name__)

Emit = Callable[..., Any]
Command = str | threading.Event
Session = Generator[dict[str, Any], None, None]


class Backend(Protocol):
    """What the service drives (dashboard_backend.DashboardBackend in the
    app, a recording fake in tests/test_dashboard_control.py)."""

    def preload(self) -> None: ...
    def open_window(self) -> str: ...
    def hide_window(self) -> None: ...
    def close_window(self) -> None: ...
    def poll_close_request(self) -> bool: ...
    def new_session(self) -> Session: ...
    def switch_game(self, variant: str) -> bool: ...
    def stop_audio(self) -> None: ...


class GameWindowService:
    """`backend` does the real work (see Backend above); `emit(event,
    payload)` talks to the browser; `log` receives the service's own status
    lines (the module logger by default)."""

    def __init__(self, backend: Backend, emit: Emit,
                 log: Callable[[str], None] = _log.info) -> None:
        self.backend = backend
        self.emit = emit
        self.log = log
        self._commands: queue.Queue[Command] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._preload_error: BaseException | None = None
        # Read from other threads (healthz, tests); written only by the game thread.
        self.testing = False
        self.session: Session | None = None
        self.steps = 0
        # The incident(s) that stopped testing, until the user resumes or resets.
        self.bug_found: list[dict[str, Any]] | None = None
        self.pause_reason: str | None = None

    # ── called from any thread ────────────────────────────────────────────
    def start(self, timeout: float | None = None) -> None:
        """Starts the game thread and waits for its pre-load. Idempotent."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name=THREAD_NAME, daemon=True)
            self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("the game thread did not finish pre-loading")
        if self._preload_error is not None:
            raise RuntimeError("pre-load failed") from self._preload_error

    def start_testing(self) -> None:
        self._commands.put('start')

    def stop_testing(self) -> None:
        self._commands.put('stop')

    def reset(self) -> None:
        self._commands.put('reset')

    def switch_game(self, variant: str) -> None:
        self._commands.put(f'switch:{variant}')

    def client_connected(self) -> None:
        self._commands.put('connect')

    def client_disconnected(self) -> None:
        self._commands.put('disconnect')

    def shutdown(self, timeout: float = 5.0) -> None:
        self._commands.put('shutdown')
        if self._thread is not None:
            self._thread.join(timeout)

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Blocks until every command posted so far has been handled."""
        done = threading.Event()
        self._commands.put(done)
        return done.wait(timeout)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict[str, Any]:
        """The backend's truth, for a browser that (re)connects: running or
        not, why it last paused, and the bug that stopped it, if any."""
        return {"testing": self.testing, "steps": self.steps,
                "pause_reason": None if self.testing else self.pause_reason,
                "bug_found": self.bug_found}

    # ── the game thread ───────────────────────────────────────────────────
    def _run(self) -> None:
        try:
            self.backend.preload()
        except BaseException as exc:
            self._preload_error = exc
            self._ready.set()
            return
        self._ready.set()
        next_step_at = 0.0
        while True:
            wait = (max(0.0, next_step_at - time.monotonic()) if self.testing
                    else IDLE_PUMP_S)
            try:
                cmd = self._commands.get(timeout=wait)
            except queue.Empty:
                cmd = None
            if cmd == 'shutdown':
                self._close_session()
                return
            if isinstance(cmd, threading.Event):
                cmd.set()
                continue
            if cmd is not None:
                self._handle(cmd)
                continue
            if self.testing and time.monotonic() >= next_step_at:
                t0 = time.monotonic()
                self._step()
                # Half speed, as before: the next step waits as long again
                # as this one took (or a 60 fps frame, whichever is longer).
                next_step_at = t0 + 2 * max(time.monotonic() - t0, TARGET_FRAME_S)
            self._check_close_button()

    def _handle(self, cmd: str) -> None:
        if cmd == 'start':
            self.log(">>> START_TESTING received!")
            try:
                self.backend.open_window()
                if self.session is None:
                    self.session = self.backend.new_session()
                self.testing = True
                self.pause_reason = None
                if self.bug_found is not None:
                    # Resuming is the user's acknowledgement. The incident
                    # itself stays on disk and in the history.
                    self.bug_found = None
                    self.emit('bug_cleared', {})
            except Exception:
                _log.exception("could not start testing")
                self._pause('error', notify=True)
        elif cmd in ('stop', 'connect', 'disconnect'):
            self._pause(cmd)
        elif cmd == 'reset':
            self._pause('reset')
            self._close_session()
            self.backend.close_window()
            if self.bug_found is not None:
                self.bug_found = None
                self.emit('bug_cleared', {})
            self.pause_reason = 'reset'
        elif cmd.startswith('switch:'):
            self._handle('reset')
            variant = cmd.split(':', 1)[1]
            try:
                changed = self.backend.switch_game(variant)
            except Exception as exc:
                _log.exception("could not switch the game to %s", variant)
                self.emit('game_switched', {'variant': variant, 'ok': False, 'error': str(exc)})
                return
            self.log(f">>> GAME SWITCHED to {variant}" if changed else f">>> {variant} already running")
            self.emit('game_switched', {'variant': variant, 'ok': True})

    def _pause(self, reason: str, notify: bool = False) -> None:
        """Stops stepping. Touches nothing else - not the window, not the
        session. `notify` tells the browser, for pauses it did not ask for.

        The recorded reason is why testing STOPPED: a browser reconnecting to
        a dashboard already stopped on a bug does not turn "bug_found" into
        "connect" - the bug is still why it is not running."""
        was_testing = self.testing
        self.testing = False
        if was_testing or self.bug_found is None:
            self.pause_reason = reason
        try:
            self.backend.stop_audio()
        except Exception:                 # silence is best-effort; the pause is not
            _log.debug("stop_audio failed", exc_info=True)
        if notify:
            self.emit('testing_paused', {'reason': reason})

    def _close_session(self) -> None:
        if self.session is not None:
            try:
                self.session.close()
            except Exception:             # a session that fails to close is still gone
                _log.debug("session close failed", exc_info=True)
            self.session = None
            self.steps = 0

    def _step(self) -> None:
        if self.session is None:          # reset between the check and the step
            self._pause('session_ended')
            return
        try:
            item = next(self.session)
        except StopIteration:
            self.session = None
            self._pause('session_ended', notify=True)
            return
        except Exception as exc:
            _log.exception("Error in the game loop: %s", exc)
            # A generator that raised is finished for good: keeping it would
            # make the next Start report "session ended" instead of playing.
            self._close_session()
            self._pause('error', notify=True)
            return
        self.steps += 1
        self.emit('video_frame', {'frame': item['frame']})
        if item.get('log'):
            self.emit('agent_log', {'log': item['log']})
        self._handle_incidents(item.get('incidents') or ())

    def _handle_incidents(self, outcomes: Any) -> None:
        """A new incident stops testing, here, before another step runs."""
        new = [o["summary"] for o in outcomes if o.get("status") == "new" and o.get("summary")]
        for o in outcomes:
            if o.get("status") == "duplicate" and o.get("summary"):
                self.emit('incident_occurrence', o["summary"])
            elif o.get("status") == "failed":
                # The detector fired but the evidence could not be written.
                # Still a stop: the user must know something happened.
                _log.error("an anomaly was detected but could not be recorded: %s", o.get("error"))
                self.emit('incident_capture_failed', {'error': o.get("error")})
        failed = any(o.get("status") == "failed" for o in outcomes)
        if new:
            self.bug_found = new
            self._pause('bug_found', notify=True)
            self.emit('bug_found', {'incidents': new})
        elif failed:
            self._pause('capture_failed', notify=True)

    def _check_close_button(self) -> None:
        """Pumps the window's events - so it stays responsive while paused -
        and honours its X button: hide, pause, keep the session."""
        try:
            closing = self.backend.poll_close_request()
        except Exception:                 # no window to poll: nothing to close
            _log.debug("close-button poll failed", exc_info=True)
            closing = False
        if not closing:
            return
        self._pause('window_closed', notify=True)
        self.backend.hide_window()
