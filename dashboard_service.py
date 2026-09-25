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
  Start   put the window in the taskbar, minimised (created or re-shown,
          centred for when the user clicks it; an already-open one is left
          as it is) and resume the SAME session.
  Stop    pause. The window stays exactly as it is - never minimised,
          hidden or closed.
  X       (the window's own close button, while running or paused) hide the
          window and pause. The session is kept; Start brings it back.
  Reset   end the session and close the window; the next Start is fresh.
          The Bug Tracker starts a new session list too (every incident
          stays saved in the store).
  Game    (the "Select Game Environment" list) a Reset, then the other game
          variant is loaded in place of this one; the next Start plays it.
  Page    opening or refreshing the page is a Reset (app.py page_opened).
          A closed tab or a dropped connection only pauses, and changes
          nothing else.

Pausing or closing never touches the agent's state: the session is a
suspended generator, so nothing about the episode, the policy or its
telemetry is discarded, and the dashboard writes no checkpoint, coverage or
lifecycle file at all.

BUG FOUND (Objective 3). When a step reports a bug - a new incident or a
repeat sighting of a known one - this thread pauses before it takes another
step - the evidence is already on disk by then
(reporting/pipeline.capture ran inside that step) - and remembers it as
`bug_found` until the user presses Start (resume) or Reset. Nothing resumes
by itself, not even when the reports finish rendering. A later sighting of an
incident already recorded is counted but does not pause.

RUN END (clean game only). When a run of the clean game ends - Mario reaches
the castle, dies, runs out of time, or the agent gets stuck - testing stops
and the result is kept as `run_result` until the user presses Start or Reset.
A run that reached the castle also has its run report ("no bugs found" when
no detector fired; reporting/run_report.py). Start plays the next run; Reset
clears the dashboard first. The bugged game plays on from one run to the
next, as before, and stops only on bugs.
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
# The dashboard's pace: one agent step every two 60 fps frames (30 steps/s)
# - what a step that fits in one frame always got. It used to be "wait as
# long again as the step took", which halved the rate again whenever a step
# ran long: measured on battery, a 22 ms step gave 44 ms per step (19/s)
# instead of 33 ms. A fixed period keeps battery at the plugged-in pace.
STEP_PERIOD_S = 2 * TARGET_FRAME_S
MIN_IDLE_S = 0.004          # always this much free time after a step, for clicks and the window
# Windows' default timer tick. A timed wait on a queue rounds UP to it
# (measured: an 18 ms wait took 31 ms), and time.monotonic() only advances in
# steps of it (GetTickCount64) - so step times read as 0, 15.6 or 31.2 ms and
# the dashboard ran at about a third of game speed instead of the intended
# half. Pacing therefore uses time.perf_counter() (0.1 us) and _next_command.
TIMER_TICK_S = 0.0156
CLICK_POLL_S = 0.004        # while sleeping to a step: how often a click is checked for
_clock = time.perf_counter

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
    def begin_incident_session(self) -> None: ...
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
        # How the last clean-game run ended (and its report), until Start or Reset.
        self.run_result: dict[str, Any] | None = None
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
                "bug_found": self.bug_found, "run_result": self.run_result}

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
            cmd = (self._next_command(next_step_at) if self.testing
                   else self._command_within(IDLE_PUMP_S))
            if cmd == 'shutdown':
                self._close_session()
                return
            if isinstance(cmd, threading.Event):
                cmd.set()
                continue
            if cmd is not None:
                self._handle(cmd)
                continue
            if self.testing and _clock() >= next_step_at:
                t0 = _clock()
                self._step()
                # One step per STEP_PERIOD_S; a step that overruns it is
                # followed by MIN_IDLE_S before the next.
                next_step_at = max(t0 + STEP_PERIOD_S, _clock() + MIN_IDLE_S)
            self._check_close_button()

    def _command_within(self, timeout: float) -> Command | None:
        try:
            return self._commands.get(timeout=timeout)
        except queue.Empty:
            return None

    def _next_command(self, deadline: float) -> Command | None:
        """The next command, or None when it is time for the next step.

        A timed queue wait can overrun by a whole timer tick, so it is used
        only while more than two ticks remain; the rest is slept in short
        time.sleep slices (high-resolution on Windows since Python 3.11) with
        the queue checked after each. Steps land on time, and a click still
        lands before the next step, within CLICK_POLL_S."""
        while True:
            remaining = deadline - _clock()
            if remaining > 2 * TIMER_TICK_S:
                cmd = self._command_within(remaining - 2 * TIMER_TICK_S)
            else:
                if remaining > 0:
                    time.sleep(min(remaining, CLICK_POLL_S))
                try:
                    cmd = self._commands.get_nowait()
                except queue.Empty:
                    cmd = None
            if cmd is not None or _clock() >= deadline:
                return cmd

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
                self._clear_run_result()
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
            self._clear_run_result()
            self.pause_reason = 'reset'
            try:
                self.backend.begin_incident_session()
            except Exception:             # the reset itself has already happened
                _log.exception("could not start a new incident session view")
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
        if was_testing or (self.bug_found is None and self.run_result is None):
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
        if item.get('run_end'):
            self._handle_run_end(item['run_end'])

    def _handle_run_end(self, run_end: dict[str, Any]) -> None:
        """A clean-game run ended: stop, and keep how it ended until the user
        starts the next run or resets. A bug found on the same step keeps the
        bug as the pause reason; the run result is still recorded."""
        self.run_result = run_end
        reason = {'level_complete': 'level_complete', 'death': 'mario_died',
                  'timeout': 'mario_died'}.get(str(run_end.get('end_reason')), 'run_ended')
        if self.testing:
            self._pause(reason, notify=True)
        self.emit('run_finished', run_end)

    def _clear_run_result(self) -> None:
        if self.run_result is not None:
            self.run_result = None
            self.emit('run_cleared', {})

    def _handle_incidents(self, outcomes: Any) -> None:
        """Every recorded bug sighting stops testing, here, before another
        step runs - a new incident, or a repeat of a known one (which only
        raises that incident's count; no second incident is created)."""
        new = [o["summary"] for o in outcomes
               if o.get("status") in ("new", "duplicate") and o.get("summary")]
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
