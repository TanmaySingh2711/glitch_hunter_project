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
  A browser refresh / closed tab pauses, and changes nothing else.

Pausing or closing never touches the agent's state: the session is a
suspended generator, so nothing about the episode, the policy or its
telemetry is discarded, and the dashboard writes no checkpoint, coverage or
lifecycle file at all.
"""
import queue
import threading
import time
import traceback

THREAD_NAME = "game-window"
IDLE_PUMP_S = 0.05          # event-pump period while paused: keeps the window responsive
TARGET_FRAME_S = 1.0 / 60.0


class GameWindowService:
    """`backend` does the real work (see agent_logic.DashboardBackend):
        preload(), open_window() -> str, hide_window(), close_window(),
        poll_close_request() -> bool, new_session() -> iterator, stop_audio()
    `emit(event, payload)` talks to the browser."""

    def __init__(self, backend, emit, log=print):
        self.backend = backend
        self.emit = emit
        self.log = log
        self._commands = queue.Queue()
        self._thread = None
        self._ready = threading.Event()
        self._preload_error = None
        # Read from other threads (healthz, tests); written only by the game thread.
        self.testing = False
        self.session = None
        self.steps = 0
        self.last_pause_reason = None

    # ── called from any thread ────────────────────────────────────────────
    def start(self, timeout=None):
        """Starts the game thread and waits for its pre-load. Idempotent."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name=THREAD_NAME, daemon=True)
            self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("the game thread did not finish pre-loading")
        if self._preload_error is not None:
            raise RuntimeError("pre-load failed") from self._preload_error

    def start_testing(self):
        self._commands.put('start')

    def stop_testing(self):
        self._commands.put('stop')

    def reset(self):
        self._commands.put('reset')

    def client_connected(self):
        self._commands.put('connect')

    def client_disconnected(self):
        self._commands.put('disconnect')

    def shutdown(self, timeout=5.0):
        self._commands.put('shutdown')
        if self._thread is not None:
            self._thread.join(timeout)

    def wait_idle(self, timeout=5.0):
        """Blocks until every command posted so far has been handled."""
        done = threading.Event()
        self._commands.put(done)
        return done.wait(timeout)

    @property
    def alive(self):
        return self._thread is not None and self._thread.is_alive()

    # ── the game thread ───────────────────────────────────────────────────
    def _run(self):
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

    def _handle(self, cmd):
        if cmd == 'start':
            self.log(">>> START_TESTING received!")
            try:
                self.backend.open_window()
                if self.session is None:
                    self.session = self.backend.new_session()
                self.testing = True
                self.last_pause_reason = None
            except Exception:
                traceback.print_exc()
                self._pause('error', notify=True)
        elif cmd in ('stop', 'connect', 'disconnect'):
            self._pause(cmd)
        elif cmd == 'reset':
            self._pause('reset')
            self._close_session()
            self.backend.close_window()

    def _pause(self, reason, notify=False):
        """Stops stepping. Touches nothing else - not the window, not the
        session. `notify` tells the browser, for pauses it did not ask for."""
        self.testing = False
        self.last_pause_reason = reason
        try:
            self.backend.stop_audio()
        except Exception:
            pass
        if notify:
            self.emit('testing_paused', {'reason': reason})

    def _close_session(self):
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass
            self.session = None
            self.steps = 0

    def _step(self):
        try:
            item = next(self.session)
        except StopIteration:
            self.session = None
            self._pause('session_ended', notify=True)
            return
        except Exception as exc:
            self.log(f"Error in the game loop: {exc}")
            traceback.print_exc()
            self._pause('error', notify=True)
            return
        self.steps += 1
        self.emit('video_frame', {'frame': item['frame']})
        if item.get('log'):
            self.emit('agent_log', {'log': item['log']})

    def _check_close_button(self):
        """Pumps the window's events - so it stays responsive while paused -
        and honours its X button: hide, pause, keep the session."""
        try:
            closing = self.backend.poll_close_request()
        except Exception:
            closing = False
        if not closing:
            return
        self._pause('window_closed', notify=True)
        self.backend.hide_window()
