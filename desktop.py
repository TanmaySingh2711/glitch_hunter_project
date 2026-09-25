"""Desktop helpers for run_dashboard.bat: speed on battery, and centred windows.

    python desktop.py center-console     centre the terminal this runs in
    python desktop.py open-dashboard     wait for the dashboard, open it in the
                                         default browser, centre that window
    python desktop.py restore-power-mode put back a power mode the dashboard
                                         changed (started by the dashboard itself)

app.py --desktop (what run_dashboard.bat starts) also calls boost_this_process()
and runs a PowerModeGuard for as long as the dashboard runs.

WHY THE DASHBOARD WAS SLOW ON BATTERY. Measured on the development laptop:
Windows' power mode is "Balanced" on AC but "Best power efficiency" on battery
(ActiveOverlayDcPowerScheme = 961cc777-...), which caps the CPU; and Windows 11
throttles processes that are not in the foreground (EcoQoS) and coarsens
their timers - the dashboard's game loop runs in a console window while the
user watches the browser, so it is exactly such a process. Hence two fixes:

  * boost_this_process() - for this process only, gone when it exits: opt out
    of EcoQoS and of timer coarsening, ask for 1 ms timers, above-normal
    priority. No administrator rights needed.
  * PowerModeGuard - while the dashboard runs, the power mode of the source
    the laptop is on (AC or battery) is "Best performance": set when it starts
    and again whenever the laptop is plugged in or unplugged - never in
    between, so a mode the user picks while it runs is left alone. The
    user's own mode is put back when it stops. Windows only lets a normal user change the
    mode of the CURRENT source, so a mode changed on battery can only be put
    back on battery: if the dashboard stops while plugged in, a small
    background process (restore-power-mode) waits for the next time the
    laptop is on battery, puts it back, and exits. The originals are kept in
    .dashboard_power.json until then, so even a crash cannot lose them.

Everything here is best-effort and Windows-only (the Windows bodies sit
inside `if sys.platform == 'win32':` blocks, as in game_window.py, so mypy
checks them only for Windows): elsewhere, or when a call is refused, it does
nothing and the dashboard runs exactly as before.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from collections.abc import Callable
from typing import Any

import game_window

log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(PROJECT_ROOT, ".dashboard_power.json")
BEST_PERFORMANCE = "ded574b5-45a0-4f42-8737-46345c09c238"
_OVERLAY_KEY = r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes"
_OVERLAY_VALUE = {"ac": "ActiveOverlayAcPowerScheme", "dc": "ActiveOverlayDcPowerScheme"}
GUARD_PERIOD_S = 3.0          # how soon a plug/unplug is noticed while running
RESTORE_PERIOD_S = 15.0       # the background restorer's check interval
PAGE_TITLE = "The Glitch Hunter"          # templates/index.html <title>
# A browser window's title starts with the page's title. Matching only these
# processes, and only a title that STARTS with PAGE_TITLE, keeps other windows
# that merely mention it - an editor with this project open - from moving.
_BROWSERS = {"msedge.exe", "chrome.exe", "brave.exe", "firefox.exe", "opera.exe",
             "vivaldi.exe", "chromium.exe", "arc.exe"}
# Hosts whose window may own a console's stand-in window (Windows Terminal).
_TERMINAL_HOSTS = {"windowsterminal.exe", "openconsole.exe"}
_POWERCFG = os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32", "powercfg.exe")
_STILL_ACTIVE = 259
_QUERY_LIMITED_INFORMATION = 0x1000


# ═══════════════════════════════════════════════════════════════════════
# THIS PROCESS
# ═══════════════════════════════════════════════════════════════════════
def boost_this_process() -> list[str]:
    """What it managed to apply (for the log); [] off Windows."""
    applied: list[str] = []
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes

        class ThrottlingState(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                              ctypes.c_void_p, wintypes.DWORD]
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        proc = k32.GetCurrentProcess()
        # ProcessPowerThrottling (4): EXECUTION_SPEED (1) + IGNORE_TIMER_RESOLUTION
        # (4) in the control mask, neither in the state mask = "never throttle me".
        state = ThrottlingState(1, 0x1 | 0x4, 0)
        try:
            if k32.SetProcessInformation(proc, 4, ctypes.byref(state), ctypes.sizeof(state)):
                applied.append("no EcoQoS throttling")
        except OSError:
            log.debug("SetProcessInformation failed", exc_info=True)
        try:
            if ctypes.WinDLL("winmm").timeBeginPeriod(1) == 0:
                applied.append("1 ms timers")
        except OSError:
            log.debug("timeBeginPeriod failed", exc_info=True)
        if k32.SetPriorityClass(proc, 0x8000):             # ABOVE_NORMAL_PRIORITY_CLASS
            applied.append("above-normal priority")
    return applied


# ═══════════════════════════════════════════════════════════════════════
# POWER MODE
# ═══════════════════════════════════════════════════════════════════════
def power_source() -> str | None:
    """'ac', 'dc' (battery) or None when it cannot be known."""
    if sys.platform == 'win32':
        import ctypes

        class PowerStatus(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte),
                        ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

        s = PowerStatus()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            return {0: "dc", 1: "ac"}.get(s.ACLineStatus)
    return None


def power_mode(source: str) -> str | None:
    """The power mode (overlay GUID) Windows uses on `source`."""
    if sys.platform == 'win32':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _OVERLAY_KEY) as key:
                return str(winreg.QueryValueEx(key, _OVERLAY_VALUE[source])[0]).lower()
        except OSError:
            return None
    else:
        return None


def set_power_mode(guid: str) -> bool:
    """Sets the mode of the CURRENT power source (all a normal user may do)."""
    if sys.platform == 'win32':
        try:
            r = subprocess.run([_POWERCFG, "/overlaysetactive", guid],  # noqa: S603 (fixed argv)
                               capture_output=True, timeout=10,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            return r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            log.debug("powercfg failed", exc_info=True)
            return False
    else:
        return False


def _load_state() -> dict[str, Any]:
    try:
        with open(STATE_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(state: dict[str, Any]) -> None:
    if not state.get("originals"):
        try:
            os.remove(STATE_FILE)
        except OSError:
            pass
        return
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, STATE_FILE)


def _pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    alive = False
    if sys.platform == 'win32':
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(_QUERY_LIMITED_INFORMATION, False, pid)
        if h:
            code = ctypes.c_ulong()
            alive = bool(k32.GetExitCodeProcess(h, ctypes.byref(code)))                 and code.value == _STILL_ACTIVE
            k32.CloseHandle(h)
    else:
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False
    return alive


def restore_due(state: dict[str, Any], source: str | None,
                setter: Callable[[str], bool] = set_power_mode,
                mode: Callable[[str], str | None] = power_mode) -> dict[str, Any]:
    """Puts back the original mode of `source`, if one is owed. Returns the
    state still owed afterwards. A mode the user has chosen since (anything
    but the Best performance this set) is theirs: it is kept, and nothing
    more is owed for that source."""
    originals = dict(state.get("originals", {}))
    if source is not None and source in originals:
        if mode(source) not in (BEST_PERFORMANCE, None):
            del originals[source]
        elif setter(originals[source]):
            log.info("[POWER] %s power mode restored",
                     "battery" if source == "dc" else "plugged-in")
            del originals[source]
    return {**state, "originals": originals}


class PowerModeGuard:
    """Best performance on whichever source the laptop is on, while running;
    the user's own modes back afterwards (see the module docstring)."""

    def __init__(self, period_s: float = GUARD_PERIOD_S,
                 source: Callable[[], str | None] = power_source,
                 mode: Callable[[str], str | None] = power_mode,
                 setter: Callable[[str], bool] = set_power_mode,
                 spawn_restorer: Callable[[], None] | None = None) -> None:
        self.period_s = period_s
        self._source, self._mode, self._setter = source, mode, setter
        self._spawn = spawn_restorer or _spawn_restorer
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._handled_source: str | None = None     # the source already set up
        # Originals still owed from an earlier run are the TRUE originals:
        # what the registry shows now may be that run's own setting.
        self.state: dict[str, Any] = {"originals": dict(_load_state().get("originals", {}))}

    def apply(self) -> None:
        """Sets Best performance for the current power source - once per
        source change, so the user can still pick another mode meanwhile."""
        with self._lock:
            src = self._source()
            if src is None or src == self._handled_source:
                return
            self._handled_source = src
            current = self._mode(src)
            if current is None or current == BEST_PERFORMANCE:
                return
            originals = self.state["originals"]
            first = src not in originals
            if first:
                originals[src] = current
                self.state["owner_pid"] = os.getpid()
                _save_state(self.state)
            if self._setter(BEST_PERFORMANCE) and first:
                log.info("[POWER] %s: power mode set to Best performance while the dashboard "
                         "runs (yours comes back when it stops)",
                         "on battery" if src == "dc" else "plugged in")

    def start(self) -> None:
        if self._thread is not None:
            return
        if self.state["originals"]:
            # Owed from an earlier run: claim them, so a background restorer
            # still waiting from then steps aside instead of racing this guard.
            self.state["owner_pid"] = os.getpid()
            _save_state(self.state)
        self.apply()
        self._thread = threading.Thread(target=self._run, name="power-guard", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.period_s):
            try:
                self.apply()
            except Exception:
                log.debug("power guard failed", exc_info=True)

    def stop(self) -> None:
        """Puts back what it can now; hands the rest to the background
        restorer. Safe to call more than once, from any thread."""
        self._stop.set()
        with self._lock:
            self.state = restore_due(self.state, self._source(), self._setter, self._mode)
            self.state.pop("owner_pid", None)
            _save_state(self.state)
            if self.state["originals"]:
                log.info("[POWER] the battery power mode will be restored the next time the "
                         "laptop runs on battery")
                self._spawn()


def _spawn_restorer() -> None:
    exe = sys.executable
    pyw = os.path.join(os.path.dirname(exe), "pythonw.exe")
    flags = 0
    if sys.platform == 'win32':
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        subprocess.Popen([pyw if os.path.isfile(pyw) else exe,  # noqa: S603 (fixed argv)
                          os.path.abspath(__file__), "restore-power-mode"],
                         cwd=PROJECT_ROOT, close_fds=True, creationflags=flags)
    except OSError:
        log.debug("could not start the power-mode restorer", exc_info=True)


def restore_power_mode_loop() -> None:
    """The background restorer: waits until each changed mode can be put
    back, puts it back, exits. Steps aside while a dashboard owns the state."""
    while True:
        state = _load_state()
        if not state.get("originals"):
            return
        if _pid_alive(state.get("owner_pid")):
            return                       # a running dashboard restores it itself
        state = restore_due(state, power_source(), set_power_mode, power_mode)
        _save_state(state)
        if not state["originals"]:
            return
        time.sleep(RESTORE_PERIOD_S)


_KEEP_ALIVE: list[Any] = []          # ctypes callbacks must outlive their registration


def install_console_close_handler(on_close: Callable[[], None]) -> None:
    """Runs on_close when the console window is closed (its X), logged off
    or shut down - atexit does not run then. Ctrl+C is left to Python."""
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes

        def handler(event: int) -> bool:
            if event in (2, 5, 6):        # CLOSE, LOGOFF, SHUTDOWN
                try:
                    on_close()
                except Exception:
                    log.debug("close handler failed", exc_info=True)
            return False

        callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)(handler)
        _KEEP_ALIVE.append(callback)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(callback, True)


# ═══════════════════════════════════════════════════════════════════════
# CENTRED WINDOWS
# ═══════════════════════════════════════════════════════════════════════
def _process_name(hwnd: int) -> str:
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        pid = wintypes.DWORD()
        ctypes.windll.user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(_QUERY_LIMITED_INFORMATION, False, pid.value)
        if not h:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(len(buf))
            if not k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                return ""
            return os.path.basename(buf.value).lower()
        finally:
            k32.CloseHandle(h)
    else:
        return ""


def center_window(hwnd: int | None) -> tuple[int, int] | None:
    """Centres a top-level window on the cursor's monitor (like the game
    window). A maximised or minimised window is left alone."""
    if sys.platform == 'win32' and hwnd:
        import ctypes
        from ctypes import wintypes
        u = ctypes.windll.user32
        h = wintypes.HWND(hwnd)
        if u.IsZoomed(h) or u.IsIconic(h):
            return None
        area = game_window.current_work_area()
        r = wintypes.RECT()
        if area is None or not u.GetWindowRect(h, ctypes.byref(r)):
            return None
        x, y = game_window.centered_origin(area, (r.right - r.left, r.bottom - r.top))
        if not u.SetWindowPos(h, None, x, y, 0, 0,
                              game_window.SWP_NOSIZE | game_window.SWP_NOZORDER):
            return None
        return (x, y)
    return None


def console_window() -> int | None:
    """The terminal window this process runs in (Windows Terminal or the
    classic console) - never an editor that hosts a terminal (VS Code)."""
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        u = ctypes.windll.user32
        k32 = ctypes.windll.kernel32
        k32.GetConsoleWindow.restype = wintypes.HWND
        u.GetAncestor.restype = wintypes.HWND
        con = k32.GetConsoleWindow()
        root = (u.GetAncestor(con, 3) or con) if con else None   # GA_ROOTOWNER
        # The classic console: the console window IS the window. (Windows
        # reports it as belonging to the program attached to it - cmd.exe -
        # not to conhost, so it is recognised by owning no other window.)
        # Windows Terminal: the console window is a hidden stand-in owned by
        # the real terminal window. Any other owner - an editor hosting a
        # terminal, like VS Code - is never moved.
        if root and u.IsWindowVisible(root) and (
                root == con or _process_name(root) in _TERMINAL_HOSTS):
            return int(root)
    return None


def find_window(title_prefix: str, processes: set[str] | None = None) -> int | None:
    """A visible top-level window whose title starts with title_prefix
    (and, if given, that belongs to one of `processes`)."""
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        u = ctypes.windll.user32
        found: list[int] = []

        def visit(hwnd: int, _lparam: int) -> bool:
            if u.IsWindowVisible(hwnd):
                buf = ctypes.create_unicode_buffer(512)
                u.GetWindowTextW(hwnd, buf, 512)
                if buf.value.startswith(title_prefix) and (
                        processes is None or _process_name(hwnd) in processes):
                    found.append(int(hwnd))
                    return False
            return True

        u.EnumWindows(ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(visit), 0)
        if found:
            return found[0]
    return None


def dashboard_url(environ: Any = os.environ) -> str:
    return f"http://localhost:{environ.get('GLITCH_HUNTER_PORT', '5000')}"


def wait_for_dashboard(url: str, timeout_s: float = 300.0) -> bool:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(f"{url}/healthz", timeout=2) as r:  # noqa: S310 (localhost)
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(0.3)
    return False


def open_dashboard(url: str | None = None, window_wait_s: float = 20.0,
                   settle_s: float = 3.0) -> bool:
    """Opens the dashboard the moment it answers, in the default browser,
    and centres that browser window once it shows the page.

    A browser puts a new window back where its last one was a moment AFTER
    showing it (measured with Edge: centred, then moved back to 10,10), so
    the window is kept centred for `settle_s` after it first appears."""
    url = url or dashboard_url()
    if not wait_for_dashboard(url):
        return False
    webbrowser.open(url)
    end = time.monotonic() + window_wait_s
    while time.monotonic() < end:
        hwnd = find_window(PAGE_TITLE, _BROWSERS)
        if hwnd:
            settled = time.monotonic() + settle_s
            while time.monotonic() < settled:
                center_window(hwnd)      # a no-op once centred; never moves a maximised one
                time.sleep(0.25)
            return True
        time.sleep(0.25)
    return True


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else ""
    if cmd == "center-console":
        center_window(console_window())
        return 0
    if cmd == "open-dashboard":
        return 0 if open_dashboard(argv[1] if len(argv) > 1 else None) else 1
    if cmd == "restore-power-mode":
        restore_power_mode_loop()
        return 0
    sys.stderr.write((__doc__ or "").split("\n\n", 2)[1] + "\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
