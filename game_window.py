"""Where the game window appears, and the window-manager calls pygame lacks.

Every platform-specific window operation in the project lives here, so the
rest can be tested without a physical display: each function degrades to a
harmless no-op (returning None/False) when there is no window, no window
manager, or no Win32.

PLACEMENT RULE. A game window is centred on the display the user is working
on - the monitor under the mouse cursor, inside its work area (so never under
the taskbar) - at the moment the window is CREATED OR RE-SHOWN. Never
afterwards: an open window is never pulled back to the centre, so the user can
move it wherever they like.

Two layers, because SDL can only centre on its own default display at
creation: request_centered_creation() asks SDL for a centred window (the
cross-platform fallback, and it avoids a flash at a stale position), then
center_on_current_display() moves the finished window onto the cursor's
monitor. On Windows the second step uses GetWindowRect, so it centres the
whole frame - title bar included - not just the client area.
"""
from __future__ import annotations

import functools
import logging
import os
import sys
from typing import Any

Area = tuple[int, int, int, int]        # (left, top, right, bottom)

# Every call here is best-effort (see the module docstring), so a failure is
# a DEBUG record - visible with GLITCH_HUNTER_LOG_LEVEL=DEBUG - never an error.
_log = logging.getLogger(__name__)

MONITOR_DEFAULTTONEAREST = 2
SWP_NOSIZE, SWP_NOZORDER, SWP_NOACTIVATE = 0x0001, 0x0004, 0x0010


def centered_origin(area: Area, size: tuple[int, int]) -> tuple[int, int]:
    """Top-left (x, y) that centres a `size` (w, h) window in `area`
    (left, top, right, bottom). A window larger than the area is pinned to
    the area's top-left, so its title bar stays reachable."""
    left, top, right, bottom = area
    w, h = size
    return (left + max(0, (right - left - w) // 2),
            top + max(0, (bottom - top - h) // 2))


def request_centered_creation() -> None:
    """Call before SDL creates a window. Clears any explicit position a
    previous caller left in the environment (it would win over centring)."""
    os.environ.pop('SDL_VIDEO_WINDOW_POS', None)
    os.environ['SDL_VIDEO_CENTERED'] = '1'


def _hwnd() -> int | None:
    try:
        import pygame as pg
        if pg.display.get_surface() is None:
            return None
        hwnd = pg.display.get_wm_info().get('window')
        return int(hwnd) if hwnd else None
    except Exception:
        _log.debug("window-manager call failed", exc_info=True)
        return None


@functools.cache
def _user32() -> Any:
    """A private user32 handle, so declaring argtypes here cannot change how
    any other module's ctypes.windll.user32 calls behave."""
    import ctypes
    from ctypes import wintypes
    u = ctypes.WinDLL('user32', use_last_error=True)
    u.MonitorFromPoint.argtypes = [wintypes.POINT, wintypes.DWORD]
    u.MonitorFromPoint.restype = wintypes.HMONITOR
    u.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.c_void_p]
    u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    u.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
    u.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
    u.IsIconic.argtypes = [wintypes.HWND]
    u.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    return u


def current_work_area() -> Area | None:
    """(left, top, right, bottom) of the work area of the monitor under the
    cursor, or None where that cannot be known."""
    if sys.platform != 'win32':
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class MONITORINFO(ctypes.Structure):
            _fields_ = [('cbSize', wintypes.DWORD), ('rcMonitor', wintypes.RECT),
                        ('rcWork', wintypes.RECT), ('dwFlags', wintypes.DWORD)]

        u = _user32()
        pt = wintypes.POINT()
        if not u.GetCursorPos(ctypes.byref(pt)):
            return None
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        if not u.GetMonitorInfoW(u.MonitorFromPoint(pt, MONITOR_DEFAULTTONEAREST),
                                 ctypes.byref(info)):
            return None
        r = info.rcWork
        return (r.left, r.top, r.right, r.bottom)
    except Exception:
        _log.debug("window-manager call failed", exc_info=True)
        return None


def center_on_current_display() -> tuple[int, int] | None:
    """Moves the pygame window so it is centred on the cursor's monitor.
    Returns the new top-left, or None if there was nothing to move. Does not
    activate the window or change its size or z-order."""
    hwnd = _hwnd()
    area = current_work_area()
    if not hwnd or area is None:
        return None
    try:
        import ctypes
        from ctypes import wintypes
        u = _user32()
        r = wintypes.RECT()
        if not u.GetWindowRect(hwnd, ctypes.byref(r)):
            return None
        x, y = centered_origin(area, (r.right - r.left, r.bottom - r.top))
        if not u.SetWindowPos(hwnd, None, x, y, 0, 0,
                              SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE):
            return None
        return (x, y)
    except Exception:
        _log.debug("window-manager call failed", exc_info=True)
        return None


# Every Window wrapper ever handed out, with the display Surface it belongs
# to. NEVER released: pygame 2.6's Window.from_display_module() stores a raw
# pointer to the wrapper in the SDL window's data ("pg_window") without
# holding a reference, and pygame dereferences it for every later window
# event. A wrapper freed while its window lives turned the next show/focus/
# minimise event into an access violation - measured, in the dashboard, at
# random points in the engine. One wrapper per window ever created costs
# nothing.
_SDL_WINDOWS: list[tuple[Any, Any]] = []


def _sdl_window() -> Any:
    try:
        import pygame as pg
        from pygame._sdl2.video import Window
        surface: pg.Surface | None = pg.display.get_surface()
        if surface is None:
            return None
        for owner, win in _SDL_WINDOWS:
            if owner is surface:
                return win
        win = Window.from_display_module()
        _SDL_WINDOWS.append((surface, win))
        return win
    except Exception:
        _log.debug("window-manager call failed", exc_info=True)
        return None


def hide() -> bool:
    """Takes the window off the screen (and the taskbar) WITHOUT destroying
    it, so everything drawn into it and every Surface stays valid."""
    if _hwnd() is None:
        return False
    w = _sdl_window()
    if w is None:
        return False
    try:
        w.hide()
        return True
    except Exception:
        _log.debug("window-manager call failed", exc_info=True)
        return False


def show() -> bool:
    """Puts a hidden window back on screen; restores it if minimised."""
    if _hwnd() is None:
        return False
    w = _sdl_window()
    if w is None:
        return False
    try:
        w.show()
        if is_minimized():
            w.restore()
        return True
    except Exception:
        _log.debug("window-manager call failed", exc_info=True)
        return False


def is_minimized() -> bool:
    hwnd = _hwnd()
    if not hwnd or sys.platform != 'win32':
        return False
    try:
        return bool(_user32().IsIconic(hwnd))
    except Exception:
        _log.debug("window-manager call failed", exc_info=True)
        return False


def bring_to_front() -> bool:
    """Best-effort foreground request (Windows only). A refusal - Windows'
    focus-stealing rules - must never break playback."""
    hwnd = _hwnd()
    if not hwnd or sys.platform != 'win32':
        return False
    try:
        import ctypes
        return bool(ctypes.windll.user32.SetForegroundWindow(hwnd))
    except Exception:
        _log.debug("window-manager call failed", exc_info=True)
        return False
