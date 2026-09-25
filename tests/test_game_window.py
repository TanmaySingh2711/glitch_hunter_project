"""game_window.py's show / hide / minimise calls, through recording fakes.

The suite runs headless (SDL's dummy driver has no OS window), so the real
calls are exercised against stand-ins for the Win32 handle, user32 and the
SDL window wrapper; tests/test_dashboard_control.py covers the same calls on
a real window where the platform has one.
"""
import sys

import pytest

import game_window


class FakeUser32:
    def __init__(self, iconic=False):
        self.calls = []
        self.iconic = iconic

    def ShowWindow(self, hwnd, cmd):
        self.calls.append(("ShowWindow", hwnd, cmd))
        self.iconic = cmd == game_window.SW_SHOWMINNOACTIVE
        return 1

    def IsIconic(self, hwnd):
        return self.iconic


class FakeSDLWindow:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a: self.calls.append(name)


@pytest.fixture
def fakes(monkeypatch):
    user32, sdl = FakeUser32(), FakeSDLWindow()
    monkeypatch.setattr(game_window, "_hwnd", lambda: 4321)
    monkeypatch.setattr(game_window, "_user32", lambda: user32)
    monkeypatch.setattr(game_window, "_sdl_window", lambda: sdl)
    return user32, sdl


def test_show_minimized_puts_the_window_in_the_taskbar_without_activating_it(fakes):
    user32, sdl = fakes
    assert game_window.show_minimized() is True
    if sys.platform == "win32":
        assert user32.calls == [("ShowWindow", 4321, game_window.SW_SHOWMINNOACTIVE)]
        assert game_window.is_minimized() is True
    else:
        assert sdl.calls == ["show", "minimize"]


def test_show_restores_a_window_minimised_behind_sdls_back(fakes):
    """show_minimized() minimises through Win32, which SDL may not have seen
    yet; restoring must not depend on SDL knowing."""
    user32, sdl = fakes
    user32.iconic = True
    assert game_window.show() is True
    assert sdl.calls[0] == "show"
    if sys.platform == "win32":
        assert ("ShowWindow", 4321, game_window.SW_RESTORE) in user32.calls
        assert game_window.is_minimized() is False
    else:                                   # minimised-ness is only known on Win32
        assert sdl.calls == ["show"]


def test_show_leaves_a_visible_window_alone(fakes):
    user32, sdl = fakes
    assert game_window.show() is True
    assert sdl.calls == ["show"] and user32.calls == []


def test_hide_goes_through_sdl(fakes):
    _user32, sdl = fakes
    assert game_window.hide() is True and sdl.calls == ["hide"]


def test_a_failing_window_call_is_reported_not_raised(fakes, monkeypatch):
    user32, _sdl = fakes

    def boom(*_a):
        raise OSError("window gone")
    monkeypatch.setattr(user32, "ShowWindow", boom)
    monkeypatch.setattr(game_window, "_sdl_window", lambda: type("W", (), {
        "show": boom, "hide": boom, "minimize": boom, "restore": boom})())
    assert game_window.show_minimized() is False
    assert game_window.hide() is False
    assert game_window.show() is False


def test_without_a_window_everything_is_a_no_op(monkeypatch):
    monkeypatch.setattr(game_window, "_hwnd", lambda: None)
    assert game_window.show_minimized() is False
    assert game_window.show() is False and game_window.hide() is False
    assert game_window.is_minimized() is False


# ── the real Win32 calls, on a real (tkinter) window ─────────────────────────
@pytest.fixture
def tk_window():
    if sys.platform != "win32":
        pytest.skip("Win32 only")
    import tkinter
    try:
        root = tkinter.Tk()
    except tkinter.TclError:
        pytest.skip("no display for a test window")
    root.title("game_window test")
    root.geometry("300x200+0+0")
    root.update()
    hwnd = int(root.frame(), 16)
    import ctypes
    top = ctypes.windll.user32.GetAncestor(hwnd, 2) or hwnd       # GA_ROOT
    yield root, top
    root.destroy()


def test_the_cursor_monitors_work_area_is_real():
    if sys.platform != "win32":
        pytest.skip("Win32 only")
    area = game_window.current_work_area()
    assert area is not None
    left, top, right, bottom = area
    assert right > left and bottom > top


def test_a_real_window_is_centred_minimised_restored_and_brought_forward(tk_window,
                                                                        monkeypatch):
    root, hwnd = tk_window
    monkeypatch.setattr(game_window, "_hwnd", lambda: hwnd)
    area = game_window.current_work_area()
    origin = game_window.center_on_current_display()
    assert origin is not None
    root.update()
    assert area[0] <= origin[0] <= area[2] and area[1] <= origin[1] <= area[3]
    monkeypatch.setattr(game_window, "_sdl_window", lambda: type("W", (), {
        "show": lambda self: None, "restore": lambda self: None})())
    assert game_window.show_minimized() is True
    root.update()
    assert game_window.is_minimized() is True
    assert game_window.show() is True                 # restores through Win32
    root.update()
    assert game_window.is_minimized() is False
    assert isinstance(game_window.bring_to_front(), bool)
