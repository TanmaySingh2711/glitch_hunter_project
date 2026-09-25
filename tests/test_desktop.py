"""desktop.py: run_dashboard.bat's speed on battery and its centred windows.

The power-mode guard is driven through fakes for the power source, the
current mode and the setter, so these tests never change this machine's
power settings; the state file lives in a temporary folder.
"""
import json
import sys

import pytest

import desktop

ORIGINAL_AC, ORIGINAL_DC = "00000000-0000-0000-0000-000000000000", "961cc777-battery-saver"


class Laptop:
    """A fake laptop: which source it is on, and the mode of each source."""

    def __init__(self, source="ac"):
        self.source = source
        self.modes = {"ac": ORIGINAL_AC, "dc": ORIGINAL_DC}
        self.spawned = 0

    def set(self, guid):
        self.modes[self.source] = guid          # Windows: current source only
        return True

    def guard(self):
        return desktop.PowerModeGuard(period_s=60, source=lambda: self.source,
                                      mode=lambda s: self.modes[s], setter=self.set,
                                      spawn_restorer=self._spawn)

    def _spawn(self):
        self.spawned += 1


@pytest.fixture(autouse=True)
def state_file(tmp_path, monkeypatch):
    path = tmp_path / "power.json"
    monkeypatch.setattr(desktop, "STATE_FILE", str(path))
    return path


def test_the_guard_sets_best_performance_and_puts_the_original_back(state_file):
    laptop = Laptop("dc")
    guard = laptop.guard()
    guard.apply()
    assert laptop.modes["dc"] == desktop.BEST_PERFORMANCE
    saved = json.loads(state_file.read_text())
    assert saved["originals"] == {"dc": ORIGINAL_DC}, "the original was not kept on disk"
    guard.apply()                                    # idempotent while running
    assert laptop.modes["dc"] == desktop.BEST_PERFORMANCE
    guard.stop()
    assert laptop.modes == {"ac": ORIGINAL_AC, "dc": ORIGINAL_DC}
    assert not state_file.exists() and laptop.spawned == 0


def test_plugging_in_while_running_covers_both_sources(state_file):
    laptop = Laptop("dc")
    guard = laptop.guard()
    guard.apply()
    laptop.source = "ac"
    guard.apply()
    assert laptop.modes == {"ac": desktop.BEST_PERFORMANCE, "dc": desktop.BEST_PERFORMANCE}
    guard.stop()                                     # on AC: only AC can be put back now
    assert laptop.modes["ac"] == ORIGINAL_AC
    assert json.loads(state_file.read_text())["originals"] == {"dc": ORIGINAL_DC}
    assert laptop.spawned == 1, "nothing was left to put the battery mode back"
    laptop.source = "dc"                             # the restorer's turn
    left = desktop.restore_due(json.loads(state_file.read_text()), "dc", laptop.set,
                               lambda s: laptop.modes[s])
    assert laptop.modes["dc"] == ORIGINAL_DC and left["originals"] == {}


def test_a_new_run_restores_the_true_original_not_the_last_runs_setting(state_file):
    state_file.write_text(json.dumps({"originals": {"dc": ORIGINAL_DC}}))
    laptop = Laptop("dc")
    laptop.modes["dc"] = desktop.BEST_PERFORMANCE    # left over from the earlier run
    guard = laptop.guard()
    guard.start()
    try:
        assert json.loads(state_file.read_text())["owner_pid"] > 0, "the owed state not claimed"
    finally:
        guard.stop()
    assert laptop.modes["dc"] == ORIGINAL_DC


def test_a_mode_the_user_picks_while_it_runs_is_left_alone(state_file):
    laptop = Laptop("dc")
    guard = laptop.guard()
    guard.apply()
    laptop.modes["dc"] = "user-picked-balanced"      # the user changes it meanwhile
    guard.apply()
    guard.apply()
    assert laptop.modes["dc"] == "user-picked-balanced", "the guard overrode the user"
    laptop.source = "ac"                             # plugged in: a new source is set up
    guard.apply()
    assert laptop.modes["ac"] == desktop.BEST_PERFORMANCE
    laptop.source = "dc"
    guard.stop()                                     # the user's own pick is not undone
    assert laptop.modes["dc"] == "user-picked-balanced"
    assert json.loads(state_file.read_text())["originals"] == {"ac": ORIGINAL_AC}


def test_a_user_already_on_best_performance_is_left_alone(state_file):
    laptop = Laptop("ac")
    laptop.modes["ac"] = desktop.BEST_PERFORMANCE
    guard = laptop.guard()
    guard.apply()
    guard.stop()
    assert laptop.modes["ac"] == desktop.BEST_PERFORMANCE and not state_file.exists()


def test_an_unknown_source_changes_nothing(state_file):
    laptop = Laptop()
    guard = desktop.PowerModeGuard(source=lambda: None, mode=lambda s: "x", setter=laptop.set)
    guard.apply()
    assert laptop.modes == {"ac": ORIGINAL_AC, "dc": ORIGINAL_DC}


def test_a_machine_without_power_modes_is_left_alone(state_file):
    # Windows Server and some desktops have no power modes: nothing to read.
    laptop = Laptop()
    guard = desktop.PowerModeGuard(source=lambda: "ac", mode=lambda s: None, setter=laptop.set,
                                   spawn_restorer=laptop._spawn)
    guard.apply()
    guard.stop()
    assert laptop.modes == {"ac": ORIGINAL_AC, "dc": ORIGINAL_DC}
    assert not state_file.exists() and laptop.spawned == 0


def test_the_restorer_steps_aside_for_a_running_dashboard(state_file, monkeypatch):
    import os
    state_file.write_text(json.dumps({"originals": {"dc": "x"}, "owner_pid": os.getpid()}))
    calls = []
    monkeypatch.setattr(desktop, "power_source", lambda: calls.append(1) or "dc")
    desktop.restore_power_mode_loop()
    assert calls == [], "it raced the dashboard that owns the state"


def test_the_restorer_puts_the_mode_back_and_exits(state_file, monkeypatch):
    state_file.write_text(json.dumps({"originals": {"dc": ORIGINAL_DC}}))
    laptop = Laptop("dc")
    laptop.modes["dc"] = desktop.BEST_PERFORMANCE    # what the dashboard left
    monkeypatch.setattr(desktop, "power_source", lambda: "dc")
    monkeypatch.setattr(desktop, "set_power_mode", laptop.set)
    monkeypatch.setattr(desktop, "power_mode", lambda s: laptop.modes[s])
    desktop.restore_power_mode_loop()
    assert laptop.modes["dc"] == ORIGINAL_DC and not state_file.exists()


def test_the_restorer_keeps_a_mode_the_user_chose_meanwhile(state_file, monkeypatch):
    state_file.write_text(json.dumps({"originals": {"dc": ORIGINAL_DC}}))
    laptop = Laptop("dc")
    laptop.modes["dc"] = "user-picked"
    monkeypatch.setattr(desktop, "power_source", lambda: "dc")
    monkeypatch.setattr(desktop, "set_power_mode", laptop.set)
    monkeypatch.setattr(desktop, "power_mode", lambda s: laptop.modes[s])
    desktop.restore_power_mode_loop()
    assert laptop.modes["dc"] == "user-picked" and not state_file.exists()


def test_this_process_boost_is_harmless_to_call():
    applied = desktop.boost_this_process()
    assert isinstance(applied, list)
    if sys.platform != "win32":
        assert applied == []


def test_window_helpers_degrade_to_no_ops():
    assert desktop.center_window(None) is None
    assert desktop.find_window("no window has this title 7f3a9c") is None
    assert desktop.dashboard_url({"GLITCH_HUNTER_PORT": "5091"}) == "http://localhost:5091"
    assert desktop.dashboard_url({}) == "http://localhost:5000"


def test_the_dashboard_is_not_opened_until_it_answers(monkeypatch):
    opened = []
    monkeypatch.setattr(desktop.webbrowser, "open", opened.append)
    assert desktop.wait_for_dashboard("http://127.0.0.1:9", timeout_s=0.5) is False
    monkeypatch.setattr(desktop, "wait_for_dashboard", lambda url: False)
    assert desktop.open_dashboard("http://127.0.0.1:9") is False and opened == []
    monkeypatch.setattr(desktop, "wait_for_dashboard", lambda url: True)
    monkeypatch.setattr(desktop, "find_window", lambda title, processes=None: None)
    assert desktop.open_dashboard("http://127.0.0.1:9", window_wait_s=0) is True
    assert opened == ["http://127.0.0.1:9"]
    centred = []
    monkeypatch.setattr(desktop, "find_window", lambda title, processes=None: 42)
    monkeypatch.setattr(desktop, "center_window", centred.append)
    assert desktop.open_dashboard("http://127.0.0.1:9", settle_s=0.6) is True
    assert centred and set(centred) == {42}, "the browser window was not kept centred"


def test_the_command_line(monkeypatch):
    monkeypatch.setattr(desktop, "console_window", lambda: None)
    assert desktop.main(["center-console"]) == 0
    monkeypatch.setattr(desktop, "open_dashboard", lambda url=None: url == "u")
    assert desktop.main(["open-dashboard", "u"]) == 0
    monkeypatch.setattr(desktop, "restore_power_mode_loop", lambda: None)
    assert desktop.main(["restore-power-mode"]) == 0
    assert desktop.main([]) == 2


# ── the real Windows calls (read-only, or setting a value to itself) ─────────
win32 = pytest.mark.skipif(sys.platform != "win32", reason="Windows APIs")


@win32
def test_the_power_source_and_its_mode_can_be_read():
    source = desktop.power_source()
    assert source in ("ac", "dc")
    mode = desktop.power_mode(source)
    if mode is None:
        # Windows Server (the CI runner) has no power modes at all; the
        # guard then leaves the machine alone (test_the_guard_* above).
        pytest.skip("this Windows edition has no power mode")
    assert len(mode) == 36
    # Setting the mode it already has changes nothing on this machine.
    assert desktop.set_power_mode(mode) is True
    assert desktop.power_mode(source) == mode


@win32
def test_a_process_is_known_alive_only_while_it_runs():
    import os
    import subprocess
    assert desktop._pid_alive(os.getpid()) is True
    assert desktop._pid_alive(0) is False and desktop._pid_alive("x") is False
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert desktop._pid_alive(p.pid) is False


@win32
def test_a_real_window_is_found_by_its_title_and_centred():
    import tkinter
    try:
        root = tkinter.Tk()
    except tkinter.TclError:
        pytest.skip("no display for a test window")
    try:
        root.title("The Glitch Hunter - desktop test window")
        root.geometry("420x260+0+0")
        root.update()
        hwnd = desktop.find_window("The Glitch Hunter - desktop test")
        assert hwnd, "the window was not found by its title prefix"
        assert desktop.find_window("desktop test window") is None, "matched mid-title"
        assert desktop.find_window("The Glitch Hunter - desktop test", {"msedge.exe"}) is None
        assert desktop._process_name(hwnd) in ("python.exe", "pythonw.exe")
        area = desktop.game_window.current_work_area()
        origin = desktop.center_window(hwnd)
        if area is None:
            assert origin is None
            return
        assert origin is not None
        root.update()
        w, h = root.winfo_width(), root.winfo_height()
        assert area[0] <= origin[0] <= area[2] and area[1] <= origin[1] <= area[3]
        assert w and h
    finally:
        root.destroy()


@win32
def test_the_console_lookup_and_close_handler_are_safe_here():
    hwnd = desktop.console_window()
    assert hwnd is None or isinstance(hwnd, int)
    desktop.install_console_close_handler(lambda: None)
    assert desktop._KEEP_ALIVE, "the callback must stay referenced while registered"


def test_the_restorer_is_started_detached(monkeypatch):
    started = []
    monkeypatch.setattr(desktop.subprocess, "Popen", lambda args, **kw: started.append((args, kw)))
    desktop._spawn_restorer()
    (args, kw), = started
    assert args[-2:] == [desktop.os.path.abspath(desktop.__file__), "restore-power-mode"]
    assert kw["cwd"] == desktop.PROJECT_ROOT
