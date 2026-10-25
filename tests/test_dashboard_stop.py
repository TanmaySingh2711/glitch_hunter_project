"""Esc, then Yes, on the dashboard page stops the dashboard.

The page asks first (a dialog whose default is No); only Yes sends
POST /api/stop, and the server then stops the way Ctrl+C in its console does,
so its clean-up runs. These tests pin the route, its refusals, and that the
page wires Esc to the question rather than straight to the stop.
"""
import os
import sys
import threading

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture
def served(monkeypatch):
    import app
    stopped = threading.Event()
    monkeypatch.setattr(app, "_interrupt_main", stopped.set)
    monkeypatch.setattr(app, "STOP_DELAY_S", 0.01)
    return app.app.test_client(), stopped


def test_post_stop_replies_then_interrupts_the_main_thread(served):
    client, stopped = served
    res = client.post("/api/stop")
    assert res.status_code == 200 and res.get_json() == {"stopping": True}
    assert stopped.wait(2), "the server was not asked to stop"


def test_another_website_cannot_stop_it(served):
    client, stopped = served
    assert client.post("/api/stop", headers={"Origin": "https://evil.example"}).status_code == 403
    assert not stopped.wait(0.2)
    assert client.post("/api/stop", headers={"Origin": "http://localhost"}).status_code == 200
    assert stopped.wait(2)


def test_reading_it_does_not_stop_it(served):
    client, stopped = served
    assert client.get("/api/stop").status_code == 405
    assert not stopped.wait(0.2)


def test_the_real_interrupt_is_the_one_ctrl_c_makes(monkeypatch):
    import app
    seen = []
    monkeypatch.setattr(app._thread, "interrupt_main", lambda: seen.append(True))
    app._interrupt_main()
    assert seen == [True]


def test_the_page_asks_before_it_stops():
    html = read("dashboard", "web", "templates", "index.html")
    js = read("dashboard", "web", "static", "js", "main.js")
    assert 'id="stop-dialog"' in html and "Stop the dashboard?" in html
    assert 'id="stop-dialog-no" class="btn btn--secondary" type="button" data-close autofocus' in html
    assert "e.key !== 'Escape'" in js and "stopDialog.showModal()" in js
    assert "dialog[open]" in js, "Esc inside another dialog must only close that dialog"
    assert js.count("/api/stop") == 1 and js.index("/api/stop") > js.index("stop-dialog-yes"), \
        "only the Yes button may send the stop"


# ── Esc in the console window ─────────────────────────────────────────────
def test_esc_in_the_console_stops_it_and_other_keys_do_not():
    from dashboard import desktop
    keys = iter(["a", "\r", "x", desktop.ESC, "never read"])
    pending = {"n": 4}
    stopped = []

    def kbhit():
        return pending["n"] > 0

    def getwch():
        pending["n"] -= 1
        return next(keys)

    desktop.watch_keys(kbhit, getwch, lambda: stopped.append(True), poll_s=0)
    assert stopped == [True] and next(keys) == "never read"


def test_the_watcher_waits_while_no_key_is_pressed_and_ends_with_the_dashboard():
    from dashboard import desktop
    polls = iter([True, True, False])
    desktop.watch_keys(lambda: False, lambda: "", lambda: pytest.fail("no key was pressed"),
                       alive=lambda: next(polls), poll_s=0)


def test_no_console_means_no_watcher(monkeypatch):
    from dashboard import desktop
    monkeypatch.setattr(desktop.sys, "stdin", None)
    assert desktop.install_escape_to_stop(lambda: None) is False


def test_the_dashboard_starts_the_watcher_and_a_clean_stop_closes_the_window():
    app_src = read("app.py")
    assert "desktop.install_escape_to_stop(desktop.interrupt_like_ctrl_c)" in app_src
    assert "except KeyboardInterrupt" in app_src
    bat = read("run_dashboard.bat")
    assert "if not errorlevel 1 exit /b 0" in bat and bat.index("exit /b 0") < bat.index("pause", bat.index("exit /b 0"))
    sh = read("run_dashboard.sh")
    assert '[ "$status" -eq 0 ] && exit 0' in sh
    assert sh.index("exit 0") < sh.index("tools/check_environment.py")


def test_the_shell_launcher_starts_the_dashboard_as_the_batch_file_does():
    sh = read("run_dashboard.sh")
    for step in ("generated/cache/pycache", "-m dashboard.desktop open-dashboard",
                 "app.py --desktop", "tools/check_environment.py"):
        assert step in sh, step
    assert '${@+"$@"}' in sh                       # bash 3.2 on macOS, set -u, no arguments
    assert 'stty "$TERMINAL"' in sh and "trap finish EXIT" in sh


def test_esc_stops_the_dashboard_like_ctrl_c(monkeypatch):
    from dashboard import desktop
    seen = []
    monkeypatch.setattr(desktop._thread, "interrupt_main", lambda: seen.append("interrupt_main"))
    monkeypatch.setattr(desktop.os, "kill", lambda pid, sig: seen.append((pid, sig)))
    for platform in ("win32", "linux", "darwin"):
        monkeypatch.setattr(desktop.sys, "platform", platform)
        desktop.interrupt_like_ctrl_c()
    me = desktop.os.getpid()
    assert seen == ["interrupt_main", (me, desktop.signal.SIGINT), (me, desktop.signal.SIGINT)]


@pytest.mark.skipif(sys.platform == "win32", reason="select() on a pipe needs Linux or macOS")
def test_the_terminal_reader_tells_esc_from_an_arrow_key():
    from dashboard import desktop
    r, w = os.pipe()
    try:
        kbhit, getwch = desktop.posix_key_reader(r)
        assert kbhit() is False
        os.write(w, b"\x1b[A")                    # an arrow key: Esc plus more bytes at once
        assert kbhit() and getwch() == ""
        assert kbhit() is False
        os.write(w, b"\x1b")                       # Esc on its own
        assert getwch() == desktop.ESC
        os.write(w, b"q")
        assert getwch() == "q"
        os.close(w)
        w = -1
        with pytest.raises(EOFError):               # the terminal went away
            getwch()
    finally:
        os.close(r)
        if w != -1:
            os.close(w)
