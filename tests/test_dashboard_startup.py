"""The dashboard is served before the brain has loaded.

run_dashboard.bat used to show the page only after torch was imported and the
brain loaded (about 10 s, measured). Now app.py starts the server at once and
the game thread loads the brain behind it: the page says "Loading AI" until
'backend_ready', and a Start pressed meanwhile waits in the game thread's
queue. These tests pin that order, the loading signal, a failed load still
stopping the process, and that the faster brain load plays exactly the same.
"""
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from dashboard import service as ds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRAIN = os.path.join(ROOT, "glitch_hunter_main_brain.zip")


class SlowBackend:
    """A backend whose pre-load waits for `release` (or raises `error`)."""

    def __init__(self, error=None):
        self.release = threading.Event()
        self.error = error
        self.calls = []

    def preload(self):
        self.release.wait(5)
        if self.error is not None:
            raise self.error
        self.calls.append("preload")

    def open_window(self):
        self.calls.append("open_window")
        return "created"

    def new_session(self):
        self.calls.append("new_session")

        def run():
            while True:
                yield {"frame": b"", "log": None}
        return run()

    def hide_window(self):
        pass

    def close_window(self):
        pass

    def poll_close_request(self):
        return False

    def switch_game(self, variant):
        return False

    def begin_incident_session(self):
        pass

    def stop_audio(self):
        pass


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


# ── the service ───────────────────────────────────────────────────────────
def test_start_without_waiting_returns_at_once_and_queues_a_start():
    backend = SlowBackend()
    s = ds.GameWindowService(backend, emit=lambda *a: None, log=lambda *a: None)
    try:
        t0 = time.monotonic()
        s.start(wait=False)
        assert time.monotonic() - t0 < 0.5, "start(wait=False) must not wait for the pre-load"
        assert not s.ready and s.status()["ready"] is False
        s.start_testing()                         # the user clicked before the brain was ready
        time.sleep(0.1)
        assert backend.calls == [], "nothing may run before the pre-load"
        backend.release.set()
        assert s.wait_ready(5) is None and s.ready
        assert wait_for(lambda: s.testing), "the early Start was lost"
        assert backend.calls[:3] == ["preload", "open_window", "new_session"]
        assert s.status()["ready"] is True
    finally:
        backend.release.set()
        s.shutdown()


def test_wait_ready_reports_a_failed_preload():
    backend = SlowBackend(error=RuntimeError("torch would not load"))
    s = ds.GameWindowService(backend, emit=lambda *a: None, log=lambda *a: None)
    s.start(wait=False)
    backend.release.set()
    error = s.wait_ready(5)
    assert isinstance(error, RuntimeError) and "torch" in str(error)
    assert not s.ready


def test_the_blocking_start_is_unchanged():
    backend = SlowBackend()
    s = ds.GameWindowService(backend, emit=lambda *a: None, log=lambda *a: None)
    threading.Timer(0.2, backend.release.set).start()
    try:
        s.start(timeout=5)
        assert s.ready and backend.calls == ["preload"]
    finally:
        s.shutdown()


# ── the server ────────────────────────────────────────────────────────────
@pytest.fixture
def app_module(monkeypatch):
    import app
    monkeypatch.setattr(app, "_startup_error", None)
    return app


def test_a_page_that_connects_while_loading_is_told_so(app_module, monkeypatch):
    monkeypatch.setattr(type(app_module.service), "ready", property(lambda self: False))
    client = app_module.socketio.test_client(app_module.app)
    try:
        names = [m["name"] for m in client.get_received()]
        assert "backend_loading" in names
        assert app_module.app.test_client().get("/healthz").get_json()["ready"] is False
    finally:
        client.disconnect()


def test_a_page_that_connects_once_loaded_is_not(app_module, monkeypatch):
    monkeypatch.setattr(type(app_module.service), "ready", property(lambda self: True))
    client = app_module.socketio.test_client(app_module.app)
    try:
        assert "backend_loading" not in [m["name"] for m in client.get_received()]
    finally:
        client.disconnect()


def test_ready_is_announced_to_every_page(app_module, monkeypatch):
    sent, counted = [], []
    monkeypatch.setattr(app_module.service, "wait_ready", lambda timeout=None: None)
    monkeypatch.setattr(app_module.socketio, "emit", lambda event, data: sent.append(event))
    monkeypatch.setattr(app_module.facts.TESTS, "start", lambda: counted.append(True))
    app_module._announce_ready()
    assert sent == ["backend_ready"] and counted == [True]
    assert app_module._startup_error is None


def test_a_failed_load_stops_the_dashboard_with_an_error(app_module, monkeypatch):
    stopped = []
    error = ImportError("DLL load failed while importing _C")
    monkeypatch.setattr(app_module.service, "wait_ready", lambda timeout=None: error)
    monkeypatch.setattr(app_module, "_interrupt_main", lambda: stopped.append(True))
    monkeypatch.setattr(app_module.socketio, "emit", lambda *a: pytest.fail("no backend_ready"))
    app_module._announce_ready()
    assert stopped == [True] and app_module._startup_error is error


def test_the_load_waits_until_the_server_listens(app_module, monkeypatch):
    started = []
    monkeypatch.setattr(app_module.service, "start", lambda **kw: started.append(kw))
    monkeypatch.setattr(app_module, "_announce_ready", lambda: None)
    import socket
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    t = threading.Thread(target=app_module._load_when_listening, args=("127.0.0.1", port))
    t.start()
    time.sleep(0.3)
    assert started == [], "the brain must not start loading before the server listens"
    listener.listen()
    t.join(5)
    listener.close()
    assert started == [{"wait": False}]


def test_the_page_has_a_loading_state():
    with open(os.path.join(ROOT, "dashboard", "web", "static", "js", "main.js"), encoding="utf-8") as fh:
        js = fh.read()
    assert "socket.on('backend_loading'" in js and "socket.on('backend_ready'" in js
    assert "loading: ['info', 'clock', 'Loading AI']" in js
    assert "state.loading || state.switching || state.testing" in js, "Start waits while loading"


# ── the backend ───────────────────────────────────────────────────────────
def test_importing_the_backend_does_not_import_torch():
    code = "import sys, dashboard.backend; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         timeout=120, check=True)
    assert out.stdout.strip() == "False", "torch must load on the game thread, not at import"


@pytest.mark.slow
def test_the_inference_only_load_plays_exactly_like_the_full_one():
    if not os.path.exists(BRAIN):
        pytest.skip("glitch_hunter_main_brain.zip is not installed")
    import torch
    from stable_baselines3 import PPO

    from dashboard import backend
    fast = PPO.load(BRAIN, device="cpu",
                    custom_objects={"policy_kwargs": backend._INFERENCE_POLICY_KWARGS})
    full = PPO.load(BRAIN, device="cpu")
    for name, tensor in full.policy.state_dict().items():
        assert torch.equal(fast.policy.state_dict()[name], tensor), name
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs = rng.integers(0, 255, size=full.observation_space.shape, dtype=np.uint8)
        with torch.no_grad():
            a = fast.policy.get_distribution(fast.policy.obs_to_tensor(obs)[0]).distribution.logits
            b = full.policy.get_distribution(full.policy.obs_to_tensor(obs)[0]).distribution.logits
        assert torch.equal(a, b)
        assert backend.choose_action(fast, obs, np.random.default_rng(1)) == \
            backend.choose_action(full, obs, np.random.default_rng(1))
