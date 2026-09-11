"""dashboard_backend.py and app.py: what the browser is sent, and from where.

The frame stream is driven with a stand-in env and model injected as the
backend's globals, so these run without loading weights; the Flask routes
and socket handlers are exercised through Flask's own test clients.
"""
import numpy as np
import pytest

import dashboard_backend as db
from agent_logic import ACTION_NAMES


class _Base:
    """The engine surface the backend touches."""

    def __init__(self):
        self.calls = []

    def render_scaled(self, size):
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    def open_window(self):
        self.calls.append("open")
        return "created"

    def hide_window(self):
        self.calls.append("hide")

    def close_window(self):
        self.calls.append("close")

    def poll_close_request(self):
        return True


class _Env:
    def __init__(self, episode_len=3, alert_at=None):
        self.unwrapped = _Base()
        self.t = 0
        self.resets = 0
        self.episode_len = episode_len
        self.alert_at = alert_at

    def reset(self, **_kw):
        self.resets += 1
        self.t = 0
        return np.zeros((4, 84, 84), dtype=np.uint8), {}

    def step(self, action):
        self.t += 1
        info = {'glitch_alert': "Mario is below the floor"} if self.t == self.alert_at else {}
        return (np.full((4, 84, 84), self.t, dtype=np.uint8), 0.5, self.t >= self.episode_len,
                False, info)


class _Model:
    def predict(self, obs, deterministic):
        assert deterministic, "the dashboard shows the greedy policy"
        return np.array(3), None


@pytest.fixture
def injected(monkeypatch):
    env = _Env(episode_len=3, alert_at=2)
    monkeypatch.setattr(db, "_global_env", env)
    monkeypatch.setattr(db, "_global_model", _Model())
    return env


def test_the_stream_yields_a_jpeg_and_a_log_line_per_step(injected):
    session = db.run_mario_agent()
    first, second = next(session), next(session)
    assert first['frame'][:2] == b"\xff\xd8", "not a JPEG"
    assert first['action'] == 3 and first['step'] == 1 and first['reward'] == 0.5
    assert first['log'] == f"Step 1: Action: {ACTION_NAMES[3]} (3) | Reward: 0.50"
    assert second['log'] == "Step 2: 🚨 BUG FOUND: Mario is below the floor"
    next(session)                                 # step 3 ends the episode ...
    fourth = next(session)                        # ... and the session resets it itself
    assert injected.resets == 2 and fourth['step'] == 4
    session.close()


def test_a_missing_frame_is_sent_as_empty_bytes():
    assert db.encode_frame(None) == b""


def test_the_backend_drives_the_engine_window(injected):
    backend = db.DashboardBackend()
    backend.preload()
    assert backend.open_window() == "created"
    backend.hide_window()
    backend.close_window()
    assert backend.poll_close_request() is True
    backend.stop_audio()                          # best-effort; must never raise
    assert injected.unwrapped.calls == ["hide", "open", "hide", "close"]
    assert next(backend.new_session())['step'] == 1


def test_the_reward_mode_follows_the_brain_on_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert db.select_checkpoint() == ("mario_brain_checkpoint.zip", "legacy_completion")
    (tmp_path / "glitch_hunter_qa.zip").write_bytes(b"")
    assert db.select_checkpoint() == ("glitch_hunter_qa.zip", "qa_exploration")


def test_legacy_playback_carries_no_coverage_and_a_bad_paired_file_is_only_a_warning(
        tmp_path, monkeypatch, caplog):
    assert db._dashboard_coverage("legacy_completion") is None
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(db.coverage_mod, "load_testable", lambda: None)
    (tmp_path / "glitch_hunter_qa_coverage.npz").write_bytes(b"not an npz")
    cov = db._dashboard_coverage("qa_exploration")
    assert cov is not None and cov.total_unique() == 0
    assert any("could not load" in r.getMessage() for r in caplog.records)


# ── app.py ─────────────────────────────────────────────────────────────────
@pytest.fixture
def app_module():
    import app
    return app


def test_the_page_and_the_health_check_are_served(app_module):
    client = app_module.app.test_client()
    page = client.get('/')
    assert page.status_code == 200 and b"socket.io" in page.data
    health = client.get('/healthz').get_json()
    assert health['status'] == "ok" and health['frame_loops'] <= 1


def test_socket_events_only_post_commands_to_the_game_thread(app_module, monkeypatch):
    posted = []
    for name in ('start_testing', 'stop_testing', 'reset', 'client_connected',
                 'client_disconnected'):
        monkeypatch.setattr(app_module.service, name,
                            lambda n=name: posted.append(n))
    client = app_module.socketio.test_client(app_module.app)
    client.emit('start_testing')
    client.emit('start_testing', {'unexpected': 'payload'})
    client.emit('stop_testing')
    client.emit('reset_game')
    client.disconnect()
    assert posted == ['client_connected', 'start_testing', 'start_testing',
                      'stop_testing', 'reset', 'client_disconnected']


def test_main_preloads_before_listening_and_warns_off_loopback(app_module, monkeypatch, caplog):
    import logging
    order = []
    monkeypatch.setattr(app_module.service, "start", lambda: order.append("preload"))
    monkeypatch.setattr(app_module.socketio, "run",
                        lambda app, **kw: order.append(("run", kw['host'], kw['port'],
                                                         kw['debug'])))
    monkeypatch.setenv("GLITCH_HUNTER_HOST", "0.0.0.0")
    monkeypatch.setenv("GLITCH_HUNTER_PORT", "5055")
    with caplog.at_level(logging.INFO):
        app_module.main()
    assert order == ["preload", ("run", "0.0.0.0", 5055, False)]
    assert any("no authentication" in r.getMessage() for r in caplog.records
               if r.levelno == logging.WARNING)
