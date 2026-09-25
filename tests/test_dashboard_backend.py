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

    class c_module:                      # noqa: N801 - the engine's module name
        SCREEN_SIZE = (800, 600)

    def __init__(self):
        self.calls = []

    def render_scaled(self, size):
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    def open_window(self, minimized=False):
        self.calls.append("open minimized" if minimized else "open")
        return "created"

    def hide_window(self):
        self.calls.append("hide")

    def close_window(self):
        self.calls.append("close")

    def poll_close_request(self):
        return True


class _Env:
    def __init__(self, episode_len=3, alert_at=None, end_info=None):
        self.unwrapped = _Base()
        self.t = 0
        self.resets = 0
        self.episode_len = episode_len
        self.alert_at = alert_at
        self.end_info = end_info or {}

    def reset(self, **_kw):
        self.resets += 1
        self.t = 0
        return np.zeros((4, 84, 84), dtype=np.uint8), {}

    def step(self, action):
        self.t += 1
        info = {'glitch_alert': "Mario is below the floor"} if self.t == self.alert_at else {}
        info['mario_rect'] = (100 * self.t, 400, 16, 16)
        done = self.t >= self.episode_len
        if done:
            info.update(self.end_info)
        return (np.full((4, 84, 84), self.t, dtype=np.uint8), 0.5, done, False, info)


class _Model:
    def predict(self, obs, deterministic):
        assert deterministic, "the dashboard shows the greedy policy"
        return np.array(3), None


@pytest.fixture
def injected(monkeypatch, tmp_path):
    env = _Env(episode_len=3, alert_at=2)
    monkeypatch.setattr(db, "_global_env", env)
    monkeypatch.setattr(db, "_global_model", _Model())
    # preload() also starts the incident pipeline: keep it off the real
    # incidents/ folder and shut it down afterwards.
    monkeypatch.setattr(db, "_config", db.DashboardConfig(incidents_dir=str(tmp_path / "inc"),
                                                          run_reports_dir=str(tmp_path / "runs")))
    monkeypatch.setattr(db, "_pipeline", None)
    monkeypatch.setattr(db, "_run_reports", None)
    monkeypatch.setattr(db, "_provenance", {})
    yield env
    if db._pipeline is not None:
        db._pipeline.close()


def _session_ending_with(monkeypatch, tmp_path, variant, end_info):
    """A backend whose episode ends on step 3 with `end_info`, on `variant`."""
    env = _Env(episode_len=3, end_info=end_info)
    monkeypatch.setattr(db, "_global_env", env)
    monkeypatch.setattr(db, "_config", db.DashboardConfig(
        game_variant=variant, incidents_dir=str(tmp_path / "inc"),
        run_reports_dir=str(tmp_path / "runs")))
    backend = db.DashboardBackend()
    backend.preload()
    return backend, backend.new_session()


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


def test_a_clean_run_that_reaches_the_castle_ends_with_a_no_bug_report(injected, monkeypatch,
                                                                       tmp_path):
    backend, session = _session_ending_with(monkeypatch, tmp_path, "mario_clean",
                                            {'flag_get': True, 'score': 1200, 'coins': 3})
    items = [next(session) for _ in range(3)]
    assert 'run_end' not in items[0] and 'run_end' not in items[1]
    end = items[2]['run_end']
    assert end['end_reason'] == 'level_complete'
    report = end['report']
    assert report['headline'] == "No Bugs Found" and report['bugs'] == 0
    assert report['agent_steps'] == 3
    backend.run_reports.wait()
    folder = tmp_path / "runs" / report['run_id']
    for name in ("run.json", "final.png", "finish.gif", "report.md", "report.pdf"):
        assert (folder / name).is_file(), name
    md = (folder / "report.md").read_text(encoding="utf-8")
    assert "No Bugs Found" in md and "world x 300" in md
    assert 'run_end' not in next(session), "the next run started without a fresh run_end"
    session.close()


def test_a_clean_run_that_dies_ends_without_a_report(injected, monkeypatch, tmp_path):
    _backend, session = _session_ending_with(monkeypatch, tmp_path, "mario_clean",
                                             {'is_dead': True, 'death_cause': 'goomba'})
    end = [next(session) for _ in range(3)][2]['run_end']
    assert end == {'end_reason': 'death', 'report': None}
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())
    session.close()


def test_the_bugged_game_plays_on_from_run_to_run(injected, monkeypatch, tmp_path):
    _backend, session = _session_ending_with(monkeypatch, tmp_path, "mario_bugged",
                                             {'flag_get': True})
    assert all('run_end' not in next(session) for _ in range(6))
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
    # Start puts the window in the taskbar, minimised (the live view is the browser's).
    assert injected.unwrapped.calls == ["hide", "open minimized", "hide", "close"]
    assert next(backend.new_session())['step'] == 1
    assert backend.pipeline is not None, "preload did not start the incident pipeline"
    assert backend.describe()["game_variant"] == db._config.game_variant


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


def test_a_run_report_that_cannot_be_written_is_logged_not_raised(injected, monkeypatch, tmp_path,
                                                                  caplog):
    backend, session = _session_ending_with(monkeypatch, tmp_path, "mario_clean",
                                            {'flag_get': True})

    def broken(*_a, **_k):
        raise OSError("disk full")
    monkeypatch.setattr(backend.run_reports, "write", broken)
    end = [next(session) for _ in range(3)][2]['run_end']
    assert end == {'end_reason': 'level_complete', 'report': None}
    assert any("could not write the run report" in r.getMessage() for r in caplog.records)
    session.close()


def test_desktop_mode_boosts_and_guards_and_restores_on_exit(app_module, monkeypatch):
    import desktop
    calls = []

    class Guard:
        def start(self):
            calls.append("start")

        def stop(self):
            calls.append("stop")
    monkeypatch.setattr(desktop, "boost_this_process", lambda: ["1 ms timers"])
    monkeypatch.setattr(desktop, "PowerModeGuard", Guard)
    monkeypatch.setattr(desktop, "install_console_close_handler",
                        lambda fn: calls.append(("close handler", fn.__self__.__class__)))
    registered = []
    monkeypatch.setattr("atexit.register", registered.append)
    app_module._desktop_mode()
    assert calls == ["start", ("close handler", Guard)]
    assert len(registered) == 1
    registered[0]()
    assert calls[-1] == "stop", "exiting did not put the power mode back"
    assert app_module.parse_args(["--desktop"]).desktop is True
    assert app_module.parse_args([]).desktop is False
