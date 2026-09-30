"""dashboard_backend.py and app.py: what the browser is sent, and from where.

The frame stream is driven with a stand-in env and model injected as the
backend's globals, so these run without loading weights; the Flask routes
and socket handlers are exercised through Flask's own test clients.
"""
import numpy as np
import pytest

import dashboard_backend as db
from agent_logic import ACTION_NAMES
from exploration import config


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


def _Model():
    from incident_helpers import FakeBrain
    return FakeBrain(action=3)


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


def test_a_freshly_loaded_page_resets_the_dashboard_first(app_module, monkeypatch):
    """A refresh is a Reset: the page is answered only once the reset ran."""
    done = []
    monkeypatch.setattr(app_module.service, "client_connected", lambda: None)
    monkeypatch.setattr(app_module.service, "client_disconnected", lambda: None)
    monkeypatch.setattr(app_module.service, "reset", lambda: done.append("reset"))
    monkeypatch.setattr(app_module.service, "wait_idle",
                        lambda timeout=5.0: done.append("waited") or True)
    client = app_module.socketio.test_client(app_module.app)
    assert client.emit('page_opened', callback=True) is True
    assert done == ["reset", "waited"], "answered before the reset had run"
    client.disconnect()


def test_main_preloads_before_listening_and_warns_off_loopback(app_module, monkeypatch, caplog):
    import logging
    order = []
    monkeypatch.setattr(app_module.service, "start", lambda: order.append("preload"))
    monkeypatch.setattr(app_module.dashboard_facts.TESTS, "start", lambda: order.append("tests"))
    monkeypatch.setattr(app_module.socketio, "run",
                        lambda app, **kw: order.append(("run", kw['host'], kw['port'],
                                                         kw['debug'])))
    monkeypatch.setenv("GLITCH_HUNTER_HOST", "0.0.0.0")
    monkeypatch.setenv("GLITCH_HUNTER_PORT", "5055")
    with caplog.at_level(logging.INFO):
        app_module.main()
    assert order == ["preload", "tests", ("run", "0.0.0.0", 5055, False)]
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


def test_switching_the_game_unloads_everything_built_on_the_old_one(monkeypatch, tmp_path):
    closed = []

    class Env(_Env):
        def close(self):
            closed.append("env")

    class Pipe:
        def close(self):
            closed.append("pipeline")
    env = Env()
    monkeypatch.setattr(db, "_global_env", env)
    monkeypatch.setattr(db, "_global_model", _Model())
    monkeypatch.setattr(db, "_pipeline", Pipe())
    monkeypatch.setattr(db, "_provenance", {"game": "old"})
    monkeypatch.setattr(db, "_config", db.DashboardConfig(game_variant="mario_clean",
                                                          incidents_dir=str(tmp_path)))
    monkeypatch.setattr(db, "release_game_variant", lambda: closed.append("game modules"))
    with pytest.raises(ValueError, match="unknown game variant"):
        db.switch_game_variant("mario_3")
    assert db.switch_game_variant("mario_clean") is False and closed == []
    assert db.switch_game_variant("mario_bugged") is True
    assert closed == ["pipeline", "env", "game modules"]
    assert env.unwrapped.calls == ["close"], "the old game's window was not closed"
    assert db._global_env is None and db._global_model is None and db._pipeline is None
    assert db._config.game_variant == "mario_bugged" and db._provenance == {}
    monkeypatch.setattr(db, "_global_env", env)
    with pytest.raises(RuntimeError, match="fixed once the env exists"):
        db.configure(db.DashboardConfig(game_variant="mario_clean"))


# ── the live panel ─────────────────────────────────────────────────────────
class _Coverage:
    """The coverage map's surface the live panel reads."""

    def __init__(self):
        self.testable = np.ones((2, 2), dtype=bool)
        self.testable_total = 1000
        self.episode_new = 0


def test_every_step_carries_the_live_panel(injected, monkeypatch):
    monkeypatch.setattr(db, "_coverage", None)
    session = db.run_mario_agent()
    first = next(session)
    t = first['telemetry']
    assert t['run'] == 1 and t['run_step'] == 1 and t['action'] == ACTION_NAMES[3]
    assert t['x'] == 100 and t['coverage'] is None and t['phase'] is None
    assert t['progress'] == 0.0                 # x 100 is behind the spawn point
    next(session)
    next(session)                               # step 3 ends the episode ...
    assert next(session)['telemetry']['run'] == 2, "... and the next run is counted"
    session.close()


def test_coverage_is_the_training_map_plus_what_is_new_since_loading(monkeypatch):
    cov = _Coverage()
    monkeypatch.setattr(db, "_coverage", cov)
    monkeypatch.setattr(db, "_coverage_base", 900)
    monkeypatch.setattr(db, "_coverage_banked", 0)
    cov.episode_new = 7
    assert db.coverage_now() == {"covered": 907, "total": 1000, "new_since_start": 7}
    db._bank_coverage()                         # a reset is about to zero the episode count
    cov.episode_new = 0
    cov.episode_new += 2
    assert db.coverage_now() == {"covered": 909, "total": 1000, "new_since_start": 9}
    cov.testable = None
    assert db.coverage_now() is None, "no reachable-space mask: nothing honest to show"
    monkeypatch.setattr(db, "_coverage", None)
    assert db.coverage_now() is None
    db._bank_coverage()                         # harmless without a map


def test_the_progress_is_measured_from_the_spawn_to_the_castle_door():
    from evaluation.completion import CASTLE_DOOR_X, SPAWN_X
    t = db.telemetry({'episode_phase': 'explore', 'mario_rect': (5000, 400, 30, 40)}, 3, 17, 4,
                     (SPAWN_X + CASTLE_DOOR_X) // 2)
    assert t['phase'] == 'explore' and t['run'] == 3 and t['run_step'] == 17
    assert t['action'] == ACTION_NAMES[4] and t['x'] == 5000
    assert t['progress'] == pytest.approx(0.5, abs=1e-3)
    assert db.telemetry({}, 1, 1, 99, 0)['action'] == "Unknown"


def test_the_brain_facts_come_from_the_loaded_model(monkeypatch):
    import torch

    class _Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Linear(3, 2)          # 8 parameters
            self.also_shared = self.shared                # counted once, as in SB3
            self.head = torch.nn.Linear(2, 1)             # 3 parameters

    class _PPO:
        policy = _Policy()

    monkeypatch.setattr(db, "_global_model", _PPO())
    monkeypatch.setattr(db, "_provenance", {"brain": {"path": "b.zip", "num_timesteps": 16,
                                                      "approved_objective2_brain": True}})
    facts = db.DashboardBackend().brain_facts()
    assert facts["parameters"] == 11 and facts["num_timesteps"] == 16 and facts["approved"]
    assert facts["actions"] == len(ACTION_NAMES)
    monkeypatch.setattr(db, "_global_model", None)
    assert db.brain_facts()["parameters"] is None


def test_the_project_facts_route_answers_from_the_backend(app_module, monkeypatch):
    monkeypatch.setattr(type(app_module.backend), "pipeline", property(lambda _s: None))
    monkeypatch.setattr(type(app_module.backend), "run_reports", property(lambda _s: None))
    monkeypatch.setattr(app_module.backend, "brain_facts", lambda: {"parameters": 5})
    monkeypatch.setattr(app_module.dashboard_facts, "code_census", lambda: {"commits": 1})
    facts = app_module.app.test_client().get('/api/project').get_json()
    assert facts["brain"] == {"parameters": 5}
    assert facts["objective3"]["evidence"]["incidents"] == 0
    assert facts["engineering"]["commits"] == 1 and "tests" in facts["engineering"]


def test_page_edits_show_on_a_refresh_without_a_restart(app_module, monkeypatch, tmp_path):
    """HTML is re-read when it changes, and the CSS/JS version follows the
    files' own modification times, so a browser refresh shows an edit."""
    import os
    assert app_module.app.config["TEMPLATES_AUTO_RELOAD"] is True
    assert app_module.app.jinja_env.auto_reload is True
    for name in app_module.STATIC_ASSETS:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x")
        os.utime(tmp_path / name, (1_000_000, 1_000_000))
    monkeypatch.setattr(app_module.app, "static_folder", str(tmp_path))
    assert app_module.asset_version() == "1000000000"
    os.utime(tmp_path / "css/style.css", (2_000_000, 2_000_000))
    after = app_module.asset_version()
    assert after == "2000000000"
    page = app_module.app.test_client().get('/').get_data(as_text=True)
    assert f"style.css?v={after}" in page and f"main.js?v={after}" in page


# ── a new route every run: actions are drawn from the brain's own policy ──
def _brain_with(probs):
    import torch
    from incident_helpers import FakeBrain
    brain = FakeBrain()
    brain.policy.logits = torch.log(torch.tensor([probs], dtype=torch.float64))
    return brain


def test_temperature_zero_is_the_old_greedy_choice():
    brain = _brain_with([0.1, 0.2, 0.7])
    rng = np.random.default_rng(0)
    assert {db.choose_action(brain, np.zeros(1), rng, temperature=0) for _ in range(50)} == {2}


def test_actions_follow_the_policy_sharpened_by_the_temperature():
    brain = _brain_with([0.3, 0.7])
    rng = np.random.default_rng(1)
    at_one = sum(db.choose_action(brain, np.zeros(1), rng, temperature=1.0) for _ in range(4000)) / 4000
    assert abs(at_one - 0.7) < 0.03                       # T = 1: the policy itself
    sharp = sum(db.choose_action(brain, np.zeros(1), rng, temperature=0.5) for _ in range(4000)) / 4000
    assert abs(sharp - 0.49 / 0.58) < 0.03                # T = 0.5: p^2, renormalised
    assert 0 < config.DASHBOARD_POLICY_TEMPERATURE < 1


def test_two_sessions_take_different_routes():
    brain = _brain_with([0.5, 0.5])
    routes = [tuple(db.choose_action(brain, np.zeros(1), np.random.default_rng(), 1.0) for _ in range(40))
              for _ in range(2)]
    assert routes[0] != routes[1]


class _StuckEnv(_Env):
    """Mario never gets past x 100."""

    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)
        info["mario_rect"] = (100, 400, 16, 16)
        return obs, reward, done, trunc, info


def _session(monkeypatch, tmp_path, variant, env):
    monkeypatch.setattr(db, "_global_env", env)
    monkeypatch.setattr(db, "_global_model", _brain_with([0.45, 0.55]))
    monkeypatch.setattr(db, "_config", db.DashboardConfig(game_variant=variant,
                                                          incidents_dir=str(tmp_path / "inc"),
                                                          run_reports_dir=str(tmp_path / "runs")))
    monkeypatch.setattr(db, "_pipeline", None)
    return db.run_mario_agent()


def test_every_clean_game_run_takes_a_new_route(monkeypatch, tmp_path):
    assert config.DASHBOARD_FIXED_ROUTE_RUNS[config.CLEAN_GAME_VARIANT] == 0
    session = _session(monkeypatch, tmp_path, config.CLEAN_GAME_VARIANT, _Env(episode_len=10_000))
    assert {next(session)["action"] for _ in range(80)} == {0, 1}
    session.close()


def test_the_bugged_games_first_run_keeps_the_fixed_route_then_routes_vary(monkeypatch, tmp_path):
    assert config.DASHBOARD_FIXED_ROUTE_RUNS[config.BUGGED_GAME_VARIANT] == 1
    session = _session(monkeypatch, tmp_path, config.BUGGED_GAME_VARIANT, _Env(episode_len=60))
    items = [next(session) for _ in range(160)]
    session.close()
    assert {i["action"] for i in items if i["telemetry"]["run"] == 1} == {1}        # the top pick
    assert {i["action"] for i in items if i["telemetry"]["run"] > 1} == {0, 1}


def test_a_drawn_route_stuck_in_a_trap_ends_early(monkeypatch, tmp_path):
    env = _StuckEnv(episode_len=10_000)
    session = _session(monkeypatch, tmp_path, config.CLEAN_GAME_VARIANT, env)
    items = [next(session) for _ in range(config.DASHBOARD_STUCK_STEPS + 1)]
    session.close()
    ended = [i for i in items if "run_end" in i]
    assert len(ended) == 1 and ended[0] is items[-1]
    assert ended[0]["run_end"]["end_reason"] == "safety_reset" and ended[0]["run_end"]["report"] is None
    assert config.DASHBOARD_STUCK_STEPS * 30 < config.QA_EPISODE_MAX_STEPS        # far sooner than the engine


def test_the_fixed_route_is_never_cut_short(monkeypatch, tmp_path):
    env = _StuckEnv(episode_len=10_000)
    session = _session(monkeypatch, tmp_path, config.BUGGED_GAME_VARIANT, env)
    for _ in range(config.DASHBOARD_STUCK_STEPS + 50):       # well past the limit
        next(session)
    session.close()
    assert env.resets == 1


def test_the_dashboard_can_be_stopped_with_sigterm():
    """SDL must not take over SIGTERM in the dashboard's process, or `kill`
    cannot stop it on Linux and macOS (tests/test_dashboard_e2e.py checks the
    real stop in CI)."""
    import os

    import app  # noqa: F401 - importing sets the hint before pygame starts
    assert os.environ.get("SDL_NO_SIGNAL_HANDLERS") == "1"


def test_a_fixed_route_config_keeps_the_top_pick_on_every_run(monkeypatch, tmp_path):
    """For tools that need the same route twice (validate_incident_pipeline)."""
    monkeypatch.setattr(db, "_global_env", _Env(episode_len=30))
    monkeypatch.setattr(db, "_global_model", _brain_with([0.45, 0.55]))
    monkeypatch.setattr(db, "_config", db.DashboardConfig(game_variant=config.CLEAN_GAME_VARIANT,
                                                          incidents_dir=str(tmp_path / "inc"),
                                                          run_reports_dir=str(tmp_path / "runs"),
                                                          fixed_route=True))
    monkeypatch.setattr(db, "_pipeline", None)
    session = db.run_mario_agent()
    items = [next(session) for _ in range(100)]
    session.close()
    assert max(i["telemetry"]["run"] for i in items) >= 3
    assert {i["action"] for i in items} == {1}
