"""training/: the callbacks and checkpoint helpers, driven without a real PPO.

Every callback is fed exactly what SB3 would hand it - num_timesteps and the
rollout's locals - through a stand-in model, so each rule can be checked
at the step it is supposed to act on.
"""
import json
import logging
import zipfile

import numpy as np
import pytest

from exploration import config
from exploration.coverage import SpatialCoverage
from training import callbacks as cbs
from training import checkpoints as ck


class _Recorder:
    """SB3's logger surface: record(key, value)."""

    def __init__(self):
        self.values = {}

    def record(self, key, value):
        self.values[key] = value


class _Model:
    def __init__(self, t=0, ent_coef=0.03, logger=None):
        self.num_timesteps = t
        self.ent_coef = ent_coef
        self.logger = logger
        self.learning_rate = None
        self.lr_schedule = None
        self.saved = []

    def get_env(self):
        return None

    def save(self, path):
        self.saved.append(path)
        with open(path, "wb") as fh:
            fh.write(b"model")


def _step(cb, t, **local):
    cb.model.num_timesteps = t
    cb.update_locals(local)
    return cb.on_step()


# ── checkpoints.py ─────────────────────────────────────────────────────────
@pytest.mark.parametrize(('name', 'steps'), [
    ("mario_brain_checkpoint_6000000_steps.zip", 6_000_000),
    ("mario_brain_checkpoint_800000_steps.zip", 800_000),
    ("mario_brain_checkpoint.zip", -1),
    ("mario_brain_checkpoint_final_steps.zip", -1),
])
def test_milestone_steps_parses_only_real_milestones(name, steps):
    assert ck.milestone_steps(name) == steps


def test_newest_milestone_sorts_numerically_not_lexically(tmp_path):
    for n in ("800000", "6000000", "5600000"):
        (tmp_path / f"brain_{n}_steps.zip").write_bytes(b"")
    (tmp_path / "brain_junk.zip").write_bytes(b"")
    (tmp_path / "other_9000000_steps.zip").write_bytes(b"")
    assert ck.newest_milestone(str(tmp_path), "brain").endswith("brain_6000000_steps.zip")
    assert ck.newest_milestone(str(tmp_path / "missing"), "brain") is None
    assert ck.newest_milestone(str(tmp_path), "nothing") is None


def test_checkpoint_timesteps_reads_the_zip_record(tmp_path):
    p = tmp_path / "m.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("data", json.dumps({"num_timesteps": 6_400_000}))
    assert ck.checkpoint_timesteps(str(p)) == 6_400_000


# ── ExactMilestoneCheckpointCallback ───────────────────────────────────────
def test_the_milestone_callback_needs_exactly_one_schedule(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        cbs.ExactMilestoneCheckpointCallback(save_path=str(tmp_path), name_prefix="m")
    with pytest.raises(ValueError, match="exactly one"):
        cbs.ExactMilestoneCheckpointCallback(targets=[1], every=1, save_path=str(tmp_path),
                                             name_prefix="m")
    with pytest.raises(ValueError, match="required"):
        cbs.ExactMilestoneCheckpointCallback(targets=[1])


def test_legacy_targets_are_saved_once_each_even_when_stepped_over(tmp_path):
    cb = cbs.ExactMilestoneCheckpointCallback(targets=[400, 800], save_path=str(tmp_path),
                                              name_prefix="m")
    cb.init_callback(_Model(0))
    _step(cb, 408)                        # 8 envs: 400 is never hit exactly
    _step(cb, 416)
    _step(cb, 808)
    assert [p.split("_")[-2] for p in cb.model.saved] == ["400", "800"]
    # A fresh callback (a restart) sees the flags and saves nothing again.
    again = cbs.ExactMilestoneCheckpointCallback(targets=[400, 800], save_path=str(tmp_path),
                                                 name_prefix="m")
    again.init_callback(_Model(808))
    _step(again, 816)
    assert again.model.saved == []


# ── WatchdogCallback ───────────────────────────────────────────────────────
def test_the_watchdog_stops_only_a_blind_agent(capsys):
    cb = cbs.WatchdogCallback()
    cb.init_callback(_Model())
    assert _step(cb, 5000, new_obs=np.ones((8, 4, 84, 84)), infos=[]) is True
    assert _step(cb, 10_000, new_obs=np.zeros((8, 4, 84, 84)), infos=[]) is False
    assert "Crunching frames" in capsys.readouterr().out


def test_the_watchdog_warns_on_collapse_at_most_once_per_window(caplog):
    cb = cbs.WatchdogCallback()
    cb.init_callback(_Model())
    good = [{"episode": {"r": 100.0}}] * 60
    _step(cb, 1, infos=good)
    bad = [{"episode": {"r": -500.0}}] * 60
    with caplog.at_level(logging.WARNING):
        _step(cb, 60_001, infos=bad)
        _step(cb, 60_002, infos=bad)
    alerts = [r for r in caplog.records if "collapse" in r.getMessage()]
    assert len(alerts) == 1


# ── ValueWarmupCallback ────────────────────────────────────────────────────
def test_warmup_runs_the_low_rate_then_restores_it():
    cb = cbs.ValueWarmupCallback(warmup_until=200, warmup_lr=1e-5, normal_lr=1e-4)
    model = _Model(100)
    cb.init_callback(model)
    cb.on_training_start({}, {})
    assert model.learning_rate == 1e-5 and model.lr_schedule(1.0) == 1e-5
    _step(cb, 150)
    assert model.learning_rate == 1e-5
    _step(cb, 208)
    assert model.learning_rate == 1e-4 and model.lr_schedule(1.0) == 1e-4


def test_warmup_already_past_starts_at_the_normal_rate():
    cb = cbs.ValueWarmupCallback(warmup_until=200, warmup_lr=1e-5, normal_lr=1e-4)
    model = _Model(500)
    cb.init_callback(model)
    cb.on_training_start({}, {})
    assert model.learning_rate == 1e-4


# ── LifecycleStatsCallback ─────────────────────────────────────────────────
def test_lifecycle_stats_count_every_ending_and_reset_per_report(caplog):
    rec = _Recorder()
    cb = cbs.LifecycleStatsCallback(every=100)
    cb.init_callback(_Model(logger=rec))
    infos = [
        {"phase_transition_reason": "T1_target_met", "episode_end_reason": "death",
         "lifecycle_agent_steps": 400},
        {"episode_end_reason": "safety_reset", "safety_reset_reason": "stuck",
         "lifecycle_agent_steps": 5001, "clock_extensions": 1},
        {"episode_end_reason": "engine_done", "TimeLimit.truncated": True},
        {},
    ]
    with caplog.at_level(logging.INFO):
        _step(cb, 8, dones=[True, True, True, False], infos=infos)
    line = next(r.getMessage() for r in caplog.records if "[LIFECYCLE]" in r.getMessage())
    assert "3 episodes" in line and "safety_reset:stuck 1" in line
    assert "time_limit 1" in line and "longest 5,001 steps, 1 outlived" in line
    assert rec.values["lifecycle/end_death"] == pytest.approx(1 / 3)
    assert rec.values["lifecycle/longest_episode_steps"] == 5001
    # Counts restart after each report; nothing is reported without episodes.
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _step(cb, 200, dones=[False], infos=[{}])
    assert not [r for r in caplog.records if "[LIFECYCLE]" in r.getMessage()]


# ── StagnationCallback ─────────────────────────────────────────────────────
def _stagnation(total, remaining):
    class _Cov:
        novelty_mult = 1.0

        def covered_testable(self):
            return total[0]

        def remaining(self):
            return remaining

        def set_novelty_mult(self, v):
            self.novelty_mult = v
    return _Cov()


def test_stagnation_escalates_only_after_enough_lean_windows(monkeypatch):
    total = [1_000_000]
    cov = _stagnation(total, remaining=2_000_000)
    model = _Model(0, ent_coef=0.03)
    cb = cbs.StagnationCallback(cov, every=10_000)
    cb.init_callback(model)
    t = 0
    for _ in range(config.STAGNATION_WINDOW):      # first report only sets the baseline
        _step(cb, t)
        t += 10_000
    assert cov.novelty_mult == 1.0
    _step(cb, t)
    assert cov.novelty_mult == pytest.approx(1.25)
    assert model.ent_coef == pytest.approx(0.03 * 1.15)


def test_stagnation_never_escalates_when_the_world_is_nearly_done():
    cov = _stagnation([0], remaining=config.STAGNATION_REMAINING_MIN - 1)
    cb = cbs.StagnationCallback(cov, every=1)
    cb.init_callback(_Model())
    for t in range(1, 20):
        _step(cb, t)
    assert cov.novelty_mult == 1.0


def test_stagnation_resets_its_count_while_discovery_is_healthy():
    total = [0]
    cov = _stagnation(total, remaining=2_000_000)
    cb = cbs.StagnationCallback(cov, every=10_000)
    cb.init_callback(_Model())
    for i in range(10):
        total[0] = i * 1_000_000                   # far above the threshold rate
        _step(cb, i * 10_000)
    assert cov.novelty_mult == 1.0


def test_callbacks_without_coverage_do_nothing():
    for cb in (cbs.StagnationCallback(None), cbs.CoverageStatsCallback(None)):
        cb.init_callback(_Model())
        assert _step(cb, 50_000, dones=[]) is True


# ── CoverageStatsCallback, late in a campaign ──────────────────────────────
def test_the_plateau_audit_and_map_are_written_near_the_end(tmp_path, caplog):
    mask = np.zeros((config.GRID_H, config.GRID_W), dtype=bool)
    mask[400:440, 400:500] = True                  # a tiny "level"
    cov = SpatialCoverage(testable_mask=mask)
    cov.visited[400:440, 400:450] = 1
    cov.invalidate_remaining()
    cov.anomalous_px = lambda: 0          # the class map is not in git (CI)
    audit, png = tmp_path / "audit.json", tmp_path / "map.png"
    cb = cbs.CoverageStatsCallback(cov, every=10, audit_path=str(audit), map_path=str(png))
    rec = _Recorder()
    cb.init_callback(_Model(logger=rec))
    with caplog.at_level(logging.INFO):
        _step(cb, 10, dones=[])
    assert json.loads(audit.read_text(encoding='utf-8'))['remaining_testable_px'] == 2000
    assert png.exists()
    assert any("[PLATEAU]" in r.getMessage() for r in caplog.records)
    assert rec.values["coverage/remaining_px"] == 2000


def test_a_failed_map_never_stops_training(tmp_path, monkeypatch):
    mask = np.zeros((config.GRID_H, config.GRID_W), dtype=bool)
    mask[400:440, 400:500] = True
    cov = SpatialCoverage(testable_mask=mask)
    cov.anomalous_px = lambda: 0
    audit = tmp_path / "audit.json"

    def broken(*_a, **_k):
        raise OSError("disk full")
    monkeypatch.setattr(cbs.lc, "write_remaining_map", broken)
    cb = cbs.CoverageStatsCallback(cov, every=10, audit_path=str(audit),
                                   map_path=str(tmp_path / "m.png"))
    cb.init_callback(_Model())
    assert _step(cb, 10, dones=[]) is True
    assert json.loads(audit.read_text(encoding='utf-8'))['map_error'] == "disk full"
