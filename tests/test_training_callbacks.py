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
    """The escalation logic itself, with escalation deliberately switched ON."""
    monkeypatch.setattr(config, "STAGNATION_ESCALATE", True)
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


def test_by_default_stagnation_is_detected_but_changes_nothing(monkeypatch, caplog):
    """The one-way ratchet is off: a stall is reported, and neither the
    novelty scale nor the entropy bonus the campaign was validated at moves."""
    import logging
    monkeypatch.setattr(config, "STAGNATION_ESCALATE", False)
    total = [1_000_000]
    cov = _stagnation(total, remaining=2_000_000)
    model = _Model(0, ent_coef=0.03)
    cb = cbs.StagnationCallback(cov, every=10_000)
    cb.init_callback(model)
    t = 0
    with caplog.at_level(logging.WARNING):
        for _ in range(config.STAGNATION_WINDOW * 3 + 1):
            _step(cb, t)
            t += 10_000
    assert cov.novelty_mult == 1.0, "the novelty scale moved while escalation was off"
    assert model.ent_coef == 0.03, "ent_coef moved while escalation was off"
    assert cb.detections >= 1, "a stall that met every condition was not even reported"
    assert any("Detected only" in r.getMessage() for r in caplog.records)


def test_a_qa_resume_pins_ent_coef_to_the_validated_base(monkeypatch):
    """A checkpoint saved after an escalation must not carry it forward."""
    import types

    import train_agent as ta
    loaded = types.SimpleNamespace(ent_coef=0.0529, target_kl=None, batch_size=None,
                                   learning_rate=None, lr_schedule=None,
                                   tensorboard_log=None, ep_info_buffer=[1],
                                   ep_success_buffer=[1])
    monkeypatch.setattr(ta, "QA_PHASE", True)
    monkeypatch.setattr(ta.PPO, "load", lambda *a, **k: loaded)
    model = ta.build_model("x.zip", None, "cpu")
    assert model.ent_coef == config.ENT_COEF_BASE


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


def test_the_live_table_prints_covered_remaining_and_percent(caplog):
    mask = np.zeros((config.GRID_H, config.GRID_W), dtype=bool)
    mask[400:440, 400:500] = True                   # 40*100 = 4,000 testable px
    cov = SpatialCoverage(testable_mask=mask)
    cov.visited[400:440, 400:500] = 1                # fully covered - no [PLATEAU] noise
    cov.invalidate_remaining()
    cov.anomalous_px = lambda: 0
    cb = cbs.CoverageStatsCallback(cov, every=10)
    cb.init_callback(_Model())
    with caplog.at_level(logging.INFO):
        _step(cb, 10, dones=[])
    table = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("  "))
    header, rule, row = table.splitlines()
    assert header.split() == ["Step", "Covered", "Testable", "Coverage%", "Remaining",
                               "New(sess)", "New/10k"]
    assert set(rule) == {"-"}
    # first-ever call: session_new and rate both baseline to 0, not covered
    assert row.split() == ["10", "4,000", "4,000", "100%", "0", "0", "0"]


def test_the_live_table_header_repeats_every_ten_rows(caplog):
    mask = np.zeros((config.GRID_H, config.GRID_W), dtype=bool)
    mask[400:440, 400:500] = True
    cov = SpatialCoverage(testable_mask=mask)
    cov.anomalous_px = lambda: 0
    cb = cbs.CoverageStatsCallback(cov, every=1)
    cb.init_callback(_Model())
    with caplog.at_level(logging.INFO):
        for i in range(1, 12):
            _step(cb, i, dones=[])
    tables = [r.getMessage() for r in caplog.records if r.getMessage().lstrip().startswith("Step")]
    # header repeats every 10 rows: row 1 and row 11 of 11 total
    assert len(tables) == 2


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


# ══════════════════════════════════════════════════════════════════════════
# ValueWarmupCallback(freeze_actor=True): the policy must be EXACTLY stationary
# ══════════════════════════════════════════════════════════════════════════
def _tiny_ppo():
    """A real, tiny PPO on a real (tiny) env: CnnPolicy with shared features,
    the same structure as the campaign's model."""
    import gymnasium as gym
    from stable_baselines3 import PPO

    class Pix(gym.Env):
        observation_space = gym.spaces.Box(0, 255, (4, 84, 84), np.uint8)
        action_space = gym.spaces.Discrete(10)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.t = 0
            return self.observation_space.sample(), {}

        def step(self, a):
            self.t += 1
            return self.observation_space.sample(), float(a == 3), self.t >= 64, False, {}

    # Seeded: an unseeded init made the anchor tests flaky - how far the
    # random policy starts from its own saved copy decided whether a large
    # test step size converged or diverged.
    return PPO("CnnPolicy", Pix(), n_steps=64, batch_size=32, n_epochs=2,
               device="cpu", seed=20260920)


def _snapshot(model):
    return {n: p.detach().clone() for n, p in model.policy.named_parameters()}


def test_a_frozen_warmup_leaves_every_actor_weight_bit_identical():
    from training.callbacks import ValueWarmupCallback
    model = _tiny_ppo()
    before = _snapshot(model)
    cb = ValueWarmupCallback(warmup_until=10**9, warmup_lr=1e-3, normal_lr=1e-4,
                             freeze_actor=True)
    model.learn(total_timesteps=64 * 3, callback=cb)
    after = _snapshot(model)
    moved = {n for n in before if not (before[n] == after[n]).all()}
    assert moved == {"value_net.weight", "value_net.bias"}, (
        f"during a frozen warm-up exactly the critic head may change; moved: {sorted(moved)}")


def test_the_release_unfreezes_everything_and_the_actor_then_moves():
    from training.callbacks import ValueWarmupCallback
    model = _tiny_ppo()
    cb = ValueWarmupCallback(warmup_until=64 * 2, warmup_lr=1e-3, normal_lr=1e-3,
                             freeze_actor=True)
    model.learn(total_timesteps=64, callback=cb)
    assert cb._frozen, "still inside the warm-up budget, the actor must still be frozen"
    mid = _snapshot(model)
    model.learn(total_timesteps=64 * 3, callback=cb, reset_num_timesteps=False)
    assert not cb._frozen, "the step budget was spent, so the actor must be released"
    assert all(p.requires_grad for p in model.policy.parameters())
    after = _snapshot(model)
    assert any(not (mid[n] == after[n]).all() for n in mid
               if n.startswith(("features_extractor", "action_net"))), \
        "after release the actor never trained"


def test_the_explained_variance_release_needs_two_consecutive_hits():
    from stable_baselines3.common.utils import configure_logger

    from training.callbacks import ValueWarmupCallback
    model = _tiny_ppo()
    model.set_logger(configure_logger(0, None, "", False))
    cb = ValueWarmupCallback(warmup_until=10**9, warmup_lr=1e-3, normal_lr=1e-4,
                             freeze_actor=True, ev_release=0.8)
    cb.init_callback(model)
    cb._freeze()
    model.logger.record("train/explained_variance", 0.9)
    cb._on_rollout_start()
    assert cb._frozen, "one good update must not release the actor"
    model.logger.record("train/explained_variance", 0.5)
    cb._on_rollout_start()
    assert cb._frozen and cb._ev_hits == 0, "a bad update must reset the streak"
    for _ in range(2):
        model.logger.record("train/explained_variance", 0.85)
        cb._on_rollout_start()
    assert not cb._frozen, "two consecutive updates at or above ev_release release it"


def test_without_freeze_actor_the_warmup_only_changes_the_learning_rate():
    """The default must stay exactly what it was: nothing frozen, ever."""
    from training.callbacks import ValueWarmupCallback
    model = _tiny_ppo()
    cb = ValueWarmupCallback(warmup_until=10**9, warmup_lr=1e-5, normal_lr=1e-4)
    model.learn(total_timesteps=64, callback=cb)
    assert not cb._frozen
    assert all(p.requires_grad for p in model.policy.parameters())


# ══════════════════════════════════════════════════════════════════════════
# AnchorConsolidationCallback: bound the drift from the healthy policy
# ══════════════════════════════════════════════════════════════════════════
def _anchor_rig(tmp_path, perturb):
    """A tiny PPO, a saved copy of it as the reference, an anchor set recorded
    for that reference, and optionally the live policy pushed away from it."""
    import torch
    from stable_baselines3.common.utils import configure_logger

    from common.fileio import canonical_sha256
    model = _tiny_ppo()
    model.set_logger(configure_logger(0, None, "", False))
    ref_path = str(tmp_path / "ref.zip")
    model.save(ref_path)
    rng = np.random.default_rng(1)
    states = rng.integers(0, 255, (300, 4, 84, 84), dtype=np.uint8)
    anchor_path = str(tmp_path / "anchors.npz")
    np.savez_compressed(anchor_path, states=states,
                        reference_sha256=np.str_(canonical_sha256(ref_path)))
    if perturb:
        with torch.no_grad():
            model.policy.action_net.bias.add_(torch.linspace(-3, 3, 10))
    return model, ref_path, anchor_path


def _cb(model, ref_path, anchor_path, target, max_steps=200):
    from training.callbacks import AnchorConsolidationCallback
    cb = AnchorConsolidationCallback(ref_path, anchor_path, target, batch=64,
                                     max_steps=max_steps, lr=5e-3, measure_n=300)
    cb.init_callback(model)
    cb._on_training_start()
    return cb


def test_consolidation_pulls_a_drifted_policy_back_under_budget(tmp_path):
    model, ref, anchors = _anchor_rig(tmp_path, perturb=True)
    cb = _cb(model, ref, anchors, target=0.05)
    rec = cb.consolidate()
    assert rec['kl_before'] > 0.05, "the rig did not actually drift the policy"
    assert rec['kl_after'] <= 0.05, (
        f"KL went {rec['kl_before']:.4f} -> {rec['kl_after']:.4f}; it must end "
        f"under the budget")
    assert rec['steps'] > 0


def test_under_budget_it_changes_nothing(tmp_path):
    model, ref, anchors = _anchor_rig(tmp_path, perturb=False)
    before = _snapshot(model)
    cb = _cb(model, ref, anchors, target=0.05)
    rec = cb.consolidate()
    assert rec['steps'] == 0, "a policy already under budget was trained anyway"
    after = _snapshot(model)
    assert all((before[n] == after[n]).all() for n in before)


def test_it_never_trains_the_value_head_or_the_reference(tmp_path):
    model, ref, anchors = _anchor_rig(tmp_path, perturb=True)
    value_before = {n: p.detach().clone() for n, p in model.policy.named_parameters()
                    if n.startswith("value_net")}
    cb = _cb(model, ref, anchors, target=0.01)
    ref_before = {n: p.detach().clone() for n, p in cb.reference.named_parameters()}
    cb.consolidate()
    for n, p in model.policy.named_parameters():
        if n in value_before:
            assert (value_before[n] == p).all(), f"consolidation trained {n}"
    for n, p in cb.reference.named_parameters():
        assert (ref_before[n] == p).all(), f"the frozen reference changed: {n}"


def test_it_refuses_an_anchor_set_recorded_for_another_reference(tmp_path):
    from training.callbacks import AnchorConsolidationCallback
    model, _ref, anchors = _anchor_rig(tmp_path, perturb=False)
    other = str(tmp_path / "other.zip")
    _tiny_ppo().save(other)
    cb = AnchorConsolidationCallback(other, anchors, 0.1)
    cb.init_callback(model)
    with pytest.raises(RuntimeError, match="recorded for reference"):
        cb._on_training_start()


def test_the_first_rollout_is_skipped_because_nothing_was_trained_yet(tmp_path):
    model, ref, anchors = _anchor_rig(tmp_path, perturb=True)
    cb = _cb(model, ref, anchors, target=0.05)
    cb._on_rollout_start()
    assert cb.history == [], "consolidated before any PPO update had happened"
    cb._on_rollout_start()
    assert len(cb.history) == 1


def test_the_final_update_is_consolidated_before_the_model_is_saved(tmp_path):
    """Consolidation runs at rollout START, so nothing follows the last update.
    train_agent saves as soon as learn() returns, so the budget has to be
    enforced once more at training end or the saved checkpoint can exceed it."""
    model, ref, anchors = _anchor_rig(tmp_path, perturb=True)
    cb = _cb(model, ref, anchors, target=0.05)
    cb._on_rollout_start()                   # first rollout: nothing trained yet
    assert cb.history == []
    cb._on_training_end()
    assert len(cb.history) == 1, "the final update was never consolidated"
    assert cb.history[-1]['kl_after'] <= 0.05


def test_consolidation_never_makes_the_drift_worse(tmp_path):
    """Minimising KL can overshoot. If it ends worse than it started, the
    actor is restored - the callback may only help or do nothing."""
    model, ref, anchors = _anchor_rig(tmp_path, perturb=True)
    cb = _cb(model, ref, anchors, target=1e-9, max_steps=200)
    cb.lr = 50.0                              # absurd on purpose: forces divergence
    import torch
    cb._opt = torch.optim.Adam(cb._params, lr=cb.lr)
    before = cb.measure()
    rec = cb.consolidate()
    assert rec['kl_after'] <= rec['kl_before'] + 1e-9, (
        f"consolidation left the policy worse: {rec['kl_before']:.4f} -> "
        f"{rec['kl_after']:.4f}")
    assert cb.measure() == pytest.approx(before, rel=1e-6), \
        "a diverged consolidation was not rolled back exactly"
