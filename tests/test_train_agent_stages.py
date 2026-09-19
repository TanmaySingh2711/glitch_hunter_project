"""train_agent.py's stages - resume, device, callbacks, banner, the training
call and how its end is reported - each driven on its own with a stand-in
model, so none of them needs workers, a GPU or minutes of wall clock.

The whole launch sequence (gates, refusals, the already-complete path) is
covered end to end in test_level_completion.py and test_level1_compliance.py.
"""
import logging
import os

import gymnasium as gym
import numpy as np
import pytest

import train_agent as ta
from training import callbacks as cbs


class _Model:
    def __init__(self, t=6_000_000, learn_raises=None):
        self.num_timesteps = t
        self.learned = None
        self.saved = []
        self.learn_raises = learn_raises

    def learn(self, total_timesteps, callback, reset_num_timesteps):
        if self.learn_raises:
            raise self.learn_raises
        self.learned = (total_timesteps, reset_num_timesteps)
        self.num_timesteps += 1000

    def save(self, path):
        self.saved.append(path)


class _Cov:
    testable_total = 4_013_723

    def __init__(self, remaining=100):
        self._remaining = remaining
        self.saved = []

    def covered_testable(self):
        return self.testable_total - self._remaining

    def remaining(self):
        return self._remaining

    def total_unique(self):
        return 5_000_000

    def save(self, path, model_timesteps=0):
        self.saved.append((path, model_timesteps))


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _messages(caplog):
    return [r.getMessage() for r in caplog.records]


# ── find_resume_point ──────────────────────────────────────────────────────
def test_fresh_start_ignores_every_checkpoint(in_tmp, monkeypatch):
    monkeypatch.setattr(ta, "FRESH_START", True)
    (in_tmp / f"{ta.CHECKPOINT_NAME}.zip").write_bytes(b"")
    assert ta.find_resume_point() == ta.Resume(None, False)


def test_the_master_wins_over_numbered_milestones(in_tmp, monkeypatch):
    monkeypatch.setattr(ta, "CHECKPOINT_DIR", str(in_tmp / "ckpts"))
    os.makedirs(in_tmp / "ckpts")
    (in_tmp / "ckpts" / f"{ta.CHECKPOINT_NAME}_6400000_steps.zip").write_bytes(b"")
    assert ta.find_resume_point().checkpoint.endswith("_6400000_steps.zip")
    (in_tmp / f"{ta.CHECKPOINT_NAME}.zip").write_bytes(b"")
    assert ta.find_resume_point() == ta.Resume(f"{ta.CHECKPOINT_NAME}.zip", False)


def test_the_first_qa_run_seeds_from_the_6m_master_read_only(in_tmp, monkeypatch):
    monkeypatch.setattr(ta, "QA_PHASE", True)
    monkeypatch.setattr(ta, "CHECKPOINT_NAME", "glitch_hunter_qa")
    monkeypatch.setattr(ta, "CHECKPOINT_DIR", str(in_tmp / "checkpoints_qa"))
    with pytest.raises(SystemExit, match="backup_6M"):
        ta.find_resume_point()
    (in_tmp / f"{ta.LEGACY_CHECKPOINT_NAME}.zip").write_bytes(b"")
    assert ta.find_resume_point() == ta.Resume(f"{ta.LEGACY_CHECKPOINT_NAME}.zip", True)


def test_legacy_mode_without_any_checkpoint_starts_from_scratch(in_tmp, monkeypatch):
    monkeypatch.setattr(ta, "QA_PHASE", False)
    monkeypatch.setattr(ta, "CHECKPOINT_NAME", ta.LEGACY_CHECKPOINT_NAME)
    monkeypatch.setattr(ta, "CHECKPOINT_DIR", str(in_tmp / "none"))
    assert ta.find_resume_point() == ta.Resume(None, False)


# ── campaign_coverage_path ─────────────────────────────────────────────────
def test_the_coverage_path_follows_the_checkpoint(in_tmp, monkeypatch):
    with pytest.raises(SystemExit, match="needs a checkpoint"):
        ta.campaign_coverage_path(None, False)
    monkeypatch.setattr(ta.xconfig, "BOOTSTRAP_COVERAGE", str(in_tmp / "boot.npz"))
    with pytest.raises(SystemExit, match="bootstrap_coverage"):
        ta.campaign_coverage_path("seed.zip", True)
    (in_tmp / "boot.npz").write_bytes(b"")
    assert ta.campaign_coverage_path("seed.zip", True).endswith("boot.npz")
    with pytest.raises(SystemExit, match="no matching"):
        ta.campaign_coverage_path("m_6400000_steps.zip", False)
    (in_tmp / "m_6400000_steps_coverage.npz").write_bytes(b"")
    assert ta.campaign_coverage_path("m_6400000_steps.zip", False) == \
        "m_6400000_steps_coverage.npz"


# ── select_device ──────────────────────────────────────────────────────────
def test_cpu_is_used_with_a_warning_when_cuda_is_missing(monkeypatch, caplog):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with caplog.at_level(logging.WARNING):
        assert ta.select_device() == "cpu"
    assert any("CUDA is not available" in m for m in _messages(caplog))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert ta.select_device() == "cuda"


# ── build_model ────────────────────────────────────────────────────────────
def test_resume_applies_this_machines_tensorboard_setting(monkeypatch):
    """A resumed run must not inherit the 6M brain's TensorBoard path.

    PPO.load() restores every saved hyperparameter, tensorboard_log included
    ('./logs/' in the 6M zip). SB3 then builds a writer inside learn() and
    raises ImportError where tensorboard is not installed - which stopped the
    +20k validation before its first step. The module-level detection is what
    decides for THIS machine, so build_model re-applies it, exactly as it
    already does for target_kl, batch_size and the learning rate.
    """
    from types import SimpleNamespace
    loaded = SimpleNamespace(tensorboard_log="./logs/", target_kl=0.03,
                             batch_size=256, learning_rate=1e-5, lr_schedule=None)
    monkeypatch.setattr(ta.PPO, "load", lambda *a, **k: loaded)
    model = ta.build_model("mario_brain_checkpoint.zip", None, "cpu")
    assert model.tensorboard_log == ta.TENSORBOARD_LOG
    # The restorations this file already relied on still happen.
    assert model.target_kl == 0.05
    assert model.batch_size == 512
    assert model.learning_rate == 1.0e-4 and model.lr_schedule(1.0) == 1.0e-4


def test_no_tensorboard_means_no_tensorboard_path(monkeypatch):
    """With the package absent, the run is configured for a plain console
    run rather than for a writer it cannot build."""
    monkeypatch.setattr(ta, "TENSORBOARD_LOG", None)
    from types import SimpleNamespace
    loaded = SimpleNamespace(tensorboard_log="./logs/", target_kl=0.03,
                             batch_size=256, learning_rate=1e-5, lr_schedule=None)
    monkeypatch.setattr(ta.PPO, "load", lambda *a, **k: loaded)
    assert ta.build_model("x.zip", None, "cpu").tensorboard_log is None


# ── build_callbacks ────────────────────────────────────────────────────────
def test_qa_runs_every_callback_and_legacy_only_the_two_it_always_had(monkeypatch, tmp_path):
    monkeypatch.setattr(ta, "CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setattr(ta, "QA_PHASE", True)
    cov = _Cov()
    lst, completion = ta.build_callbacks(_Model(), cov, {'global_timestep': 6_000_000,
                                                         'covered_testable_px': 0})
    kinds = [type(c) for c in lst.callbacks]
    assert kinds == [cbs.ExactMilestoneCheckpointCallback, cbs.WatchdogCallback,
                     cbs.Level1CompletionCallback, cbs.ValueWarmupCallback,
                     cbs.CoverageStatsCallback, cbs.LifecycleStatsCallback,
                     cbs.RewardTelemetryCallback, cbs.StagnationCallback]
    assert isinstance(completion, cbs.Level1CompletionCallback)
    assert lst.callbacks[0].every == ta.QA_CHECKPOINT_EVERY

    monkeypatch.setattr(ta, "QA_PHASE", False)
    monkeypatch.setattr(ta, "CHECKPOINT_MILESTONES", ta.LEGACY_CHECKPOINT_MILESTONES)
    lst, completion = ta.build_callbacks(_Model(), None, None)
    assert [type(c) for c in lst.callbacks] == [cbs.ExactMilestoneCheckpointCallback,
                                                cbs.WatchdogCallback]
    assert completion is None
    assert lst.callbacks[0].targets == ta.LEGACY_CHECKPOINT_MILESTONES


def test_the_reward_telemetry_lands_beside_the_coverage_trail():
    """Both are append-only records of one QA campaign, so they live in the
    same place - the QA checkpoint directory, which is not tracked by git."""
    for path in (ta.REWARD_TELEMETRY_PATH, ta.COVERAGE_TRAIL_PATH):
        assert os.path.dirname(os.path.normpath(path)) == \
            os.path.normpath(ta.CHECKPOINT_DIR)
    assert ta.REWARD_TELEMETRY_PATH.endswith("reward_telemetry.jsonl")


# ── log_banner ─────────────────────────────────────────────────────────────
def test_the_banner_states_success_as_coverage_not_steps(monkeypatch, caplog):
    monkeypatch.setattr(ta, "QA_PHASE", True)
    with caplog.at_level(logging.INFO):
        ta.log_banner(_Model(), ta.Resume("mario_brain_checkpoint.zip", True), _Cov(),
                      6_020_000, 20_000)
    text = "\n".join(_messages(caplog))
    assert "QA EXPLORATION" in text and "(NOT a step count)" in text
    assert "SEED - read only" in text and "6,020,000 (a cut-off, never completion)" in text
    assert "NOT written in this phase" in text

    caplog.clear()
    monkeypatch.setattr(ta, "QA_PHASE", False)
    with caplog.at_level(logging.INFO):
        ta.log_banner(_Model(1_000), ta.Resume(None, False), None, None, 5_999_000)
    text = "\n".join(_messages(caplog))
    assert "LEGACY COMPLETION" in text and "remaining        : 5,999,000" in text


# ── run_training and log_outcome ───────────────────────────────────────────
def test_nothing_is_trained_or_saved_when_no_steps_remain(monkeypatch, caplog):
    monkeypatch.setattr(ta, "QA_PHASE", True)
    model = _Model()
    with caplog.at_level(logging.INFO):
        ta.run_training(model, None, None, _Cov(), 6_000_000, 0, False)
    assert model.learned is None and model.saved == []
    assert any("Safety cap 6,000,000 already reached" in m for m in _messages(caplog))

    monkeypatch.setattr(ta, "QA_PHASE", False)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        ta.run_training(model, None, None, None, None, 0, False)
    assert any("Already at TOTAL_TIMESTEPS" in m for m in _messages(caplog))


def test_training_saves_the_pair_and_reports_the_safety_cap(monkeypatch, caplog):
    monkeypatch.setattr(ta, "QA_PHASE", True)
    model, cov = _Model(6_019_000), _Cov()
    with caplog.at_level(logging.INFO):
        ta.run_training(model, "callbacks", None, cov, 6_020_000, 1_000, False)
    assert model.learned == (1_000, False)
    assert model.saved == [ta.FINAL_MODEL_PATH]
    assert cov.saved == [(ta.COVERAGE_FINAL_PATH, 6_020_000)]
    assert any("Stopped at the SAFETY CAP (6,020,000)" in m for m in _messages(caplog))


def test_ctrl_c_saves_and_exits_cleanly(monkeypatch, caplog):
    monkeypatch.setattr(ta, "QA_PHASE", True)
    model, cov = _Model(learn_raises=KeyboardInterrupt()), _Cov()
    with caplog.at_level(logging.INFO):
        ta.run_training(model, None, None, cov, None, ta._SB3_OPEN_ENDED, False)
    assert model.saved == [ta.FINAL_MODEL_PATH] and len(cov.saved) == 1
    assert any("Saved successfully" in m for m in _messages(caplog))


@pytest.mark.parametrize(('qa', 'completed', 'cap', 'expect'), [
    (False, False, None, "Training complete!"),
    (True, True, None, "Level 1 complete."),
    (True, False, 6_000_500, "Stopped at the SAFETY CAP"),
    (True, False, None, "before Level 1 was complete"),
])
def test_every_way_training_can_end_is_named(monkeypatch, caplog, qa, completed, cap, expect):
    monkeypatch.setattr(ta, "QA_PHASE", qa)

    class _Done:
        pass
    done = _Done()
    done.completed = completed
    with caplog.at_level(logging.INFO):
        ta.log_outcome(_Model(6_001_000), done, _Cov() if qa else None, cap)
    assert any(expect in m for m in _messages(caplog))


def test_save_pair_writes_the_model_before_its_coverage(monkeypatch):
    order = []

    class _M(_Model):
        def save(self, path):
            order.append("model")

    class _C(_Cov):
        def save(self, path, model_timesteps=0):
            order.append("coverage")
    ta.save_pair(_M(), _C())
    ta.save_pair(_M(), None)
    assert order == ["model", "coverage", "model"]


def test_workers_get_one_blas_thread_unless_told_otherwise(monkeypatch):
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    ta.limit_worker_blas_threads()
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "4")
    ta.limit_worker_blas_threads()
    assert os.environ["OPENBLAS_NUM_THREADS"] == "4"


# ── build_model: the hyperparameters every incident note is about ──────────
class _ImageEnv(gym.Env):
    """The smallest env with the policy's real observation and action spaces."""
    observation_space = gym.spaces.Box(0, 255, (4, 84, 84), np.uint8)
    action_space = gym.spaces.Discrete(10)

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}


@pytest.fixture
def vec_env():
    from stable_baselines3.common.vec_env import DummyVecEnv
    v = DummyVecEnv([_ImageEnv])
    yield v
    v.close()


def _hyper(model):
    return (model.learning_rate, model.lr_schedule(1.0), model.batch_size, model.target_kl)


def test_a_fresh_model_gets_the_tuned_hyperparameters(vec_env, monkeypatch):
    monkeypatch.setattr(ta, "TENSORBOARD_LOG", None)
    model = ta.build_model(None, vec_env, "cpu")
    assert _hyper(model) == (1.0e-4, 1.0e-4, 512, 0.05)
    assert model.ent_coef == ta.xconfig.ENT_COEF_BASE
    assert (model.n_steps, model.n_epochs, model.gamma) == (2048, 4, 0.99)


@pytest.mark.slow
def test_a_resumed_model_is_retuned_after_load(vec_env):
    """PPO.load() restores the checkpoint's own settings, including a
    learning-rate SCHEDULE the optimizer reads - so each has to be re-applied,
    the schedule included, or the tuning silently does nothing."""
    path = os.path.join(os.path.dirname(ta.__file__), "mario_brain_checkpoint.zip")
    model = ta.build_model(path, vec_env, "cpu")
    assert _hyper(model) == (1.0e-4, 1.0e-4, 512, 0.05)
    assert model.num_timesteps == 6_000_000
