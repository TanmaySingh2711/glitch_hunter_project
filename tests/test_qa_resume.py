"""QA-phase checkpointing, resumption, and the 6M master's safety.

The failures these guard against are all of the same shape: they do not
raise, they do not look wrong in the logs, and you discover them hours later
with the original brain already overwritten or the coverage map silently
wrong. Three specific ones:

  * The QA phase writing over mario_brain_checkpoint.zip, which would destroy
    the only copy of the completion-phase brain outside backup_6M/.
  * A model resuming beside the wrong coverage map, which either re-pays the
    agent for ground it already explored or starves it of reward for ground
    it has not.
  * Training appearing to restart from step 0 rather than continuing past
    6,000,000.
"""

import os

import numpy as np
import pytest

import train_agent
from exploration import config
from exploration.coverage import CoverageCheckpointMismatch, SpatialCoverage, create_shared


# ── Test 13: phase separation ─────────────────────────────────────────────
def test_qa_phase_follows_reward_mode():
    """One switch for the whole retrofit; the two can never disagree."""
    assert (config.REWARD_MODE == "qa_exploration") == train_agent.QA_PHASE


def test_qa_phase_never_writes_the_6m_master():
    """The single most destructive thing this retrofit could do."""
    if not train_agent.QA_PHASE:
        pytest.skip("only meaningful in QA mode")
    assert train_agent.CHECKPOINT_NAME != train_agent.LEGACY_CHECKPOINT_NAME
    assert train_agent.CHECKPOINT_DIR != train_agent.LEGACY_CHECKPOINT_DIR
    assert "mario_brain" not in train_agent.FINAL_MODEL_PATH
    assert "mario_brain" not in train_agent.COVERAGE_FINAL_PATH


def test_qa_has_no_step_target_and_continues_past_6m():
    """The completion phase ended at exactly its budget, so remaining was 0.

    QA does not replace that with a bigger number (Phase 4F): it has no step
    target at all - Level-1 coverage ends it - and its checkpoints continue
    the same 400k cadence in absolute lifetime steps, open-ended, rather than
    restarting at 400k and colliding with the completion phase's files.
    """
    if not train_agent.QA_PHASE:
        pytest.skip("only meaningful in QA mode")
    assert not hasattr(train_agent, "TOTAL_TIMESTEPS_QA")
    assert not hasattr(train_agent, "TOTAL_TIMESTEPS")
    assert train_agent.CHECKPOINT_MILESTONES is None
    assert train_agent.QA_CHECKPOINT_EVERY == 400_000
    seed = train_agent.TOTAL_TIMESTEPS_LEGACY
    first = (seed // train_agent.QA_CHECKPOINT_EVERY + 1) * train_agent.QA_CHECKPOINT_EVERY
    assert first == 6_400_000
    assert [400_000 * i for i in range(1, 16)] == train_agent.LEGACY_CHECKPOINT_MILESTONES


# ── Test 9: milestones save model and coverage as a matched pair ──────────
class _FakeModel:
    def __init__(self, steps):
        self.num_timesteps = steps
        self.saved = []

    def save(self, path):
        self.saved.append(path)
        with open(path, "w") as fh:
            fh.write("model")


def test_milestone_saves_model_coverage_and_flag(tmp_path):
    cov = SpatialCoverage()
    cov.begin_episode()
    cov.record({'cur_rect': (1000, 400, 30, 40)})

    cb = train_agent.ExactMilestoneCheckpointCallback(
        targets=[6_400_000], save_path=str(tmp_path),
        name_prefix="glitch_hunter_qa", coverage=cov)
    # The PUBLIC entry points, not the underscore ones: on_step() is what
    # syncs num_timesteps from the model, and init_callback() is what binds
    # the model in the first place. Driving the private methods directly
    # would test a code path SB3 never actually takes.
    cb.init_callback(_FakeModel(6_400_008))
    assert cb.on_step() is True

    zip_path = tmp_path / "glitch_hunter_qa_6400000_steps.zip"
    npz_path = tmp_path / "glitch_hunter_qa_6400000_steps_coverage.npz"
    flag = tmp_path / ".milestone_saved_glitch_hunter_qa_6400000.flag"
    assert zip_path.exists() and npz_path.exists() and flag.exists()

    # The flag is the commit point, so it must be the LAST thing written.
    assert flag.stat().st_mtime >= npz_path.stat().st_mtime


def test_milestone_is_saved_exactly_once(tmp_path):
    """The flag files are the only record; re-saving would rewrite history."""
    cov = SpatialCoverage()
    cb = train_agent.ExactMilestoneCheckpointCallback(
        targets=[6_400_000], save_path=str(tmp_path),
        name_prefix="glitch_hunter_qa", coverage=cov)
    cb.init_callback(_FakeModel(6_400_008))
    cb.on_step()
    cb.on_step()
    assert len(cb.model.saved) == 1

    # A fresh callback object (i.e. a restart) must honour the flag on disk.
    cb2 = train_agent.ExactMilestoneCheckpointCallback(
        targets=[6_400_000], save_path=str(tmp_path),
        name_prefix="glitch_hunter_qa", coverage=cov)
    cb2.init_callback(_FakeModel(6_500_000))
    cb2.on_step()
    assert cb2.model.saved == []


# ── Test 8: resume restores coverage and refuses a mismatch ───────────────
def test_resume_restores_coverage_exactly(tmp_path):
    cov = SpatialCoverage()
    cov.begin_episode()
    for i in range(50):
        cov.record({'cur_rect': (2000 + i * 9, 350, 30, 40)})
    total, raw = cov.total_unique(), cov.visited.copy()
    path = str(tmp_path / "qa_6400000_coverage.npz")
    cov.save(path, model_timesteps=6_400_000)

    resumed = SpatialCoverage()
    resumed.load(path, model=_FakeModel(6_400_000))
    assert resumed.total_unique() == total
    assert np.array_equal(resumed.visited, raw)


def test_resume_refuses_a_coverage_map_from_a_different_checkpoint(tmp_path):
    """Naming both step counts matters: the message has to be actionable."""
    cov = SpatialCoverage()
    cov.begin_episode()
    cov.record({'cur_rect': (3000, 350, 30, 40)})
    path = str(tmp_path / "cov.npz")
    cov.save(path, model_timesteps=6_400_000)

    fresh = SpatialCoverage()
    with pytest.raises(CoverageCheckpointMismatch) as exc:
        fresh.load(path, model=_FakeModel(7_200_000))
    assert "6,400,000" in str(exc.value) and "7,200,000" in str(exc.value)


def test_shared_coverage_survives_a_save_load_round_trip():
    """The trainer saves from the SHARED map the workers were writing into."""
    cov, _names = create_shared()
    try:
        cov.begin_episode()
        cov.record({'cur_rect': (4000, 350, 30, 40)})
        total = cov.total_unique()
        d = cov.state_dict(model_timesteps=6_400_000)
        assert int(d['total_covered']) == total
    finally:
        cov.close()
        cov.unlink()


# ── the value head reset ──────────────────────────────────────────────────
@pytest.mark.slow
def test_value_head_reset_preserves_the_policy():
    """Requirement: preserve locomotion. Only the critic may be touched.

    This loads the real 6M baseline, which is the only way to assert against
    the actual architecture rather than a stand-in.
    """
    if not os.path.exists(config.BASELINE_MODEL):
        pytest.skip(f"{config.BASELINE_MODEL} not present")
    import torch
    from stable_baselines3 import PPO

    model = PPO.load(config.BASELINE_MODEL, device="cpu")
    policy = model.policy

    action_before = {k: v.clone() for k, v in policy.action_net.state_dict().items()}
    features_before = {k: v.clone()
                       for k, v in policy.features_extractor.state_dict().items()}
    value_before = policy.value_net.weight.clone()

    assert train_agent.reset_value_head(model) is True

    for k, v in policy.action_net.state_dict().items():
        assert torch.equal(v, action_before[k]), f"action head changed at {k}"
    for k, v in policy.features_extractor.state_dict().items():
        assert torch.equal(v, features_before[k]), f"CNN features changed at {k}"
    assert not torch.equal(policy.value_net.weight, value_before)
    assert torch.all(policy.value_net.bias == 0)
    # It should now predict close to nothing, which is much nearer the truth
    # for the QA reward than anything it previously believed.
    assert float(policy.value_net.weight.abs().mean()) < 0.05
