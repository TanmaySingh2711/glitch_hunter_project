"""evaluation/completion.py's small parts, on stand-ins, so every CI run
checks them (the protocol itself is played in the slow tests): how actions
are chosen, what is recorded about a checkpoint, and the interval maths."""
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluation import completion as ce


class _Policy:
    def __init__(self, probs):
        self.probs = torch.tensor([probs], dtype=torch.float64)
        self.training = True

    def set_training_mode(self, mode):
        self.training = mode

    def obs_to_tensor(self, obs):
        return torch.as_tensor(np.asarray(obs))[None], True

    def get_distribution(self, _t):
        return SimpleNamespace(distribution=SimpleNamespace(probs=self.probs))


def _model(probs, steps=16_000_000):
    return SimpleNamespace(policy=_Policy(probs), num_timesteps=steps)


def test_the_actor_is_inference_only():
    model = _model([0.5, 0.5])
    ce.PolicyActor(model)
    assert model.policy.training is False


def test_greedy_takes_the_top_action_and_sampled_follows_the_policy():
    model = _model([0.2, 0.8])
    rng = np.random.default_rng(3)
    greedy = ce.PolicyActor(model, "greedy")
    assert {greedy(np.zeros(1), rng) for _ in range(20)} == {1}
    sampled = ce.PolicyActor(model, "sampled")
    share = sum(sampled(np.zeros(1), rng) for _ in range(4000)) / 4000
    assert abs(share - 0.8) < 0.03


def test_the_same_seed_gives_the_same_actions():
    actor = ce.PolicyActor(_model([0.25, 0.25, 0.25, 0.25]))
    runs = [[actor(np.zeros(1), rng) for _ in range(30)]
            for rng in (np.random.default_rng(7), np.random.default_rng(7))]
    assert runs[0] == runs[1]


def test_an_unknown_action_mode_is_refused():
    with pytest.raises(ValueError, match="action mode"):
        ce.PolicyActor(_model([1.0]), "clever")


def test_checkpoint_metadata_reads_the_file_and_its_name(tmp_path, monkeypatch):
    path = tmp_path / "glitch_hunter_qa_16000000_steps.zip"
    path.write_bytes(b"not really a brain")
    monkeypatch.setattr(ce, "ROOT", str(tmp_path))
    meta = ce.checkpoint_meta(str(path), _model([1.0], steps=16_000_128))
    assert meta["path"] == "glitch_hunter_qa_16000000_steps.zip"
    assert meta["sha256"] == ce.sha256_of(str(path))
    assert (meta["num_timesteps"], meta["filename_timesteps"]) == (16_000_128, 16_000_000)
    assert meta["is_6m_master"] is False
    other = tmp_path / "brain.zip"
    other.write_bytes(b"x")
    assert ce.checkpoint_meta(str(other), _model([1.0]))["filename_timesteps"] is None
    assert os.path.exists(path)                   # read, never moved or written


def test_the_wilson_interval():
    assert ce.wilson(0, 0) == [0.0, 0.0]
    low, high = ce.wilson(283, 500)               # the 16M brain's completion rate
    assert low < 283 / 500 < high
    assert (round(low, 3), round(high, 3)) == (0.522, 0.609)
