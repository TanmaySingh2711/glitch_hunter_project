"""training/value_head.py on a tiny stand-in policy, so it runs in every CI
run (tests/test_qa_resume.py checks the same reset on the real brain, as a
slow test): only the critic's output layer changes, and only its optimizer
state is dropped."""
from types import SimpleNamespace

import torch

from training.value_head import reset_value_head


def _model():
    torch.manual_seed(0)
    features = torch.nn.Linear(4, 8)
    action_net = torch.nn.Linear(8, 10)
    value_net = torch.nn.Linear(8, 1)
    params = [*features.parameters(), *action_net.parameters(), *value_net.parameters()]
    optimizer = torch.optim.Adam(params, lr=1e-3)
    loss = (value_net(features(torch.ones(2, 4))) ** 2).sum() + action_net(features(torch.ones(2, 4))).sum()
    loss.backward()
    optimizer.step()                       # every parameter now has Adam state
    policy = SimpleNamespace(value_net=value_net, action_net=action_net, features=features,
                             optimizer=optimizer)
    return SimpleNamespace(policy=policy)


def test_only_the_value_head_is_reset():
    model = _model()
    p = model.policy
    kept = {name: t.detach().clone() for name, t in (("features.w", p.features.weight),
                                                    ("action.w", p.action_net.weight),
                                                    ("action.b", p.action_net.bias))}
    old = p.value_net.weight.detach().clone()
    assert reset_value_head(model) is True
    assert not torch.equal(p.value_net.weight, old)
    assert float(p.value_net.weight.detach().abs().mean()) < float(old.abs().mean())   # near-zero start
    assert torch.count_nonzero(p.value_net.bias) == 0
    assert torch.equal(p.features.weight, kept["features.w"])
    assert torch.equal(p.action_net.weight, kept["action.w"])
    assert torch.equal(p.action_net.bias, kept["action.b"])


def test_only_the_value_heads_optimizer_state_is_cleared():
    model = _model()
    p = model.policy
    state = p.optimizer.state
    assert p.value_net.weight in state and p.action_net.weight in state
    reset_value_head(model)
    assert p.value_net.weight not in state and p.value_net.bias not in state
    assert p.action_net.weight in state and p.features.weight in state


def test_a_policy_without_a_value_head_is_left_alone(caplog):
    model = SimpleNamespace(policy=SimpleNamespace())
    assert reset_value_head(model) is False
    assert "no value_net" in caplog.text
