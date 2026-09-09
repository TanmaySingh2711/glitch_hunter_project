"""Shared fixtures.

The Mario clone creates its pygame window ONCE per process, at module-import
time (see mario_clone/data/setup.py). So every test in a run has to share a
single environment instance - building a second one would not get a second
window, and tearing one down would pull the display out from under the rest.
Hence session scope.

Nothing here loads the trained PPO model on purpose: these are fast
structural/behavioural checks, and pulling 20MB of weights onto the GPU would
turn a 5-second suite into a slow one for no extra coverage.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def env():
    from custom_mario_env import CustomMarioEnv
    e = CustomMarioEnv()
    yield e
    e.close_window()


@pytest.fixture
def fresh(env):
    """The shared env, reset to a known-good start state."""
    env.reset()
    return env


@pytest.fixture
def patch_step(env):
    """Swap env.step for one test, guaranteed restored afterwards.

    Several reward tests drive the wrapper with synthetic info dicts by
    assigning env.step. The env fixture is SESSION-scoped (the Mario clone
    creates its pygame window once per process, so it has to be), which means
    an unrestored assignment leaks into every later test in the run - the
    shared env keeps replaying one frozen frame, and unrelated tests fail with
    values that look like a reward bug rather than a fixture bug.

    Assigning env.step only shadows the bound method on the instance, so
    deleting the instance attribute is enough to restore it.
    """
    def _patch(fn):
        env.step = fn
        return fn
    yield _patch
    env.__dict__.pop('step', None)
