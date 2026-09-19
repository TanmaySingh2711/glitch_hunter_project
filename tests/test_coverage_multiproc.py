"""Test 10: coverage is genuinely global across real OS processes.

This is the claim the whole retrofit rests on. Eight SubprocVecEnv workers
each hold their OWN Python interpreter, so a module-level bitmap would give
each worker a private notion of "explored" - and every worker would be paid
full novelty for the same opening stretch of the level, eight times over,
forever. Nothing in a single-process test can catch that.

So this test spawns REAL processes with the `spawn` start method, which is
what SubprocVecEnv uses on Windows and what SB3 defaults to here.
"""

import multiprocessing as mp
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mp_workers

from exploration import config
from exploration.coverage import CTRL_NOVELTY_MULT, create_shared


@pytest.fixture
def shared():
    cov, names = create_shared()
    try:
        yield cov, names
    finally:
        cov.close()
        cov.unlink()


def test_two_processes_share_one_coverage_bitmap(shared):
    """B must see A's pixels as already explored, and be paid nothing."""
    _cov, names = shared
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    marked, released = ctx.Event(), ctx.Event()

    a = ctx.Process(target=mp_workers.marker, args=(names, q, marked, released))
    b = ctx.Process(target=mp_workers.follower, args=(names, q, marked, released))
    a.start()
    b.start()
    b.join(timeout=60)
    a.join(timeout=60)
    assert a.exitcode == 0 and b.exitcode == 0, "a worker process crashed"

    results = {}
    while not q.empty():
        k, v = q.get()
        results[k] = v

    assert results['A'] == mp_workers.AREA, "A did not claim virgin ground"
    assert results['B'] == 0, (
        "process B was paid novelty for pixels process A had already "
        "explored - coverage is NOT shared across workers")
    assert results['B_total'] == mp_workers.AREA


def test_parent_sees_worker_writes(shared):
    """The parent is the authoritative reader for every reported metric."""
    cov, names = shared
    assert cov.total_unique() == 0
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    marked, released = ctx.Event(), ctx.Event()
    released.set()   # A need not wait for anyone here

    a = ctx.Process(target=mp_workers.marker, args=(names, q, marked, released))
    a.start()
    a.join(timeout=60)
    assert a.exitcode == 0

    # total_unique() is recomputed from the shared bitmap, not summed from
    # per-worker novelty claims, so it is exact even though those claims race.
    assert cov.total_unique() == mp_workers.AREA


def test_control_slot_reaches_workers(shared):
    """StagnationCallback's escalation must cross the process boundary.

    A plain config.NOVELTY_WEIGHT_MULT assignment in the parent would leave
    every worker on its spawned-in value, so the callback would look like it
    was working while changing nothing about the rollouts.
    """
    cov, names = shared
    cov.set_novelty_mult(2.0)
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    marked, released = ctx.Event(), ctx.Event()
    marked.set()

    b = ctx.Process(target=mp_workers.follower, args=(names, q, marked, released))
    b.start()
    b.join(timeout=60)
    assert b.exitcode == 0

    results = {}
    while not q.empty():
        k, v = q.get()
        results[k] = v
    assert results['B_mult'] == pytest.approx(2.0)


def test_set_novelty_mult_is_clamped(shared):
    cov, _names = shared
    cov.set_novelty_mult(99.0)
    assert cov.control[CTRL_NOVELTY_MULT] == config.NOVELTY_MULT_MAX
    cov.set_novelty_mult(0.01)
    assert cov.control[CTRL_NOVELTY_MULT] == 1.0


def test_remaining_is_never_stale_on_a_shared_bitmap():
    """A shared instance is NOT the only writer, so it may not cache.

    The bug this pins stopped the first 8-worker QA run at its first
    10,000-step report: remaining() is invalidated by record() on the same
    instance, but under shared memory the other workers' pixels arrive with
    no record() call here at all. The parent went on reporting the count it
    cached when the bootstrap map loaded, while covered_testable() read the
    live bitmap, and assert_consistent() refused to continue.
    """
    from exploration.coverage import SpatialCoverage
    mask = np.zeros((config.GRID_H, config.GRID_W), dtype=bool)
    mask[400:500, 400:9000] = True
    cov, _names = create_shared(testable_mask=mask, tag="staletest")
    try:
        before = cov.remaining()
        assert before == cov.testable_total
        # Another worker discovers 1,000 px: the bitmap moves under this
        # instance, exactly as SubprocVecEnv workers move it.
        cov.visited[400:410, 400:500] = 1
        assert cov.remaining() == cov.testable_total - cov.covered_testable()
        assert cov.remaining() == before - 1_000
        cov.assert_consistent()
    finally:
        cov.close()
        cov.unlink()

    # A private bitmap has exactly one writer, so it still caches: record()
    # is the only way its pixels can change, and it invalidates.
    local = SpatialCoverage(testable_mask=mask)
    local.remaining()
    local.visited[400:410, 400:500] = 1          # a direct edit, not a record()
    assert local.remaining() == local.testable_total, "the local cache was dropped"
    local.invalidate_remaining()                  # what a direct edit must call
    assert local.remaining() == local.testable_total - 1_000


def test_shared_and_local_backends_agree(shared):
    """One API, two backings - the call sites must not be able to tell."""
    from exploration.coverage import SpatialCoverage
    cov, _names = shared
    local = SpatialCoverage()
    r = {'cur_rect': (2000, 200, 30, 40)}
    cov.begin_episode()
    local.begin_episode()
    assert cov.record(r) == local.record(r)
    assert cov.total_unique() == local.total_unique()
    assert np.array_equal(cov.visited, local.visited)
