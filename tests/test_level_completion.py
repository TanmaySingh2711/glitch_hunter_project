"""Phase 4F: QA training ends on Level-1 coverage, not on a step count.

    complete  <=>  covered_testable_pixels == 4,013,723      (exactly)

What has to hold for that to be trustworthy: the condition is exact (one
missing pixel is not complete, whatever a rounded percentage says), hitting
it stops training before another PPO update and writes an immutable snapshot
at once, a finished campaign cannot be trained further, a resumed campaign
continues from its exact cumulative count - and refuses, loudly, any coverage
state it cannot vouch for. Checkpoints keep global numbering, the 6M master
is never touched, and legacy mode does not change.

Built on the real verified testable mask (exploration_data/, not in git), so
these skip on a fresh clone. Numbers in brackets are the brief's test IDs.
"""
import hashlib
import inspect
import json
import os
import stat
import zipfile

import numpy as np
import pytest

import train_agent
from exploration import config
from exploration import coverage as coverage_mod
from exploration import level_completion as lc
from exploration.coverage import (
    CoverageCheckpointMismatch,
    CoverageCorrupted,
    CoverageFormatMismatch,
    SpatialCoverage,
)

TOTAL = config.TESTABLE_TOTAL
SIX_M_SHA256 = "690d57022c1fb444454b0b47f9d1e4ff1bc1444d7110c0bc3320b7fe53a188b3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def mask():
    m = coverage_mod.load_testable()
    if m is None:
        pytest.skip("exploration_data/reachable_mask.npz is not in the repository")
    return m


def _cov(mask, missing=0, shared=False):
    """A coverage state over the real mask: everything testable covered but
    the first `missing` testable pixels."""
    if shared:
        cov, _names = coverage_mod.create_shared(testable_mask=mask)
    else:
        cov = SpatialCoverage(testable_mask=mask)
    cov.visited[:] = mask
    if missing:
        ys, xs = np.nonzero(mask)
        cov.visited[ys[:missing], xs[:missing]] = 0
    cov.invalidate_remaining()
    return cov


def _first_testable(mask, k):
    ys, xs = np.nonzero(mask)
    return ys[k], xs[k]


class _FakeModel:
    """Just what the callbacks touch: a timestep, save(), and SB3's n_steps."""

    def __init__(self, steps):
        self.num_timesteps = steps
        self.saved = []

    def save(self, path):
        self.saved.append(path)
        with open(path, "wb") as fh:
            fh.write(f"model@{self.num_timesteps}".encode())


def _fake_checkpoint(path, steps):
    """A zip carrying SB3's 'data' record - all checkpoint_timesteps() reads."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data", json.dumps({"num_timesteps": steps}))


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _writable(root):
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            os.chmod(os.path.join(dirpath, f), stat.S_IREAD | stat.S_IWRITE)


@pytest.fixture
def snap_dir(tmp_path):
    d = tmp_path / lc.SNAPSHOT_DIRNAME
    yield str(d)
    _writable(str(tmp_path))            # snapshots are read-only; let pytest clean up


def _completion_cb(cov, snap_dir, check_every=1, start=(6_000_000, 0)):
    return train_agent.Level1CompletionCallback(
        cov, snapshot_dir=snap_dir, name_prefix="glitch_hunter_qa",
        session_start={'global_timestep': start[0], 'covered_testable_px': start[1]},
        check_every=check_every)


# ══════════════════════════════════════════════════════════════════════════
# [1] COVERAGE, NOT TIMESTEP
# ══════════════════════════════════════════════════════════════════════════
def test_coverage_not_timestep_decides_completion(mask, snap_dir):          # [1]
    # No step target anywhere in QA: no constant, no horizon that depends on it.
    assert not hasattr(train_agent, "TOTAL_TIMESTEPS_QA")
    assert "12_000_000" not in inspect.getsource(train_agent)
    open_ended = train_agent.steps_to_run(True, 6_000_000, False)
    assert open_ended == train_agent.steps_to_run(True, 50_000_000, False) >= 2 ** 60

    # A full map at an early step completes...
    cb = _completion_cb(_cov(mask), snap_dir)
    cb.init_callback(_FakeModel(6_000_008))
    assert cb.on_step() is False and cb.completed
    # ...an incomplete one at any step does not.
    cb = _completion_cb(_cov(mask, missing=1), snap_dir + "_b")
    cb.init_callback(_FakeModel(50_000_000))
    assert cb.on_step() is True and not cb.completed


# ══════════════════════════════════════════════════════════════════════════
# [2] [3] THE STOP AND THE SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════
def test_hitting_the_denominator_stops_training_before_any_update(mask, snap_dir):  # [2]
    """A real SB3 learn() loop. The last pixel is claimed at step 10 of the
    first rollout; learn() must end right there, with the policy weights
    identical to before - no PPO update ran after completion."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    cov = _cov(mask, missing=1)
    y, x = _first_testable(mask, 0)

    class ClaimLastPixelAt(BaseCallback):
        def _on_step(self):
            if self.n_calls == 10:
                cov.visited[y, x] = 1
            return True

    model = PPO("MlpPolicy", "CartPole-v1", n_steps=64, batch_size=64, n_epochs=1,
                device="cpu", seed=0)
    before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    done = _completion_cb(cov, snap_dir, check_every=4)
    model.learn(total_timesteps=10_000, callback=CallbackList([ClaimLastPixelAt(), done]))

    assert done.completed
    assert model.num_timesteps == 12, "stopped at the first check after completion"
    for k, v in model.policy.state_dict().items():
        assert torch.equal(v, before[k]), f"{k} was updated after completion"
    assert os.path.exists(done.snapshot['_paths']['metadata'])

    # Control: without completion the same loop DOES update - so the check
    # above can tell the difference.
    control = PPO("MlpPolicy", "CartPole-v1", n_steps=64, batch_size=64, n_epochs=1,
                  device="cpu", seed=0)
    ref = {k: v.clone() for k, v in control.policy.state_dict().items()}
    control.learn(total_timesteps=64,
                  callback=_completion_cb(_cov(mask, missing=1), snap_dir + "_c"))
    assert any(not torch.equal(v, ref[k]) for k, v in control.policy.state_dict().items())


def test_completion_is_seen_on_the_last_step_of_every_rollout(mask, snap_dir):
    """The guarantee behind 'no update after completion': whatever the
    cadence, the last collection step of a rollout is always checked."""
    cb = _completion_cb(_cov(mask), snap_dir, check_every=10 ** 9)
    cb.init_callback(_FakeModel(6_100_000))
    cb.update_locals({"n_steps": 5, "n_rollout_steps": 2048})
    assert cb.on_step() is True                        # mid-rollout, off-cadence
    cb.update_locals({"n_steps": 2047, "n_rollout_steps": 2048})
    assert cb.on_step() is False and cb.completed      # last step: always checked


def test_completion_between_checkpoints_writes_an_immediate_snapshot(mask, snap_dir, tmp_path):  # [3]
    ckpt_dir = tmp_path / "checkpoints_qa"
    steps = 6_512_344                                   # between 6.4M and 6.8M
    model = _FakeModel(6_500_000)
    milestones = train_agent.ExactMilestoneCheckpointCallback(
        every=400_000, save_path=str(ckpt_dir), name_prefix="glitch_hunter_qa")
    cov = _cov(mask, missing=3)
    done = _completion_cb(cov, snap_dir, check_every=1, start=(6_500_000, TOTAL - 3))
    milestones.init_callback(model)
    done.init_callback(model)
    model.num_timesteps = 6_510_000
    assert done.on_step() is True
    cov.visited[:] = mask
    model.num_timesteps = steps
    assert milestones.on_step() is True and done.on_step() is False

    assert not [p for p in os.listdir(ckpt_dir) if p.endswith(".zip")], "no milestone was due"
    paths = done.snapshot['_paths']
    for p in paths.values():
        assert os.path.exists(p) and not os.access(p, os.W_OK), f"{p} is not read-only"
        assert f"level1_complete_{steps}_steps" in os.path.basename(p)
    with open(paths['metadata'], encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta['kind'] == lc.SNAPSHOT_KIND
    assert (meta['covered_testable_px'], meta['testable_total'], meta['remaining_testable_px']) \
        == (TOTAL, TOTAL, 0)
    assert meta['coverage_pct'] == 100.0 and meta['global_timestep'] == steps
    assert meta['detected_after'] == {'global_timestep': 6_510_000,
                                      'covered_testable_px': TOTAL - 3}
    assert meta['policy_updates_since_completion'] == 0
    assert meta['session']['new_px_this_session'] == 3
    assert meta['model']['sha256'] == _sha(paths['model'])
    assert meta['coverage']['sha256'] == _sha(paths['coverage'])
    assert meta['reachability']['testable_fingerprint'] == config.TESTABLE_FINGERPRINT
    # The coverage file proves it on its own: it re-verifies to exactly 100%.
    back = SpatialCoverage(testable_mask=mask).load_verified(paths['coverage'], steps)
    assert back.covered_testable() == TOTAL
    # Immutable: a second completion is refused, nothing is overwritten.
    with pytest.raises(lc.LevelAlreadyCompleted):
        lc.write_completion_snapshot(model, cov, snap_dir, "glitch_hunter_qa", steps + 8,
                                     {'global_timestep': 0, 'covered_testable_px': 0}, None)


# ══════════════════════════════════════════════════════════════════════════
# [4] 99.99% IS NOT 100%
# ══════════════════════════════════════════════════════════════════════════
def test_99_99_percent_is_not_complete(mask, snap_dir):                    # [4]
    for missing in (1, 402):                           # 99.99998% and 99.990%
        assert not lc.is_level_complete(TOTAL - missing, TOTAL)
        assert lc.pct_text(TOTAL - missing) != "100%"
        assert not lc.pct_text(TOTAL - missing).startswith("100")
    assert lc.pct_text(TOTAL - 1) == "99.9999%"
    assert f"{100 * (TOTAL - 1) / TOTAL:.2f}%" == "100.00%", "why truncation is needed"
    assert lc.pct_text(TOTAL) == "100%" and lc.is_level_complete(TOTAL, TOTAL)
    with pytest.raises(ValueError):
        lc.is_level_complete(TOTAL, TOTAL + 1)          # only the verified denominator

    cov = _cov(mask, missing=1)
    cb = _completion_cb(cov, snap_dir)
    cb.init_callback(_FakeModel(9_999_992))
    assert cb.on_step() is True and not os.path.exists(snap_dir)
    with pytest.raises(ValueError):
        lc.write_completion_snapshot(_FakeModel(1), cov, snap_dir, "x", 1,
                                     {'global_timestep': 0, 'covered_testable_px': 0}, None)


def test_near_completion_remaining_pixels_are_exposed(mask):
    """Three separate leftover patches: counted, located, never removed."""
    cov = _cov(mask)
    ys, xs = np.nonzero(mask)
    picks = [0, len(ys) // 2, len(ys) - 1]              # far apart along the level
    for i in picks:
        cov.visited[ys[i], xs[i]] = 0
    cov.invalidate_remaining()
    r = lc.remaining_regions(cov)
    assert r['remaining_px'] == 3 and r['regions'] == 3
    boxes = {tuple(g['world_bbox']) for g in r['largest']}
    want = {(int(xs[i] + cov.x0), int(ys[i] + cov.y0)) * 2 for i in picks}
    assert boxes == want
    assert cov.covered_testable() == TOTAL - 3, "reporting must not change coverage"

    t = lc.PlateauTracker(window=3)
    states = [t.update(s, c) for s, c in ((10_000, 5), (20_000, 7), (30_000, 7),
                                          (40_000, 7), (50_000, 7))]
    assert [s['stalled'] for s in states] == [False, False, False, False, True]
    assert states[-1]['steps_since_last_gain'] == 30_000
    report = lc.plateau_report(cov, 50_000, states[-1])
    assert report['remaining_testable_px'] == 3 and report['level1_complete'] is False
    assert report['coverage_pct_text'] == "99.9999%" and report['stalled'] is True


# ══════════════════════════════════════════════════════════════════════════
# [5] [6] [7] RESUME
# ══════════════════════════════════════════════════════════════════════════
def test_resume_preserves_cumulative_coverage(mask, tmp_path, monkeypatch):  # [5]
    """Save at 6.8M, resume through the trainer's own preflight into a
    SHARED map, keep discovering: the count carries on from where it was."""
    saved = _cov(mask, missing=500_000)
    before = saved.covered_testable()
    monkeypatch.chdir(tmp_path)
    _fake_checkpoint("glitch_hunter_qa.zip", 6_800_000)
    saved.save("glitch_hunter_qa_coverage.npz", model_timesteps=6_800_000)

    resumed, _names = coverage_mod.create_shared(testable_mask=mask)
    try:
        path, t = train_agent.prepare_qa_coverage(resumed, "glitch_hunter_qa.zip", False)
        assert (path, t) == ("glitch_hunter_qa.zip".replace(".zip", "_coverage.npz"), 6_800_000)
        assert resumed.covered_testable() == before
        assert np.array_equal(resumed.visited, saved.visited)
        y, x = _first_testable(mask, 0)
        resumed.record({'cur_rect': (int(x + resumed.x0), int(y + resumed.y0), 1, 1)})
        assert resumed.covered_testable() == before + 1
        s = lc.status(resumed, 6_800_008, session_start_covered=before)
        assert s['session_new_px'] == 1 and s['covered_testable_px'] == before + 1
    finally:
        resumed.close()
        resumed.unlink()


def test_the_real_bootstrap_resumes_exactly():
    """The file the first QA run will load, under the strict checks."""
    boot = os.path.join(ROOT, config.BOOTSTRAP_COVERAGE)
    m = coverage_mod.load_testable()
    if m is None or not os.path.exists(boot):
        pytest.skip("exploration_data/ is not in the repository")
    digest = _sha(boot)
    cov = SpatialCoverage(testable_mask=m).load_verified(boot, expected_timesteps=6_000_000)
    assert cov.covered_testable() == 1_872_441
    assert _sha(boot) == digest


def _tamper(src, dst, **changes):
    with np.load(src, allow_pickle=False) as npz:
        d = {k: npz[k] for k in npz.files}
    for k, v in changes.items():
        if v is None:
            d.pop(k)
        else:
            d[k] = v(d[k]) if callable(v) else v
    np.savez_compressed(dst, **d)
    return dst if dst.endswith(".npz") else dst + ".npz"


def _flip_bits(packed):
    """Change one bitmap byte so its popcount is guaranteed to change (a plain
    XOR 0xFF would leave a 4-bits-set byte's count as it was)."""
    out = packed.copy()
    i = len(out) // 2
    out[i] = 0 if out[i] else 0xFF
    return out


@pytest.mark.parametrize(("name", "changes", "error"), [
    ("wrong_fingerprint", {'testable_fingerprint': np.str_("0" * 64)}, CoverageFormatMismatch),
    ("no_fingerprint", {'testable_fingerprint': None}, CoverageFormatMismatch),
    ("wrong_denominator", {'testable_total': np.int64(TOTAL - 1)}, CoverageFormatMismatch),
    ("wrong_config", {'config_hash': np.str_("f" * 64)}, CoverageFormatMismatch),
    ("bits_disagree", {'visited_packed': _flip_bits}, CoverageCorrupted),
    ("short_bitmap", {'visited_packed': lambda p: p[:-10]}, CoverageCorrupted),
    ("wrong_checkpoint", {'model_timesteps': np.int64(7_200_000)}, CoverageCheckpointMismatch),
])
def test_incompatible_coverage_fails_safely(mask, tmp_path, name, changes, error):  # [7]
    good = str(tmp_path / "good.npz")
    _cov(mask, missing=1000).save(good, model_timesteps=6_800_000)
    bad = _tamper(good, str(tmp_path / f"{name}.npz"), **changes)
    digest = _sha(bad)
    live = SpatialCoverage(testable_mask=mask)
    with pytest.raises(error):
        live.load_verified(bad, expected_timesteps=6_800_000)
    assert live.covered_testable() == 0, "a refused file must not half-load"
    assert _sha(bad) == digest, "a refused file must not be rewritten"


def test_unreadable_coverage_and_wrong_live_mask_fail_safely(mask, tmp_path):
    junk = tmp_path / "junk.npz"
    junk.write_bytes(b"PK\x03\x04 this is not a coverage file")
    with pytest.raises(CoverageCorrupted):
        SpatialCoverage(testable_mask=mask).load_verified(str(junk), 6_000_000)
    good = str(tmp_path / "good.npz")
    _cov(mask).save(good, model_timesteps=6_000_000)
    other = mask.copy()
    other[_first_testable(mask, 0)] = False
    with pytest.raises(CoverageFormatMismatch):
        SpatialCoverage(testable_mask=other).load_verified(good, 6_000_000)


def _launch_in(tmp_path, monkeypatch, mask):
    """main() in an isolated directory. Starting workers or building a model
    means training was about to begin - both fail the test."""
    def no_training(*_a, **_k):
        raise AssertionError("training was started")
    # The launch banner reads the noncoverage class map through a relative
    # path; point it at the real one before leaving the project root.
    monkeypatch.setattr(config, "REACHABLE_MASK_PATH",
                        os.path.join(ROOT, config.REACHABLE_MASK_PATH))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_agent, "SubprocVecEnv", no_training)
    monkeypatch.setattr(train_agent, "build_model", no_training)
    monkeypatch.setattr(coverage_mod, "load_testable", lambda: mask)
    monkeypatch.setattr(train_agent, "QA_PHASE", True)


def test_already_complete_coverage_prevents_training(mask, tmp_path, monkeypatch, capsys):  # [6]
    _launch_in(tmp_path, monkeypatch, mask)
    _fake_checkpoint("glitch_hunter_qa.zip", 9_600_000)
    _cov(mask).save("glitch_hunter_qa_coverage.npz", model_timesteps=9_600_000)
    before = {f: _sha(f) for f in os.listdir(".")}
    train_agent.main()
    out = capsys.readouterr().out
    assert "LEVEL 1 IS ALREADY COMPLETE" in out and "Not training further" in out
    assert {f: _sha(f) for f in os.listdir(".") if os.path.isfile(f)} == before


def test_launch_refuses_corrupted_coverage(mask, tmp_path, monkeypatch):     # [7]
    _launch_in(tmp_path, monkeypatch, mask)
    _fake_checkpoint("glitch_hunter_qa.zip", 6_800_000)
    _cov(mask, missing=10).save("good.npz", model_timesteps=6_800_000)
    _tamper("good.npz", "glitch_hunter_qa_coverage.npz", visited_packed=_flip_bits)
    digest = _sha("glitch_hunter_qa_coverage.npz")
    with pytest.raises(SystemExit) as exc:
        train_agent.main()
    assert "Refusing to resume" in str(exc.value) and "CoverageCorrupted" in str(exc.value)
    assert _sha("glitch_hunter_qa_coverage.npz") == digest


# ══════════════════════════════════════════════════════════════════════════
# [8] PERIODIC CHECKPOINTS
# ══════════════════════════════════════════════════════════════════════════
def test_periodic_checkpoints_are_global_and_open_ended(tmp_path):          # [8]
    d = tmp_path / "checkpoints_qa"

    def run(start, *steps):
        model = _FakeModel(start)
        cb = train_agent.ExactMilestoneCheckpointCallback(
            every=train_agent.QA_CHECKPOINT_EVERY, save_path=str(d),
            name_prefix="glitch_hunter_qa")
        cb.init_callback(model)
        for s in steps:
            model.num_timesteps = s
            cb.on_step()
        return [os.path.basename(p) for p in model.saved]

    assert run(6_000_000, 6_000_008, 6_399_992, 6_400_000, 6_400_008) == \
        ["glitch_hunter_qa_6400000_steps.zip"]
    # Past the old 12M "target" it simply carries on.
    assert run(12_000_000, 12_000_008, 12_400_000) == ["glitch_hunter_qa_12400000_steps.zip"]
    # Resuming at 6.5M: 6.4M is behind this run - never rewritten, even if its
    # flag has gone missing - and 6.8M is next.
    (d / ".milestone_saved_glitch_hunter_qa_6400000.flag").unlink()
    assert run(6_500_000, 6_500_008, 6_799_992, 6_800_000) == \
        ["glitch_hunter_qa_6800000_steps.zip"]
    # A restart never duplicates a milestone it already saved.
    assert run(6_790_000, 6_800_000, 6_800_008) == []
    names = sorted(p for p in os.listdir(d) if p.endswith(".zip"))
    assert names == ["glitch_hunter_qa_12400000_steps.zip",
                     "glitch_hunter_qa_6400000_steps.zip",
                     "glitch_hunter_qa_6800000_steps.zip"]
    # The completion snapshot lives in its own subdirectory, so the resume
    # search (top-level milestone zips only) can never pick it up.
    assert os.path.dirname(os.path.normpath(train_agent.LEVEL1_SNAPSHOT_DIR)) == \
        os.path.normpath(train_agent.CHECKPOINT_DIR)


# ══════════════════════════════════════════════════════════════════════════
# [9] [10] WHAT MUST NOT MOVE
# ══════════════════════════════════════════════════════════════════════════
def test_original_6m_checkpoints_are_untouched():                          # [9]
    files = ["mario_brain_checkpoint.zip", config.BASELINE_MODEL]
    found = {f: _sha(os.path.join(ROOT, f)) for f in files
             if os.path.exists(os.path.join(ROOT, f))}
    assert "mario_brain_checkpoint.zip" in found
    assert set(found.values()) == {SIX_M_SHA256}
    if train_agent.QA_PHASE:
        for p in (train_agent.LEVEL1_SNAPSHOT_DIR, train_agent.REMAINING_AUDIT_PATH):
            assert "mario_brain" not in p
            assert os.path.normpath(p).startswith(os.path.normpath(train_agent.CHECKPOINT_DIR))


def test_legacy_mode_is_unaffected():                                      # [10]
    """Legacy keeps its step budget, its formula and its fixed milestones;
    a QA safety cap never reaches it."""
    s = train_agent.steps_to_run
    assert train_agent.TOTAL_TIMESTEPS_LEGACY == 6_000_000
    assert s(False, 5_200_000, False) == 800_000
    assert s(False, 6_000_000, False) == 0
    assert s(False, 0, True) == 6_000_000
    assert s(False, 5_200_000, False, safety_cap=5_300_000) == 800_000
    assert [400_000 * i for i in range(1, 16)] == train_agent.LEGACY_CHECKPOINT_MILESTONES
    # QA's cap, by contrast, is a cut-off relative to where the run is.
    assert s(True, 6_000_000, False, safety_cap=6_020_000) == 20_000
    assert s(True, 6_030_000, False, safety_cap=6_020_000) == 0
