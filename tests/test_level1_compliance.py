"""Level 1-1 compliance: what surrounds completion, not the condition itself.

  * provenance    bootstrap-known vs QA-discovered pixels, recorded, never
                  weighted, never fabricated
  * remaining map an offline picture of what is left, for spotting islands
  * audit trail   an append-only history of coverage growth across restarts
  * launch gate   no unrestricted QA campaign starts by default
  * verification  a completed snapshot is a candidate until it passes
                  integrity, health and completion retention

Built on the real verified testable mask (exploration_data/, not in git), so
most of these skip on a fresh clone.
"""
import json
import os
import stat
import zipfile

import numpy as np
import pytest

import train_agent
from evaluation import level1_verification as lv
from exploration import config
from exploration import coverage as coverage_mod
from exploration import level_completion as lc
from exploration.coverage import SpatialCoverage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOTAL = config.TESTABLE_TOTAL


@pytest.fixture(scope="module")
def mask():
    m = coverage_mod.load_testable()
    if m is None:
        pytest.skip("exploration_data/reachable_mask.npz is not in the repository")
    return m


@pytest.fixture(scope="module")
def bootstrap(mask):
    path = os.path.join(ROOT, config.BOOTSTRAP_COVERAGE)
    if not os.path.exists(path):
        pytest.skip("the bootstrap coverage is not in the repository")
    cov = SpatialCoverage(testable_mask=mask)
    cov.load(path)
    return path, cov


def _full(mask, missing=0):
    cov = SpatialCoverage(testable_mask=mask)
    cov.visited[:] = mask
    if missing:
        ys, xs = np.nonzero(mask)
        cov.visited[ys[:missing], xs[:missing]] = 0
    cov.invalidate_remaining()
    return cov


def _writable(root):
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            os.chmod(os.path.join(dirpath, f), stat.S_IREAD | stat.S_IWRITE)


# ══════════════════════════════════════════════════════════════════════════
# PROVENANCE
# ══════════════════════════════════════════════════════════════════════════
def test_bootstrap_known_and_qa_discovered_are_told_apart(mask, bootstrap):
    path, boot = bootstrap
    p = lc.provenance(boot, path)
    assert p['available'] and p['bootstrap']['model_timesteps'] == 6_000_000
    assert p['bootstrap_known_testable_px'] == boot.covered_testable() == 1_872_441
    assert p['qa_discovered_testable_px'] == 0

    # QA training then finds 500 px the bootstrap never touched...
    grown = SpatialCoverage(testable_mask=mask)
    grown.visited[:] = boot.visited
    fresh = np.argwhere(mask & (boot.visited == 0))[:500]
    grown.visited[fresh[:, 0], fresh[:, 1]] = 1
    q = lc.provenance(grown, path)
    assert q['qa_discovered_testable_px'] == 500
    assert q['bootstrap_px_still_covered'] == q['bootstrap_known_testable_px']
    # ...and every one of them counts toward completion exactly like the rest.
    assert grown.covered_testable() == 1_872_441 + 500 == q['covered_testable_px']


def test_provenance_never_fabricates_missing_history(mask, tmp_path):
    """No bootstrap, no split - not a guess at what 0-6M might have covered."""
    p = lc.provenance(_full(mask), str(tmp_path / "absent.npz"))
    assert p == {'available': False, 'reason': f"{tmp_path / 'absent.npz'} not found"}


# ══════════════════════════════════════════════════════════════════════════
# THE REMAINING MAP
# ══════════════════════════════════════════════════════════════════════════
def test_a_single_leftover_pixel_is_visible_on_the_map(mask, tmp_path):
    cov = _full(mask, missing=1)
    ys, xs = np.nonzero(mask)
    img = lc.remaining_map_image(cov, scale=4)
    assert tuple(img[ys[0] // 4, xs[0] // 4]) == (40, 40, 255), "the island is not red"
    assert int(np.count_nonzero((img == (40, 40, 255)).all(axis=2))) == 1
    png = lc.write_remaining_map(cov, str(tmp_path / "m.png"), lc.remaining_regions(cov))
    assert os.path.getsize(png) > 0
    assert cov.covered_testable() == TOTAL - 1, "drawing the map changed coverage"


def test_the_offline_map_tool_reads_any_coverage_file(mask, bootstrap, tmp_path):
    from tools import remaining_coverage_map as tool
    src = str(tmp_path / "cov.npz")
    _full(mask, missing=3).save(src, model_timesteps=7_000_000)
    before = lc.sha256_of(src)
    assert tool.main([src, '--out', str(tmp_path / 'out')]) == 0
    with open(tmp_path / 'out' / 'cov_remaining_regions.json', encoding='utf-8') as fh:
        report = json.load(fh)
    assert report['remaining_testable_px'] == 3 and report['level1_complete'] is False
    assert report['coverage_pct_text'] == "99.9999%"
    assert (tmp_path / 'out' / 'cov_remaining_map.png').exists()
    assert lc.sha256_of(src) == before, "the tool wrote to the coverage file"


# ══════════════════════════════════════════════════════════════════════════
# THE AUDIT TRAIL
# ══════════════════════════════════════════════════════════════════════════
class _Model:
    logger = None

    def __init__(self, t):
        self.num_timesteps = t


def _report(cb, t, dones=()):
    cb.model.num_timesteps = t
    cb.update_locals({'dones': list(dones)})
    cb.on_step()


def test_coverage_growth_is_appended_across_restarts(mask, tmp_path):
    trail = str(tmp_path / "coverage_audit_trail.jsonl")
    cov = _full(mask, missing=5_000)

    class _Ckpt:
        last_saved = "checkpoints_qa/glitch_hunter_qa_6400000_steps.zip"

    cb = train_agent.CoverageStatsCallback(cov, every=10_000, trail_path=trail,
                                           session_start_covered=TOTAL - 6_000,
                                           checkpoints=_Ckpt())
    cb.init_callback(_Model(6_400_000))
    _report(cb, 6_400_008, dones=[True, False, True])
    ys, xs = np.nonzero(mask)
    cov.visited[ys[:1_000], xs[:1_000]] = 1              # 1,000 new pixels
    cov.invalidate_remaining()
    _report(cb, 6_410_008)

    # A restart: a new callback appends to the same file, never rewrites it.
    cb2 = train_agent.CoverageStatsCallback(cov, every=10_000, trail_path=trail,
                                            session_start_covered=cov.covered_testable())
    cb2.init_callback(_Model(6_410_008))
    _report(cb2, 6_420_008)

    with open(trail, encoding='utf-8') as fh:
        rows = [json.loads(line) for line in fh]
    assert [r['global_timestep'] for r in rows] == [6_400_008, 6_410_008, 6_420_008]
    assert [r['remaining_testable_px'] for r in rows] == [5_000, 4_000, 4_000]
    assert rows[1]['covered_testable_px'] - rows[0]['covered_testable_px'] == 1_000
    assert rows[1]['new_px_per_10k'] == 1_000.0
    assert rows[0]['session_new_px'] == 1_000 and rows[2]['session_new_px'] == 0
    assert rows[0]['episodes_this_session'] == 2
    assert rows[0]['last_checkpoint_this_session'].endswith("6400000_steps.zip")
    assert rows[1]['coverage_pct_text'] == lc.pct_text(TOTAL - 4_000)
    for key in ('testable_total', 'bootstrap_known_px', 'qa_discovered_px', 'stalled',
                'written_at'):
        assert key in rows[0]


# ══════════════════════════════════════════════════════════════════════════
# THE LAUNCH GATE
# ══════════════════════════════════════════════════════════════════════════
def test_an_unrestricted_campaign_needs_explicit_approval():
    with pytest.raises(SystemExit) as exc:
        train_agent.check_launch_gate(True, None, False)
    msg = str(exc.value)
    assert "--safety-cap-timesteps 6020000" in msg and "--unrestricted" in msg
    assert "Nothing was trained" in msg
    train_agent.check_launch_gate(True, 6_020_000, False)       # the validation run
    train_agent.check_launch_gate(True, None, True)             # explicitly approved
    train_agent.check_launch_gate(False, None, False)           # legacy: its own budget


def test_the_gate_stops_a_real_launch_before_any_worker(mask, tmp_path, monkeypatch):
    def started(*_a, **_k):
        raise RuntimeError("workers started")
    monkeypatch.setattr(config, "REACHABLE_MASK_PATH", os.path.join(ROOT, config.REACHABLE_MASK_PATH))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_agent, "SubprocVecEnv", started)
    monkeypatch.setattr(coverage_mod, "load_testable", lambda: mask)
    monkeypatch.setattr(train_agent, "QA_PHASE", True)
    with zipfile.ZipFile("glitch_hunter_qa.zip", "w") as z:
        z.writestr("data", json.dumps({"num_timesteps": 6_020_000}))
    _full(mask, missing=10).save("glitch_hunter_qa_coverage.npz", model_timesteps=6_020_000)
    with pytest.raises(SystemExit, match="Refusing an unrestricted QA campaign"):
        train_agent.main([])
    assert sorted(os.listdir(".")) == ["glitch_hunter_qa.zip", "glitch_hunter_qa_coverage.npz"]
    with pytest.raises(RuntimeError, match="workers started"):
        train_agent.main(["--safety-cap-timesteps", "6040000"])
    with pytest.raises(RuntimeError, match="workers started"):
        train_agent.main(["--unrestricted"])


# ══════════════════════════════════════════════════════════════════════════
# VERIFICATION — complete is not yet final
# ══════════════════════════════════════════════════════════════════════════
class _ZipModel:
    """save() writes a zip with SB3's 'data' record, which is all the
    integrity check reads from it."""

    def __init__(self, t):
        self.num_timesteps = t

    def save(self, path):
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("data", json.dumps({"num_timesteps": self.num_timesteps}))


class _Policy:
    def __init__(self, value=0.5):
        import torch
        self.policy = torch.nn.Linear(4, 2)
        with torch.no_grad():
            self.policy.weight.fill_(value)


@pytest.fixture
def snapshot(mask, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(coverage_mod, "load_testable", lambda: mask)
    snap = os.path.join("checkpoints_qa", lc.SNAPSHOT_DIRNAME)
    meta = lc.write_completion_snapshot(
        _ZipModel(7_203_328), _full(mask), snap, "glitch_hunter_qa", 7_203_328,
        {'global_timestep': 6_000_000, 'covered_testable_px': 1_872_441}, (7_201_280, TOTAL - 9))
    yield meta['_paths']
    _writable(str(tmp_path))


def _verify(snapshot, tmp_path, verdict='HEALTHY', policy=None, calls=None):
    def evaluate(model_path, protocol, workers=1):
        if calls is not None:
            calls.append(model_path)
        return {'protocol': protocol, 'checkpoint': {'num_timesteps': 7_203_328}}

    def compare(result, baseline):
        return {'verdict': verdict, 'reasons': [] if verdict == 'HEALTHY' else ['x'],
                'completion_rate': 0.9, 'baseline_completion_rate': 0.9,
                'completion_delta': 0.0, 'mean_progress': 0.95,
                'baseline_mean_progress': 0.95, 'greedy_completes': True}

    return lv.verify(snapshot['metadata'], root=str(tmp_path), evaluate=evaluate,
                     compare=compare, load_policy=lambda _p: policy or _Policy(),
                     results_dir=str(tmp_path / "results"), log=lambda *_a: None)


def test_a_healthy_completion_becomes_the_final_brain(snapshot, tmp_path):
    before = {k: lc.sha256_of(p) for k, p in snapshot.items()}
    rec = _verify(snapshot, tmp_path)
    assert rec['verdict'] == lv.VERIFIED
    assert all(c['passed'] for c in rec['checks'].values())
    assert rec['final_level1_brain']['sha256'] == before['model']
    assert not os.access(rec['_path'], os.W_OK), "the record is not read-only"
    assert {k: lc.sha256_of(p) for k, p in snapshot.items()} == before, "the snapshot changed"
    found = lc.final_level1_brain(os.path.dirname(snapshot['metadata']))
    assert found and found[1]['verdict'] == lv.VERIFIED
    # The candidate list is unaffected: the record is not a second completion.
    assert len(lc.committed_snapshots(os.path.dirname(snapshot['metadata']))) == 1
    with pytest.raises(FileExistsError):
        _verify(snapshot, tmp_path)


@pytest.mark.parametrize(("retention", "verdict"), [('WARNING', lv.NEEDS_REVIEW),
                                               ('REGRESSED', lv.REJECTED)])
def test_lost_completion_ability_is_not_a_final_brain(snapshot, tmp_path, retention, verdict):
    rec = _verify(snapshot, tmp_path, verdict=retention)
    assert rec['verdict'] == verdict and rec['final_level1_brain'] is None
    assert lc.final_level1_brain(os.path.dirname(snapshot['metadata'])) is None


def test_a_tampered_snapshot_is_rejected_before_any_evaluation(snapshot, tmp_path):
    os.chmod(snapshot['model'], stat.S_IREAD | stat.S_IWRITE)
    _ZipModel(7_203_329).save(snapshot['model'])
    calls = []
    rec = _verify(snapshot, tmp_path, calls=calls)
    assert rec['verdict'] == lv.REJECTED and calls == []
    assert any("changed since completion" in f for f in rec['checks']['integrity']['failures'])


def test_an_unhealthy_policy_is_rejected_before_any_evaluation(snapshot, tmp_path):
    calls = []
    rec = _verify(snapshot, tmp_path, policy=_Policy(float('nan')), calls=calls)
    assert rec['verdict'] == lv.REJECTED and calls == []
    assert rec['checks']['health']['failures']
