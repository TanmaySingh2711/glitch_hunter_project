"""tools/migrate_coverage.py: re-stamping a coverage file to a new mask.

A mask change (tools/build_reachability.py) makes every saved coverage file
unloadable, by design - it names the denominator it was scored against. The
migration must be lossless and must never take coverage away.
"""
import importlib.util
import os

import numpy as np
import pytest

from exploration import config
from exploration import coverage as cov_mod
from exploration.reachability import mask_fingerprint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location(
        "migrate_coverage", os.path.join(ROOT, "tools", "migrate_coverage.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mask(rows):
    m = np.zeros((config.GRID_H, config.GRID_W), dtype=bool)
    m[rows] = True
    return m


def _use_mask(monkeypatch, mask):
    monkeypatch.setattr(config, "TESTABLE_TOTAL", int(mask.sum()))
    monkeypatch.setattr(config, "TESTABLE_FINGERPRINT", mask_fingerprint(mask))
    monkeypatch.setattr(cov_mod, "load_testable", lambda: mask)


def _source(tmp_path, monkeypatch, old_mask, visited_rows, steps=6_032_768):
    _use_mask(monkeypatch, old_mask)
    cov = cov_mod.SpatialCoverage(testable_mask=old_mask)
    cov.visited[visited_rows, 100:200] = 1
    path = str(tmp_path / "src.npz")
    cov.save(path, model_timesteps=steps)
    return path, cov


def test_every_visit_is_preserved_and_the_numerator_is_recounted(tool, tmp_path, monkeypatch):
    old = _mask(slice(400, 600))
    src, cov = _source(tmp_path, monkeypatch, old, slice(420, 440))
    before_bits = cov.visited.copy()
    new = _mask(slice(400, 500))                 # the new mask drops rows 500-599
    _use_mask(monkeypatch, new)
    dst = str(tmp_path / "dst.npz")
    r = tool.migrate(src, dst)

    probe = cov_mod.SpatialCoverage(testable_mask=new)
    probe.load_verified(dst, expected_timesteps=6_032_768)      # the campaign would load it
    assert np.array_equal(probe.visited, before_bits), "a visit was added or dropped"
    assert r['covered_before'] == r['covered_after'] == 20 * 100
    assert r['total_after'] == int(new.sum()) < r['total_before'] == int(old.sum())
    with np.load(dst) as d:
        assert int(d['model_timesteps']) == 6_032_768
        assert str(d['migrated_from_fingerprint']) == mask_fingerprint(old)
        assert int(d['migrated_from_covered_testable']) == 2000


def test_a_migration_that_would_take_coverage_away_is_refused(tool, tmp_path, monkeypatch):
    """Coverage may never go down. Covered pixels leaving the mask means
    something reached space the new mask calls unreachable - investigate."""
    old = _mask(slice(400, 600))
    src, _ = _source(tmp_path, monkeypatch, old, slice(550, 560))   # visits in rows 550-559
    _use_mask(monkeypatch, _mask(slice(400, 500)))                  # ...which the new mask drops
    dst = tmp_path / "dst.npz"
    with pytest.raises(tool.MigrationRefused, match="may never go down"):
        tool.migrate(src, str(dst))
    assert not dst.exists(), "a refused migration still wrote a file"


def test_it_never_overwrites_its_source_or_an_existing_file(tool, tmp_path, monkeypatch):
    old = _mask(slice(400, 600))
    src, _ = _source(tmp_path, monkeypatch, old, slice(420, 440))
    with open(src, "rb") as fh:
        before = fh.read()
    with pytest.raises(tool.MigrationRefused, match="overwrite the source"):
        tool.migrate(src, src)
    existing = tmp_path / "taken.npz"
    existing.write_bytes(b"do not touch")
    with pytest.raises(tool.MigrationRefused, match="already exists"):
        tool.migrate(src, str(existing))
    with open(src, "rb") as fh:
        assert fh.read() == before
    assert existing.read_bytes() == b"do not touch"


def test_it_refuses_when_the_live_mask_is_not_the_configured_one(tool, tmp_path, monkeypatch):
    old = _mask(slice(400, 600))
    src, _ = _source(tmp_path, monkeypatch, old, slice(420, 440))
    monkeypatch.setattr(cov_mod, "load_testable", lambda: _mask(slice(0, 10)))
    with pytest.raises(tool.MigrationRefused, match="does not match"):
        tool.migrate(src, str(tmp_path / "dst.npz"))


def test_a_file_without_a_stored_numerator_needs_its_own_mask(tool, tmp_path, monkeypatch):
    """The 6M bootstrap predates covered_testable. The never-go-down check then
    needs the ORIGINAL mask - and only that exact mask is accepted."""
    old = _mask(slice(400, 600))
    src, _ = _source(tmp_path, monkeypatch, old, slice(420, 440))
    with np.load(src) as d:
        stripped = {k: d[k] for k in d.files if k != 'covered_testable'}
    legacy = str(tmp_path / "legacy.npz")
    np.savez_compressed(legacy, **stripped)
    old_path = str(tmp_path / "old_mask.npz")
    np.savez_compressed(old_path, testable_packed=np.packbits(old))
    wrong_path = str(tmp_path / "wrong_mask.npz")
    np.savez_compressed(wrong_path, testable_packed=np.packbits(_mask(slice(0, 5))))

    _use_mask(monkeypatch, _mask(slice(400, 500)))
    with pytest.raises(tool.MigrationRefused, match="source-mask"):
        tool.migrate(legacy, str(tmp_path / "a.npz"))
    with pytest.raises(tool.MigrationRefused, match="not the mask this file"):
        tool.migrate(legacy, str(tmp_path / "b.npz"), source_mask=wrong_path)
    r = tool.migrate(legacy, str(tmp_path / "c.npz"), source_mask=old_path)
    assert r['covered_before'] == r['covered_after'] == 2000
