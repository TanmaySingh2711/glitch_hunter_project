"""artifacts.json: the protected files are still the bytes that were recorded.

Checked for every listed file that exists in this checkout - in git that is
the 6M master and the completion baseline; locally also the milestones,
backups, masks and bootstrap map. Missing files are not failures: most are
git-ignored by design.
"""
import json
import os

import pytest

from common.fileio import canonical_sha256
from evaluation.completion import SIX_M_SHA256

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "artifacts.json"), encoding="utf-8") as _fh:
    MANIFEST = json.load(_fh)


def test_the_manifest_covers_the_tracked_artifacts():
    assert "mario_brain_checkpoint.zip" in MANIFEST
    assert "evaluation/completion_baseline_6M.json" in MANIFEST


def test_every_copy_of_the_6m_brain_is_the_6m_brain():
    six_m = [k for k in MANIFEST if k.endswith(("mario_brain_checkpoint.zip",
                                                "_6000000_steps.zip"))]
    assert len(six_m) == 4
    assert {MANIFEST[k]['sha256'] for k in six_m} == {SIX_M_SHA256}


@pytest.mark.parametrize("rel", sorted(MANIFEST))
def test_the_artifact_is_unchanged(rel):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        pytest.skip(f"{rel} is not in this checkout")
    assert canonical_sha256(path) == MANIFEST[rel]['sha256'], (
        f"{rel} changed since it was recorded; if that was deliberate, re-record "
        f"with `python tools/verify_artifacts.py --record` and say why in the commit")


def test_recording_keeps_entries_that_are_not_in_protected(tmp_path, monkeypatch):
    """--record must not silently drop what was added later (the mask inputs, the
    frozen Objective-2 pair): their notes are the only record of why they exist."""
    from tools import verify_artifacts as va

    monkeypatch.setattr(va, "ROOT", str(tmp_path))
    (tmp_path / "later_added.bin").write_bytes(b"frozen pair")
    (tmp_path / "changed.bin").write_bytes(b"new bytes")
    previous = {
        "later_added.bin": {"sha256": canonical_sha256(str(tmp_path / "later_added.bin")),
                            "recorded": "2026-09-21", "note": "why it exists"},
        "changed.bin": {"sha256": "0" * 64, "recorded": "2026-09-01", "note": "kept too"},
        "not_in_this_checkout.bin": {"sha256": "1" * 64, "recorded": "2026-09-01"},
    }
    out = va.record(previous)
    assert out["later_added.bin"] == previous["later_added.bin"]        # untouched: same bytes, same date
    assert out["changed.bin"]["note"] == "kept too"
    assert out["changed.bin"]["sha256"] == canonical_sha256(str(tmp_path / "changed.bin"))
    assert out["changed.bin"]["recorded"] != "2026-09-01"               # the bytes changed, so re-dated
    assert "not_in_this_checkout.bin" not in out                        # absent files are not recorded
