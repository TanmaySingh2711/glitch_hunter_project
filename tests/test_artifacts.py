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
