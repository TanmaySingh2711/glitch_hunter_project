"""Objective 3 reads the final Objective-2 state and writes nowhere but its
own incident store.

A full incident cycle on the real game - capture, GIF, Markdown, PDF and the
replay subprocess - runs against a scratch store, and afterwards every file
in the project (the frozen Objective-2 folder, every checkpoint, the masks,
the manifests, the source) must be exactly as it was: same bytes, same size,
same modification time. Separately, the frozen pair is re-hashed against
its closure record.
"""
import json
import os

import pytest

from common.fileio import sha256_of
from exploration import config
from reporting.events import SyntheticProbe
from reporting.pipeline import IncidentPipeline, SessionRecorder
from reporting.provenance import session_provenance
from reporting.store import IncidentStore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SKIP_DIRS = {"venv_gpu", ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
              config.INCIDENTS_DIR, "logs"}


def _snapshot():
    snap = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            path = os.path.join(dirpath, name)
            st = os.stat(path)
            snap[os.path.relpath(path, ROOT)] = (st.st_size, st.st_mtime_ns)
    return snap


def test_a_full_incident_cycle_writes_nothing_outside_its_store(env, tmp_path):
    before = _snapshot()
    env.enable_evidence()
    env.add_detector(SyntheticProbe(700))
    env.reset()
    rec = SessionRecorder()
    rec.begin_episode(env.episode_index)
    prov = session_provenance(config.FINAL_BRAIN_PATH if os.path.exists(config.FINAL_BRAIN_PATH)
                              else None, "qa_exploration", config.CLEAN_GAME_VARIANT)
    pipe = IncidentPipeline(IncidentStore(str(tmp_path / "incidents")), reproduce=True)
    try:
        for _ in range(200):
            for _ in range(config.SUBSTEPS_PER_AGENT_STEP):
                env.step(3)
            rec.step_taken(3)
            found = env.drain_detections()
            if found:
                out = pipe.capture(found[0], rec.context("qa_exploration", prov))
                break
        assert out.status == "new"
        assert pipe.wait_idle(300)
        summary = pipe.summary(out.incident_id)
        assert summary["finalized"] and summary["reproduction"] == "reproduced"
    finally:
        pipe.close()
    after = _snapshot()
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    assert changed == [], f"Objective 3 wrote outside its store: {changed}"


def test_the_frozen_objective2_pair_still_matches_its_closure_record():
    record_path = os.path.join(ROOT, config.FINAL_OBJECTIVE2_RECORD)
    if not os.path.exists(record_path):
        pytest.skip("the frozen Objective-2 folder is not in this checkout")
    with open(record_path, encoding="utf-8") as fh:
        record = json.load(fh)
    assert sha256_of(os.path.join(ROOT, config.FINAL_BRAIN_PATH)) == record["brain"]["sha256"]
    assert sha256_of(os.path.join(ROOT, config.FINAL_COVERAGE_PATH)) == record["coverage"]["sha256"]
    for path in (config.FINAL_BRAIN_PATH, config.FINAL_COVERAGE_PATH, config.FINAL_OBJECTIVE2_RECORD):
        assert not os.access(os.path.join(ROOT, path), os.W_OK), f"{path} is writable"


def test_provenance_recognises_the_approved_brain():
    if not os.path.exists(os.path.join(ROOT, config.FINAL_BRAIN_PATH)):
        pytest.skip("the frozen Objective-2 brain is not in this checkout")
    prov = session_provenance(config.FINAL_BRAIN_PATH, "qa_exploration", config.CLEAN_GAME_VARIANT)
    assert prov["brain"]["approved_objective2_brain"] is True
    assert prov["brain"]["num_timesteps"] == 16_000_000
    assert prov["game"]["matches_pinned_clean_tree"] is True
