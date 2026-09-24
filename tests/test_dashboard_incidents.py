"""The dashboard side of Objective 3.

  * the service: a new incident stops testing on the game thread, before the
    next step, and stays "bug found" until the USER resumes or resets
  * the backend: each step's detections reach the pipeline with the right
    context - frames from before the trigger only
  * the web routes: status, history and evidence files, and nothing outside
    the incident store can be fetched through them
  * brain selection: the approved Objective-2 brain, and only if its hash matches
"""
import hashlib
import io
import os
import time
import zipfile

import numpy as np
import pytest
from incident_helpers import context, detection
from test_dashboard_control import FakeBackend, wait_for

import dashboard_backend as db
import dashboard_service as ds
from exploration import config
from reporting.pipeline import IncidentPipeline
from reporting.store import IncidentStore


# ══════════════════════════════════════════════════════════════════════════
# The service: Bug Found is backend state, and only the user clears it
# ══════════════════════════════════════════════════════════════════════════
class IncidentBackend(FakeBackend):
    """A session that reports the given outcomes on the given steps."""

    def __init__(self, plan):
        super().__init__()
        self.plan = plan

    def new_session(self):
        self._rec('new_session')

        def run():
            n = 0
            while True:
                n += 1
                yield {'frame': b'', 'log': f"step {n}", 'incidents': self.plan.get(n, [])}
        return run()


def _summary(i="INC-20260923-120000-aaaaaa", **kw):
    return {"incident_id": i, "title": "Impossible horizontal speed", "category": "speed",
            "occurrences": 1, **kw}


@pytest.fixture
def make_svc():
    made = []

    def make(plan):
        backend, events = IncidentBackend(plan), []
        s = ds.GameWindowService(backend, emit=lambda e, p: events.append((e, p)),
                                 log=lambda *a: None)
        s.start(timeout=5)
        s.events = events
        made.append(s)
        return s
    yield make
    for s in made:
        s.shutdown()


def _names(s):
    return [e for e, _p in s.events]


def test_a_new_incident_stops_testing_before_another_step(make_svc):
    s = make_svc({3: [{"status": "new", "incident_id": "INC-20260923-120000-aaaaaa",
                       "summary": _summary()}]})
    s.start_testing()
    assert wait_for(lambda: s.bug_found is not None)
    assert s.testing is False and s.steps == 3, "stepped past the bug"
    time.sleep(0.2)
    assert s.steps == 3, "testing resumed by itself"
    st = s.status()
    assert st["pause_reason"] == "bug_found" and st["bug_found"][0]["category"] == "speed"
    assert ("testing_paused", {"reason": "bug_found"}) in s.events
    assert ("bug_found", {"incidents": [_summary()]}) in s.events


def test_only_the_user_resumes_and_the_session_carries_on(make_svc):
    s = make_svc({2: [{"status": "new", "summary": _summary()}]})
    s.start_testing()
    assert wait_for(lambda: s.bug_found is not None)
    s.start_testing()
    assert wait_for(lambda: s.steps >= 6)
    assert s.bug_found is None and "bug_cleared" in _names(s)
    logs = [p["log"] for e, p in s.events if e == "agent_log"]
    assert logs[:6] == [f"step {n}" for n in range(1, 7)], "resume started a new session"


def test_reset_clears_the_bug_but_not_the_evidence(make_svc):
    s = make_svc({1: [{"status": "new", "summary": _summary()}]})
    s.start_testing()
    assert wait_for(lambda: s.bug_found is not None)
    s.reset()
    s.wait_idle()
    assert s.bug_found is None and s.status()["pause_reason"] == "reset"


def test_a_repeat_sighting_stops_testing_too_and_is_announced(make_svc):
    """Every detection stops testing (the owner's rule); a repeat is the same
    incident with a higher count, not a new one."""
    s = make_svc({2: [{"status": "duplicate", "summary": _summary(occurrences=2)}]})
    s.start_testing()
    assert wait_for(lambda: not s.testing and s.steps == 2)
    assert s.status()["pause_reason"] == "bug_found"
    assert s.bug_found == [_summary(occurrences=2)]
    assert ("incident_occurrence", _summary(occurrences=2)) in s.events
    s.start_testing()
    assert wait_for(lambda: s.steps >= 5) and s.bug_found is None


def test_a_known_bug_seen_first_time_in_a_new_session_stops_testing(make_svc):
    """After a Reset the Bug Tracker is empty: a bug already in the store is
    new to this session, so it stops testing like a new one."""
    s = make_svc({2: [{"status": "duplicate", "first_in_session": True,
                       "summary": _summary(occurrences=3)}]})
    s.start_testing()
    assert wait_for(lambda: not s.testing and s.steps == 2)
    assert s.status()["pause_reason"] == "bug_found"
    assert s.bug_found == [_summary(occurrences=3)]


def test_an_unrecorded_detection_still_stops_testing_and_says_why(make_svc):
    s = make_svc({2: [{"status": "failed", "error": "OSError: disk full", "summary": None}]})
    s.start_testing()
    assert wait_for(lambda: not s.testing and s.steps == 2)
    assert s.status()["pause_reason"] == "capture_failed"
    assert ("incident_capture_failed", {"error": "OSError: disk full"}) in s.events


# ══════════════════════════════════════════════════════════════════════════
# The backend: detections become incidents with the right context
# ══════════════════════════════════════════════════════════════════════════
class _Base:
    def __init__(self, plan):
        self.plan, self.step_no, self.episode_index = plan, 0, 0
        self.pending = []

    def render_scaled(self, size):
        return np.full((size[1], size[0], 3), self.step_no % 255, dtype=np.uint8)

    def drain_detections(self):
        out, self.pending = self.pending, []
        return out


class _Env:
    def __init__(self, plan, episode_len=50):
        self.unwrapped = _Base(plan)
        self.t, self.episode_len = 0, episode_len

    def reset(self, **_kw):
        self.t = 0
        self.unwrapped.episode_index += 1
        return np.zeros((4, 84, 84), dtype=np.uint8), {}

    def step(self, action):
        self.t += 1
        b = self.unwrapped
        b.step_no += 1
        if (b.episode_index, self.t) in b.plan:
            b.pending.append(b.plan[(b.episode_index, self.t)])
        return np.zeros((4, 84, 84), dtype=np.uint8), 0.0, self.t >= self.episode_len, False, {}


class _Model:
    def predict(self, obs, deterministic):
        return np.array(3), None


@pytest.fixture
def live_pipeline(tmp_path, monkeypatch):
    from incident_helpers import provenance
    pipe = IncidentPipeline(IncidentStore(str(tmp_path / "incidents")), reproduce=False)
    monkeypatch.setattr(db, "_pipeline", pipe)
    monkeypatch.setattr(db, "_provenance", provenance())
    monkeypatch.setattr(db, "_reward_mode", "qa_exploration")
    yield pipe
    pipe.close()


def test_each_detection_becomes_an_incident_with_only_earlier_frames(live_pipeline, monkeypatch):
    sub = config.SUBSTEPS_PER_AGENT_STEP
    plan = {(1, 10): detection(kind="speed", x=2000, substep=10 * sub - 2, episode=1),
            (2, 10): detection(kind="speed", x=2003, substep=10 * sub - 2, episode=2)}
    monkeypatch.setattr(db, "_global_env", _Env(plan, episode_len=20))
    monkeypatch.setattr(db, "_global_model", _Model())
    session = db.run_mario_agent()
    items = [next(session) for _ in range(30)]
    session.close()
    first = items[9]["incidents"]
    assert [o["status"] for o in first] == ["new"]
    record = live_pipeline.store.load_record(first[0]["incident_id"])
    assert record["evidence"]["context_frames"]["count"] == 9, "the trigger step's own frame leaked in"
    assert record["evidence"]["context_frames"]["last_episode_agent_step"] == 9
    assert record["timing"]["episode_agent_step"] == 10 and record["consistency"]["episode_match"]
    second = items[29]["incidents"]
    assert [o["status"] for o in second] == ["duplicate"]
    assert second[0]["incident_id"] == first[0]["incident_id"]
    assert all(it["incidents"] == [] for n, it in enumerate(items) if n not in (9, 29))


def test_detections_without_a_pipeline_are_reported_as_failures(monkeypatch):
    monkeypatch.setattr(db, "_pipeline", None)
    monkeypatch.setattr(db, "_global_env", _Env({(1, 1): detection()}))
    monkeypatch.setattr(db, "_global_model", _Model())
    session = db.run_mario_agent()
    item = next(session)
    session.close()
    assert item["incidents"][0]["status"] == "failed"


# ══════════════════════════════════════════════════════════════════════════
# The web routes
# ══════════════════════════════════════════════════════════════════════════
@pytest.fixture
def served(live_pipeline, monkeypatch):
    import app
    ids = []
    for i, kind in enumerate(("speed", "below_world", "synthetic_probe")):
        det = detection(kind=kind, x=1000 + 800 * i, synthetic=kind == "synthetic_probe")
        ids.append(live_pipeline.capture(det, context(det)).incident_id)
    assert live_pipeline.wait_idle(120)
    return app, app.app.test_client(), ids


def test_status_reports_the_backend_truth(served, monkeypatch):
    app, client, _ = served
    monkeypatch.setattr(app.service, "bug_found", [{"incident_id": "X"}])
    monkeypatch.setattr(app.service, "testing", False)
    monkeypatch.setattr(app.service, "pause_reason", "bug_found")
    st = client.get("/api/status").get_json()
    assert st["bug_found"] == [{"incident_id": "X"}] and st["pause_reason"] == "bug_found"
    assert st["game_variant"] == config.CLEAN_GAME_VARIANT


def test_the_history_lists_every_incident_newest_first(served):
    _app, client, ids = served
    listed = client.get("/api/incidents?all=1").get_json()["incidents"]
    assert [i["incident_id"] for i in listed] == sorted(ids, reverse=True)
    synthetic = [i for i in listed if i["synthetic"]]
    assert len(synthetic) == 1 and synthetic[0]["confidence"] == "not_applicable"
    for inc in listed:
        assert {"report.pdf", "report.md", "trigger.png", "context.gif"} <= set(inc["available"])
        assert inc["finalized"] and inc["occurrences"] == 1


def test_the_bug_tracker_lists_this_session_and_reset_empties_it(served):
    """Reset starts a new session list: the tracker is empty again, yet
    every incident is still in the store (?all=1) and still served."""
    app, client, ids = served
    pipe = db._pipeline
    pipe.begin_session()
    body = client.get("/api/incidents").get_json()
    assert body == {"incidents": [], "scope": "session"}
    assert len(client.get("/api/incidents?all=1").get_json()["incidents"]) == len(ids)
    assert client.get(f"/incidents/{ids[0]}/report.md").status_code == 200
    app.backend.begin_incident_session()                 # what the service's Reset calls
    assert client.get("/api/incidents").get_json()["incidents"] == []


def test_a_new_or_repeated_sighting_joins_the_session_list():
    import threading

    from reporting.pipeline import IncidentPipeline
    pipe = IncidentPipeline.__new__(IncidentPipeline)
    pipe._lock, pipe._session = threading.RLock(), {"INC-A": 1, "INC-B": 2}
    pipe.summary = lambda i: {"incident_id": i}              # type: ignore[method-assign]
    assert [s["incident_id"] for s in pipe.session_summaries()] == ["INC-B", "INC-A"]
    assert pipe.session_summaries()[0]["seen_this_session"] == 2
    pipe.begin_session()
    assert pipe.session_summaries() == []


def test_detail_and_files_are_served(served):
    _app, client, ids = served
    detail = client.get(f"/api/incidents/{ids[0]}").get_json()
    assert detail["record"]["incident_id"] == ids[0] and detail["manifest"]["finalized"]
    pdf = client.get(f"/incidents/{ids[0]}/report.pdf")
    assert pdf.status_code == 200 and pdf.mimetype == "application/pdf"
    assert pdf.data.startswith(b"%PDF-")
    assert pdf.headers["X-Content-Type-Options"] == "nosniff"
    assert "attachment" not in pdf.headers.get("Content-Disposition", "")
    md = client.get(f"/incidents/{ids[0]}/report.md")
    assert md.mimetype == "text/plain" and md.data.startswith(b"# Incident")
    dl = client.get(f"/incidents/{ids[0]}/report.md?download=1")
    assert "attachment" in dl.headers["Content-Disposition"]
    assert f"{ids[0]}_report.md" in dl.headers["Content-Disposition"]
    png = client.get(f"/incidents/{ids[0]}/trigger.png")
    assert png.mimetype == "image/png" and png.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_bundle_download_holds_the_evidence_and_nothing_else(served):
    _app, client, ids = served
    folder = db._pipeline.store.bundle_dir(ids[1])
    with open(os.path.join(folder, "stray-notes.txt"), "w", encoding="utf-8") as fh:
        fh.write("not evidence")
    res = client.get(f"/incidents/{ids[1]}/bundle.zip")
    assert res.status_code == 200 and res.mimetype == "application/zip"
    with zipfile.ZipFile(io.BytesIO(res.data)) as zf:
        names = sorted(zf.namelist())
    assert f"{ids[1]}/incident.json" in names and f"{ids[1]}/report.pdf" in names
    assert not any("stray" in n for n in names)


@pytest.mark.parametrize("url", [
    "/incidents/{id}/..%2Fmanifest.json",
    "/incidents/{id}/%2e%2e%2f%2e%2e%2fapp.py",
    "/incidents/{id}/..\\..\\app.py",
    "/incidents/..%2F..%2Fapp.py/incident.json",
    "/incidents/{id}/app.py",
    "/incidents/{id}/stray-notes.txt",
    "/incidents/INC-20260923-120000-abcdef/report.pdf",
    "/incidents/not-an-id/report.pdf",
    "/incidents/{id}/report.v1.pdf",
    "/api/incidents/..%2F..%2Fapp.py",
    "/api/incidents/INC-20260923-120000-abcdef",
    "/incidents/INC-20260923-120000-abcdef/bundle.zip",
])
def test_nothing_outside_the_store_can_be_fetched(served, url):
    _app, client, ids = served
    folder = db._pipeline.store.bundle_dir(ids[0])
    with open(os.path.join(folder, "stray-notes.txt"), "w", encoding="utf-8") as fh:
        fh.write("not evidence")
    res = client.get(url.replace("{id}", ids[0]))
    assert res.status_code == 404, (url, res.status_code)


def test_the_routes_work_before_the_pipeline_exists(monkeypatch):
    import app
    monkeypatch.setattr(db, "_pipeline", None)
    client = app.app.test_client()
    assert client.get("/api/incidents").get_json() == {"incidents": [], "scope": "session"}
    assert client.get("/incidents/INC-20260923-120000-abcdef/report.pdf").status_code == 404


def test_the_command_line_picks_game_and_probes():
    import app
    args = app.parse_args(["--game", "mario_bugged", "--synthetic-probe", "900",
                           "--synthetic-probe", "2000", "--no-reproduce"])
    assert (args.game, args.synthetic_probe, args.no_reproduce) == ("mario_bugged", [900, 2000], True)
    assert app.parse_args([]).game == config.CLEAN_GAME_VARIANT
    with pytest.raises(SystemExit):
        app.parse_args(["--game", "mario_clone"])


# ══════════════════════════════════════════════════════════════════════════
# Which brain the dashboard plays
# ══════════════════════════════════════════════════════════════════════════
def test_the_approved_brain_is_chosen_only_when_its_hash_matches(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    final = tmp_path / config.FINAL_BRAIN_PATH
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_bytes(b"the approved weights")
    (tmp_path / "glitch_hunter_qa.zip").write_bytes(b"a working copy")
    good = hashlib.sha256(b"the approved weights").hexdigest()
    monkeypatch.setattr(db, "objective2_record", lambda: {"brain": {"sha256": good}})
    assert db.select_checkpoint() == (config.FINAL_BRAIN_PATH, "qa_exploration")
    monkeypatch.setattr(db, "objective2_record", lambda: {"brain": {"sha256": "0" * 64}})
    assert db.select_checkpoint() == ("glitch_hunter_qa.zip", "qa_exploration")
    assert any("does not match" in r.getMessage() for r in caplog.records)


def test_the_frozen_brain_brings_its_frozen_coverage(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(db.coverage_mod, "load_testable", lambda: None)
    monkeypatch.setattr(db.os.path, "exists", lambda p: seen.append(p) or False)
    db._dashboard_coverage("qa_exploration", config.FINAL_BRAIN_PATH)
    assert seen == [config.FINAL_COVERAGE_PATH]


def test_a_browser_reconnecting_does_not_rewrite_why_testing_stopped(make_svc):
    s = make_svc({1: [{"status": "new", "summary": _summary()}]})
    s.start_testing()
    assert wait_for(lambda: s.bug_found is not None)
    s.client_connected()
    s.client_disconnected()
    s.wait_idle()
    assert s.status()["pause_reason"] == "bug_found" and s.bug_found is not None
