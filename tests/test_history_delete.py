"""Bug History's Delete and Clear history.

Saved evidence leaves the disk in exactly one way: the user deletes it from
the Bug History page. These tests pin what that removes (the whole bundle and
everything remembered about it), and the three things it must leave alone:
the bug or run Live Testing is stopped on, a report still being written, and
anything asked for by another website.
"""
import os

import numpy as np
import pytest
from incident_helpers import context, detection
from test_run_report import jpeg, record

from common import fileio
from dashboard import backend as db
from reporting import run_report as rr
from reporting.pipeline import DeleteRefused, IncidentPipeline
from reporting.store import IncidentStore, StoreError


@pytest.fixture
def pipe(tmp_path, monkeypatch):
    from incident_helpers import provenance
    p = IncidentPipeline(IncidentStore(str(tmp_path / "incidents")), reproduce=False)
    monkeypatch.setattr(db, "_pipeline", p)
    monkeypatch.setattr(db, "_provenance", provenance())
    yield p
    p.close()


def _capture(pipe, kind="speed", x=1000):
    det = detection(kind=kind, x=x)
    return pipe.capture(det, context(det))


@pytest.fixture
def served(pipe):
    import app
    ids = [_capture(pipe, kind, 1000 + 800 * i).incident_id
           for i, kind in enumerate(("speed", "below_world"))]
    assert pipe.wait_idle(120)
    return app, app.app.test_client(), ids


@pytest.fixture
def runs(tmp_path, monkeypatch):
    import app
    store = rr.RunReports(str(tmp_path / "runs"))
    ids = []
    for run_id in ("RUN-20260925-100000-abc123", "RUN-20260925-110000-def456"):
        store.write({**record(), "run_id": run_id}, np.zeros((60, 80, 3), np.uint8), [jpeg(3)],
                    background=False)
        ids.append(run_id)
    monkeypatch.setattr(type(app.backend), "run_reports", property(lambda _s: store))
    return app, app.app.test_client(), store, ids


# ── the store and the pipeline ────────────────────────────────────────────
def test_a_deleted_incident_is_gone_from_disk_and_from_every_list(pipe):
    first = _capture(pipe).incident_id
    assert pipe.wait_idle(120)
    again = _capture(pipe)                      # the same bug, seen a second time
    assert again.status == "duplicate" and pipe.summary(first)["occurrences"] == 2
    events = []
    pipe.notify = lambda e, p: events.append((e, p))
    bundle = pipe.store.bundle_dir(first)
    assert not os.access(os.path.join(bundle, "incident.json"), os.W_OK), "evidence was writable"
    pipe.delete(first)
    assert not os.path.exists(bundle), "read-only evidence survived the delete"
    assert os.listdir(pipe.store.root) == ["occurrences.jsonl"], "something was left behind"
    assert list(pipe.store.occurrences()) == [], "its sightings were kept"
    assert pipe.incident_ids() == [] and pipe.summaries() == [] and pipe.session_summaries() == []
    assert events == [("incident_deleted", {"incident_id": first})]
    with pytest.raises(StoreError):
        pipe.delete(first)


def test_the_same_bug_seen_after_a_delete_is_a_new_incident(pipe):
    first = _capture(pipe).incident_id
    assert pipe.wait_idle(120)
    pipe.delete(first)
    again = _capture(pipe)
    assert again.status == "new" and again.incident_id != first
    assert pipe.summary(again.incident_id)["occurrences"] == 1


def test_deleting_one_incident_leaves_the_others_and_their_sightings(pipe):
    keep = _capture(pipe, "speed", 1000).incident_id
    drop = _capture(pipe, "below_world", 3000).incident_id
    _capture(pipe, "speed", 1000)
    _capture(pipe, "below_world", 3000)
    assert pipe.wait_idle(120)
    pipe.delete(drop)
    assert pipe.incident_ids() == [keep]
    assert [o["incident_id"] for o in pipe.store.occurrences()] == [keep]
    assert pipe.summary(keep)["occurrences"] == 2
    assert pipe.store.verify(keep) == []


def test_an_incident_whose_reports_are_still_being_written_is_not_deleted(tmp_path):
    pipe = IncidentPipeline(IncidentStore(str(tmp_path)), reproduce=False, start_worker=False)
    det = detection()
    incident_id = pipe.capture(det, context(det)).incident_id      # queued, never rendered
    with pytest.raises(DeleteRefused):
        pipe.delete(incident_id)
    assert os.path.isdir(pipe.store.bundle_dir(incident_id))


def test_a_delete_that_could_not_finish_is_swept_at_the_next_start(tmp_path):
    left = tmp_path / f"{fileio.RETIRED_PREFIX}INC-20260923-120000-aaaaaa-1a2b3c"
    left.mkdir()
    (left / "trigger.png").write_bytes(b"x")
    fileio.make_read_only(left / "trigger.png")
    IncidentStore(str(tmp_path)).recover()
    assert not left.exists()


# ── the web routes: incidents ─────────────────────────────────────────────
def test_delete_removes_one_incident(served):
    _app, client, ids = served
    res = client.delete(f"/api/incidents/{ids[0]}")
    assert res.status_code == 200 and res.get_json() == {"deleted": [ids[0]], "kept": []}
    listed = [i["incident_id"] for i in client.get("/api/incidents?all=1").get_json()["incidents"]]
    assert listed == [ids[1]]
    assert client.get(f"/incidents/{ids[0]}/report.pdf").status_code == 404
    assert client.delete(f"/api/incidents/{ids[0]}").status_code == 404


def test_delete_refuses_anything_that_is_not_an_incident(served):
    _app, client, ids = served
    for bad in ("INC-20260923-120000-ffffff", "..%2F..%2Fapp.py", "not-an-id", "_incomplete"):
        assert client.delete(f"/api/incidents/{bad}").status_code == 404
    assert len(client.get("/api/incidents?all=1").get_json()["incidents"]) == len(ids)


def test_the_bug_live_testing_is_stopped_on_is_kept(served, monkeypatch):
    app, client, ids = served
    monkeypatch.setattr(app.service, "bug_found", [{"incident_id": ids[0]}])
    res = client.delete(f"/api/incidents/{ids[0]}")
    assert res.status_code == 409
    assert res.get_json()["kept"] == [{"id": ids[0], "why": app.OPEN_ON_LIVE}]
    cleared = client.delete("/api/incidents").get_json()
    assert cleared["deleted"] == [ids[1]] and [k["id"] for k in cleared["kept"]] == [ids[0]]
    assert client.get(f"/incidents/{ids[0]}/report.pdf").status_code == 200


def test_clear_history_removes_every_incident(served):
    _app, client, ids = served
    body = client.delete("/api/incidents").get_json()
    assert sorted(body["deleted"]) == sorted(ids) and body["kept"] == []
    assert client.get("/api/incidents?all=1").get_json()["incidents"] == []
    assert client.delete("/api/incidents").get_json() == {"deleted": [], "kept": []}


def test_another_website_cannot_delete(served, runs):
    _app, client, ids = served
    evil = {"Origin": "http://evil.example"}
    assert client.delete(f"/api/incidents/{ids[0]}", headers=evil).status_code == 403
    assert client.delete("/api/incidents", headers=evil).status_code == 403
    assert client.delete(f"/api/runs/{runs[3][0]}", headers=evil).status_code == 403
    assert client.delete("/api/runs", headers=evil).status_code == 403
    assert len(client.get("/api/incidents?all=1").get_json()["incidents"]) == len(ids)
    assert len(client.get("/api/runs").get_json()["runs"]) == 2
    own = {"Origin": "http://localhost"}                    # the dashboard's own page
    assert client.delete(f"/api/incidents/{ids[0]}", headers=own).status_code == 200


def test_reading_routes_do_not_delete(served):
    _app, client, ids = served
    assert client.get(f"/api/incidents/{ids[0]}").status_code == 200
    assert client.post(f"/api/incidents/{ids[0]}").status_code == 405
    assert client.post("/api/incidents").status_code == 405
    assert len(client.get("/api/incidents?all=1").get_json()["incidents"]) == len(ids)


# ── the web routes: clean-run reports ─────────────────────────────────────
def test_delete_removes_one_run_report(runs):
    _app, client, store, ids = runs
    res = client.delete(f"/api/runs/{ids[0]}")
    assert res.status_code == 200 and res.get_json() == {"deleted": [ids[0]], "kept": []}
    assert [r["run_id"] for r in client.get("/api/runs").get_json()["runs"]] == [ids[1]]
    assert os.listdir(store.root) == [ids[1]]
    assert client.delete(f"/api/runs/{ids[0]}").status_code == 404
    for bad in ("RUN-20260925-100123-ffffff", "..%2Fruns", "not-a-run"):
        assert client.delete(f"/api/runs/{bad}").status_code == 404


def test_the_run_live_testing_is_showing_is_kept(runs, monkeypatch):
    app, client, _store, ids = runs
    monkeypatch.setattr(app.service, "run_result", {"report": {"run_id": ids[1]}})
    res = client.delete(f"/api/runs/{ids[1]}")
    assert res.status_code == 409 and res.get_json()["kept"][0]["why"] == app.OPEN_ON_LIVE
    cleared = client.delete("/api/runs").get_json()
    assert cleared["deleted"] == [ids[0]] and [k["id"] for k in cleared["kept"]] == [ids[1]]


def test_a_run_report_still_being_written_is_kept(runs):
    _app, client, store, ids = runs
    store._renders[ids[0]] = {"report.pdf": "pending"}
    assert client.delete(f"/api/runs/{ids[0]}").status_code == 409
    with pytest.raises(rr.RunReportBusy):
        store.delete(ids[0])
    assert os.path.isdir(store.folder(ids[0]))


def test_clear_history_removes_every_run_report(runs):
    _app, client, store, ids = runs
    body = client.delete("/api/runs").get_json()
    assert sorted(body["deleted"]) == sorted(ids) and body["kept"] == []
    assert client.get("/api/runs").get_json() == {"runs": []} and os.listdir(store.root) == []


# ── the page ──────────────────────────────────────────────────────────────
def test_the_page_asks_before_it_deletes():
    import app
    page = app.app.test_client().get("/").get_data(as_text=True)
    for needed in ('id="clear-incidents"', 'id="clear-runs"', 'id="confirm-dialog"',
                   'id="confirm-dialog-ok"'):
        assert needed in page
    with open(os.path.join(app.app.static_folder, "js", "main.js"), encoding="utf-8") as fh:
        script = fh.read()
    assert script.count("method: 'DELETE'") == 1, "a delete that does not go through the confirm dialog"
    assert "askToDelete(" in script
