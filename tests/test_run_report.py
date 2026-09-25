"""reporting/run_report.py: the clean game's "no bugs found" run report.

The record is built from measurements only; the Markdown and PDF read the same
rows, so they cannot disagree; the store serves only real runs and
allow-listed names; and the verdict never claims more than the run showed.
"""
import datetime
import json

import cv2
import numpy as np
import pytest

from reporting import run_report as rr

T0 = datetime.datetime(2026, 9, 25, 10, 0, 0, tzinfo=datetime.UTC)
T1 = T0 + datetime.timedelta(seconds=83)
PROVENANCE = {
    "brain": {"present": True, "path": "glitch_hunter_main_brain.zip", "sha256": "ab" * 32,
              "approved_objective2_brain": True},
    "game": {"variant": "mario_clean", "tree_sha256": "cd" * 32,
             "matches_pinned_clean_tree": True},
}
INFO = {"mario_rect": (8850, 452, 32, 32), "score": 4200, "coins": 7, "hud_time": 212,
        "status": "tall", "flag_get": True}


def record(bugs=()):
    return rr.build_run_record(
        run_id="RUN-20260925-100123-abc123", created=T1, started=T0, session_id="S-1",
        env_episode_index=2, agent_steps=1504, end_reason="level_complete", info=INFO,
        furthest_x=8850, bugs=bugs, reward_mode="qa_exploration", provenance=PROVENANCE)


def jpeg(value):
    ok, buf = cv2.imencode(".jpg", np.full((360, 480, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


def test_the_record_holds_the_measured_run():
    r = record()
    assert r["schema"] == rr.RUN_SCHEMA and r["verdict"] == "no_bugs_found"
    assert r["duration_s"] == 83.0 and r["engine_frames"] == 1504 * 4
    assert r["outcome"] == {"end_reason": "level_complete", "label": "Level complete",
                            "level_complete": True}
    assert r["final_state"]["furthest_x"] == 8850 and r["final_state"]["score"] == 4200
    assert "synthetic" not in " ".join(r["detectors"]).lower(), "the probe is not a detector"
    assert len(r["detectors"]) == 12
    json.dumps(r)                                   # the canonical file is plain JSON


def test_bugs_in_the_run_change_the_verdict():
    r = record(bugs=[{"incident_id": "INC-1", "title": "Mario inside a pipe",
                      "category": "clip_into_pipe"}])
    assert r["verdict"] == "bugs_found" and rr.headline(r) == "1 Bug Found"
    assert "INC-1" in dict(rr.overview_rows(r))["Bugs recorded in this run"]


def test_markdown_and_pdf_carry_the_verdict_and_its_limits():
    r, now = record(), T1
    md = rr.render_markdown(r, now)
    assert md.startswith("# Glitch Hunter run report - No Bugs Found")
    assert "none of the 12 detectors fired" in md
    assert "not proof that the game has no bugs" in md
    for key, value in rr.overview_rows(r):
        assert f"| {key} | {value} |" in md
    pdf = rr.render_pdf(r, None, now)
    assert pdf.startswith(b"%PDF")


def test_the_gif_ends_on_the_final_frame_held():
    from PIL import Image
    final = cv2.imencode(".png", np.zeros((600, 800, 3), np.uint8))[1].tobytes()
    gif = rr.render_finish_gif([jpeg(10), jpeg(200)], final)
    img = Image.open(__import__("io").BytesIO(gif))
    assert img.n_frames == 3
    img.seek(2)
    assert img.info["duration"] == 2000
    with pytest.raises(ValueError, match="no frames"):
        rr.render_finish_gif([], None)


def test_the_store_writes_every_file_and_serves_only_allow_listed_names(tmp_path):
    events = []
    store = rr.RunReports(str(tmp_path), notify=lambda e, p: events.append((e, p)))
    r = record()
    summary = store.write(r, np.zeros((600, 800, 3), np.uint8), [jpeg(50)] * 4, background=False)
    assert summary["headline"] == "No Bugs Found" and summary["bugs"] == 0
    assert set(summary["available"]) == set(rr.FILES)
    assert summary["renders"] == dict.fromkeys(rr.DERIVED_FILES, "done")
    assert events and events[-1][0] == "run_report_updated"
    assert store.path(r["run_id"], "report.pdf").endswith("report.pdf")
    for run_id, name in ((r["run_id"], "../run.json"), (r["run_id"], "secret.txt"),
                         ("../../etc", "run.json"), ("RUN-20260925-100123-ffffff", "run.json")):
        with pytest.raises(KeyError):
            store.path(run_id, name)


def test_a_failed_render_is_recorded_not_raised(tmp_path, monkeypatch):
    store = rr.RunReports(str(tmp_path))
    monkeypatch.setattr(rr, "render_pdf", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    summary = store.write(record(), None, [jpeg(1)], background=False)
    assert summary["renders"]["report.pdf"] == "failed"
    assert "report.pdf" not in summary["available"] and "report.md" in summary["available"]


def test_the_app_serves_run_files_and_404s_everything_else(tmp_path, monkeypatch):
    import app as app_module
    store = rr.RunReports(str(tmp_path))
    r = record()
    store.write(r, np.zeros((60, 80, 3), np.uint8), [jpeg(9)], background=False)
    monkeypatch.setattr(type(app_module.backend), "run_reports", property(lambda _s: store))
    client = app_module.app.test_client()
    pdf = client.get(f"/runs/{r['run_id']}/report.pdf")
    assert pdf.status_code == 200 and pdf.data.startswith(b"%PDF")
    md = client.get(f"/runs/{r['run_id']}/report.md")
    assert md.mimetype == "text/plain" and b"No Bugs Found" in md.data
    assert client.get(f"/runs/{r['run_id']}/report.md?download=1").mimetype == "text/markdown"
    assert client.get(f"/runs/{r['run_id']}/secret.txt").status_code == 404
    assert client.get("/runs/not-a-run/report.pdf").status_code == 404
