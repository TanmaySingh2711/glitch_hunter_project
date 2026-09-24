"""The incident pipeline without the game: schema, store, de-duplication,
classification, rendering, failure handling and crash recovery.

The engine-level half (exact trigger frames, observation invariance) is in
test_incident_capture.py; replay in test_incident_reproduce.py.
"""
import datetime
import io
import json
import os
import zipfile

import pytest
from incident_helpers import context, detection, provenance

from exploration import config
from reporting import analysis, render, schema
from reporting.pipeline import IncidentPipeline, SessionRecorder
from reporting.store import IncidentStore, StoreError


@pytest.fixture
def store(tmp_path):
    return IncidentStore(str(tmp_path / "incidents"))


@pytest.fixture
def pipe(store):
    p = IncidentPipeline(store, reproduce=False)
    yield p
    p.close()


def capture(pipe, det, **kw):
    return pipe.capture(det, context(det, **kw))


# ── schema ───────────────────────────────────────────────────────────────────
def test_a_captured_record_is_valid_and_deterministic(pipe):
    out = capture(pipe, detection())
    record = pipe.store.load_record(out.incident_id)
    assert schema.validate_incident(record) == []
    assert schema.dumps(record) == schema.dumps(json.loads(schema.dumps(record)))
    assert schema.INCIDENT_ID_RE.fullmatch(out.incident_id)


@pytest.mark.parametrize(("breakage", "message"), [
    (lambda r: r.pop("timing"), "missing timing"),
    (lambda r: r.update(schema="glitch-hunter.incident/0"), "schema"),
    (lambda r: r.update(incident_id="../../etc"), "malformed"),
    (lambda r: r["classification"].update(severity="catastrophic"), "severity"),
    (lambda r: r["timing"].update(episode_substep="12"), "episode_substep"),
    (lambda r: r["evidence"].update(evil={"file": "../x.png"}), "evidence.evil"),
    (lambda r: r.update(synthetic=True), "synthetic incident cannot carry"),
])
def test_validation_rejects_malformed_records(pipe, breakage, message):
    record = pipe.store.load_record(capture(pipe, detection()).incident_id)
    breakage(record)
    assert any(message in e for e in schema.validate_incident(record))


# ── capture: raw evidence first, immutable, unique ───────────────────────────
def test_capture_writes_the_raw_evidence_immutably(pipe):
    det = detection()
    out = capture(pipe, det)
    assert out.status == "new"
    bundle = pipe.store.bundle_dir(out.incident_id)
    for name in schema.RAW_ARTIFACTS:
        path = os.path.join(bundle, name)
        assert os.path.isfile(path), name
        assert not os.access(path, os.W_OK), f"{name} is still writable"
    record = pipe.store.load_record(out.incident_id)
    ev = record["evidence"]["trigger_frame"]
    assert (ev["width"], ev["height"]) == (800, 600)
    from reporting.evidence import decode_png, pixel_sha256
    with open(os.path.join(bundle, "trigger.png"), "rb") as fh:
        assert pixel_sha256(decode_png(fh.read())) == ev["pixel_sha256"] == pixel_sha256(det.frame)
    with pytest.raises(StoreError, match="never overwritten"):
        pipe.store.write_artifact(bundle, "trigger.png", b"replacement")


def test_the_record_carries_what_the_brief_asks_for(pipe):
    det = detection(kind="speed", x=1234, y=321, substep=207)
    record = pipe.store.load_record(capture(pipe, det).incident_id)
    assert record["location"]["world_collider"] == {"x": 1234, "y": 321, "w": 30, "h": 40}
    assert record["location"]["screen"]["x"] == 1234 - det.state["viewport_x"]
    t = record["timing"]
    assert (t["episode_substep"], t["episode_agent_step"], t["substep_in_agent_step"]) == \
        (207, 207 // config.SUBSTEPS_PER_AGENT_STEP + 1, 207 % config.SUBSTEPS_PER_AGENT_STEP)
    assert t["session_id"] == "S-TEST" and t["episode_index"] == det.episode_index
    assert record["gameplay"]["agent_action"] == {"id": 3, "name": "Run Right"}
    assert record["detector"]["metrics"]["x_vel"] == 60.0
    assert record["provenance"]["brain"]["approved_objective2_brain"] is True
    assert record["reproduction_inputs"]["replayable"] is True
    assert record["consistency"] == {"episode_match": True, "agent_step_match": True, "notes": []}


def test_a_mismatched_caller_context_is_flagged_not_hidden(pipe):
    det = detection(substep=207, episode=3)
    ctx = context(det)
    import dataclasses
    ctx = dataclasses.replace(ctx, env_episode_index=2, episode_agent_step=5)
    record = pipe.store.load_record(pipe.capture(det, ctx).incident_id)
    assert record["consistency"]["episode_match"] is False
    assert record["consistency"]["agent_step_match"] is False
    assert len(record["consistency"]["notes"]) == 2


def test_many_incidents_never_share_or_overwrite_a_bundle(pipe):
    ids = [capture(pipe, detection(kind="speed", x=200 + 500 * i)).incident_id for i in range(12)]
    assert len(set(ids)) == 12
    assert sorted(ids) == pipe.incident_ids()
    for i in ids:
        assert pipe.store.load_record(i)["incident_id"] == i
        assert pipe.store.verify(i) == []


def test_ids_stay_unique_when_the_clock_does_not_move(store):
    frozen = datetime.datetime(2026, 9, 23, 12, 0, 0, tzinfo=datetime.UTC)
    ids = {store.create_bundle(frozen)[0] for _ in range(50)}
    assert len(ids) == 50


# ── duplicates ───────────────────────────────────────────────────────────────
def test_the_same_anomaly_at_the_same_site_is_one_incident_with_occurrences(pipe):
    first = capture(pipe, detection(kind="below_world", x=3000, y=610, episode=1))
    again = capture(pipe, detection(kind="below_world", x=3000 + config.DEDUP_RADIUS_X - 1,
                                    y=612, episode=2))
    third = capture(pipe, detection(kind="below_world", x=2990, y=605, episode=3))
    assert (first.status, again.status, third.status) == ("new", "duplicate", "duplicate")
    assert again.incident_id == third.incident_id == first.incident_id
    assert pipe.summary(first.incident_id)["occurrences"] == 3
    assert len(pipe.occurrences(first.incident_id)) == 2
    assert len(pipe.incident_ids()) == 1


@pytest.mark.parametrize("second", [
    {"kind": "speed", "x": 3000, "y": 610},                                # other kind, same site
    {"kind": "below_world", "x": 3000 + config.DEDUP_RADIUS_X + 1, "y": 610},  # next site
    {"kind": "below_world", "x": 3000, "y": 610 + config.DEDUP_RADIUS_Y + 1},
    {"kind": "below_world", "x": 3000, "y": 610, "rect": False},           # no position: no merge
])
def test_different_anomalies_are_never_merged(pipe, second):
    capture(pipe, detection(kind="below_world", x=3000, y=610))
    assert capture(pipe, detection(**second)).status == "new"


def test_synthetic_and_real_events_never_merge(pipe):
    capture(pipe, detection(kind="speed", x=1000))
    out = capture(pipe, detection(kind="synthetic_probe", x=1000, synthetic=True))
    assert out.status == "new"


def test_a_different_game_is_a_different_incident(pipe):
    capture(pipe, detection(kind="speed", x=1000))
    assert capture(pipe, detection(kind="speed", x=1000), tree="0" * 64).status == "new"


def test_duplicates_are_recognised_across_a_restart(store):
    p1 = IncidentPipeline(store, reproduce=False)
    first = capture(p1, detection(kind="speed", x=5000))
    capture(p1, detection(kind="speed", x=5001))
    p1.wait_idle()
    p1.close()
    p2 = IncidentPipeline(store, reproduce=False)
    try:
        again = capture(p2, detection(kind="speed", x=5002))
        assert (again.status, again.incident_id) == ("duplicate", first.incident_id)
        assert p2.summary(first.incident_id)["occurrences"] == 3
    finally:
        p2.close()


# ── severity and confidence ──────────────────────────────────────────────────
def test_severity_is_per_kind_and_never_critical_today():
    assert {analysis.severity(k)[0] for k in analysis.SEVERITY} <= {"high", "medium", "low", "none"}
    assert analysis.severity("below_world")[0] == "high"
    assert analysis.severity("score_drop")[0] == "low"
    assert analysis.severity("something_new")[0] == "unclassified"


def test_confidence_is_earned_by_margin_not_by_firing():
    gap = config.NORMAL_PLAY_MAX_ABS_X_VEL
    barely = analysis.detector_confidence(detection(metrics={"x_vel": 25.5}))
    far = analysis.detector_confidence(detection(metrics={"x_vel": 25 + (25 - gap) * 1.2}))
    assert barely["level"] == "medium" and far["level"] == "high"
    assert 0 < barely["margin"] < 1 <= far["margin"]


def test_confidence_drops_for_untrustworthy_readings():
    dead = analysis.detector_confidence(detection(kind="score_drop", dead=True))
    settling = analysis.detector_confidence(detection(kind="score_drop", substep=1))
    clean = analysis.detector_confidence(detection(kind="score_drop"))
    assert clean["level"] == "high"
    for c in (dead, settling):
        assert c["level"] == "medium" and len(c["downgrades"]) == 1


def test_synthetic_events_carry_no_confidence_and_no_severity(pipe):
    record = pipe.store.load_record(
        capture(pipe, detection(kind="synthetic_probe", synthetic=True)).incident_id)
    assert record["synthetic"] is True
    assert record["classification"]["severity"] == "none"
    assert record["classification"]["detector_confidence"]["level"] == "not_applicable"


# ── rendering ────────────────────────────────────────────────────────────────
def test_every_derived_artifact_is_rendered_and_finalized(pipe):
    out = capture(pipe, detection())
    assert pipe.wait_idle(120)
    bundle = pipe.store.bundle_dir(out.incident_id)
    manifest = pipe.store.read_manifest(bundle)
    assert manifest["finalized"] is True
    assert {k: v["status"] for k, v in manifest["renders"].items()} == {
        "context.gif": "done", "report.md": "done", "report.pdf": "done",
        "reproduction.json": "skipped"}
    assert pipe.store.verify(out.incident_id) == []
    assert not os.access(os.path.join(bundle, "manifest.json"), os.W_OK)
    with open(os.path.join(bundle, "report.pdf"), "rb") as fh:
        pdf = fh.read()
    assert pdf.startswith(b"%PDF-") and pdf.count(b"/Type /Page") >= 2


def test_the_gif_ends_on_the_exact_trigger_frame(pipe):
    from PIL import Image
    det = detection(substep=203)
    out = capture(pipe, det, frames=8)
    pipe.wait_idle(120)
    with open(os.path.join(pipe.store.bundle_dir(out.incident_id), "context.gif"), "rb") as fh:
        gif = Image.open(io.BytesIO(fh.read()))
        n = getattr(gif, "n_frames", 1)
        gif.seek(n - 1)
        last_duration = gif.info.get("duration")
    assert n == 8 + 1
    assert last_duration == config.GIF_TRIGGER_HOLD_MS


def test_the_markdown_separates_facts_from_interpretation(pipe):
    out = capture(pipe, detection(kind="speed"))
    pipe.wait_idle(120)
    with open(os.path.join(pipe.store.bundle_dir(out.incident_id), "report.md"),
              encoding="utf-8") as fh:
        md = fh.read()
    for heading in ("## What was observed (measured)", "## Interpretation (inferred, not measured)",
                    "## Reproduction", "## Limitations", "Root cause: unknown"):
        assert heading in md
    assert "SYNTHETIC" not in md
    assert out.incident_id in md and "x_vel" in md


def test_a_synthetic_report_says_so_prominently(pipe):
    out = capture(pipe, detection(kind="synthetic_probe", synthetic=True))
    pipe.wait_idle(120)
    with open(os.path.join(pipe.store.bundle_dir(out.incident_id), "report.md"),
              encoding="utf-8") as fh:
        md = fh.read()
    assert md.splitlines()[2].startswith("> **SYNTHETIC TEST EVENT - NOT A GAME BUG.**")
    assert "Root cause" not in md


def test_pdf_text_is_made_latin1_safe():
    assert render.pdf_text("a \u2014 b \u2192 c \U0001f6a8") == "a - b -> c ?"


# ── failures never lose the incident ─────────────────────────────────────────
def test_a_failing_pdf_library_loses_the_report_not_the_incident(store, monkeypatch):
    monkeypatch.setattr(config, "RENDER_RETRY_DELAYS_S", (0.0, 0.0))
    pipe = IncidentPipeline(store, reproduce=False)

    def broken(*_a, **_kw):
        raise RuntimeError("PDF engine exploded")
    pipe.render_pdf = broken
    try:
        out = capture(pipe, detection())
        assert out.status == "new"
        assert pipe.wait_idle(60)
        bundle = pipe.store.bundle_dir(out.incident_id)
        manifest = pipe.store.read_manifest(bundle)
        assert manifest["renders"]["report.pdf"]["status"] == "failed"
        assert manifest["renders"]["report.pdf"]["attempts"] == config.RENDER_MAX_ATTEMPTS
        assert "exploded" in manifest["renders"]["report.pdf"]["last_error"]
        assert manifest["renders"]["report.md"]["status"] == "done"
        assert pipe.store.verify(out.incident_id) == []
        assert pipe.incident_ids() == [out.incident_id], "a render failure became an incident"
        assert pipe.summary(out.incident_id)["renders"]["report.pdf"] == "failed"
    finally:
        pipe.close()


def test_a_transient_failure_is_retried(store, monkeypatch):
    monkeypatch.setattr(config, "RENDER_RETRY_DELAYS_S", (0.0, 0.0))
    pipe = IncidentPipeline(store, reproduce=False)
    calls = []
    real = pipe.render_markdown

    def flaky(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError("file locked by an indexer")
        return real(*a, **kw)
    pipe.render_markdown = flaky
    try:
        out = capture(pipe, detection())
        pipe.wait_idle(60)
        manifest = pipe.store.read_manifest(pipe.store.bundle_dir(out.incident_id))
        assert manifest["renders"]["report.md"] == {"status": "done", "attempts": 2}
    finally:
        pipe.close()


def test_a_failed_capture_reports_failure_and_leaves_no_phantom_incident(pipe, monkeypatch):
    def full_disk(*_a, **_kw):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(pipe.store, "write_artifact", full_disk)
    out = capture(pipe, detection(x=7000))
    assert out.status == "failed" and "No space left" in out.error
    monkeypatch.undo()
    # The half-made bundle is not an incident, and the site is not "known":
    assert pipe.incident_ids() == []
    assert capture(pipe, detection(x=7000)).status == "new"


# ── crash recovery ───────────────────────────────────────────────────────────
def test_recovery_keeps_interrupted_captures_and_finishes_interrupted_renders(store):
    p1 = IncidentPipeline(store, reproduce=False, start_worker=False)
    done = capture(p1, detection(x=1000))                       # captured, never rendered
    crashed_id, crashed = store.create_bundle(schema.utc_now())   # capture died mid-way
    store.write_artifact(crashed, "trigger.png", b"\x89PNG partial")
    with open(os.path.join(store.bundle_dir(done.incident_id), "report.md.tmp"), "wb") as fh:
        fh.write(b"half a report")
    p2 = IncidentPipeline(store, reproduce=False)
    try:
        assert crashed_id in p2.recovered["moved_incomplete"]
        assert os.path.isfile(os.path.join(store.root, "_incomplete", crashed_id, "trigger.png"))
        assert done.incident_id in p2.recovered["requeued"]
        assert any(r.endswith("report.md.tmp") for r in p2.recovered["removed_temp"])
        assert p2.wait_idle(120)
        assert p2.summary(done.incident_id)["finalized"] is True
        assert p2.incident_ids() == [done.incident_id]
    finally:
        p2.close()


def test_a_bundle_whose_manifest_was_never_written_is_repaired(store):
    p1 = IncidentPipeline(store, reproduce=False, start_worker=False)
    out = capture(p1, detection())
    os.remove(os.path.join(store.bundle_dir(out.incident_id), "manifest.json"))
    p2 = IncidentPipeline(store, reproduce=False)
    try:
        assert out.incident_id in p2.recovered["manifest_rebuilt"]
        assert p2.wait_idle(120)
        assert p2.store.verify(out.incident_id) == []
    finally:
        p2.close()


def test_a_fully_written_artifact_left_unrecorded_is_adopted_not_rerendered(store):
    p1 = IncidentPipeline(store, reproduce=False, start_worker=False)
    out = capture(p1, detection())
    bundle = store.bundle_dir(out.incident_id)
    store.write_artifact(bundle, "report.md", b"# written before the crash\n")
    p2 = IncidentPipeline(store, reproduce=False)
    try:
        p2.wait_idle(120)
        with open(os.path.join(bundle, "report.md"), encoding="utf-8") as fh:
            assert fh.read() == "# written before the crash\n"
        history = store.read_manifest(bundle)["history"]
        assert any("adopted" in h["event"] for h in history)
    finally:
        p2.close()


# ── the store's paths are closed to the outside ──────────────────────────────
@pytest.mark.parametrize(("incident_id", "name"), [
    ("../incidents", "incident.json"),
    ("INC-20260923-120000-zzzzzz", "incident.json"),
    ("INC-20260923-120000-abcdef", "incident.json"),          # well-formed, does not exist
    (None, "incident.json"),
])
def test_bad_ids_are_refused(pipe, incident_id, name):
    with pytest.raises(StoreError):
        pipe.store.artifact_path(incident_id, name)


@pytest.mark.parametrize("name", ["../manifest.json", "..\\incident.json", "/etc/passwd",
                                  "C:\\Windows\\win.ini", "trigger.png/..", "report.exe",
                                  "incident.json\x00.png", "", ".", "report.v1.pdf"])
def test_bad_file_names_are_refused(pipe, name):
    incident_id = capture(pipe, detection()).incident_id
    with pytest.raises(StoreError):
        pipe.store.artifact_path(incident_id, name)


def test_versioned_rerenders_never_replace_the_original(store):
    pipe = IncidentPipeline(store, reproduce=False, start_worker=False)
    bundle = store.bundle_dir(capture(pipe, detection()).incident_id)
    store.write_artifact(bundle, "report.md", b"v1")
    assert store.next_version_name(bundle, "report.md") == "report.v2.md"
    store.write_artifact(bundle, "report.v2.md", b"v2")
    assert store.next_version_name(bundle, "report.md") == "report.v3.md"
    with open(os.path.join(bundle, "report.md"), "rb") as fh:
        assert fh.read() == b"v1"


def test_a_torn_occurrence_line_is_skipped(store):
    os.makedirs(store.root, exist_ok=True)
    store.append_occurrence({"incident_id": "INC-20260923-120000-abcdef"})
    with open(os.path.join(store.root, "occurrences.jsonl"), "a", encoding="utf-8") as fh:
        fh.write('{"incident_id": "INC-2026')                # the process died mid-line
    assert [o["incident_id"] for o in store.occurrences()] == ["INC-20260923-120000-abcdef"]


# ── the caller-side recorder ─────────────────────────────────────────────────
def test_context_frames_come_from_before_the_trigger_and_one_episode_only():
    rec = SessionRecorder(context_steps=4)
    rec.begin_episode(1)
    for i in range(6):
        rec.step_taken(3)
        rec.push_frame(bytes([i]))
    rec.begin_episode(2)                                  # a reset: nothing carries over
    rec.step_taken(1)
    rec.push_frame(b"e2-1")
    rec.step_taken(2)                                     # the triggering step...
    ctx = rec.context("qa_exploration", provenance())     # ...before its own frame is pushed
    assert [f.jpeg for f in ctx.context_frames] == [b"e2-1"]
    assert ctx.episode_agent_step == 2 and ctx.session_agent_step == 8
    assert ctx.recent_agent_actions == (1, 2) and ctx.agent_action == 2


def test_the_context_zip_is_byte_deterministic():
    from reporting.evidence import context_zip
    det = detection()
    a = context_zip(context(det).context_frames)
    b = context_zip(context(det).context_frames)
    assert a == b
    with zipfile.ZipFile(io.BytesIO(a)) as zf:
        index = json.loads(zf.read("frames.json"))
    assert len(index["frames"]) == 8


# ── audit findings ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("text", [
    "INC-٢٠٢٦٠٩٢٣-120000-abcdef",   # non-ASCII digits
    "INC-20260923-120000-abcdef\n",                                         # trailing newline
])
def test_ids_are_ascii_and_matched_whole(text):
    assert not schema.INCIDENT_ID_RE.fullmatch(text)


def test_file_names_are_matched_whole():
    assert not schema.ARTIFACT_NAME_RE.fullmatch("report.pdf\n")
    assert not schema.ARTIFACT_NAME_RE.fullmatch("report.v٢.pdf")
    assert schema.ARTIFACT_NAME_RE.fullmatch("report.v12.pdf")


def test_a_placeholder_reading_is_low_confidence_whatever_else_holds():
    conf = analysis.detector_confidence(detection(kind="score_drop", rect=False))
    assert conf["level"] == "low"


def test_a_failed_capture_still_leaves_a_record_of_the_detection(pipe, monkeypatch):
    def full_disk(*_a, **_kw):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(pipe.store, "write_artifact", full_disk)
    capture(pipe, detection(kind="below_world", x=4321, y=610))
    monkeypatch.undo()
    path = os.path.join(pipe.store.root, "capture_failures.jsonl")
    with open(path, encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh]
    assert len(lines) == 1
    entry = lines[0]
    assert entry["kind"] == "below_world" and "No space left" in entry["error"]
    assert entry["state"]["mario_rect"] == [4321, 610, 30, 40]


def test_a_bundle_deleted_by_hand_drops_out_of_the_list_without_breaking_it(pipe):
    import shutil
    import stat as st
    keep = capture(pipe, detection(x=1000)).incident_id
    gone = capture(pipe, detection(x=5000)).incident_id
    pipe.wait_idle(120)
    folder = pipe.store.bundle_dir(gone)
    for name in os.listdir(folder):
        os.chmod(os.path.join(folder, name), st.S_IREAD | st.S_IWRITE)
    shutil.rmtree(folder)
    assert [s["incident_id"] for s in pipe.summaries()] == [keep]


def test_detail_reads_an_incident_another_process_wrote(store):
    writer = IncidentPipeline(store, reproduce=False, start_worker=False)
    reader = IncidentPipeline(store, reproduce=False, start_worker=False, recover=False)
    incident_id = capture(writer, detection()).incident_id
    assert reader.detail(incident_id)["record"]["incident_id"] == incident_id


def test_first_in_session_marks_the_sightings_that_stop_the_dashboard(pipe):
    det = detection()
    first = capture(pipe, det)
    again = capture(pipe, det)
    assert first.status == "new" and first.first_in_session
    assert again.status == "duplicate" and not again.first_in_session
    pipe.begin_session()                                   # the dashboard's Reset
    after_reset = capture(pipe, det)
    assert after_reset.status == "duplicate" and after_reset.first_in_session
    assert not capture(pipe, det).first_in_session
    assert [s["incident_id"] for s in pipe.session_summaries()] == [first.incident_id]
