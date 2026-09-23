"""End-to-end proof of the incident pipeline on the REAL dashboard stack.

    python tools/validate_incident_pipeline.py            (~30 s, headless)

Runs the dashboard's own game thread (dashboard_service), backend
(dashboard_backend) and web routes (app.py) with the approved Objective-2
brain playing the CLEAN game, headless. Two SYNTHETIC probes stand in for
bugs - the game itself is untouched, and every incident they produce is
labelled synthetic - and every stage of Objective 3 is asserted:

    probe fires -> testing stops before another step -> raw evidence on disk
    -> dashboard reports "bug found" -> GIF / Markdown / PDF rendered
    -> replay reproduces it -> files download, nothing else does
    -> resume -> a second, different incident -> resume -> the first one's
       site again in a later episode: counted, NOT a new incident, no stop
    -> the frozen Objective-2 files are byte-identical afterwards

Writes only its own store (default incidents/_validation/<time>/, which the
dashboard's history ignores) and a validation_report.json beside it. This is
PIPELINE validation: nothing here is, or claims to be, a discovered game bug.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.cli import prepare_tool

# Headless unless --windowed: that one flag has to be known before pygame loads.
ROOT = prepare_tool(headless="--windowed" not in sys.argv)

from common.fileio import sha256_of
from exploration import config

# Where the agent reliably is early in every episode: past the first goomba,
# and short of the first pipe (x ~1200) - and then well past it.
DEFAULT_PROBES = (1000, 2000)


class Check:
    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []

    def __call__(self, name: str, ok: bool, detail: Any = "") -> bool:
        self.results.append({"check": name, "ok": bool(ok), "detail": str(detail)[:400]})
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))
        return bool(ok)

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [r for r in self.results if not r["ok"]]


def _wait(cond: Any, timeout: float, poll: float = 0.05) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(poll)
    return bool(cond())


def _protected() -> dict[str, str]:
    """SHA-256 of the frozen Objective-2 files and every artifacts.json entry present."""
    with open(os.path.join(ROOT, "artifacts.json"), encoding="utf-8") as fh:
        names = set(json.load(fh)) | {config.FINAL_BRAIN_PATH, config.FINAL_COVERAGE_PATH,
                                      config.FINAL_OBJECTIVE2_RECORD}
    return {n: sha256_of(os.path.join(ROOT, n)) for n in sorted(names)
            if os.path.isfile(os.path.join(ROOT, n))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=None, help="incident store for this run "
                    "(default: incidents/_validation/<utc time>/)")
    ap.add_argument("--probes", type=int, nargs=2, default=list(DEFAULT_PROBES), metavar="X",
                    help="world x of the two synthetic probes")
    ap.add_argument("--timeout", type=float, default=240.0, help="seconds allowed per stage")
    ap.add_argument("--windowed", action="store_true",
                    help="use a real game window, as the dashboard does (the replay stays "
                         "headless): checks the trigger frame survives that difference")
    args = ap.parse_args(argv)

    import app
    import dashboard_backend as db
    from dashboard_service import GameWindowService
    from reporting.schema import utc_now
    out = args.out or os.path.join(ROOT, config.INCIDENTS_DIR, "_validation",
                                   f"{utc_now():%Y%m%d-%H%M%S}")
    check = Check()
    events: list[tuple[str, Any]] = []
    started = time.monotonic()
    print(f"validation store: {out}")

    protected_before = _protected()
    db.configure(db.DashboardConfig(game_variant=config.CLEAN_GAME_VARIANT,
                                    synthetic_probes=tuple(args.probes), incidents_dir=out))
    app.backend.notify = lambda e, p: events.append((e, p))
    svc = GameWindowService(app.backend, emit=lambda e, p: events.append((e, p)),
                            log=lambda *_a: None)
    app.service = svc                                  # the routes read the module global
    client = app.app.test_client()
    svc.start(timeout=args.timeout)
    pipe = app.backend.pipeline
    assert pipe is not None
    desc = app.backend.describe()
    check("the approved Objective-2 brain is playing", desc["brain_approved"] is True, desc)
    check("the clean game is running and matches its pin", desc["game_is_clean_baseline"] is True,
          desc)

    # ── 1. the first probe: stop, evidence, bug found ─────────────────────
    print("stage 1: first incident")
    svc.start_testing()
    got = _wait(lambda: svc.bug_found is not None, args.timeout)
    if not check("a synthetic probe produced an incident", got):
        svc.shutdown()
        return _finish(check, out, started, events)
    assert svc.bug_found is not None
    first = svc.bug_found[0]
    steps_at_stop = svc.steps
    time.sleep(1.0)
    check("testing stopped and stayed stopped", not svc.testing and svc.steps == steps_at_stop,
          f"steps {steps_at_stop} -> {svc.steps}")
    check("the stop is backend state: pause_reason bug_found",
          svc.status()["pause_reason"] == "bug_found")
    check("the dashboard was told", any(e == "bug_found" for e, _p in events))
    check("it is labelled synthetic", first["synthetic"] is True
          and first["title"].startswith("SYNTHETIC"), first["title"])
    bundle1 = pipe.store.bundle_dir(first["incident_id"])
    raw = [n for n in ("incident.json", "trigger.png", "trajectory.json", "context_frames.zip")
           if os.path.isfile(os.path.join(bundle1, n))]
    check("raw evidence was on disk when testing stopped", len(raw) == 4, raw)
    status = client.get("/api/status").get_json()
    check("/api/status reports the bug", (status.get("bug_found") or [{}])[0].get("incident_id")
          == first["incident_id"])

    # ── 2. the derived artifacts ──────────────────────────────────────────
    print("stage 2: reports and replay")
    check("reports rendered", pipe.wait_idle(args.timeout))
    s1 = pipe.summary(first["incident_id"])
    check("GIF, Markdown and PDF all rendered",
          all(s1["renders"].get(n) == "done" for n in ("context.gif", "report.md", "report.pdf")),
          s1["renders"])
    check("the replay reproduced it exactly", s1["reproduction"] == "reproduced", s1["reproduction"])
    check("the bundle verifies against its manifest", pipe.store.verify(first["incident_id"]) == [])
    check("the bundle is finalized", s1["finalized"])
    record1 = pipe.store.load_record(first["incident_id"])
    check("the record names the approved brain and clean game",
          record1["provenance"]["brain"]["approved_objective2_brain"]
          and record1["provenance"]["game"]["variant"] == config.CLEAN_GAME_VARIANT)
    check("detection and caller agree on episode and step",
          record1["consistency"]["episode_match"] and record1["consistency"]["agent_step_match"],
          record1["consistency"])
    from PIL import Image
    with open(os.path.join(bundle1, "context.gif"), "rb") as fh:
        gif = Image.open(io.BytesIO(fh.read()))
        n_frames = getattr(gif, "n_frames", 1)
    ctx_count = record1["evidence"].get("context_frames", {}).get("count", 0)
    check("the GIF is the context frames then the trigger", n_frames == ctx_count + 1,
          f"{n_frames} frames, {ctx_count} context")

    # ── 3. the web routes ─────────────────────────────────────────────────
    print("stage 3: open and download")
    listed = client.get("/api/incidents").get_json()["incidents"]
    check("/api/incidents lists it", [i["incident_id"] for i in listed] == [first["incident_id"]])
    for name, mime in (("report.pdf", "application/pdf"), ("report.md", "text/plain"),
                       ("trigger.png", "image/png"), ("context.gif", "image/gif")):
        res = client.get(f"/incidents/{first['incident_id']}/{name}")
        check(f"{name} opens", res.status_code == 200 and res.mimetype == mime,
              f"{res.status_code} {res.mimetype}")
    res = client.get(f"/incidents/{first['incident_id']}/report.pdf?download=1")
    check("report.pdf downloads as an attachment",
          "attachment" in res.headers.get("Content-Disposition", ""))
    res = client.get(f"/incidents/{first['incident_id']}/bundle.zip")
    with zipfile.ZipFile(io.BytesIO(res.data)) as zf:
        zipped = len(zf.namelist())
    check("the whole bundle downloads", res.status_code == 200 and zipped >= 9, zipped)
    for url in (f"/incidents/{first['incident_id']}/..%2F..%2Fapp.py",
                f"/incidents/{first['incident_id']}/..%2Foccurrences.jsonl",
                "/incidents/..%2F..%2Fartifacts.json/incident.json"):
        check(f"refused: {url}", client.get(url).status_code == 404)

    # ── 4. resume: a second, different incident ──────────────────────────
    print("stage 4: resume, second incident")
    hash1 = {n: sha256_of(os.path.join(bundle1, n)) for n in os.listdir(bundle1)}
    svc.start_testing()
    check("resuming cleared the banner", _wait(lambda: svc.bug_found is None, 10)
          and any(e == "bug_cleared" for e, _p in events))
    got = _wait(lambda: svc.bug_found is not None, args.timeout)
    second = svc.bug_found[0] if got and svc.bug_found else None
    check("the second probe produced a NEW incident",
          second is not None and second["incident_id"] != first["incident_id"],
          None if second is None else second["incident_id"])
    check("the session carried on rather than restarting", svc.steps > steps_at_stop)

    # ── 5. the first site again, in a later episode: counted, no stop ─────
    print("stage 5: a repeat sighting")
    svc.start_testing()
    got = _wait(lambda: any(e == "incident_occurrence" for e, _p in events), args.timeout * 2)
    check("a later episode's sighting was recorded as an occurrence", got)
    time.sleep(0.5)
    repeats = [p for e, p in events if e == "incident_occurrence"]
    check("a repeat does not stop testing", svc.testing or svc.status()["pause_reason"] != "bug_found"
          or (svc.bug_found or [{}])[0].get("incident_id") not in {r["incident_id"] for r in repeats})
    svc.stop_testing()
    svc.wait_idle()
    pipe.wait_idle(args.timeout)
    counts = {s["incident_id"]: s["occurrences"] for s in pipe.summaries()}
    check("the repeat raised the count, not the number of incidents",
          any(counts.get(r["incident_id"], 0) >= 2 for r in repeats) and len(counts) >= 2,
          counts)
    check("incident 1's files were never touched by what came after",
          {n: sha256_of(os.path.join(bundle1, n)) for n in hash1} == hash1)
    svc.reset()
    svc.wait_idle()
    check("reset cleared the stop", svc.bug_found is None)
    svc.shutdown()
    pipe.close()

    # ── 6. Objective 2 untouched ──────────────────────────────────────────
    after = _protected()
    check("every protected Objective-2 file is byte-identical",
          after == protected_before, sorted(k for k in after if after.get(k) != protected_before.get(k)))
    return _finish(check, out, started, events)


def _finish(check: Check, out: str, started: float, events: list[tuple[str, Any]]) -> int:
    os.makedirs(out, exist_ok=True)
    report = {"tool": "tools/validate_incident_pipeline.py",
              "kind": "SYNTHETIC pipeline validation - not a game-bug discovery",
              "display": "windowed" if "--windowed" in sys.argv else "headless (SDL dummy)",
              "elapsed_s": round(time.monotonic() - started, 1),
              "passed": not check.failed, "checks": check.results,
              "events": sorted({e for e, _p in events})}
    with open(os.path.join(out, "validation_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"\n{len(check.results) - len(check.failed)}/{len(check.results)} checks passed "
          f"in {report['elapsed_s']} s -> {os.path.join(out, 'validation_report.json')}")
    return 0 if not check.failed else 1


if __name__ == "__main__":
    sys.exit(main())
