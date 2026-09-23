"""Inspect and maintain the incident store (Objective 3) from the command line.

    python tools/incidents.py list                  every incident, newest first
    python tools/incidents.py show INC-...          the summary + where its files are
    python tools/incidents.py verify [INC-...]      re-hash every bundle file against its manifest
    python tools/incidents.py rerender INC-... --reason "..."
                                                    render the reports again as NEW versions
                                                    (report.v2.md, ...); originals untouched
    python tools/incidents.py reproduce INC-...     replay the episode again (reproduction.vN.json)
    python tools/incidents.py recover               finish what an interrupted run left behind

The dashboard does all of this on its own; the tool is for a developer
looking at the store directly. Only `rerender`, `reproduce` and `recover`
write, and only inside the store: a re-render never replaces a file - it
adds a versioned one and records why in the bundle's manifest history.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.cli import prepare_tool

ROOT = prepare_tool()

from exploration import config
from reporting import render, schema
from reporting.pipeline import IncidentPipeline
from reporting.reproduce import run_reproduction
from reporting.store import IncidentStore, StoreError

DEFAULT_DIR = os.path.join(ROOT, config.INCIDENTS_DIR)


def _pipeline(root: str, writes: bool) -> IncidentPipeline:
    """list/show/verify only read: no recovery, no worker, nothing moved."""
    return IncidentPipeline(IncidentStore(root), start_worker=False, recover=writes)


def cmd_list(p: IncidentPipeline, _args: argparse.Namespace) -> int:
    rows = p.summaries()
    if not rows:
        print(f"no incidents in {p.store.root}")
        return 0
    for s in rows:
        tag = " [SYNTHETIC]" if s["synthetic"] else ""
        print(f"{s['incident_id']}  {s['category']:<16} sev {s['severity']:<7} "
              f"conf {s['confidence']:<14} repro {s['reproduction']:<22} x{s['occurrences']:<3} "
              f"at ({s['location']['x']}, {s['location']['y']}){tag}")
    return 0


def cmd_show(p: IncidentPipeline, args: argparse.Namespace) -> int:
    s = p.summary(args.incident_id)
    bundle = p.store.bundle_dir(args.incident_id)
    for key in ("incident_id", "title", "description", "created_utc", "category", "synthetic",
                "severity", "confidence", "reproduction", "occurrences", "location",
                "game_variant", "brain_approved", "renders", "finalized"):
        print(f"{key:>15}: {s[key]}")
    print(f"{'folder':>15}: {bundle}")
    for name in sorted(os.listdir(bundle)):
        print(f"{'':>17}{name}")
    return 0


def cmd_verify(p: IncidentPipeline, args: argparse.Namespace) -> int:
    ids = [args.incident_id] if args.incident_id else p.store.incident_ids()
    bad = 0
    for incident_id in ids:
        problems = p.store.verify(incident_id)
        errors = schema.validate_incident(p.store.load_record(incident_id))
        for msg in problems + [f"incident.json: {e}" for e in errors]:
            print(f"  {incident_id}: {msg}")
        bad += bool(problems or errors)
        if not (problems or errors):
            print(f"  OK  {incident_id}")
    print(f"{len(ids)} incident(s), {bad} with problems")
    return 1 if bad else 0


def _record_new_version(p: IncidentPipeline, incident_id: str, base: str, data: bytes,
                        reason: str) -> str:
    bundle = p.store.bundle_dir(incident_id)
    name = p.store.next_version_name(bundle, base)
    meta = p.store.write_artifact(bundle, name, data)
    p.store.unlock_manifest(bundle)
    manifest = p.store.read_manifest(bundle)
    manifest["artifacts"][name] = meta
    manifest["history"].append({"utc": schema.iso_utc(schema.utc_now()),
                                "event": f"{base} re-rendered as {name}: {reason}"})
    p.store.write_manifest(bundle, manifest)
    return name


def cmd_rerender(p: IncidentPipeline, args: argparse.Namespace) -> int:
    incident_id = args.incident_id
    bundle = p.store.bundle_dir(incident_id)
    record = p.store.load_record(incident_id)
    manifest, occ, traj, rep = p._render_inputs(bundle, record)
    now = schema.utc_now()
    trig, ctx = p._read(bundle, "trigger.png"), p._read(bundle, "context_frames.zip")
    outputs = {
        "report.md": render.render_markdown(record, manifest, occ, traj, rep, now).encode("utf-8"),
        "report.pdf": render.render_pdf(record, manifest, occ, traj, rep, trig, ctx, now),
    }
    if trig or ctx:
        outputs["context.gif"] = render.render_gif(record, ctx, trig)
    for base in args.what or outputs:
        print(f"  {_record_new_version(p, incident_id, base, outputs[base], args.reason)}")
    return 0


def cmd_reproduce(p: IncidentPipeline, args: argparse.Namespace) -> int:
    bundle = p.store.bundle_dir(args.incident_id)
    result = run_reproduction(bundle)
    name = _record_new_version(p, args.incident_id, "reproduction.json", schema.dumps(result),
                               args.reason)
    print(f"  {result['status']}: {result.get('detail')}  -> {name}")
    return 0


def cmd_recover(root: str) -> int:
    p = IncidentPipeline(IncidentStore(root), start_worker=True)
    print(f"  {p.recovered}")
    p.wait_idle(timeout=600)
    p.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default=DEFAULT_DIR, help="the incident store (default: incidents/)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="every incident, newest first")
    s = sub.add_parser("show", help="one incident's summary and files")
    s.add_argument("incident_id")
    v = sub.add_parser("verify", help="re-hash bundles against their manifests")
    v.add_argument("incident_id", nargs="?")
    r = sub.add_parser("rerender", help="render reports again as new versions")
    r.add_argument("incident_id")
    r.add_argument("--reason", required=True, help="why - recorded in the manifest history")
    r.add_argument("--what", nargs="+", choices=("report.md", "report.pdf", "context.gif"))
    q = sub.add_parser("reproduce", help="replay the episode again, as a new version")
    q.add_argument("incident_id")
    q.add_argument("--reason", default="manual re-run")
    sub.add_parser("recover", help="finish renders an interrupted run left behind")
    args = ap.parse_args(argv)
    if args.cmd == "recover":
        return cmd_recover(args.dir)
    p = _pipeline(args.dir, writes=args.cmd in ("rerender", "reproduce"))
    try:
        return {"list": cmd_list, "show": cmd_show, "verify": cmd_verify,
                "rerender": cmd_rerender, "reproduce": cmd_reproduce}[args.cmd](p, args)
    except StoreError as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
