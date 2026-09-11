"""Check that the protected artifacts are still the bytes they were.

    python tools/verify_artifacts.py            # verify every artifact present
    python tools/verify_artifacts.py --record   # (re)write artifacts.json - deliberately

artifacts.json lists the files the project's results depend on - the 6M
brain and its backups, the milestone checkpoints, the testable mask, the
bootstrap map, the frozen completion baseline - with the SHA-256 each must
have. This re-hashes whichever of them exist in this checkout:

    OK        matches the manifest
    CHANGED   exists but differs: a result built on it is no longer what was measured
    absent    not in this checkout (most are git-ignored; that is normal)

Exit code 0 when nothing CHANGED, 1 otherwise. Read-only unless --record is
given, and --record only ever rewrites artifacts.json itself.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.cli import prepare_tool

ROOT = prepare_tool(headless=False)

from common.fileio import canonical_sha256, read_json, write_json_atomic

MANIFEST = os.path.join(ROOT, "artifacts.json")

# What --record captures: every file under these paths that exists.
PROTECTED = (
    "mario_brain_checkpoint.zip",
    "backup_6M/mario_brain_checkpoint.zip",
    "backup_6M/mario_brain_checkpoint_6000000_steps.zip",
    *(f"checkpoints/mario_brain_checkpoint_{400_000 * i}_steps.zip" for i in range(1, 16)),
    "exploration_data/reachable_mask.npz",
    "exploration_data/coverage_bootstrap_6000000.npz",
    "exploration_data/coverage_bootstrap_6000000_raw_v1.npz",
    "evaluation/completion_baseline_6M.json",
)


def verify(manifest: dict[str, dict[str, str]]) -> int:
    changed = 0
    for rel, entry in sorted(manifest.items()):
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print(f"  absent   {rel}")
            continue
        actual = canonical_sha256(path)
        if actual == entry['sha256']:
            print(f"  OK       {rel}")
        else:
            changed += 1
            print(f"  CHANGED  {rel}\n           expected {entry['sha256']}\n"
                  f"           found    {actual}")
    return changed


def record() -> dict[str, dict[str, str]]:
    stamp = datetime.date.today().isoformat()
    return {rel: {"sha256": canonical_sha256(os.path.join(ROOT, rel)), "recorded": stamp}
            for rel in PROTECTED if os.path.exists(os.path.join(ROOT, rel))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--record', action='store_true',
                    help='rewrite artifacts.json from the files present now')
    args = ap.parse_args(argv)
    if args.record:
        recorded = record()
        write_json_atomic(recorded, MANIFEST)
        print(f"recorded {len(recorded)} artifacts in {MANIFEST}")
        return 0
    if not os.path.exists(MANIFEST):
        print(f"{MANIFEST} not found; create it with --record")
        return 1
    manifest: dict[str, dict[str, str]] = read_json(MANIFEST)
    changed = verify(manifest)
    print(f"{len(manifest)} artifacts in the manifest, {changed} changed")
    return 1 if changed else 0


if __name__ == '__main__':
    sys.exit(main())
