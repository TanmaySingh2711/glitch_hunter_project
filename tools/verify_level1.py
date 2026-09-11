"""Verify a Level-1 completion snapshot before it counts as the final brain.

    python tools/verify_level1.py                  # the committed snapshot
    python tools/verify_level1.py PROOF.json --workers 4

Checks integrity (hashes, exact re-verified coverage), policy health and
completion retention against the frozen 6M baseline - see
evaluation/level1_verification.py. Writes <proof>_verification.json once,
read-only, beside the snapshot; the snapshot itself is never modified.

Exit codes: 0 VERIFIED, 1 NEEDS_REVIEW, 2 REJECTED, 3 nothing to verify.
Inference only: nothing is trained, and no other level is started.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.cli import prepare_tool

ROOT = prepare_tool(headless=False)

from evaluation import completion as ce
from evaluation import level1_verification as lv
from exploration import config
from exploration import level_completion as lc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('proof', nargs='?', help='completion proof .json (default: the committed one)')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--baseline', default=ce.BASELINE_PATH)
    args = ap.parse_args(argv)
    os.chdir(ROOT)

    proof = args.proof
    if proof is None:
        found = lc.committed_snapshots(os.path.join(config.CHECKPOINT_DIR_QA,
                                                    lc.SNAPSHOT_DIRNAME))
        if not found:
            print("No committed Level-1 completion snapshot to verify.")
            return 3
        proof = found[0][0]
    existing = lv.verification_path(proof)
    if os.path.exists(existing):
        rec = ce.load(existing)
        print(f"Already verified: {existing}\nVERDICT {rec['verdict']}  {rec.get('reasons')}")
        return {'VERIFIED': 0, 'NEEDS_REVIEW': 1}.get(rec['verdict'], 2)

    rec = lv.verify(proof, baseline_path=args.baseline, workers=args.workers, root=ROOT)
    print(f"\nVERDICT {rec['verdict']}")
    for reason in rec.get('reasons') or []:
        print(f"  - {reason}")
    if rec['final_level1_brain']:
        print(f"Final Level-1 brain: {rec['final_level1_brain']['path']}")
    print(f"record: {rec['_path']}")
    return {'VERIFIED': 0, 'NEEDS_REVIEW': 1}.get(rec['verdict'], 2)


if __name__ == '__main__':
    sys.exit(main())
