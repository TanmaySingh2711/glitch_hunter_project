"""Re-evaluate the bootstrap coverage against the corrected testable mask.

    python tools/recheck_bootstrap.py

The bootstrap was recorded before the denominator was corrected, so its
headline "1,882,128 px = 24.71%" was measured against a mask (7,606,986)
that was larger than the world itself. The RAW EVIDENCE is still perfectly
good - it is a record of where Mario's collider actually went, which does
not change when the denominator is fixed. Only the interpretation changes.

So this does not re-run the 40 episodes. It re-reads the recorded bitmap,
re-scores it against the adopted mask, splits out the pixels that are now
correctly classified as anomalous, and rewrites the file in the current
format. The original is kept alongside as `_raw_v1.npz`, because deleting
the evidence to make the numbers tidy would be the wrong trade.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.cli import prepare_tool

ROOT = prepare_tool(headless=False)

import numpy as np

from exploration import config
from exploration.coverage import SpatialCoverage, load_testable


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.parse_args(argv)
    path = config.BOOTSTRAP_COVERAGE
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found - run tools/bootstrap_coverage.py first.")

    testable = load_testable()
    if testable is None:
        raise SystemExit("Run tools/build_reachability.py first.")

    raw = np.load(path, allow_pickle=False)
    n = config.GRID_W * config.GRID_H
    visited = np.unpackbits(raw['visited_packed'])[:n].reshape(
        config.GRID_H, config.GRID_W)
    model_steps = int(raw['model_timesteps'])
    recorded = int(visited.sum())

    cov = SpatialCoverage(testable_mask=testable)
    cov.visited[:] = visited
    if 'visit_counts' in raw and cov.visit_count is not None:
        cov.visit_count[:] = raw['visit_counts']
    cov.oob_events = int(raw['oob_events'])
    cov.episode_new_history = [float(v) for v in raw['episode_new_history']]
    # np.load on an .npz keeps the zip handle open lazily. On Windows that
    # holds a lock on the very file the atomic os.replace() below targets,
    # so it must be closed before rewriting.
    raw.close()

    covered = cov.covered_testable()
    breakdown = cov.noncoverage_breakdown()
    cov.assert_consistent()

    backup = path.replace('.npz', '_raw_v1.npz')
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    cov.save(path, model_timesteps=model_steps)

    print("=" * 70)
    print("BOOTSTRAP RE-SCORED AGAINST THE CORRECTED DENOMINATOR")
    print("=" * 70)
    print(f"  paired with model     : {model_steps:,} steps")
    print(f"  raw recorded px       : {recorded:,}   (unchanged evidence)")
    print()
    print(f"  world_raster_px       : {config.WORLD_RASTER_PX:>10,}"
          f"   informational only")
    print(f"  testable_coverable_px : {config.TESTABLE_TOTAL:>10,}"
          f"   method {config.ADOPTED_METHOD}")
    print(f"  covered_testable_px   : {covered:>10,}")
    print(f"  remaining_testable_px : {cov.remaining():>10,}")
    print(f"  noncoverage_px        : {breakdown['total']:>10,}   NOT coverage")
    print()
    print(f"  COVERAGE              : {cov.coverage_pct():.2f}%"
          f"   (covered_testable / testable_coverable)")
    print()
    print("  noncoverage breakdown - most of this is NORMAL, not glitches:")
    for group, note in (('expected', 'proven-normal engine behaviour'),
                        ('model_gap', 'reachability model, not the game'),
                        ('anomalous', 'genuinely impossible -> glitch system')):
        print(f"    {group:<20}{breakdown[group + '_total']:>8,}   {note}")
        for k, v in breakdown[group].items():
            print(f"      {k:<26}{v:>8,}")
    print()
    print("  FOR COMPARISON, the same evidence under the retired denominator:")
    print("    reported before     : 1,882,128 px = 24.71% of 7,606,986")
    print("    that denominator exceeded the world raster by 2,154,786 px")
    print()
    print(f"  raw evidence preserved at {backup}")
    print(f"  rewritten in format v{config.COVERAGE_FORMAT_VERSION}: {path}")


if __name__ == "__main__":
    main()
