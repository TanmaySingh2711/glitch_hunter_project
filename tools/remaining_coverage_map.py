"""Where are the testable pixels nobody has covered yet? Offline, from any
saved coverage file.

    python tools/remaining_coverage_map.py glitch_hunter_qa_coverage.npz
    python tools/remaining_coverage_map.py checkpoints_qa/glitch_hunter_qa_6400000_steps_coverage.npz --top 25

Writes, into --out (default coverage_audits/):
    <name>_remaining_map.png       the level, one pixel per 4x4 block: red =
                                   still-unvisited testable pixels, grey =
                                   covered, black = outside the testable mask;
                                   the largest regions boxed and numbered
    <name>_remaining_regions.json  the same regions with exact pixel counts and
                                   world bounding boxes, plus the provenance
                                   split (bootstrap-known vs QA-discovered)

Made for the end of a campaign: small isolated red islands are exactly where
to look for an unreachable ledge or a mistake in the reachability mask. It
only READS the coverage file; nothing is removed, reclassified or rewritten.
"""
import argparse
import datetime
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from exploration import config
from exploration import coverage as coverage_mod
from exploration import level_completion as lc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('coverage', help='a saved coverage .npz')
    ap.add_argument('--out', default=os.path.join(ROOT, 'coverage_audits'))
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--scale', type=int, default=4)
    args = ap.parse_args(argv)

    mask = coverage_mod.load_testable()
    if mask is None:
        sys.exit(f"{config.REACHABLE_MASK_PATH} not found; build it with tools/build_reachability.py")
    cov = coverage_mod.SpatialCoverage(testable_mask=mask)
    cov.load(args.coverage)          # refuses a file recorded against another mask
    regions = lc.remaining_regions(cov, top=args.top)
    stem = os.path.splitext(os.path.basename(args.coverage))[0]
    png = lc.write_remaining_map(cov, os.path.join(args.out, f"{stem}_remaining_map.png"),
                                 regions, scale=args.scale)
    covered = cov.covered_testable()
    report = {
        'kind': 'glitch_hunter_level1_remaining_map',
        'source': os.path.relpath(os.path.abspath(args.coverage), ROOT).replace('\\', '/'),
        'testable_total': cov.testable_total,
        'covered_testable_px': covered,
        'remaining_testable_px': cov.remaining(),
        'coverage_pct_text': lc.pct_text(covered, cov.testable_total),
        'level1_complete': lc.is_level_complete(covered, cov.testable_total),
        'remaining_regions': regions,
        'provenance': lc.provenance(cov),
        'map_png': os.path.relpath(png, ROOT).replace('\\', '/'),
        'map_scale_px_per_block': args.scale,
        'testable_fingerprint': config.TESTABLE_FINGERPRINT,
        'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    js = os.path.join(args.out, f"{stem}_remaining_regions.json")
    lc.write_json_atomic(report, js)
    print(f"{covered:,} / {cov.testable_total:,} covered ({report['coverage_pct_text']}), "
          f"{cov.remaining():,} remaining in {regions['regions']} region(s)")
    for i, r in enumerate(regions['largest'][:10], start=1):
        x0, y0, x1, y1 = r['world_bbox']
        print(f"  #{i:<2} {r['pixels']:>9,} px   x {x0}-{x1}, y {y0}-{y1}")
    print(f"map     {png}\nregions {js}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
