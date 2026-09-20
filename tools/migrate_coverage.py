"""Re-stamp a campaign coverage file to the CURRENT testable mask, losslessly.

    python tools/migrate_coverage.py SRC DST

Needed whenever the testable mask changes (tools/build_reachability.py), because
every coverage file names the mask it was scored against and
SpatialCoverage.load_verified() refuses a mismatch - correctly: a percentage
computed against a different denominator describes a different quantity.

What a mask change does and does not touch:

  * `visited_packed` records every world pixel the collider ever swept. It is
    independent of the mask and is copied BYTE-FOR-BYTE - no visit is ever
    fabricated or dropped.
  * Exactly four fields depend on the mask: `covered_testable`,
    `testable_total`, `testable_fingerprint` and `config_hash`. They are
    recounted / re-stamped against the live mask.
  * Everything else (model_timesteps, oob_events, episode history, visit
    counts, geometry, format, the original saved_at) is copied unchanged.

Two guarantees, both checked before anything is reported as done:

  1. A covered pixel never becomes uncovered. If any previously covered
     testable pixel falls outside the new mask, the migration REFUSES: a
     numerator that goes down would be coverage being taken away, and for the
     flag-trigger correction it would mean Mario had passed the flagpole
     without the flag sequence - a glitch to investigate, not to paper over.
  2. The result must load through load_verified() at its own model timestep,
     i.e. the campaign would actually resume from it.

The source is never written. DST must not already exist.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.cli import prepare_tool

ROOT = prepare_tool()

import numpy as np

from common.fileio import atomic_write, canonical_sha256
from exploration import config
from exploration import coverage as cov_mod
from exploration.reachability import mask_fingerprint

# Copied unchanged. Everything a campaign file carries except the four fields
# that describe WHICH mask it was scored against.
PRESERVED = ('visited_packed', 'total_covered', 'oob_events', 'episode_new_history',
             'grid_geom', 'world_dims', 'format_version', 'model_timesteps',
             'saved_at', 'visit_counts')


class MigrationRefused(RuntimeError):
    """The migration would lose coverage or produce an unloadable file."""


def migrate(src: str, dst: str, source_mask: str | None = None) -> dict[str, Any]:
    if os.path.abspath(src) == os.path.abspath(dst):
        raise MigrationRefused("refusing to overwrite the source; give a new DST")
    if os.path.exists(dst):
        raise MigrationRefused(f"{dst} already exists; refusing to overwrite it")

    mask = cov_mod.load_testable()
    if mask is None:
        raise MigrationRefused("no live testable mask; run tools/build_reachability.py")
    live_fp = mask_fingerprint(mask)
    if live_fp != config.TESTABLE_FINGERPRINT or int(mask.sum()) != config.TESTABLE_TOTAL:
        raise MigrationRefused(
            "the live mask does not match config.TESTABLE_TOTAL / TESTABLE_FINGERPRINT; "
            "rebuild it before migrating")

    with np.load(src, allow_pickle=False) as npz:
        d = {k: npz[k] for k in npz.files}
    n = config.GRID_W * config.GRID_H
    visited = np.unpackbits(d['visited_packed'])[:n].reshape(
        config.GRID_H, config.GRID_W).astype(bool)
    if int(np.count_nonzero(visited)) != int(d['total_covered']):
        raise MigrationRefused(f"{src}: bitmap does not re-count to its total_covered")

    before = _covered_before(d, visited, source_mask)
    after = int(np.count_nonzero(visited & mask))
    if after < before:
        lost = before - after
        raise MigrationRefused(
            f"{lost:,} previously covered testable px fall outside the new mask. "
            f"Coverage may never go down; investigate before migrating.")

    out = {k: d[k] for k in PRESERVED if k in d}
    out.update({
        'covered_testable': np.int64(after),
        'testable_total': np.int64(config.TESTABLE_TOTAL),
        'testable_fingerprint': np.str_(config.TESTABLE_FINGERPRINT),
        'config_hash': np.str_(cov_mod._config_hash()),
        # Provenance: where this came from and what it said there.
        'migrated_from_path': np.str_(os.path.relpath(src, ROOT)),
        'migrated_from_sha256': np.str_(canonical_sha256(src)),
        'migrated_from_fingerprint': np.str_(str(d['testable_fingerprint'])),
        'migrated_from_testable_total': np.int64(int(d['testable_total'])),
        'migrated_from_covered_testable': np.int64(before),
        'migrated_at': np.str_(datetime.datetime.now().isoformat()),
    })
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with atomic_write(dst) as fh:
        np.savez_compressed(fh, **out)

    # The file is only a migration if the campaign would actually load it.
    probe = cov_mod.SpatialCoverage(testable_mask=mask)
    probe.load_verified(dst, expected_timesteps=int(d['model_timesteps']))
    if int(np.count_nonzero(probe.visited)) != int(d['total_covered']):
        raise MigrationRefused("round trip changed the visited bitmap")
    return {'src': src, 'dst': dst, 'model_timesteps': int(d['model_timesteps']),
            'covered_before': before, 'covered_after': after,
            'total_before': int(d['testable_total']), 'total_after': config.TESTABLE_TOTAL}


def _covered_before(d: dict[str, Any], visited: Any, source_mask: str | None) -> int:
    """The numerator the file had under the mask it was recorded against.

    Older files (the 6M bootstrap) predate the stored `covered_testable`, so
    the check that coverage never goes down needs the ORIGINAL mask to recount
    it. Supplied explicitly, and only accepted if its fingerprint is exactly
    the one the file names - recounting against any other mask would make
    the check meaningless.
    """
    if 'covered_testable' in d:
        return int(d['covered_testable'])
    if source_mask is None:
        raise MigrationRefused(
            "this file has no stored covered_testable; pass --source-mask with "
            "the mask it was recorded against so the numerator can be checked")
    n = config.GRID_W * config.GRID_H
    with np.load(source_mask, allow_pickle=False) as m:
        old = np.unpackbits(m['testable_packed'])[:n].reshape(
            config.GRID_H, config.GRID_W).astype(bool)
    if mask_fingerprint(old) != str(d['testable_fingerprint']):
        raise MigrationRefused(
            f"--source-mask {source_mask} is not the mask this file was recorded "
            f"against (fingerprint {mask_fingerprint(old)[:16]}... vs "
            f"{str(d['testable_fingerprint'])[:16]}...)")
    return int(np.count_nonzero(visited & old))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--source-mask', help='the mask SRC was recorded against; needed '
                    'only for files without a stored covered_testable')
    args = ap.parse_args(argv)
    try:
        r = migrate(args.src, args.dst, args.source_mask)
    except MigrationRefused as exc:
        print(f"REFUSED: {exc}")
        return 1
    print(f"migrated  {r['src']}\n       -> {r['dst']}")
    print(f"  model timestep    : {r['model_timesteps']:,}")
    print(f"  covered testable  : {r['covered_before']:,} -> {r['covered_after']:,}  "
          f"({'unchanged' if r['covered_before'] == r['covered_after'] else 'CHANGED'})")
    print(f"  testable total    : {r['total_before']:,} -> {r['total_after']:,}")
    print(f"  coverage          : {100 * r['covered_before'] / r['total_before']:.4f}% -> "
          f"{100 * r['covered_after'] / r['total_after']:.4f}%")
    print("  load_verified     : OK (the campaign would resume from it)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
