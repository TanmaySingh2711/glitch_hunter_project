"""Build exploration_data/reachable_mask.npz — the TESTABLE-pixel denominator.

    python tools/build_reachability.py

Runs all three reachability methods, reconciles them, adopts one, and writes
the mask plus the full provenance. Re-run only if mario_clone's level layout
changes; the solid-geometry counts are asserted so such a change fails loudly
rather than silently moving the coverage denominator.

The adopted total is written back into exploration/config.py so nothing has
to hardcode it, and a fingerprint of the mask goes into the .npz so coverage
states recorded against a different denominator cannot silently load.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.cli import prepare_tool

ROOT = prepare_tool()

from exploration import config, reachability

CONFIG_PATH = os.path.join(ROOT, "exploration", "config.py")


# ══════════════════════════════════════════════════════════════════════════
# THE SCRIPTED FLAG CORRIDOR - recorded from the engine, not derived
#
# Past the flag trigger Mario is under script control (slide, walk to the
# door), so what he can occupy there is not a geometry question. It is
# recorded here by replaying the grab in the real engine through the real
# coverage recorder, and passed to reachability.build_testable, which keeps
# the grab band unchanged and only this corridor beyond it.
#
# The approach set is deliberately dense. A two-replay version (one small
# and one big Mario) was measured and REJECTED: it missed 3,871 of the
# 22,524 corridor px, because the approach - not only Mario's size - changes
# where the script walks him. These 672 approaches (small and big, ground
# hops and leaps from the three upper staircase steps, 7 run-ups x 6 jump
# lengths) reproduce every past-trigger pixel real play has ever covered.
# ══════════════════════════════════════════════════════════════════════════
FLAG_LAUNCHES = ((8300, 538), (8360, 538), (8420, 538),
                 (8070, 194), (8100, 194), (8110, 194),
                 (8025, 237), (7980, 280))
FLAG_RUNUPS = (0, 3, 6, 10, 14, 20, 30)
FLAG_JUMPS = (0, 4, 8, 14, 22, 30)


def flag_approaches() -> list[tuple[bool, int, int, int, int]]:
    """(big, start_x, surface_y, run-up frames, jump frames) for every replay."""
    return [(big, x, surf, pre, jf)
            for big in (False, True) for x, surf in FLAG_LAUNCHES
            for pre in FLAG_RUNUPS for jf in FLAG_JUMPS]


def replay_flag_approach(args: tuple[bool, int, int, int, int]) -> Any:
    """One approach to the pole; returns the packed visited GRID bitmap.

    Mario is placed ON a surface (rect.bottom), never inside a solid - a
    teleport into the staircase is resolved by the engine in ways that do not
    correspond to any real trajectory.
    """
    import numpy as np

    from custom_mario_env import CustomMarioEnv
    from exploration.coverage import SpatialCoverage
    big, start_x, surface, pre, jf = args
    run, run_jump = 3, 4
    env = CustomMarioEnv()
    try:
        env.episode_time_units = config.QA_EPISODE_TIME_UNITS
        env.end_on_level_complete = config.QA_END_ON_LEVEL_COMPLETE
        env.reset()
        mario = env.game.state.mario
        mario.rect.x = start_x
        if big:
            mario.become_big()
        mario.rect.bottom = surface
        mario.y_vel = 0
        for _ in range(30):
            env.step(0)
        cov = SpatialCoverage(testable_mask=None)
        for action in [run] * pre + [run_jump] * jf + [run] * 1200:
            _o, _r, term, trunc, info = env.step(action)
            rect = info.get("mario_rect")
            if rect:
                cov.record({"cur_rect": tuple(rect),
                            "viewport_x": info.get("viewport_x", 0),
                            "x_vel": info.get("x_vel", 0.0),
                            "on_ground": info.get("on_ground", True)})
            if term or trunc:
                break
        return np.packbits(cov.visited.astype(bool))
    finally:
        env.close()


def record_flag_corridor(workers: int = 8) -> Any:
    """Union of every approach, as a WORLD-raster mask."""
    import multiprocessing as mp

    import numpy as np
    n = config.GRID_W * config.GRID_H
    union = np.zeros((config.GRID_H, config.GRID_W), dtype=bool)
    approaches = flag_approaches()
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, packed in enumerate(pool.imap_unordered(replay_flag_approach, approaches)):
            union |= np.unpackbits(packed)[:n].reshape(config.GRID_H, config.GRID_W).astype(bool)
            if (i + 1) % 96 == 0:
                print(f"    flag corridor: {i + 1}/{len(approaches)} approaches replayed")
    y0, x0 = -config.GRID_Y0, -config.GRID_X0
    return union[y0:y0 + config.LEVEL_H, x0:x0 + config.LEVEL_W]


def write_config(stats: dict[str, Any], fingerprint: str) -> None:
    """Records the adopted denominator and its provenance in config.py."""
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        src = fh.read()
    block = "\n".join([
        "# ─── ADOPTED BY tools/build_reachability.py ───",
        (f"#   world raster        = {stats['world_raster_px']:,}"
        f"   (9087 x 600, informational only - NEVER a denominator)"),
        (f"#   solid px            = {stats['solid_px']:,}"
        f"   ({stats['n_solid_rects']} collider rects)"),
        f"#   valid anchors       = {stats['valid_anchors']:,}",
        f"#   standable anchors   = {stats['standable_anchors']:,}",
        "#",
        (f"#   Method A (geometric)  = {stats['method_a_px']:,}"
        f"   informational; no gravity, no jump limit"),
        f"#   Method B (jump env.)  = {stats['method_b_px']:,}",
        f"#   Method C (BFS)        = {stats['method_c_px']:,}",
        f"#   B vs C delta          = {stats['bc_delta_pct']:.2f}%",
        (f"#   flag trigger          = -{stats.get('flag_removed_px') or 0:,}"
        f"   past x {(stats.get('flag_trigger') or [0])[0]}, off the scripted path"),
        *([(f"#   real-arc envelope     = -{stats['arcs_removed_px']:,}"
            f"   no measured jump arc reaches it (real play keeps {stats['observed_px']:,})")]
          if stats.get('arcs_removed_px') is not None else []),
        (f"#   ADOPTED               = {stats['testable_total']:,}   (Method C + flag trigger"
         f"{' + real-arc envelope' if stats.get('arcs_removed_px') is not None else ''})"),
        "#",
        "# Coverage percentage is ALWAYS covered_testable / TESTABLE_TOTAL.",
        "# Pixels outside this mask are noncoverage_px, never coverage. Most",
        "# of that is NORMAL (jump arcs, pit deaths, collision tolerance);",
        "# only the genuinely impossible subset is anomalous_px.",
        f"TESTABLE_TOTAL = {stats['testable_total']}",
        f'TESTABLE_FINGERPRINT = "{fingerprint}"',
        f'ADOPTED_METHOD = "{stats["adopted_method"]}"',
    ])
    # Strip the previous block AND the previous standalone assignments
    # FIRST. Doing it afterwards deleted the very lines the new block had
    # just written, because those assignments live inside the block.
    src = re.sub(r"^# ─── ADOPTED BY tools/build_reachability\.py "
                 r"───\n(?:#.*\n)*", "", src, flags=re.MULTILINE)
    src = re.sub(r"^TESTABLE_FINGERPRINT = .*\n", "", src, flags=re.MULTILINE)
    src = re.sub(r"^ADOPTED_METHOD = .*\n", "", src, flags=re.MULTILINE)
    src = re.sub(r"^TESTABLE_TOTAL = .*$", block, src, count=1, flags=re.MULTILINE)
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        fh.write(src)


def reclassify_only(level_state: Any, dry_run: bool = False) -> None:
    """Recompute ONLY the noncoverage taxonomy inside an existing bundle.

    The full build re-derives the denominator from scratch. When the fix is
    to the TAXONOMY rather than to the geometry - as it was for all three of
    the measured false-positive causes - re-running it would put a
    4,013,723-pixel denominator and its fingerprint back on the line for no
    reason. This path cannot move them: every array in the file except
    `class_map` is copied straight back out of it, and every pixel whose
    class changed must match one of `allowed_moves` below or nothing is
    written at all.
    """
    import numpy as np

    path = config.REACHABLE_MASK_PATH
    with np.load(path, allow_pickle=False) as d:
        stored = {k: d[k] for k in d.files}

    gh, gw = config.GRID_H, config.GRID_W
    y0, x0 = -config.GRID_Y0, -config.GRID_X0
    grid = (gh, gw)

    def unpack(key: str) -> Any:
        full = np.unpackbits(stored[key])[:gh * gw].reshape(grid).astype(bool)
        return full[y0:y0 + config.LEVEL_H, x0:x0 + config.LEVEL_W]

    solid = unpack('solid_packed')
    testable = unpack('testable_packed')
    old = stored['class_map']

    print(f"  reading             : {path}")
    print(f"  testable_total      : {int(stored['testable_total']):,}")
    print(f"  fingerprint         : {str(stored['testable_fingerprint'])[:16]}...")
    print("  recomputing Method B for the connectivity-gap class...")
    b_px, _b = reachability.method_b(solid)
    print("  building mutable-solid mask...")
    mutable = reachability.mutable_solid_mask(level_state)
    print("  building sweep-coverable mask (exhaustive; a few minutes)...")
    sweep = reachability.sweep_coverable(solid)

    region = None
    if 'flag_trigger' in stored and 'flag_corridor_packed' in stored:
        tx, ty, tw, th = (int(v) for v in stored['flag_trigger'])
        trigger = (tx, ty, tw, th)
        corridor = unpack('flag_corridor_packed')
        region = reachability.beyond_flag_region(trigger, corridor)
        print(f"  flag trigger        : {trigger} (stored corridor re-applied)")

    new = reachability.classify_noncoverage(
        solid, testable, b_px, mutable_world=mutable, sweep_world=sweep,
        beyond_flag_world=region)

    # Exactly which reclassifications this path may make. Anything else means
    # the taxonomy moved somewhere nobody intended, and writing it would put a
    # silently different denominator-adjacent artifact on disk.
    allowed_moves = (
        (reachability.CLS_DEEP_PENETRATION, reachability.CLS_MUTABLE_SOLID),
        (reachability.CLS_DEEP_PENETRATION, reachability.CLS_SWEEP_ARTIFACT),
        (reachability.CLS_FLOOR_CLIP, reachability.CLS_PIT_FALL),
    )
    changed = new != old
    n_changed = int(changed.sum())
    legal = np.zeros_like(changed)
    print()
    print(f"  pixels reclassified : {n_changed:,}")
    for src, dst in allowed_moves:
        move = changed & (old == src) & (new == dst)
        legal |= move
        if move.any():
            print(f"    {reachability.CLASS_NAMES[src]:>18} -> "
                  f"{reachability.CLASS_NAMES[dst]:<16}{int(move.sum()):>10,}")
    illegal = int((changed & ~legal).sum())
    for cls in (reachability.CLS_DEEP_PENETRATION, reachability.CLS_FLOOR_CLIP):
        print(f"  {reachability.CLASS_NAMES[cls]:<18}: "
              f"{int((old == cls).sum()):>9,} -> {int((new == cls).sum()):>9,}")
    if illegal:
        raise SystemExit(
            f"REFUSING TO WRITE: {illegal:,} pixels changed class outside the "
            f"moves this path is allowed to make. The taxonomy change is not "
            f"confined.")
    print("  confinement check   : OK (only the allowed moves occurred)")

    if dry_run:
        print("\n--dry-run: nothing written.")
        return

    from common.fileio import atomic_write
    stored['class_map'] = new
    with atomic_write(path) as fh:
        np.savez_compressed(fh, **stored)

    with np.load(path, allow_pickle=False) as d:
        assert int(d['testable_total']) == config.TESTABLE_TOTAL, "denominator moved"
        assert str(d['testable_fingerprint']) == config.TESTABLE_FINGERPRINT, \
            "fingerprint moved"
        assert np.array_equal(d['testable_packed'], stored['testable_packed'])
        assert np.array_equal(d['solid_packed'], stored['solid_packed'])
    reachability._CLASS_MAP_CACHE = None
    print(f"\n  written             : {path}")
    print(f"  denominator         : {config.TESTABLE_TOTAL:,}  UNCHANGED")
    print("  fingerprint         : UNCHANGED")


def load_visited_world(coverage_path: str, expect_fingerprint: str) -> Any:
    """The visited bitmap of a saved coverage state, as a WORLD-raster mask."""
    import numpy as np

    with np.load(coverage_path, allow_pickle=False) as d:
        if str(d['testable_fingerprint']) != expect_fingerprint:
            # The visited bitmap does not depend on any mask (it is every pixel
            # the collider ever swept), so an older state is still valid input
            # here. It just cannot be RESUMED from without tools/migrate_coverage.py.
            print(f"  note: {coverage_path} is stamped with another mask "
                  f"({str(d['testable_fingerprint'])[:12]}...); its bitmap is used as is")
        n = config.GRID_W * config.GRID_H
        grid = np.unpackbits(d['visited_packed'])[:n].reshape(
            config.GRID_H, config.GRID_W).astype(bool)
    y0, x0 = -config.GRID_Y0, -config.GRID_X0
    return grid[y0:y0 + config.LEVEL_H, x0:x0 + config.LEVEL_W]


def tighten(level_state: Any, spawn: tuple[int, int], coverage_files: list[str],
            dry_run: bool = False) -> None:
    """Apply the real-arc envelope to the EXISTING mask bundle.

    Equivalent to a full rebuild with the arcs (the flag trigger and the arc
    filter both only remove pixels, so their order cannot matter) without
    re-recording the flag corridor or re-running Method C at a 1 px lattice.
    Every array except the testable mask, its total/fingerprint and the
    taxonomy is copied straight back; the taxonomy may change ONLY on the
    pixels that left the mask, and only to CONNECTIVITY_GAP.

    `coverage_files` are the saved coverage states the new mask must stay
    consistent with. Whatever they cover that no arc reaches is real play the
    model denies: it is kept testable (and recorded in observed_reach.npz), so
    tools/migrate_coverage.py can carry them across losslessly - coverage may
    never go down.
    """
    import numpy as np

    from common.fileio import atomic_write

    path = config.REACHABLE_MASK_PATH
    with np.load(path, allow_pickle=False) as d:
        stored = {k: d[k] for k in d.files}
    if 'arcs_removed_px' in stored:
        raise SystemExit(f"{path} already carries the arc envelope. Restore the pre-arc "
                         f"bundle (its archive) or rebuild before tightening again.")
    if not coverage_files:
        raise SystemExit("--tighten needs at least one --coverage file: the coverage "
                         "states the new mask must stay consistent with")

    gh, gw = config.GRID_H, config.GRID_W
    y0, x0 = -config.GRID_Y0, -config.GRID_X0

    def unpack(key: str) -> Any:
        full = np.unpackbits(stored[key])[:gh * gw].reshape(gh, gw).astype(bool)
        return full[y0:y0 + config.LEVEL_H, x0:x0 + config.LEVEL_W]

    def embed(world: Any) -> Any:
        return reachability._embed_in_grid(world)

    solid, testable = unpack('solid_packed'), unpack('testable_packed')
    old_class = stored['class_map']
    fp_before = str(stored['testable_fingerprint'])
    total_before = int(stored['testable_total'])
    print(f"  reading             : {path}")
    print(f"  testable_total      : {total_before:,}   ({fp_before[:16]}...)")

    arcs = reachability.load_jump_arcs()
    print(f"  jump arcs           : {arcs.shape[0]} arcs x {arcs.shape[1]} frames")
    print("  real-arc reachability, both forms (a few minutes)...")
    _t, arc_px, _s = reachability.apply_arc_envelope(testable, solid, spawn, arcs, None)

    observed = reachability.load_observed_reach()
    observed = np.zeros_like(testable) if observed is None else observed
    visited_by = {}
    for cov in coverage_files:
        visited = load_visited_world(cov, fp_before)
        visited_by[cov] = visited
        observed |= visited & testable & ~arc_px
    corridor = unpack('flag_corridor_packed') if 'flag_corridor_packed' in stored else None
    keep = observed | (reachability.fill_corridor(corridor) if corridor is not None
                       else np.zeros_like(testable))
    new_testable = testable & (arc_px | keep)
    removed = testable & ~new_testable
    for cov, visited in visited_by.items():
        lost = int((visited & testable & ~new_testable).sum())
        if lost:
            raise SystemExit(f"{cov}: {lost:,} covered pixels would leave the mask")
    total_after = int(new_testable.sum())

    print("  recomputing Method B (both forms) for the connectivity-gap class...")
    b_px = np.logical_or.reduce([reachability.method_b(solid, mw, mh)[0] for mw, mh in
                                 ((config.MARIO_SMALL_W, config.MARIO_SMALL_H),
                                  (config.MARIO_BIG_W, config.MARIO_BIG_H))])
    print("  building mutable-solid mask...")
    mutable = reachability.mutable_solid_mask(level_state)
    print("  building sweep-coverable mask (exhaustive; a few minutes)...")
    sweep = reachability.sweep_coverable(solid)
    region = None
    if 'flag_trigger' in stored and corridor is not None:
        tx, ty, tw, th = (int(v) for v in stored['flag_trigger'])
        region = reachability.beyond_flag_region((tx, ty, tw, th), corridor)
    new_class = reachability.classify_noncoverage(
        solid, new_testable, b_px, mutable_world=mutable, sweep_world=sweep,
        beyond_flag_world=region)

    changed = new_class != old_class
    expected = embed(removed)
    stray = int((changed & ~expected).sum())
    wrong = int((expected & (new_class != reachability.CLS_CONNECTIVITY_GAP)).sum())
    print()
    print(f"  pixels leaving the mask       : {int(removed.sum()):>10,}")
    print(f"  kept as observed real play    : {int((observed & testable & ~arc_px).sum()):>10,}"
          f"   (real play the arcs deny)")
    print(f"  taxonomy pixels changed       : {int(changed.sum()):>10,}"
          f"   (expected {int(expected.sum()):,})")
    if stray or wrong:
        raise SystemExit(
            f"REFUSING TO WRITE: {stray:,} pixels changed class outside the removed set "
            f"and {wrong:,} removed pixels are not CONNECTIVITY_GAP. The taxonomy stored "
            f"in the bundle no longer matches what the classifier produces.")
    print("  confinement check             : OK (taxonomy moved only on removed pixels)")

    fingerprint = reachability.mask_fingerprint(embed(new_testable))
    print()
    print(f"  denominator                   : {total_before:,} -> {total_after:,}"
          f"   ({total_after - total_before:+,})")
    print(f"  fingerprint                   : {fingerprint[:16]}...")
    for cov, visited in visited_by.items():
        covered = int((visited & new_testable).sum())
        was = int((visited & testable).sum())
        print(f"  {cov}")
        print(f"      covered {was:,} / {total_before:,} = {100 * was / total_before:.4f}%"
              f"   ->   {covered:,} / {total_after:,} = {100 * covered / total_after:.4f}%")
    if dry_run:
        print("\n--dry-run: nothing written.")
        return

    import shutil
    archive_dir = os.path.join(config.EXPLORATION_DATA_DIR, "archive_mask_v3_rectangle")
    archive = os.path.join(archive_dir, os.path.basename(path))
    if os.path.exists(archive):
        raise SystemExit(f"{archive} already exists; refusing to overwrite the archive")
    os.makedirs(archive_dir)
    shutil.copy2(path, archive)
    reachability.save_observed_reach(
        config.OBSERVED_REACH_PATH, observed & testable & ~arc_px,
        [os.path.basename(c) for c in coverage_files])

    stored['testable_packed'] = np.packbits(embed(new_testable))
    stored['testable_total'] = np.int64(total_after)
    stored['testable_fingerprint'] = np.str_(fingerprint)
    stored['class_map'] = new_class
    stored['arcs_removed_px'] = np.int64(removed.sum())
    stored['arc_reach_px'] = np.int64(arc_px.sum())
    stored['observed_px'] = np.int64((observed & testable & ~arc_px).sum())
    with atomic_write(path) as fh:
        np.savez_compressed(fh, **stored)
    _solid_back, t, meta = reachability.load_masks()
    assert int(t.sum()) == total_after
    assert meta['testable_fingerprint'] == fingerprint
    flat = {k: (stored[k].item() if hasattr(stored[k], 'item') else stored[k])
            for k in stored if stored[k].ndim == 0}
    flat['flag_trigger'] = ([int(v) for v in stored['flag_trigger']]
                            if 'flag_trigger' in stored else None)
    write_config(flat, fingerprint)
    reachability._CLASS_MAP_CACHE = None
    print(f"\n  archived old mask   : {archive}")
    print(f"  written             : {path}")
    print("  TESTABLE_TOTAL / TESTABLE_FINGERPRINT written to exploration/config.py")
    print("\n  NEXT: migrate every coverage file you will resume from with "
          "tools/migrate_coverage.py")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="build and reconcile the mask, print the result, write nothing")
    ap.add_argument("--workers", type=int, default=8,
                    help="processes for the flag-corridor engine replays")
    ap.add_argument("--class-map-only", action="store_true",
                    help="recompute ONLY the noncoverage taxonomy in the "
                         "existing bundle; the solid mask, the testable mask, "
                         "the denominator and its fingerprint are copied back "
                         "unchanged and verified")
    ap.add_argument("--tighten", action="store_true",
                    help="apply the real-arc envelope (tools/collect_jump_arcs.py) to the "
                         "existing mask: the denominator shrinks to what a measured jump "
                         "can reach, plus whatever real play has demonstrated")
    ap.add_argument("--coverage", nargs="+", default=[], metavar="NPZ",
                    help="with --tighten: the saved coverage states the new mask must "
                         "stay consistent with (each must carry the CURRENT mask's "
                         "fingerprint)")
    args = ap.parse_args(argv)
    from custom_mario_env import CustomMarioEnv

    if args.tighten:
        print("Tightening the testable mask with the real-arc envelope...\n")
        env = CustomMarioEnv()
        env.reset()
        try:
            mario = env.game.state.mario
            tighten(env.game.state, (mario.rect.x, mario.rect.y), args.coverage, args.dry_run)
        finally:
            env.close()
        return

    if args.class_map_only:
        print("Recomputing the noncoverage taxonomy only...\n")
        env = CustomMarioEnv()
        env.reset()
        try:
            reclassify_only(env.game.state, args.dry_run)
        finally:
            env.close()
        return

    print("Building the testable-pixel mask from live level geometry...\n")
    env = CustomMarioEnv()
    env.reset()
    try:
        mario = env.game.state.mario
        spawn = (mario.rect.x, mario.rect.y)
        print(f"  spawn anchor        : {spawn}")
        print(f"  collider            : {mario.rect.size}")
        print(f"  jump envelope       : rise {config.JUMP_RISE_PX} px, "
              f"reach {config.JUMP_REACH_PX} px")
        print(f"  lattice             : {config.REACHABILITY_LATTICE} px"
              f"   (1 = exact, no discretisation)\n")

        print(f"  recording the scripted flag corridor "
              f"({len(flag_approaches())} engine replays)...")
        corridor = record_flag_corridor(args.workers)
        solid, testable, class_map, stats = reachability.build_testable(
            env.game.state, spawn, flag_corridor=corridor,
            arcs=reachability.load_jump_arcs(),
            observed=reachability.load_observed_reach())

        for label, got, want in (
            ("solid rects", stats['n_solid_rects'], config.EXPECTED_SOLID_RECTS),
            ("solid px", stats['solid_px'], config.EXPECTED_SOLID_PX),
        ):
            if got != want:
                raise SystemExit(
                    f"Level geometry changed: {label} got {got:,}, expected "
                    f"{want:,}.\nIf mario_clone's layout genuinely changed, "
                    f"update the EXPECTED_* constants and say so in the "
                    f"commit - the coverage denominator moves with them.")

        w = stats['world_raster_px']
        print("=" * 70)
        print("RECONCILIATION")
        print("=" * 70)
        print(f"  world raster            {w:>12,}   (informational only)")
        print(f"  solid px                {stats['solid_px']:>12,}")
        print(f"  valid anchors           {stats['valid_anchors']:>12,}")
        print(f"  standable anchors       {stats['standable_anchors']:>12,}")
        print()
        for m in ('a', 'b', 'c'):
            px = stats[f'method_{m}_px']
            print(f"  Method {m.upper()}                {px:>12,}   "
                  f"{100.0 * px / w:>6.2f}% of world")
        print()
        print("  ordering C <= B <= A    OK")
        print(f"  B vs C delta            {stats['bc_delta_pct']:>11.2f}%")
        print(f"  lattice                 {stats['lattice']:>11} px")
        print(f"  C overshoot above B     "
              f"{stats['c_lattice_overshoot_px']:>11,} px"
              f"   {'(exact - no lattice term)' if stats['lattice'] == 1 else ''}")
        print(f"  Method C (trimmed)      {stats['method_c_trimmed_px']:>12,} px")
        if stats.get('flag_trigger') is not None:
            print(f"  flag trigger            {stats['flag_trigger']}   grab band to x "
                  f"{stats['flag_band_right']}")
            print(f"  - past it, off the scripted path  {stats['flag_removed_px']:>9,} px")
        if stats.get('arcs_removed_px') is not None:
            print(f"  - no measured jump arc reaches it {stats['arcs_removed_px']:>9,} px"
                  f"   (real play keeps {stats['observed_px']:,})")
        print(f"  ADOPTED                 Method {stats['adopted_method']} + flag trigger"
              f" + real-arc envelope  =  {stats['testable_total']:,} px")
        print("=" * 70)

        if args.dry_run:
            print("\n--dry-run: nothing written.")
            return
        reachability.save_masks(config.REACHABLE_MASK_PATH, solid, testable,
                                class_map, stats)
        _s, t, meta = reachability.load_masks()
        assert int(t.sum()) == stats['testable_total'], "round-trip mismatch"
        write_config(stats, meta['testable_fingerprint'])

        size = os.path.getsize(config.REACHABLE_MASK_PATH) / 1e6
        print(f"\nWrote {config.REACHABLE_MASK_PATH} ({size:.2f} MB)")
        print(f"  fingerprint : {meta['testable_fingerprint'][:32]}...")
        print("  TESTABLE_TOTAL written to exploration/config.py")
    finally:
        env.close_window()


if __name__ == "__main__":
    main()
