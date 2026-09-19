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
        (f"#   Method C (BFS)        = {stats['method_c_px']:,}"
        f"   <- ADOPTED"),
        f"#   B vs C delta          = {stats['bc_delta_pct']:.2f}%",
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

    new = reachability.classify_noncoverage(
        solid, testable, b_px, mutable_world=mutable, sweep_world=sweep)

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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="build and reconcile the mask, print the result, write nothing")
    ap.add_argument("--class-map-only", action="store_true",
                    help="recompute ONLY the noncoverage taxonomy in the "
                         "existing bundle; the solid mask, the testable mask, "
                         "the denominator and its fingerprint are copied back "
                         "unchanged and verified")
    args = ap.parse_args(argv)
    from custom_mario_env import CustomMarioEnv

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

        solid, testable, class_map, stats = reachability.build_testable(
            env.game.state, spawn)

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
        print(f"  ADOPTED                 Method {stats['adopted_method']}"
              f"  =  {stats['testable_total']:,} px")
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
