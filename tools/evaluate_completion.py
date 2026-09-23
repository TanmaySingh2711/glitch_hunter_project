"""Completion retention: evaluate any checkpoint against the 6M baseline.

    # A QA checkpoint (or any other) - the everyday use:
    python tools/evaluate_completion.py checkpoints_qa/glitch_hunter_qa_6400000_steps.zip

    # Re-compare a result already on disk, without replaying it:
    python tools/evaluate_completion.py --compare evaluation/results/<file>.json

    # (Re)build the baseline itself - done once, from the frozen 6M brain:
    python tools/evaluate_completion.py --make-baseline --episodes 500 --seed 20260911

The protocol - episode count, seeds, action mode, engine settings - is read
from the baseline file on every evaluation, never from the command line, so
every checkpoint is played exactly the way the baseline was. --workers only
changes how fast that happens: each episode is a function of (checkpoint,
seed) alone, so the result is identical for any worker count.

Inference only. Results go to evaluation/results/; nothing under
a brain, checkpoints_qa/ or exploration_data/ is written.
See evaluation/completion.py for the protocol and why it is what it is.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.cli import prepare_tool


# ─── HEADLESS BY DEFAULT ───
# This used to force a visible window (headless=False), and with --workers 12
# every worker opened its own - twelve game windows filling the screen for the
# length of a 500-episode run. The window never affected the result, and that
# was measured rather than assumed: the healthy 6,032,768-step checkpoint
# replayed headless (SDL dummy driver, 8 worker processes) over the first 40
# protocol seeds reproduced the stored WINDOWED result exactly, 40 of 40
# episodes identical in end reason, agent steps, substeps and max_x.
# --windowed restores the old behaviour for anyone who wants to watch.
# Read from argv directly because the driver must be chosen before pygame
# is imported, i.e. before argparse runs; spawned workers inherit it.
def wants_window(argv: list[str]) -> bool:
    return '--windowed' in argv


ROOT = prepare_tool(headless=not wants_window(sys.argv[1:]))

from evaluation import completion as ce
from exploration import config

# How far below the baseline, in units of the protocol's own no-change noise
# (evaluation.completion.derive_thresholds), a checkpoint may fall.
Z_WARNING = 2.0
Z_REGRESSED = 4.0


def _report(result: dict[str, Any], comparison: dict[str, Any] | None = None) -> None:
    s = result['summary']
    ck = result['checkpoint']
    print(f"\ncheckpoint  {ck['path']}  ({ck['num_timesteps']:,} steps, sha256 {ck['sha256'][:12]})")
    lo, hi = s['completion_rate_ci95']
    print(f"completion  {s['completed']}/{s['episodes']} = {s['completion_rate']:.1%} "
          f"(95% CI {lo:.1%}-{hi:.1%}); castle door {s['castle_door']}, flagpole {s['flagpole']}")
    print(f"endings     {s['ends']}   deaths by cause {s['death_causes']}")
    print(f"max-x       mean {s['max_x']['mean']:.0f}  median {s['max_x']['median']:.0f} | "
          f"progress mean {s['progress']['mean']:.3f} median {s['progress']['median']:.3f}")
    if s['completion_agent_steps']:
        c = s['completion_agent_steps']
        print(f"to finish   agent steps mean {c['mean']:.0f} median {c['median']:.0f} "
              f"(p10 {c['p10']:.0f}, p90 {c['p90']:.0f}); within the original 401-unit "
              f"clock: {s['completed_within_legacy_clock']}/{s['completed']}")
    g = result['greedy']
    print(f"greedy      {g['end']} after {g['agent_steps']} steps, max-x {g['max_x']}")
    if comparison:
        c = comparison
        print(f"\nvs 6M       completion {c['completion_rate']:.1%} vs {c['baseline_completion_rate']:.1%} "
              f"({c['completion_delta']:+.1%}) | mean progress {c['mean_progress']:.3f} vs "
              f"{c['baseline_mean_progress']:.3f} ({c['progress_delta']:+.3f})")
        print(f"VERDICT     {c['verdict']}" + (f"  - {'; '.join(c['reasons'])}" if c['reasons'] else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('checkpoint', nargs='?', help='checkpoint .zip to evaluate')
    ap.add_argument('--baseline', default=ce.BASELINE_PATH)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--out', help='result path (default evaluation/results/<name>.json)')
    ap.add_argument('--compare', metavar='RESULT', help='compare an existing result file')
    ap.add_argument('--make-baseline', action='store_true')
    ap.add_argument('--episodes', type=int, help='baseline only')
    ap.add_argument('--seed', type=int, help='baseline only')
    ap.add_argument('--force', action='store_true', help='overwrite an existing baseline')
    ap.add_argument('--windowed', action='store_true',
                    help='show the game window(s); results are identical either way')
    args = ap.parse_args(argv)

    if args.make_baseline:
        if os.path.exists(args.baseline) and not args.force:
            sys.exit(f"{args.baseline} exists; the baseline is fixed once measured (--force to redo)")
        if args.episodes is None or args.seed is None:
            sys.exit("--make-baseline needs --episodes and --seed")
        model = args.checkpoint or os.path.join(ROOT, config.BASELINE_MODEL)
        result = ce.evaluate(model, ce.make_protocol(args.episodes, args.seed),
                             workers=args.workers)
        if not result['checkpoint']['is_6m_master']:
            sys.exit("the baseline must be the 6M master (sha256 mismatch)")
        result['kind'] = 'completion_retention_baseline'
        result['thresholds'] = ce.derive_thresholds(result['episodes'], Z_WARNING, Z_REGRESSED)
        ce.save(result, args.baseline)
        _report(result)
        print(f"\nthresholds  {json.dumps(result['thresholds'], indent=1)}")
        print(f"baseline written to {args.baseline}")
        return 0

    baseline = ce.load(args.baseline)
    if args.compare:
        result = ce.load(args.compare)
    elif args.checkpoint:
        result = ce.evaluate(args.checkpoint, baseline['protocol'], workers=args.workers)
        stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        out = args.out or os.path.join(
            ce.RESULTS_DIR, f"{stem}_{result['checkpoint']['sha256'][:8]}.json")
        result['comparison'] = ce.compare(result, baseline)
        ce.save(result, out)
        print(f"result written to {out}")
    else:
        ap.error("give a checkpoint, --compare RESULT, or --make-baseline")
    comparison = ce.compare(result, baseline)
    _report(result, comparison)
    return {'HEALTHY': 0, 'WARNING': 1, 'REGRESSED': 2}[comparison['verdict']]


if __name__ == '__main__':
    sys.exit(main())
