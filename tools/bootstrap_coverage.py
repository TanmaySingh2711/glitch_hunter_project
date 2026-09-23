"""Seed the coverage map with the 6M brain's habitual routes.

    python tools/bootstrap_coverage.py

═══════════════════════════════════════════════════════════════════════════
READ THIS BEFORE TRUSTING THE OUTPUT
═══════════════════════════════════════════════════════════════════════════
This does NOT reconstruct where the agent actually went during its first six
million steps. That data was never recorded - no world-space visitation was
persisted anywhere in this repository, at any point, and no amount of
processing can recover it after the fact.

What this produces is an APPROXIMATION of the 6M policy's habitual routes:
the same brain, replayed now, under its own native legacy reward, for a few
dozen episodes. It answers "where does this policy tend to go" - which is the
question that actually matters for seeding the map - but it is a fresh
measurement, not a recovered history.

Two concrete ways it differs from the truth:

  * It is far SMALLER than the real historical footprint. Six million steps
    of training included a long, exploratory, high-entropy early phase that
    wandered into places the converged policy now never visits. Those places
    will look unexplored, and the agent will be paid novelty for rediscovering
    them. That is a deliberate, acceptable trade: over-crediting a little
    early novelty is harmless, whereas under-crediting - marking pixels as
    explored that the agent has never actually reached - would starve it of
    the signal for genuinely new ground.

  * It samples from the policy rather than acting greedily. A deterministic
    replay would produce forty near-identical episodes and a single thin
    ribbon of coverage, which would badly understate even the current
    policy's own range.

The bootstrap is therefore a FLOOR on "already familiar", not a ceiling.

The model is used for inference only. learn() is never called, no gradient is
taken, and nothing here writes to mario_brain_checkpoint.zip, checkpoints/ or
the main brain - it only reads the baseline and writes one .npz.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.cli import prepare_tool

ROOT = prepare_tool()

from typing import Any

import gymnasium as gym
import numpy as np

from exploration import config
from exploration.coverage import SpatialCoverage, load_testable


def build_env(coverage: SpatialCoverage) -> gym.Env[Any, Any]:
    from gymnasium.wrappers import TimeLimit

    from agent_logic import GlitchHunterWrapper
    from custom_mario_env import CustomMarioEnv, wrap_observation

    # LEGACY reward on purpose. The point is to record the routes this brain
    # actually learned, and it learned them under this reward. Running it
    # under the QA reward would be recording where a differently-motivated
    # agent goes, which is not the thing being approximated.
    env = wrap_observation(GlitchHunterWrapper(CustomMarioEnv(), reward_mode="legacy_completion",
                                               coverage=coverage))
    return TimeLimit(env, max_episode_steps=config.LEGACY_EPISODE_MAX_STEPS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=config.BOOTSTRAP_EPISODES)
    ap.add_argument("--model", default=config.BASELINE_MODEL)
    ap.add_argument("--out", default=config.BOOTSTRAP_COVERAGE)
    ap.add_argument("--deterministic", action="store_true",
                    help="greedy actions; produces a much thinner map (see "
                         "the module docstring) - not the default for a reason")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(
            f"{args.model} not found.\n"
            f"The bootstrap reads the immutable 6M baseline. Restore it from "
            f"git (git checkout -- mario_brain_checkpoint.zip) if it is missing.")

    reachable = load_testable()
    if reachable is None:
        print("[WARN] exploration_data/reachable_mask.npz not found - the "
              "coverage %/remaining figures below will be unavailable. Run "
              "tools/build_reachability.py first if you want them.")

    coverage = SpatialCoverage(testable_mask=reachable)
    env = build_env(coverage)

    import torch
    from stable_baselines3 import PPO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} (inference only)...")
    model = PPO.load(args.model, device=device)
    print(f"  model num_timesteps = {model.num_timesteps:,}")

    per_episode = []
    t0 = time.perf_counter()
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            before = coverage.total_unique()
            steps = 0
            while True:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, _r, terminated, truncated, _info = env.step(int(action))
                steps += 1
                if terminated or truncated:
                    break
            gained = coverage.total_unique() - before
            per_episode.append(gained)
            print(f"  episode {ep + 1:>3}/{args.episodes}: {steps:>5} steps, "
                  f"+{gained:>7,} new px, total {coverage.total_unique():>9,}",
                  flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted - saving what was recorded so far.")
    finally:
        env.close()

    total = coverage.total_unique()
    if total == 0:
        raise SystemExit("Recorded zero coverage - refusing to write an empty "
                         "bootstrap file. Something is wrong with the env.")

    # Paired with the BASELINE's step count, not with 0. The coverage file and
    # the model checkpoint are loaded as a matched pair, and a mismatch is a
    # hard error (see SpatialCoverage.load) precisely so a bootstrap can never
    # be silently attached to the wrong brain.
    coverage.save(args.out, model_timesteps=model.num_timesteps)

    print("\n" + "=" * 68)
    print("BOOTSTRAP COVERAGE WRITTEN")
    print("=" * 68)
    print(f"  file                 : {args.out}")
    print(f"  size                 : {os.path.getsize(args.out) / 1e6:.2f} MB")
    print(f"  paired with          : num_timesteps={model.num_timesteps:,}")
    print(f"  episodes             : {len(per_episode)}"
          f"{'  (deterministic)' if args.deterministic else '  (sampled)'}")
    print(f"  unique world px      : {total:,}")
    if per_episode:
        print(f"  new px per episode   : first={per_episode[0]:,}  "
              f"median={int(np.median(per_episode)):,}  "
              f"last={per_episode[-1]:,}")
    if coverage.testable_total:
        coverage.assert_consistent()
        print(f"  world raster px      : {config.WORLD_RASTER_PX:,}"
              f"   (informational only)")
        print(f"  testable px          : {coverage.testable_total:,}"
              f"   (method {config.ADOPTED_METHOD})")
        print(f"  covered testable px  : {coverage.covered_testable():,}")
        print(f"  remaining testable px: {coverage.remaining():,}")
        nb = coverage.noncoverage_breakdown()
        print(f"  noncoverage px       : {nb['total']:,}   (NOT coverage)")
        print(f"    expected           : {nb['expected_total']:,}"
              f"   normal engine behaviour")
        print(f"    model gap          : {nb['model_gap_total']:,}")
        print(f"    anomalous          : {nb['anomalous_total']:,}"
              f"   -> glitch evidence")
        print(f"  COVERAGE             : {coverage.coverage_pct():.2f}%")
    print(f"  out-of-grid events   : {coverage.oob_events}")
    print(f"  wall clock           : {time.perf_counter() - t0:.0f}s")
    print()
    print("  COVERAGE is covered_testable / testable_coverable. The denominator")
    print("  is Method C at a 1 px lattice - connectivity-checked from spawn and")
    print("  bounded by the measured jump envelope, so it is not a loose")
    print("  geometric bound. See exploration/config.py for the derivation.")
    print("  This file approximates the 6M policy's habitual routes. It is not")
    print("  a recovered history - see the module docstring.")


if __name__ == "__main__":
    main()
