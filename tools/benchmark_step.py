"""How long one agent step takes, and where the time goes.

    python tools/benchmark_step.py                 # 1,000 agent steps per stack
    python tools/benchmark_step.py --steps 3000 --profile

Times the same random action sequence through three stacks, each under the
training observation chain (wrap_observation):

    bare     CustomMarioEnv alone - the engine, rendering and the 84x84 view
    legacy   + GlitchHunterWrapper in legacy_completion mode
    qa       + GlitchHunterWrapper in qa_exploration mode (coverage, lifecycle)

The difference between the rows is what the project's own code costs on top
of the engine. --profile adds a cProfile table of the qa stack, which is how
the lazy frame capture in custom_mario_env.SkipObservation was found.
Headless and read-only: nothing is written.
"""
from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.cli import prepare_tool

ROOT = prepare_tool()

import gymnasium as gym
import numpy as np

from agent_logic import GlitchHunterWrapper
from custom_mario_env import CustomMarioEnv, wrap_observation
from exploration.coverage import SpatialCoverage, load_testable

MOVES = (1, 2, 3, 4, 6, 8)


def build(kind: str, base: CustomMarioEnv) -> gym.Env[Any, Any]:
    if kind == "bare":
        return wrap_observation(base)
    coverage = SpatialCoverage(testable_mask=load_testable()) if kind == "qa" else None
    mode = "qa_exploration" if kind == "qa" else "legacy_completion"
    return wrap_observation(GlitchHunterWrapper(base, reward_mode=mode, coverage=coverage))


def run(env: gym.Env[Any, Any], steps: int, seed: int) -> float:
    """Seconds for `steps` agent steps of a fixed random action sequence."""
    rng = np.random.default_rng(seed)
    actions = rng.choice(MOVES, size=steps)
    env.reset()
    t0 = time.perf_counter()
    for a in actions:
        _obs, _r, terminated, truncated, _info = env.step(int(a))
        if terminated or truncated:
            env.reset()
    return time.perf_counter() - t0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument("--steps", type=int, default=1000, help="agent steps per stack")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", action="store_true", help="cProfile the qa stack")
    args = ap.parse_args(argv)

    base = CustomMarioEnv()
    rows = []
    for kind in ("bare", "legacy", "qa"):
        env = build(kind, base)
        run(env, 50, args.seed)                         # warm caches and the window
        rows.append((kind, run(env, args.steps, args.seed)))
        base.episode_time_units, base.end_on_level_complete = None, False

    print(f"{args.steps:,} agent steps ({args.steps * 4:,} engine frames) per stack\n")
    print(f"  {'stack':<8}{'ms / agent step':>18}{'agent steps / s':>18}{'vs bare':>10}")
    bare = rows[0][1]
    for kind, secs in rows:
        per = 1000.0 * secs / args.steps
        print(f"  {kind:<8}{per:>18.3f}{args.steps / secs:>18.1f}{secs / bare:>9.2f}x")

    if args.profile:
        env = build("qa", base)
        profiler = cProfile.Profile()
        profiler.enable()
        run(env, args.steps, args.seed)
        profiler.disable()
        print()
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(25)
    return 0


if __name__ == "__main__":
    sys.exit(main())
