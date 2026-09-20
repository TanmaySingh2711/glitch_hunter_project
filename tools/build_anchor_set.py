"""Record the ANCHOR states: what a healthy policy sees when it plays Level 1-1.

    python tools/build_anchor_set.py            # from the healthy 6,032,768 pair

Completion retention is measured by tools/evaluate_completion.py, and across
six checkpoints it is predicted almost exactly by one number: the mean
KL(healthy || candidate) on the states the HEALTHY policy visits when it plays
the retention protocol (Pearson r = -0.962; retention ~= 46.7% - 51.5 * KL).
AnchorConsolidationCallback keeps that number under a budget during training,
so it needs a fixed set of such states to measure it on. This tool records
them.

The seeds are DISJOINT from the retention protocol's (20260911 + 0..499), so
the training signal never sees the exact episodes it is later judged by.
Recorded through the protocol env itself (bare engine, sampled actions), so the
observations are exactly what the policy would see.

Writes exploration_data/anchor_states.npz with the reference checkpoint's
sha256 inside it; the callback refuses an anchor set recorded for a different
reference.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.cli import prepare_tool

ROOT = prepare_tool()

import numpy as np

from common.fileio import atomic_write, canonical_sha256
from exploration import config

DEFAULT_REFERENCE = os.path.join("checkpoints_qa", "pre_main_6032768", "glitch_hunter_qa.zip")
ANCHOR_SEED0 = 1_000                  # retention protocol seeds are 20260911 + i
ANCHOR_EPISODES = 24
ANCHOR_STEP_CAP = 900
ANCHOR_EVERY = 2                      # consecutive frames are near-duplicates


def record(reference: str, episodes: int, seed0: int, step_cap: int,
           every: int) -> np.ndarray:
    from custom_mario_env import CustomMarioEnv
    from evaluation import completion as ce
    model = ce.load_policy(reference)
    actor = ce.PolicyActor(model, "sampled")
    proto = ce.make_protocol(episodes, seed0)
    proto["max_agent_steps"] = step_cap
    base = CustomMarioEnv()
    states = []
    with ce.protocol_env(base, proto) as (env, _probe):
        for i in range(episodes):
            rng = np.random.default_rng(seed0 + i)
            obs, _ = env.reset()
            done, t = False, 0
            while not done:
                if t % every == 0:
                    states.append(np.asarray(obs, dtype=np.uint8))
                obs, _r, term, trunc, _ = env.step(actor(obs, rng))
                done, t = term or trunc, t + 1
    return np.stack(states)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", default=DEFAULT_REFERENCE)
    ap.add_argument("--out", default=config.ANCHOR_STATES_PATH)
    ap.add_argument("--episodes", type=int, default=ANCHOR_EPISODES)
    args = ap.parse_args(argv)
    states = record(args.reference, args.episodes, ANCHOR_SEED0, ANCHOR_STEP_CAP, ANCHOR_EVERY)
    seeds = np.arange(ANCHOR_SEED0, ANCHOR_SEED0 + args.episodes, dtype=np.int64)
    with atomic_write(args.out) as fh:
        np.savez_compressed(fh, states=states, seeds=seeds,
                            reference_path=np.str_(args.reference),
                            reference_sha256=np.str_(canonical_sha256(args.reference)),
                            step_cap=np.int64(ANCHOR_STEP_CAP), every=np.int64(ANCHOR_EVERY))
    size = os.path.getsize(args.out) / 1e6
    print(f"recorded {len(states):,} anchor states from {args.episodes} episodes "
          f"(seeds {seeds[0]}..{seeds[-1]}) of {args.reference}")
    print(f"wrote {args.out} ({size:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
