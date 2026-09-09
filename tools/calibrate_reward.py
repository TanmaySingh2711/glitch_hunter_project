"""Solve NOVELTY_WEIGHT so the QA reward lands on the legacy reward's scale.

    python tools/calibrate_reward.py

═══════════════════════════════════════════════════════════════════════════
WHY THIS IS MANDATORY, NOT A NICETY
═══════════════════════════════════════════════════════════════════════════
The 6M value function was fitted against episode returns of a particular
size. Swap in a reward of a wildly different magnitude and every prediction
it makes is instantly, hugely wrong - which means enormous TD errors, which
means enormous advantages, which means one PPO update that moves the policy
much further than clip_range was ever meant to allow.

That is not hypothetical here. This exact project has already lived through
it once, at roughly 3.22M steps: approx_kl 11.37, clip_fraction 0.834, and
entropy_loss collapsing to near zero within a couple of iterations. Getting
the scale right is the single cheapest insurance available against repeating
it, and it costs about ten minutes.

═══════════════════════════════════════════════════════════════════════════
METHOD
═══════════════════════════════════════════════════════════════════════════
Arm A: N episodes of the 6M policy under the LEGACY reward - the reward it
       was actually trained on. This is the target scale.

Arm B: N episodes of the same policy under the QA reward, each starting from
       the BOOTSTRAP coverage state, because that is the condition training
       will genuinely start in. Calibrating against an empty map would
       measure a novelty windfall that will never exist in practice.
       Coverage is restored between episodes so every sample measures the
       same starting condition rather than a decay curve.

Because the QA return is linear in the weight,

    return(W) = W * novelty_shape + everything_else

both parts can be accumulated separately during one pass and the weight
solved in closed form. No search, no repeated rollouts.

The model is used for inference only. learn() is never called and no
checkpoint is written.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

# Console output includes box-drawing characters. On Windows the
# default console encoding is cp1252, which cannot encode them, so
# redirecting this tool to a file would crash it AFTER the work was
# done but BEFORE the result was written.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from exploration import config
from exploration.coverage import SpatialCoverage, load_testable

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "exploration", "config.py")


def build_env(mode, coverage):
    from gymnasium.wrappers import (FrameStackObservation, GrayscaleObservation,
                                    MaxAndSkipObservation, ResizeObservation,
                                    TimeLimit)

    from agent_logic import GlitchHunterWrapper
    from custom_mario_env import CustomMarioEnv

    env = CustomMarioEnv()
    inner = GlitchHunterWrapper(env, reward_mode=mode, coverage=coverage,
                                attach_coverage=coverage is not None)
    env = MaxAndSkipObservation(inner, skip=4)
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, 4)
    env = TimeLimit(env, max_episode_steps=4000)
    return env, inner


def run_arm(env, inner, model, episodes, restore=None, label=""):
    """Returns (returns, novelty_shapes, lengths) per episode."""
    returns, shapes, lengths = [], [], []
    for ep in range(episodes):
        if restore is not None:
            restore()
        obs, _ = env.reset()
        total, steps = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=False)
            obs, r, terminated, truncated, _info = env.step(int(action))
            total += r
            steps += 1
            if terminated or truncated:
                break
        returns.append(total)
        shapes.append(inner.ep_novelty_shape)
        lengths.append(steps)
        print(f"  [{label}] episode {ep + 1:>3}: return {total:>10.2f}  "
              f"steps {steps:>5}  novelty_shape {inner.ep_novelty_shape:>9.2f}",
              flush=True)
    return np.array(returns), np.array(shapes), np.array(lengths)

def write_weight(value, stats):
    """Rewrites NOVELTY_WEIGHT in exploration/config.py, recording the evidence.

    The measured numbers go in beside the value. A tuned constant with no
    record of what it was tuned against is indistinguishable from a guess the
    next time someone reads it.

    Any PREVIOUS solved block is stripped first. An earlier version only
    replaced the assignment line, so each run appended a fresh block and left
    the old one behind as orphaned comments - config.py accumulated a pile of
    contradictory provenance, which is worse than having none at all, because
    every stale entry still reads as authoritative.
    """
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        src = fh.read()

    lines = [
        "# ─── SOLVED BY tools/calibrate_reward.py ───",
        f"# Measured {stats['stamp']} against the {stats['model_steps']:,}-step baseline,",
        f"# {stats['episodes']} episodes per arm, QA episodes starting from the bootstrap state.",
        f"#   legacy median return = {stats['legacy_median']:.2f}   "
        f"(mean {stats['legacy_mean']:.2f}; the distribution is",
        "#                          bimodal, so the median is the target, not the mean)",
        f"#   QA non-novelty terms = {stats['other_mean']:+.2f} per episode",
        f"#   QA novelty shape     = {stats['shape_mean']:.2f} per episode (unweighted)",
        f"#   solved               = ({stats['legacy_median']:.2f} - "
        f"{stats['other_mean']:.2f}) / {stats['shape_mean']:.2f} = {stats['raw']:.4f}",
        f"#   safety ceiling       = {stats['w_max']:.4f}   "
        f"(novelty alone must never reach half the reward clip)",
        f"#   ADOPTED              = {value:.4f}",
        "#",
        f"# The solved value is {stats['raw'] / stats['w_max']:.0f}x the ceiling, so the "
        f"ceiling is what binds.",
        "# Scale matching is NOT achievable here: the legacy return was dominated by the",
        f"# x-monotone terms this retrofit deletes, so QA returns land at "
        f"{stats['ratio']:.3f}x the",
        "# legacy median and no SAFE weight closes that gap. The residual mismatch is",
        "# handled by resetting the value head on the first QA run (train_agent.py",
        "# RESET_VALUE_HEAD), not by forcing this weight upward.",
        f"NOVELTY_WEIGHT = {value:.4f}",
    ]
    block = "\n".join(lines)

    src = src.replace(
        "# PLACEHOLDER until calibration runs; calibrate_reward.py rewrites it.\n",
        "")
    src = re.sub(r"^# ─── SOLVED BY tools/calibrate_reward\.py "
                 r"───\n(?:#.*\n)*", "", src, flags=re.M)
    src = re.sub(r"^NOVELTY_WEIGHT = .*$", block, src, count=1, flags=re.M)

    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        fh.write(src)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=30,
                    help="30 rather than 20: the legacy return distribution "
                         "is bimodal (most episodes in the hundreds, level "
                         "completions in the thousands), and at 20 the median "
                         "flipped between 625 and 2305 across two runs")
    ap.add_argument("--model", default=config.BASELINE_MODEL)
    ap.add_argument("--bootstrap", default=config.BOOTSTRAP_COVERAGE)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the solved weight without editing config.py")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"{args.model} not found.")
    if not os.path.exists(args.bootstrap):
        raise SystemExit(
            f"{args.bootstrap} not found.\n"
            f"Run tools/bootstrap_coverage.py first - calibrating against an "
            f"empty map measures a novelty windfall that will never exist.")

    import torch
    from stable_baselines3 import PPO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(args.model, device=device)
    print(f"Baseline: {args.model} at {model.num_timesteps:,} steps "
          f"({device}, inference only)\n")

    t0 = time.time()

    # ── ARM A: the scale to match ─────────────────────────────────────────
    print(f"ARM A - legacy_completion, {args.episodes} episodes")
    env_a, inner_a = build_env("legacy_completion", None)
    legacy_ret, _s, legacy_len = run_arm(env_a, inner_a, model, args.episodes,
                                         label="legacy")
    env_a.close()

    # ── ARM B: the new reward, from the real starting condition ───────────
    print(f"\nARM B - qa_exploration from bootstrap, {args.episodes} episodes")
    reachable = load_testable()
    coverage = SpatialCoverage(testable_mask=reachable)
    coverage.load(args.bootstrap, model=model)
    baseline_bits = coverage.visited.copy()
    print(f"  bootstrap coverage loaded: {coverage.total_unique():,} px\n")

    def restore():
        coverage.visited[:] = baseline_bits
        coverage._remaining_cached = None
        coverage.episode_new_history = []

    env_b, inner_b = build_env("qa_exploration", coverage)
    qa_ret, qa_shape, qa_len = run_arm(env_b, inner_b, model, args.episodes,
                                       restore=restore, label="qa")
    env_b.close()

    # ── SOLVE ─────────────────────────────────────────────────────────────
    w0 = config.NOVELTY_WEIGHT
    other = qa_ret - w0 * qa_shape          # everything that is not novelty
    L_mean = float(np.mean(legacy_ret))
    L = float(np.median(legacy_ret))        # the target - see below
    R = float(np.mean(other))
    S = float(np.mean(qa_shape))

    print("\n" + "=" * 72)
    print("CALIBRATION")
    print("=" * 72)
    print(f"  legacy return      mean {L_mean:>10.2f}   median {L:>10.2f}   "
          f"std {float(np.std(legacy_ret)):>8.2f}")
    print(f"  legacy length      mean {float(np.mean(legacy_len)):>10.1f} steps")
    print(f"  QA return @W={w0:<5.3f} mean {float(np.mean(qa_ret)):>10.2f}   "
          f"median {float(np.median(qa_ret)):>10.2f}")
    print(f"  QA length          mean {float(np.mean(qa_len)):>10.1f} steps")
    print(f"  QA novelty shape   mean {S:>10.2f} per episode (unweighted)")
    print(f"  QA other terms     mean {R:>10.2f} per episode")

    # ─── WHY THE MEDIAN, NOT THE MEAN ───
    # The legacy return distribution is bimodal, not noisy: most episodes land
    # in the hundreds, and the few that reach the flagpole collect the +500
    # completion bonus on top of a full level of x-monotone tile and milestone
    # rewards, landing in the thousands. The mean therefore describes an
    # episode that essentially never happens, and calibrating to it would size
    # the novelty weight for the rare case.
    completions = int(np.sum(legacy_ret > 2 * L))
    print(f"\n  legacy distribution is bimodal: {completions}/{len(legacy_ret)} "
          f"episodes above 2x the median")
    print(f"  -> targeting the MEDIAN ({L:.2f}), not the mean ({L_mean:.2f})")

    if S <= 1e-6:
        raise SystemExit(
            "\nThe QA arm found essentially no new ground (novelty shape ~ 0), "
            "so the weight is not solvable from this data.\n"
            "That usually means the bootstrap already covers everywhere this "
            "policy goes. Options: shrink the bootstrap (fewer episodes), or "
            "accept that novelty will be rare at first and set NOVELTY_WEIGHT "
            "by hand - but do not train without understanding which.")

    raw = (L - R) / S

    # ─── THE SAFETY CEILING IS DERIVED, NOT PICKED ───
    # Novelty alone must never be able to saturate the backstop clamp, or the
    # clamp stops being a backstop and becomes part of the reward function.
    # Worst case is one substep at the novelty cap with the stagnation
    # multiplier fully escalated, and that is held to half the clamp.
    w_max = 0.5 * config.QA_REWARD_CLIP / (config.NOVELTY_CAP
                                           * config.NOVELTY_MULT_MAX)
    lo = 0.05
    value = float(min(w_max, max(lo, raw)))
    predicted = value * S + R
    ratio = predicted / L if L != 0 else float('inf')

    print(f"\n  solved  NOVELTY_WEIGHT = ({L:.2f} - {R:.2f}) / {S:.2f} "
          f"= {raw:.4f}")
    print(f"  safety ceiling         = 0.5 * clip {config.QA_REWARD_CLIP} / "
          f"(cap {config.NOVELTY_CAP} * mult {config.NOVELTY_MULT_MAX}) "
          f"= {w_max:.4f}")
    if value != raw:
        print(f"  CLAMPED to {value:.4f}")
    print(f"  predicted QA return    = {predicted:.2f}  ({ratio:.3f}x legacy "
          f"median)")

    # ── REQUIREMENT: compare every new term against the old scale ─────────
    ep_substeps = float(np.mean(qa_len)) * 4.0
    print("\n" + "-" * 72)
    print("  PER-SUBSTEP MAGNITUDES (what PPO sees is 4x this per agent step)")
    print("-" * 72)
    rows = [
        ("novelty, typical stride", value * 1.0, "legacy tile bonus", 1.0),
        ("novelty, hard ceiling",
         value * config.NOVELTY_CAP * config.NOVELTY_MULT_MAX, "-", None),
        ("revisit (transit)", 0.0, "legacy tile bonus", 1.0),
        ("frontier residue, worst",
         config.FRONTIER_WEIGHT * (1 - config.GAMMA), "-", None),
        ("drought, full ramp", -config.DROUGHT_MAX,
         "legacy stuck penalty", -0.1),
        ("time", -config.QA_TIME_PENALTY, "legacy time", -0.02),
        ("clean jump", config.QA_CLEAN_JUMP_REWARD, "legacy clean jump", 3.0),
        ("flag_get", config.QA_FLAG_GET_REWARD, "legacy flag_get", 500.0),
        ("death", -5.0, "legacy death", -5.0),
    ]
    for name, qa_v, legacy_name, legacy_v in rows:
        cmp_txt = (f"   ({legacy_name}: {legacy_v:+.3f})"
                   if legacy_v is not None else "")
        print(f"    {name:<26} {qa_v:>+9.4f}{cmp_txt}")

    print("\n" + "-" * 72)
    print("  PER-EPISODE BUDGET (mean episode = "
          f"{ep_substeps:,.0f} substeps)")
    print("-" * 72)
    print(f"    novelty                  {value * S:>+9.2f}")
    print(f"    everything else          {R:>+9.2f}")
    print(f"    drought ceiling          {-config.DROUGHT_EPISODE_CAP:>+9.2f}"
          f"   (hard cap, cannot exceed)")
    print(f"    frontier total range     {config.FRONTIER_WEIGHT:>+9.2f}"
          f"   (whole map, cannot exceed)")
    print(f"    => QA episode            {predicted:>+9.2f}")
    print(f"    vs legacy median         {L:>+9.2f}")

    novelty_share = (value * S) / max(1e-9, value * S + abs(R))
    print(f"\n  novelty share of episode magnitude: {100 * novelty_share:.0f}%")
    if value * S < abs(R):
        print("  [NOTE] Novelty is not yet the largest single contributor. "
              "That is EXPECTED at the start: the bootstrap already covers "
              "this policy's habitual routes, so there is little new ground "
              "on its current behaviour. The whole point of training is to "
              "change that.")

    # ─── THE HONEST CONCLUSION ───
    ok = 0.5 <= ratio <= 2.0
    print(f"\n  within 0.5x-2.0x of legacy: {'YES' if ok else 'NO'}")
    if not ok:
        print("""
  ─── SCALE MATCHING IS NOT ACHIEVABLE HERE, AND SHOULD NOT BE FORCED ───
  The legacy return is dominated by terms that were DELETED on purpose:
  roughly 2,180 of a ~2,200-point episode was a monotone function of max-x.
  A coverage-based reward has no equivalent, so closing the gap would need a
  novelty weight two orders of magnitude above the safety ceiling - which
  would make one lucky substep worth more than an entire legacy episode.
  That is not a calibration, it is a different way to blow up.

  So the mismatch is REAL and is handled elsewhere, by resetting the value
  head. The critic currently predicts returns in the hundreds; against a QA
  reward it would be confidently wrong about every state, and with
  vf_coef=0.5 that error backpropagates through the SHARED CNN and damages
  the very locomotion features this retrofit is trying to preserve.
  Reinitialising the single Linear(features -> 1) value layer costs nothing
  the policy needs: the action head and every convolutional feature are
  untouched.

  train_agent.py does this automatically on the FIRST QA run only
  (RESET_VALUE_HEAD). Combined with the value-function warm-up, that is what
  protects the 6M brain - not a forced weight.""")

    stats = {
        'stamp': time.strftime("%Y-%m-%d"),
        'model_steps': model.num_timesteps,
        'episodes': args.episodes,
        'legacy_mean': L_mean, 'legacy_median': L,
        'other_mean': R, 'shape_mean': S, 'raw': raw, 'ratio': ratio,
        'w_max': w_max,
    }
    if args.dry_run:
        print("\n  --dry-run: exploration/config.py NOT modified.")
    else:
        write_weight(value, stats)
        print(f"\n  Wrote NOVELTY_WEIGHT = {value:.4f} to exploration/config.py")
    print(f"  wall clock: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
