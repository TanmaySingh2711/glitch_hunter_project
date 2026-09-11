"""Train the Glitch Hunter agent.

    python train_agent.py --safety-cap-timesteps 6020000   # the controlled +20k validation
    python train_agent.py --unrestricted                   # the QA campaign, after review

This file is the entry point: it decides WHICH run happens - the phase, the
checkpoint to resume, the coverage that belongs to it, whether the launch is
allowed at all - and wires the pieces in training/ together around one PPO
model. Everything it prints also goes to logs/train.log once training starts.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from common.logging_setup import configure_logging
from exploration import config as xconfig
from exploration import coverage as coverage_mod
from exploration import level_completion as lc
from exploration.coverage import SpatialCoverage

# Re-exported: the callbacks and helpers moved to training/ but remain part of
# this module's surface (tests and tools address them as train_agent.X).
from training.callbacks import (
    CoverageStatsCallback,
    ExactMilestoneCheckpointCallback,
    Level1CompletionCallback,
    LifecycleStatsCallback,
    StagnationCallback,
    ValueWarmupCallback,
    WatchdogCallback,
)
from training.checkpoints import checkpoint_timesteps, newest_milestone
from training.value_head import reset_value_head

__all__ = [
    'CoverageStatsCallback', 'ExactMilestoneCheckpointCallback', 'Level1CompletionCallback',
    'LifecycleStatsCallback', 'StagnationCallback', 'ValueWarmupCallback', 'WatchdogCallback',
    'build_model', 'campaign_coverage_path', 'check_launch_gate',
    'checkpoint_timesteps', 'main', 'make_env', 'prepare_qa_coverage', 'reset_value_head',
    'steps_to_run',
]

log = logging.getLogger(__name__)


def make_env(rank: int, shm_names: dict[str, str] | None = None,
             reward_mode: str | None = None) -> Callable[[], gym.Env[Any, Any]]:
    """Returns a function that creates a single wrapped env instance.

    `shm_names` is a dict of multiprocessing.shared_memory block NAMES, not
    objects. That distinction is the whole reason this signature changed: a
    SharedMemory object cannot be cloudpickled into a SubprocVecEnv worker,
    but its name is just a string, and the worker re-attaches by name on the
    other side. All 8 workers therefore write into ONE bitmap.

    Without that, each worker would hold a private copy of "explored", every
    one of them would be paid full novelty for the same opening stretch of
    the level, and coverage would be counted eight times over forever.
    """
    # Resolved HERE, in the parent, and closed over as plain ints - so the
    # worker receives numbers rather than having to re-derive the mode.
    # LEGACY: TimeLimit is a BACKSTOP strictly above the engine timer's
    # measured cap (the engine clock is authoritative - see exploration/
    # config.py EPISODE LENGTH). It used to be 4000 in both modes, which was
    # dead code: the engine always timed out first, at 2,451 agent steps.
    # QA: none. A step count alone may not end a QA episode (the respawn
    # rule, config QA_TIMEOUT_ENDS_EPISODE); the lifecycle's evidence-based
    # safety reset is what ends an unproductive one.
    mode = reward_mode or xconfig.REWARD_MODE
    max_steps = (None if mode == "qa_exploration"
                 else xconfig.LEGACY_EPISODE_MAX_STEPS)
    skip = xconfig.SUBSTEPS_PER_AGENT_STEP

    def _init() -> gym.Env[Any, Any]:
        import time
        time.sleep(rank * 0.5)  # Stagger window creation to prevent Windows DWM race conditions
        from gymnasium.wrappers import TimeLimit
        from stable_baselines3.common.monitor import Monitor

        from agent_logic import GlitchHunterWrapper
        from custom_mario_env import CustomMarioEnv, wrap_observation
        env: gym.Env[Any, Any] = wrap_observation(GlitchHunterWrapper(
            CustomMarioEnv(), reward_mode=reward_mode, shm_names=shm_names), skip=skip)
        if max_steps is not None:
            env = TimeLimit(env, max_episode_steps=max_steps)
        return Monitor(env)
    return _init

# ═══════════════════════════════════════════════════════════════════════
# PHASE SELECTION
#
# QA_PHASE follows exploration/config.py REWARD_MODE, so there is exactly ONE
# switch for the whole retrofit and no way for the reward and the checkpoint
# naming to disagree with each other.
#
# ─── THE 6M MASTER IS NEVER WRITTEN IN QA MODE ───
# mario_brain_checkpoint.zip and checkpoints/ belong to the completion-phase
# brain. QA training reads the master once, to seed itself, and from then on
# writes only to checkpoints_qa/ and glitch_hunter_qa.zip. Backing the
# retrofit out is therefore always possible: flip REWARD_MODE back and the
# original files are exactly where they were, untouched, byte for byte.
# ═══════════════════════════════════════════════════════════════════════
QA_PHASE = (xconfig.REWARD_MODE == "qa_exploration")

LEGACY_CHECKPOINT_DIR = "./checkpoints/"
LEGACY_CHECKPOINT_NAME = "mario_brain_checkpoint"

if QA_PHASE:
    CHECKPOINT_DIR = f"./{xconfig.CHECKPOINT_DIR_QA}/"
    CHECKPOINT_NAME = xconfig.CHECKPOINT_NAME_QA
else:
    CHECKPOINT_DIR = LEGACY_CHECKPOINT_DIR
    # Master checkpoint written on clean finish and on Ctrl+C. The checkpoint
    # discovery below prefers this file over the numbered milestones, so it is
    # always the resume point unless it is deleted or moved aside.
    CHECKPOINT_NAME = LEGACY_CHECKPOINT_NAME

FINAL_MODEL_PATH = f"./{CHECKPOINT_NAME}"
COVERAGE_FINAL_PATH = f"./{CHECKPOINT_NAME}_coverage.npz"
# The run's own log, beside the TensorBoard curves. Opened only once every
# launch gate has passed, so a refused launch writes nothing at all.
TRAIN_LOG_PATH = os.path.join("logs", "train.log")

# ═══════════════════════════════════════════════════════════════════════
# FRESH_START — set True to force a brand-new model from step 0, ignoring
# ANY checkpoint on disk. This is the explicit, unambiguous way to start
# over, rather than relying on remembering to delete/move files out of the
# way. Set it back to False to resume normally; reset_num_timesteps is then
# computed automatically from whether a checkpoint was actually found, so
# that never needs hand-toggling either.
# ═══════════════════════════════════════════════════════════════════════
FRESH_START = False

# ═══════════════════════════════════════════════════════════════════════
# RESET_VALUE_HEAD — applies only on the FIRST QA run (the one seeded from
# the completion brain), and only in QA mode. See training/value_head.py
# for the full reasoning; in short, the 6M critic is confidently wrong about
# a reward it has never seen, and its error would backpropagate through the
# shared CNN into the locomotion features the retrofit is trying to keep.
#
# Set False only if you want to observe that failure deliberately.
# ═══════════════════════════════════════════════════════════════════════
RESET_VALUE_HEAD = True

# ═══════════════════════════════════════════════════════════════════════
# WHEN TRAINING ENDS
#
# LEGACY: a step budget, unchanged. The completion phase finished at exactly
# 6,000,000, so its remaining budget is 0 and it refuses to train again.
#
# QA (Phase 4F): there is NO step target. QA training succeeds when, and only
# when, every verified testable pixel of Level 1 is covered:
#
#     covered_testable_pixels == 4,013,723      (exploration/level_completion.py)
#
# Level1CompletionCallback stops training the moment that holds and writes
# the immutable completion snapshot. Timesteps keep numbering checkpoints and
# logs and nothing else. (This replaced a fixed 12,000,000 lifetime target,
# which said nothing about whether the level had actually been explored.)
#
# QA_SAFETY_CAP_TIMESTEPS is a SAFETY CAP ONLY - an absolute global step at
# which a run is cut off whether or not the level is done (e.g. to bound a
# short validation run). Reaching it is never Level-1 completion and is
# reported as a stop with pixels still remaining. None = no cap.
# `--safety-cap-timesteps N` sets it for one launch without editing this file.
#
# THE LAUNCH GATE. A QA launch with no cap at all is an unrestricted long
# campaign, and one of those only starts with `--unrestricted`: before it,
# the controlled validation (python train_agent.py --safety-cap-timesteps
# 6020000, i.e. 6.0M -> 6.02M) has to be run and its coverage growth, reward
# balance, lifecycle behaviour, resume integrity and completion retention
# reviewed. The gate cannot check that the review happened - only a person
# can - but it makes sure an unbounded run is never started by default.
# ═══════════════════════════════════════════════════════════════════════
TOTAL_TIMESTEPS_LEGACY = 6_000_000
QA_SAFETY_CAP_TIMESTEPS: int | None = None
# SB3's learn() must be given SOME integer; with no cap this is it. It is not
# a target and no message ever presents it as one.
_SB3_OPEN_ENDED = 2 ** 62
NUM_ENVS = 8                          # Parallel environments

# ═══════════════════════════════════════════════════════════════════════
# EXACT CHECKPOINT MILESTONES — you get exactly one .zip per number below,
# named "mario_brain_checkpoint_{N}_steps.zip", no more and no less,
# regardless of how many times training is stopped and resumed in between.
# See ExactMilestoneCheckpointCallback (training/callbacks.py) for why this
# is reliable where a frequency-based approach isn't.
# ═══════════════════════════════════════════════════════════════════════
LEGACY_CHECKPOINT_MILESTONES = [400_000 * i for i in range(1, 16)]  # 400k .. 6.0M
# QA: every multiple of 400k GLOBAL timesteps - 6.4M, 6.8M, 7.2M, ... - with
# no last one, since QA has no step target. Absolute lifetime counts, in
# checkpoints_qa/ under the QA name, so they never collide with the
# completion phase's 400k-6.0M files.
QA_CHECKPOINT_EVERY = 400_000
CHECKPOINT_MILESTONES = None if QA_PHASE else LEGACY_CHECKPOINT_MILESTONES

# The Level-1 completion snapshot: its own directory beside the periodic
# checkpoints, written once, read-only (exploration/level_completion.py). The
# remaining-pixel audit is a live file, rewritten late in a campaign.
LEVEL1_SNAPSHOT_DIR = os.path.join(CHECKPOINT_DIR, lc.SNAPSHOT_DIRNAME)
REMAINING_AUDIT_PATH = os.path.join(CHECKPOINT_DIR, "level1_remaining_audit.json")
REMAINING_MAP_PATH = os.path.join(CHECKPOINT_DIR, "level1_remaining_map.png")
# Append-only history of coverage growth, one JSON line per report, across
# every run of the campaign.
COVERAGE_TRAIL_PATH = os.path.join(CHECKPOINT_DIR, "coverage_audit_trail.jsonl")

# ═══════════════════════════════════════════════════════════════════════
# OPTIONAL TENSORBOARD LOGGING
# Stable-Baselines3 hard-fails ("tensorboard is not installed") the moment
# tensorboard_log is set to a path without the tensorboard package present.
# tensorboard is a large dependency that is only useful for inspecting
# training curves, so requirements.txt does NOT force everyone who just
# wants to watch the agent play to install it. Detecting it here means
# training works either way: you get the curves if you have it, and a plain
# console run if you don't, instead of a crash on startup.
#   To enable:  pip install tensorboard
#   Then view:  tensorboard --logdir ./logs/
# ═══════════════════════════════════════════════════════════════════════
try:
    import tensorboard  # noqa: F401  (imported for its presence, not its API)
    TENSORBOARD_LOG: str | None = "./logs/"
except ImportError:
    TENSORBOARD_LOG = None

VALIDATION_CAP_TIMESTEPS = 6_020_000        # the controlled +20k run from the 6M seed


def steps_to_run(qa: bool, num_timesteps: int, reset_num_timesteps: bool,
                 safety_cap: int | None = None) -> int:
    """What to hand SB3's learn().

    LEGACY: the remaining lifetime budget - the formula this project already
    used, unchanged (SB3 adds num_timesteps on top when resuming, so passing
    the raw total would re-aim for that many MORE steps each restart).

    QA: open-ended; Level1CompletionCallback ends it. With a safety cap, the
    steps left until the cap - which is a cut-off, never a success.
    """
    if not qa:
        return (TOTAL_TIMESTEPS_LEGACY if reset_num_timesteps
                else max(0, TOTAL_TIMESTEPS_LEGACY - num_timesteps))
    if safety_cap is None:
        return _SB3_OPEN_ENDED
    return max(0, int(safety_cap) - int(num_timesteps))


def campaign_coverage_path(latest_checkpoint: str | None, seeded_from_legacy: bool) -> str:
    """The coverage file that belongs to the checkpoint being resumed."""
    if latest_checkpoint is None:
        raise SystemExit(
            "QA mode needs a checkpoint to resume and the coverage that belongs "
            "to it; there is none (FRESH_START?). Refusing to start from an "
            "empty map.")
    if seeded_from_legacy:
        cov_path = xconfig.BOOTSTRAP_COVERAGE
        if not os.path.exists(cov_path):
            raise SystemExit(
                f"{cov_path} not found.\n"
                f"The first QA run starts from the bootstrap map. Run "
                f"`python tools/bootstrap_coverage.py` first - starting "
                f"from an empty map would pay the agent full novelty "
                f"for re-walking everywhere it already knows.")
        return cov_path
    paired = f"{os.path.splitext(latest_checkpoint)[0]}_coverage.npz"
    if os.path.exists(paired):
        return paired
    if os.path.exists(COVERAGE_FINAL_PATH):
        return COVERAGE_FINAL_PATH
    raise SystemExit(
        f"Resuming {latest_checkpoint} but no matching "
        f"coverage file was found (looked for {paired} and "
        f"{COVERAGE_FINAL_PATH}).\n"
        f"Refusing to continue with an empty map - that would "
        f"silently re-reward everywhere already explored.")


def prepare_qa_coverage(coverage: SpatialCoverage, latest_checkpoint: str | None,
                        seeded_from_legacy: bool) -> tuple[str, int]:
    """Loads and VERIFIES the campaign's cumulative coverage into `coverage`
    before any worker starts. Returns (path, checkpoint timesteps).

    Anything incompatible or corrupted - wrong mask fingerprint, wrong
    denominator, a bitmap that does not re-count to its own totals, a file
    paired with a different checkpoint - stops the launch with the reason.
    It is never reset, never repaired, and nothing is written.
    """
    cov_path = campaign_coverage_path(latest_checkpoint, seeded_from_legacy)
    assert latest_checkpoint is not None        # campaign_coverage_path exits otherwise
    try:
        expected = checkpoint_timesteps(latest_checkpoint)
        coverage.load_verified(cov_path, expected_timesteps=expected)
    except (coverage_mod.CoverageFormatMismatch, coverage_mod.CoverageCorrupted,
            coverage_mod.CoverageCheckpointMismatch, FileNotFoundError,
            zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise SystemExit(
            f"[COVERAGE] Refusing to resume: {cov_path} is not a usable "
            f"campaign coverage state for {latest_checkpoint}.\n"
            f"  {type(exc).__name__}: {exc}\n"
            f"Nothing was trained and no file was written. Restore the "
            f"matching coverage file rather than starting from an empty map.") from exc
    return cov_path, expected


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train the Glitch Hunter agent.")
    ap.add_argument("--safety-cap-timesteps", type=int, metavar="N", default=None,
                    help="QA: stop at global step N whether or not Level 1 is complete "
                         f"(a cut-off, never success). The validation run is "
                         f"--safety-cap-timesteps {VALIDATION_CAP_TIMESTEPS}.")
    ap.add_argument("--unrestricted", action="store_true",
                    help="QA: explicitly approve a campaign with no safety cap. Only "
                         "after the controlled validation run has been reviewed.")
    return ap.parse_args(list(argv))


def check_launch_gate(qa: bool, safety_cap: int | None, unrestricted: bool) -> None:
    """No QA campaign runs unbounded unless someone said so explicitly."""
    if not qa or safety_cap is not None or unrestricted:
        return
    raise SystemExit(
        "[LAUNCH] Refusing an unrestricted QA campaign: no safety cap was given "
        "and it was not explicitly approved.\n"
        "  1. Run the controlled validation from a clean state and review it:\n"
        f"       python train_agent.py --safety-cap-timesteps {VALIDATION_CAP_TIMESTEPS}\n"
        "  2. Only after reviewing its coverage growth, reward balance, lifecycle\n"
        "     behaviour, resume integrity and completion retention:\n"
        "       python train_agent.py --unrestricted\n"
        "Nothing was trained and no file was written.")


def report_level_complete(coverage: SpatialCoverage) -> None:
    """Launch-time: the campaign is already done. Says so, points at the
    proof if there is one, and trains nothing."""
    rule = "=" * 68
    log.info(rule)
    log.info("LEVEL 1 IS ALREADY COMPLETE: %s / %s testable px covered (100%%).",
             f"{coverage.covered_testable():,}", f"{coverage.testable_total:,}")
    found = lc.committed_snapshots(LEVEL1_SNAPSHOT_DIR)
    if found:
        path, meta = found[0]
        log.info("Completed at global step %s; proof: %s",
                 f"{meta['global_timestep']:,}", path)
    else:
        log.info("(no completion snapshot found in %s)", LEVEL1_SNAPSHOT_DIR)
    final = lc.final_level1_brain(LEVEL1_SNAPSHOT_DIR)
    if final:
        log.info("Final Level-1 brain (VERIFIED): %s", final[1]['final_level1_brain']['path'])
    else:
        log.info("Not yet the final Level-1 brain: run `python tools/verify_level1.py`.")
    log.info("Not training further, and not re-saving anything. No other level is started.")
    log.info(rule)


def build_model(latest_checkpoint: str | None, vec_env: VecEnv, device: str) -> PPO:
    """Loads or constructs the PPO model.

    Shared by both phases unchanged - every hyperparameter note below was
    written from a real incident in this project's training history and none
    of it should drift between the two phases.
    """
    # Resume or fresh start
    if latest_checkpoint:
        log.info("Resuming from checkpoint: %s", latest_checkpoint)
        model = PPO.load(latest_checkpoint, env=vec_env, device=device)
        # PPO.load() restores hyperparameters from the checkpoint itself, so
        # anything set in the fresh-start branch below never reaches a
        # resumed run unless it's applied here too.
        #
        # ─── target_kl 0.03 -> 0.05 (post-5.2M tuning) ───
        # Around 4.6M-5.2M steps, "Early stopping at step 0" was firing on
        # most iterations with the triggering per-minibatch KL sitting at
        # 0.05-0.07 - well above the 1.5*target_kl=0.045 break threshold,
        # but nowhere near the 11.37 that caused the real collapse at
        # ~3.22M. With batch_size=256, each rollout (16384 samples) splits
        # into 64 tiny minibatches, and a single noisy one can trip the
        # ceiling even when the overall update is fine. 0.05 keeps a large
        # (~220x) safety margin below the value that actually caused
        # instability, while no longer treating this routine minibatch
        # noise as a violation.
        model.target_kl = 0.05
        # ─── batch_size 256 -> 512 (post-5.2M tuning) ───
        # Larger minibatches average the KL estimate over more samples,
        # directly reducing the per-minibatch noise described above -
        # addresses the actual noise source rather than just raising the
        # ceiling to tolerate it.
        model.batch_size = 512
        # Same trap for learning_rate: PPO.load() also restores an internal
        # lr_schedule closure built from the checkpoint's saved rate.
        # Setting model.learning_rate alone does NOT change what the
        # optimizer actually uses each update - lr_schedule has to be
        # rebuilt explicitly too, or this silently has zero effect.
        # Left at 1e-4 (not lowered further to 1e-5): the early-stopping
        # pattern above was present from the very first iteration after the
        # 3.2M resume, not something that emerged as the policy "got more
        # advanced" - so it's minibatch noise, not a step-size problem. A
        # 10x cut this late (~800k steps left of the 6M budget) risked
        # stalling real progress in the final stretch for a problem it
        # wasn't actually fixing.
        model.learning_rate = 1.0e-4
        model.lr_schedule = FloatSchedule(1.0e-4)
        return model

    log.info("Starting fresh training...")
    return PPO(
        "CnnPolicy", vec_env,
        learning_rate=1.0e-4,      # Lowered from 2.5e-4 for the same reason
                                   # noted in the resume branch above: too-
                                   # large per-epoch updates were hitting
                                   # target_kl on every iteration.
        n_steps=2048,             # Longer rollouts = more stable learning
        batch_size=512,           # Raised from 256 for the same reason
                                   # noted in the resume branch above:
                                   # larger minibatches average out KL
                                   # noise instead of letting a single
                                   # noisy one trip target_kl early.
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=xconfig.ENT_COEF_BASE,  # 0.03, raised from 0.02: with 2 new
                                   # actions (Left+Sprint, Left+Jump) and
                                   # reshaped rewards, the policy needs a
                                   # bit more entropy to actually sample
                                   # and discover the new momentum-based
                                   # play patterns instead of collapsing
                                   # onto the old 8-action habits.
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.05,           # Hard stop: if a rollout's mean KL
                                   # divergence blows past this, SB3 cuts
                                   # the remaining epochs for that update
                                   # instead of continuing to push the
                                   # policy further. clip_range alone
                                   # doesn't guarantee this - a bad batch
                                   # can still overwhelm the clipping
                                   # (this is what caused the collapse at
                                   # ~3.22M steps: approx_kl hit 11.37,
                                   # clip_fraction 0.834, and the policy
                                   # collapsed to near-zero entropy).
                                   # Raised from 0.03 to 0.05 after
                                   # ~4.6M-5.2M steps showed routine
                                   # minibatch noise (0.05-0.07) tripping
                                   # early stopping almost every
                                   # iteration - still a ~220x margin
                                   # below the value that actually caused
                                   # the real collapse.
        verbose=1,
        device=device,
        tensorboard_log=TENSORBOARD_LOG,
    )


def save_pair(model: PPO, coverage: SpatialCoverage | None) -> None:
    """Saves the model and, in QA mode, its matching coverage map.

    Order matters for the same reason it does at milestones: the model is
    written first, then the coverage. A coverage file is only ever useful
    beside the model it was recorded with, and load() enforces that pairing.
    """
    model.save(FINAL_MODEL_PATH)
    if coverage is not None:
        coverage.save(COVERAGE_FINAL_PATH, model_timesteps=model.num_timesteps)
        log.info("[COVERAGE] Saved %s: %s unique world px",
                 COVERAGE_FINAL_PATH, f"{coverage.total_unique():,}")


# ══════════════════════════════════════════════════════════════════════════
# main() AND ITS STAGES
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Resume:
    """Where this run starts from."""
    checkpoint: str | None
    seeded_from_legacy: bool


def find_resume_point() -> Resume:
    """The checkpoint to resume (skipped entirely if FRESH_START). The master
    file wins over numbered milestones when it exists."""
    if FRESH_START:
        log.info("FRESH_START is True — ignoring any existing checkpoints, training from step 0.")
        return Resume(None, False)
    latest: str | None = None
    if os.path.exists(f"{CHECKPOINT_NAME}.zip"):
        latest = f"{CHECKPOINT_NAME}.zip"
    else:
        latest = newest_milestone(CHECKPOINT_DIR, CHECKPOINT_NAME)
    # ─── QA SEEDING ───
    # First QA run only: there is no QA checkpoint yet, so the completion
    # phase's master is READ to seed the weights. It is never written
    # back - from here on every save goes to checkpoints_qa/ and
    # glitch_hunter_qa.zip. num_timesteps rides along inside the zip, so
    # training continues from 6,000,001 rather than appearing to restart.
    if latest is None and QA_PHASE:
        if not os.path.exists(f"{LEGACY_CHECKPOINT_NAME}.zip"):
            raise SystemExit(
                f"QA phase needs a brain to continue from, but neither a "
                f"QA checkpoint nor {LEGACY_CHECKPOINT_NAME}.zip was "
                f"found.\n"
                f"Restore the 6M master (backup_6M/ holds a copy) before "
                f"starting the QA phase.")
        return Resume(f"{LEGACY_CHECKPOINT_NAME}.zip", True)
    return Resume(latest, False)


def log_loaded_coverage(coverage: SpatialCoverage, cov_path: str, paired_at: int) -> None:
    """The verified starting state of the campaign, before any worker."""
    covered = coverage.covered_testable()
    total = int(coverage.testable_total or 0)
    log.info("[COVERAGE] Loaded %s (verified: mask fingerprint %s..., paired at step %s)",
             cov_path, xconfig.TESTABLE_FINGERPRINT[:12], f"{paired_at:,}")
    rows = [("world raster", xconfig.WORLD_RASTER_PX, "informational only"),
            ("testable", total, f"method {xconfig.ADOPTED_METHOD}"),
            ("covered", covered, lc.pct_text(covered, total)),
            ("remaining", coverage.remaining() or 0, "")]
    nb = coverage.noncoverage_breakdown()
    rows += [("noncoverage", nb['total'], "NOT coverage"),
             ("  expected", nb['expected_total'], "normal engine behaviour, not glitches"),
             ("  model gap", nb['model_gap_total'], "reachability model, not the game"),
             ("  anomalous", nb['anomalous_total'], "-> glitch system")]
    for label, value, note in rows:
        log.info("           %-15s %12s   %s", label, f"{value:,}", note)


def limit_worker_blas_threads() -> None:
    """One OpenBLAS thread for every env worker spawned after this call.

    An env worker never does linear algebra - stepping the engine and the
    reward is elementwise numpy - but numpy's OpenBLAS still reserves a
    buffer per core the moment it loads. Measured on this machine (16 logical
    cores, one QA env, headless): 512 MB of private memory committed before
    the first step, 30 MB with one thread; step time unchanged. x NUM_ENVS
    that is ~3.8 GB for nothing. Set in the PARENT's environment because
    SubprocVecEnv spawns workers that import numpy before any of this
    project's code runs in them; the parent's own BLAS - already loaded, and
    the one PPO's updates actually use - is unaffected. An explicit value
    in the environment wins.
    """
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def select_device() -> str:
    """CUDA when it is actually available; CPU with a loud warning otherwise.

    Training on CPU is dramatically slower, so this warns rather than failing
    silently — if you see it, something is wrong with your CUDA/driver setup
    and training will be much slower than expected until it's fixed.
    """
    import torch
    if torch.cuda.is_available():
        return "cuda"
    log.warning("CUDA is not available. Training on CPU will be MUCH slower. "
                "Check your PyTorch/CUDA installation if this is unexpected.")
    return "cpu"


def build_callbacks(model: PPO, coverage: SpatialCoverage | None,
                    session_start: dict[str, int] | None
                    ) -> tuple[CallbackList, Level1CompletionCallback | None]:
    """Every callback of the run, and the completion stop (QA only)."""
    # Auto-save at exact milestones (see ExactMilestoneCheckpointCallback
    # for why this replaces SB3's built-in CheckpointCallback).
    # QA: every 400k global steps, open-ended; legacy: its fixed list.
    if QA_PHASE:
        checkpoint_callback = ExactMilestoneCheckpointCallback(
            every=QA_CHECKPOINT_EVERY, save_path=CHECKPOINT_DIR,
            name_prefix=CHECKPOINT_NAME, coverage=coverage)
    else:
        checkpoint_callback = ExactMilestoneCheckpointCallback(
            targets=CHECKPOINT_MILESTONES, save_path=CHECKPOINT_DIR,
            name_prefix=CHECKPOINT_NAME, coverage=coverage)

    callbacks: list[BaseCallback] = [checkpoint_callback, WatchdogCallback()]
    completion = None
    if QA_PHASE:
        assert coverage is not None
        assert session_start is not None
        completion = Level1CompletionCallback(
            coverage, snapshot_dir=LEVEL1_SNAPSHOT_DIR,
            name_prefix=CHECKPOINT_NAME, session_start=session_start)
        callbacks += [
            completion,
            ValueWarmupCallback(warmup_until=model.num_timesteps + xconfig.VF_WARMUP_STEPS,
                                warmup_lr=xconfig.VF_WARMUP_LR,
                                normal_lr=xconfig.NORMAL_LR),
            CoverageStatsCallback(coverage,
                                  session_start_covered=session_start['covered_testable_px'],
                                  audit_path=REMAINING_AUDIT_PATH, map_path=REMAINING_MAP_PATH,
                                  trail_path=COVERAGE_TRAIL_PATH, checkpoints=checkpoint_callback),
            LifecycleStatsCallback(),
            StagnationCallback(coverage),
        ]
    return CallbackList(callbacks), completion


def log_banner(model: PPO, resume: Resume, coverage: SpatialCoverage | None,
               safety_cap: int | None, n_steps: int) -> None:
    rule = "=" * 68
    log.info(rule)
    log.info("PHASE            : %s", 'QA EXPLORATION' if QA_PHASE else 'LEGACY COMPLETION')
    log.info("REWARD_MODE      : %s", xconfig.REWARD_MODE)
    log.info("resuming from    : %s%s", resume.checkpoint or 'scratch',
             '  (SEED - read only, never written)' if resume.seeded_from_legacy else '')
    log.info("current step     : %s", f"{model.num_timesteps:,}")
    if QA_PHASE and coverage is not None:
        covered = coverage.covered_testable()
        total = int(coverage.testable_total or 0)
        log.info("success          : Level 1 covered = %s testable px (NOT a step count)",
                 f"{total:,}")
        log.info("covered          : %s  (%s), %s remaining", f"{covered:,}",
                 lc.pct_text(covered, total), f"{coverage.remaining() or 0:,}")
        log.info("safety cap       : %s",
                 'none (approved with --unrestricted)' if safety_cap is None
                 else f'{safety_cap:,} (a cut-off, never completion)')
        log.info("coverage trail   : %s (appended every 10,000 steps)", COVERAGE_TRAIL_PATH)
        nxt = (model.num_timesteps // QA_CHECKPOINT_EVERY + 1) * QA_CHECKPOINT_EVERY
        log.info("checkpoints      : every %s global steps (next %s) -> %s",
                 f"{QA_CHECKPOINT_EVERY:,}", f"{nxt:,}", CHECKPOINT_DIR)
        log.info("on completion    : immutable snapshot -> %s", LEVEL1_SNAPSHOT_DIR)
    else:
        log.info("lifetime target  : %s", f"{TOTAL_TIMESTEPS_LEGACY:,}")
        log.info("remaining        : %s", f"{n_steps:,}")
    log.info("writes model to  : %s.zip and %s", FINAL_MODEL_PATH, CHECKPOINT_DIR)
    if QA_PHASE:
        log.info("master preserved : %s.zip and %s are NOT written in this phase",
                 LEGACY_CHECKPOINT_NAME, LEGACY_CHECKPOINT_DIR)
    log.info("run log          : %s", TRAIN_LOG_PATH)
    log.info(rule)


def run_training(model: PPO, callback_list: CallbackList, completion: Level1CompletionCallback | None,
                 coverage: SpatialCoverage | None, safety_cap: int | None, n_steps: int,
                 reset_num_timesteps: bool) -> None:
    """learn(), then save and say how it ended. Ctrl+C saves and exits cleanly."""
    try:
        if n_steps == 0:
            # Nothing to run. Return WITHOUT training and WITHOUT saving:
            # learn(0) still collects a rollout, and re-saving here would
            # overwrite the master with a model stepped past its point for
            # no reason.
            if QA_PHASE and coverage is not None and safety_cap is not None:
                log.info("Safety cap %s already reached at step %s. Level 1 is NOT complete "
                         "(%s px remain). Raise the cap (--safety-cap-timesteps) to continue.",
                         f"{safety_cap:,}", f"{model.num_timesteps:,}",
                         f"{coverage.remaining() or 0:,}")
            else:
                log.info("Already at TOTAL_TIMESTEPS_LEGACY (%s); current step: %s. Nothing "
                         "to train — raise TOTAL_TIMESTEPS_LEGACY to continue.",
                         f"{TOTAL_TIMESTEPS_LEGACY:,}", f"{model.num_timesteps:,}")
            return
        if QA_PHASE:
            log.info("Training until Level 1 is fully covered. Ctrl+C to stop and resume later.")
        else:
            log.info("Training to %s total steps (%s remaining). Ctrl+C to stop and resume later.",
                     f"{TOTAL_TIMESTEPS_LEGACY:,}", f"{n_steps:,}")
        model.learn(total_timesteps=n_steps, callback=callback_list,
                    reset_num_timesteps=reset_num_timesteps)
        save_pair(model, coverage)
        log_outcome(model, completion, coverage, safety_cap)
    except KeyboardInterrupt:
        log.info("Training paused by user. Saving current brain state...")
        save_pair(model, coverage)
        log.info("Saved successfully! Run this script again to resume.")


def log_outcome(model: PPO, completion: Level1CompletionCallback | None,
                coverage: SpatialCoverage | None, safety_cap: int | None) -> None:
    if not QA_PHASE or coverage is None:
        log.info("Training complete! Model saved to %s.zip", FINAL_MODEL_PATH)
        return
    remaining = f"{coverage.remaining() or 0:,}"
    if completion is not None and completion.completed:
        log.info("Level 1 complete. Master %s.zip holds the same policy as the snapshot; "
                 "relaunching will not train.", FINAL_MODEL_PATH)
        log.info("It is NOT yet the final Level-1 brain: verify it first with "
                 "`python tools/verify_level1.py`. No other level is started.")
    elif safety_cap is not None and model.num_timesteps >= safety_cap:
        log.info("Stopped at the SAFETY CAP (%s). Level 1 is NOT complete: %s px remain (%s).",
                 f"{safety_cap:,}", remaining,
                 lc.pct_text(coverage.covered_testable(), int(coverage.testable_total or 0)))
    else:
        log.info("Training stopped at step %s before Level 1 was complete (%s px remain). "
                 "Saved; run again to resume.", f"{model.num_timesteps:,}", remaining)


def main(argv: Sequence[str] = ()) -> None:
    configure_logging()
    if TENSORBOARD_LOG is None:
        log.info("[INFO] tensorboard not installed - training will run without logging "
                 "curves. Install it with `pip install tensorboard` if you want them.")
    args = parse_args(argv)
    # The command line wins over the constant; either is a cut-off only.
    safety_cap = (args.safety_cap_timesteps if args.safety_cap_timesteps is not None
                  else QA_SAFETY_CAP_TIMESTEPS)
    resume = find_resume_point()

    # ═══════════════════════════════════════════════════════════════════
    # SHARED COVERAGE (QA phase only)
    #
    # Created BEFORE SubprocVecEnv, because the workers attach to it by name
    # as they start. The parent keeps its own handle for two reasons: it is
    # the authoritative reader for every reported metric (total_unique() is
    # recomputed from the bitmap, so it is exact even though the per-worker
    # novelty claims race), and it is the only process allowed to unlink.
    # On Windows the blocks are refcounted and vanish when the last handle
    # closes, so the parent's handle is also what keeps them alive.
    # ═══════════════════════════════════════════════════════════════════
    coverage: SpatialCoverage | None = None
    shm_names: dict[str, str] | None = None
    if QA_PHASE:
        reachable = coverage_mod.load_testable()
        if reachable is None:
            raise SystemExit(
                "exploration_data/reachable_mask.npz not found.\n"
                "Run `python tools/build_reachability.py` first - without it "
                "there is no frontier and no coverage denominator.")
        coverage, shm_names = coverage_mod.create_shared(testable_mask=reachable)
        log.info("[COVERAGE] Shared grid allocated: %s", shm_names['visited'])

    vec_env: VecEnv | None = None
    try:
        # ═══════════════════════════════════════════════════════════════
        # LOAD THE COVERAGE THAT BELONGS TO THIS CHECKPOINT - before any
        # worker starts (Phase 4F)
        #
        # The model and its coverage are a MATCHED PAIR. Pairing coverage with
        # the wrong brain corrupts every downstream number silently: the agent
        # gets re-paid for ground it already explored, or starved of reward
        # for ground it has not, and either way it looks like ordinary
        # training rather than a bug. load_verified also checks the mask
        # fingerprint, the denominator and the bitmap's own counts, and
        # prepare_qa_coverage turns any refusal into a clean exit.
        #
        # Loading first also means a finished campaign never spawns a worker.
        # ═══════════════════════════════════════════════════════════════
        session_start: dict[str, int] | None = None
        if QA_PHASE and coverage is not None:
            cov_path, expected_timesteps = prepare_qa_coverage(
                coverage, resume.checkpoint, resume.seeded_from_legacy)
            session_start = {'global_timestep': expected_timesteps,
                             'covered_testable_px': coverage.covered_testable()}
            log_loaded_coverage(coverage, cov_path, expected_timesteps)
            if lc.is_level_complete(session_start['covered_testable_px'],
                                    int(coverage.testable_total or 0)):
                report_level_complete(coverage)
                return
            # After the coverage checks (which write nothing), before any
            # worker, model or file.
            check_launch_gate(QA_PHASE, safety_cap, args.unrestricted)

        # Every gate has passed: from here on the run is real, so it gets a
        # log file that outlives this terminal.
        configure_logging(log_file=TRAIN_LOG_PATH)

        limit_worker_blas_threads()
        vec_env = SubprocVecEnv([make_env(i, shm_names=shm_names,
                                          reward_mode=xconfig.REWARD_MODE)
                                 for i in range(NUM_ENVS)])
        model = build_model(resume.checkpoint, vec_env, select_device())
        if session_start is not None and model.num_timesteps != session_start['global_timestep']:
            raise SystemExit(
                f"{resume.checkpoint} loaded at {model.num_timesteps:,} steps but "
                f"its zip record said {session_start['global_timestep']:,}; "
                f"refusing to pair it with coverage verified for the latter.")

        # ─── FIRST QA RUN ONLY ───
        # Done exactly once, when the QA phase is seeded from the completion
        # brain, because that is the only moment the critic is fitted to the
        # wrong objective. Resuming a QA checkpoint must NOT reset it again -
        # by then it is fitted to the reward actually in use, and throwing it
        # away would discard real progress.
        if QA_PHASE and resume.seeded_from_legacy and RESET_VALUE_HEAD:
            reset_value_head(model)

        callback_list, completion = build_callbacks(model, coverage, session_start)

        # Computed automatically: True only when we're actually starting a
        # brand-new model (no checkpoint found, or FRESH_START forced it).
        # False whenever resuming an existing checkpoint, so the step counter
        # keeps climbing correctly instead of restarting at 0. Seeding the QA
        # phase from the 6M master counts as resuming, which is what makes
        # training continue at 6,000,001 rather than at 1.
        reset_num_timesteps = resume.checkpoint is None

        # See steps_to_run(): legacy passes its remaining budget (SB3 adds
        # num_timesteps on top when resuming); QA is open-ended and ends on
        # Level-1 coverage, or at the safety cap if one is set.
        n_steps = steps_to_run(QA_PHASE, model.num_timesteps, reset_num_timesteps,
                               safety_cap)
        log_banner(model, resume, coverage, safety_cap, n_steps)
        run_training(model, callback_list, completion, coverage, safety_cap, n_steps,
                     reset_num_timesteps)
    finally:
        if vec_env is not None:
            vec_env.close()
        if coverage is not None:
            # Workers close their own handles as they shut down; the parent is
            # the only one allowed to destroy the blocks, and only after every
            # worker is gone.
            coverage.close()
            coverage.unlink()


if __name__ == "__main__":
    main(sys.argv[1:])
