import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import get_schedule_fn
import numpy as np
from collections import deque

class ExactMilestoneCheckpointCallback(BaseCallback):
    """
    Saves a checkpoint at EXACT cumulative step counts — not "every N calls
    since this callback object was created."

    ─── WHY THIS REPLACES SB3's BUILT-IN CheckpointCallback ───
    SB3's CheckpointCallback decides when to save by counting its OWN
    `self.n_calls` since the callback object was instantiated, not by
    looking at the model's persistent `self.num_timesteps`. Every time
    train_agent.py is re-run (crash, manual stop, laptop sleep, anything),
    a BRAND NEW CheckpointCallback is created with n_calls starting back at
    0 — while num_timesteps correctly keeps climbing from the loaded
    checkpoint. The result: the "next save" lands at (wherever training
    happened to be when you restarted) + 50,000, not at the next clean
    round number. This is exactly why checkpoints appeared at ugly numbers
    like 827344 and 1227344 instead of 800000 and 1200000 — nothing was
    lost or corrupted, the save points were just measured from the wrong
    reference point.

    This callback fixes it by comparing against `self.num_timesteps`
    directly (which persists correctly across restarts as long as
    reset_num_timesteps=False) against a fixed, explicit list of target
    step counts, with a one-shot-per-target sentinel file so a target can
    never be saved twice even if num_timesteps jumps past it in one batch
    step (NUM_ENVS=8 means num_timesteps advances 8 at a time, so it can
    step over an exact target value without ever equaling it — this uses
    >= specifically to handle that).

    Crucially: this callback NEVER returns False. It only saves a file and
    keeps going — no pauses, ever, matching "I don't want any pauses."
    """
    def __init__(self, targets, save_path, name_prefix, verbose=0):
        super().__init__(verbose)
        self.targets = sorted(targets)
        self.save_path = save_path
        self.name_prefix = name_prefix
        self._saved = set()

    def _sentinel_path(self, target):
        return os.path.join(self.save_path, f".milestone_saved_{self.name_prefix}_{target}.flag")

    def _init_callback(self) -> None:
        # ─── DO NOT DELETE THE .flag FILES IN checkpoints/ ───
        # They are 0 bytes and look like junk, but they are the ONLY record
        # of which milestones have already been saved. Deleting them makes
        # this set empty, so on the very next step every target at or below
        # the current step count is treated as unsaved and gets rewritten —
        # e.g. resuming at 6,000,000 with no flags overwrites all 15
        # milestone .zip files with copies of the current model, destroying
        # the entire training history in one step.
        os.makedirs(self.save_path, exist_ok=True)
        self._saved = set(t for t in self.targets if os.path.exists(self._sentinel_path(t)))

    def _on_step(self) -> bool:
        for target in self.targets:
            if self.num_timesteps >= target and target not in self._saved:
                path = os.path.join(self.save_path, f"{self.name_prefix}_{target}_steps.zip")
                self.model.save(path)
                open(self._sentinel_path(target), "w").close()
                self._saved.add(target)
                print(f"\n[CHECKPOINT] Saved exact milestone: {path} (at {self.num_timesteps:,} steps)")
        return True


class WatchdogCallback(BaseCallback):
    """
    Passive training monitor. It never stops training except in the one case
    where continuing is provably pointless (a blind agent).

    1. Detects a blind agent (all-black observations) and halts — this means
       the render pipeline broke, so every further step would train on
       garbage.
    2. Warns on possible policy collapse (average reward fell below half of
       the all-time peak), rate-limited to one alert per 50k steps.
    3. Tracks episode analytics (totals, short-death counts) for the alerts.

    ─── NOTE ON READING COLLAPSE ALERTS ───
    `peak_reward` is an all-time running maximum that never decays, so once
    the agent hits a lucky high streak, ANY normal regression below half of
    it keeps re-triggering this warning even when nothing is actually wrong.
    Treat the alert as "look at the numbers", not "something broke". The
    signature of a REAL collapse is different and much more specific:
    `approx_kl` spiking into double digits and `entropy_loss` crashing to
    near-zero within one or two iterations (see the target_kl comment below
    for the incident this was written from).
    """
    def __init__(self, verbose=0):
        super(WatchdogCallback, self).__init__(verbose)
        self.reward_history = deque(maxlen=100)  # Rolling window of episode rewards
        self.peak_reward = -float('inf')
        self.last_alert_step = 0
        self.short_episode_count = 0
        self.total_episodes = 0

    def _on_step(self) -> bool:
        # ─── LIVE PROGRESS INDICATOR ───
        if self.num_timesteps % 1000 == 0:
            print(f"Crunching frames... {self.num_timesteps:,} steps collected", end="\r", flush=True)

        # ─── BLIND CHECK (every 5000 steps) ───
        if self.num_timesteps % 5000 == 0:
            obs = self.locals.get("new_obs")
            if obs is not None:
                mean_pixel = np.mean(obs)
                if mean_pixel == 0.0:
                    print("CRITICAL: AI IS BLIND (Mean pixel = 0.0). Stopping training.")
                    return False

        # ─── EPISODE ANALYTICS ───
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_len = info["episode"]["l"]
                ep_reward = info["episode"]["r"]
                self.total_episodes += 1
                self.reward_history.append(ep_reward)
                
                # Track peak reward
                avg_reward = np.mean(self.reward_history)
                if avg_reward > self.peak_reward:
                    self.peak_reward = avg_reward

                # Count short-death episodes
                if ep_len < 30:
                    self.short_episode_count += 1
                
                # ─── COLLAPSE DETECTION ───
                # If we have enough data and reward has dropped >50% from peak
                if len(self.reward_history) >= 50 and self.peak_reward > 0:
                    if avg_reward < self.peak_reward * 0.5:
                        # Only alert once per 50k steps to avoid spam
                        if self.num_timesteps - self.last_alert_step > 50000:
                            self.last_alert_step = self.num_timesteps
                            print(f"\n[WARNING] WATCHDOG ALERT: Possible brain collapse detected!")
                            print(f"    Peak reward: {self.peak_reward:.1f}")
                            print(f"    Current avg: {avg_reward:.1f}")
                            print(f"    Step: {self.num_timesteps:,}")
                            print(f"    Continuing training (entropy should help recovery)...\n")
        
        return True


def make_env(rank):
    """Returns a function that creates a single wrapped env instance."""
    def _init():
        import time
        time.sleep(rank * 0.5)  # Stagger window creation to prevent Windows DWM race conditions
        from custom_mario_env import CustomMarioEnv
        from agent_logic import GlitchHunterWrapper
        from gymnasium.wrappers import (
            MaxAndSkipObservation, GrayscaleObservation,
            ResizeObservation, FrameStackObservation, TimeLimit
        )
        from stable_baselines3.common.monitor import Monitor
        env = CustomMarioEnv()
        env = GlitchHunterWrapper(env)
        env = MaxAndSkipObservation(env, skip=4)
        env = GrayscaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, 4)
        env = TimeLimit(env, max_episode_steps=4000)
        env = Monitor(env)
        return env
    return _init

CHECKPOINT_DIR = "./checkpoints/"

# Master checkpoint written on clean finish and on Ctrl+C. The checkpoint
# discovery below prefers this file over the numbered milestones, so it is
# always the resume point unless it is deleted or moved aside.
CHECKPOINT_NAME = "mario_brain_checkpoint"
FINAL_MODEL_PATH = f"./{CHECKPOINT_NAME}"

# ═══════════════════════════════════════════════════════════════════════
# FRESH_START — set True to force a brand-new model from step 0, ignoring
# ANY checkpoint on disk. This is the explicit, unambiguous way to start
# over, rather than relying on remembering to delete/move files out of the
# way. Set it back to False to resume normally; reset_num_timesteps is then
# computed automatically from whether a checkpoint was actually found, so
# that never needs hand-toggling either.
# ═══════════════════════════════════════════════════════════════════════
FRESH_START = False

TOTAL_TIMESTEPS = 6_000_000
NUM_ENVS = 8                          # Parallel environments

# ═══════════════════════════════════════════════════════════════════════
# EXACT CHECKPOINT MILESTONES — you get exactly one .zip per number below,
# named "mario_brain_checkpoint_{N}_steps.zip", no more and no less,
# regardless of how many times training is stopped and resumed in between.
# See ExactMilestoneCheckpointCallback above for why this is reliable where
# a frequency-based approach isn't.
# ═══════════════════════════════════════════════════════════════════════
CHECKPOINT_MILESTONES = [400_000 * i for i in range(1, 16)]  # 400k .. 6.0M


def _milestone_steps(filename):
    """Step count encoded in 'mario_brain_checkpoint_{N}_steps.zip', else -1.

    Returning -1 for unparseable names keeps them sorted below every real
    milestone, so a stray file can never be picked as "latest" — the old
    version fell back to lexicographic order on a parse failure, which
    silently ranks '800000' above '6000000'.
    """
    parts = filename[:-len(".zip")].split("_")
    if len(parts) >= 2 and parts[-1] == "steps":
        try:
            return int(parts[-2])
        except ValueError:
            return -1
    return -1


if __name__ == "__main__":
    # Look for the latest checkpoint (skipped entirely if FRESH_START).
    # The master file wins over numbered milestones when it exists.
    latest_checkpoint = None
    if not FRESH_START:
        if os.path.exists(f"{CHECKPOINT_NAME}.zip"):
            latest_checkpoint = f"{CHECKPOINT_NAME}.zip"
        elif os.path.exists(CHECKPOINT_DIR):
            candidates = [f for f in os.listdir(CHECKPOINT_DIR)
                          if f.endswith(".zip") and f.startswith(CHECKPOINT_NAME)
                          and _milestone_steps(f) >= 0]
            if candidates:
                newest = max(candidates, key=_milestone_steps)
                latest_checkpoint = os.path.join(CHECKPOINT_DIR, newest)
    else:
        print("FRESH_START is True — ignoring any existing checkpoints, training from step 0.")

    # Create 8 parallel environments
    vec_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])

    # Use CUDA if it's actually available; fall back to CPU instead of
    # crashing outright. Training on CPU is dramatically slower, so this
    # prints a loud warning rather than failing silently — if you see this
    # warning, something is wrong with your CUDA/driver setup and training
    # will be much slower than expected until it's fixed.
    import torch
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        print("=" * 60)
        print("WARNING: CUDA is not available. Training on CPU will be")
        print("MUCH slower. Check your PyTorch/CUDA installation if this")
        print("is unexpected.")
        print("=" * 60)

    # Resume or fresh start
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
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
        model.lr_schedule = get_schedule_fn(1.0e-4)
    else:
        print("Starting fresh training...")
        model = PPO(
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
            ent_coef=0.03,            # Raised slightly from 0.02: with 2 new
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
            tensorboard_log="./logs/"
        )

    # Auto-save at exact milestones (see ExactMilestoneCheckpointCallback
    # above for why this replaces SB3's built-in CheckpointCallback)
    checkpoint_callback = ExactMilestoneCheckpointCallback(
        targets=CHECKPOINT_MILESTONES,
        save_path=CHECKPOINT_DIR,
        name_prefix=CHECKPOINT_NAME,
    )

    # Combine callbacks
    callback_list = CallbackList([checkpoint_callback, WatchdogCallback()])

    # Computed automatically: True only when we're actually starting a
    # brand-new model (no checkpoint found, or FRESH_START forced it).
    # False whenever resuming an existing checkpoint, so the step counter
    # keeps climbing correctly instead of restarting at 0.
    reset_num_timesteps = latest_checkpoint is None

    # ─── BUGFIX: TOTAL_TIMESTEPS was silently a moving target ───
    # SB3's learn() adds the model's current num_timesteps on top of
    # whatever total_timesteps it's given whenever reset_num_timesteps=False
    # (every resume). Passing the raw TOTAL_TIMESTEPS=6_000_000 constant
    # unchanged meant each restart re-aimed for "6,000,000 MORE steps from
    # right now" instead of "6,000,000 steps total, ever" - e.g. resuming
    # from 5,200,000 actually targeted 11,200,000 internally, which is why
    # training sailed straight through the intended 6M mark without
    # stopping. Passing the remaining budget instead keeps the absolute
    # lifetime target fixed no matter how many times the script restarts.
    steps_to_run = TOTAL_TIMESTEPS if reset_num_timesteps else max(0, TOTAL_TIMESTEPS - model.num_timesteps)

    try:
        if steps_to_run == 0:
            # Budget already met. Return WITHOUT training and WITHOUT saving:
            # learn(0) still collects a rollout, and re-saving here would
            # overwrite the finished master checkpoint with a model that has
            # been stepped past the milestone for no reason.
            print(f"Already at TOTAL_TIMESTEPS ({TOTAL_TIMESTEPS:,}); current step: "
                  f"{model.num_timesteps:,}. Nothing to train — raise TOTAL_TIMESTEPS to continue.")
        else:
            print(f"Training to {TOTAL_TIMESTEPS:,} total steps "
                  f"({steps_to_run:,} remaining). Ctrl+C to stop and resume later.")
            model.learn(
                total_timesteps=steps_to_run,
                callback=callback_list,
                reset_num_timesteps=reset_num_timesteps,
            )
            model.save(FINAL_MODEL_PATH)
            print(f"Training complete! Model saved to {FINAL_MODEL_PATH}.zip")
    except KeyboardInterrupt:
        print("\nTraining paused by user. Saving current brain state...")
        model.save(FINAL_MODEL_PATH)
        print("Saved successfully! Run this script again to resume.")
    finally:
        vec_env.close()
