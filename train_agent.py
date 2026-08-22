import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
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


WATCHDOG_MILESTONES = []


class WatchdogCallback(BaseCallback):
    """
    Smart watchdog that:
    1. Pauses training once at each step-milestone in WATCHDOG_MILESTONES for
       manual review (see the sentinel-file note below for why this is safe
       to leave in across resumed/migrated runs).
    2. Detects blind AI (all-black observations)
    3. Detects loop collapse (reward drops >50% from peak over 50k steps)
    4. Detects instant-death loops (episode length consistently <30 frames)
    5. Logs warnings but does NOT spam — only alerts on real problems

    ─── BUGFIX NOTE (post-migration audit) ───
    The original version checked `if self.num_timesteps in [400000]:`. This
    had two real bugs:
      (a) With NUM_ENVS=8, num_timesteps advances in steps of 8 per callback
          call (399992 -> 400000 -> 400008, or similar depending on exact
          rollout boundaries), so the counter can jump straight past the
          exact value 400000 and the `in [...]` check would then NEVER fire.
      (b) Every time training is resumed with reset_num_timesteps=True, the
          counter restarts at 0 and will eventually hit 400000 again,
          re-triggering the "pause for review" every single time — this is
          almost certainly why training appeared to repeatedly stall at the
          same 400k mark.
    Fixed by (a) using >= instead of exact equality, and (b) writing a small
    sentinel flag file to CHECKPOINT_DIR the first time each milestone is
    reviewed, so it only ever pauses once per milestone, permanently,
    regardless of how many times the run is resumed or reset_num_timesteps
    is toggled. Delete the `.watchdog_reviewed_*.flag` files in checkpoints/
    if you ever want a milestone to pause again on purpose.
    """
    def __init__(self, verbose=0):
        super(WatchdogCallback, self).__init__(verbose)
        self.reward_history = deque(maxlen=100)  # Rolling window of episode rewards
        self.peak_reward = -float('inf')
        self.last_alert_step = 0
        self.short_episode_count = 0
        self.total_episodes = 0
        self._paused_milestones = set(
            m for m in WATCHDOG_MILESTONES if os.path.exists(self._sentinel_path(m))
        )

    def _sentinel_path(self, milestone):
        return os.path.join(CHECKPOINT_DIR, f".watchdog_reviewed_{milestone}.flag")

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

        # ─── MILESTONE PAUSES (one-time-ever per milestone, see docstring) ───
        for milestone in WATCHDOG_MILESTONES:
            if self.num_timesteps >= milestone and milestone not in self._paused_milestones:
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                open(self._sentinel_path(milestone), "w").close()
                self._paused_milestones.add(milestone)
                print(f"\n{'='*60}")
                print(f"WATCHDOG: Reached {self.num_timesteps:,} steps (milestone {milestone:,}).")
                print(f"Peak reward so far: {self.peak_reward:.1f}")
                print(f"Recent avg reward: {np.mean(self.reward_history) if self.reward_history else 0:.1f}")
                print(f"Total episodes: {self.total_episodes}")
                print(f"Short-death episodes (<30 frames): {self.short_episode_count}")
                print(f"{'='*60}")
                print("WATCHDOG: Pausing for manual review. This milestone will NOT pause again on resume.")
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
# ═══════════════════════════════════════════════════════════════════════
# RENAMED AGAIN: "mario_brain_v2_checkpoint" -> "mario_brain_v3_checkpoint".
#
# v1 = original pre-migration run (8 actions)
# v2 = migrated run (10 actions, warm-started from v1's CNN weights) — this
#      lineage is being ABANDONED. Video review after ~2M steps of
#      post-migration training showed the agent still pure-speedrunning
#      (zero mushrooms/fire flowers/stars collected, zero enemies killed,
#      just flying over everything) despite the corrected reward function.
#      Most likely explanation: the inherited CNN feature extractor was so
#      specialized on "detect gaps, jump over them, ignore everything else"
#      from v1's speedrun-only reward that it never developed the visual
#      features needed to even notice a mushroom or a nearby enemy — no
#      amount of new reward signal helps if the network was never trained
#      to see the thing the reward is about. Rather than fight that, this
#      is a clean restart.
# v3 = this run: a genuine fresh start, 10 actions, current reward function
#      (including the death-memory system), trained from scratch so the
#      CNN learns to see mushrooms/enemies/blocks from the very beginning
#      alongside everything else, instead of retrofitting them onto a
#      network that already decided none of that matters.
#
# The v3 prefix guarantees these files can never collide with the old v1 or
# v2 checkpoints already sitting in ./checkpoints/, regardless of what step
# numbers either lineage passes through.
# ═══════════════════════════════════════════════════════════════════════
CHECKPOINT_NAME = "mario_brain_v3_checkpoint"
FINAL_MODEL_PATH = "./mario_brain_v3_checkpoint"

# ═══════════════════════════════════════════════════════════════════════
# FRESH_START — set True to force a brand-new model from step 0, ignoring
# ANY checkpoint that might exist on disk (v1, v2, or even a partial v3
# run). This is the explicit, unambiguous way to start over, rather than
# relying on remembering to delete/move files out of the way. Once you've
# begun real v3 training and want to resume it normally later, set this
# back to False — the checkpoint-discovery logic below will then correctly
# find and continue the latest v3 checkpoint, and reset_num_timesteps is
# computed automatically based on whether a checkpoint was actually found,
# so you never need to hand-toggle that again either.
# ═══════════════════════════════════════════════════════════════════════
FRESH_START = True

TOTAL_TIMESTEPS = 6_000_000
NUM_ENVS = 8                          # Parallel environments

# ═══════════════════════════════════════════════════════════════════════
# EXACT CHECKPOINT MILESTONES — replaces the old CHECKPOINT_FREQ-based
# periodic saving entirely. You will get exactly one .zip per number below,
# named "mario_brain_v3_checkpoint_{N}_steps.zip", no more and no less,
# regardless of how many times training is stopped and resumed in between.
# See ExactMilestoneCheckpointCallback above for exactly why this is now
# reliable where the old approach wasn't.
# ═══════════════════════════════════════════════════════════════════════
CHECKPOINT_MILESTONES = [400_000 * i for i in range(1, 16)]  # 400k .. 6.0M

# ═══════════════════════════════════════════════════════════════════════
# ACTION SPACE: 10 actions (see custom_mario_env.py). This run starts fresh
# and trains directly on the 10-action space from step 0 — no migration
# involved, see the CHECKPOINT_NAME comment above for why.
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Look for the latest v3 checkpoint (skipped entirely if FRESH_START)
    latest_checkpoint = None
    if not FRESH_START:
        if os.path.exists(f"{CHECKPOINT_NAME}.zip"):
            latest_checkpoint = f"{CHECKPOINT_NAME}.zip"
        elif os.path.exists(CHECKPOINT_DIR):
            checkpoints = [f for f in os.listdir(CHECKPOINT_DIR)
                           if f.endswith(".zip") and f.startswith(CHECKPOINT_NAME)]
            if checkpoints:
                try:
                    checkpoints.sort(key=lambda x: int(x.split("_")[-2]))
                    latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
                except (ValueError, IndexError):
                    latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
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
    else:
        print("Starting fresh training...")
        model = PPO(
            "CnnPolicy", vec_env,
            learning_rate=2.5e-4,
            n_steps=2048,             # Longer rollouts = more stable learning
            batch_size=256,
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
    # False whenever resuming an existing v3 checkpoint, so the step
    # counter keeps climbing correctly instead of restarting at 0 — you no
    # longer need to remember to hand-toggle this between runs.
    reset_num_timesteps = latest_checkpoint is None

    try:
        # Train
        print("Starting training loop... You can stop this anytime with Ctrl+C and resume later.")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback_list,
            reset_num_timesteps=reset_num_timesteps,
        )

        # Save final model
        model.save(FINAL_MODEL_PATH)
        print(f"Training complete! Model saved to {FINAL_MODEL_PATH}.zip")
    except KeyboardInterrupt:
        print("\nTraining paused by user. Saving current brain state...")
        model.save(FINAL_MODEL_PATH)
        print("Saved successfully! Run this script again to resume.")
    finally:
        vec_env.close()
