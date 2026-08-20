import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

def make_env(rank):
    """Returns a function that creates a single wrapped env instance."""
    def _init():
        from custom_mario_env import CustomMarioEnv
        from agent_logic import GlitchHunterWrapper
        from gymnasium.wrappers import (
            MaxAndSkipObservation, GrayscaleObservation,
            ResizeObservation, FrameStackObservation
        )
        env = CustomMarioEnv()
        env = GlitchHunterWrapper(env)
        env = MaxAndSkipObservation(env, skip=4)
        env = GrayscaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, 4)
        return env
    return _init

CHECKPOINT_DIR = "./checkpoints/"
CHECKPOINT_NAME = "mario_brain_checkpoint"
FINAL_MODEL_PATH = "./mario_brain_checkpoint"
TOTAL_TIMESTEPS = 2_500_000          # ~7.5 hours with 8 envs + frame skip
CHECKPOINT_FREQ = 50_000              # Save every 50k steps (~every 9 min based on Python clone 93 FPS)
NUM_ENVS = 8                          # Parallel environments

# Look for the latest checkpoint
latest_checkpoint = None
if os.path.exists(f"{CHECKPOINT_NAME}.zip"):
    latest_checkpoint = f"{CHECKPOINT_NAME}.zip"
elif os.path.exists(CHECKPOINT_DIR):
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".zip")]
    if checkpoints:
        # Sort by step number embedded in filename if possible
        try:
            checkpoints.sort(key=lambda x: int(x.split("_")[-2]))
            latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
        except (ValueError, IndexError):
            latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[-1])

if __name__ == "__main__":
    # Create 8 parallel environments
    vec_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])

    # Resume or fresh start
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=vec_env, device="cuda")
    else:
        print("Starting fresh training...")
        model = PPO(
            "CnnPolicy", vec_env,
            learning_rate=2.5e-4,
            n_steps=512,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            device="cuda",
        )

    # Auto-save callback
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix=CHECKPOINT_NAME,
    )

    try:
        # Train
        print("Starting training loop... You can stop this anytime with Ctrl+C and resume later.")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            reset_num_timesteps=False,    # Critical for resume — keeps the step counter
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
