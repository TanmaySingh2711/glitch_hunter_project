# Glitch Hunter Project

A web-based visualizer for a Super Mario Bros reinforcement learning agent. Uses Flask and WebSockets to stream the agent's gameplay frames and live logs to a browser interface in real-time.

The agent plays a Python/Pygame clone of Super Mario Bros 1-1, trained with PPO (Proximal Policy Optimization) via Stable Baselines3.

## Project Structure
- `app.py`: Flask backend with WebSocket event handlers for real-time streaming.
- `agent_logic.py`: RL reward shaping (`GlitchHunterWrapper`) — exploration rewards, stuck detection, powerup shaping, adaptive death memory — plus the dashboard's playback generator.
- `custom_mario_env.py`: Gymnasium-compatible wrapper around the Pygame Mario clone.
- `train_agent.py`: Training script using 8 parallel environments with PPO (CnnPolicy).
- `mario_clone/`: The full Python/Pygame Super Mario Bros 1-1 clone (graphics, sounds, game logic).
- `static/` & `templates/`: Frontend dashboard (HTML/CSS/JS).
- `IMPLEMENTATION.md`: Template for writing up the next change before building it (currently empty).

## Checkpoints

| Path | What it is |
|---|---|
| `mario_brain_checkpoint.zip` | **Master checkpoint.** Written on clean finish and on Ctrl+C. Training resumes from this if present, and the dashboard always plays this one. |
| `checkpoints/mario_brain_checkpoint_{N}_steps.zip` | Immutable milestone snapshots every 400k steps (400k … 6.0M). Never overwritten. |
| `checkpoints/.milestone_saved_*.flag` | Sentinels ensuring each milestone is saved exactly once, across any number of restarts. **Do not delete these** — they are 0 bytes, but without them a resumed run rewrites every milestone `.zip` with the current model, wiping the training history. |

To roll back to an earlier milestone, copy it over the master file:
```bash
cp checkpoints/mario_brain_checkpoint_5200000_steps.zip mario_brain_checkpoint.zip
```

## How to Run

`venv_gpu/` is not in this repo (see below) — you're creating it fresh, not
activating one that came with the clone.

1. **Create the virtual environment, using Python 3.12 specifically**
   (see `requirements.txt` for why — short version: PyTorch's CUDA wheels
   don't exist for Python 3.14 yet, and your system's default `python` may
   well be 3.14 even if 3.12 is also installed):
   ```bash
   py -3.12 -m venv venv_gpu       # Windows
   python3.12 -m venv venv_gpu     # Linux/Mac
   venv_gpu\Scripts\activate       # Windows
   source venv_gpu/bin/activate    # Linux/Mac
   ```

2. **Install Dependencies, in this exact order** (see `requirements.txt`
   for why plain `pip install -r requirements.txt` alone doesn't work here —
   short version: `stable-baselines3==2.9.0` requires `torch>=2.8`, but no
   CUDA 12.1 build of torch newer than 2.5.1 exists, so it must be installed
   with `--no-deps` or pip will silently swap in a non-CUDA torch build):
   ```bash
   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   pip install stable-baselines3==2.9.0 --no-deps
   pip install -r requirements.txt
   ```
   No NVIDIA GPU? Drop the `--index-url` line entirely (installs a CPU
   build) — the dashboard runs fine on CPU either way; only training
   benefits meaningfully from a GPU.

3. **Train the Agent** (optional — takes several hours, needs a GPU to be
   practical):
   ```bash
   python train_agent.py
   ```
   Training runs to `TOTAL_TIMESTEPS` (6M) *in total*, not 6M more per restart. Stop anytime with Ctrl+C and rerun to resume. Set `FRESH_START = True` to ignore all checkpoints and train from step 0.

4. **Run the Dashboard** — no GPU required. Loading and running the already-
   trained checkpoint to pick one action per frame is cheap; PyTorch falls
   back to CPU automatically if no CUDA device is found (see
   `agent_logic.py`, `device="auto"`), and the game itself is plain
   software-rendered pygame, not GPU-accelerated:
   ```bash
   python app.py
   ```

5. **View in Browser**:
   Open `http://localhost:5000` and click "Start Testing" to watch the agent play.
   If `mario_brain_checkpoint.zip` isn't present, this still runs, but with
   an **untrained, randomly-acting** policy — check the terminal for a
   `[WARNING] ... running an UNTRAINED policy` line.

## Upgrading dependencies later

The exact versions pinned in `requirements.txt` (and Python 3.12 itself)
were tested together and are known to work — nothing here needs touching
just because time has passed; PyPI doesn't delete old package versions, so
these pins keep installing indefinitely. There's a real trigger point for
when an upgrade becomes *necessary* rather than optional, though:

- **Python 3.12 reaches end-of-life on 2028-10-31** (official schedule:
  https://devguide.python.org/versions/). After that it stops receiving
  security patches. That's the real deadline, not "eventually."
- Sooner than that, only if a specific pinned package gets a CVE that
  matters for how this project is actually used.

When either happens, upgrade like this rather than just deleting the pins
and hoping:

1. Do it in a **new venv**, so the current working one is untouched until
   the new one is proven.
2. Re-resolve the same chain that's pinned today (check the current
   `torch>=` constraint declared by whatever `stable-baselines3` version
   you're moving to, and whether a CUDA build exists for the torch version
   it now requires — this exact mismatch already happened once, see the
   `requirements.txt` comments).
3. Run a smoke test before trusting it: load `mario_brain_checkpoint.zip`
   with the new `stable_baselines3`, `env.reset()`, step it a few times, and
   confirm `torch.cuda.is_available()` is still `True` if you have a GPU.
4. Only then update the pins in `requirements.txt` and commit — keep it as
   one deliberate, tested change, not a moving target.
