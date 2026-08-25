# Glitch Hunter Project

A trained AI plays a Super Mario Bros clone, and you watch it live in your browser.

---

## Setup (copy-paste these, in order)

**1. Get Python 3.12** from [python.org](https://www.python.org/downloads/) if you don't have it. (Not 3.13, not 3.14 — must be 3.12.)

**2. Open a terminal in this folder and create a virtual environment:**
```bash
py -3.12 -m venv venv_gpu
venv_gpu\Scripts\activate
```
*(Mac/Linux: `python3.12 -m venv venv_gpu` then `source venv_gpu/bin/activate`)*

**3. Install everything, in this exact order** (don't skip ahead or combine these):
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install stable-baselines3==2.9.0 --no-deps
pip install -r requirements.txt
```
No NVIDIA GPU? Use this instead for the first line — everything still works, just training would be slow (watching the AI play is unaffected either way):
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

**4. Run it:**
```bash
python app.py
```

**5. Open your browser** to **http://localhost:5000** and click **"START TESTING"**.

That's it. A game window will pop up and the same footage streams live to your browser.

---

## Using the dashboard

- **Start Testing** — pops up the game window and starts streaming it to the browser.
- **Stop Testing** — pauses the game (both the pop-up window and the browser view freeze on the same frame).
- **Reset Dashboard** / refreshing the page — closes the pop-up window.
- **BUG TRACKER** (red panel) — stays empty unless the game actually breaks a
  rule it's supposed to follow: Mario alive below the floor, moving at an
  impossible speed, or the score/coin counter running backwards. An empty
  panel is the normal, healthy result — it only speaks up for real problems.
- **LOG TERMINAL** (green panel) — every action the AI takes, with its reward.

If you ever see the AI acting completely random instead of playing well, check the terminal for a `[WARNING] ... running an UNTRAINED policy` line — it means `mario_brain_checkpoint.zip` (the trained brain) is missing from the folder.

---

## Project Structure

- `app.py` — the web server (Flask + WebSockets) that streams the game to your browser.
- `agent_logic.py` — how the AI is rewarded during training, plus the code that runs it live for the dashboard.
- `custom_mario_env.py` — connects the Mario game to the AI training library.
- `train_agent.py` — trains the AI from scratch (takes hours — most people will never need to run this).
- `mario_clone/` — the actual Super Mario Bros game (Python/Pygame). Not written by us — see Credits below.
- `static/` & `templates/` — the dashboard's look (HTML/CSS/JS).
- `mario_brain_checkpoint.zip` — the trained AI's "brain". Needed for the AI to play well; see above if it's missing.
- `checkpoints/` — snapshots from training, not needed just to watch the AI play.

---

## Credits & Licensing

The game in `mario_clone/` was written by **Justin Meister**
([Mario-Level-1](https://github.com/justinmeister/Mario-Level-1)) — not by us. It has
**no open-source license**, and its author states it is "intended for non-commercial
educational purposes." The artwork and sounds are Nintendo's property.

**So: learn from this, don't sell it.** Our own code (`app.py`, `agent_logic.py`,
`custom_mario_env.py`, `train_agent.py`, the dashboard) is MIT-licensed.

Full details in [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

---

## Technical Reference

Everything below this line is background/troubleshooting detail — not needed for normal setup or use.

### Why the install is 3 steps, not just `pip install -r requirements.txt`

`stable-baselines3==2.9.0` requires `torch>=2.8`, but no CUDA-enabled build of torch newer than 2.5.1 exists (checked directly against PyTorch's own package index). Installing stable-baselines3 normally would silently replace your GPU-enabled torch with a non-GPU one to satisfy that requirement — no error, just much slower training with no explanation why. `--no-deps` stops it from touching torch at all.

### Why Python 3.12 specifically

PyTorch's GPU-enabled (CUDA) builds don't exist yet for Python 3.14 (checked directly against `https://download.pytorch.org/whl/cu121/` — wheels exist for 3.10–3.13, not 3.14). Many systems default to whatever the newest installed Python is, so even if 3.12 is also on your machine, you may need to request it explicitly with `py -3.12` as shown above.

### GPU requirement

Only **training** meaningfully needs a GPU. **Watching the AI play does not** — loading the trained brain and picking one action per frame is cheap, and it automatically runs on CPU if no GPU is found. The game itself is plain 2D rendering, not GPU-accelerated.

### Checkpoints

| Path | What it is |
|---|---|
| `mario_brain_checkpoint.zip` | **The trained brain.** Training resumes from this if present, and the dashboard always plays this one. |
| `checkpoints/mario_brain_checkpoint_{N}_steps.zip` | Snapshots taken every 400k training steps. Never overwritten. |
| `checkpoints/.milestone_saved_*.flag` | Bookkeeping for the snapshots above. **Do not delete** — they're 0 bytes but deleting them causes a resumed training run to overwrite every snapshot with the current model, destroying the training history. |

To roll back to an earlier snapshot:
```bash
cp checkpoints/mario_brain_checkpoint_5200000_steps.zip mario_brain_checkpoint.zip
```

### Upgrading dependencies later

The pinned versions above were tested together and don't need touching just because time has passed — PyPI doesn't delete old package versions, so these keep installing indefinitely. The one real deadline: **Python 3.12 loses security support on 2028-10-31** ([official schedule](https://devguide.python.org/versions/)). Before then, upgrade in a *new* venv first, re-check for the same kind of torch-version conflict described above, and smoke-test (load the checkpoint, step the environment a few times) before trusting it.
