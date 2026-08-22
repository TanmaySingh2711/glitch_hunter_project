# Glitch Hunter — Implementation Report #3
### The checkpoint-numbering bug (permanently fixed), why the migration is being abandoned, and a full solutions catalog for death/loophole/speedrunning problems

---

## PART A — The checkpoint files: good news first

I checked. **Nothing was overwritten this time, and nothing was lost.** What actually
happened:

```
checkpoints/mario_brain_v2_checkpoint_400000_steps.zip     (from before)
checkpoints/mario_brain_v2_checkpoint_827344_steps.zip     (new)
checkpoints/mario_brain_v2_checkpoint_1227344_steps.zip    (new)
checkpoints/mario_brain_v2_checkpoint_1627344_steps.zip    (new)
checkpoints/mario_brain_v2_checkpoint_2027344_steps.zip    (new)
```

Every file has a unique name — the v2 rename from last round worked exactly as intended.
What went wrong is just that the numbers are ugly (`827344` instead of `800000`) instead
of colliding. Here's precisely why, and why it's now permanently fixed regardless of
which lineage you train.

### A.1 — Root cause: SB3's `CheckpointCallback` counts from the wrong reference point

Stable-Baselines3's built-in `CheckpointCallback` decides when to save by counting its
own internal `self.n_calls` — a counter that starts at 0 the moment the callback
**object** is created. It does **not** look at the model's actual cumulative
`num_timesteps`.

Every time `train_agent.py` gets re-run — whether that's a manual stop, a crash, a
laptop going to sleep, anything — a **brand new** `CheckpointCallback` object gets
created, and its `n_calls` restarts at 0. Meanwhile `num_timesteps` correctly keeps
climbing from wherever the loaded checkpoint left off. So the "next save" always lands at
*(wherever you happened to restart) + 50,000*, not at the next clean round number. That's
exactly the `827344`, `1227344`, etc. pattern — each one is ~50,000 (or a multiple of it)
past some earlier restart point, not past a round number.

**This is not data corruption or loss — it's just a measurement-reference bug.** Nothing
to recover, nothing was destroyed.

### A.2 — Permanent fix: a custom callback that only looks at absolute step count

I replaced SB3's `CheckpointCallback` entirely with a new
`ExactMilestoneCheckpointCallback` in `train_agent.py`. It:

- Takes an explicit list of target step counts (your exact list — see Part D).
- Compares each target against `self.num_timesteps` directly (which persists correctly
  across restarts, since it belongs to the model, not the callback).
- Uses `>=` instead of exact equality, so it can't be skipped over even though
  `num_timesteps` jumps by 8 at a time (`NUM_ENVS=8`).
- Writes a one-shot sentinel flag file per milestone, so a target can never be saved
  twice, no matter how many times the script restarts in between.
- **Never returns `False`.** It only ever saves a file and continues — no pauses, ever.

I verified this directly: I simulated saving past the 400,000 mark, then **simulated a
full script restart** (a brand new callback object, exactly like re-running
`train_agent.py`), and confirmed the next save still landed exactly on 800,000 — not on
some restart-shifted number:

```
[CHECKPOINT] Saved exact milestone: test_v3_400000_steps.zip (at 400,006 steps)
[CHECKPOINT] Saved exact milestone: test_v3_800000_steps.zip (at 800,003 steps)
EXACT MILESTONE TEST PASSED — no duplicate saves, exact round numbers, survives restart
```

This is a structural fix, not a one-off patch — it will hold regardless of how many times
training gets interrupted going forward.

---

## PART B — Why the migration is being abandoned (and why that's the right call)

I pulled frames from your new video (`Screen Recording 2026-08-22 111953.mp4`) the same
way as before. At ~2,000,000 steps of post-migration training under the corrected reward
function:

- Score sits at `000100`, coins at `x00`, for the entire clip.
- Mario flies straight over a row of four "?" blocks and two Goombas without touching
  either — same exact pattern as the pre-fix video from last round.
- He clears pipes by pure altitude, never engaging with anything at ground level.

**The corrected reward function alone wasn't enough to undo this**, even after ~1.6M
steps of training under it. Here's the most likely reason, and why I think restarting is
the right call rather than continuing to patch the migrated model:

When you migrate a checkpoint, only the network's weights transfer — but *which visual
features the CNN even learned to recognize* is itself a product of what the old reward
function made useful to notice. The original run's reward was almost entirely about
"is there a gap ahead, when should I jump" — so that's what the convolutional layers
became good at detecting. They were never pushed to develop strong internal
representations for "there's a mushroom bouncing nearby" or "there's a Goomba two tiles to
my right," because under the old reward, noticing those things never used to matter.
Warm-starting from those weights hands the new reward function a brain that's
functionally near-blind to the exact things it now needs to reward — no amount of correct
reward signal fixes that quickly, because the policy can't act on a feature its vision
never learned to extract in the first place. It's not stubbornness, it's closer to the
network genuinely not having built the right eyes yet, and building genuinely new visual
features on top of an already-converged CNN is much slower than learning them from
scratch alongside everything else.

**Fresh start it is.** `train_agent.py` now has `FRESH_START = True` and a `v3` checkpoint
naming lineage, completely separate from the abandoned `v2` migration and the original
`v1` pre-migration run:

- `v1` = original 8-action run (your very first training)
- `v2` = the migration attempt — **abandoned**, kept only as a reference
- `v3` = this run — genuine fresh start, 10 actions, current reward function (including
  the new death-memory system in Part C), the CNN learns to see mushrooms/enemies/blocks
  from step 0 alongside everything else instead of retrofitting them on top of a network
  that already decided none of it mattered.

`FRESH_START = True` means the very first run of `train_agent.py` will ignore any
existing checkpoint entirely and build a brand-new model, no matter what's sitting in
`checkpoints/` or at the top level. Once real `v3` training has begun and you want to
resume it normally, set `FRESH_START = False` — checkpoint discovery and
`reset_num_timesteps` are both computed automatically from there, so you won't need to
hand-toggle anything else between resumes.

Your `v1` and `v2` checkpoints are untouched and still on disk if you ever want to go
back and look at them — nothing about this deletes them.

---

## PART C — Adaptive death memory: implementing your "10 deaths in a row" idea

You proposed: *"if Mario dies 10 times in a row he should change the way he is playing."*
I researched this properly and it maps directly onto real reinforcement-learning
concepts — this section covers the full landscape of options, then what I actually built.

### C.1 — The full solution space for "stop repeating the same fatal mistake"

| Approach | How it works | Feasible here? |
|---|---|---|
| **Let vanilla PPO handle it** | Policy gradient methods naturally lower the probability of state-action pairs that lead to low reward, over enough training. Repeated deaths at the same spot *should* eventually get trained away on their own. | Already happening in the background, but "eventually, given enough steps" isn't the same as "guaranteed within 10 tries" — too slow/unreliable for your requirement on its own. |
| **Adaptive entropy scheduling** | Temporarily raise PPO's `ent_coef` (exploration strength) when the agent seems stuck in a bad pattern, then lower it again once it escapes. | Possible in principle, but `ent_coef` is a training hyperparameter set at model-construction time — not something the environment/reward wrapper can reach into and adjust per-location. Would require a custom callback that rebuilds parts of the optimizer mid-training; fragile and not attempted here. |
| **Intrinsic curiosity / novelty bonuses (ICM, RND)** | A second neural network learns to predict what should happen next; wherever its predictions are most wrong (novel/surprising states) gets an extra reward, pulling the agent toward unexplored behavior generally. | The "proper" academic answer to "explore more, don't repeat failures," but it's a genuinely new subsystem (a second trained network) — real implementation effort, listed as future work, not done here. |
| **Count-based / pseudo-count exploration bonuses** | Reward states inversely proportional to how many times they've been visited. | Similar weight to RND, same "future work" bucket — meaningfully more infrastructure than the other options. |
| **Curriculum learning / automatic domain randomization** | Vary starting conditions (e.g. start episodes closer to a hard obstacle sometimes) so the agent gets more practice reps at exactly the hard part. | Would require modifying the game's level-loading/spawn logic, which is a deeper change to `mario_clone` itself. Possible, but bigger scope — listed as future work. |
| **Location+cause-keyed adaptive shaping (what I built)** | Track *where* and *why* Mario keeps dying; when the same (place, cause) kills him 10 times in a row, temporarily strengthen the specific rewards relevant to surviving that exact situation, right there, for a stretch of episodes. | **Implemented now** — see C.2. This is the closest match to your literal proposal, doesn't touch training hyperparameters or need a second network, and is fully environment-side so it works with the existing PPO setup unchanged. |

I went with the last option because it's the most direct implementation of what you
actually asked for, is achievable without new infrastructure, and is verifiable (see
C.3).

### C.2 — What was built: per-cause, per-location adaptive shaping

Every death is now tagged with an exact **cause** — this required a small, targeted
change to the underlying game code (`mario_clone/data/states/level1.py`), tagging
`mario.death_cause` at each of the four places the game can actually kill Mario:

| Cause | Where it's detected |
|---|---|
| `'goomba'` | Mario walks into a Goomba while small (non-stomp contact) |
| `'koopa'` | Mario walks into a Koopa while small |
| `'koopa_shell'` | Mario walks into a moving/kicked shell |
| `'pit'` | Mario's y-position goes below the bottom of the screen |
| `'timeout'` | The level clock reaches 0 |

(These are the **only four ways Mario can die** in this clone — I read through every
death-triggering code path in `level1.py` to confirm there's no fifth cause like crushing
or off-screen-left death that was missed.)

`custom_mario_env.py` exposes this as `info['death_cause']`. `agent_logic.py`'s wrapper
tracks a `(location_bucket, cause)` streak counter that **persists across episodes**
(deliberately not reset each episode, since the whole point is noticing a *pattern* across
many separate lives). When the exact same cause kills Mario at the exact same ~200px
patch of the level **10 times in a row**, that patch becomes an active "danger zone" for
the next 15 episodes, during which:

- **Pit deaths** → the momentum/sprint reward and the "clean running jump" bonus are
  **doubled** specifically in that zone, and the stuck-penalty threshold is relaxed there
  (100 frames of patience instead of 40) — giving room to deliberately back up and line up
  a proper running jump instead of being time-pressured into repeating the same rushed
  attempt.
- **Goomba / Koopa / shell deaths** → a small bonus is added for attempting a jump while
  airborne in that zone — a nudge toward trying a stomp instead of walking straight in
  again.
- **Timeout deaths** → the time penalty is halved in that zone, since repeatedly running
  out of clock there means the agent needs more breathing room, not more pressure.

The boost decays automatically — one episode closer to expiring every time an episode
ends, regardless of outcome — so it never lingers forever if the agent has clearly moved
past that problem.

### C.3 — Verified working

I ran the environment with a deterministic "always run right, never jump" policy so it
would reliably die to the same Goomba at the same spot every episode:

```
ep 0: death_cause=goomba x_pos=843  danger_zones={}
ep 1: death_cause=goomba x_pos=843  danger_zones={}
...
ep 8: death_cause=goomba x_pos=843  danger_zones={}
ep 9: death_cause=goomba x_pos=843  danger_zones={4: {'cause': 'goomba', 'episodes_left': 15}}
ep 10: ...                          danger_zones={4: {'cause': 'goomba', 'episodes_left': 14}}
```

Triggers exactly on the 10th consecutive identical death, and counts down correctly
afterward. This is a genuinely new capability, not just documentation — you can watch it
happen live in the training console (it doesn't print anything currently; let me know if
you'd like a print statement added when a zone activates/expires, for visibility during
the 8-window training sessions).

---

## PART D — Exact checkpoint milestones, as you specified

`train_agent.py`'s `CHECKPOINT_MILESTONES` is now exactly:

```
400000, 800000, 1200000, 1600000, 2000000, 2400000, 2800000, 3200000,
3600000, 4000000, 4400000, 4800000, 5200000, 5600000, 6000000
```

— generated as `[400_000 * i for i in range(1, 16)]`, matching your list of 15 targets
exactly, ending at `TOTAL_TIMESTEPS = 6,000,000`. You will get **exactly one `.zip` per
number**, named `mario_brain_v3_checkpoint_{N}_steps.zip`, no more and no less, regardless
of how many times the 8 windows get stopped and restarted along the way — that's the
whole point of Part A's fix.

---

## PART E — Everything else, addressed directly

- **"I don't want any pauses"** — `WATCHDOG_MILESTONES` is already an empty list in the
  version you uploaded (this was correctly removed already, likely by Antigravity
  following the last round's instructions), and I confirmed the new
  `ExactMilestoneCheckpointCallback` never pauses either — it only saves and returns
  `True`. The one thing left that *can* still stop training is the existing "AI IS BLIND"
  check (all-black observation frames) — that's a crash/bug detector, not a review pause,
  and I left it in deliberately since the alternative is silently burning hours of GPU
  time training on a broken screen. Tell me if you want that removed too.
- **File/folder cleanup** — I re-audited the whole zip. Last round's cleanup instructions
  were followed correctly: no leftover duplicate `_FIXED` folder, no `scratch/`, no
  `mario_brain_checkpoint_migrated.zip`. The only thing sitting around now is the
  ~100MB of `v2` migration checkpoints, which are no longer going to be trained further
  but aren't hurting anything — your call whether to archive/delete them once you're
  confident `v3` is working out. `.git` is still ~91MB from old history (see last round's
  report, Part F step 5, for the exact optional purge commands — still your call, still
  not run automatically since it's a destructive, remote-affecting operation).
- **"Explore the game, don't speedrun"** — unchanged from before: the tile-exploration
  reward, generalized milestones, block-reveal bonus, and potential-based powerup pull are
  all still in place and still the dominant reward sources over a full playthrough; the
  time penalty is still light (`-0.02/step`). The fresh start gives these signals a real
  chance to shape the CNN from the beginning instead of fighting an already-specialized
  network.

---

## What I could not do / verify from here

- I can't run your actual 6,000,000-step training session — that has to happen on your
  laptops. Everything above is verified via smoke tests and code review, not a live full
  training run.
- I can't confirm how the adaptive death-memory system behaves under a *real* trained
  policy (as opposed to the scripted "always walk right" test) — a real policy dies in
  more varied ways, so the streak-trigger will fire less mechanically than in the test,
  which is expected and fine, just worth knowing the verification was structural, not
  behavioral-at-scale.
- Curiosity-based exploration (RND/ICM) and curriculum/spawn-randomization, listed in
  C.1, are real, legitimate next steps if the location-based adaptive shaping alone turns
  out not to be enough — but they're a meaningfully bigger implementation effort than
  anything in this round, and I didn't want to build speculative infrastructure you didn't
  ask for without checking first.
