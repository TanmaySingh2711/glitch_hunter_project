# Anomaly: acceleration state leaks from the ground into the air

**Status:** confirmed in the engine, reproducible, not fixed (the clone is
read-only). Found 2026-09-20 while testing whether the staircase well is a
soft-lock.

**Severity for QA:** it breaks two stated movement limits at once — the walk
speed cap and the run-up required for a full-height jump — and it makes a spot
that looks inescapable escapable. Any reachability model built from the
documented physics will be wrong wherever this is used.

## What happens

`mario.py` sets the horizontal acceleration and the speed cap in `walking()`,
which only runs while Mario is **on the ground**:

* pressing the direction **opposite** to his motion sets
  `x_accel = SMALL_TURNAROUND` (0.35), more than twice `WALK_ACCEL` (0.15);
* holding the sprint key sets `max_x_vel = MAX_RUN_SPEED` (800) instead of
  `MAX_WALK_SPEED` (6).

`jumping()` and `falling()` reset **neither**. They apply whatever `x_accel`
and `max_x_vel` the last grounded frame happened to leave behind:

```python
elif keys[tools.keybinding['right']]:
    if self.x_vel < self.max_x_vel:
        self.x_vel += self.x_accel      # 0.35/frame, capped at 800
```

So a turnaround immediately followed by a jump carries the turnaround
acceleration into the air with the walk cap lifted. Mid-air, Mario then
accelerates at 0.35/frame instead of 0.15 and is not held to 6.0.

## Measured consequences

Ordinary, documented movement (`docs/evidence/runup.py`, engine ground truth):

| | measured |
|---|---|
| acceleration from a standstill (walk **and** sprint) | 0.15 / frame |
| ground needed to reach the running-jump threshold (`x_vel` > 4.5) | **70 px** |
| jump rise below the threshold / at or above it | **166 px / 183 px** |

With the leak, in an 84 px well with only 54 px of run room — less than the
70 px the threshold needs — Mario reached **`x_vel` 7.65**, above the 6.0 walk
cap, and cleared a 172 px wall that a 166 px standing jump cannot.

## Reproduction

The scripts are kept beside this file in `docs/evidence/`; each runs headless
from a clone with no arguments.

`docs/evidence/well_trace.py` replays it deterministically (seeded, no policy
involved) and prints the frame-by-frame state. The escape:

```
f148  Left+Jump   x_vel -0.50  accel 0.35  max 6     # turnaround sets 0.35...
f156  fall        x_vel  0.00  accel 0.35            # ...and the jump keeps it
f172  fall        x_vel  4.20  accel 0.35            # accelerating in mid-air
f177  land        x_vel  5.95                        # lands above the 4.5 threshold
f178  jump        max 800                            # -> full 183 px rise
f217  stands on the wall top (y 326)                 # 172 px wall cleared
f244  x = 6105, clear of the well
```

Structured strategies that only use documented movement failed **873 of 873**
attempts from inside the well. Seeded random play escaped **5 times in 436**
(~1%), every time through this route.

## Why it matters beyond the well

1. **Reachability.** `JUMP_RISE_PX = 183` is applied from every standable
   anchor, including anchors with no run-up. That looked too generous, and a
   run-up-aware model (183 only with ≥70 px of ground) was built to tighten
   it. This anomaly **refutes** that model: speed can be built in the air, so
   the generous 183 is right and the tighter model would have excluded
   reachable space. The run-up model was withdrawn on this evidence.
2. **Policy behaviour.** The recurring timeouts of every trained policy at
   max_x 1,943 / 2,415 / 5,971 are all Mario pressed flush against a 172 px
   obstacle with no room to run. The documented escape is to back off ~220 px;
   this undocumented one needs no room at all. Neither is learned today.

## Not yet established

* Whether the real NES game permits an equivalent (this is a clone).
* Whether the leak is reachable through the agent's 10-action space as easily
  as through the raw key combinations used here — the trace used Left+Jump
  then Right, which **are** in the action space, so it is at least available.
* Whether any other level geometry depends on it.
