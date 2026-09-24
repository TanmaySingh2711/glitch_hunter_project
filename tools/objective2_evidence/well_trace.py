"""Replay the exact action sequence of well_escape.py random attempt 61 and
trace x_vel, x_accel, max_x_vel, state and position frame by frame, to see how
Mario built >4.5 speed and got out of an 84 px well."""
import os, sys
os.environ["SDL_VIDEODRIVER"] = "dummy"; os.environ["SDL_AUDIODRIVER"] = "dummy"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np
from custom_mario_env import CustomMarioEnv
NAMES = ["NOOP","WR","WRJ","RR","RRJ","J","WL","CROUCH","RL","LJ"]
rng = np.random.default_rng(20260920)
starts = (5915, 5940, 5955)
for i in range(62):                       # regenerate attempts 0..61 identically
    s = int(rng.choice(starts)); acts = []
    while len(acts) < 360:
        acts += [int(rng.integers(0, 10))] * int(rng.integers(4, 31))
    acts = acts[:360]
print(f"attempt 61: start x={s}")
env = CustomMarioEnv(); env.reset(); env.game.state.mario.rect.x = s
for _ in range(60): env.step(0)
m = env.game.state.mario
prev = None; top_min = 10**9
for f, a in enumerate(acts):
    _o,_r,term,_tr,info = env.step(a)
    r = info["mario_rect"]; top_min = min(top_min, r[1])
    row = (NAMES[a], m.state, round(m.x_vel,2), round(m.x_accel,3), m.max_x_vel, r[0], r[1])
    if prev is None or row[:2] != prev[:2] or abs(m.x_vel) > 4.4 or r[0] > 6000:
        print(f"  f{f:>3} {row[0]:<6} {row[1]:<10} x_vel {row[2]:>6} accel {row[3]:>6} max {row[4]:>4} rect ({row[5]},{row[6]})")
    prev = row
    if r[0] > 6100 or term: break
print(f"highest collider top reached: {top_min}")
