"""Minimum run-up for a full-height (running) jump, measured in the engine.

A running jump needs |x_vel| > 4.5 at take-off (mario.py: y_vel = JUMP_VEL-0.5).
For walk and for sprint, how many frames and how many PIXELS of ground does it
take to exceed 4.5 from a standstill, and what rise does the jump then give?
"""
import os, sys
os.environ["SDL_VIDEODRIVER"] = "dummy"; os.environ["SDL_AUDIODRIVER"] = "dummy"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from custom_mario_env import CustomMarioEnv
NOOP, WR, WRJ, RR, RRJ = 0, 1, 2, 3, 4
env = CustomMarioEnv()
def settle(x):
    env.reset(); env.game.state.mario.rect.x = x
    for _ in range(40): _o,_r,_t,_tr,info = env.step(NOOP)
    return info
for label, run, jump in (("walk", WR, WRJ), ("sprint", RR, RRJ)):
    print(f"--- {label} ---")
    info = settle(250); x0 = info["mario_rect"][0]; first = None
    for f in range(1, 60):
        _o,_r,_t,_tr,info = env.step(run)
        v = abs(info["x_vel"]); d = info["mario_rect"][0] - x0
        if first is None and v > 4.5:
            first = (f, d, v)
        if f in (5,10,15,20,25,30,35,40) or (first and first[0]==f):
            print(f"  frame {f:>2}: x_vel {v:>5.2f}  travelled {d:>4} px" + ("   <- crosses 4.5" if first and first[0]==f else ""))
    print(f"  => running-jump speed first reached after {first[0]} frames and {first[1]} px")
    # rise from exactly that take-off
    for runf in (first[0]-2, first[0], first[0]+2):
        info = settle(250)
        for _ in range(runf): _o,_r,_t,_tr,info = env.step(run)
        v = abs(info["x_vel"]); top0 = info["mario_rect"][1]; tops=[]
        for _ in range(60):
            _o,_r,_t,_tr,info = env.step(jump); tops.append(info["mario_rect"][1])
        print(f"  jump after {runf:>2} frames (x_vel {v:.2f}): rise {top0-min(tops)} px")
env.close()
