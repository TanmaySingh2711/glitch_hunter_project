"""Is the staircase well (x ~5904-5988) a soft-lock? Exhaustive engine search.

Every one of the healthy policy's 32/500 failures at max_x 5971 is a TIMEOUT,
never a death. Hypothesis: the well is 84 px wide with ~172 px walls on both
sides, and 84 - 30 (collider) = 54 px of run room is not enough to reach the
running-jump threshold |x_vel| > 4.5, so no jump can reach 172 px.

Escape = the collider's left edge passes the RIGHT wall (x > 6043), or its top
rises above the wall tops (the rim), in any direction. Structured strategies
first, then seeded random search. No reward wrapper, one frame per step.
"""
import os
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

from custom_mario_env import CustomMarioEnv

NOOP, WR, WRJ, RR, RRJ, J, WL, CROUCH, RL, LJ = range(10)
RIGHT_WALL = 6043
RIM_TOP = 358 - 40          # collider top above this = standing on a wall top


def place(env, x):
    env.reset()
    env.game.state.mario.rect.x = x
    info = None
    for _ in range(60):
        _o, _r, _t, _tr, info = env.step(NOOP)
    return info


def run(env, x, actions):
    info = place(env, x)
    x0, y0 = info["mario_rect"][0], info["mario_rect"][1]
    best_x, best_top, maxv = x0, y0, 0.0
    for a in actions:
        _o, _r, term, _tr, info = env.step(a)
        r = info["mario_rect"]
        best_x, best_top = max(best_x, r[0]), min(best_top, r[1])
        maxv = max(maxv, abs(info["x_vel"]))
        if term:
            break
    esc = best_x > RIGHT_WALL or best_top < RIM_TOP
    return esc, best_x, best_top, maxv, (x0, y0)


if __name__ == "__main__":
    env = CustomMarioEnv()
    starts = (5915, 5940, 5955)
    info = place(env, 5940)
    print(f"placed at 5940 -> collider {info['mario_rect']}, on_ground {info['on_ground']}")

    print("\n=== structured strategies ===")
    strategies = {}
    for back in (5, 10, 20, 30, 40, 54):
        for runf in range(4, 40, 3):
            for hold in (30, 45):
                strategies[f"left{back} sprint{runf} jump{hold}"] = (
                    [RL] * back + [NOOP] * 2 + [RR] * runf + [RRJ] * hold + [RR] * 40)
                strategies[f"left{back} walk{runf} jump{hold}"] = (
                    [WL] * back + [NOOP] * 2 + [WR] * runf + [WRJ] * hold + [WR] * 40)
    strategies["hold jump"] = [J] * 120
    strategies["sprint-right+jump"] = [RRJ] * 200
    strategies["left+jump"] = [LJ] * 200
    esc_any, best = [], (0, 10**9, 0.0)
    for s in starts:
        for name, acts in strategies.items():
            esc, bx, bt, mv, _ = run(env, s, acts)
            best = (max(best[0], bx), min(best[1], bt), max(best[2], mv))
            if esc:
                esc_any.append((s, name, bx, bt))
    print(f"  {len(strategies) * len(starts)} structured attempts; escapes: {len(esc_any)}")
    for e in esc_any[:10]:
        print(f"    ESCAPED from x={e[0]} via '{e[1]}' -> max_x {e[2]}, top {e[3]}")
    print(f"  best over all: furthest x {best[0]}, highest top y {best[1]} "
          f"(rim needs < {RIM_TOP}), peak |x_vel| {best[2]:.2f}")

    print("\n=== seeded random search ===")
    rng = np.random.default_rng(20260920)
    n_rand, esc_r, best_r = 1500, 0, (0, 10**9, 0.0)
    for i in range(n_rand):
        s = int(rng.choice(starts))
        acts = []
        while len(acts) < 360:          # random macro-actions held 4-30 frames
            acts += [int(rng.integers(0, 10))] * int(rng.integers(4, 31))
        esc, bx, bt, mv, _ = run(env, s, acts[:360])
        best_r = (max(best_r[0], bx), min(best_r[1], bt), max(best_r[2], mv))
        if esc:
            esc_r += 1
            print(f"    ESCAPED on random attempt {i} from x={s}: max_x {bx}, top {bt}")
            if esc_r >= 5:
                break
    print(f"  {i + 1} random attempts; escapes: {esc_r}")
    print(f"  best over all: furthest x {best_r[0]}, highest top y {best_r[1]}, "
          f"peak |x_vel| {best_r[2]:.2f}")
    env.close()
