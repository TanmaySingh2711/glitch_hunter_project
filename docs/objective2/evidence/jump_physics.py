"""Engine ground truth for the 172 px obstacles (tall pipes, staircase top).

Every recurring TIMEOUT stuck point of the 6M, 6.03M and 6.4M policies sits
exactly 32 px left of an obstacle 172 px tall (x=1975, 2447, 6003). The claim
under test:

  * a STANDING jump (y_vel -10) cannot clear 172 px;
  * a RUNNING jump (|x_vel| > 4.5 -> y_vel -10.5) can, if jump is held;
  * so Mario pressed flush against the wall is stuck unless he backs off.

Uses the bare engine, one engine frame per step, no reward wrapper.
"""
import os
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from custom_mario_env import CustomMarioEnv

NOOP, WALK_R, WALK_R_JUMP, RUN_R, RUN_R_JUMP, JUMP, WALK_L = 0, 1, 2, 3, 4, 5, 6


def fresh(env, x):
    env.reset()
    m = env.game.state.mario
    m.rect.x = x
    for _ in range(40):                      # settle onto the ground, camera follows
        _o, _r, term, _t, info = env.step(NOOP)
    return info


def rect(info):
    return info["mario_rect"]


def measure_jump(env, x, runup_frames, run_action, jump_action, hold=60):
    """Run for `runup_frames`, then press jump and HOLD it. Returns
    (x_vel at take-off, rise in px measured on the collider TOP)."""
    info = fresh(env, x)
    for _ in range(runup_frames):
        _o, _r, _te, _tr, info = env.step(run_action)
    vx = info["x_vel"]
    top0 = rect(info)[1]
    tops = []
    for _ in range(hold):
        _o, _r, term, _tr, info = env.step(jump_action)
        tops.append(rect(info)[1])
        if term:
            break
    return vx, top0 - min(tops)


def try_clear(env, start_x, pipe_right, frames, policy):
    """Drive `policy(frame)->action` from start_x. Returns the furthest x the
    collider's LEFT edge reached, and whether it got past the obstacle."""
    info = fresh(env, start_x)
    best = rect(info)[0]
    for f in range(frames):
        _o, _r, term, _tr, info = env.step(policy(f))
        best = max(best, rect(info)[0])
        if term:
            break
    return best, best > pipe_right


if __name__ == "__main__":
    env = CustomMarioEnv()
    print("=== flat-ground jump height (collider top), jump HELD ===")
    for label, runup, run_a, jump_a in (
            ("standing jump", 0, NOOP, JUMP),
            ("walk 20 frames then jump", 20, WALK_R, WALK_R_JUMP),
            ("walk 40 frames then jump", 40, WALK_R, WALK_R_JUMP),
            ("walk 80 frames then jump", 80, WALK_R, WALK_R_JUMP),
            ("run 40 frames then jump", 40, RUN_R, RUN_R_JUMP)):
        vx, rise = measure_jump(env, 250, runup, run_a, jump_a)
        print(f"  {label:<28} take-off x_vel {vx:>5.2f}   rise {rise:>4} px")

    for name, wall_left, wall_right in (("tall pipe x=1975", 1975, 2059),
                                        ("tall pipe x=2447", 2447, 2531),
                                        ("staircase x=6003", 6003, 6043)):
        flush = wall_left - 32
        print(f"\n=== {name} (172 px): Mario's LEFT edge must pass x={wall_right} ===")
        # 1. flush against the wall, pressing right+jump the way a stuck policy does
        best, ok = try_clear(env, flush, wall_right, 600,
                             lambda f: WALK_R_JUMP if (f // 30) % 2 == 0 else WALK_R)
        print(f"  flush (x={flush}), right+jump repeatedly   : furthest {best:>5}  cleared={ok}")
        best, ok = try_clear(env, flush, wall_right, 600,
                             lambda f: RUN_R_JUMP if (f // 30) % 2 == 0 else RUN_R)
        print(f"  flush (x={flush}), SPRINT+jump repeatedly  : furthest {best:>5}  cleared={ok}")
        # 2. back off, then a committed running jump
        for back in (40, 70, 100, 150, 220):
            start = flush - back
            dist = wall_left - 30 - start          # px of run available before the wall

            def run_then_jump(f, dist=dist):
                return RUN_R if f < max(1, dist // 6) else RUN_R_JUMP
            best, ok = try_clear(env, start, wall_right, 400, run_then_jump)
            print(f"  backed off {back:>3} px (start x={start}), run then hold jump: "
                  f"furthest {best:>5}  cleared={ok}")
    env.close()
