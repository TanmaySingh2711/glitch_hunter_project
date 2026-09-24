"""Runs every injected-bug scenario on ONE game variant and prints the
measurements as JSON. Run by tests/test_injected_bugs.py, once per variant
(both import the game as `data`, so each gets a fresh process):

    python tests/bug_probe.py mario_clean
    python tests/bug_probe.py mario_bugged

A scenario resets the level, places Mario at a spot, feeds a fixed key
sequence one engine frame at a time and measures what happened - from the
OUTSIDE: Mario's collider against the geometry as it is DRAWN (the clean
level's rectangles, listed below), never by asking the bugged code whether its
bug fired. Every bug has a matching CONTROL scenario at a similar spot the bug
does not touch; those must come out identical on both games.

Placing Mario directly skips the enemies earlier checkpoints would have
spawned; the Goomba scenarios spawn their pair exactly the way checkpoint
'9' / '10' does (level1.check_points_check). Each scenario runs twice and
reports whether the two runs were identical.
"""
import hashlib
import json
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame as pg

from custom_mario_env import CustomMarioEnv

NOOP, RIGHT, JUMP, LEFT = 0, 1, 5, 6
GROUND = 538

# The level as DRAWN (mario_clean's colliders): what the player sees as solid.
PIPE3 = (1973, 366, 83, 170)
PIPE4 = (2445, 366, 83, 170)
STEP4_TOP_BLOCKS = (5874, 366, 40, 86)    # the column's two drawn top blocks
STEP5_TOP_BLOCKS = (6001, 366, 40, 86)
BRICK20 = (5058, 365, 43, 43)
BRICK18 = (4287, 365, 43, 43)
WALL_SPOT = (4412, 452, 40, 86)          # where the invisible wall stands
OPEN_SPOT = (1330, 452, 40, 86)          # an equally empty patch between pipes 1 and 2


def overlap(rect, box):
    x, y, w, h = rect
    bx, by, bw, bh = box
    ox = min(x + w, bx + bw) - max(x, bx)
    oy = min(y + h, by + bh) - max(y, by)
    return (ox, oy) if ox > 0 and oy > 0 else (0, 0)


def gap(a, b):
    """Pixels of clear space between two rects (0 = touching or overlapping)."""
    return max(b[0] - (a[0] + a[2]), a[0] - (b[0] + b[2]),
               b[1] - (a[1] + a[3]), a[1] - (b[1] + b[3]), 0)


def rect(sprite):
    r = sprite.rect
    return (int(r.x), int(r.y), int(r.w), int(r.h))


class Scene:
    def __init__(self, env):
        self.env = env
        env.enable_evidence()           # the Objective-3 detectors watch every scenario
        env.reset()
        env.drain_detections()
        self.level = env.game.state
        self.mario = self.level.mario
        self.trace = hashlib.sha256()
        self.rows = []

    def place(self, x, bottom, on_ground):
        m, lvl = self.mario, self.level
        m.rect.x, m.rect.bottom = x, bottom
        m.x_vel = m.y_vel = 0
        # 'standing', not 'walk': from 'walk' with no direction key the engine
        # overwrites a fresh jump with STAND (mario.walking), so a standing
        # jump would be swallowed.
        m.state = "standing" if on_ground else "fall"
        lvl.viewport.x = max(0, x - 300)

    def spawn_pair(self, group_number, distance):
        """What checkpoint `group_number` does when Mario touches it, except
        that the pair starts `distance` px ahead of Mario rather than at the
        edge of the screen (the same walk, just shorter)."""
        lvl = self.level
        group = lvl.enemy_group_list[group_number - 1]
        for index, enemy in enumerate(group):
            enemy.rect.x = self.mario.rect.right + distance + (index * 60)
        lvl.enemy_group.add(group)
        lvl.mario_and_enemy_group.add(lvl.enemy_group)
        return list(group)

    def run(self, actions):
        for a in actions:
            _obs, _r, done, _t, info = self.env.step(a)
            r = rect(self.mario)
            self.rows.append((r, self.mario.state, bool(info.get("is_dead"))))
            self.trace.update(repr(self.rows[-1]).encode())
            if info.get("is_dead"):
                return "death", info.get("death_cause")
            if done:
                return "done", None
        return "ok", None

    def screen_crop(self, box):
        """The drawn pixels of a level-space box on the current frame."""
        x, y, w, h = box
        sx = x - self.level.viewport.x
        sub = pg.display.get_surface().subsurface(pg.Rect(sx, y, w, h)).copy()
        return hashlib.sha256(pg.image.tobytes(sub, "RGB")).hexdigest()


# ── scenarios ────────────────────────────────────────────────────────────────
def pipe_walk_top(env, pipe):
    """Stand on the pipe's left rim and walk right along its top."""
    s = Scene(env)
    s.place(pipe[0] + 2, pipe[1], True)
    end = s.run([RIGHT] * 45)
    body = (pipe[0], pipe[1], pipe[2], pipe[3])
    deepest = max(overlap(r, body)[1] for r, _st, _d in s.rows)
    return {"end": end, "max_depth_inside_pipe_px": deepest,
            "min_bottom_while_over_pipe": min(r[1] + r[3] for r, _st, _d in s.rows
                                              if r[0] < body[0] + body[2] and r[0] + r[2] > body[0]),
            "trace": s.trace.hexdigest()}


def pipe_walk_in_from_right(env, pipe):
    s = Scene(env)
    s.place(pipe[0] + pipe[2] + 40, GROUND, True)
    end = s.run([LEFT] * 60)
    return {"end": end, "min_left_x": min(r[0] for r, _st, _d in s.rows),
            "pipe_right_edge": pipe[0] + pipe[2], "trace": s.trace.hexdigest()}


def land_on_column(env, block):
    """Drop Mario straight onto a stair column's top block."""
    s = Scene(env)
    s.place(block[0] + 5, block[1] - 60, False)
    end = s.run([NOOP] * 40)
    r = s.rows[-1][0]
    return {"end": end, "rest_bottom": r[1] + r[3], "drawn_top": block[1],
            "sunk_px": r[1] + r[3] - block[1], "overlap_with_block": overlap(r, block),
            "trace": s.trace.hexdigest()}


def walk_into_column_top(env):
    """Stand on step3 and walk right into step4's top blocks."""
    s = Scene(env)
    s.place(5836, 409, True)
    end = s.run([RIGHT] * 40)
    right = max(r[0] + r[2] for r, _st, _d in s.rows)
    return {"end": end, "max_right_edge": right, "drawn_face_x": STEP4_TOP_BLOCKS[0],
            "max_overlap_with_block": max((overlap(r, STEP4_TOP_BLOCKS) for r, _s, _d in s.rows),
                                          key=lambda o: o[0] * o[1]),
            "trace": s.trace.hexdigest()}


def jump_under(env, brick):
    """Stand under a brick and hold jump."""
    s = Scene(env)
    s.place(brick[0] + 6, GROUND, True)
    end = s.run([JUMP] * 40 + [NOOP] * 60)
    tops = [r[1] for r, _st, _d in s.rows]
    ov = max((overlap(r, brick) for r, _st, _d in s.rows), key=lambda o: o[0] * o[1])
    return {"end": end, "highest_top_y": min(tops), "brick_bottom": brick[1] + brick[3],
            "max_overlap_with_brick": ov, "trace": s.trace.hexdigest()}


def walk_across(env, spot):
    """Walk right on open ground across `spot`, then check what is drawn there."""
    s = Scene(env)
    s.place(spot[0] - 70, GROUND, True)
    s.run([NOOP])
    drawn = s.screen_crop(spot)
    end = s.run([RIGHT] * 70)
    right = max(r[0] + r[2] for r, _st, _d in s.rows)
    return {"end": end, "max_right_edge": right, "spot_left_x": spot[0],
            "stopped_at_spot": right == spot[0], "drawn_pixels_sha256": drawn,
            "trace": s.trace.hexdigest()}


def goomba_approach(env, group_number, which):
    """Mario stands still on open ground; checkpoint group `group_number`
    spawns and walks into him. Measures the clear space left at the hit."""
    s = Scene(env)
    x = {9: 5300, 10: 7100}[group_number]
    s.place(x, GROUND, True)
    s.run([NOOP])
    pair = s.spawn_pair(group_number, 150)
    target = pair[which]
    for other in pair:
        if other is not target:
            other.kill()
    end = s.run([NOOP] * 600)
    return {"end": end, "gap_at_hit_px": gap(s.rows[-1][0], rect(target)) if end[0] == "death" else None,
            "trace": s.trace.hexdigest()}


def jump_from(env, x, hold):
    s = Scene(env)
    s.place(x, GROUND, True)
    env.enable_evidence()
    end = s.run([JUMP] * hold + [NOOP] * 260)
    tops = [r[1] for r, _st, _d in s.rows]
    kinds = sorted({d.kind for d in env.drain_detections()})
    return {"end": end, "highest_top_y": min(tops), "detections": kinds,
            "trace": s.trace.hexdigest()}


SCENARIOS = {
    # BUG pipe-clip + its control (pipe 3)
    "pipe4_walk_top": lambda env: pipe_walk_top(env, PIPE4),
    "pipe4_walk_in_from_right": lambda env: pipe_walk_in_from_right(env, PIPE4),
    "control_pipe3_walk_top": lambda env: pipe_walk_top(env, PIPE3),
    "control_pipe3_walk_in_from_right": lambda env: pipe_walk_in_from_right(env, PIPE3),
    # BUG stair-clip + its control (step13, the second staircase's column)
    "land_on_step4": lambda env: land_on_column(env, STEP4_TOP_BLOCKS),
    "land_on_step5": lambda env: land_on_column(env, STEP5_TOP_BLOCKS),
    "walk_step3_into_step4": walk_into_column_top,
    "control_land_on_step13": lambda env: land_on_column(env, (6517, 366, 40, 86)),
    # BUG ceiling-clip + its control (another lone plain brick)
    "jump_under_brick20": lambda env: jump_under(env, BRICK20),
    "control_jump_under_brick18": lambda env: jump_under(env, BRICK18),
    # BUG invisible-wall + its control (an equally empty patch)
    "walk_across_wall_spot": lambda env: walk_across(env, WALL_SPOT),
    "control_walk_across_open_spot": lambda env: walk_across(env, OPEN_SPOT),
    # BUG false-goomba-hit + its controls (its partner, and another pair)
    "goomba14_walks_into_mario": lambda env: goomba_approach(env, 10, 0),
    "control_goomba15_walks_into_mario": lambda env: goomba_approach(env, 10, 1),
    "control_goomba12_walks_into_mario": lambda env: goomba_approach(env, 9, 0),
    # BUG open-sky-jump (tap and full hold) + its control (outside the zone)
    "tap_jump_at_x300": lambda env: jump_from(env, 300, 4),
    "held_jump_at_x300": lambda env: jump_from(env, 300, 60),
    "control_held_jump_at_x600": lambda env: jump_from(env, 600, 60),
}


def main(variant):
    env = CustomMarioEnv(game_variant=variant)
    out = {"variant": variant}
    for name, fn in SCENARIOS.items():
        first = fn(env)
        first.setdefault("detections", sorted({d.kind for d in env.drain_detections()}))
        second = fn(env)
        env.drain_detections()
        first["repeatable"] = first["trace"] == second["trace"]
        out[name] = first
    print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1])
