"""Collision and jump-physics invariants: what the engine does to Mario must
agree with what is DRAWN and with the engine's own laws of motion. Four rules,
each true everywhere in any correct build of the level and none aware of where
(or whether) a bug was injected:

  SOLID PENETRATION   Mario's collider is never more than PENETRATION_TOL px
                      inside a solid, in both axes. Solids are the level's
                      designed ground, pipes and steps (reporting/level_design)
                      and the live bricks and ? boxes (sprites, drawn at their
                      rects). The engine resolves every contact by snapping
                      Mario flush to the edge, so in a correct build the depth
                      is 0. Kinds: clip_into_step / clip_into_pipe /
                      clip_into_ground / clip_into_block (+ the side he came in
                      from: a block entered from below is a ceiling clip).
  COLLISION WITH NOTHING  Every collider the engine resolves Mario against must
                      be drawn. When its collision response visibly acts -
                        side     he was moving (|x_vel| >= 1), x_vel was zeroed
                                 and a collider is flush on that side,
                        support  he is standing on a collider,
                        ceiling  a rise became the head-bump fall (y_vel 7)
                                 under a collider flush above -
                      the collider touching him there must lie on a drawn
                      solid. The running game's colliders are used only as
                      proof that a collision happened; whether it was legal is
                      judged against what is drawn. Kind: invisible_collision.
  HIT WITHOUT CONTACT An enemy hurt Mario (a death by an enemy, or big -> small)
                      although no enemy or shell came within HIT_CONTACT_TOL px
                      of him at any point of that frame: the swept boxes of
                      their previous and current positions do not touch. Kind:
                      hit_without_contact.
  IMPOSSIBLE JUMP     No legal move sends Mario up faster than the engine's own
                      fastest declared jump (constants.FAST_JUMP_VEL, 12.5
                      px/frame; ordinary jumps take off at 10-10.5, stomp
                      bounces at 7), nor higher above where he last stood than
                      that speed can carry him under the rising gravity
                      (12.5^2 / (2 x 0.31) = 252 px, +TOL). Kind: impossible_jump;
                      one jump is one report, whichever limit it breaks first.

Suspended where the engine itself suspends collision: Mario dead or dying, on
the flagpole or walking to the castle, and during the grow/shrink/fire
transitions (level1 skips his vertical move and collision checks then).

Read-only: every value comes from the level's sprites and the previous
frame's snapshot this object keeps. Run by CustomMarioEnv only while evidence
is on (the dashboard and incident replays), so training and evaluation are
exactly as before.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from exploration import config

Rect = tuple[int, int, int, int]
PENETRATION_TOL = config.PENETRATION_TOL    # the reachability model's own tolerance
HIT_CONTACT_TOL = 2                          # px of clear space still counted as a touch
PROBE = 1                                    # px beside / below / above Mario: flush contact
MIN_STOPPED_SPEED = 1.0                      # |x_vel| a side collision must have killed
HEAD_BUMP_Y_VEL = 7                          # what level1 sets on every head bump
# The engine's laws of jumping (mario_clean/data/constants.py): the fastest
# take-off it declares, and the gravity while a jump is held.
FASTEST_JUMP_SPEED = 12.5                    # -constants.FAST_JUMP_VEL
JUMP_GRAVITY = 0.31                          # constants.JUMP_GRAVITY
MAX_JUMP_RISE = FASTEST_JUMP_SPEED ** 2 / (2 * JUMP_GRAVITY) + PENETRATION_TOL   # 258 px
ENEMY_DEATH_CAUSES = frozenset(("goomba", "koopa", "koopa_shell", "enemy"))
SUSPENDED_STATES = frozenset(("death jump", "flag pole", "walking to castle", "end of level fall",
                              "small to big", "big to small", "big to fire"))
DETECTOR = "collision_invariants"

_CLIP_KIND = {"step": "clip_into_step", "pipe": "clip_into_pipe", "ground": "clip_into_ground",
              "brick": "clip_into_block", "coin_box": "clip_into_block"}
_SOLID_NAME = {"step": "stair step", "pipe": "pipe", "ground": "ground",
               "brick": "brick", "coin_box": "? block"}

Hit = tuple[str, str, dict[str, Any], str, str]


def _rect(sprite: Any) -> Rect:
    r = sprite.rect
    return (int(r.x), int(r.y), int(r.w), int(r.h))


def overlap(a: Rect, b: Rect) -> tuple[int, int]:
    """Depth of a inside b on each axis (0, 0 when they do not overlap)."""
    ox = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    oy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return (ox, oy) if ox > 0 and oy > 0 else (0, 0)


def intersection(a: Rect, b: Rect) -> Rect | None:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return (x0, y0, x1 - x0, y1 - y0) if x1 > x0 and y1 > y0 else None


def gap(a: Rect, b: Rect) -> int:
    """Clear pixels between two rects (0 = touching or overlapping)."""
    return max(b[0] - (a[0] + a[2]), a[0] - (b[0] + b[2]),
               b[1] - (a[1] + a[3]), a[1] - (b[1] + b[3]), 0)


def union(a: Rect, b: Rect) -> Rect:
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    return (x0, y0, max(a[0] + a[2], b[0] + b[2]) - x0, max(a[1] + a[3], b[1] + b[3]) - y0)


def _undrawn_contact(probe: Rect, colliders: Iterable[Rect], drawn: Sequence[tuple[str, Rect]]
                     ) -> tuple[bool, Rect | None]:
    """(a collider touches the probe, the touched part of one that no drawn
    solid covers - or None when every touch is on something drawn)."""
    touched = False
    for c in colliders:
        part = intersection(probe, c)
        if part is None:
            continue
        touched = True
        if not any(intersection(part, r) for _k, r in drawn):
            return True, part
    return touched, None


class CollisionInvariants:
    """One per env. check() is called after every engine frame; reset() at
    every episode start."""

    def __init__(self, design: Sequence[tuple[str, Rect]]) -> None:
        self.design: tuple[tuple[str, Rect], ...] = tuple(
            (str(k), (int(r[0]), int(r[1]), int(r[2]), int(r[3]))) for k, r in design)
        self._prev: dict[str, Any] | None = None
        self._support_bottom: int | None = None

    def reset(self) -> None:
        self._prev = None
        self._support_bottom = None

    def check(self, level: Any) -> list[Hit]:
        """[(kind, message, metrics, detector id, report-once key)]."""
        mario = getattr(level, "mario", None)
        if mario is None or getattr(mario, "rect", None) is None:
            self.reset()
            return []
        blocks = [("brick", _rect(s)) for s in getattr(level, "brick_group", ())]
        blocks += [("coin_box", _rect(s)) for s in getattr(level, "coin_box_group", ())]
        now = {"rect": _rect(mario), "x_vel": float(getattr(mario, "x_vel", 0.0)),
               "y_vel": float(getattr(mario, "y_vel", 0.0)), "state": str(getattr(mario, "state", "")),
               "dead": bool(getattr(mario, "dead", False)),
               "death_cause": getattr(mario, "death_cause", None),
               "enemies": {id(e): _rect(e) for grp in ("enemy_group", "shell_group")
                           for e in getattr(level, grp, ())},
               "blocks": blocks}
        prev, self._prev = self._prev, now
        hits: list[Hit] = []
        if prev is not None:
            hits += self._hit_without_contact(prev, now)
        suspended = (now["dead"] or now["state"] in SUSPENDED_STATES
                     or bool(getattr(mario, "in_transition_state", False))
                     or bool(getattr(mario, "in_castle", False)))
        if suspended:
            self._support_bottom = None
            return hits
        drawn = list(self.design) + blocks
        hits += self._penetration(prev, now, drawn)
        if prev is not None and not prev["dead"] and prev["state"] not in SUSPENDED_STATES:
            # A block smashed or created this frame was there when the engine
            # resolved the contact: judge against both frames' blocks.
            drawn_either = drawn + [b for b in prev["blocks"] if b not in blocks]
            colliders = [_rect(s) for s in getattr(level, "ground_step_pipe_group", ())]
            colliders += [r for _k, r in blocks]
            hits += self._collision_with_nothing(prev, now, colliders, drawn_either)
        hits += self._impossible_jump(now)
        return hits

    # ── rule 1 ────────────────────────────────────────────────────────────
    def _penetration(self, prev: dict[str, Any] | None, now: dict[str, Any],
                     solids: Sequence[tuple[str, Rect]]) -> list[Hit]:
        rect = now["rect"]
        deepest = None
        for solid, r in solids:
            ox, oy = overlap(rect, r)
            if ox > PENETRATION_TOL and oy > PENETRATION_TOL and (deepest is None or ox * oy > deepest[0]):
                deepest = (ox * oy, solid, r, ox, oy)
        if deepest is None:
            return []
        _a, solid, r, ox, oy = deepest
        entry = "unknown"
        if prev is not None:
            p = prev["rect"]
            if p[1] + p[3] <= r[1] + PENETRATION_TOL:
                entry = "top"
            elif p[1] >= r[1] + r[3] - PENETRATION_TOL:
                entry = "bottom"
            elif p[0] + p[2] <= r[0] + PENETRATION_TOL:
                entry = "left side"
            elif p[0] >= r[0] + r[2] - PENETRATION_TOL:
                entry = "right side"
            else:
                entry = "already inside"
        kind = _CLIP_KIND[solid]
        how = " - through its underside (a ceiling clip)" if entry == "bottom" else ""
        message = (f"Mario is inside a solid {_SOLID_NAME[solid]}: his collider is {ox}x{oy} px deep "
                   f"in it (entered from the {entry}){how}; the engine should have stopped him at its edge.")
        metrics = {"solid": solid, "solid_rect": list(r), "mario_rect": list(rect),
                   "depth_x": ox, "depth_y": oy, "entered_from": entry,
                   "tolerance_px": PENETRATION_TOL}
        return [(kind, message, metrics, f"{DETECTOR}/solid_penetration", f"{kind}@{solid}:{r[0]},{r[1]}")]

    # ── rule 2 ────────────────────────────────────────────────────────────
    def _collision_with_nothing(self, prev: dict[str, Any], now: dict[str, Any],
                                colliders: Sequence[Rect], drawn: Sequence[tuple[str, Rect]]) -> list[Hit]:
        x, y, w, h = now["rect"]
        found = []
        if abs(prev["x_vel"]) >= MIN_STOPPED_SPEED and now["x_vel"] == 0:
            # level1 resolves the horizontal move BEFORE the vertical one, so
            # the side contact happened at the previous frame's height: probe
            # the whole vertical span he covered this frame.
            right = prev["x_vel"] > 0
            p = prev["rect"]
            top, bottom = min(y, p[1]), max(y + h, p[1] + p[3])
            probe = (x + w, top, PROBE, bottom - top) if right else (x - PROBE, top, PROBE, bottom - top)
            what = (f"stopped while moving {'right' if right else 'left'} at "
                    f"{abs(prev['x_vel']):.1f} px/frame")
            found.append(("side", probe, what))
        if now["state"] in ("walk", "standing") and now["y_vel"] == 0:
            found.append(("support", (x, y + h, w, PROBE), "standing on it"))
        if prev["y_vel"] < 0 and now["y_vel"] == HEAD_BUMP_Y_VEL and now["state"] == "fall":
            found.append(("ceiling", (x, y - PROBE, w, PROBE), "his head bumped into it"))
        hits: list[Hit] = []
        for contact, probe, what in found:
            touched, undrawn = _undrawn_contact(probe, colliders, drawn)
            if not touched or undrawn is None:
                continue
            message = (f"Mario was stopped by something that is not drawn: {what}, where no "
                       f"solid is drawn ({contact} contact).")
            hits.append(("invisible_collision", message,
                         {"contact": contact, "mario_rect": list(now["rect"]),
                          "undrawn_contact_rect": list(undrawn),
                          "x_vel_before": prev["x_vel"], "x_vel_after": now["x_vel"],
                          "y_vel_before": prev["y_vel"], "y_vel_after": now["y_vel"],
                          "state": now["state"]},
                         f"{DETECTOR}/invisible_collision",
                         f"invisible_collision/{contact}@{undrawn[0] // 43},{undrawn[1] // 43}"))
        return hits

    # ── rule 3 ────────────────────────────────────────────────────────────
    def _hit_without_contact(self, prev: dict[str, Any], now: dict[str, Any]) -> list[Hit]:
        died = now["dead"] and not prev["dead"] and now["death_cause"] in ENEMY_DEATH_CAUSES
        shrank = now["state"] == "big to small" and prev["state"] != "big to small"
        if not (died or shrank):
            return []
        swept_mario = union(prev["rect"], now["rect"])
        best: tuple[int, Rect] | None = None
        for eid, rect in now["enemies"].items():
            swept = union(prev["enemies"].get(eid, rect), rect)
            g = gap(swept_mario, swept)
            if best is None or g < best[0]:
                best = (g, rect)
        if best is not None and best[0] <= HIT_CONTACT_TOL:
            return []
        what = f"killed by a {now['death_cause']}" if died else "hurt (shrank from big to small)"
        clear = ("no enemy was on screen" if best is None
                 else f"the nearest enemy never came within {best[0]} px of him")
        message = (f"Mario was {what} without touching it: {clear} during that frame "
                   f"(contact means <= {HIT_CONTACT_TOL} px).")
        metrics = {"event": "death" if died else "shrink", "death_cause": now["death_cause"],
                   "gap_px": None if best is None else best[0],
                   "nearest_enemy_rect": None if best is None else list(best[1]),
                   "mario_rect": list(now["rect"]), "contact_tolerance_px": HIT_CONTACT_TOL}
        return [("hit_without_contact", message, metrics, f"{DETECTOR}/hit_without_contact",
                 "hit_without_contact")]

    # ── rule 4 ────────────────────────────────────────────────────────────
    def _impossible_jump(self, now: dict[str, Any]) -> list[Hit]:
        bottom = now["rect"][1] + now["rect"][3]
        if now["state"] in ("walk", "standing") and now["y_vel"] == 0:
            self._support_bottom = bottom
        hits: list[Hit] = []
        rising = -now["y_vel"]
        if rising > FASTEST_JUMP_SPEED:
            message = (f"Mario is rising at {rising:.1f} px/frame, faster than the engine's fastest "
                       f"jump ({FASTEST_JUMP_SPEED:g} px/frame).")
            hits.append(("impossible_jump", message,
                         {"rule": "rise_speed", "rise_speed": round(rising, 3),
                          "limit": FASTEST_JUMP_SPEED, "mario_rect": list(now["rect"]),
                          "state": now["state"]},
                         f"{DETECTOR}/impossible_jump", "impossible_jump"))
        if self._support_bottom is not None:
            climb = self._support_bottom - bottom
            if climb > MAX_JUMP_RISE:
                message = (f"Mario is {climb} px above where he last stood; the engine's fastest jump "
                           f"can climb at most {MAX_JUMP_RISE:.0f} px.")
                hits.append(("impossible_jump", message,
                             {"rule": "rise_height", "climb_px": climb, "limit": round(MAX_JUMP_RISE, 1),
                              "support_bottom": self._support_bottom, "mario_rect": list(now["rect"]),
                              "state": now["state"]},
                             f"{DETECTOR}/impossible_jump", "impossible_jump"))
        return hits
