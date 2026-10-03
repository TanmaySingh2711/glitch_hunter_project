"""Where to fix it: the most likely place in the game's own code for the cause
of an incident. INFERRED, never measured - every report labels it so.

It is worked out when a report is rendered, from two things only:
  * what the incident recorded: the detector's measurements and the colliders
    and sprites in view at the trigger frame (incident.json), and
  * the game's source as it is on disk (<variant>/data/...), read as text and
    parsed with `ast` - never imported, never run.

The reasoning follows the detectors (reporting/collision_invariants.py),
which compare the engine with what is DRAWN:

  Mario inside a drawn pipe, step or ground
      no collider there              -> the level never built one: its setup
                                        function
      a collider smaller than the drawing
                                     -> the line that builds that collider,
                                        with the drawn size to give it
      a collider as big as the drawing
                                     -> the collision checks let him through:
                                        where they look the collider up
  Mario inside a brick or ? block    -> the block is drawn at its own collider,
                                        so the collision checks let him
                                        through; plus every line that changes
                                        that block after it is built
  stopped by something not drawn     -> the line that builds that collider
                                        and the line that adds it to the level
  hurt with no enemy touching him    -> where the hurt check decides which
                                        enemy touched him
  a jump faster or higher than the engine's own
                                     -> Mario's take-off code

Nothing here knows where, or whether, a bug was injected: the bugged game's
INJECTED_BUGS.json is never read. tests/test_fix_hint.py checks, on the six
benchmark bugs' real incidents, that each lead lands on the injected code.
"""
from __future__ import annotations

import ast
import itertools
import os
import re
import textwrap
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reporting.variants import PROJECT_ROOT, game_dir

Rect = tuple[int, int, int, int]

LEVEL_FILE = ("data", "states", "level1.py")
PLAYER_FILE = ("data", "components", "mario.py")

# What the level calls each kind of solid, and the sprite group that holds it.
_GROUP = {"ground": "ground_group", "pipe": "pipe_group", "step": "step_group",
          "brick": "brick_group", "coin_box": "coin_box_group"}
_NAME = {"ground": "ground", "pipe": "pipe", "step": "stair step", "brick": "brick",
         "coin_box": "? block"}
_DESIGNED = ("ground", "pipe", "step")          # drawn by the level art; colliders built by hand
_COLLIDER_GROUP = "ground_step_pipe_group"      # what the collision checks look them up in
MAX_CODE_LINES = 6                              # lines of code quoted per place
BASIS = ("Inferred from what the incident recorded and from the game's source code. "
         "It is a lead, not a proven root cause: fix it, then run the test again to confirm.")


# ═══════════════════════════════════════════════════════════════════════
# THE GAME'S SOURCE, READ AS TEXT
# ═══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class _Function:
    name: str               # "Level1.setup_steps"
    start: int              # first line (the def), 1-based
    end: int                # last line
    node: ast.FunctionDef


class _Source:
    """One source file: its lines, and its functions with their line spans."""

    def __init__(self, path: str, text: str) -> None:
        self.path = path
        self.lines = text.splitlines()
        self.tree = ast.parse(text)
        self.parents = {child: parent for parent in ast.walk(self.tree)
                        for child in ast.iter_child_nodes(parent)}
        self.functions: list[_Function] = []
        for cls in [n for n in self.tree.body if isinstance(n, ast.ClassDef)] + [self.tree]:
            prefix = f"{cls.name}." if isinstance(cls, ast.ClassDef) else ""
            for node in cls.body:
                if isinstance(node, ast.FunctionDef):
                    self.functions.append(_Function(prefix + node.name, node.lineno,
                                                    node.end_lineno or node.lineno, node))

    def function_at(self, line: int) -> _Function | None:
        return next((f for f in self.functions if f.start <= line <= f.end), None)

    def text(self, first: int, last: int) -> list[str]:
        return [self.lines[i - 1].rstrip() for i in range(first, min(last, first + MAX_CODE_LINES - 1) + 1)
                if 0 < i <= len(self.lines)]

    def removable(self, st: ast.stmt) -> ast.stmt:
        """What must go for `st` to go: `st` itself, or the `if` whose only
        statement it is (an `if` with an empty body is not valid Python)."""
        parent = self.parents.get(st)
        while isinstance(parent, ast.If) and parent.body == [st] and not parent.orelse:
            st, parent = parent, self.parents.get(parent)
        return st

    def statements(self, fn: _Function | None = None) -> Iterable[ast.stmt]:
        for node in ast.walk(fn.node if fn is not None else self.tree):
            if isinstance(node, ast.stmt) and not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                yield node


_CACHE: dict[str, tuple[float, _Source]] = {}
_CACHE_LOCK = threading.Lock()


def _source(variant: str, parts: Sequence[str]) -> _Source | None:
    """The parsed file, cached until it changes on disk; None if unreadable."""
    try:
        path = os.path.join(game_dir(variant), *parts)
        mtime = os.path.getmtime(path)
    except (OSError, ValueError):
        return None
    with _CACHE_LOCK:
        hit = _CACHE.get(path)
        if hit is not None and hit[0] == mtime:
            return hit[1]
    try:
        with open(path, encoding="utf-8") as fh:
            src = _Source(path, fh.read())
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    with _CACHE_LOCK:
        _CACHE[path] = (mtime, src)
    return src


def _segment(src: _Source, node: ast.AST) -> str:
    return ast.get_source_segment("\n".join(src.lines), node) or ""


def _place(src: _Source, first: int, last: int, why: str,
           focus: str | None = None, suspect: bool = True,
           fix: dict[str, Any] | None = None) -> dict[str, Any]:
    """One place in the code. `focus`: of a statement too long to quote,
    only the lines that mention this name. `suspect`: a line that builds or
    changes the thing that misbehaved (look here first); False for context,
    such as the ordinary lookup a suspect line then overrides. `fix`: the
    edit that removes the cause here (_delete / _replace), when one is known."""
    if focus is not None and last - first + 1 > MAX_CODE_LINES:
        word = re.compile(r"\b" + re.escape(focus) + r"\b")
        hits = [i for i in range(first, last + 1) if word.search(src.lines[i - 1])]
        if hits:
            first = last = hits[0]
    fn = src.function_at(first)
    return {"file": os.path.relpath(src.path, PROJECT_ROOT).replace("\\", "/"),
            "line": first, "end_line": last,
            "function": fn.name if fn is not None else None,
            "code": textwrap.dedent("\n".join(src.text(first, last))).splitlines(), "why": why,
            "suspect": suspect, "fix": fix}


# ── the edits a fix is made of ────────────────────────────────────────────
def _delete(first: int, last: int) -> dict[str, Any]:
    return {"do": "delete", "first": first, "last": last}


def _replace(src: _Source, line: int, new: str) -> dict[str, Any]:
    """Line `line` becomes `new` (kept at that line's indentation)."""
    old = src.lines[line - 1]
    indent = old[:len(old) - len(old.lstrip())]
    return {"do": "replace", "first": line, "last": line, "code": [indent + new.strip()]}


def _without_comment(line: str) -> str:
    return line.split("#", 1)[0].rstrip()


def _removal(src: _Source, st: ast.stmt, why: str, focus: str | None = None) -> dict[str, Any]:
    """A place whose fix is to delete `st` (with the `if` it alone fills)."""
    whole = src.removable(st)
    a, b = _of(whole)
    return _place(src, a, b, why, focus=focus if whole is st else None, fix=_delete(a, b))


def _resized(src: _Source, st: ast.stmt, drawn: Rect) -> dict[str, Any] | None:
    """`name = Collider(x, y, w, h)` rewritten with the drawn rectangle."""
    call = getattr(st, "value", None)
    if not isinstance(call, ast.Call) or st.lineno != st.end_lineno or len(call.args) < 4:
        return None
    args = call.args[:4]
    if not all(isinstance(a, ast.Constant) and isinstance(a.value, (int, float)) for a in args):
        return None
    line = src.lines[st.lineno - 1]
    for arg, value in reversed(list(zip(args, drawn, strict=True))):
        line = line[:arg.col_offset] + str(value) + line[arg.end_col_offset or arg.col_offset:]
    return _replace(src, st.lineno, _without_comment(line))


def _unlisted(src: _Source, use: ast.stmt, var: str) -> dict[str, Any]:
    """The place where `var` is handed to a sprite group, fixed by taking it
    out of that list."""
    name = re.compile(r"\b" + re.escape(var) + r"\b")
    a, b = _of(use)
    for i in range(a, b + 1):
        if name.search(src.lines[i - 1]):
            rest = re.sub(r"\b" + re.escape(var) + r"\b\s*,?\s*", "", _without_comment(src.lines[i - 1]))
            fix = _delete(i, i) if not rest.strip() else _replace(src, i, rest)
            fix["say"] = f"Take {var} out of this list on line {i}."
            return _place(src, i, i, f"puts {var} into the level", fix=fix)
    return _place(src, a, b, f"puts {var} into the level", focus=var)


def fix_text(place: Mapping[str, Any]) -> str | None:
    """The fix as one plain instruction."""
    fix = place.get("fix")
    if not fix:
        return None
    if fix.get("say"):
        return str(fix["say"])
    span =(f"line {fix['first']}" if fix["first"] == fix["last"]
            else f"lines {fix['first']}-{fix['last']}")
    if fix["do"] == "delete":
        return f"Delete {span}."
    return f"Change {span} to: {' '.join(c.strip() for c in fix['code'])}"


def _of(node: ast.stmt) -> tuple[int, int]:
    return node.lineno, node.end_lineno or node.lineno


# ── finding things in the source ──────────────────────────────────────────
def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _numbers(call: ast.Call) -> list[int | None]:
    return [int(arg.value) if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float))
            else None for arg in call.args]


def _builders(src: _Source, x: int, y: int | None) -> list[tuple[str, ast.Assign]]:
    """`name = Something(x, y, ...)` statements whose first two numeric
    arguments are this position (y None: x alone, for rows built on a named
    height such as the ground)."""
    found = []
    for st in src.statements():
        if (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
                and isinstance(st.value, ast.Call)):
            nums = _numbers(st.value)
            if len(nums) >= 2 and nums[0] == x and (y is None or nums[1] == y):
                found.append((st.targets[0].id, st))
    return found


def _uses(src: _Source, name: str, skip: ast.stmt) -> list[ast.stmt]:
    """Top-level statements (inside any function) other than `skip` that use
    the variable `name` - where it is changed, or handed to the level."""
    out = []
    for fn in src.functions:
        for st in fn.node.body:
            for inner in ast.walk(st):
                if inner is skip:
                    break
            else:
                if name in _names(st):
                    out.append(_innermost(st, name))
    return out


def _innermost(st: ast.stmt, name: str) -> ast.stmt:
    """The smallest statement inside `st` that still mentions `name`."""
    best = st
    for node in ast.walk(st):
        if (isinstance(node, ast.stmt) and node is not st and name in _names(node)
                and (node.end_lineno or node.lineno) - node.lineno
                <= (best.end_lineno or best.lineno) - best.lineno):
            best = node
    return best


def _setup_function(src: _Source, group: str) -> _Function | None:
    """The function that builds `self.<group>`."""
    for fn in src.functions:
        for st in src.statements(fn):
            if isinstance(st, ast.Assign) and any(
                    isinstance(t, ast.Attribute) and t.attr == group for t in st.targets):
                return fn
    return None


def _lookups(src: _Source, group: str) -> list[tuple[_Function, list[ast.stmt]]]:
    """For each function that looks Mario up in `self.<group>`
    (`v = pg.sprite.spritecollideany(self.mario, self.<group>)`): the lookup
    and every later statement in it that sets `v` again - the only places
    that decide what Mario collides with there."""
    out = []
    for fn in src.functions:
        found: list[ast.stmt] = []
        var = None
        for st in src.statements(fn):
            if (var is None and isinstance(st, ast.Assign) and len(st.targets) == 1
                    and isinstance(st.targets[0], ast.Name) and "spritecollideany" in _segment(src, st.value)
                    and re.search(rf"self\.mario\s*,\s*self\.{group}\b", _segment(src, st.value))):
                var = st.targets[0].id
                found.append(st)
        if var is None:
            continue
        for st in src.statements(fn):
            if (st is not found[0] and isinstance(st, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == var for t in st.targets)):
                found.append(st)
        out.append((fn, sorted(found, key=lambda s: s.lineno)))
    return out


def _lookup_places(src: _Source, group: str, what: str, first: str | None,
                   only: Sequence[str] = ()) -> list[dict[str, Any]]:
    places = []
    for fn, sts in _lookups(src, group):
        if only and not any(re.search(p, _segment(src, fn.node)) for p in only):
            continue
        for i, st in enumerate(sts):
            if i == 0:
                a, b = _of(st)
                places.append(_place(src, a, b, f"looks up the {what} Mario touches here",
                                     suspect=False))
            else:
                places.append(_removal(src, st, f"...and changes that result here: whatever "
                                                f"this sets is what this check treats as the {what}"))
    if first:
        places.sort(key=lambda p: 0 if p["function"] and first in p["function"] else 1)
    return places


# ═══════════════════════════════════════════════════════════════════════
# GEOMETRY
# ═══════════════════════════════════════════════════════════════════════
def _rect(v: Any) -> Rect | None:
    if isinstance(v, Mapping):
        v = [v.get("x"), v.get("y"), v.get("w"), v.get("h")]
    if not isinstance(v, (list, tuple)) or len(v) != 4 or not all(isinstance(n, (int, float)) for n in v):
        return None
    return (int(v[0]), int(v[1]), int(v[2]), int(v[3]))


def _intersection(a: Rect, b: Rect) -> Rect | None:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    return (x0, y0, x1 - x0, y1 - y0) if x1 > x0 and y1 > y0 else None


def _covered(part: Rect, by: Sequence[Rect]) -> bool:
    """True when every pixel of `part` lies in one of `by` (exact, on the
    grid of their edges)."""
    xs = sorted({part[0], part[0] + part[2], *(v for r in by for v in (r[0], r[0] + r[2])
                                              if part[0] < v < part[0] + part[2])})
    ys = sorted({part[1], part[1] + part[3], *(v for r in by for v in (r[1], r[1] + r[3])
                                              if part[1] < v < part[1] + part[3])})
    for x0, x1 in itertools.pairwise(xs):
        for y0, y1 in itertools.pairwise(ys):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if not any(r[0] <= cx < r[0] + r[2] and r[1] <= cy < r[1] + r[3] for r in by):
                return False
    return True


def _fmt(r: Rect) -> str:
    return f"x {r[0]}, y {r[1]}, {r[2]} x {r[3]} px"


def _differences(collider: Rect, drawn: Rect) -> list[str]:
    out = []
    if collider[1] != drawn[1]:
        out.append(f"its top is at y {collider[1]} instead of {drawn[1]}")
    if collider[1] + collider[3] != drawn[1] + drawn[3]:
        out.append(f"its bottom is at y {collider[1] + collider[3]} instead of {drawn[1] + drawn[3]}")
    if collider[0] != drawn[0]:
        out.append(f"its left edge is at x {collider[0]} instead of {drawn[0]}")
    if collider[0] + collider[2] != drawn[0] + drawn[2]:
        out.append(f"its right edge is at x {collider[0] + collider[2]} instead of {drawn[0] + drawn[2]}")
    return out


def _geometry(record: Mapping[str, Any], groups: Iterable[str]) -> list[tuple[str, Rect]]:
    wanted = set(groups)
    out = []
    for g in record.get("geometry", []) or []:
        r = _rect(g)
        if r is not None and g.get("group") in wanted:
            out.append((str(g["group"]), r))
    return out


# ═══════════════════════════════════════════════════════════════════════
# ONE RULE PER KIND OF FINDING
# ═══════════════════════════════════════════════════════════════════════
def _vertical_first(entered: str) -> bool:
    return entered in ("top", "bottom")


def _clip(record: Mapping[str, Any], m: Mapping[str, Any], variant: str) -> dict[str, Any] | None:
    solid = str(m.get("solid"))
    drawn, mario = _rect(m.get("solid_rect")), _rect(m.get("mario_rect"))
    level = _source(variant, LEVEL_FILE)
    if drawn is None or mario is None or level is None or solid not in _GROUP:
        return None
    name, entered = _NAME[solid], str(m.get("entered_from", "unknown"))
    check_first = "check_mario_y_collisions" if _vertical_first(entered) else "check_mario_x_collisions"
    inside = _intersection(mario, drawn) or drawn
    evidence = [f"Drawn {name}: {_fmt(drawn)}.",
                (f"Mario's collider was {m.get('depth_x')} x {m.get('depth_y')} px inside it "
                f"(entered from the {entered}).")]

    if solid in _DESIGNED:
        colliders = [r for g, r in _geometry(record, [solid]) if _intersection(r, drawn)]
        if not colliders:
            fn = _setup_function(level, _GROUP[solid])
            places = [_place(level, fn.start, fn.start, f"builds every "
                             f"{name} collider in the level")] if fn else []
            return {"diagnosis": "missing_collider",
                    "summary": f"This {name} is drawn but has no collider, so nothing stops Mario there.",
                    "evidence": [*evidence, f"No {name} collider overlaps the drawing."],
                    "places": places,
                    "suggestion": f"Add a collider with the drawn rectangle ({_fmt(drawn)}).",
                    "basis": BASIS}
        if not _covered(inside, colliders):
            col = max(colliders, key=lambda r: (_intersection(r, drawn) or (0, 0, 0, 0))[2]
                      * (_intersection(r, drawn) or (0, 0, 0, 0))[3])
            diffs = _differences(col, drawn)
            places = []
            builders = (_builders(level, col[0], col[1])
                        or (_builders(level, col[0], None) if solid == "ground" else []))
            for var, st in builders[:1]:
                a, b = _of(st)
                places.append(_place(level, a, b, f"builds this {name}'s collider ({_fmt(col)})",
                                     fix=_resized(level, st, drawn)))
                for use in _uses(level, var, st):
                    if not isinstance(use, ast.Assign) or "Group(" not in _segment(level, use):
                        places.append(_removal(level, use, f"changes {var} after it is built", focus=var))
            if not places:
                fn = _setup_function(level, _GROUP[solid])
                if fn:
                    places.append(_place(level, fn.start, fn.start,
                                         f"builds the {name} colliders"))
            return {"diagnosis": "collider_smaller_than_drawing",
                    "summary": (f"This {name}'s collider is smaller than its drawing "
                                f"({'; '.join(diffs) or 'it does not cover the drawing'}), so Mario "
                                f"can move into the part that has no collision."),
                    "evidence": [*evidence, f"Its collider: {_fmt(col)}."],
                    "places": places,
                    "suggestion": (f"Give the collider the drawn rectangle: x {drawn[0]}, y {drawn[1]}, "
                                   f"width {drawn[2]}, height {drawn[3]}."),
                    "basis": BASIS}
        places = _lookup_places(level, _COLLIDER_GROUP, "ground, pipe or step", check_first)
        return {"diagnosis": "collision_check",
                "summary": (f"This {name}'s collider matches its drawing, so the collision check "
                            f"let Mario through."),
                "evidence": [*evidence, "Its collider covers the whole drawing."],
                "places": places,
                "suggestion": ("Make sure these checks always use what the lookup found; nothing "
                               "should clear or replace it."),
                "basis": BASIS}

    # A brick or ? block is drawn exactly at its own collider.
    places = []
    for var, st in _builders(level, drawn[0], drawn[1])[:1]:
        a, b = _of(st)
        places.append(_place(level, a, b, f"builds this {name}", suspect=False))
        for use in _uses(level, var, st):
            if not isinstance(use, ast.Assign) or "Group(" not in _segment(level, use):
                places.append(_removal(level, use, f"changes {var} after it is built", focus=var))
    places += _lookup_places(level, _GROUP[solid], name, check_first)
    how = " from below (a ceiling clip)" if entered == "bottom" else ""
    return {"diagnosis": "collision_check",
            "summary": (f"A {name} is drawn exactly at its own collider, so the collision check "
                        f"let Mario pass into it{how}."),
            "evidence": evidence,
            "places": places,
            "suggestion": ("The check must treat this block as solid from every side: remove "
                           "anything that clears the lookup's result or marks the block as "
                           "passable."),
            "basis": BASIS}


def _invisible(record: Mapping[str, Any], m: Mapping[str, Any], variant: str) -> dict[str, Any] | None:
    part = _rect(m.get("undrawn_contact_rect"))
    level = _source(variant, LEVEL_FILE)
    if part is None or level is None:
        return None
    hits = [(g, r) for g, r in _geometry(record, _DESIGNED)
            if _intersection((part[0] - 1, part[1] - 1, part[2] + 2, part[3] + 2), r)]
    evidence = [(f"Mario was stopped ({m.get('contact')} contact) by a collider at "
                f"{_fmt(part)}, where nothing is drawn.")]
    if not hits:
        return {"diagnosis": "undrawn_collider",
                "summary": "Something with collision but no drawing stopped Mario.",
                "evidence": evidence, "places": [],
                "suggestion": "Find the collider at this position and remove it, or draw it.",
                "basis": BASIS}
    group, col = hits[0]
    name = _NAME[group]
    places = []
    for var, st in _builders(level, col[0], col[1])[:1]:
        places.append(_removal(level, st, f"builds this collider ({_fmt(col)}); nothing is drawn there"))
        for use in _uses(level, var, st):
            if isinstance(use, ast.Assign) and "Group(" in _segment(level, use):
                places.append(_unlisted(level, use, var))
            else:
                places.append(_removal(level, use, f"changes {var} after it is built", focus=var))
    if not places:
        fn = _setup_function(level, _GROUP[group])
        if fn:
            places.append(_place(level, fn.start, fn.start,
                                 f"builds the {name} colliders"))
    return {"diagnosis": "undrawn_collider",
            "summary": (f"A {name} collider ({_fmt(col)}) has nothing drawn on it, so it is an "
                        f"invisible obstacle."),
            "evidence": [*evidence, f"The collider it belongs to: {_fmt(col)}."],
            "places": places,
            "suggestion": (f"Delete this collider and take it out of the {name} group - or, if the "
                           f"obstacle is meant to be there, draw it."),
            "basis": BASIS}


def _hit(record: Mapping[str, Any], m: Mapping[str, Any], variant: str) -> dict[str, Any] | None:
    level = _source(variant, LEVEL_FILE)
    if level is None:
        return None
    gap = m.get("gap_px")
    enemy = _rect(m.get("nearest_enemy_rect"))
    evidence = [f"Mario was {'killed' if m.get('event') == 'death' else 'hurt'} by a "
                f"{m.get('death_cause') or 'enemy'}; "
                + (f"the nearest enemy ({_fmt(enemy)}) never came within {gap} px of him."
                   if enemy is not None and gap is not None else "no enemy was on screen.")]
    # The hurt path: the check that looks the enemy up AND records what hurt Mario.
    places = _lookup_places(level, "enemy_group", "enemy", None, only=(r"self\.mario\.(death_cause|start_death_jump)",))
    return {"diagnosis": "hurt_without_contact",
            "summary": (f"Mario was hurt {gap} px away from the nearest enemy, so the code that "
                        f"decides he touched an enemy uses something other than their real "
                        f"boxes." if gap is not None else
                        "Mario was hurt with no enemy on screen."),
            "evidence": evidence, "places": places,
            "suggestion": ("Decide contact only from the sprites' own rectangles (the lookup "
                           "line); remove any extra reach added after it."),
            "basis": BASIS}


def _stomp(record: Mapping[str, Any], m: Mapping[str, Any], variant: str) -> dict[str, Any] | None:
    level = _source(variant, LEVEL_FILE)
    if level is None:
        return None
    gap = m.get("gap_px")
    enemy = _rect(m.get("enemy_rect"))
    seen = _fmt(enemy) if enemy is not None else "its box was not recorded"
    evidence = [f"An enemy ({seen}) was stomped while Mario's feet were still {gap} px above it."]
    # The stomp path: the check that looks the enemy up AND hands it to the stomp.
    places = _lookup_places(level, "enemy_group", "enemy", None,
                            only=(r"adjust_mario_for_y_enemy_collisions",))
    return {"diagnosis": "stomp_without_contact",
            "summary": (f"An enemy was stomped {gap} px below Mario's feet, so the code that "
                        f"decides he landed on an enemy uses something other than their real "
                        f"boxes." if gap is not None else "An enemy was stomped without contact."),
            "evidence": evidence, "places": places,
            "suggestion": ("Decide a stomp only from the sprites' own rectangles (the lookup "
                           "line); remove any extra reach added after it."),
            "basis": BASIS}


_JUMP_STATE = r"self\.state\s*=\s*c\.JUMP"


def _plain_takeoff(player: _Source, st: ast.stmt) -> bool:
    """A statement a take-off consists of in the engine as written: entering
    the JUMP state, or setting the rise speed from the jump constants (also
    inside an if/else that picks between them)."""
    if isinstance(st, ast.Assign):
        text = _segment(player, st)
        return bool(re.fullmatch(_JUMP_STATE, text)
                    or re.fullmatch(r"self\.y_vel\s*=\s*c\.(FAST_)?JUMP_VEL\b[^\n]*", text))
    if isinstance(st, ast.If):
        return all(_plain_takeoff(player, s) for s in [*st.body, *st.orelse])
    return False


def _takeoffs(player: _Source) -> list[dict[str, Any]]:
    """Every take-off: from `self.state = c.JUMP` to the end of that block, in
    each function that sets a jump velocity - and, as suspects, whatever else
    that block does after the take-off itself."""
    places = []
    for fn in player.functions:
        for st in player.statements(fn):
            body = next((getattr(st, f) for f in ("body", "orelse") if any(
                isinstance(s, ast.Assign) and re.fullmatch(_JUMP_STATE, _segment(player, s))
                for s in getattr(st, f, []))), None)
            if not body or not re.search(r"y_vel\s*=\s*c\.(FAST_)?JUMP_VEL", _segment(player, st)):
                continue
            start = next(s for s in body if re.fullmatch(_JUMP_STATE, _segment(player, s)))
            a, b = start.lineno, max((s.end_lineno or s.lineno) for s in body)
            places.append(_place(player, a, b, "Mario takes off here; everything in this block "
                                               "sets the jump's speed", suspect=False))
            places.extend(_removal(player, extra, "runs right after the take-off: the engine's "
                                                  "own take-off is only the lines above")
                          for extra in body[body.index(start) + 1:]
                          if not _plain_takeoff(player, extra))
    return places


def _jump(record: Mapping[str, Any], m: Mapping[str, Any], variant: str, kind: str) -> dict[str, Any] | None:
    player = _source(variant, PLAYER_FILE)
    if player is None:
        return None
    places = _takeoffs(player)
    rising = next((fn for fn in player.functions
                   if re.search(r"self\.gravity\s*=\s*c\.JUMP_GRAVITY", _segment(player, fn.node))), None)
    if kind == "impossible_jump" and m.get("rule") == "rise_speed":
        summary = (f"Mario rose at {m.get('rise_speed')} px/frame; the engine's fastest jump is "
                   f"{m.get('limit')}. Something at take-off makes the jump too strong.")
        evidence = [f"Rise speed {m.get('rise_speed')} px/frame, limit {m.get('limit')} px/frame."]
    else:
        if rising is not None:
            for st in player.statements(rising):
                if isinstance(st, ast.If) and any(
                        re.fullmatch(r"self\.state\s*=\s*c\.FALL", _segment(player, b)) for b in st.body):
                    a, b = _of(st)
                    places.append(_place(player, a, b, "the rise "
                                                       "ends here (the jump turns into a fall)",
                                         suspect=False))
        if kind == "impossible_jump":
            summary = (f"Mario climbed {m.get('climb_px')} px above where he last stood; the "
                       f"fastest jump can climb {m.get('limit')} px.")
            evidence = [f"Climb {m.get('climb_px')} px, limit {m.get('limit')} px."]
        else:
            summary = (f"Mario went above the top of the level (y {m.get('y')}); only a jump can "
                       f"send him up, so the jump is too strong or lasts too long.")
            evidence = [f"y {m.get('y')}, limit {m.get('threshold_y')}."]
    return {"diagnosis": "jump_too_strong", "summary": summary, "evidence": evidence,
            "places": places,
            "suggestion": ("A take-off must use the jump constants unchanged (constants.JUMP_VEL, "
                           "FAST_JUMP_VEL); remove anything in these blocks that makes it stronger."),
            "basis": BASIS}


def _search(src: _Source | None, pattern: str, why: str, limit: int = 4) -> list[dict[str, Any]]:
    if src is None:
        return []
    out = []
    for i, line in enumerate(src.lines, start=1):
        if re.search(pattern, line):
            out.append(_place(src, i, i, why))
            if len(out) == limit:
                break
    return out


def _other(kind: str, m: Mapping[str, Any], variant: str) -> dict[str, Any] | None:
    level, player = _source(variant, LEVEL_FILE), _source(variant, PLAYER_FILE)
    if kind == "below_world":
        return {"diagnosis": "death_plane", "summary": ("Mario is alive below the bottom of the "
                                                       "screen: the rule that kills him there did not act."),
                "evidence": [f"Measured: {dict(m)}."],
                "places": _search(level, r"rect\.y\s*>\s*c\.SCREEN_HEIGHT",
                                  "the check that kills Mario below the screen"),
                "suggestion": "Make sure this check runs on every frame Mario can fall.",
                "basis": BASIS}
    if kind == "speed":
        return {"diagnosis": "speed_cap", "summary": ("Mario moved faster than the engine's own "
                                                     "speed cap."),
                "evidence": [f"Measured: {dict(m)}."],
                "places": _search(player, r"max_x_vel\s*=", "sets Mario's top speed"),
                "suggestion": "Keep every top-speed setting at the engine's constants.",
                "basis": BASIS}
    if kind in ("score_drop", "coin_drop"):
        key = "SCORE" if kind == "score_drop" else "COIN_TOTAL"
        return {"diagnosis": "bookkeeping", "summary": (f"The {'score' if key == 'SCORE' else 'coin total'} "
                                                       f"went down; the game should only ever add to it."),
                "evidence": [f"Measured: {dict(m)}."],
                "places": _search(level, rf"\[c\.{key}\]\s*(-=|=(?!=))", "lowers or overwrites it"),
                "suggestion": "Remove any code that lowers or overwrites it during play.",
                "basis": BASIS}
    return None


def fix_hint(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Where to fix the cause of this incident, or None when there is nothing
    to fix (a synthetic test event) or no lead can be worked out."""
    if record.get("synthetic"):
        return None
    kind = str(record.get("fingerprint", {}).get("kind") or record.get("summary", {}).get("category"))
    metrics = record.get("detector", {}).get("metrics", {}) or {}
    variant = record.get("provenance", {}).get("game", {}).get("variant")
    if not isinstance(variant, str):
        return None
    try:
        if kind.startswith("clip_into_"):
            return _clip(record, metrics, variant)
        if kind == "invisible_collision":
            return _invisible(record, metrics, variant)
        if kind == "hit_without_contact":
            return _hit(record, metrics, variant)
        if kind == "stomp_without_contact":
            return _stomp(record, metrics, variant)
        if kind in ("impossible_jump", "above_world"):
            return _jump(record, metrics, variant, kind)
        return _other(kind, metrics, variant)
    except (KeyError, TypeError, ValueError, StopIteration):
        return None


def headline(hint: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The short form the dashboard shows: the finding and the first place."""
    if not hint:
        return None
    places = list(hint.get("places") or [])
    first = next((p for p in places if p.get("suspect")), places[0] if places else None)
    where = _where(first) if first else None
    # Every concrete edit, in the order a developer would make them.
    fixes = [{"where": _where(p), "fix": fix_text(p)} for p in places if p.get("suspect") and p.get("fix")]
    return {"summary": hint.get("summary"), "where": where, "suggestion": hint.get("suggestion"),
            "fixes": fixes}


def _where(place: Mapping[str, Any]) -> str:
    fn = f" in {place['function'].split('.')[-1]}()" if place.get("function") else ""
    return f"{place['file']} line {place['line']}{fn}"
