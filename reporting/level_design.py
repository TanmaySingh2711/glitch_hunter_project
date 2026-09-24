"""Level 1-1 as DESIGNED: where the static solids (ground, pipes, steps) are.

The collision detectors (reporting/collision_invariants.py) must know what the
player SEES as solid, independently of what the running game's colliders say -
otherwise a collider that is wrong (too small, too low, stray) would be its
own alibi. The static solids are painted into the level's background image,
so their place is a fact of the level's design, not of any collider at
runtime. This module holds that design:

    reporting/level1_design.json   every ground, pipe and step rectangle,
                                   extracted from mario_clean (the pinned
                                   baseline) by tools/build_level_design.py

and tests/test_collision_invariants.py proves, on every run, that it is
exactly mario_clean's geometry and that every rectangle is drawn (its
background pixels are not sky). Bricks, ? boxes and enemies are sprites: the
detectors read those live, since a sprite is drawn exactly at its rect.

Nothing here names a bug or a place a bug is: it is the whole level's
geometry, used the same way everywhere.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from typing import Any

DESIGN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "level1_design.json")
DESIGN_FORMAT = 1
# The level's static collider groups (level1.py), and what each one is.
STATIC_GROUPS = (("ground", "ground_group"), ("pipe", "pipe_group"), ("step", "step_group"))

Rect = tuple[int, int, int, int]


def design_from_level(level: Any) -> list[dict[str, Any]]:
    """The static solids of a live Level1, sorted: [{solid, x, y, w, h}]."""
    out = []
    for solid, attr in STATIC_GROUPS:
        for sprite in getattr(level, attr):
            r = sprite.rect
            out.append({"solid": solid, "x": int(r.x), "y": int(r.y), "w": int(r.w), "h": int(r.h)})
    return sorted(out, key=lambda s: (s["solid"], s["x"], s["y"], s["w"], s["h"]))


def load_design(path: str = DESIGN_PATH) -> list[tuple[str, Rect]]:
    """[(solid kind, (x, y, w, h))] from the design file. Refuses a file in
    another format rather than guessing."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, Mapping) or doc.get("format") != DESIGN_FORMAT:
        raise ValueError(f"{path}: not a level design file (format {DESIGN_FORMAT})")
    return [(str(s["solid"]), (int(s["x"]), int(s["y"]), int(s["w"]), int(s["h"])))
            for s in doc["solids"]]


def design_document(solids: Iterable[Mapping[str, Any]], source: Mapping[str, Any]) -> dict[str, Any]:
    return {"format": DESIGN_FORMAT,
            "note": ("Level 1-1's static solids (ground, pipes, steps) as designed and drawn. "
                     "Generated from mario_clean by tools/build_level_design.py; "
                     "tests/test_collision_invariants.py checks it against mario_clean and "
                     "against the drawn background on every run. Do not edit by hand."),
            "source": dict(source), "solids": list(solids)}
