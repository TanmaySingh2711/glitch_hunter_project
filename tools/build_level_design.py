"""Write reporting/level1_design.json from mario_clean: Level 1-1's static
solids (ground, pipes, steps) as designed and drawn.

    python tools/build_level_design.py            # writes it (refuses if unchanged)
    python tools/build_level_design.py --check    # exit 1 if the file is stale

Always reads mario_clean - the pinned baseline - never a variant under test,
so a defective collider in mario_bugged can never become "the design". Re-run
only if mario_clean's level layout changes (its pin would change with it).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.cli import prepare_tool

ROOT = prepare_tool(headless=True)

from custom_mario_env import CustomMarioEnv
from exploration import config
from reporting import level_design, variants


def build() -> dict:
    env = CustomMarioEnv(game_variant=config.CLEAN_GAME_VARIANT)
    env.reset()
    solids = level_design.design_from_level(env.game.state)
    tree = variants.game_tree_sha256(variants.game_dir(config.CLEAN_GAME_VARIANT))
    if tree != config.CLEAN_GAME_TREE_SHA256:
        raise SystemExit("mario_clean does not match its pin; refusing to take the design from it")
    return level_design.design_document(solids, {"variant": config.CLEAN_GAME_VARIANT,
                                                 "game_tree_sha256": tree})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="only report whether the file is current")
    args = ap.parse_args(argv)
    doc = build()
    text = json.dumps(doc, indent=1) + "\n"
    current = None
    if os.path.exists(level_design.DESIGN_PATH):
        with open(level_design.DESIGN_PATH, encoding="utf-8") as fh:
            current = fh.read()
    if args.check:
        print("current" if current == text else "STALE")
        return 0 if current == text else 1
    with open(level_design.DESIGN_PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"wrote {level_design.DESIGN_PATH}: {len(doc['solids'])} solids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
