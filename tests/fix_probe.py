"""Runs tests/bug_probe.py's scenarios on a PATCHED copy of the bugged game.

    python tests/fix_probe.py ROOT

ROOT holds a copy of games/mario_bugged/ (at ROOT/games/mario_bugged) with the fixes "Where to fix it" suggests
applied (tests/test_fix_hint.py makes it). The game is loaded from that copy
instead of the project's, in this fresh process; everything else - the env,
the detectors, the scenarios - is the project's own. Prints the same JSON as
bug_probe.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import custom_mario_env

custom_mario_env.PROJECT_ROOT = os.path.abspath(sys.argv[1])

import bug_probe

if __name__ == "__main__":
    bug_probe.main("mario_bugged")
