"""The six benchmark bugs, end to end through Objective 3 (slow: needs the
approved brain).

One dashboard episode per game (tests/bug_incident_run.py): the approved brain
plays greedily, every built-in detector is live, and each detection becomes a
full incident - evidence, GIF, Markdown and PDF reports, and a replay in a
separate process. On mario_bugged every bug must come out as its own incident,
replayed exactly; on mario_clean nothing at all may be reported.
"""
import json
import os
import subprocess
import sys

import pytest

from exploration import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, config.FINAL_BRAIN_PATH)),
                                 reason="the approved brain is not in this checkout")]

# One incident per bug, from the generic detectors; the stair bug is on two
# columns (two sites), and the sky jump also crosses the older above_world line.
EXPECTED = {"impossible_jump": 1, "above_world": 1, "clip_into_pipe": 1, "invisible_collision": 1,
            "clip_into_block": 1, "clip_into_step": 2, "hit_without_contact": 1}


def _run(variant, tmp_path):
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", PYTHONUTF8="1")
    result = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "bug_incident_run.py"),
                             variant, str(tmp_path / variant)], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=1500, check=False)
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_every_bug_becomes_a_reproduced_reported_incident(tmp_path):
    out = _run(config.BUGGED_GAME_VARIANT, tmp_path)
    found: dict[str, int] = {}
    for inc in out["incidents"]:
        found[inc["kind"]] = found.get(inc["kind"], 0) + 1
        assert not inc["synthetic"] and inc["variant"] == config.BUGGED_GAME_VARIANT
        assert inc["reproduction"] == "reproduced", inc
        assert all(inc["renders"][n] == "done" for n in ("context.gif", "report.md", "report.pdf")), inc
        assert inc["verify"] == [], inc
    assert found == EXPECTED


def test_the_clean_game_reports_nothing(tmp_path):
    assert _run(config.CLEAN_GAME_VARIANT, tmp_path)["incidents"] == []
