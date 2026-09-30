"""The facts the dashboard's Overview, How It Works and Bug History pages show.

Every number is read from the project's own files - the frozen Objective-2
closure record, the 6M retention baseline, the declared benchmark bugs, the
incident and run-report stores, the source tree and git - never typed in.
Anything that cannot be read is sent as None and the page leaves it out; it
is never guessed.

The slow parts (the source-tree census, git, the automated-test count) are
worked out once per dashboard process and never on the game thread. The test
count is pytest's own collection, run in a low-priority child process in the
background (about 10 s); until it finishes the page says "counting".
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
from collections.abc import Iterable, Mapping
from typing import Any

from exploration import config
from reporting.provenance import objective2_record
from reporting.variants import BUG_MANIFEST_NAME

log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# evaluation.completion.BASELINE_PATH, not imported from there: that module
# loads pygame, and app.py imports this module before custom_mario_env has set
# pygame's environment up - so nothing here may load pygame at import time
# (tests/test_dashboard_facts.py pins both).
BASELINE_PATH = os.path.join(PROJECT_ROOT, "evaluation", "completion_baseline_6M.json")
# The two game copies are vendored third-party code, not this project's work.
GAME_DIRS = tuple(config.GAME_VARIANTS)
# The detectors custom_mario_env._detect_glitches runs itself; every other
# real detector is a collision or jump-physics invariant
# (reporting/collision_invariants.py).
ENGINE_KINDS = frozenset(("below_world", "above_world", "speed", "score_drop", "coin_drop"))
SYNTHETIC_KIND = "synthetic_probe"
_KIND_RE = re.compile(r"kind '([a-z_]+)'")
_COLLECTED_RE = re.compile(r"^tests[/\\]\S+\.py: (\d+)$", re.MULTILINE)
TEST_COUNT_TIMEOUT_S = 180.0


def _read_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ── the three objectives ──────────────────────────────────────────────────
def objective1() -> dict[str, Any] | None:
    """The 6M brain's measured result: the frozen 500-episode baseline."""
    base = _read_json(BASELINE_PATH)
    if base is None:
        return None
    summary = base.get("summary", {})
    return {"timesteps": base.get("checkpoint", {}).get("num_timesteps"),
            "episodes": summary.get("episodes"), "completed": summary.get("completed"),
            "completion_rate": summary.get("completion_rate"),
            "mean_progress": summary.get("progress", {}).get("mean")}


def objective2() -> dict[str, Any] | None:
    """The final QA brain as approved: FINAL_OBJECTIVE2.json (installed with it)."""
    record = objective2_record()
    if record is None:
        return None
    cov, ret = record.get("coverage", {}), record.get("retention", {})
    return {"timesteps": record.get("brain", {}).get("num_timesteps"),
            "coverage_percent": cov.get("percent_of_testable"),
            "covered": cov.get("covered_testable"), "reachable": cov.get("testable_total"),
            "episodes": ret.get("episodes"), "completed": ret.get("completed"),
            "completion_rate": ret.get("completion_rate"),
            "completion_ci95": ret.get("completion_rate_ci95"),
            "mean_progress": ret.get("mean_progress"),
            "baseline_completion_rate": ret.get("baseline_completion_rate"),
            "verdict": ret.get("verdict"), "approved_on": record.get("approved_on")}


def detectors() -> list[dict[str, str]]:
    """Every real detector the dashboard runs, grouped as the code groups them."""
    from reporting.evidence import TITLES  # loads pygame: see BASELINE_PATH
    return [{"kind": kind, "title": title,
             "group": "engine" if kind in ENGINE_KINDS else "collision"}
            for kind, title in TITLES.items() if kind != SYNTHETIC_KIND]


def benchmark_bugs() -> list[dict[str, Any]]:
    """The bugs declared in games/mario_bugged/INJECTED_BUGS.json, with the detector
    kinds each one's declaration says catch it."""
    manifest = _read_json(os.path.join(PROJECT_ROOT, config.GAMES_DIR, config.BUGGED_GAME_VARIANT,
                                       BUG_MANIFEST_NAME))
    if manifest is None:
        return []
    return [{"id": bug.get("id"), "number": bug.get("number"), "name": bug.get("name"),
             "summary": bug.get("summary"),
             "kinds": _KIND_RE.findall(str(bug.get("objective3_detection", "")))}
            for bug in manifest.get("bugs", [])]


def evidence(incidents: Iterable[Mapping[str, Any]], runs: Iterable[Mapping[str, Any]],
             bugs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """What the evidence on disk shows: real (non-synthetic) incidents per
    game, which declared benchmark bugs have an incident of their own kind on
    the bugged game, and the clean-run reports."""
    real = [i for i in incidents if not i.get("synthetic")]
    per_game = {v: sum(1 for i in real if i.get("game_variant") == v)
                for v in config.GAME_VARIANTS}
    bugged_kinds = {i.get("category") for i in real
                    if i.get("game_variant") == config.BUGGED_GAME_VARIANT}
    bug_list = list(bugs)
    caught = [b["id"] for b in bug_list if set(b.get("kinds") or ()) & bugged_kinds]
    run_list = list(runs)
    clean_runs = [r for r in run_list if r.get("game_variant") == config.CLEAN_GAME_VARIANT]
    return {"incidents": len(real), "incidents_per_game": per_game,
            "benchmark_bugs_with_evidence": caught, "benchmark_bugs_declared": len(bug_list),
            "run_reports": len(run_list),
            "clean_runs_no_bugs": sum(1 for r in clean_runs if r.get("bugs") == 0)}


# ── the engineering behind it ─────────────────────────────────────────────
def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", PROJECT_ROOT, *args],  # noqa: S603, S607 (fixed argv)
                             capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _python_files() -> list[str]:
    """The project's own Python files (the vendored game copies are not)."""
    listed = _git("ls-files", "*.py")
    if listed is not None:
        names = [n.strip() for n in listed.splitlines() if n.strip()]
    else:                                 # not a git checkout: walk the tree
        skip = {"venv_gpu", ".venv", ".git", "__pycache__", ".mypy_cache", ".ruff_cache",
                ".pytest_cache", *GAME_DIRS}
        names = []
        for folder, dirs, files in os.walk(PROJECT_ROOT):
            dirs[:] = [d for d in dirs if d not in skip]
            names += [os.path.relpath(os.path.join(folder, f), PROJECT_ROOT).replace(os.sep, "/")
                      for f in files if f.endswith(".py")]
    return [n for n in names if n.split("/", 1)[0] not in GAME_DIRS]


def _lines(name: str) -> int:
    try:
        with open(os.path.join(PROJECT_ROOT, name), encoding="utf-8", errors="replace") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


_census: dict[str, Any] | None = None
_census_lock = threading.Lock()


def code_census() -> dict[str, Any]:
    """Files and non-blank lines of the project's Python, and git commits.
    Worked out once per process."""
    global _census
    with _census_lock:
        if _census is None:
            files = _python_files()
            tests = [f for f in files if f.startswith("tests/")]
            source = [f for f in files if not f.startswith("tests/")]
            commits = _git("rev-list", "--count", "HEAD")
            _census = {"source_files": len(source), "source_lines": sum(map(_lines, source)),
                       "test_files": len(tests), "test_lines": sum(map(_lines, tests)),
                       "commits": int(commits) if commits and commits.strip().isdigit() else None}
        return dict(_census)


class TestCount:
    """pytest's own count of the automated tests, collected once in the
    background: status 'idle' -> 'counting' -> 'done' (count set) or
    'unavailable' (pytest not installed, or collection failed)."""

    __test__ = False                      # a helper, not a pytest test class

    def __init__(self) -> None:
        self.status = "idle"
        self.count: int | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self.status != "idle":
                return
            self.status = "counting"
        threading.Thread(target=self._run, name="test-count", daemon=True).start()

    def _run(self) -> None:
        flags = 0x4000 if sys.platform == "win32" else 0     # BELOW_NORMAL_PRIORITY_CLASS
        counts: list[int] = []
        try:
            out = subprocess.run(
                [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=TEST_COUNT_TIMEOUT_S,
                check=False, creationflags=flags)
            if out.returncode == 0:
                counts = [int(n) for n in _COLLECTED_RE.findall(out.stdout)]
        except (OSError, subprocess.SubprocessError):
            log.debug("could not count the tests", exc_info=True)
        with self._lock:
            if counts:
                self.count, self.status = sum(counts), "done"
            else:
                self.status = "unavailable"

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {"status": self.status, "count": self.count}


TESTS = TestCount()


def project_facts(incidents: Iterable[Mapping[str, Any]], runs: Iterable[Mapping[str, Any]],
                  brain: Mapping[str, Any]) -> dict[str, Any]:
    """Everything the Overview page shows, in one answer."""
    bugs = benchmark_bugs()
    return {"brain": dict(brain),
            "objective1": objective1(), "objective2": objective2(),
            "objective3": {"detectors": detectors(), "benchmark_bugs": bugs,
                           "evidence": evidence(incidents, runs, bugs)},
            "engineering": {**code_census(), "tests": TESTS.state()}}
