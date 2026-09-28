"""dashboard_facts.py: the numbers the dashboard's Overview shows.

Every one must come from the project's own files, and anything that cannot
be read must come back as None or empty - never a guess.
"""
import json
import subprocess
import sys

import pytest

import dashboard_facts as df
from exploration import config
from reporting.evidence import TITLES


def test_the_baseline_path_is_the_evaluation_modules_own():
    from evaluation.completion import BASELINE_PATH
    assert df.BASELINE_PATH == BASELINE_PATH


def test_importing_it_does_not_load_pygame():
    # app.py imports this module before custom_mario_env has set pygame up.
    out = subprocess.run([sys.executable, "-c",
                          "import sys, dashboard_facts; print('pygame' in sys.modules)"],
                         cwd=df.PROJECT_ROOT, capture_output=True, text=True, timeout=60,
                         check=True)
    assert out.stdout.strip() == "False"


def test_the_detectors_are_the_evidence_titles_without_the_synthetic_probe():
    dets = df.detectors()
    assert [d["kind"] for d in dets] == [k for k in TITLES if k != df.SYNTHETIC_KIND]
    assert {d["kind"] for d in dets if d["group"] == "engine"} == df.ENGINE_KINDS
    assert all(d["title"] == TITLES[d["kind"]] for d in dets)


def test_every_declared_benchmark_bug_names_the_detector_that_catches_it():
    from reporting.variants import injected_bugs
    bugs = df.benchmark_bugs()
    assert [b["id"] for b in bugs] == [b["id"] for b in injected_bugs(config.BUGGED_GAME_VARIANT)]
    for b in bugs:
        assert b["kinds"], f"{b['id']} names no detector kind"
        assert set(b["kinds"]) <= set(TITLES), b["kinds"]


def test_a_missing_bug_declaration_gives_no_bugs(monkeypatch, tmp_path):
    monkeypatch.setattr(df, "PROJECT_ROOT", str(tmp_path))
    assert df.benchmark_bugs() == []


def test_the_evidence_counts_real_incidents_per_game_and_the_bugs_they_cover():
    bugs = [{"id": "a", "kinds": ["clip_into_step"]}, {"id": "b", "kinds": ["impossible_jump"]},
            {"id": "c", "kinds": ["hit_without_contact"]}]
    incidents = [
        {"game_variant": "mario_bugged", "category": "clip_into_step", "synthetic": False},
        {"game_variant": "mario_bugged", "category": "clip_into_step", "synthetic": False},
        {"game_variant": "mario_bugged", "category": "impossible_jump", "synthetic": True},
        {"game_variant": "mario_clean", "category": "hit_without_contact", "synthetic": False},
    ]
    runs = [{"game_variant": "mario_clean", "bugs": 0}, {"game_variant": "mario_clean", "bugs": 2},
            {"game_variant": None, "bugs": 0}]
    ev = df.evidence(incidents, runs, bugs)
    assert ev["incidents"] == 3, "a synthetic probe is not a bug"
    assert ev["incidents_per_game"] == {"mario_clean": 1, "mario_bugged": 2}
    # Only the bugged game's incidents count as catching a benchmark bug.
    assert ev["benchmark_bugs_with_evidence"] == ["a"]
    assert ev["benchmark_bugs_declared"] == 3
    assert ev["run_reports"] == 3 and ev["clean_runs_no_bugs"] == 1


def test_the_objectives_read_the_frozen_records(monkeypatch, tmp_path):
    o1 = df.objective1()
    assert o1 is not None and o1["timesteps"] == 6_000_000 and o1["episodes"] == 500
    assert 0 < o1["completion_rate"] < 1
    record = {"approved_on": "2026-09-21", "brain": {"num_timesteps": 16},
              "coverage": {"percent_of_testable": 84.2, "covered_testable": 3,
                           "testable_total": 4},
              "retention": {"episodes": 500, "completed": 283, "completion_rate": 0.566,
                            "baseline_completion_rate": 0.468, "verdict": "HEALTHY"}}
    monkeypatch.setattr(df, "objective2_record", lambda: record)
    o2 = df.objective2()
    assert o2 is not None
    assert (o2["timesteps"], o2["coverage_percent"], o2["verdict"]) == (16, 84.2, "HEALTHY")
    monkeypatch.setattr(df, "objective2_record", lambda: None)
    assert df.objective2() is None, "the final brain is not installed: nothing to show"
    monkeypatch.setattr(df, "BASELINE_PATH", str(tmp_path / "missing.json"))
    assert df.objective1() is None


def test_the_code_census_counts_the_projects_python_not_the_game(monkeypatch):
    monkeypatch.setattr(df, "_census", None)
    census = df.code_census()
    assert census["source_files"] > 0 and census["source_lines"] > census["source_files"]
    assert census["test_files"] > 0
    files = df._python_files()
    assert not [f for f in files if f.split("/", 1)[0] in df.GAME_DIRS]
    assert "dashboard_backend.py" in files


def test_the_census_walks_the_tree_without_git(monkeypatch):
    monkeypatch.setattr(df, "_git", lambda *a: None)
    monkeypatch.setattr(df, "_census", None)
    files = df._python_files()
    assert "dashboard_backend.py" in files and "tests/test_dashboard_facts.py" in files
    assert not [f for f in files if f.split("/", 1)[0] in df.GAME_DIRS or "venv" in f]
    assert df.code_census()["commits"] is None


class _Done:
    def __init__(self, returncode, stdout):
        self.returncode, self.stdout = returncode, stdout


@pytest.mark.parametrize(("returncode", "stdout", "expected"), [
    (0, "tests/test_a.py: 3\ntests/test_b.py: 4\n\n", ("done", 7)),
    (0, "no tests ran\n", ("unavailable", None)),
    (1, "tests/test_a.py: 3\n", ("unavailable", None)),
])
def test_the_test_count_is_pytests_own_collection(monkeypatch, returncode, stdout, expected):
    calls = []

    def fake_run(argv, **kw):
        calls.append((argv, kw))
        return _Done(returncode, stdout)

    monkeypatch.setattr(df.subprocess, "run", fake_run)
    counter = df.TestCount()
    assert counter.state() == {"status": "idle", "count": None}
    counter._run()
    assert (counter.state()["status"], counter.state()["count"]) == expected
    argv, kw = calls[0]
    assert argv[1:4] == ["-m", "pytest", "--collect-only"] and kw["cwd"] == df.PROJECT_ROOT


def test_a_test_count_that_cannot_run_is_unavailable(monkeypatch):
    def broken(*_a, **_kw):
        raise OSError("no python")

    monkeypatch.setattr(df.subprocess, "run", broken)
    counter = df.TestCount()
    counter._run()
    assert counter.state() == {"status": "unavailable", "count": None}


def test_the_test_count_starts_only_once(monkeypatch):
    started = []
    monkeypatch.setattr(df.threading, "Thread",
                        lambda **kw: type("T", (), {"start": lambda _s: started.append(kw)})())
    counter = df.TestCount()
    counter.start()
    counter.start()
    assert len(started) == 1 and counter.state()["status"] == "counting"


def test_the_project_facts_are_one_json_answer(monkeypatch):
    monkeypatch.setattr(df, "code_census", lambda: {"commits": 3})
    facts = df.project_facts(incidents=[], runs=[], brain={"parameters": 1})
    json.dumps(facts)
    assert facts["brain"] == {"parameters": 1}
    assert facts["engineering"]["commits"] == 3 and "status" in facts["engineering"]["tests"]
    assert len(facts["objective3"]["detectors"]) == len(TITLES) - 1
