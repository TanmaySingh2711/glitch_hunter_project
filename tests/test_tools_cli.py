"""Every script in tools/ starts, parses its arguments and documents itself.

Most tools replay the real game for minutes against git-ignored artifacts,
so their full runs are not tests. What is tested is what breaks silently when
the modules under them move: the imports, the shared start-up
(common/cli.py), and the argument parser - each tool run as the README says,
`python tools/<name>.py`, from a different working directory.
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = sorted(f for f in os.listdir(os.path.join(ROOT, "tools"))
               if f.endswith(".py") and not f.startswith("_"))


def test_every_tool_is_listed_in_the_tools_readme():
    with open(os.path.join(ROOT, "tools", "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    missing = [t for t in TOOLS if f"`{t}`" not in readme]
    assert not missing, f"undocumented tools: {missing}"


@pytest.mark.parametrize("tool", TOOLS)
def test_the_tool_starts_and_prints_its_usage(tool, tmp_path):
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy",
               PYTHONUTF8="1")
    result = subprocess.run([sys.executable, os.path.join(ROOT, "tools", tool), "--help"],
                            cwd=tmp_path, env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=120, check=False)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage:" in result.stdout
    # --help must never do the tool's work: nothing may appear in the cwd.
    assert not any(tmp_path.iterdir()), f"--help wrote {list(tmp_path.iterdir())}"


def test_check_runs_the_same_gates_as_ci():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    try:
        import check
    finally:
        sys.path.pop(0)
    fast = [name for name, _cmd in check.gates(full=False)]
    assert fast == ["lint", "types", "types (linux)", "tests", "artifacts"]
    assert dict(check.gates(full=False))["types (linux)"][-2:] == ["--platform", "linux"]
    fast_tests = dict(check.gates(full=False))["tests"]
    full_tests = dict(check.gates(full=True))["tests"]
    assert "not slow" in fast_tests
    assert f"--cov-fail-under={check.COVERAGE_FLOOR}" in full_tests
