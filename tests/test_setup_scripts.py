"""setup.bat and setup.sh stay in step with the project they install.

CI runs both scripts for real on clean machines (the "One-click setup" job);
these checks catch drift in every test run: the torch pin, the brain files
the dashboard needs, and the details that make a batch file run at all."""
import os
import re
import tomllib

import pytest

from tools import final_brain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name):
    with open(os.path.join(ROOT, name), "rb") as fh:
        return fh.read()


def _pinned(package):
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
        deps = tomllib.load(fh)["project"]["dependencies"]
    return next(d for d in deps if d.startswith(f"{package}=="))


@pytest.mark.parametrize("script", ["setup.bat", "setup.sh"])
def test_the_script_installs_the_pinned_torch(script):
    text = _read(script).decode("ascii")
    assert re.findall(r"torch==[\d.]+", text) == [_pinned("torch")]
    assert "-r requirements.txt" in text


@pytest.mark.parametrize("script", ["setup.bat", "setup.sh"])
def test_the_script_checks_every_file_the_dashboard_needs(script):
    text = _read(script).decode("ascii").replace("\\", "/")
    for name in final_brain.FILES:
        assert name in text, name
    assert "tools/final_brain.py install" in text


def test_the_batch_file_runs_on_any_windows():
    data = _read("setup.bat")
    data.decode("ascii")                               # no characters cmd.exe could misread
    assert data.count(b"\n") == data.count(b"\r\n")    # CRLF: .gitattributes keeps it so
    assert b"norun" in data and b"call run_dashboard.bat" in data


def test_the_shell_script_is_lf_and_strict():
    data = _read("setup.sh")
    assert b"\r" not in data
    assert data.startswith(b"#!/usr/bin/env bash\n") and b"set -euo pipefail" in data
