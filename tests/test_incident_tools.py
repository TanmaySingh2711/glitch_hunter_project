"""tools/incidents.py and tools/validate_incident_pipeline.py.

The maintenance tool must only READ for list/show/verify, catch a tampered
bundle, and re-render without ever replacing a file. The end-to-end
validation runs the real dashboard stack with the approved brain, so it is
marked slow and skips where that brain is not in the checkout.
"""
import importlib.util
import json
import os
import stat

import pytest
from incident_helpers import context, detection

from exploration import config
from reporting.pipeline import IncidentPipeline
from reporting.store import IncidentStore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_tool(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "tools", f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def filled(tmp_path):
    store = IncidentStore(str(tmp_path / "incidents"))
    pipe = IncidentPipeline(store, reproduce=False)
    ids = []
    for kind, x in (("speed", 1000), ("below_world", 3000)):
        det = detection(kind=kind, x=x)
        ids.append(pipe.capture(det, context(det)).incident_id)
    pipe.wait_idle(120)
    pipe.close()
    return store, ids


def _snapshot(root):
    return {os.path.join(d, f): (os.stat(os.path.join(d, f)).st_size,
                                 os.stat(os.path.join(d, f)).st_mtime_ns)
            for d, _dirs, files in os.walk(root) for f in files}


def test_list_show_and_verify_only_read(filled, capsys):
    store, ids = filled
    tool = _load_tool("incidents")
    before = _snapshot(store.root)
    assert tool.main(["--dir", store.root, "list"]) == 0
    out = capsys.readouterr().out
    assert all(i in out for i in ids)
    assert tool.main(["--dir", store.root, "show", ids[0]]) == 0
    assert "report.pdf" in capsys.readouterr().out
    assert tool.main(["--dir", store.root, "verify"]) == 0
    assert "2 incident(s), 0 with problems" in capsys.readouterr().out
    assert _snapshot(store.root) == before, "a read-only command wrote to the store"


def test_verify_catches_a_tampered_file(filled, capsys):
    store, ids = filled
    path = os.path.join(store.bundle_dir(ids[1]), "report.md")
    os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\nedited after the fact\n")
    assert _load_tool("incidents").main(["--dir", store.root, "verify"]) == 1
    assert "report.md: SHA-256 differs" in capsys.readouterr().out


def test_rerender_adds_a_version_and_says_why(filled):
    store, ids = filled
    bundle = store.bundle_dir(ids[0])
    with open(os.path.join(bundle, "report.pdf"), "rb") as fh:
        original = fh.read()
    assert _load_tool("incidents").main(
        ["--dir", store.root, "rerender", ids[0], "--reason", "template fix", "--what",
         "report.pdf"]) == 0
    with open(os.path.join(bundle, "report.pdf"), "rb") as fh:
        assert fh.read() == original
    assert os.path.isfile(os.path.join(bundle, "report.v2.pdf"))
    manifest = store.read_manifest(bundle)
    assert "report.v2.pdf" in manifest["artifacts"]
    assert any("re-rendered as report.v2.pdf: template fix" in h["event"]
               for h in manifest["history"])
    assert not os.access(os.path.join(bundle, "manifest.json"), os.W_OK), "left unlocked"
    assert store.verify(ids[0]) == []


def test_an_unknown_incident_is_an_error_not_a_crash(filled, capsys):
    store, _ids = filled
    assert _load_tool("incidents").main(
        ["--dir", store.root, "show", "INC-20260923-120000-abcdef"]) == 2
    assert "no such incident" in capsys.readouterr().out


@pytest.mark.slow
def test_the_end_to_end_validation_passes(tmp_path):
    """The real stack, headless: the approved brain on the clean game.

    In its OWN process, as it is run for real. In-process it would share this
    test session's pygame display - and its final Reset closes the game
    window, which took the display away from every test after it (measured:
    118 'video system not initialized' errors in one full run)."""
    import subprocess
    import sys
    if not os.path.exists(os.path.join(ROOT, config.FINAL_BRAIN_PATH)):
        pytest.skip("the approved Objective-2 brain is not in this checkout")
    out = tmp_path / "e2e"
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy", PYTHONUTF8="1")
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "validate_incident_pipeline.py"),
         "--out", str(out)], cwd=ROOT, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=900, check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    with open(out / "validation_report.json", encoding="utf-8") as fh:
        report = json.load(fh)
    assert report["passed"] and len(report["checks"]) >= 30
    assert report["kind"].startswith("SYNTHETIC")
