"""The dashboard in a real browser, end to end: the page, its JavaScript and
the server together, the way a user drives it.

It starts app.py on the bugged game (scratch folders for the evidence), opens
the page in headless Chromium, clicks Start testing, and waits for the first
benchmark bug to stop testing - then checks the Bug found card shows the bug
and where and what to fix, and that the page raised no JavaScript error.

Needs Playwright and its Chromium (`pip install playwright` then
`playwright install chromium`); without them it skips. CI runs it in the
"Dashboard in a browser" job. GLITCH_HUNTER_PYTHON picks the interpreter that
runs app.py (default: this one).
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 5063
PILL = "document.querySelector('#system-pill .pill__text').textContent"


def _status(base):
    with urllib.request.urlopen(base + "/api/status", timeout=5) as r:
        return json.load(r)


@pytest.fixture(scope="module")
def dashboard(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("e2e")
    env = dict(os.environ, GLITCH_HUNTER_PORT=str(PORT), SDL_VIDEODRIVER="dummy",
               SDL_AUDIODRIVER="dummy", PYTHONUTF8="1")
    python = os.environ.get("GLITCH_HUNTER_PYTHON", sys.executable)
    app = subprocess.Popen([python, "app.py", "--game", "mario_bugged",
                            "--incidents-dir", str(scratch / "incidents"),
                            "--run-reports-dir", str(scratch / "runs")],
                           cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{PORT}"
    try:
        for _ in range(180):
            time.sleep(1)
            try:
                if _status(base).get("brain_path"):
                    break
            except OSError:
                continue
        else:
            pytest.fail("the dashboard did not come up within 3 minutes")
        yield base
    finally:
        app.terminate()
        try:
            app.wait(30)
        except subprocess.TimeoutExpired:
            app.kill()
            app.wait(30)
            pytest.fail("the dashboard ignored a request to stop (SIGTERM)")


def test_a_bug_stops_testing_and_the_card_says_where_and_what_to_fix(dashboard):
    with sync_api.sync_playwright() as p:
        # PLAYWRIGHT_CHANNEL=msedge uses an installed Edge instead of Chromium.
        browser = p.chromium.launch(channel=os.environ.get("PLAYWRIGHT_CHANNEL") or None)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(dashboard + "/#live")
        page.wait_for_function(f"{PILL} === 'Ready'", timeout=120_000)
        assert page.inner_text("#monitor-empty").strip().startswith("NO SIGNAL")
        page.click("#start-btn")
        page.wait_for_function(f"{PILL} === 'Bug found'", timeout=300_000)
        page.wait_for_selector("#result-card .fix", timeout=60_000)
        card = page.inner_text("#result-card")
        fix = page.inner_text("#result-card .fix")
        browser.close()
    assert "BUG FOUND" in card.upper()
    assert "WHERE TO FIX IT" in fix.upper()
    assert "mario_bugged/data/" in fix and ("Delete line" in fix or "Change line" in fix)
    status = _status(dashboard)
    assert status["bug_found"] and status["brain_approved"] is True
    assert not errors, errors
