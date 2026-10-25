#!/usr/bin/env bash
# ====================================================================
#  Glitch Hunter - start the dashboard on Linux and macOS.
#  Run "bash run_dashboard.sh" in the project folder (after bash setup.sh).
#  The browser opens http://localhost:5000 the moment the dashboard is ready;
#  the page is served within about a second and the AI loads behind it.
#  To stop the dashboard press Esc in this terminal (one key, no Enter), or
#  press Esc on the dashboard page and choose Yes. Ctrl+C works too.
#  Extra arguments go to app.py, for example:
#
#    bash run_dashboard.sh --game mario_bugged
# ====================================================================
set -euo pipefail
cd "$(dirname "$0")"
# No __pycache__ folders in the project (see common/__init__.py).
export PYTHONPYCACHEPREFIX="$PWD/generated/cache/pycache"

VPY=venv_gpu/bin/python
if [ ! -x "$VPY" ]; then
    echo "venv_gpu was not found in $PWD."
    echo "Create it first with:  bash setup.sh   (see \"Installation\" in README.md)"
    exit 1
fi

# The dashboard reads Esc one key at a time and puts the terminal back as it
# was when it stops; the settings saved here put it back even if it crashes.
TERMINAL=$(stty -g 2>/dev/null || true)

# Opening the browser runs alongside the dashboard, so nothing waits for it.
# It waits for the server to answer, opens the page and exits; if the dashboard
# stops first, it is stopped too.
"$VPY" -m dashboard.desktop open-dashboard &
OPENER=$!
finish() {
    kill "$OPENER" 2>/dev/null || true
    if [ -n "$TERMINAL" ]; then stty "$TERMINAL" 2>/dev/null || true; fi
}
trap finish EXIT

status=0
# ${@+"$@"}: with no arguments, the bash 3.2 that macOS ships fails on a bare "$@" under set -u.
"$VPY" app.py --desktop ${@+"$@"} || status=$?
# A normal stop (Esc, Ctrl+C, Stop on the page) ends here quietly.
[ "$status" -eq 0 ] && exit 0
# A stop with an error: say in plain words what is wrong (a library that will
# not load, for example), instead of leaving only a traceback. Silent when
# everything loads (tools/check_environment.py).
"$VPY" tools/check_environment.py || true
echo
echo "The dashboard stopped with an error (see above)."
exit "$status"
