#!/usr/bin/env bash
# ====================================================================
#  Glitch Hunter - one-command setup for Linux and macOS.
#  Run "bash setup.sh" in the project folder: it installs everything, then
#  starts the dashboard. Safe to run again: it only adds what is missing.
#
#    bash setup.sh          install (small CPU build of PyTorch), then run
#    bash setup.sh gpu      install the NVIDIA GPU build instead (Linux, for training)
#    bash setup.sh norun    install only, do not start the dashboard (CI uses it)
# ====================================================================
set -euo pipefail
cd "$(dirname "$0")"

TORCH="torch==2.14.0"
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
RUN=1
for arg in "$@"; do
    case "$arg" in
        gpu) TORCH_INDEX="https://download.pytorch.org/whl/cu126" ;;
        norun) RUN=0 ;;
    esac
done

echo "[1/5] Looking for Python 3.12 ..."
PY=""
for candidate in python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' 2>/dev/null; then
        PY="$candidate"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "Python 3.12 is needed. Install it, then run bash setup.sh again:"
    echo "  Ubuntu/Debian:  sudo apt install python3.12 python3.12-venv"
    echo "  macOS:          brew install python@3.12"
    exit 1
fi
echo "      Using $PY"

echo "[2/5] Creating the environment (venv_gpu) ..."
[ -x venv_gpu/bin/python ] || "$PY" -m venv venv_gpu
VPY=venv_gpu/bin/python
# Bytecode caches go to generated/, never next to the source (tools/pycache_hook.py).
"$VPY" tools/pycache_hook.py >/dev/null
"$VPY" -m pip install --quiet --upgrade pip

echo "[3/5] Installing PyTorch (the biggest download, please wait) ..."
if [ "$(uname)" = "Darwin" ]; then
    "$VPY" -m pip install --quiet "$TORCH"                  # macOS builds come from PyPI
else
    "$VPY" -m pip install --quiet "$TORCH" --index-url "$TORCH_INDEX"
fi

echo "[4/5] Installing the other libraries ..."
"$VPY" -m pip install --quiet -r requirements.txt

echo "[5/5] Checking the AI brain files ..."
MISSING=""
for f in glitch_hunter_main_brain.zip glitch_hunter_main_brain_coverage.npz \
         checkpoints_qa/final_objective2_16000000/FINAL_OBJECTIVE2.json \
         exploration_data/reachable_mask.npz; do
    [ -f "$f" ] || MISSING=1
done
if [ -n "$MISSING" ]; then
    echo "      Some brain files are missing. Downloading them ..."
    "$VPY" tools/final_brain.py install
fi
if ! "$VPY" -c "import cv2" >/dev/null 2>&1; then
    echo "OpenCV needs the system OpenGL library. Run this, then bash setup.sh again:"
    echo "  sudo apt-get install -y libgl1"
    exit 1
fi
"$VPY" tools/check_environment.py
"$VPY" app.py --help >/dev/null

echo
echo "Setup finished. Next time, start the dashboard with:  venv_gpu/bin/python app.py"
echo
if [ "$RUN" = "1" ]; then
    ( sleep 5; xdg-open http://localhost:5000 >/dev/null 2>&1 || open http://localhost:5000 >/dev/null 2>&1 || true ) &
    exec "$VPY" app.py
fi
