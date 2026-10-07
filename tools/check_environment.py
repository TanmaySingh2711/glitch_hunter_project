"""Checks that the libraries the dashboard needs can be loaded, and says in
plain words what to do when one cannot.

    python tools/check_environment.py

Silent and exit code 0 when everything loads. Otherwise it prints what is
wrong and how to fix it, and exits 1. run_dashboard.bat runs it when the
dashboard stops with an error, and setup.bat / setup.sh run it before they
start the dashboard, so a broken machine gets an answer instead of a
traceback.

Why it exists: on 2026-10-07 run_dashboard.bat stopped working with
"WinError 4551: An Application Control policy has blocked this file ...
torch_python.dll". Windows' Smart App Control had started refusing PyTorch's
unsigned DLL. Nothing in the project had changed (a bare `import torch`
failed too), and the traceback gave no hint what to do.
"""
import argparse
import importlib
import os
import sys

sys.pycache_prefix = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "generated", "cache", "pycache")

# (import name, why the dashboard needs it). torch first: it is the one that breaks.
REQUIRED = (
    ("torch", "the agent's brain"),
    ("stable_baselines3", "loads the brain"),
    ("gymnasium", "the game as an environment"),
    ("numpy", "numbers"),
    ("pygame", "the game"),
    ("cv2", "video frames (OpenCV)"),
    ("PIL", "GIFs (Pillow)"),
    ("fpdf", "PDF reports"),
    ("flask", "the dashboard server"),
    ("flask_socketio", "the live connection"),
)

SMART_APP_CONTROL = """\
Windows is blocking PyTorch.
{what}

This is Windows' Smart App Control, a security setting. It refuses files that
are not signed by a company it knows, and PyTorch's torch_python.dll is one of
them. Nothing is wrong with the project, and a reinstall will not help.

To fix it, turn Smart App Control off:
  1. Start > Settings > Privacy & security > Windows Security
  2. App & browser control > Smart App Control settings
  3. Choose "Off"
  4. Double-click run_dashboard.bat again.

Windows may not let you turn it back on later without resetting the PC, so
decide whether you are happy with that first. (Windows records each block in
Event Viewer > Applications and Services Logs > Microsoft > Windows >
CodeIntegrity > Operational.)
"""

MISSING_LIBRARY = """\
The library "{name}" ({why}) is missing or damaged:
  {error}

Run setup.bat (Windows) or `bash setup.sh` (Linux and macOS) again. It only
installs what is missing.
"""

OTHER_ERROR = """\
The library "{name}" ({why}) could not be loaded:
  {error}

If a security program or Windows policy is mentioned above, allow that file
and try again. Otherwise run setup.bat (Windows) or `bash setup.sh` (Linux
and macOS) again; it only installs what is missing.
"""


def blocked_by_application_control(error: BaseException) -> bool:
    """True for Windows refusing a file on policy grounds (WinError 4551 and
    the other Application Control / Smart App Control messages)."""
    text = str(error).lower()
    return (getattr(error, "winerror", None) == 4551 or "winerror 4551" in text
            or "application control policy" in text or "smart app control" in text)


def diagnose(name: str, why: str, error: BaseException) -> str:
    """The advice for one library that failed to load."""
    one_line = " ".join(str(error).split())
    if blocked_by_application_control(error):
        # Windows' own sentence, without the long file path after it.
        said = one_line.split(" Error loading")[0].replace("[WinError 4551] ", "")
        return SMART_APP_CONTROL.format(what=f"Windows says: {said[:120]}")
    if isinstance(error, ModuleNotFoundError):
        return MISSING_LIBRARY.format(name=name, why=why, error=one_line[:300])
    return OTHER_ERROR.format(name=name, why=why, error=one_line[:300])


def check(required: tuple[tuple[str, str], ...] = REQUIRED) -> str | None:
    """None when every library loads, else the advice for the first one that
    does not."""
    for name, why in required:
        try:
            importlib.import_module(name)
        except Exception as error:      # any load failure is the answer
            return diagnose(name, why, error)
    return None


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0]).parse_args(argv)
    advice = check()
    if advice is None:
        return 0
    print()
    print("=" * 70)
    print(advice)
    print("=" * 70)
    return 1


if __name__ == "__main__":
    sys.exit(main())
