"""Every quality gate, in one command - the same ones CI runs.

    python tools/check.py           # lint, types, fast tests, artifact hashes
    python tools/check.py --full    # ... with the slow tests and a coverage floor

Stops at the first gate that fails and exits non-zero, so it can sit in
front of a commit or a training launch. Each gate is the plain command it
prints, so a failure can be re-run and investigated on its own.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The floor --full enforces on first-party line+branch coverage. Set just
# under the measured level so a change that drops tests fails loudly, while
# ordinary churn does not. Raise it when coverage rises; never lower it to
# make a change pass.
COVERAGE_FLOOR = 90


def gates(full: bool) -> list[tuple[str, list[str]]]:
    py = sys.executable
    tests = [py, "-m", "pytest", "-p", "no:cacheprovider"]
    if full:
        tests += ["--cov", f"--cov-fail-under={COVERAGE_FLOOR}"]
    else:
        tests += ["-m", "not slow"]
    # mypy judges `sys.platform` checks for the platform it runs on. CI type-
    # checks on Linux while development happens on Windows, so both are run
    # here - a Windows-only name outside its platform guard fails locally,
    # not first on CI.
    return [
        ("lint", [py, "-m", "ruff", "check", "."]),
        ("types", [py, "-m", "mypy"]),
        ("types (linux)", [py, "-m", "mypy", "--platform", "linux"]),
        ("tests", tests),
        ("artifacts", [py, os.path.join("tools", "verify_artifacts.py")]),
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument("--full", action="store_true",
                    help="include the slow tests and enforce the coverage floor")
    args = ap.parse_args(argv)
    for name, cmd in gates(args.full):
        print(f"\n=== {name}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        code = subprocess.call(cmd, cwd=ROOT)  # noqa: S603  (fixed argv, built above)
        print(f"=== {name}: {'passed' if code == 0 else f'FAILED ({code})'} "
              f"in {time.time() - t0:.0f}s", flush=True)
        if code != 0:
            return code
    print("\nall gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
