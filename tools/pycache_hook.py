"""Makes this project's Python environment keep its bytecode in
generated/cache/pycache/, whatever starts Python: run_dashboard.bat, a tool,
bare pytest, an editor's "Run file", or `python -m dashboard.desktop` typed by hand.

    python tools/pycache_hook.py           install the hook into the running venv
    python tools/pycache_hook.py --check   say whether it is installed (exit 1 if not)

The hook is one line in a .pth file in the venv's site-packages. Python runs
such a line at start-up, before any other import, so no __pycache__ folder can
appear next to the project's source. setup.bat and setup.sh install it right
after they create venv_gpu. It only ever writes inside a virtual environment:
outside one it refuses, so the system Python is never changed.

Why it exists: entry points used to set sys.pycache_prefix themselves
(common/__init__.py), which covers only what goes through them. On 2026-10-07
a quick `import desktop` from a scratch script left __pycache__/ in the
project root, and bare pytest always left tests/__pycache__/ (pytest compiles
conftest.py before conftest can set the prefix). The in-code lines stay, for a
Python without the hook (CI's runners, a system Python).
"""
import argparse
import os
import sys
import sysconfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "generated", "cache", "pycache")
PTH_NAME = "glitch_hunter_pycache.pth"


def pth_line(cache: str = CACHE) -> str:
    """The start-up line. It keeps a prefix already set (PYTHONPYCACHEPREFIX)."""
    return f"import sys; sys.pycache_prefix = sys.pycache_prefix or {cache!r}\n"


def site_dir() -> str:
    return sysconfig.get_paths()["purelib"]


def in_venv() -> bool:
    return sys.prefix != sys.base_prefix


def installed(site: str, cache: str = CACHE) -> bool:
    try:
        with open(os.path.join(site, PTH_NAME), encoding="utf-8") as fh:
            return fh.read() == pth_line(cache)
    except OSError:
        return False


def install(site: str, cache: str = CACHE) -> str:
    path = os.path.join(site, PTH_NAME)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(pth_line(cache))
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="only say whether it is installed")
    args = ap.parse_args(argv)
    if not in_venv():
        print("Not inside a virtual environment: nothing changed. Run it with "
              "venv_gpu's Python (setup.bat and setup.sh do).")
        return 1
    site = site_dir()
    if args.check:
        ok = installed(site)
        print(f"{'installed' if ok else 'NOT installed'}: {os.path.join(site, PTH_NAME)}")
        return 0 if ok else 1
    print("bytecode goes to", CACHE, "- hook:", install(site))
    return 0


if __name__ == "__main__":
    sys.exit(main())
