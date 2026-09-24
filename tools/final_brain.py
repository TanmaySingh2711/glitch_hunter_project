"""Install (or bundle) the final Objective-2 brain: the approved 16M QA brain
and the three files the dashboard needs with it.

    python tools/final_brain.py install                  # download from the GitHub Release
    python tools/final_brain.py install --zip FILE.zip   # or use a bundle you downloaded
    python tools/final_brain.py bundle OUT.zip           # maintainers: build the release asset

The four files are too large for git, so only their SHA-256 hashes are
tracked (artifacts.json). `install` refuses any bundle whose files do not
match those hashes exactly, writes nothing until every file has been checked,
never overwrites a different file already in place, and leaves the installed
files read-only, as the frozen originals are. Nothing is trained or modified.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.cli import prepare_tool

ROOT = prepare_tool(headless=True)

from common.fileio import _TEXT_SUFFIXES, canonical_sha256

RELEASE_TAG = "v1.0.0"
BUNDLE_NAME = "glitch_hunter_final_brain_16M.zip"
RELEASE_URL = (f"https://github.com/TanmaySingh2711/glitch_hunter_project/releases/download/"
               f"{RELEASE_TAG}/{BUNDLE_NAME}")
# Where each file must be for the dashboard to find and approve the brain.
FILES = ("glitch_hunter_main_brain.zip",
         "glitch_hunter_main_brain_coverage.npz",
         "checkpoints_qa/final_objective2_16000000/FINAL_OBJECTIVE2.json",
         "exploration_data/reachable_mask.npz")


def _expected(root: str) -> dict[str, str]:
    with open(os.path.join(root, "artifacts.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    missing = [f for f in FILES if f not in manifest]
    if missing:
        raise SystemExit(f"artifacts.json has no hash for {missing}; this checkout cannot verify them")
    return {f: manifest[f]["sha256"] for f in FILES}


def _digest(name: str, data: bytes) -> str:
    """The same hash artifacts.json records (text: CRLF folded to LF)."""
    if os.path.splitext(name)[1].lower() in _TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def install(root: str, bundle: bytes) -> list[str]:
    """Verifies every file of `bundle`, then writes them under `root`.
    Returns one line per file saying what was done."""
    expected = _expected(root)
    with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
        names = {n for n in zf.namelist() if not n.endswith("/")}
        missing = [f for f in FILES if f not in names]
        unknown = sorted(names - set(FILES))
        if missing or unknown:
            raise SystemExit(f"not the final-brain bundle: missing {missing}, unexpected {unknown}")
        contents = {f: zf.read(f) for f in FILES}
    for f, data in contents.items():
        if _digest(f, data) != expected[f]:
            raise SystemExit(f"{f} does not match its SHA-256 in artifacts.json; nothing was written")
    done = []
    for f, data in contents.items():
        dest = os.path.join(root, *f.split("/"))
        if os.path.exists(dest):
            if canonical_sha256(dest) == expected[f]:
                done.append(f"already in place  {f}")
                continue
            raise SystemExit(f"{f} already exists with DIFFERENT content; move it away first "
                             f"(nothing was overwritten)")
        os.makedirs(os.path.dirname(dest) or root, exist_ok=True)
        tmp = dest + ".part"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, dest)
        os.chmod(dest, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        done.append(f"installed         {f}")
    return done


def bundle(root: str, out: str) -> None:
    """Builds the release asset from this checkout's verified files."""
    expected = _expected(root)
    for f in FILES:
        path = os.path.join(root, *f.split("/"))
        if not os.path.isfile(path):
            raise SystemExit(f"{f} is not in this checkout")
        if canonical_sha256(path) != expected[f]:
            raise SystemExit(f"{f} does not match artifacts.json; refusing to bundle it")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in FILES:
            zf.write(os.path.join(root, *f.split("/")), arcname=f)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_in = sub.add_parser("install", help="download (or take) the bundle, verify it, put the files in place")
    p_in.add_argument("--zip", help=f"a downloaded {BUNDLE_NAME} (default: download it from the release)")
    p_in.add_argument("--url", default=RELEASE_URL, help="where to download the bundle from")
    p_b = sub.add_parser("bundle", help="maintainers: build the release asset from this checkout")
    p_b.add_argument("out", help=f"where to write the bundle (e.g. {BUNDLE_NAME})")
    args = ap.parse_args(argv)
    if args.cmd == "bundle":
        bundle(ROOT, args.out)
        print(f"wrote {args.out} ({os.path.getsize(args.out):,} bytes); every file matches artifacts.json")
        return 0
    if args.zip:
        with open(args.zip, "rb") as fh:
            data = fh.read()
    else:
        print(f"downloading {args.url} ...")
        with urllib.request.urlopen(args.url, timeout=120) as resp:  # noqa: S310 - fixed https URL
            data = resp.read()
    for line in install(ROOT, data):
        print(line)
    print("every file matches artifacts.json - the dashboard will now use the final 16M brain")
    return 0


if __name__ == "__main__":
    sys.exit(main())
