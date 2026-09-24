"""tools/final_brain.py: the final 16M brain reaches a fresh clone intact.

The four files are git-ignored; only their hashes are tracked. These tests use
stand-in files and a stand-in artifacts.json in a scratch project root, so
they run anywhere (CI has none of the real files)."""
import importlib.util
import io
import json
import os
import stat
import zipfile

import pytest

from common.fileio import canonical_sha256
from exploration import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("final_brain", os.path.join(ROOT, "tools", "final_brain.py"))
assert _spec is not None and _spec.loader is not None
fb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fb)


def test_the_bundle_holds_exactly_what_the_dashboard_needs_for_the_final_brain():
    assert fb.FILES == (config.FINAL_BRAIN_PATH, config.FINAL_COVERAGE_PATH,
                        config.FINAL_OBJECTIVE2_RECORD, config.REACHABLE_MASK_PATH)


def test_the_real_manifest_tracks_every_bundled_file():
    with open(os.path.join(ROOT, "artifacts.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert all(f in manifest for f in fb.FILES)


CONTENT = {f: (b'{"stand": "in"}\r\n' if f.endswith(".json") else f"bytes of {f}".encode())
           for f in fb.FILES}


@pytest.fixture
def root(tmp_path):
    for f, data in CONTENT.items():                        # hash the stand-ins as git would
        p = tmp_path / "src" / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    manifest = {f: {"sha256": canonical_sha256(tmp_path / "src" / f)} for f in fb.FILES}
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "artifacts.json").write_text(json.dumps(manifest), encoding="utf-8")
    return proj


def _zip(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_install_verifies_places_and_locks_every_file(root):
    lines = fb.install(str(root), _zip(CONTENT))
    assert all(line.startswith("installed") for line in lines) and len(lines) == 4
    for f, data in CONTENT.items():
        p = root / f
        assert p.read_bytes() == data
        assert not os.stat(p).st_mode & stat.S_IWRITE          # read-only, like the frozen originals
    again = fb.install(str(root), _zip(CONTENT))
    assert all(line.startswith("already in place") for line in again)


def test_a_tampered_bundle_writes_nothing(root):
    bad = dict(CONTENT)
    bad[fb.FILES[0]] = b"not the approved brain"
    with pytest.raises(SystemExit, match="does not match"):
        fb.install(str(root), _zip(bad))
    assert not any((root / f).exists() for f in fb.FILES)


def test_a_bundle_with_missing_or_extra_files_is_refused(root):
    partial = {f: d for f, d in CONTENT.items() if f != fb.FILES[1]}
    with pytest.raises(SystemExit, match="missing"):
        fb.install(str(root), _zip(partial))
    with pytest.raises(SystemExit, match="unexpected"):
        fb.install(str(root), _zip({**CONTENT, "evil.py": b"print(1)"}))


def test_a_different_file_already_in_place_is_never_overwritten(root):
    target = root / fb.FILES[0]
    target.write_bytes(b"someone's own brain")
    with pytest.raises(SystemExit, match="DIFFERENT"):
        fb.install(str(root), _zip(CONTENT))
    assert target.read_bytes() == b"someone's own brain"


def test_bundle_refuses_files_that_do_not_match_the_manifest(root, tmp_path):
    for f, data in CONTENT.items():
        p = root / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    out = tmp_path / "b.zip"
    fb.bundle(str(root), str(out))
    with zipfile.ZipFile(out) as zf:
        assert sorted(zf.namelist()) == sorted(fb.FILES)
    (root / fb.FILES[2]).write_bytes(b"edited")
    with pytest.raises(SystemExit, match="refusing"):
        fb.bundle(str(root), str(tmp_path / "c.zip"))
