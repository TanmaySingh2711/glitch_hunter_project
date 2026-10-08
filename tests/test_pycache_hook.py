"""tools/pycache_hook.py: whatever starts Python in the project's venv, its
bytecode goes to generated/cache/pycache/, never into a __pycache__ folder
next to the source (on 2026-10-07 a plain `import desktop` left one in the
project root, and bare pytest always left tests/__pycache__/)."""
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import pycache_hook as hook


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def test_a_process_with_the_hook_writes_no_pycache_next_to_the_source(tmp_path):
    """The real thing: a fresh Python reads the hook at start-up (site), then
    imports a module - its bytecode lands in the cache folder only."""
    site, src, cache = tmp_path / "site", tmp_path / "src", tmp_path / "cache"
    site.mkdir()
    src.mkdir()
    (src / "some_module.py").write_text("VALUE = 1\n", encoding="utf-8")
    hook.install(str(site), str(cache))
    code = ("import site, sys; site.addsitedir(sys.argv[1]); sys.path.insert(0, sys.argv[2]); "
            "import some_module; print(sys.pycache_prefix)")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPYCACHEPREFIX"}
    # -S: start without the venv's own site-packages (and so without its own
    # hook, which would set the prefix first); addsitedir then reads ours.
    out = subprocess.run([sys.executable, "-S", "-c", code, str(site), str(src)], env=env,
                         capture_output=True, text=True, timeout=60, check=True)
    assert out.stdout.strip() == str(cache)
    assert not (src / "__pycache__").exists()
    assert any(p.name.startswith("some_module.") for p in cache.rglob("*.pyc"))


def test_the_hook_keeps_a_prefix_that_is_already_set():
    ns = {}
    sys_like = type("S", (), {"pycache_prefix": "/already/set"})()
    exec(hook.pth_line("/x").replace("import sys; ", ""), {"sys": sys_like}, ns)
    assert sys_like.pycache_prefix == "/already/set"


def test_install_and_check(tmp_path):
    assert not hook.installed(str(tmp_path))
    path = hook.install(str(tmp_path))
    assert os.path.basename(path) == hook.PTH_NAME and hook.installed(str(tmp_path))
    assert os.path.join(ROOT, "generated", "cache", "pycache") == hook.CACHE


def test_outside_a_venv_it_changes_nothing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(hook, "in_venv", lambda: False)
    monkeypatch.setattr(hook, "site_dir", lambda: str(tmp_path))
    assert hook.main([]) == 1 and not any(tmp_path.iterdir())
    assert "nothing changed" in capsys.readouterr().out


def test_both_setup_scripts_install_it_before_any_package():
    for name, line in (("setup.bat", r'"%VPY%" tools\pycache_hook.py'),
                       ("setup.sh", '"$VPY" tools/pycache_hook.py')):
        s = read(name)
        assert line in s, name
        assert s.index(line) < s.index("pip install --quiet --upgrade pip"), name


@pytest.mark.parametrize("tool", sorted(f for f in os.listdir(os.path.join(ROOT, "tools"))
                                        if f.endswith(".py")))
def test_no_tool_sends_bytecode_into_tools(tool):
    """16 tools pointed sys.pycache_prefix at tools/generated/ (one dirname
    short), which grew to 38 MB of stray bytecode."""
    for m in re.finditer(r"sys\.pycache_prefix = os\.path\.join\((.*?), \"generated\"", read("tools", tool)):
        assert m.group(1) == "os.path.dirname(os.path.dirname(os.path.abspath(__file__)))", tool
