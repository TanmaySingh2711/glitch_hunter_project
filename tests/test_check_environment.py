"""tools/check_environment.py: a library that will not load gets a plain
answer instead of a traceback (the 2026-10-07 case: Windows' Smart App
Control blocking PyTorch's DLL made run_dashboard.bat fail with WinError 4551)."""
import importlib
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import check_environment as ce

BLOCKED = OSError('[WinError 4551] An Application Control policy has blocked this file. Error '
                  r'loading "C:\x\torch\lib\torch_python.dll" or one of its dependencies.')


def read(name):
    with open(os.path.join(ROOT, name), "rb") as fh:
        return fh.read().decode("ascii")


def test_windows_blocking_a_dll_is_recognised_however_it_is_worded():
    assert ce.blocked_by_application_control(BLOCKED)
    assert ce.blocked_by_application_control(OSError("blocked by Smart App Control"))
    err = OSError("x")
    err.winerror = 4551
    assert ce.blocked_by_application_control(err)
    assert not ce.blocked_by_application_control(ImportError("No module named 'torch'"))
    assert not ce.blocked_by_application_control(OSError("[WinError 126] module not found"))


def test_the_smart_app_control_answer_says_what_it_is_and_where_to_switch_it_off():
    advice = ce.diagnose("torch", "the agent's brain", BLOCKED)
    for needed in ("Smart App Control", "Privacy & security", "Off", "run_dashboard.bat",
                   "Windows says: An Application Control policy has blocked this file."):
        assert needed in advice
    assert "site-packages" not in advice and "dependencies" not in advice, "the long path is not shown"


def test_a_missing_library_points_at_the_setup_script():
    advice = ce.diagnose("flask", "the dashboard server", ModuleNotFoundError("No module named 'flask'"))
    assert '"flask"' in advice and "setup.bat" in advice and "setup.sh" in advice
    assert "Smart App Control" not in advice


def test_any_other_load_failure_is_reported_with_its_error():
    advice = ce.diagnose("cv2", "video frames", ImportError("DLL load failed while importing cv2"))
    assert "DLL load failed" in advice and "setup.bat" in advice


def test_check_is_silent_when_everything_loads_and_names_the_first_failure(monkeypatch):
    assert ce.check((("os", "x"), ("sys", "y"))) is None
    real = importlib.import_module

    def broken(name, *a, **kw):
        if name == "gymnasium":
            raise BLOCKED
        return real(name, *a, **kw)

    monkeypatch.setattr(ce.importlib, "import_module", broken)
    advice = ce.check((("os", "x"), ("gymnasium", "the environment"), ("sys", "y")))
    assert advice is not None and "Smart App Control" in advice


def test_main_prints_the_advice_and_fails(monkeypatch, capsys):
    monkeypatch.setattr(ce, "check", lambda: "ADVICE")
    assert ce.main([]) == 1 and "ADVICE" in capsys.readouterr().out
    monkeypatch.setattr(ce, "check", lambda: None)
    assert ce.main([]) == 0 and capsys.readouterr().out == ""


def test_the_scripts_run_the_check():
    bat = read("run_dashboard.bat")
    assert bat.index("app.py --desktop") < bat.index(r"if errorlevel 1 venv_gpu\Scripts\python.exe tools\check_environment.py")
    assert r"tools\check_environment.py || goto :fail" in read("setup.bat")
    assert 'tools/check_environment.py' in read("setup.sh")


@pytest.mark.parametrize("name", [n for n, _ in ce.REQUIRED])
def test_every_required_name_is_a_real_import_name(name):
    """A typo here would report a healthy machine as broken."""
    assert importlib.util.find_spec(name) is not None or name == "torch"
