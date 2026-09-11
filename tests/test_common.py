"""common/: atomic file I/O, the logging set-up, and the tools' start-up."""
import json
import logging
import os
import stat
import sys

import pytest

from common import fileio
from common import logging_setup as ls


# ── fileio ─────────────────────────────────────────────────────────────────
def test_sha256_matches_hashlib_for_multi_block_files(tmp_path):
    import hashlib
    p = tmp_path / "blob.bin"
    data = os.urandom(3 * (1 << 20) + 17)          # crosses the 1 MiB block edge
    p.write_bytes(data)
    assert fileio.sha256_of(p) == hashlib.sha256(data).hexdigest()


def test_json_is_written_atomically_and_reads_back(tmp_path):
    target = tmp_path / "nested" / "record.json"
    fileio.write_json_atomic({'a': 1, 'b': [1, 2]}, target)
    assert fileio.read_json(target) == {'a': 1, 'b': [1, 2]}
    assert not (tmp_path / "nested" / "record.json.tmp").exists()


def test_a_failed_write_leaves_the_previous_file_intact(tmp_path):
    target = tmp_path / "record.json"
    fileio.write_json_atomic({'v': 1}, target)
    with pytest.raises(TypeError):
        fileio.write_json_atomic({'v': object()}, target)   # not serialisable
    assert json.loads(target.read_text(encoding='utf-8')) == {'v': 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["record.json"], "temp file left behind"


def test_atomic_write_replaces_on_success_and_keeps_the_old_file_on_failure(tmp_path):
    target = tmp_path / "blob.bin"
    with fileio.atomic_write(target) as fh:
        fh.write(b"first")
    assert target.read_bytes() == b"first"

    def crash_mid_write():
        with fileio.atomic_write(target) as fh:
            fh.write(b"half-written")
            raise RuntimeError("crash mid-write")
    with pytest.raises(RuntimeError):
        crash_mid_write()
    assert target.read_bytes() == b"first"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["blob.bin"]


def test_make_read_only_drops_the_write_bit(tmp_path):
    p = tmp_path / "proof.json"
    p.write_text("{}", encoding='utf-8')
    fileio.make_read_only(p)
    try:
        assert not os.stat(p).st_mode & stat.S_IWRITE
        with pytest.raises(PermissionError):
            open(p, 'w').close()
    finally:
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD)


# ── logging_setup ──────────────────────────────────────────────────────────
@pytest.fixture
def root_logger():
    """configure_logging() mutates the root logger; put it back afterwards."""
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)
    yield root
    for h in list(root.handlers):
        if h not in saved[0]:
            root.removeHandler(h)
            h.close()
    root.setLevel(saved[1])


def test_console_lines_read_exactly_like_the_old_prints(root_logger, capsys):
    ls.configure_logging()
    logging.getLogger("x").info("[COVERAGE] step %s", "6,010,000")
    logging.getLogger("x").warning("could not load %s", "f.npz")
    out = capsys.readouterr().out.splitlines()
    assert out == ["[COVERAGE] step 6,010,000", "[WARNING] could not load f.npz"]


def test_configuring_twice_does_not_duplicate_lines(root_logger, capsys):
    ls.configure_logging()
    ls.configure_logging()
    logging.getLogger("x").info("once")
    assert capsys.readouterr().out == "once\n"


def test_a_progress_line_is_ended_before_the_next_record(root_logger, capsys):
    ls.configure_logging()
    ls.write_progress("Crunching frames... 1,000")
    logging.getLogger("x").info("[CHECKPOINT] saved")
    assert capsys.readouterr().out == "Crunching frames... 1,000\r\n[CHECKPOINT] saved\n"


def test_the_log_file_carries_time_level_and_module(root_logger, tmp_path, capsys):
    log_file = tmp_path / "logs" / "train.log"
    ls.configure_logging(log_file=str(log_file))
    logging.getLogger("training.callbacks").info("[COVERAGE] line")
    for h in root_logger.handlers:
        h.flush()
    line = log_file.read_text(encoding='utf-8').strip()
    assert line.endswith(
        "INFO    training.callbacks.test_the_log_file_carries_time_level_and_module: "
        "[COVERAGE] line")
    assert line[:4].isdigit()                       # starts with the timestamp


def test_level_comes_from_the_environment(root_logger, monkeypatch, capsys):
    monkeypatch.setenv(ls.ENV_LEVEL, "warning")
    ls.configure_logging()
    logging.getLogger("x").info("hidden")
    logging.getLogger("x").warning("shown")
    assert capsys.readouterr().out == "[WARNING] shown\n"
    monkeypatch.setenv(ls.ENV_LEVEL, "not-a-level")
    assert ls.configure_logging().level == logging.INFO


def test_the_console_follows_a_replaced_stdout(root_logger, monkeypatch):
    import io
    ls.configure_logging()
    swapped = io.StringIO()
    monkeypatch.setattr(sys, "stdout", swapped)
    logging.getLogger("x").info("after the swap")
    assert swapped.getvalue() == "after the swap\n"


# ── cli ────────────────────────────────────────────────────────────────────
def test_prepare_tool_sets_the_shared_start_up(root_logger, monkeypatch):
    from common import cli
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.delenv("SDL_AUDIODRIVER", raising=False)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != cli.ROOT])
    assert cli.prepare_tool() == cli.ROOT
    assert os.environ["SDL_VIDEODRIVER"] == "dummy"
    assert os.environ["SDL_AUDIODRIVER"] == "dummy"
    assert sys.path[0] == cli.ROOT
    assert any(isinstance(h, ls.ConsoleHandler) for h in root_logger.handlers)


def test_an_explicit_video_driver_wins_and_windowed_tools_keep_theirs(root_logger, monkeypatch):
    from common import cli
    monkeypatch.setenv("SDL_VIDEODRIVER", "windows")
    cli.prepare_tool()
    assert os.environ["SDL_VIDEODRIVER"] == "windows"
    monkeypatch.delenv("SDL_VIDEODRIVER")
    cli.prepare_tool(headless=False)
    assert "SDL_VIDEODRIVER" not in os.environ
