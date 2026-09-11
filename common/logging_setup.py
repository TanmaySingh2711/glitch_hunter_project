"""One logging configuration for every entry point.

Library modules only ever do ``log = logging.getLogger(__name__)`` and call
it; they never configure anything. The entry points (app.py, train_agent.py,
the tools) call configure_logging() once, which gives:

  * the CONSOLE exactly what it always showed: the bare message, one line per
    record - "[COVERAGE] step 6,010,000 | covered ..." reads the same as it
    did when it was a print(). WARNING and above are prefixed with their
    level so they stand out in a long training scroll.
  * optionally a LOG FILE with a timestamp, level, module and function on
    every line, so a training run's history survives the terminal it was
    started in.

The console handler looks up sys.stdout when it WRITES, not when it is
created. Test harnesses (and app.py's line-buffered UTF-8 reconfigure) swap
sys.stdout after import; a handler bound to the old stream would write into
a closed or stale object.

Levels come from the GLITCH_HUNTER_LOG_LEVEL environment variable (default
INFO), so a quieter or chattier run needs no code change.
"""
from __future__ import annotations

import logging
import os
import sys

ENV_LEVEL = 'GLITCH_HUNTER_LOG_LEVEL'
FILE_FORMAT = '%(asctime)s %(levelname)-7s %(name)s.%(funcName)s: %(message)s'


class _ConsoleFormatter(logging.Formatter):
    """The bare message for INFO and below; '[WARNING] ...' above that."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if record.levelno >= logging.WARNING:
            return f"[{record.levelname}] {message}"
        return message


_progress_pending = False       # a write_progress() line is on screen, unterminated


class ConsoleHandler(logging.Handler):
    """Writes to whatever sys.stdout is at the moment of each record.

    If a write_progress() status line is still on screen it is ended first,
    so a record never lands on top of it (messages used to carry a leading
    os.linesep for exactly this, which left blank lines in every log file).
    """

    def emit(self, record: logging.LogRecord) -> None:
        global _progress_pending
        try:
            stream = sys.stdout
            if _progress_pending:
                stream.write('\n')
                _progress_pending = False
            stream.write(self.format(record) + '\n')
            stream.flush()
        except Exception:
            self.handleError(record)


class _RunLogFileHandler(logging.FileHandler):
    """The optional per-run log file; its own type so it can be found again."""


_OWN_HANDLERS = (ConsoleHandler, _RunLogFileHandler)


def _level_from_env(default: int) -> int:
    name = os.environ.get(ENV_LEVEL, '').strip().upper()
    if not name:
        return default
    level = logging.getLevelName(name)
    return level if isinstance(level, int) else default


def configure_logging(level: int = logging.INFO, log_file: str | None = None) -> logging.Logger:
    """Installs the project's handlers on the root logger. Idempotent.

    Calling it again (a test running main() twice, a tool importing another
    tool) replaces the handlers this function installed earlier instead of
    stacking a second copy of every line.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, _OWN_HANDLERS):
            root.removeHandler(handler)
            handler.close()

    console = ConsoleHandler()
    console.setFormatter(_ConsoleFormatter('%(message)s'))
    root.addHandler(console)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        to_file = _RunLogFileHandler(log_file, encoding='utf-8')
        to_file.setFormatter(logging.Formatter(FILE_FORMAT))
        root.addHandler(to_file)

    root.setLevel(_level_from_env(level))
    return root


def write_progress(message: str) -> None:
    """A live, self-overwriting status line (carriage return, no newline).

    Deliberately NOT a log record: it is redrawn thousands of times per run,
    so it belongs on the terminal and nowhere else.
    """
    global _progress_pending
    sys.stdout.write(message + '\r')
    sys.stdout.flush()
    _progress_pending = True
