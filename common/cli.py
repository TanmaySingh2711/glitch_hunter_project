"""The start-up every script in tools/ shares:

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common.cli import prepare_tool
    ROOT = prepare_tool()

(The one sys.path line has to stay in the tool itself: until it runs, this
module cannot be imported.) Each tool used to repeat everything below on its
own - six copies, drifting apart:

  * UTF-8 stdout. The reports use box-drawing characters and Windows'
    default cp1252 console cannot encode them, so redirecting a tool to a
    file crashed it AFTER the work was done but BEFORE the result was written.
  * Headless SDL. A tool replaying the game needs a pygame surface but no
    window and no sound card; the dummy drivers give exactly that. Only a
    default - an explicit SDL_VIDEODRIVER in the environment wins.
  * The project root on sys.path, so `python tools/x.py` can import the
    project's own modules from any working directory.
  * The project's logging, so library log lines (coverage, lifecycle,
    evaluation progress) appear alongside the tool's own report.
"""
from __future__ import annotations

import os
import sys

from common.logging_setup import configure_logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # the project root


def prepare_tool(headless: bool = True) -> str:
    """Applies the shared start-up (see the module docstring); returns ROOT."""
    reconfigure = getattr(sys.stdout, 'reconfigure', None)
    if reconfigure is not None:
        reconfigure(encoding='utf-8', errors='replace')
    if headless:
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    configure_logging()
    return ROOT
