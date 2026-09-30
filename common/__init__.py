"""Small, dependency-free helpers shared by every layer of the project.

Nothing in here knows about Mario, rewards or coverage: file integrity
(fileio) and console/file logging (logging_setup). Anything that does know
about those belongs in exploration/, evaluation/ or training/ instead.

Importing it also stops Python writing __pycache__ folders for the project
(and, through the environment, for the processes it starts): everything the
project generates lives in generated/ instead (exploration/config.py,
GENERATED_DIR). Every entry point imports `common` before its other project
modules. The cost is recompiling the project's own source at start-up, well
under a second; installed libraries keep the caches they were installed with.
"""
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
