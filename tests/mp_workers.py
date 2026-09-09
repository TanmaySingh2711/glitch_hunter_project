"""Worker entry points for the multi-process coverage test.

These live in their own module rather than inside the test file because
`spawn` (which is what SubprocVecEnv uses, and what this test deliberately
matches) re-imports the target function's module in a fresh interpreter.
Importing a pytest test module in that fresh interpreter works only by
accident of sys.path ordering; importing a plain module does not.

`spawn` ships the parent's sys.path to the child, and pytest has already put
tests/ on it, so `import mp_workers` resolves in the child.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RECT = (1000, 300, 30, 40)
AREA = RECT[2] * RECT[3]


def _attach(shm_names):
    from exploration.coverage import open_shared
    return open_shared(shm_names)


def marker(shm_names, q, marked, released):
    """Process A: claims RECT, then stays attached until B has finished."""
    cov = _attach(shm_names)
    try:
        cov.begin_episode()
        q.put(('A', cov.record({'cur_rect': RECT})))
        marked.set()
        released.wait(timeout=30)
    finally:
        cov.close()


def follower(shm_names, q, marked, released):
    """Process B: attaches to the same block and re-walks A's exact region."""
    cov = _attach(shm_names)
    try:
        marked.wait(timeout=30)
        cov.begin_episode()
        q.put(('B', cov.record({'cur_rect': RECT})))
        q.put(('B_total', cov.total_unique()))
        q.put(('B_mult', cov.novelty_mult))
    finally:
        cov.close()
        released.set()
