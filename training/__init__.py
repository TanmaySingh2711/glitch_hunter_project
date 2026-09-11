"""Training building blocks used by train_agent.py.

train_agent.py is the entry point and keeps the run-level decisions: which
phase is active, where checkpoints go, when a launch is refused, and the
main() that wires it together. What lives here is everything it composes:

    callbacks.py    the SB3 callbacks - milestone checkpoints, the watchdog,
                    the value-function warm-up, coverage / lifecycle reporting,
                    stagnation escalation and the Level-1 completion stop
    checkpoints.py  reading what a checkpoint zip or its file name says,
                    without loading any weights
    value_head.py   the one-off critic reset for the first QA run
"""
