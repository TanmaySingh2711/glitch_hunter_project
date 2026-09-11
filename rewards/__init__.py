"""The two reward objectives GlitchHunterWrapper (agent_logic.py) can run.

    shared.py   state and shaping both modes use: locomotion tracking, the
                powerup/score bookkeeping and the cross-episode ADAPTIVE
                DEATH MEMORY
    qa.py       "qa_exploration" - persistent world-space coverage, phase-gated
                by the EXPLORE -> COMPLETE lifecycle. The current objective.
    legacy.py   "legacy_completion" - the reward the 6M brain was trained
                under, preserved exactly. The baseline and the fallback.

Each is a mixin over the wrapper: it owns its own state (declared on the
class, initialised by _reset_*_state) and one method that turns a substep's
info dict into (reward, done). The wrapper itself only records coverage,
drives the lifecycle and dispatches on the mode.
"""
