"""Persistent world-space exploration tracking for the Glitch Hunter agent.

The agent was trained to 6,000,000 steps to COMPLETE the level. This package
changes what it optimises next: persistent spatial coverage of the world that
survives episode resets, process restarts and worker boundaries, so the agent
behaves as an autonomous QA explorer rather than a speedrunner.

Public surface:
    config          - every tunable constant, in one place
    coverage        - SpatialCoverage: the shared bitmap, frontier, targets
    reachability    - the solid/coverable masks derived from level geometry
"""
