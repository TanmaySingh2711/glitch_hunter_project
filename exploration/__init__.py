"""Persistent world-space exploration tracking for the Glitch Hunter agent.

The agent was trained to 6,000,000 steps to COMPLETE the level. This package
changes what it optimises next: persistent spatial coverage of the world that
survives episode resets, process restarts and worker boundaries, so the agent
behaves as an autonomous QA explorer rather than a speedrunner.

Public surface:
    config            - every tunable constant, in one place, with its evidence
    coverage          - SpatialCoverage: the shared bitmap, frontier, targets
    reachability      - the solid/coverable masks derived from level geometry
    lifecycle         - EpisodeLifecycle: EXPLORE -> COMPLETE, and why an
                        episode ended
    level_completion  - "Level 1 is complete": covered == TESTABLE_TOTAL,
                        and the snapshot written when it happens
"""
