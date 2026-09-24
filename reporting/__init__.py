"""Objective 3: turn a detected QA anomaly into a durable, reviewable incident.

    detector (custom_mario_env) -> Detection -> IncidentPipeline.capture()
        -> raw evidence persisted (IncidentStore) -> GIF / Markdown / PDF,
           replay-based reproduction -> dashboard

See docs/OBJECTIVE3.md for the architecture and the reasoning behind it.
"""
