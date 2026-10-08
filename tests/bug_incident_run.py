"""One dashboard episode on ONE game variant, through the real Objective-3
pipeline, printed as JSON. Run by tests/test_bug_incidents.py (fresh process
per variant: both import the game as `data`).

    python tests/bug_incident_run.py mario_bugged <store dir>

The approved brain draws its moves exactly as the dashboard does, from a
fixed seed (route_seed) so the run is the same every time, with evidence on
and every built-in detector live; each detection is captured,
rendered (GIF, Markdown, PDF) and replayed in a separate process. No probe,
no teleport: every incident here is a real detection a replay can reproduce.
"""
import json
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard import backend as db

# A drawn route that meets every benchmark bug in its first run (measured,
# 2026-10-03: seeds 1, 2 and 8 do so in one run; 3-7 need two or three).
ROUTE_SEED = 1


def main(variant, store_dir, max_steps=700):
    db.configure(db.DashboardConfig(game_variant=variant, incidents_dir=store_dir,
                                    run_reports_dir=os.path.join(store_dir, "run_reports"),
                                    route_seed=ROUTE_SEED))
    env, _model = db._ensure_global_env_and_model()
    pipe = db.ensure_pipeline()
    base = db._base(env)
    first_episode = None
    session = db.run_mario_agent()
    for _ in range(max_steps):
        next(session)
        if first_episode is None:
            first_episode = base.episode_index
        if base.episode_index != first_episode:        # one full episode, then stop
            break
    pipe.wait_idle(600)
    out = []
    for s in pipe.summaries():
        rec = pipe.store.load_record(s["incident_id"])
        out.append({"kind": s["category"], "detector": rec["detector"]["id"],
                    "reproduction": s["reproduction"], "renders": s["renders"],
                    "verify": pipe.store.verify(s["incident_id"]), "synthetic": s["synthetic"],
                    "variant": rec["provenance"]["game"]["variant"]})
    pipe.close()
    print(json.dumps({"variant": variant, "incidents": out}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
