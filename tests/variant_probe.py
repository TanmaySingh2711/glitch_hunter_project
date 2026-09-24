"""Plays one game variant in a fresh process and prints its behavioural
fingerprint as JSON. Run by tests/test_game_variants.py, once per variant:
one process can host only one variant (both import the game as `data`).

    python tests/variant_probe.py mario_clean
    python tests/variant_probe.py mario_bugged mario_clean   # load clean first,
                                  # release it, then probe bugged (the game switch)

Fingerprints, each a SHA-256 so two variants compare with one equality:
  noop600          the 600 raw observations holding NOOP from reset - the same
                   measurement as test_hud_timer.LEGACY_NOOP_600_SHA256
  scripted_state   every substep's physics state (collider, x_vel, y_vel,
                   Mario's state, score, coins, clock, death) over a scripted
                   run that walks, sprints, jumps into walls and meets enemies
  scripted_frames  every 25th full-resolution frame of that run (rendering)
  geometry         every collider in the level at reset (level layout)
"""
import hashlib
import json
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame as pg

from custom_mario_env import CustomMarioEnv, claim_game_variant, release_game_variant

# A fixed, varied action script: sprint right with jumps (into the first pipe
# and over the goombas), retreat, walk, crouch, stand. Deterministic.
SCRIPT = ([3] * 40 + [4] * 30 + [3] * 60 + [4] * 25 + [0] * 20 + [8] * 30
          + [9] * 20 + [1] * 60 + [2] * 30 + [7] * 15 + [5] * 25 + [4] * 80
          + [3] * 100 + [4] * 40 + [6] * 30 + [3] * 120)


def main(variant, switch_from=None):
    if switch_from:
        first = CustomMarioEnv(game_variant=switch_from)
        first.reset()
        for _ in range(60):
            first.step(3)
        del first
        release_game_variant()
    env = CustomMarioEnv(game_variant=variant)
    claim_game_variant(variant)
    import data
    loaded = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(data.__file__))))
    out = {"variant": variant, "loaded_from": loaded, "switched_from": switch_from}

    env.reset()
    h = hashlib.sha256()
    for _ in range(600):
        obs, _r, done, _t, _i = env.step(0)
        h.update(obs.tobytes())
        assert not done
    out["noop600"] = h.hexdigest()

    env.reset()
    state = env.game.state
    geo = hashlib.sha256()
    rects = []
    for group in ("ground_group", "pipe_group", "step_group", "brick_group", "coin_box_group"):
        for s in sorted(getattr(state, group), key=lambda s: (s.rect.x, s.rect.y, s.rect.w)):
            geo.update(f"{group}:{s.rect.x},{s.rect.y},{s.rect.w},{s.rect.h};".encode())
            rects.append([group, s.rect.x, s.rect.y, s.rect.w, s.rect.h])
    out["geometry"] = geo.hexdigest()
    out["geometry_rects"] = rects

    st, fr = hashlib.sha256(), hashlib.sha256()
    deaths = 0
    per_substep, per_frame = [], []
    for i, action in enumerate(SCRIPT):
        _o, _r, done, _t, info = env.step(action)
        mario = env.game.state.mario
        row = json.dumps([info.get("mario_rect"), round(float(info.get("x_vel", 0)), 6),
                          round(float(getattr(mario, "y_vel", 0)), 6), str(mario.state),
                          info.get("score"), info.get("coins"), info.get("time_left"),
                          info.get("is_dead")]).encode()
        st.update(row)
        per_substep.append([info.get("mario_rect"), hashlib.sha256(row).hexdigest()[:16]])
        if i % 25 == 0:
            frame = pg.surfarray.array3d(pg.display.get_surface()).tobytes()
            fr.update(frame)
            per_frame.append([i, hashlib.sha256(frame).hexdigest()[:16]])
        if done:
            deaths += 1
            env.reset()
    out.update(scripted_state=st.hexdigest(), scripted_frames=fr.hexdigest(),
               scripted_substeps=len(SCRIPT), scripted_episode_ends=deaths,
               scripted_per_substep=per_substep, scripted_per_frame=per_frame)
    print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
