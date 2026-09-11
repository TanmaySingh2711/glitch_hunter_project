"""What a checkpoint file says about itself, read without loading weights."""
from __future__ import annotations

import json
import os
import zipfile


def milestone_steps(filename: str) -> int:
    """Step count encoded in 'mario_brain_checkpoint_{N}_steps.zip', else -1.

    Returning -1 for unparseable names keeps them sorted below every real
    milestone, so a stray file can never be picked as "latest" — the old
    version fell back to lexicographic order on a parse failure, which
    silently ranks '800000' above '6000000'.
    """
    parts = filename[:-len(".zip")].split("_")
    if len(parts) >= 2 and parts[-1] == "steps":
        try:
            return int(parts[-2])
        except ValueError:
            return -1
    return -1


def newest_milestone(checkpoint_dir: str, name_prefix: str) -> str | None:
    """Path of the highest-numbered '<prefix>_<N>_steps.zip' in a directory,
    or None when there is no directory or no parseable milestone in it."""
    if not os.path.exists(checkpoint_dir):
        return None
    candidates = [f for f in os.listdir(checkpoint_dir)
                  if f.endswith(".zip") and f.startswith(name_prefix)
                  and milestone_steps(f) >= 0]
    if not candidates:
        return None
    return os.path.join(checkpoint_dir, max(candidates, key=milestone_steps))


def checkpoint_timesteps(path: str) -> int:
    """The global timestep saved inside an SB3 checkpoint zip, read from its
    'data' record without loading any weights (~1 ms) - so the coverage that
    belongs to it can be verified before a single worker starts."""
    with zipfile.ZipFile(path) as z:
        return int(json.loads(z.read("data"))["num_timesteps"])
