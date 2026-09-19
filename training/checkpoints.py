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


def retention_verdicts(results_dir: str) -> dict[str, dict[str, object]]:
    """Known completion-retention verdicts, keyed by checkpoint SHA-256.

    "Newest wins" is only safe while newer also means better, and the
    6,400,000-step checkpoint is the counter-example: it is the newest thing
    on disk AND it is REGRESSED (completion 6.0% against the 6M baseline's
    46.8%, mean progress 0.280 against 0.710). Resuming it would have
    continued training a brain that had lost the ability to finish the level.

    So selection reads the verdicts the evaluator has already written. Keyed
    by SHA-256 rather than by path because a checkpoint gets copied and
    renamed - the root master and checkpoints_qa/pre_main_6032768/ are the
    same bytes - and the verdict belongs to the weights, not the filename.

    A checkpoint with no result file is simply unknown, never assumed good or
    bad; the caller decides what to do with that.
    """
    out: dict[str, dict[str, object]] = {}
    if not os.path.isdir(results_dir):
        return out
    for name in sorted(os.listdir(results_dir)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(results_dir, name), encoding="utf-8") as fh:
                doc = json.load(fh)
            sha = str(doc["checkpoint"]["sha256"])
            verdict = str(doc["comparison"]["verdict"])
        except (OSError, ValueError, KeyError, TypeError):
            continue          # an unreadable result is not a verdict
        out[sha] = {
            "verdict": verdict,
            "result_file": name,
            "num_timesteps": doc["checkpoint"].get("num_timesteps"),
            "completion_rate": doc.get("comparison", {}).get("completion_rate"),
        }
    return out
