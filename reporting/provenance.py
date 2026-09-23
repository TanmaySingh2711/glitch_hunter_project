"""Which brain, which game and which code produced an incident.

Computed once per session and copied into every incident, so a report can be
traced back without trusting anything that could have changed since: the
brain by SHA-256 (checked against the frozen Objective-2 record), the game by
its tree hash (reporting/variants.py), the code by git commit.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import zipfile
from importlib import metadata
from typing import Any

from common.fileio import sha256_of
from exploration import config
from reporting import variants

PROJECT_ROOT = variants.PROJECT_ROOT


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def objective2_record() -> dict[str, Any] | None:
    """The frozen Objective-2 closure record, or None when it is not in
    this checkout (checkpoints_qa/ is git-ignored)."""
    path = _abs(config.FINAL_OBJECTIVE2_RECORD)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def brain_identity(path: str | None) -> dict[str, Any]:
    """What a report needs to name the brain: path, SHA-256, training steps,
    and whether it IS the approved Objective-2 brain."""
    if path is None or not os.path.isfile(_abs(path)):
        return {"path": path, "present": False, "sha256": None, "num_timesteps": None,
                "approved_objective2_brain": False,
                "note": "no trained brain was loaded; the policy was untrained"}
    full = _abs(path)
    digest = sha256_of(full)
    steps: int | None
    try:
        with zipfile.ZipFile(full) as zf:
            steps = int(json.loads(zf.read("data"))["num_timesteps"])
    except (zipfile.BadZipFile, KeyError, ValueError, OSError):
        steps = None
    record = objective2_record()
    approved_sha = (record or {}).get("brain", {}).get("sha256")
    return {"path": os.path.relpath(full, PROJECT_ROOT).replace(os.sep, "/"), "present": True,
            "sha256": digest, "num_timesteps": steps,
            "approved_objective2_brain": approved_sha is not None and digest == approved_sha,
            "approval_record": config.FINAL_OBJECTIVE2_RECORD if record else None}


def code_identity() -> dict[str, Any]:
    """The git commit the code came from, and whether the tree had edits."""
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", PROJECT_ROOT, *args],  # noqa: S603, S607 (fixed argv)
                                 capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": commit, "git_available": commit is not None,
            "uncommitted_changes": None if status is None else bool(status)}


def runtime_identity() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for dist in ("numpy", "pygame", "torch", "stable-baselines3", "gymnasium", "fpdf2",
                 "pillow", "opencv-python"):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "packages": versions}


def session_provenance(brain_path: str | None, reward_mode: str,
                       game_variant: str) -> dict[str, Any]:
    return {"brain": brain_identity(brain_path),
            "game": variants.variant_identity(game_variant),
            "reward_mode": reward_mode,
            "code": code_identity(),
            "runtime": runtime_identity()}
