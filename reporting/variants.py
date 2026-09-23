"""The two game variants, and the provenance that says which one ran.

    mario_clean/    the canonical game. The trustworthy baseline: it never
                    receives an intentional bug, and its content is pinned
                    (config.CLEAN_GAME_TREE_SHA256), so any edit fails the suite.
    mario_bugged/   starts as a byte-identical copy of mario_clean. Deliberate
                    bugs go here - and only here - and each one must be declared
                    in mario_bugged/INJECTED_BUGS.json, naming the files it
                    touches. Until one is declared the two games are identical.

A variant's IDENTITY is the SHA-256 of its game tree: every file under data/
and resources/ (the code and the assets the engine loads), by relative path.
Text files are hashed with CRLF folded to LF so a Windows and a Linux
checkout of the same commit agree. Bytecode caches are ignored; the
per-variant metadata files at the variant root (VARIANT.md,
INJECTED_BUGS.json) are not part of the game and are ignored too.

Both variants are imported as the top-level package `data` (the upstream
layout, left untouched), so one process can host only ONE of them -
custom_mario_env enforces that.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from exploration import config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_TREE_PARTS = ("data", "resources")
_TEXT_SUFFIXES = frozenset((".py", ".txt", ".md", ".json"))
BUG_MANIFEST_NAME = "INJECTED_BUGS.json"


def game_dir(variant: str) -> str:
    """Absolute path of a registered variant; refuses anything else."""
    if variant not in config.GAME_VARIANTS:
        raise ValueError(f"unknown game variant {variant!r}; expected one of "
                         f"{', '.join(config.GAME_VARIANTS)}")
    return os.path.join(PROJECT_ROOT, variant)


def _file_digest(path: str) -> str:
    with open(path, "rb") as fh:
        data = fh.read()
    if os.path.splitext(path)[1].lower() in _TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def game_tree_files(root: str) -> dict[str, str]:
    """{relative posix path: SHA-256} for every game file under `root`."""
    files: dict[str, str] = {}
    for part in GAME_TREE_PARTS:
        base = os.path.join(root, part)
        if not os.path.isdir(base):
            raise FileNotFoundError(f"{base} is missing: {root} is not a game variant")
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for name in sorted(filenames):
                if name.endswith((".pyc", ".pyo")):
                    continue
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                files[rel] = _file_digest(full)
    return dict(sorted(files.items()))


def tree_sha256(files: dict[str, str]) -> str:
    """One digest for a whole tree listing (see game_tree_files)."""
    h = hashlib.sha256()
    for rel, digest in sorted(files.items()):
        h.update(rel.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
    return h.hexdigest()


def game_tree_sha256(root: str) -> str:
    return tree_sha256(game_tree_files(root))


def injected_bugs(variant: str) -> list[dict[str, Any]]:
    """The bugs declared for a variant. mario_clean has none by definition;
    mario_bugged lists its own in INJECTED_BUGS.json (a missing or malformed
    file is an error, never "no bugs")."""
    if variant == config.CLEAN_GAME_VARIANT:
        return []
    path = os.path.join(game_dir(variant), BUG_MANIFEST_NAME)
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    bugs = doc.get("bugs") if isinstance(doc, dict) else None
    if not isinstance(bugs, list):
        raise ValueError(f"{path}: expected an object with a 'bugs' list")
    for bug in bugs:
        if not (isinstance(bug, dict) and isinstance(bug.get("id"), str)
                and isinstance(bug.get("files"), list)):
            raise ValueError(f"{path}: every bug needs a string 'id' and a 'files' list")
    return bugs


def undeclared_differences(variant: str) -> list[str]:
    """Game files where `variant` differs from mario_clean without a declared
    bug naming that file. Empty is the only acceptable answer: a difference
    nobody declared is either an accident or an undocumented bug."""
    clean = game_tree_files(game_dir(config.CLEAN_GAME_VARIANT))
    other = game_tree_files(game_dir(variant))
    declared = {f for bug in injected_bugs(variant) for f in bug["files"]}
    changed = {f for f in set(clean) | set(other) if clean.get(f) != other.get(f)}
    return sorted(changed - declared)


def variant_identity(variant: str) -> dict[str, Any]:
    """Everything an incident needs to prove which game produced it."""
    root = game_dir(variant)
    files = game_tree_files(root)
    digest = tree_sha256(files)
    bugs = injected_bugs(variant)
    return {
        "variant": variant,
        "directory": variant,
        "tree_sha256": digest,
        "file_count": len(files),
        "is_clean_baseline": variant == config.CLEAN_GAME_VARIANT,
        "matches_pinned_clean_tree": digest == config.CLEAN_GAME_TREE_SHA256,
        "declared_injected_bugs": [b["id"] for b in bugs],
    }
