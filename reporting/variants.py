"""The two game variants, and the provenance that says which one ran.

    mario_clean/    the canonical game. The trustworthy baseline: it never
                    receives an intentional bug, and its content is pinned
                    (config.CLEAN_GAME_TREE_SHA256), so any edit fails the suite.
    mario_bugged/   starts as a byte-identical copy of mario_clean. Deliberate
                    bugs go here - and only here - and each one must be declared
                    in mario_bugged/INJECTED_BUGS.json, naming the files it
                    touches.

A difference from mario_clean is accepted only if it is ACCOUNTED FOR, at
three levels (tests/test_game_variants.py):
  * file   - a declared bug names the file (undeclared_differences);
  * line   - every changed block of lines carries the marker
             "INJECTED BUG <id>" of a declared bug that names that file
             (unattributed_changes), so an extra edit inside a declared file
             is still caught;
  * whole  - the full clean->bugged diff hashes to the manifest's
             `diff_sha256` (bug_diff_sha256): the implementation is exactly
             the one that was validated, and re-validating is the only way to
             change it.

A variant's IDENTITY is the SHA-256 of its game tree: every file under data/
and resources/ (the code and the assets the engine loads), by relative path.
Text files are hashed with CRLF folded to LF so a Windows and a Linux
checkout of the same commit agree. Bytecode caches are ignored; the
per-variant metadata files at the variant root (VARIANT.md,
INJECTED_BUGS.json) are not part of the game and are ignored too.

Both variants are imported as the top-level package `data` (the upstream
layout, left untouched), so a process hosts ONE of them at a time -
custom_mario_env enforces that (release_game_variant unloads one first).
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from typing import Any

from exploration import config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_TREE_PARTS = ("data", "resources")
_TEXT_SUFFIXES = frozenset((".py", ".txt", ".md", ".json"))
BUG_MANIFEST_NAME = "INJECTED_BUGS.json"
BUG_MARKER = re.compile(r"INJECTED BUG ([a-z0-9][a-z0-9-]*)")


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


def _text_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return fh.read().replace("\r\n", "\n").split("\n")


def _changed_files(variant: str) -> list[str]:
    clean = game_tree_files(game_dir(config.CLEAN_GAME_VARIANT))
    other = game_tree_files(game_dir(variant))
    return sorted(f for f in set(clean) | set(other) if clean.get(f) != other.get(f))


def unattributed_changes(variant: str) -> list[str]:
    """Changed blocks of lines ("file:first-last", in the variant) that do not
    carry the marker of a declared bug naming that file. Empty is the only
    acceptable answer. A pure deletion has no line to carry a marker, so it is
    always reported: a bug that removes code must leave a marked line."""
    names = {b["id"]: set(b["files"]) for b in injected_bugs(variant)}
    out = []
    for rel in _changed_files(variant):
        a_path = os.path.join(game_dir(config.CLEAN_GAME_VARIANT), rel)
        b_path = os.path.join(game_dir(variant), rel)
        if not (os.path.exists(a_path) and os.path.exists(b_path)) or \
                os.path.splitext(rel)[1].lower() not in _TEXT_SUFFIXES:
            out.append(f"{rel}: added, removed or binary")
            continue
        a, b = _text_lines(a_path), _text_lines(b_path)
        matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            ids = {m for line in b[j1:j2] for m in BUG_MARKER.findall(line)}
            if not any(rel in names.get(i, ()) for i in ids):
                out.append(f"{rel}:{j1 + 1}-{max(j1 + 1, j2)}")
    return out


def bug_diff_sha256(variant: str) -> str:
    """SHA-256 of the unified clean->variant diff of every changed game file
    (text, LF line endings, files in path order)."""
    h = hashlib.sha256()
    for rel in _changed_files(variant):
        a_path = os.path.join(game_dir(config.CLEAN_GAME_VARIANT), rel)
        b_path = os.path.join(game_dir(variant), rel)
        a = _text_lines(a_path) if os.path.exists(a_path) else []
        b = _text_lines(b_path) if os.path.exists(b_path) else []
        for line in difflib.unified_diff(a, b, f"a/{rel}", f"b/{rel}", n=3, lineterm=""):
            h.update(line.encode("utf-8") + b"\n")
    return h.hexdigest()


def declared_diff_sha256(variant: str) -> str | None:
    """The `diff_sha256` the variant's bug manifest pins (None: no bugs, no pin)."""
    if variant == config.CLEAN_GAME_VARIANT:
        return None
    with open(os.path.join(game_dir(variant), BUG_MANIFEST_NAME), encoding="utf-8") as fh:
        pinned = json.load(fh).get("diff_sha256")
    return pinned if isinstance(pinned, str) else None


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
