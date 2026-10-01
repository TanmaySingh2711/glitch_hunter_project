"""Atomic writes, content hashes and write-protection for project artifacts.

These operations used to be implemented separately wherever a record,
coverage map or mask was written - identical code in several places, so a
fix to one (say, closing the temp file before the rename on Windows) would
silently miss the others. They live here once.

Every write here is ATOMIC: the payload goes to a sibling temp file which is
flushed to disk and then renamed over the target. A crash mid-write leaves
the old file, never a half-written one that would look valid.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import shutil
import stat
from collections.abc import Iterator
from typing import IO, Any

_HASH_BLOCK = 1 << 20          # 1 MiB: bounded memory for 20 MB checkpoints


def sha256_of(path: str | os.PathLike[str]) -> str:
    """Hex SHA-256 of a file's bytes, read in 1 MiB blocks."""
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(_HASH_BLOCK), b''):
            h.update(block)
    return h.hexdigest()


_TEXT_SUFFIXES = frozenset(('.json', '.jsonl', '.md', '.txt', '.toml', '.yml', '.yaml'))


def canonical_sha256(path: str | os.PathLike[str]) -> str:
    """SHA-256 of a file as git stores it, so a hash recorded on one platform
    verifies on every other.

    Binaries (checkpoints, masks) are hashed byte for byte. Text files are
    hashed with CRLF folded to LF: .gitattributes normalises them, so a
    Windows checkout and a Linux one hold different bytes for the same
    committed content, and a raw hash would call one of them corrupted.
    """
    if os.path.splitext(os.fspath(path))[1].lower() not in _TEXT_SUFFIXES:
        return sha256_of(path)
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read().replace(b'\r\n', b'\n')).hexdigest()


def ensure_parent_dir(path: str | os.PathLike[str]) -> None:
    """Creates the directory `path` will be written into, if it is missing."""
    os.makedirs(os.path.dirname(os.fspath(path)) or '.', exist_ok=True)


@contextlib.contextmanager
def atomic_write(path: str | os.PathLike[str], mode: str = 'wb',
                 encoding: str | None = None) -> Iterator[IO[Any]]:
    """Opens a sibling temp file for writing and, when the block succeeds,
    renames it over `path`.

    The temp file is flushed to disk before the rename - otherwise a power
    cut just after it can leave the NEW name pointing at an empty file - and
    is removed if the block raises, so a failed write leaves the old file
    and no debris behind.
    """
    path = os.fspath(path)
    ensure_parent_dir(path)
    tmp = path + '.tmp'
    try:
        with open(tmp, mode, encoding=encoding) as fh:
            yield fh
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp)
        raise


def write_json_atomic(obj: Any, path: str | os.PathLike[str], indent: int = 1) -> None:
    """Writes `obj` as JSON to `path` atomically (see atomic_write)."""
    with atomic_write(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=indent)


def read_json(path: str | os.PathLike[str]) -> Any:
    """Parses a UTF-8 JSON file."""
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def make_read_only(path: str | os.PathLike[str]) -> None:
    """Drops every write permission bit. Used on records written exactly once
    (completion snapshots, verification records) so an accidental second
    write fails loudly instead of replacing the evidence."""
    os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)


# A folder the user deleted is first renamed to this prefix, then emptied.
RETIRED_PREFIX = "_deleted-"


def retire_dir(path: str | os.PathLike[str]) -> str:
    """Renames a folder out of the way in one step and returns the new path.

    A delete that removed files one by one would leave half a folder behind
    if it were interrupted, and a listing could show it. One rename takes the
    whole folder out of every listing at once; the files go afterwards
    (remove_tree). Raises OSError when something still has a file in it open
    (Windows), and then nothing has changed."""
    parent, name = os.path.split(os.path.normpath(os.fspath(path)))
    retired = os.path.join(parent, f"{RETIRED_PREFIX}{name}-{secrets.token_hex(3)}")
    os.replace(path, retired)
    return retired


def remove_tree(path: str | os.PathLike[str]) -> bool:
    """Deletes a folder and everything in it, read-only files included.
    Returns False when something could not be removed (it is left in place
    under its retired name, and the next sweep tries again)."""
    def _writable_then_retry(func: Any, target: str, _exc: BaseException) -> None:
        os.chmod(target, stat.S_IREAD | stat.S_IWRITE)
        func(target)

    try:
        shutil.rmtree(path, onexc=_writable_then_retry)
    except OSError:
        return False
    return True


def sweep_retired(root: str | os.PathLike[str]) -> list[str]:
    """Removes folders an earlier delete could not finish emptying."""
    done: list[str] = []
    try:
        names = os.listdir(root)
    except OSError:
        return done
    for name in names:
        path = os.path.join(root, name)
        if name.startswith(RETIRED_PREFIX) and os.path.isdir(path) and remove_tree(path):
            done.append(name)
    return done
