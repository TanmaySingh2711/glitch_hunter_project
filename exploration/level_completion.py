"""Level 1 is complete when coverage says so - not when a step count does.

THE CONDITION, exactly:

    covered_testable_pixels == TESTABLE_TOTAL          (4,013,723)

Integer equality against the verified denominator (Method C, fingerprinted).
Never a percentage, never rounded: at 4,013,722 covered the level is NOT
complete, even though a two-decimal percentage would print "100.00%". Every
coverage figure shown during QA training goes through pct_text(), which can
only print 100% when the two integers are equal.

Timesteps still number the checkpoints and the logs. They are not the target.

AT COMPLETION (write_completion_snapshot) three files are written, once, into
their own directory beside the periodic checkpoints - never on top of them,
and never anywhere near the 6M master:

    <name>_level1_complete_<T>_steps.zip            the policy, as it was
    <name>_level1_complete_<T>_steps_coverage.npz   the exact coverage state
    <name>_level1_complete_<T>_steps.json           the proof, written LAST

The JSON is the commit point: it is only written after both other files, and
after the coverage file has been read back and re-counted. All three are then
made read-only. A directory that already holds a committed completion
refuses a second one.

BELOW 100% (remaining_regions / plateau_report) the leftover testable pixels
are exposed for a later audit - how many, where they cluster, and whether
discovery has stalled - without deleting or reclassifying any of them.
write_remaining_map draws the same thing as a picture of the level.

COMPLETE IS NOT YET FINAL. A completed snapshot is a candidate: it becomes
the final Level-1 brain only when tools/verify_level1.py has re-checked its
files and shown the policy can still finish the level (evaluation/
level1_verification.py). Nothing starts on another level either way.
"""
from __future__ import annotations

import datetime
import glob
import json
import math
import os
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

from common.fileio import make_read_only, sha256_of, write_json_atomic

from . import config

if TYPE_CHECKING:
    from .coverage import SpatialCoverage

__all__ = ['LevelAlreadyCompleted', 'PlateauTracker', 'committed_snapshots',
           'final_level1_brain', 'is_level_complete', 'pct_text', 'plateau_report',
           'provenance', 'remaining_regions', 'sha256_of', 'status',
           'write_completion_snapshot', 'write_json_atomic', 'write_remaining_map']

SNAPSHOT_KIND = "glitch_hunter_level1_completion"
SNAPSHOT_DIRNAME = "level1_complete"
VERIFICATION_KIND = "glitch_hunter_level1_verification"


class LevelAlreadyCompleted(RuntimeError):
    """A committed Level-1 completion snapshot already exists."""


# ══════════════════════════════════════════════════════════════════════════
# THE CONDITION
# ══════════════════════════════════════════════════════════════════════════
def is_level_complete(covered: int, testable_total: int) -> bool:
    """True exactly when every verified testable pixel is covered.

    Refuses a denominator other than the verified one rather than answering
    against it: a completion claimed on a different mask is not Level 1.
    """
    if int(testable_total) != config.TESTABLE_TOTAL:
        raise ValueError(
            f"testable_total {int(testable_total):,} is not the verified "
            f"denominator {config.TESTABLE_TOTAL:,}")
    covered = int(covered)
    if covered > config.TESTABLE_TOTAL:
        raise ValueError(f"covered {covered:,} exceeds the denominator - impossible")
    return covered == config.TESTABLE_TOTAL


def pct_text(covered: int, total: int = config.TESTABLE_TOTAL, places: int = 4) -> str:
    """Coverage as a percentage that can never round UP to 100.

    Truncated, not rounded: 4,013,722 of 4,013,723 is "99.9999%", where
    f"{pct:.2f}" would have said "100.00%" about a level that is not done.
    """
    covered, total = int(covered), int(total)
    if total <= 0:
        return "n/a"
    if covered == total:
        return "100%"
    scale = 10 ** places
    truncated = math.floor(covered * 100 * scale / total) / scale
    return f"{truncated:.{places}f}%"


def status(coverage: SpatialCoverage, timesteps: int,
           session_start_covered: int) -> dict[str, Any]:
    """Every number the trainer reports, from one exact count."""
    covered = coverage.covered_testable()
    total = int(coverage.testable_total or 0)
    return {
        'global_timestep': int(timesteps),
        'testable_total': int(total),
        'covered_testable_px': int(covered),
        'remaining_testable_px': int(total - covered),
        'coverage_pct_text': pct_text(covered, total),
        'session_new_px': int(covered - session_start_covered),
        'level1_complete': is_level_complete(covered, total),
    }


# ══════════════════════════════════════════════════════════════════════════
# BELOW 100%: WHAT IS LEFT, WHERE, AND WHETHER IT IS MOVING
# ══════════════════════════════════════════════════════════════════════════
def remaining_mask(coverage: SpatialCoverage) -> np.ndarray:
    """Testable pixels not yet visited, on the padded grid."""
    if coverage.testable is None:
        raise ValueError("remaining pixels need the testable mask")
    rem: np.ndarray = coverage.testable & (coverage.visited == 0)
    return rem


def remaining_regions(coverage: SpatialCoverage, top: int = 10,
                      cell: int = config.FRONTIER_CELL) -> dict[str, Any]:
    """Clusters of the remaining pixels, cheaply.

    Pixels are binned into cell x cell blocks (the frontier's 40 px), and
    blocks that touch (8-neighbour) form one region. Each region reports its
    exact pixel count and its exact world-space pixel bounding box. Nothing
    here removes or relabels a pixel - it only says where they are.
    """
    rem = remaining_mask(coverage)
    total = int(np.count_nonzero(rem))
    if total == 0:
        return {'remaining_px': 0, 'regions': 0, 'largest': []}
    h, w = rem.shape
    ph, pw = -h % cell, -w % cell                   # pad, never crop: every pixel counts
    padded = np.pad(rem, ((0, ph), (0, pw)))
    ch, cw = padded.shape[0] // cell, padded.shape[1] // cell
    counts = padded.reshape(ch, cell, cw, cell).sum(axis=(1, 3))

    label = np.zeros((ch, cw), dtype=np.int32)
    regions: list[dict[str, Any]] = []
    for sy, sx in zip(*np.nonzero(counts), strict=True):
        if label[sy, sx]:
            continue
        rid = len(regions) + 1
        label[sy, sx] = rid
        stack, cells = [(sy, sx)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < ch and 0 <= nx < cw and counts[ny, nx]
                            and not label[ny, nx]):
                        label[ny, nx] = rid
                        stack.append((ny, nx))
        # Exact pixel bbox from the region's own blocks only - never a
        # full-grid pass per region.
        bx0 = by0 = 1 << 30
        bx1 = by1 = -1
        pixels = 0
        for y, x in cells:
            py, px = np.nonzero(rem[y * cell:(y + 1) * cell, x * cell:(x + 1) * cell])
            pixels += py.size
            by0, by1 = min(by0, y * cell + py.min()), max(by1, y * cell + py.max())
            bx0, bx1 = min(bx0, x * cell + px.min()), max(bx1, x * cell + px.max())
        regions.append({
            'pixels': int(pixels),
            'cells': len(cells),
            'world_bbox': [int(bx0 + coverage.x0), int(by0 + coverage.y0),
                           int(bx1 + coverage.x0), int(by1 + coverage.y0)],
        })
    regions.sort(key=lambda r: -int(r['pixels']))
    return {'remaining_px': total, 'regions': len(regions), 'largest': regions[:top]}


class PlateauTracker:
    """Whether discovery has stalled, on the trainer's report cadence.

    Stalled means the exact covered count did not rise across the last
    `window` reports - the same STAGNATION_WINDOW the stagnation callback
    uses, so "stalled" means one thing in this project, not two.
    """

    def __init__(self, window: int = config.STAGNATION_WINDOW) -> None:
        self.window = window
        self.gains: deque[int] = deque(maxlen=window)
        self.last_covered: int | None = None
        self.last_gain_timestep: int | None = None

    def update(self, timesteps: int, covered: int) -> dict[str, Any]:
        if self.last_covered is not None:
            gain = covered - self.last_covered
            self.gains.append(gain)
            if gain > 0:
                self.last_gain_timestep = timesteps
        elif self.last_gain_timestep is None:
            self.last_gain_timestep = timesteps
        self.last_covered = covered
        stalled = len(self.gains) == self.window and not any(self.gains)
        last_gain = timesteps if self.last_gain_timestep is None else self.last_gain_timestep
        return {
            'stalled': stalled,
            'recent_window_gains_px': list(self.gains),
            'steps_since_last_gain': int(timesteps - last_gain),
        }


def plateau_report(coverage: SpatialCoverage, timesteps: int, tracker_state: dict[str, Any],
                   top: int = 10) -> dict[str, Any]:
    """What a remaining-pixel audit needs; written by the trainer late in a campaign."""
    total = int(coverage.testable_total or 0)
    covered = coverage.covered_testable()        # one full-grid count, used three ways
    return {
        'kind': 'glitch_hunter_level1_remaining_audit',
        'global_timestep': int(timesteps),
        'testable_total': total,
        'covered_testable_px': covered,
        'remaining_testable_px': total - covered,
        'coverage_pct_text': pct_text(covered, total),
        'level1_complete': False,
        'note': ("Remaining pixels are reported, never removed or reclassified. "
                 "The exact set is testable & ~visited from the latest saved coverage."),
        **tracker_state,
        'remaining_regions': remaining_regions(coverage, top=top),
        'testable_fingerprint': config.TESTABLE_FINGERPRINT,
        'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
    }


def remaining_map_image(coverage: SpatialCoverage, scale: int = 4) -> np.ndarray:
    """The whole padded grid as a BGR picture, one pixel per scale x scale
    block: RED where any testable pixel is still unvisited, grey where the
    testable pixels are all covered, black outside the testable mask.

    A block is red if it holds even ONE remaining pixel, so an isolated
    island - a single ledge nobody reached, or a sliver the reachability mask
    may have got wrong - is still visible at a glance."""
    rem = remaining_mask(coverage)
    testable = coverage.testable
    assert testable is not None                  # remaining_mask() refuses otherwise
    h, w = rem.shape
    ph, pw = -h % scale, -w % scale

    def blocks(a: np.ndarray) -> np.ndarray:
        a = np.pad(a, ((0, ph), (0, pw)))
        return a.reshape(a.shape[0] // scale, scale, a.shape[1] // scale, scale).any(axis=(1, 3))

    img = np.zeros(((h + ph) // scale, (w + pw) // scale, 3), dtype=np.uint8)
    img[blocks(testable)] = (90, 90, 90)
    img[blocks(rem)] = (40, 40, 255)
    return img


def write_remaining_map(coverage: SpatialCoverage, png_path: str,
                        regions: dict[str, Any] | None = None,
                        scale: int = 4) -> str:
    """Writes remaining_map_image() with the largest remaining regions boxed
    in yellow and numbered as in `regions` (remaining_regions()). Atomic."""
    import cv2
    img = remaining_map_image(coverage, scale)
    for i, r in enumerate((regions or {}).get('largest', []), start=1):
        x0, y0, x1, y1 = r['world_bbox']
        p0 = ((x0 - coverage.x0) // scale - 2, (y0 - coverage.y0) // scale - 2)
        p1 = ((x1 - coverage.x0) // scale + 2, (y1 - coverage.y0) // scale + 2)
        cv2.rectangle(img, p0, p1, (0, 230, 255), 1)
        cv2.putText(img, str(i), (p0[0], max(10, p0[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (0, 230, 255), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(png_path) or '.', exist_ok=True)
    tmp = png_path[:-4] + '.tmp.png'
    if not cv2.imwrite(tmp, img):
        raise OSError(f"could not write {tmp}")
    os.replace(tmp, png_path)
    return png_path


# ══════════════════════════════════════════════════════════════════════════
# PROVENANCE: BOOTSTRAP-KNOWN vs QA-DISCOVERED
# ══════════════════════════════════════════════════════════════════════════
BootstrapBits = tuple[np.ndarray, dict[str, Any]]
_BOOTSTRAP_BITS: dict[tuple[str, float, tuple[int, ...]], BootstrapBits] = {}


def _bootstrap_bits(path: str, testable: np.ndarray) -> tuple[BootstrapBits | None, str | None]:
    """The bootstrap's covered testable pixels (bool, padded grid) and the
    file's facts, or (None, reason). Cached per file version."""
    if not os.path.exists(path):
        return None, f"{path} not found"
    key = (os.path.abspath(path), os.path.getmtime(path), testable.shape)
    if key not in _BOOTSTRAP_BITS:
        with np.load(path, allow_pickle=False) as npz:
            fp = str(npz['testable_fingerprint'])
            if fp != config.TESTABLE_FINGERPRINT:
                return None, f"{path} was recorded against mask {fp[:16]}..."
            n = testable.size
            bits = np.unpackbits(npz['visited_packed'])[:n].reshape(testable.shape)
            facts = {'file': path.replace('\\', '/'),
                     'model_timesteps': int(npz['model_timesteps'])}
        _BOOTSTRAP_BITS.clear()
        _BOOTSTRAP_BITS[key] = (bits.astype(bool) & testable, facts)
    return _BOOTSTRAP_BITS[key], None


def provenance(coverage: SpatialCoverage,
               bootstrap_path: str = config.BOOTSTRAP_COVERAGE) -> dict[str, Any]:
    """Splits the covered testable pixels by where they came from.

    Every covered pixel counts toward completion the same way; this only
    RECORDS the split. 'Bootstrap-known' means covered by the replay of the
    frozen 6M policy at step 6,000,000 (tools/bootstrap_coverage.py), which
    seeded the campaign. Coverage during 0-6M training was never recorded,
    and nothing here reconstructs it: anything that replay missed counts as
    discovered by QA training, when QA training finds it.

    bootstrap_px_still_covered should always equal bootstrap_known: a
    covered pixel is never uncovered within a campaign.
    """
    if coverage.testable is None:
        return {'available': False, 'reason': 'no testable mask'}
    found, reason = _bootstrap_bits(bootstrap_path, coverage.testable)
    if found is None:
        return {'available': False, 'reason': reason}
    boot, facts = found
    covered = (coverage.visited != 0) & coverage.testable
    total = int(np.count_nonzero(covered))
    from_boot = int(np.count_nonzero(covered & boot))
    known = int(np.count_nonzero(boot))
    return {
        'available': True,
        'bootstrap': facts,
        'bootstrap_known_testable_px': known,
        'bootstrap_px_still_covered': from_boot,
        'qa_discovered_testable_px': total - from_boot,
        'covered_testable_px': total,
    }


# ══════════════════════════════════════════════════════════════════════════
# AT 100%: THE IMMUTABLE SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════
Record = tuple[str, dict[str, Any]]


def _json_records(pattern: str, kind: str) -> list[Record]:
    """(path, record) for every readable JSON file matching `pattern` whose
    'kind' is `kind`, sorted by path. Unreadable files are skipped: a stray or
    half-written file must not hide - or impersonate - a real record."""
    found = []
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p, encoding='utf-8') as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict) and rec.get('kind') == kind:
            found.append((p, rec))
    return found


def committed_snapshots(snapshot_dir: str) -> list[Record]:
    """The completion proofs already in `snapshot_dir`, oldest first."""
    return _json_records(os.path.join(snapshot_dir, '*.json'), SNAPSHOT_KIND)


def final_level1_brain(snapshot_dir: str) -> Record | None:
    """(verification path, record) for a VERIFIED completion, else None.

    The final Level-1 brain is never simply the latest checkpoint, nor the
    completion snapshot on its own: only a verification record that passed
    integrity, health AND completion retention names one
    (evaluation/level1_verification.py)."""
    for p, rec in _json_records(os.path.join(snapshot_dir, '*_verification.json'),
                                VERIFICATION_KIND):
        if rec.get('verdict') == 'VERIFIED':
            return p, rec
    return None


def write_completion_snapshot(model: Any, coverage: SpatialCoverage, snapshot_dir: str,
                              name_prefix: str, timesteps: int, session_start: dict[str, int],
                              previous_check: tuple[int, int] | None) -> dict[str, Any]:
    """Saves the policy, the exact coverage and the proof of completion.

    Called the moment completion is detected, between periodic checkpoints
    if that is when it happens. `previous_check` is (timesteps, covered) at
    the last check that was NOT complete: completion happened in between,
    and no PPO update ran in that interval (see Level1CompletionCallback).
    """
    covered = coverage.covered_testable()
    if coverage.testable is None or not is_level_complete(covered,
                                                          int(coverage.testable_total or 0)):
        raise ValueError(f"not complete: {covered:,} of {coverage.testable_total:,}")
    if committed_snapshots(snapshot_dir):
        raise LevelAlreadyCompleted(
            f"{snapshot_dir} already holds a committed Level-1 completion; "
            f"it is immutable and will not be replaced")

    os.makedirs(snapshot_dir, exist_ok=True)
    stem = os.path.join(snapshot_dir, f"{name_prefix}_level1_complete_{int(timesteps)}_steps")
    zip_path, cov_path, meta_path = stem + '.zip', stem + '_coverage.npz', stem + '.json'

    # Model, then coverage, then the proof - the same commit order as every
    # periodic checkpoint in this project.
    model.save(zip_path)
    coverage.save(cov_path, model_timesteps=int(timesteps))

    # Read the coverage back and count it again: the proof describes the file
    # on disk, not the array in memory.
    n = coverage.w * coverage.h
    with np.load(cov_path, allow_pickle=False) as d:
        bits = np.unpackbits(d['visited_packed'])[:n].reshape(coverage.h, coverage.w)
        written = {'config_hash': str(d['config_hash']),
                   'model_timesteps': int(d['model_timesteps'])}
    on_disk = int(np.count_nonzero(bits.astype(bool) & coverage.testable))
    if on_disk != config.TESTABLE_TOTAL:
        raise RuntimeError(f"written coverage re-counts to {on_disk:,}, not "
                           f"{config.TESTABLE_TOTAL:,}; completion NOT committed")

    try:
        noncov: dict[str, Any] = {'noncoverage_px': coverage.noncoverage_px(),
                                  'anomalous_px': coverage.anomalous_px()}
    except Exception as exc:                      # the class map is diagnostic only
        noncov = {'noncoverage_px': coverage.noncoverage_px(),
                  'anomalous_px': None, 'anomalous_error': str(exc)}
    try:
        origin = provenance(coverage)
    except Exception as exc:                      # a record, never a gate
        origin = {'available': False, 'reason': str(exc)}

    root = os.getcwd()

    def rel(p: str) -> str:
        return os.path.relpath(os.path.abspath(p), root).replace('\\', '/')

    meta = {
        'kind': SNAPSHOT_KIND,
        'level': '1-1',
        'condition': 'covered_testable_pixels == testable_total',
        'covered_testable_px': on_disk,
        'testable_total': config.TESTABLE_TOTAL,
        'remaining_testable_px': 0,
        'coverage_pct': 100.0,
        'global_timestep': int(timesteps),
        'detected_after': ({'global_timestep': int(previous_check[0]),
                            'covered_testable_px': int(previous_check[1])}
                           if previous_check else None),
        'policy_updates_since_completion': 0,
        'session': {**session_start,
                    'new_px_this_session': on_disk - int(session_start['covered_testable_px'])},
        'model': {'path': rel(zip_path), 'sha256': sha256_of(zip_path)},
        'coverage': {'path': rel(cov_path), 'sha256': sha256_of(cov_path), **written},
        'reachability': {'testable_fingerprint': config.TESTABLE_FINGERPRINT,
                         'adopted_method': config.ADOPTED_METHOD},
        'provenance': origin,
        **noncov,
        # Coverage is complete; that alone does not make this the final
        # Level-1 brain. tools/verify_level1.py decides that, separately.
        'verification': 'pending - run tools/verify_level1.py',
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    with open(meta_path, 'x', encoding='utf-8') as fh:      # 'x': never over an existing proof
        json.dump(meta, fh, indent=1)
    for p in (zip_path, cov_path, meta_path):
        make_read_only(p)
    meta['_paths'] = {'model': zip_path, 'coverage': cov_path, 'metadata': meta_path}
    return meta
