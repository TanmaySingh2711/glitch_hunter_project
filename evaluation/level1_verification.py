"""Is a Level-1 completion snapshot the FINAL Level-1 brain?

Covering every testable pixel (exploration/level_completion.py) produces a
CANDIDATE. It becomes the final Level-1 brain only when all of these hold,
checked independently of the training run that produced it:

  1. INTEGRITY  the policy and coverage files still hash to what the proof
                recorded; the coverage re-verifies against the live mask
                (fingerprint, denominator, config hash, its own counts, the
                checkpoint's timestep) and re-counts to exactly 4,013,723; the
                zip's own timestep is the proof's; no policy update ran after
                completion.
  2. HEALTH     the policy loads and every parameter is finite.
  3. RETENTION  the policy can still finish Level 1-1: the frozen completion
                protocol (evaluation/completion.py - same seeds, same sampled
                play, same engine settings) against the 6M baseline gives
                HEALTHY.

Verdicts: VERIFIED (all three, HEALTHY), NEEDS_REVIEW (1-2 pass, retention
WARNING - a person decides), REJECTED (anything else). Retention is not even
attempted on a candidate that fails 1 or 2.

The record is written ONCE, read-only, beside the snapshot - the snapshot's
own three files are never touched. Only a VERIFIED record names a final
brain. Nothing here trains, and nothing starts on another level.
"""
import datetime
import json
import os
import stat
import zipfile

from evaluation import completion as ce
from exploration import config
from exploration import coverage as coverage_mod
from exploration import level_completion as lc

VERIFIED, NEEDS_REVIEW, REJECTED = "VERIFIED", "NEEDS_REVIEW", "REJECTED"
_BY_RETENTION = {'HEALTHY': VERIFIED, 'WARNING': NEEDS_REVIEW, 'REGRESSED': REJECTED}


def verification_path(proof_path):
    return proof_path[:-len('.json')] + '_verification.json'


def _resolve(path, root):
    return path if os.path.isabs(path) else os.path.join(root, path)


def check_integrity(meta, root):
    """(failures, facts) for the snapshot a proof describes."""
    failures, facts = [], {}
    expect = {'kind': lc.SNAPSHOT_KIND, 'testable_total': config.TESTABLE_TOTAL,
              'covered_testable_px': config.TESTABLE_TOTAL, 'remaining_testable_px': 0,
              'policy_updates_since_completion': 0}
    for key, want in expect.items():
        if meta.get(key) != want:
            failures.append(f"proof {key} is {meta.get(key)!r}, expected {want!r}")
    t = int(meta.get('global_timestep', -1))
    model = _resolve(meta['model']['path'], root)
    cov_path = _resolve(meta['coverage']['path'], root)
    for label, path, digest in (('policy', model, meta['model']['sha256']),
                                ('coverage', cov_path, meta['coverage']['sha256'])):
        if not os.path.exists(path):
            failures.append(f"{label} file missing: {path}")
            continue
        actual = lc.sha256_of(path)
        facts[f'{label}_sha256'] = actual
        if actual != digest:
            failures.append(f"{label} file changed since completion (sha256 {actual[:12]}..., "
                            f"proof says {digest[:12]}...)")
    if failures:
        return failures, facts
    try:
        with zipfile.ZipFile(model) as z:
            zip_t = int(json.loads(z.read('data'))['num_timesteps'])
    except Exception as exc:
        failures.append(f"policy zip unreadable: {exc}")
        return failures, facts
    if zip_t != t:
        failures.append(f"policy zip is at step {zip_t:,}, the proof says {t:,}")
    mask = coverage_mod.load_testable()
    if mask is None:
        failures.append("the verified testable mask is not available")
        return failures, facts
    try:
        cov = coverage_mod.SpatialCoverage(testable_mask=mask)
        cov.load_verified(cov_path, expected_timesteps=t)
        covered = cov.covered_testable()
        facts['covered_testable_px'] = covered
        if not lc.is_level_complete(covered, cov.testable_total):
            failures.append(f"coverage re-counts to {covered:,}, not {config.TESTABLE_TOTAL:,}")
        facts['provenance'] = lc.provenance(cov)
    except Exception as exc:
        failures.append(f"coverage does not re-verify: {type(exc).__name__}: {exc}")
    return failures, facts


def check_health(model):
    """(failures, facts): every parameter of the policy must be finite."""
    import torch
    n, bad = 0, []
    for name, p in model.policy.named_parameters():
        n += p.numel()
        if not bool(torch.isfinite(p).all()):
            bad.append(name)
    facts = {'parameters': n}
    return ([f"non-finite parameters: {', '.join(bad)}"] if bad else []), facts


def verify(proof_path, baseline_path=ce.BASELINE_PATH, workers=1, root=None,
           evaluate=ce.evaluate, compare=ce.compare, load_policy=ce.load_policy,
           results_dir=ce.RESULTS_DIR, log=print):
    """Runs the three checks and writes the record. Returns the record."""
    root = root or os.getcwd()
    out = verification_path(proof_path)
    if os.path.exists(out):
        raise FileExistsError(f"{out} already exists; a verification is written once")
    with open(proof_path, encoding='utf-8') as fh:
        meta = json.load(fh)
    model_path = _resolve(meta['model']['path'], root)
    record = {'kind': lc.VERIFICATION_KIND, 'level': '1-1', 'proof': proof_path.replace('\\', '/'),
              'global_timestep': meta.get('global_timestep'), 'checks': {}}

    failures, facts = check_integrity(meta, root)
    record['checks']['integrity'] = {'passed': not failures, 'failures': failures, **facts}
    if not failures:
        log("[VERIFY] integrity: passed")
        model = load_policy(model_path)
        failures, facts = check_health(model)
        record['checks']['health'] = {'passed': not failures, 'failures': failures, **facts}
    if failures:
        record['verdict'] = REJECTED
        record['reasons'] = failures
    else:
        log("[VERIFY] health: passed; running the completion-retention protocol...")
        baseline = ce.load(baseline_path)
        result = evaluate(model_path, baseline['protocol'], workers=workers)
        comparison = compare(result, baseline)
        result['comparison'] = comparison
        saved = os.path.join(results_dir, f"level1_verification_{int(meta['global_timestep'])}"
                                          f"_{meta['model']['sha256'][:8]}.json")
        ce.save(result, saved)
        record['checks']['retention'] = {
            'passed': comparison['verdict'] == 'HEALTHY', 'result': saved.replace('\\', '/'),
            'baseline': os.path.relpath(baseline_path, root).replace('\\', '/'),
            **{k: comparison[k] for k in ('verdict', 'reasons', 'completion_rate',
                                          'baseline_completion_rate', 'completion_delta',
                                          'mean_progress', 'baseline_mean_progress',
                                          'greedy_completes')}}
        record['verdict'] = _BY_RETENTION[comparison['verdict']]
        record['reasons'] = comparison['reasons']
    record['final_level1_brain'] = (
        {'path': meta['model']['path'], 'sha256': meta['model']['sha256'],
         'coverage': meta['coverage']['path']}
        if record['verdict'] == VERIFIED else None)
    record['created'] = datetime.datetime.now().isoformat(timespec='seconds')
    with open(out, 'x', encoding='utf-8') as fh:            # written once
        json.dump(record, fh, indent=1)
    os.chmod(out, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    record['_path'] = out
    return record
