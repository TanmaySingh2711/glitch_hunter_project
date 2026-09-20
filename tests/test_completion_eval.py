"""Completion retention evaluation (evaluation/completion.py).

The evaluator plays checkpoints on the bare engine and compares them against
the 6M baseline. What has to hold for that comparison to mean anything:
episodes are classified right, max-x is the real max, the same checkpoint and
seeds give the same answer, and evaluating touches neither the weights nor
any QA campaign state - nor anything outside the evaluator.

Scripted actors drive the classification tests, so they are exact and fast.
The tests marked slow load the real checkpoint. Numbers in brackets are the
brief's test IDs.
"""
import hashlib
import os

import numpy as np
import pytest

from evaluation import completion as ce
from exploration import config
from exploration.lifecycle import EndReason

ROOT = ce.ROOT
PROTOCOL = ce.make_protocol(episodes=2, seed=900_000)
SIX_M_FILES = ("mario_brain_checkpoint.zip",
               "checkpoints/mario_brain_checkpoint_6000000_steps.zip",
               "backup_6M/mario_brain_checkpoint.zip",
               "backup_6M/mario_brain_checkpoint_6000000_steps.zip")
BASELINE_MODEL = os.path.join(ROOT, config.BASELINE_MODEL)
needs_6m = pytest.mark.skipif(not os.path.exists(BASELINE_MODEL),
                              reason="backup_6M/ is not in the repository")


def scripted(actions, then=0):
    """An actor that plays a fixed agent-step script, then `then` forever."""
    it = iter(actions)

    def act(_obs, _rng):
        return next(it, then)
    return act


def clock(units):
    def set_clock(base):
        base.game.state.overhead_info_display.time = units
    return set_clock


def at_flagpole(base):
    base.game.state.mario.rect.x = 8400


def play(env, actor, seed=0, on_reset=None, protocol=PROTOCOL):
    with ce.protocol_env(env, protocol) as (e, probe):
        return ce.run_episode(e, probe, actor, seed, on_reset)


def _files_digest(paths):
    out = {}
    for p in paths:
        full = os.path.join(ROOT, p)
        if os.path.exists(full):
            out[p] = ce.sha256_of(full)
    return out


def _campaign_files():
    """Every file a QA campaign keeps between runs, as far as it exists."""
    # Files only, and into sub-folders: exploration_data/ and checkpoints_qa/
    # both now hold archive directories (the superseded mask, the per-run
    # telemetry), and hashing a directory raises PermissionError on Windows.
    paths = []
    for folder, _dirs, names in sorted(os.walk(os.path.join(ROOT, 'exploration_data'))):
        paths += [os.path.relpath(os.path.join(folder, n), ROOT) for n in sorted(names)]
    qa_dir = os.path.join(ROOT, config.CHECKPOINT_DIR_QA)
    if os.path.isdir(qa_dir):
        # Walk into sub-folders: checkpoints_qa/pre_main_6032768/ holds the
        # healthy rollback pair, and it is exactly what must stay byte-identical.
        for folder, _dirs, names in sorted(os.walk(qa_dir)):
            paths += [os.path.relpath(os.path.join(folder, n), ROOT) for n in sorted(names)]
    paths += ["glitch_hunter_qa.zip", "glitch_hunter_qa_coverage.npz"]
    return paths


# ══════════════════════════════════════════════════════════════════════════
# [4] [5] [6] CLASSIFICATION AND MEASUREMENT
# ══════════════════════════════════════════════════════════════════════════
def test_castle_completion_is_counted(env):                             # [4]
    r = play(env, scripted([], then=2), on_reset=at_flagpole)
    assert r['end'] == EndReason.LEVEL_COMPLETE
    assert r['castle_door'] and r['flagpole']
    assert r['completion_agent_step'] == r['agent_steps']
    assert r['max_x'] >= ce.CASTLE_DOOR_X - 6 and r['progress'] == 1.0
    assert r['death_cause'] is None
    s = ce.summarize([r])
    assert (s['completed'], s['castle_door'], s['flagpole']) == (1, 1, 1)
    assert s['completion_rate'] == 1.0


def test_death_and_timeout_are_classified_separately(env):              # [5]
    died = play(env, scripted([1] * 15))                # walk into the first goomba
    timed_out = play(env, scripted([]), on_reset=clock(3))
    assert died['end'] == EndReason.DEATH and died['death_cause'] not in (None, 'timeout')
    assert timed_out['end'] == EndReason.TIMEOUT and timed_out['death_cause'] is None
    for r in (died, timed_out):
        assert not r['castle_door'] and r['completion_agent_step'] is None
    s = ce.summarize([died, timed_out])
    assert s['ends'] == {EndReason.DEATH: 1, EndReason.TIMEOUT: 1}
    assert s['completed'] == 0 and 'timeout' not in s['death_causes']


def test_max_x_is_the_max_over_every_substep(env, patch_step):         # [6]
    """MaxAndSkip passes up one info in four; the max must come from all of
    them. Walk right, then back left, until a short clock ends it."""
    xs, real = [], env.step

    def recording(action):
        out = real(action)
        xs.append(out[4]['x_pos'])
        return out
    patch_step(recording)
    r = play(env, scripted([1] * 10 + [6] * 30), on_reset=clock(3))
    assert r['end'] == EndReason.TIMEOUT
    assert r['max_x'] == max(xs) and xs[-1] < max(xs), "the max was not mid-episode"
    assert r['substeps'] == len(xs)
    assert r['progress'] == round(ce.progress_of(max(xs)), 5)
    assert ce.progress_of(ce.SPAWN_X) == 0.0 and ce.progress_of(ce.CASTLE_DOOR_X) == 1.0


def test_the_evaluation_env_is_the_bare_engine(env):
    """No reward wrapper, so no coverage, novelty or lifecycle can reach it."""
    from agent_logic import GlitchHunterWrapper
    with ce.protocol_env(env, PROTOCOL) as (e, _probe):
        layer, layers = e, []
        while hasattr(layer, 'env'):
            layers.append(type(layer).__name__)
            assert not isinstance(layer, GlitchHunterWrapper)
            layer = layer.env
        assert layer is env
        assert env.episode_time_units == config.QA_EPISODE_TIME_UNITS
        assert env.end_on_level_complete is True
    assert layers == ['TimeLimit', 'FrameStackObservation', 'ResizeObservation',
                      'GrayscaleObservation', 'SkipObservation', 'EpisodeProbe']


# ══════════════════════════════════════════════════════════════════════════
# [7] [8] COMPARISON AND VERDICTS
# ══════════════════════════════════════════════════════════════════════════
THRESHOLDS = {'completion_rate': {'warning_below': 0.40, 'regressed_below': 0.30},
              'mean_progress': {'warning_below': 0.65, 'regressed_below': 0.55}}


def _summary(rate, progress, **extra):
    return {'completion_rate': rate, 'progress': {'mean': progress},
            'max_x': {'median': 5000.0}, **extra}


@pytest.mark.parametrize(("rate", "progress", "want"), [
    (0.46, 0.72, 'HEALTHY'),
    (0.40, 0.65, 'HEALTHY'),          # exactly on a line is not below it
    (0.39, 0.72, 'WARNING'),
    (0.30, 0.72, 'WARNING'),
    (0.29, 0.72, 'REGRESSED'),
    (0.46, 0.60, 'WARNING'),          # progress fallback can downgrade...
    (0.46, 0.50, 'REGRESSED'),
    (0.35, 0.60, 'WARNING'),          # ...and never double-counts
    (0.29, 0.99, 'REGRESSED'),        # ...but can never rescue completion
])
def test_verdict_follows_the_thresholds(rate, progress, want):          # [8]
    assert ce.classify(_summary(rate, progress), THRESHOLDS)[0] == want


def test_coverage_gains_cannot_make_a_checkpoint_healthy():
    s = _summary(0.10, 0.40, coverage_pct=99.0, new_px=10 ** 9)
    verdict, reasons = ce.classify(s, THRESHOLDS)
    assert verdict == 'REGRESSED' and reasons


def _result(protocol, rate, progress, steps):
    return {'protocol': protocol, 'summary': _summary(rate, progress),
            'checkpoint': {'num_timesteps': steps},
            'greedy': {'end': EndReason.DEATH}, 'thresholds': THRESHOLDS}


def test_a_future_checkpoint_is_compared_against_the_baseline():        # [7]
    base = _result(PROTOCOL, 0.46, 0.72, 6_000_000)
    c = ce.compare(_result(PROTOCOL, 0.35, 0.70, 6_400_000), base)
    assert c['verdict'] == 'WARNING'
    assert (c['checkpoint_timesteps'], c['baseline_timesteps']) == (6_400_000, 6_000_000)
    assert c['completion_delta'] == pytest.approx(-0.11)
    assert c['progress_delta'] == pytest.approx(-0.02)


def test_a_different_protocol_is_refused():
    base = _result(PROTOCOL, 0.46, 0.72, 6_000_000)
    other = dict(PROTOCOL, episodes=PROTOCOL['episodes'] + 1)
    with pytest.raises(ce.ProtocolMismatch):
        ce.compare(_result(other, 0.46, 0.72, 6_400_000), base)


def test_thresholds_are_the_protocol_noise_in_units_of_z():
    """A Bernoulli rate p over n episodes: the no-change spread of (run A -
    run B) is sqrt(2 p (1 - p) / n). The bootstrap must land on it."""
    n, k = 500, 230
    recs = ([{'end': EndReason.LEVEL_COMPLETE, 'progress': 1.0}] * k
            + [{'end': EndReason.DEATH, 'progress': 0.5}] * (n - k))
    t = ce.derive_thresholds(recs, 2.0, 4.0, rounds=20_000)
    p = k / n
    sd = (2 * p * (1 - p) / n) ** 0.5
    got = t['completion_rate']
    assert got['noise_sd_of_difference'] == pytest.approx(sd, rel=0.03)
    assert got['warning_below'] == pytest.approx(p - 2 * got['noise_sd_of_difference'], abs=1e-3)
    assert got['regressed_below'] == pytest.approx(p - 4 * got['noise_sd_of_difference'], abs=1e-3)


def test_the_stored_baseline_is_self_consistent():
    """The file every future comparison reads: the 6M master, its summary
    and thresholds re-derivable from its own episodes."""
    if not os.path.exists(ce.BASELINE_PATH):
        pytest.skip("baseline not measured yet")
    b = ce.load(ce.BASELINE_PATH)
    assert b['kind'] == 'completion_retention_baseline'
    assert b['checkpoint']['is_6m_master'] and b['checkpoint']['sha256'] == ce.SIX_M_SHA256
    assert b['protocol'] == ce.make_protocol(b['protocol']['episodes'], b['protocol']['seed'])
    assert [r['seed'] for r in b['episodes']] == ce.episode_seeds(b['protocol'])
    assert ce.summarize(b['episodes']) == b['summary']
    t = b['thresholds']
    assert ce.derive_thresholds(b['episodes'], t['z_warning'], t['z_regressed'],
                                rounds=t['bootstrap_rounds']) == t
    # The baseline must itself be HEALTHY against its own thresholds.
    assert ce.classify(b['summary'], t)[0] == 'HEALTHY'


# ══════════════════════════════════════════════════════════════════════════
# [9] NOTHING OUTSIDE THE EVALUATOR MOVES
# ══════════════════════════════════════════════════════════════════════════
def test_the_shared_env_is_left_as_found(fresh):                       # [9]
    """Engine settings are restored after the episode, and after a failure
    inside one; legacy then still draws the frames it always drew."""
    from test_hud_timer import LEGACY_NOOP_600_SHA256, _noop_600_sha256
    before = (fresh.episode_time_units, fresh.end_on_level_complete)
    play(fresh, scripted([]), on_reset=clock(2))
    assert (fresh.episode_time_units, fresh.end_on_level_complete) == before

    def boom(_obs, _rng):
        raise RuntimeError("actor failed")
    with pytest.raises(RuntimeError):
        play(fresh, boom)
    assert (fresh.episode_time_units, fresh.end_on_level_complete) == before
    fresh.reset()
    assert _noop_600_sha256(fresh.step)[0] == LEGACY_NOOP_600_SHA256


def test_every_six_m_master_file_is_unchanged():                       # [10]
    found = _files_digest(SIX_M_FILES)
    assert "mario_brain_checkpoint.zip" in found, "the tracked master is missing"
    assert set(found.values()) == {ce.SIX_M_SHA256}, found


# ══════════════════════════════════════════════════════════════════════════
# THE REAL CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def six_m():
    if not os.path.exists(BASELINE_MODEL):
        pytest.skip("backup_6M/ is not in the repository")
    return ce.load_policy(BASELINE_MODEL)


@pytest.mark.slow
def test_evaluation_never_updates_the_weights(env, six_m):              # [1]
    before = {k: v.detach().clone() for k, v in six_m.policy.state_dict().items()}
    file_before = ce.sha256_of(BASELINE_MODEL)
    ce.run_seeds(env, six_m, PROTOCOL, [1, 2], on_reset=clock(20))
    assert not six_m.policy.training
    after = six_m.policy.state_dict()
    assert before.keys() == after.keys()
    for k, v in before.items():
        assert np.array_equal(v.cpu().numpy(), after[k].cpu().numpy()), f"{k} changed"
    assert ce.sha256_of(BASELINE_MODEL) == file_before


@pytest.mark.slow
def test_same_checkpoint_and_seeds_reproduce(env, six_m):               # [3]
    """Per seed, the same record - down to the hash of every action - in any
    order. Order independence is what makes a worker pool equal to a serial
    run. A different seed plays differently."""
    a = ce.run_seeds(env, six_m, PROTOCOL, [11, 12], on_reset=clock(25))
    b = ce.run_seeds(env, six_m, PROTOCOL, [12, 11], on_reset=clock(25))
    assert a == b[::-1]
    assert a[0]['actions_sha1'] != a[1]['actions_sha1']


@pytest.mark.slow
@needs_6m
def test_evaluate_touches_no_campaign_state(env, monkeypatch):          # [2]
    """A whole evaluation, end to end, with coverage made impossible to
    construct: nothing it reads can have come from, or gone to, a coverage
    map - and every file a QA campaign keeps is byte-identical afterwards.
    Seed 900000 is also pinned: the same record as the pilot run, in another
    process, weeks of other episodes earlier."""
    import custom_mario_env
    from exploration import coverage

    def no_coverage(*_a, **_k):
        raise AssertionError("the evaluator built a coverage map")
    monkeypatch.setattr(coverage.SpatialCoverage, '__init__', no_coverage)
    monkeypatch.setattr(custom_mario_env, 'CustomMarioEnv', lambda: env)
    before = _files_digest(_campaign_files() + list(SIX_M_FILES))
    result = ce.evaluate(BASELINE_MODEL, ce.make_protocol(1, 900_000), base=env, log=None)
    assert _files_digest(_campaign_files() + list(SIX_M_FILES)) == before
    (r,) = result['episodes']
    assert (r['end'], r['agent_steps'], r['max_x']) == (EndReason.LEVEL_COMPLETE, 412, 8751)
    assert result['checkpoint']['num_timesteps'] == 6_000_000
    assert result['checkpoint']['is_6m_master']


@pytest.mark.slow
@needs_6m
def test_cli_compares_another_checkpoint_to_a_baseline(env, tmp_path, monkeypatch):  # [7]
    """The real path a QA checkpoint will take, with a stand-in: the 5.6M
    completion checkpoint against a 2-episode baseline under a short clock,
    so the whole thing runs in seconds."""
    import custom_mario_env
    from tools import evaluate_completion as cli
    other = os.path.join(ROOT, "checkpoints", "mario_brain_checkpoint_5600000_steps.zip")
    if not os.path.exists(other):
        pytest.skip("checkpoints/ is not in the repository")
    monkeypatch.setattr(custom_mario_env, 'CustomMarioEnv', lambda: env)
    protocol = dict(ce.make_protocol(2, 900_000), engine_time_units=40)
    baseline = ce.evaluate(BASELINE_MODEL, protocol, base=env, log=None)
    baseline['kind'] = 'completion_retention_baseline'
    baseline['thresholds'] = ce.derive_thresholds(baseline['episodes'], 2.0, 4.0, rounds=500)
    ce.save(baseline, str(tmp_path / "baseline.json"))
    code = cli.main([other, '--baseline', str(tmp_path / "baseline.json"),
                     '--out', str(tmp_path / "result.json")])
    result = ce.load(str(tmp_path / "result.json"))
    assert result['checkpoint']['num_timesteps'] == 5_600_000
    assert result['checkpoint']['filename_timesteps'] == 5_600_000
    assert not result['checkpoint']['is_6m_master']
    assert result['protocol'] == baseline['protocol']
    c = result['comparison']
    assert c['baseline_timesteps'] == 6_000_000 and c['verdict'] in ce.VERDICTS
    assert code == {'HEALTHY': 0, 'WARNING': 1, 'REGRESSED': 2}[c['verdict']]
    with open(other, 'rb') as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == result['checkpoint']['sha256']
