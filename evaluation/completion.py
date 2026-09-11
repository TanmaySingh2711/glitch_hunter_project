"""Completion retention: can this checkpoint still finish Level 1-1?

The frozen 6M completion brain is the BASELINE. Every later checkpoint - the
QA ones in particular - is played under one fixed protocol and compared
against it, so that a checkpoint which gains coverage but loses the ability
to finish the level is caught rather than waved through.

THE PROTOCOL. It is stored in the baseline file and every comparison reuses
it from there; nothing about it is taken from the command line.

  * The bare engine. No GlitchHunterWrapper: no reward, no coverage, no
    novelty, no EXPLORE/COMPLETE lifecycle, no safety reset, no legacy stuck
    rule. Only the game can end an episode - the castle door, a death, or the
    clock - so no QA campaign state can reach this test, and this test has no
    coverage object to write to in the first place.
  * The QA episode lifetime (QA_EPISODE_TIME_UNITS, Phase 3), with the
    Phase 4D TIME box, so the observation is the legacy one throughout.
    A slow but successful run counts as a completion: speed is recorded, not
    judged. Whether the run would ALSO have beaten the original 401-unit clock
    is recorded per episode.
  * Episodes end at the castle door (end_on_level_complete): the victory
    animation after it is a fixed countdown no action can change.
  * The training observation chain, exactly: MaxAndSkip(4) > Grayscale >
    Resize(84x84) > FrameStack(4), with the mode's TimeLimit backstop on top.
  * SAMPLED actions from a per-episode numpy RNG. The engine has no
    randomness at all, so greedy play is one trajectory however many times it
    is repeated - a single yes/no, not a rate. The policy's own action
    distribution - the one PPO acts with in training - is the only source of
    variety that exercises it. The greedy trajectory is recorded once as a
    supplement.
  * Every episode starts from a black frame. The env's reset() returns
    whatever was last on the display - the previous episode's final frame -
    so without this an episode would depend on the one before it. Blanked,
    each episode is a function of (checkpoint, seed) alone: episode order
    and worker count cannot change a result.
  * CPU, one torch thread: bit-identical action probabilities run to run.

Weights are only ever read. Nothing here calls learn() or save(), or opens a
checkpoint for writing.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import platform
import re
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym
import numpy as np
import pygame as pg

from common.fileio import read_json, sha256_of, write_json_atomic
from exploration import config
from exploration.lifecycle import EndReason, classify_end

if TYPE_CHECKING:
    from stable_baselines3 import PPO

    from custom_mario_env import CustomMarioEnv

__all__ = ['BASELINE_PATH', 'RESULTS_DIR', 'EpisodeProbe', 'PolicyActor', 'ProtocolMismatch',
           'classify', 'compare', 'derive_thresholds', 'evaluate', 'load', 'load_policy',
           'make_protocol', 'run_episode', 'run_seeds', 'save', 'sha256_of', 'summarize']

_log = logging.getLogger(__name__)

Protocol = dict[str, Any]          # the frozen play protocol (make_protocol)
Record = dict[str, Any]            # one episode's outcome (run_episode)
Result = dict[str, Any]            # a whole evaluation (evaluate)
Log = Callable[[str], None]
Actor = Callable[[Any, np.random.Generator], int]

PROTOCOL_VERSION = 1
SPAWN_X = 110                                   # level1.setup_mario: viewport.x + 110
# Every completed calibration run (Phase 4B) peaked at 8,745-8,751: the door.
CASTLE_DOOR_X = SPAWN_X + config.LEVEL_COMPLETE_SPAN_PX
SIX_M_SHA256 = "690d57022c1fb444454b0b47f9d1e4ff1bc1444d7110c0bc3320b7fe53a188b3"
N_ACTIONS = 10

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_PATH = os.path.join(ROOT, "evaluation", "completion_baseline_6M.json")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

VERDICTS = ("HEALTHY", "WARNING", "REGRESSED")


def make_protocol(episodes: int, seed: int) -> Protocol:
    return {
        'version': PROTOCOL_VERSION,
        'episodes': int(episodes),
        'seed': int(seed),
        'seeds': 'seed + i for i in range(episodes); numpy default_rng per episode',
        'action_mode': 'sampled',
        'env': 'CustomMarioEnv, bare engine: no reward wrapper, no coverage, no lifecycle',
        'reward_mode': None,
        'engine_time_units': config.QA_EPISODE_TIME_UNITS,
        'hud': 'Phase 4D legacy-equivalent TIME box',
        'end_on_level_complete': True,
        'max_agent_steps': config.QA_EPISODE_MAX_STEPS,
        'substeps_per_agent_step': config.SUBSTEPS_PER_AGENT_STEP,
        'observation': 'MaxAndSkip(4) > Grayscale > Resize(84x84) > FrameStack(4)',
        'reset_frame': 'black',
        'device': 'cpu',
        'torch_threads': 1,
    }


def episode_seeds(protocol: Protocol) -> list[int]:
    return [protocol['seed'] + i for i in range(protocol['episodes'])]


# ══════════════════════════════════════════════════════════════════════════
# ONE EPISODE
# ══════════════════════════════════════════════════════════════════════════
class EpisodeProbe(gym.Wrapper[Any, Any, Any, Any]):
    """Sits directly on the engine, UNDER MaxAndSkip, so it sees every
    substep: MaxAndSkip passes up only the last of four infos, and max-x or
    the flagpole can fall on any of them. Read-only - passes everything
    through untouched."""

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        super().__init__(env)
        self.clear()

    def clear(self) -> None:
        self.substeps = 0
        self.max_x = 0
        self.flagpole_substep: int | None = None
        self.castle_substep: int | None = None
        self.last_info: dict[str, Any] = {}

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self.clear()
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.substeps += 1
        if info.get('x_pos') is not None:
            self.max_x = max(self.max_x, int(info['x_pos']))
        if self.flagpole_substep is None and self._past_the_pole(info):
            self.flagpole_substep = self.substeps
        if self.castle_substep is None and info.get('flag_get'):
            self.castle_substep = self.substeps
        self.last_info = info
        return obs, reward, terminated, truncated, info

    def _past_the_pole(self, info: dict[str, Any]) -> bool:
        base: Any = self.env.unwrapped
        c = base.c_module
        mario = getattr(getattr(base.game, 'state', None), 'mario', None)
        state = getattr(mario, 'state', None)
        return bool(info.get('flag_get')) or state in (
            c.FLAGPOLE, c.WALKING_TO_CASTLE, c.END_OF_LEVEL_FALL)


@contextlib.contextmanager
def protocol_env(base: CustomMarioEnv,
                 protocol: Protocol) -> Iterator[tuple[gym.Env[Any, Any], EpisodeProbe]]:
    """The protocol's env over `base`, yielding (env, probe). The engine
    settings it needs are put back on exit, so a shared env is left exactly
    as it was found."""
    from gymnasium.wrappers import TimeLimit

    from custom_mario_env import wrap_observation
    saved = (base.episode_time_units, base.end_on_level_complete)
    base.episode_time_units = protocol['engine_time_units']
    base.end_on_level_complete = protocol['end_on_level_complete']
    try:
        probe = EpisodeProbe(base)
        env = wrap_observation(probe, skip=protocol['substeps_per_agent_step'])
        env = TimeLimit(env, max_episode_steps=protocol['max_agent_steps'])
        yield env, probe
    finally:
        base.episode_time_units, base.end_on_level_complete = saved


class PolicyActor:
    """Chooses actions from a loaded SB3 policy. Inference only: eval mode,
    no_grad, and nothing that could reach an optimizer."""

    def __init__(self, model: PPO, mode: str = 'sampled') -> None:
        if mode not in ('sampled', 'greedy'):
            raise ValueError(f"action mode {mode!r}")
        self.model, self.mode = model, mode
        model.policy.set_training_mode(False)

    def __call__(self, obs: Any, rng: np.random.Generator) -> int:
        import torch
        with torch.no_grad():
            t, _ = self.model.policy.obs_to_tensor(obs)
            dist: Any = self.model.policy.get_distribution(t).distribution
            p = dist.probs[0].cpu().numpy().astype(np.float64)
        if self.mode == 'greedy':
            return int(np.argmax(p))
        return int(rng.choice(len(p), p=p / p.sum()))


def progress_of(max_x: int) -> float:
    return float(np.clip((max_x - SPAWN_X) / (CASTLE_DOOR_X - SPAWN_X), 0.0, 1.0))


def run_episode(env: gym.Env[Any, Any], probe: EpisodeProbe, actor: Actor, seed: int,
                on_reset: Callable[[Any], None] | None = None) -> Record:
    """One episode; returns its record. `actor(obs, rng) -> action`.
    `on_reset(base_env)` runs after the reset - a test hook for placing Mario
    or setting the clock, never used by the protocol itself."""
    surface = pg.display.get_surface()
    if surface is not None:
        surface.fill((0, 0, 0))
    obs, _ = env.reset()
    base = cast('CustomMarioEnv', env.unwrapped)
    if on_reset is not None:
        on_reset(base)
    rng = np.random.default_rng(seed)
    steps, trace = 0, hashlib.sha1(usedforsecurity=False)
    terminated = truncated = False
    while not (terminated or truncated):
        action = actor(obs, rng)
        trace.update(bytes((action,)))
        obs, _r, terminated, truncated, _info = env.step(action)
        steps += 1
    info = probe.last_info
    end = (classify_end(info, False) if terminated else EndReason.TIME_LIMIT)
    time_left = info.get('time_left')
    units = base.episode_time_units or config.ENGINE_TIME_UNITS_DEFAULT
    used = None if time_left is None else units - int(time_left)
    completed = end == EndReason.LEVEL_COMPLETE
    return {
        'seed': int(seed),
        'end': end,
        'death_cause': info.get('death_cause') if end == EndReason.DEATH else None,
        'agent_steps': steps,
        'substeps': probe.substeps,
        'max_x': probe.max_x,
        'progress': round(progress_of(probe.max_x), 5),
        'flagpole': probe.flagpole_substep is not None,
        'castle_door': probe.castle_substep is not None,
        'completion_agent_step': steps if completed else None,
        'engine_units_used': used,
        # The original game's clock would still have been running at the door.
        'within_legacy_clock': (completed and used is not None
                                and used < config.ENGINE_TIME_UNITS_DEFAULT),
        'actions_sha1': trace.hexdigest()[:16],
    }


# ══════════════════════════════════════════════════════════════════════════
# A WHOLE EVALUATION
# ══════════════════════════════════════════════════════════════════════════
def load_policy(path: str) -> PPO:
    """Loads a checkpoint for inference on the CPU, one thread."""
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    return PPO.load(path, device='cpu')


def checkpoint_meta(path: str, model: PPO) -> dict[str, Any]:
    digest = sha256_of(path)
    m = re.search(r'_(\d+)_steps', os.path.basename(path))
    return {
        'path': os.path.relpath(os.path.abspath(path), ROOT).replace('\\', '/'),
        'sha256': digest,
        'num_timesteps': int(model.num_timesteps),
        'filename_timesteps': int(m.group(1)) if m else None,
        'is_6m_master': digest == SIX_M_SHA256,
    }


def run_seeds(base: CustomMarioEnv, model: PPO, protocol: Protocol, seeds: Sequence[int],
              mode: str = 'sampled', on_reset: Callable[[Any], None] | None = None,
              log: Log | None = None) -> list[Record]:
    actor = PolicyActor(model, mode)
    records = []
    with protocol_env(base, protocol) as (env, probe):
        for i, seed in enumerate(seeds):
            records.append(run_episode(env, probe, actor, seed, on_reset))
            if log:
                r = records[-1]
                log(f"  [{i + 1}/{len(seeds)}] seed {seed}: {r['end']:<14} "
                    f"steps {r['agent_steps']:>5}  max-x {r['max_x']:>5}")
    return records


_WORKER: dict[str, Any] = {}


def _worker_init(model_path: str, protocol: Protocol) -> None:
    from custom_mario_env import CustomMarioEnv
    _WORKER['base'] = CustomMarioEnv()
    _WORKER['model'] = load_policy(model_path)
    _WORKER['protocol'] = protocol


def _worker_run(seed: int) -> Record:
    return run_seeds(_WORKER['base'], _WORKER['model'], _WORKER['protocol'], [seed])[0]


def evaluate(model_path: str, protocol: Protocol, workers: int = 1,
             base: CustomMarioEnv | None = None, log: Log | None = _log.info) -> Result:
    """Plays the protocol with the checkpoint at `model_path`; returns the
    full result (metadata, per-episode records, summary)."""
    import stable_baselines3
    import torch
    t0 = time.perf_counter()
    model = load_policy(model_path)
    seeds = episode_seeds(protocol)
    if workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(workers, _worker_init, (model_path, protocol)) as pool:
            records = []
            for i, rec in enumerate(pool.imap(_worker_run, seeds, chunksize=1)):
                records.append(rec)
                if log is not None and (i + 1) % 25 == 0:
                    log(f"  {i + 1}/{len(seeds)} episodes")
    else:
        if base is None:
            from custom_mario_env import CustomMarioEnv
            base = CustomMarioEnv()
        records = run_seeds(base, model, protocol, seeds, log=log)
    if base is None:
        from custom_mario_env import CustomMarioEnv
        base = CustomMarioEnv()
    greedy = run_seeds(base, model, protocol, [protocol['seed']], mode='greedy')[0]
    records.sort(key=lambda r: r['seed'])
    return {
        'kind': 'completion_retention_result',
        'protocol': protocol,
        'checkpoint': checkpoint_meta(model_path, model),
        'summary': summarize(records),
        'greedy': greedy,
        'episodes': records,
        'software': {'python': platform.python_version(), 'torch': torch.__version__,
                     'stable_baselines3': stable_baselines3.__version__,
                     'numpy': np.__version__, 'gymnasium': gym.__version__,
                     'pygame': pg.version.ver},
        'workers': workers,
        'elapsed_s': round(time.perf_counter() - t0, 1),
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY, COMPARISON, VERDICT
# ══════════════════════════════════════════════════════════════════════════
def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson interval for k successes in n (a list: it round-trips JSON)."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(centre - half, 4), round(centre + half, 4)]


def _stats(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'mean': round(float(a.mean()), 2), 'median': float(np.median(a)),
            'p10': float(np.percentile(a, 10)), 'p90': float(np.percentile(a, 90)),
            'min': float(a.min()), 'max': float(a.max())}


def summarize(records: Sequence[Record]) -> dict[str, Any]:
    n = len(records)
    done = [r for r in records if r['end'] == EndReason.LEVEL_COMPLETE]
    failed = [r for r in records if r['end'] != EndReason.LEVEL_COMPLETE]
    return {
        'episodes': n,
        'completed': len(done),
        'completion_rate': round(len(done) / n, 4) if n else 0.0,
        'completion_rate_ci95': wilson(len(done), n),
        'castle_door': sum(r['castle_door'] for r in records),
        'flagpole': sum(r['flagpole'] for r in records),
        'ends': dict(Counter(r['end'] for r in records)),
        'death_causes': dict(Counter(r['death_cause'] for r in records if r['death_cause'])),
        'max_x': _stats([r['max_x'] for r in records]),
        'progress': _stats([r['progress'] for r in records]),
        'progress_of_failures': _stats([r['progress'] for r in failed]),
        'completion_agent_steps': _stats([r['completion_agent_step'] for r in done]),
        'completed_within_legacy_clock': sum(r['within_legacy_clock'] for r in done),
        'episode_agent_steps': _stats([r['agent_steps'] for r in records]),
    }


class ProtocolMismatch(ValueError):
    """Two results that were not played under the same protocol."""


def _protocol_key(protocol: Protocol) -> str:
    return json.dumps(protocol, sort_keys=True)


def classify(candidate_summary: dict[str, Any],
             thresholds: dict[str, Any]) -> tuple[str, list[str]]:
    """HEALTHY / WARNING / REGRESSED from a summary and the baseline's
    measured thresholds. Completion rate decides; mean progress - the
    fallback for runs that do not finish - can only make it worse. Coverage
    plays no part: it is not measured here at all."""
    rate = candidate_summary['completion_rate']
    prog = candidate_summary['progress']['mean']
    t_rate, t_prog = thresholds['completion_rate'], thresholds['mean_progress']
    reasons = []
    verdict = 'HEALTHY'
    if rate < t_rate['regressed_below']:
        verdict = 'REGRESSED'
        reasons.append(f"completion rate {rate:.3f} < {t_rate['regressed_below']:.3f}")
    elif rate < t_rate['warning_below']:
        verdict = 'WARNING'
        reasons.append(f"completion rate {rate:.3f} < {t_rate['warning_below']:.3f}")
    if prog < t_prog['regressed_below']:
        verdict = 'REGRESSED'
        reasons.append(f"mean progress {prog:.3f} < {t_prog['regressed_below']:.3f}")
    elif prog < t_prog['warning_below'] and verdict == 'HEALTHY':
        verdict = 'WARNING'
        reasons.append(f"mean progress {prog:.3f} < {t_prog['warning_below']:.3f}")
    return verdict, reasons


def compare(candidate: Result, baseline: Result) -> dict[str, Any]:
    """The candidate result against the baseline file. Refuses outright if
    the two were not played under the identical protocol."""
    if _protocol_key(candidate['protocol']) != _protocol_key(baseline['protocol']):
        raise ProtocolMismatch("candidate and baseline were played under different "
                               "protocols; re-run the candidate with the baseline's")
    cs, bs = candidate['summary'], baseline['summary']
    verdict, reasons = classify(cs, baseline['thresholds'])
    return {
        'verdict': verdict,
        'reasons': reasons,
        'checkpoint_timesteps': candidate['checkpoint']['num_timesteps'],
        'baseline_timesteps': baseline['checkpoint']['num_timesteps'],
        'completion_rate': cs['completion_rate'],
        'baseline_completion_rate': bs['completion_rate'],
        'completion_delta': round(cs['completion_rate'] - bs['completion_rate'], 4),
        'mean_progress': cs['progress']['mean'],
        'baseline_mean_progress': bs['progress']['mean'],
        'progress_delta': round(cs['progress']['mean'] - bs['progress']['mean'], 4),
        'median_max_x': cs['max_x']['median'],
        'baseline_median_max_x': bs['max_x']['median'],
        'greedy_completes': candidate['greedy']['end'] == EndReason.LEVEL_COMPLETE,
    }


def derive_thresholds(records: Sequence[Record], z_warning: float, z_regressed: float,
                      rounds: int = 20_000, seed: int = 0) -> dict[str, Any]:
    """Thresholds from the baseline's own spread, not picked by hand.

    For each metric, the no-change noise of the protocol is the spread of
    (run A - run B) between two independent runs of the SAME policy at the
    baseline's episode count - bootstrapped from the baseline episodes. A
    candidate's shortfall is then read in units of that noise: within
    z_warning of it the drop is indistinguishable from re-running the
    baseline; beyond z_regressed it is not noise at any sensible confidence.
    """
    rng = np.random.default_rng(seed)
    n = len(records)
    done = np.array([r['end'] == EndReason.LEVEL_COMPLETE for r in records], float)
    prog = np.array([r['progress'] for r in records], float)
    out: dict[str, Any] = {}
    for name, v in (('completion_rate', done), ('mean_progress', prog)):
        a = v[rng.integers(0, n, (rounds, n))].mean(1)
        b = v[rng.integers(0, n, (rounds, n))].mean(1)
        sd = float(np.std(a - b))
        centre = float(v.mean())
        out[name] = {
            'baseline': round(centre, 4),
            'noise_sd_of_difference': round(sd, 4),
            'warning_below': round(centre - z_warning * sd, 4),
            'regressed_below': round(centre - z_regressed * sd, 4),
        }
    out['z_warning'], out['z_regressed'] = z_warning, z_regressed
    out['bootstrap_rounds'] = rounds
    return out


def save(result: Result, path: str) -> None:
    """Writes a result (or the baseline) atomically."""
    write_json_atomic(result, path)


def load(path: str) -> Result:
    """Reads a result, the baseline or a verification record."""
    loaded: Result = read_json(path)
    return loaded
