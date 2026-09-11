"""Measure the phase-gated QA reward on real Level 1-1 trajectories.

    python tools/calibrate_phase_reward.py --arms ppo_boot,c_ppo --out runs/

Phase 4B. Runs the 6M policy (inference only - learn() is never called and
nothing is written back to any checkpoint) and a handful of scripted
controllers through the REAL environment under the qa_exploration reward,
and records every reward channel separately, per episode, per phase and per
agent step (agent_logic.QA_CHANNELS). The per-step records are what the
leakage and clipping analysis in --report is computed from, so a report can
be regenerated without re-running anything.

Every arm starts from the bootstrap coverage map unless it says otherwise,
because that is the condition QA training genuinely starts in. "restore"
arms put the map back before every episode (a stationary sample of the
start of training); "persist" arms let it fill, which is the campaign.

The adaptive-target history starts EMPTY in every arm, as it does in a real
SubprocVecEnv worker (the history list is per process; only the bitmap is
shared), and accumulates across the arm's episodes.

Phase forcing. Some scenarios need the episode in a phase the controller
would not reach on its own - COMPLETE from the spawn, or EXPLORE pinned for
a whole run. Those arms disable the lifecycle's natural transition check
and move the phase by hand at a chosen agent step, with the credit computed
by the lifecycle's own _credit_for() on the real coverage state (or a fixed
credit where the arm says so). Everything else about the reward is the
production code path.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.cli import prepare_tool

ROOT = prepare_tool()

from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np

from agent_logic import GlitchHunterWrapper
from exploration import config
from exploration.coverage import SpatialCoverage, load_testable
from exploration.lifecycle import EpisodeLifecycle, EpisodePhase, Transition

N_ACTIONS = 10


# ── controllers ─────────────────────────────────────────────────────────────
Info = dict[str, Any]
ArmSpec = dict[str, Any]
Episode = dict[str, Any]


class Controller:
    uses_model = False

    def reset(self) -> None:
        pass

    def act(self, obs: Any, t: int, info: Info) -> int:
        raise NotImplementedError


class Policy(Controller):
    """The 6M policy, sampled stochastically as PPO does during rollouts."""
    uses_model = True

    def __init__(self, model: Any, eps: float = 0.0,
                 rng: np.random.Generator | None = None) -> None:
        self.model, self.eps = model, eps
        self.rng = rng or np.random.default_rng(0)

    def act(self, obs: Any, t: int, info: Info) -> int:
        if self.eps and self.rng.random() < self.eps:
            return int(self.rng.integers(N_ACTIONS))
        action, _ = self.model.predict(obs, deterministic=False)
        return int(action)


class Hold(Controller):
    def __init__(self, action: int) -> None:
        self.action = action

    def act(self, obs: Any, t: int, info: Info) -> int:
        return self.action


class Oscillate(Controller):
    """Runs right, running-jumps, runs back left, jumps - forever, locally.

    The densest direction-agnostic locomotion farm available from the spawn:
    every leg ends in a running jump, which is what QA_CLEAN_JUMP_REWARD
    pays for, and the drift is near zero so it never reaches new ground.
    """
    PATTERN = ((3, 10), (4, 6), (8, 10), (9, 6))   # (action, agent steps)
    # Found by searching 162 patterns on the real engine: the one that kept
    # paying clean-jump reward for a whole 9,782-step episode (35.7 before
    # QA_LOCOMOTION_EPISODE_CAP existed).
    FARM = ((3, 6), (4, 4), (8, 8), (5, 2))

    def __init__(self, pattern: Sequence[tuple[int, int]] | None = None) -> None:
        self.pattern = pattern or self.PATTERN
        self.reset()

    def reset(self) -> None:
        self.i, self.left = 0, self.pattern[0][1]

    def act(self, obs: Any, t: int, info: Info) -> int:
        if self.left <= 0:
            self.i = (self.i + 1) % len(self.pattern)
            self.left = self.pattern[self.i][1]
        self.left -= 1
        return self.pattern[self.i][0]


class Switch(Controller):
    """`first` until `until(t, info, lifecycle)` is true, then `second`."""

    def __init__(self, first: Controller, second: Controller,
                 until: Callable[[int, Info, EpisodeLifecycle | None], bool]) -> None:
        self.first, self.second, self.until = first, second, until
        self.uses_model = first.uses_model or second.uses_model
        self.lifecycle: EpisodeLifecycle | None = None
        self.switched = False

    def reset(self) -> None:
        self.switched = False
        self.first.reset()
        self.second.reset()

    def act(self, obs: Any, t: int, info: Info) -> int:
        if not self.switched and self.until(t, info, self.lifecycle):
            self.switched = True
        return (self.second if self.switched else self.first).act(obs, t, info)


# ── arms ────────────────────────────────────────────────────────────────────
def _in_complete(t: int, info: Info, lc: EpisodeLifecycle | None) -> bool:
    return lc is not None and lc.is_complete


def arm_specs(model: Any) -> dict[str, ArmSpec]:
    """name -> dict(controller, episodes, map, forcing).

    map:      'bootstrap' | 'empty' | 'full'   (starting bitmap)
    persist:  keep the bitmap between episodes (the campaign) instead of
              restoring it
    force:    None (natural lifecycle) | dict(at=agent step, reason=...,
              credit=None for the lifecycle's own formula, or a number;
              pin=True keeps EXPLORE forever)
    """
    rng = np.random.default_rng(1234)

    def ppo(eps: float = 0.0) -> Policy:
        return Policy(model, eps=eps, rng=rng)

    t1, t2, t4 = (Transition.TARGET_MET, Transition.YIELD_EXHAUSTED,
                  Transition.EXPLORE_BACKSTOP)
    return {
        # natural lifecycle ------------------------------------------------
        'ppo_boot':  {"ctl": ppo(), "n": 30, "map": 'bootstrap'},
        'eps_boot':  {"ctl": ppo(0.3), "n": 20, "map": 'bootstrap'},
        'ppo_empty': {"ctl": ppo(), "n": 8, "map": 'empty'},
        'campaign':  {"ctl": ppo(), "n": 40, "map": 'bootstrap', "persist": True},
        'ppo_full':  {"ctl": ppo(), "n": 6, "map": 'full'},
        'idle':      {"ctl": Hold(0), "n": 2, "map": 'bootstrap'},
        'oscillate': {"ctl": Oscillate(), "n": 2, "map": 'bootstrap'},
        # the measured locomotion farm, EXPLORE pinned and in COMPLETE
        'farm':      {"ctl": Oscillate(Oscillate.FARM), "n": 1, "map": 'bootstrap',
                          "force": {"pin": True}},
        'c_farm':    {"ctl": Oscillate(Oscillate.FARM), "n": 1, "map": 'bootstrap',
                          "force": {"at": 0, "reason": Transition.TARGET_MET,
                                     "credit": 1.0}},
        # T2 on purpose: stand still until the lifecycle gives up, then run
        'idle_then_ppo': {"ctl": Switch(Hold(0), ppo(), _in_complete), "n": 8,
                              "map": 'bootstrap'},
        # forced phase -----------------------------------------------------
        # EXPLORE for the whole episode: the straightforward speedrun check
        'e_ppo':     {"ctl": ppo(), "n": 20, "map": 'bootstrap',
                          "force": {"pin": True}},
        'e_suicide': {"ctl": Hold(1), "n": 3, "map": 'bootstrap',
                          "force": {"pin": True}},
        # COMPLETE from the spawn, full credit (a legitimate T1)
        'c_ppo':     {"ctl": ppo(), "n": 20, "map": 'bootstrap',
                          "force": {"at": 0, "reason": t1, "credit": 1.0}},
        'c_idle':    {"ctl": Hold(0), "n": 2, "map": 'bootstrap',
                          "force": {"at": 0, "reason": t1, "credit": 1.0}},
        'c_oscillate': {"ctl": Oscillate(), "n": 2, "map": 'bootstrap',
                            "force": {"at": 0, "reason": t1, "credit": 1.0}},
        'c_suicide': {"ctl": Hold(1), "n": 3, "map": 'bootstrap',
                          "force": {"at": 0, "reason": t1, "credit": 1.0}},
        # COMPLETE mid-level, then walk away from the castle
        'c_backward': {"ctl": Switch(ppo(), Hold(8), _in_complete), "n": 4,
                           "map": 'bootstrap',
                           "force": {"at": 150, "reason": t1, "credit": 1.0}},
        # T4 by the backstop clock, credit from the real formula
        't4_osc_then_ppo': {"ctl": Switch(Oscillate(), ppo(), _in_complete),
                                "n": 2, "map": 'bootstrap',
                                "force": {"at": config.MAX_EXPLORE_STEPS,
                                           "reason": t4, "credit": None}},
        # T2 by hand at the earliest point idling could reach it, real credit
        't2_forced_ppo': {"ctl": ppo(), "n": 8, "map": 'bootstrap',
                              "force": {"at": config.YIELD_WINDOWS
                                         * config.LIFECYCLE_WINDOW,
                                         "reason": t2, "credit": None}},
    }


# ── env ─────────────────────────────────────────────────────────────────────
def build_env(coverage: SpatialCoverage) -> tuple[gym.Env[Any, Any], GlitchHunterWrapper]:
    from gymnasium.wrappers import TimeLimit

    from custom_mario_env import CustomMarioEnv, wrap_observation

    inner = GlitchHunterWrapper(CustomMarioEnv(), reward_mode="qa_exploration",
                                coverage=coverage, attach_coverage=True)
    env: gym.Env[Any, Any] = TimeLimit(wrap_observation(inner), max_episode_steps=config.QA_EPISODE_MAX_STEPS)
    return env, inner


def make_coverage(kind: str, bootstrap: str) -> SpatialCoverage:
    cov = SpatialCoverage(testable_mask=load_testable())
    if kind in ('bootstrap', 'full'):
        cov.load(bootstrap)
    if kind == 'full' and cov.testable is not None:
        cov.visited[cov.testable] = 1
    # A fresh worker's history: empty (see module docstring).
    cov.episode_new_history = []
    cov.invalidate_remaining()
    return cov


def restore(cov: SpatialCoverage, bits: np.ndarray) -> None:
    cov.visited[:] = bits
    cov.invalidate_remaining()
    cov.invalidate_frontier()     # frontier snapshot is stale


# ── one arm ─────────────────────────────────────────────────────────────────
def _flat(ep_channels: dict[str, dict[str, float]]) -> np.ndarray:
    from agent_logic import QA_CHANNELS
    return np.array([[ep_channels[p.value][k] for k in QA_CHANNELS]
                     for p in EpisodePhase])            # (2, n_channels)


def run_arm(name: str, spec: ArmSpec, bootstrap: str, seed: int) -> list[Episode]:
    from agent_logic import QA_CHANNELS
    cov = make_coverage(spec['map'], bootstrap)
    bits = cov.visited.copy()
    env, inner = build_env(cov)
    lc = inner.lifecycle
    assert lc is not None                      # a QA wrapper always has one
    force = spec.get('force')
    if force:
        lc.auto_transition = False
    ctl = spec['ctl']
    if isinstance(ctl, Switch):
        ctl.lifecycle = lc

    # The lifecycle's own inputs, substep by substep, and every window verdict
    # it closed. The policy never sees the phase and the env never reads it,
    # so a trajectory is the same whatever the transition rule is - which is
    # what lets --replay re-run candidate rules on this exact stream.
    trace: dict[str, Any] = {}
    real_observe = lc.observe

    def observe(info: Info, n_new: int) -> None:
        before_window, before_phase = lc.last_window, lc.phase
        real_observe(info, n_new)
        trace['new'].append(int(n_new))
        rect = info.get('mario_rect')
        trace['pos'].append((rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)
                            if rect else trace['pos'][-1] if trace['pos'] else (0, 0))
        w = lc.last_window
        if w is not None and w is not before_window:
            trace['windows'].append((lc.substeps, w['new_px'], w['in_transit'],
                                     w['exhausted'], w['is_stuck'],
                                     -1 if w['cells_ahead'] is None else w['cells_ahead']))
        if lc.phase is not before_phase:
            trace['switch_substep'] = lc.substeps
    lc.observe = observe                       # type: ignore[method-assign]  # tracing hook

    episodes = []
    for ep in range(spec['n']):
        if not spec.get('persist'):
            restore(cov, bits)
        clip0 = inner.qa_clip_events
        trace.update(new=[], pos=[], windows=[], switch_substep=None)
        obs, info = env.reset(seed=seed + ep)
        ctl.reset()
        informed = lc.target_informed
        covered0 = cov.covered_testable()
        target = lc.target
        prev = _flat(inner.ep_channels)
        steps_ch: list[np.ndarray] = []
        steps_phase: list[int] = []
        steps_x: list[int] = []
        steps_r: list[float] = []
        t, t0 = 0, time.perf_counter()
        while True:
            if force and not force.get('pin') and t == force['at']:
                lc.force_complete(force['reason'], force['credit'])
            action = ctl.act(obs, t, info)
            obs, r, term, trunc, info = env.step(action)
            cur = _flat(inner.ep_channels)
            steps_ch.append((cur - prev).sum(axis=0))
            prev = cur
            steps_phase.append(1 if lc.is_complete else 0)
            steps_x.append(info.get('x_pos', 0))
            steps_r.append(float(r))
            t += 1
            if term or trunc:
                break
        chans = info.get('qa_channels') or {
            p.value: dict(zip(QA_CHANNELS, row, strict=True))
            for p, row in zip(EpisodePhase, cur, strict=True)}
        end = info.get('episode_end_reason')
        if trunc and not term:
            end = 'time_limit'
        rec = {
            "arm": name, "episode": ep, "steps": t, "ret": float(np.sum(steps_r)),
            "end": end, "transition": lc.transition_reason,
            "transition_step": lc.transition_step,
            "credit": lc.completion_credit if lc.is_complete else None,
            "episode_new": int(cov.episode_new), "target": int(target or 0),
            "covered_before": covered0, "max_x": int(inner.ep_max_x),
            "flag": bool(info.get('flag_get')), "clip_events": inner.qa_clip_events - clip0,
            "channels": chans, "wall": time.perf_counter() - t0,
            "step_channels": np.asarray(steps_ch, dtype=np.float32),
            "step_phase": np.asarray(steps_phase, dtype=np.int8),
            "step_x": np.asarray(steps_x, dtype=np.int32),
            "step_reward": np.asarray(steps_r, dtype=np.float32),
            "target_informed": bool(informed),
            "sub_new": np.asarray(trace['new'], dtype=np.int32),
            "sub_pos": np.asarray(trace['pos'], dtype=np.int16).reshape(-1, 2),
            "windows": np.asarray(trace['windows'], dtype=np.int64).reshape(-1, 6),
            "switch_substep": trace['switch_substep'],
        }
        episodes.append(rec)
        e, c = chans['explore'], chans['complete']
        credit_txt = '-' if rec['credit'] is None else f"{rec['credit']:.2f}"
        print(f"[{name}] ep {ep + 1:>2}/{spec['n']}  steps {t:>5}  "
              f"ret {rec['ret']:>8.2f}  end {end:<14} "
              f"trans {lc.transition_reason!s:<20} "
              f"@{lc.transition_step!s:>5} credit {credit_txt:>4}  "
              f"new {rec['episode_new']:>6}/{target:<6} "
              f"E[nov {e['novelty']:6.2f}] C[prog {c['progress']:6.2f} "
              f"flag {c['flag']:5.2f}] maxx {rec['max_x']:>5} "
              f"({rec['wall']:.0f}s)", flush=True)
    env.close()
    return episodes


# ── report ──────────────────────────────────────────────────────────────────
# The arms whose lifecycle ran naturally from the bootstrap map: the closest
# available sample of what QA training will actually feed PPO at its start.
NATURAL_MIX = ('ppo_boot', 'eps_boot', 'campaign')

# What each phase is FOR. Exploration-driven: paid for finding space.
# Completion-driven: paid for finishing. The rest (locomotion, interaction,
# and the failure costs) is shared by both phases and belongs to neither.
EXPLORE_DRIVEN = ('novelty', 'frontier')
COMPLETE_DRIVEN = ('progress', 'flag', 'time')


def _stats(values: Sequence[float] | np.ndarray) -> str:
    v = np.asarray(values, dtype=float)
    if not v.size:
        return "n=0"
    return (f"mean {v.mean():8.2f}  median {np.median(v):8.2f}  "
            f"p10 {np.percentile(v, 10):8.2f}  p90 {np.percentile(v, 90):8.2f}  "
            f"min {v.min():8.2f}  max {v.max():8.2f}  (n={v.size})")


def _discounted(x: np.ndarray, gamma: float) -> np.ndarray:
    g, out = 0.0, np.empty_like(x, dtype=float)
    for i in range(len(x) - 1, -1, -1):
        g = x[i] + gamma * g
        out[i] = g
    return out


def report(run_dir: str, gamma: float = config.GAMMA, bucket: int = 512) -> None:
    import collections
    import glob

    from agent_logic import QA_CHANNELS
    idx = {k: i for i, k in enumerate(QA_CHANNELS)}
    arms = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "*.pkl"))):
        with open(p, "rb") as fh:
            # Only ever this tool's own --out files (see main); never
            # point --report at pickles from anywhere else.
            arms[os.path.splitext(os.path.basename(p))[0]] = pickle.load(fh)  # noqa: S301
    short = {'death': 'death', 'novelty': 'nov', 'frontier': 'front',
             'drought': 'drought', 'safety_reset': 'safety',
             'locomotion': 'loco', 'time': 'time', 'progress': 'prog',
             'interaction': 'inter', 'flag': 'flag', 'shortfall': 'short',
             'clip': 'clip'}

    print("=" * 110)
    print("1. PER-ARM CHANNEL ACCOUNTING  (mean per episode; E = EXPLORE, C = COMPLETE)")
    print("=" * 110)
    head = "".join(f"{short[k]:>8}" for k in QA_CHANNELS)
    for arm, eps in arms.items():
        rets = [e['ret'] for e in eps]
        ends = collections.Counter(e['end'] for e in eps)
        trans = collections.Counter(e['transition'] or 'none' for e in eps)
        arm_credits = [e['credit'] for e in eps if e['credit'] is not None]
        print(f"\n[{arm}] {len(eps)} ep, steps mean {np.mean([e['steps'] for e in eps]):.0f}, "
              f"flag {sum(e['flag'] for e in eps)}/{len(eps)}, "
              f"new px mean {np.mean([e['episode_new'] for e in eps]):,.0f}")
        print(f"   return {_stats(rets)}")
        print(f"   ends {dict(ends)}   transitions {dict(trans)}")
        if arm_credits:
            print(f"   credit {_stats(arm_credits)}")
        print(f"      {head}{'total':>9}")
        for ph in ('explore', 'complete'):
            m = np.mean([[e['channels'][ph][k] for k in QA_CHANNELS] for e in eps],
                        axis=0)
            print(f"   {ph[0].upper()}  " + "".join(f"{v:8.2f}" for v in m)
                  + f"{m.sum():9.2f}")

    mix = [e for a in NATURAL_MIX if a in arms for e in arms[a]]
    if mix:
        print("\n" + "=" * 110)
        print(f"2. NATURAL MIX ({'+'.join(a for a in NATURAL_MIX if a in arms)}, "
              f"{len(mix)} episodes): phase totals and shares")
        print("=" * 110)
        e_tot = [sum(e['channels']['explore'].values()) for e in mix]
        c_tot = [sum(e['channels']['complete'].values()) for e in mix]
        print(f"   EXPLORE-phase total per episode   {_stats(e_tot)}")
        print(f"   COMPLETE-phase total per episode  {_stats(c_tot)}")
        ed = sum(e['channels']['explore'][k] for e in mix for k in EXPLORE_DRIVEN)
        cd = sum(e['channels']['complete'][k] for e in mix for k in COMPLETE_DRIVEN)
        ef = sum(e['channels']['explore']['flag'] for e in mix)
        cn = sum(e['channels']['complete']['novelty'] for e in mix)
        loco = sum(e['channels'][p]['locomotion'] for e in mix
                   for p in ('explore', 'complete'))
        print(f"   exploration-driven (E novelty+frontier)     {ed:10.2f}")
        print(f"   completion-driven  (C progress+flag+time)   {cd:10.2f}")
        print(f"   hidden completion in EXPLORE (E flag)       {ef:10.2f}")
        print(f"   COMPLETE novelty (tie-breaker)              {cn:10.2f}")
        print(f"   locomotion, both phases                     {loco:10.2f}")
        print(f"   completion-driven / exploration-driven      "
              f"{cd / max(ed, 1e-9):10.3f}")
        print(f"   completion share of (E-driven + C-driven)   "
              f"{cd / max(ed + cd, 1e-9):10.3f}")

        # Return-scale pressure: the spread of each group's DISCOUNTED return
        # over the states visited. The critic learns the mean; the spread is
        # what is left for the advantage, i.e. for the policy gradient.
        parts_e: list[np.ndarray] = []
        parts_c: list[np.ndarray] = []
        parts_all: list[np.ndarray] = []
        for e in mix:
            sc, ph = e['step_channels'], e['step_phase']
            xe = sc[:, [idx[k] for k in EXPLORE_DRIVEN]].sum(1) * (ph == 0)
            xc = sc[:, [idx[k] for k in COMPLETE_DRIVEN]].sum(1) * (ph == 1)
            parts_e.append(_discounted(xe, gamma))
            parts_c.append(_discounted(xc, gamma))
            parts_all.append(_discounted(sc.sum(1), gamma))
        g_e, g_c, g_all = (np.concatenate(g) for g in (parts_e, parts_c, parts_all))
        print(f"\n   discounted-return spread (gamma {gamma}, per agent step), "
              f"{g_e.size:,} states:")
        print(f"     exploration-driven  std {g_e.std():8.3f}")
        print(f"     completion-driven   std {g_c.std():8.3f}")
        print(f"     all channels        std {g_all.std():8.3f}")
        print(f"     completion / exploration spread  {g_c.std() / max(g_e.std(), 1e-9):.3f}")

        # The same two ratios on steady-state episodes only: once a worker
        # has TARGET_MIN_HISTORY episodes behind it, T1 is armed, which is
        # every episode of a long run but its first few.
        steady = [e for e in mix if e['episode'] >= config.TARGET_MIN_HISTORY]
        ed_s = sum(e['channels']['explore'][k] for e in steady for k in EXPLORE_DRIVEN)
        cd_s = sum(e['channels']['complete'][k] for e in steady for k in COMPLETE_DRIVEN)
        parts_ge: list[np.ndarray] = []
        parts_gc: list[np.ndarray] = []
        for e in steady:
            sc, ph = e['step_channels'], e['step_phase']
            parts_ge.append(_discounted(sc[:, [idx[k] for k in EXPLORE_DRIVEN]].sum(1)
                                    * (ph == 0), gamma))
            parts_gc.append(_discounted(sc[:, [idx[k] for k in COMPLETE_DRIVEN]].sum(1)
                                        * (ph == 1), gamma))
        ge_s, gc_s = np.concatenate(parts_ge), np.concatenate(parts_gc)
        print(f"\n   steady state ({len(steady)} episodes, target informed): "
              f"completion / exploration summed {cd_s / max(ed_s, 1e-9):.3f}, "
              f"spread {gc_s.std() / max(ge_s.std(), 1e-9):.3f}")
        if 'campaign' in arms:
            for lo, hi in ((config.TARGET_MIN_HISTORY, 20), (20, 40)):
                part = [e for e in arms['campaign'] if lo <= e['episode'] < hi]
                ed_p = sum(e['channels']['explore'][k] for e in part for k in EXPLORE_DRIVEN)
                cd_p = sum(e['channels']['complete'][k] for e in part for k in COMPLETE_DRIVEN)
                print(f"   campaign episodes {lo}-{hi - 1} (map filling): "
                      f"exploration {ed_p:7.2f}  completion {cd_p:7.2f}  "
                      f"ratio {cd_p / max(ed_p, 1e-9):.3f}")

        # Leakage by place: the policy cannot see the phase, but it can see
        # where it is. Where COMPLETE pays for running right at a place EXPLORE
        # also visits, that pay is what bleeds into EXPLORE behaviour there.
        print(f"\n   by position ({bucket}px buckets): E-driven vs C-driven reward, "
              f"and agent steps per phase")
        nb = config.LEVEL_W // bucket + 1
        be, bc = np.zeros(nb), np.zeros(nb)
        se, sc_ = np.zeros(nb), np.zeros(nb)
        for e in mix:
            b = np.clip(e['step_x'] // bucket, 0, nb - 1)
            sc, ph = e['step_channels'], e['step_phase']
            xe = sc[:, [idx[k] for k in EXPLORE_DRIVEN]].sum(1) * (ph == 0)
            xc = sc[:, [idx[k] for k in COMPLETE_DRIVEN]].sum(1) * (ph == 1)
            np.add.at(be, b, xe)
            np.add.at(bc, b, xc)
            np.add.at(se, b, ph == 0)
            np.add.at(sc_, b, ph == 1)
        for i in range(nb):
            if se[i] + sc_[i] == 0:
                continue
            print(f"     x {i * bucket:>5}-{(i + 1) * bucket - 1:<5}  E {be[i]:8.2f} "
                  f"({int(se[i]):>6} st)   C {bc[i]:8.2f} ({int(sc_[i]):>6} st)   "
                  f"C/E {bc[i] / be[i] if be[i] > 0 else float('inf'):7.2f}")

    print("\n" + "=" * 110)
    print("3. CLIPPING  (substeps clamped at +/-QA_REWARD_CLIP)")
    print("=" * 110)
    for arm, eps in arms.items():
        clips = sum(e['clip_events'] for e in eps)
        subs = sum(e['steps'] for e in eps) * config.SUBSTEPS_PER_AGENT_STEP
        mx = max(float(np.abs(e['step_reward']).max()) for e in eps)
        print(f"   {arm:<16} {clips:>4} clipped of ~{subs:>9,} substeps   "
              f"max |agent-step reward| {mx:6.2f}")


    print("\n" + "=" * 110)
    print("4. TRANSITIONS AND COMPLETION CREDIT, every arm")
    print("=" * 110)
    by: dict[str, list[tuple[str, Episode]]] = {}
    for arm, eps in arms.items():
        for e in eps:
            if e['transition']:
                by.setdefault(e['transition'], []).append((arm, e))
    for reason in sorted(by):
        rows = by[reason]
        cr = [e['credit'] for _a, e in rows]
        paid = [sum(e['channels']['complete'][k] for k in COMPLETE_DRIVEN)
                for _a, e in rows]
        step = [e['transition_step'] for _a, e in rows]
        print(f"   {reason:<22} n={len(rows):>3}  credit {_stats(cr)}")
        print(f"   {'':<22}        at agent step median {np.median(step):.0f}, "
              f"COMPLETE-driven paid median {np.median(paid):.2f} max {max(paid):.2f}  "
              f"arms {sorted({a for a, _e in rows})}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default=None, metavar="DIR",
                    help="summarise the arm pickles in DIR instead of running")
    ap.add_argument("--arms", default="")
    ap.add_argument("--episodes", type=int, default=None,
                    help="override every arm's episode count")
    ap.add_argument("--out", default="calibration_runs")
    ap.add_argument("--model", default=config.BASELINE_MODEL)
    ap.add_argument("--bootstrap", default=config.BOOTSTRAP_COVERAGE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.report:
        report(args.report)
        return

    import torch
    from stable_baselines3 import PPO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model = PPO.load(args.model, device=device)
    print(f"model {args.model} @ {model.num_timesteps:,} steps "
          f"({device}, inference only)", flush=True)
    specs = arm_specs(model)
    names = [a for a in args.arms.split(",") if a] or list(specs)
    os.makedirs(args.out, exist_ok=True)
    for name in names:
        spec = dict(specs[name])
        if args.episodes:
            spec['n'] = args.episodes
        eps = run_arm(name, spec, args.bootstrap, args.seed)
        with open(os.path.join(args.out, f"{name}.pkl"), "wb") as fh:
            pickle.dump(eps, fh)
    print("done:", ", ".join(names))


if __name__ == "__main__":
    main()
