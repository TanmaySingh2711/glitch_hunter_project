"""The Stable-Baselines3 callbacks train_agent.py composes into a run.

Each one has a single job and never stops training on its own initiative,
with two deliberate exceptions, both documented on the class: the watchdog
halts a BLIND agent, and Level1CompletionCallback stops the QA phase the
moment Level 1 is fully covered.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from collections import deque
from typing import Any, cast

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import FloatSchedule

from common.logging_setup import write_progress
from exploration import config as xconfig
from exploration import level_completion as lc
from exploration.coverage import SpatialCoverage

log = logging.getLogger(__name__)


class ExactMilestoneCheckpointCallback(BaseCallback):
    """
    Saves a checkpoint at EXACT cumulative step counts — not "every N calls
    since this callback object was created."

    ─── WHY THIS REPLACES SB3's BUILT-IN CheckpointCallback ───
    SB3's CheckpointCallback decides when to save by counting its OWN
    `self.n_calls` since the callback object was instantiated, not by
    looking at the model's persistent `self.num_timesteps`. Every time
    train_agent.py is re-run (crash, manual stop, laptop sleep, anything),
    a BRAND NEW CheckpointCallback is created with n_calls starting back at
    0 — while num_timesteps correctly keeps climbing from the loaded
    checkpoint. The result: the "next save" lands at (wherever training
    happened to be when you restarted) + 50,000, not at the next clean
    round number. This is exactly why checkpoints appeared at ugly numbers
    like 827344 and 1227344 instead of 800000 and 1200000 — nothing was
    lost or corrupted, the save points were just measured from the wrong
    reference point.

    This callback fixes it by comparing against `self.num_timesteps`
    directly (which persists correctly across restarts as long as
    reset_num_timesteps=False) against a fixed, explicit list of target
    step counts, with a one-shot-per-target sentinel file so a target can
    never be saved twice even if num_timesteps jumps past it in one batch
    step (NUM_ENVS=8 means num_timesteps advances 8 at a time, so it can
    step over an exact target value without ever equaling it — this uses
    >= specifically to handle that).

    Crucially: this callback NEVER returns False. It only saves a file and
    keeps going — no pauses, ever, matching "I don't want any pauses."

    ─── TWO MODES ───
    `targets`  an explicit list - the legacy completion phase, unchanged.
    `every`    open-ended: every multiple of `every` global timesteps, with no
               last one (QA mode, Phase 4F). Only milestones CROSSED DURING
               THIS RUN are saved - multiples above the step the run started
               at - so a missing flag can never relabel today's model as an
               older milestone, and nothing at or below the 6M seed is ever
               written.
    """

    def __init__(self, targets: list[int] | None = None, save_path: str | None = None,
                 name_prefix: str | None = None, coverage: SpatialCoverage | None = None,
                 every: int | None = None, verbose: int = 0) -> None:
        super().__init__(verbose)
        if (targets is None) == (every is None):
            raise ValueError("give exactly one of targets= or every=")
        if save_path is None or name_prefix is None:
            raise ValueError("save_path= and name_prefix= are required")
        self.targets = sorted(targets) if targets is not None else None
        self.every = every
        self._start = 0
        self.save_path = save_path
        self.name_prefix = name_prefix
        # When present, the exploration bitmap is saved as a MATCHED PAIR with
        # every model zip. A checkpoint without its coverage is not resumable:
        # reloading it against the wrong map would either re-pay the agent for
        # ground it already explored or starve it of reward for ground it has
        # not, and both look like ordinary training rather than like a bug.
        self.coverage = coverage
        self._saved: set[int] = set()
        self.last_saved: str | None = None          # the newest checkpoint this run wrote

    def _sentinel_path(self, target: int) -> str:
        return os.path.join(self.save_path, f".milestone_saved_{self.name_prefix}_{target}.flag")

    def _init_callback(self) -> None:
        # ─── DO NOT DELETE THE .flag FILES IN checkpoints/ ───
        # They are 0 bytes and look like junk, but they are the ONLY record
        # of which milestones have already been saved. Deleting them makes
        # this set empty, so on the very next step every target at or below
        # the current step count is treated as unsaved and gets rewritten —
        # e.g. resuming at 6,000,000 with no flags overwrites all 15
        # milestone .zip files with copies of the current model, destroying
        # the entire training history in one step.
        os.makedirs(self.save_path, exist_ok=True)
        if self.targets is not None:
            self._saved = {t for t in self.targets if os.path.exists(self._sentinel_path(t))}
        else:
            self._saved = set()
            self._start = int(self.model.num_timesteps)

    def _due(self) -> list[int]:
        if self.targets is not None:
            return [t for t in self.targets
                    if self.num_timesteps >= t and t not in self._saved]
        every = cast(int, self.every)
        target = (self.num_timesteps // every) * every
        if target <= self._start or target in self._saved:
            return []
        if os.path.exists(self._sentinel_path(target)):
            self._saved.add(target)
            return []
        return [target]

    def _on_step(self) -> bool:
        for target in self._due():
            path = os.path.join(self.save_path, f"{self.name_prefix}_{target}_steps.zip")
            # ─── SAVE ORDER IS LOAD-BEARING ───
            # model zip -> coverage npz -> flag. The flag is the commit
            # point: written last, so a crash partway through leaves the
            # milestone unflagged and it is simply redone on the next run.
            # Writing the flag first would permanently mark a milestone
            # done while its coverage file was missing or half-written.
            self.model.save(path)
            if self.coverage is not None:
                cov_path = os.path.join(
                    self.save_path,
                    f"{self.name_prefix}_{target}_steps_coverage.npz")
                self.coverage.save(cov_path,
                                   model_timesteps=self.num_timesteps)
            with open(self._sentinel_path(target), "w"):
                pass
            self._saved.add(target)
            self.last_saved = path
            log.info("[CHECKPOINT] Saved exact milestone: %s (at %s steps)",
                     path, f"{self.num_timesteps:,}")
            if self.coverage is not None and self.coverage.testable_total:
                covered = self.coverage.covered_testable()
                log.info("[CHECKPOINT] Paired coverage: %s / %s testable px (%s)",
                         f"{covered:,}", f"{self.coverage.testable_total:,}",
                         lc.pct_text(covered, self.coverage.testable_total))
            elif self.coverage is not None:
                log.info("[CHECKPOINT] Paired coverage: %s unique world px",
                         f"{self.coverage.total_unique():,}")
        return True


class WatchdogCallback(BaseCallback):
    """
    Passive training monitor. It never stops training except in the one case
    where continuing is provably pointless (a blind agent).

    1. Detects a blind agent (all-black observations) and halts — this means
       the render pipeline broke, so every further step would train on
       garbage.
    2. Warns on possible policy collapse (average reward fell below half of
       the all-time peak), rate-limited to one alert per 50k steps.

    ─── NOTE ON READING COLLAPSE ALERTS ───
    `peak_reward` is an all-time running maximum that never decays, so once
    the agent hits a lucky high streak, ANY normal regression below half of
    it keeps re-triggering this warning even when nothing is actually wrong.
    Treat the alert as "look at the numbers", not "something broke". The
    signature of a REAL collapse is different and much more specific:
    `approx_kl` spiking into double digits and `entropy_loss` crashing to
    near-zero within one or two iterations (see the target_kl comment in
    train_agent.build_model for the incident this was written from).
    """

    PROGRESS_EVERY = 1000
    BLIND_CHECK_EVERY = 5000
    ALERT_EVERY = 50_000
    MIN_HISTORY = 50

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.reward_history: deque[float] = deque(maxlen=100)  # rolling episode rewards
        self.peak_reward = -float('inf')
        self.last_alert_step = 0

    def _on_step(self) -> bool:
        # ─── LIVE PROGRESS INDICATOR ───
        if self.num_timesteps % self.PROGRESS_EVERY == 0:
            write_progress(f"Crunching frames... {self.num_timesteps:,} steps collected")

        # ─── BLIND CHECK ───
        if self.num_timesteps % self.BLIND_CHECK_EVERY == 0:
            obs = self.locals.get("new_obs")
            if obs is not None and float(np.mean(obs)) == 0.0:
                log.critical("AI IS BLIND (Mean pixel = 0.0). Stopping training.")
                return False

        # ─── EPISODE ANALYTICS ───
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            self.reward_history.append(float(info["episode"]["r"]))
            avg_reward = float(np.mean(self.reward_history))
            self.peak_reward = max(self.peak_reward, avg_reward)

            # ─── COLLAPSE DETECTION ───
            # Enough data, and the average dropped more than 50% below the
            # peak - alerted at most once per ALERT_EVERY steps.
            if (len(self.reward_history) >= self.MIN_HISTORY and self.peak_reward > 0
                    and avg_reward < self.peak_reward * 0.5
                    and self.num_timesteps - self.last_alert_step > self.ALERT_EVERY):
                self.last_alert_step = self.num_timesteps
                log.warning("WATCHDOG ALERT: Possible brain collapse detected! "
                            "Peak reward %.1f, current avg %.1f, step %s. Continuing "
                            "training (entropy should help recovery).",
                            self.peak_reward, avg_reward, f"{self.num_timesteps:,}")
        return True


class ValueWarmupCallback(BaseCallback):
    """Runs the first stretch of the QA phase at a reduced learning rate.

    ─── WHY THIS EXISTS ───
    The 6M value function was fitted against completion-phase returns. The QA
    reward is a different function, so on the very first update every value
    prediction it makes is wrong - not slightly wrong, wrong about a quantity
    it has never seen. Large TD errors become large advantages, and large
    advantages are how one PPO update moves the policy much further than
    clip_range was ever meant to permit.

    This project has already been through that exact failure once, at roughly
    3.22M steps: approx_kl 11.37, clip_fraction 0.834, entropy_loss collapsing
    to near zero. A lower learning rate for the first VF_WARMUP_STEPS lets the
    critic re-fit to the new reward scale before the actor is allowed to move
    far on its estimates. Calibration (tools/calibrate_reward.py) shrinks the
    initial error; this limits the damage from whatever error is left.

    The learning rate is set in TWO places on purpose. PPO.load() restores an
    lr_schedule CLOSURE built from the checkpoint's saved rate, and the
    optimizer reads the schedule, not the attribute - so assigning
    model.learning_rate alone silently does nothing at all.
    """

    def __init__(self, warmup_until: int, warmup_lr: float, normal_lr: float,
                 verbose: int = 0) -> None:
        super().__init__(verbose)
        self.warmup_until = warmup_until
        self.warmup_lr = warmup_lr
        self.normal_lr = normal_lr
        self._restored = False

    def _apply(self, lr: float) -> None:
        self.model.learning_rate = lr
        self.model.lr_schedule = FloatSchedule(lr)

    def _on_training_start(self) -> None:
        if self.num_timesteps >= self.warmup_until:
            self._restored = True
            self._apply(self.normal_lr)
            log.info("[WARMUP] Past %s; running at lr=%g.",
                     f"{self.warmup_until:,}", self.normal_lr)
        else:
            self._apply(self.warmup_lr)
            log.info("[WARMUP] Value-function warm-up active: lr=%g until %s steps, then %g.",
                     self.warmup_lr, f"{self.warmup_until:,}", self.normal_lr)

    def _on_step(self) -> bool:
        if not self._restored and self.num_timesteps >= self.warmup_until:
            self._restored = True
            self._apply(self.normal_lr)
            log.info("[WARMUP] Warm-up complete at %s steps; learning rate restored to %g.",
                     f"{self.num_timesteps:,}", self.normal_lr)
        return True


class CoverageStatsCallback(BaseCallback):
    """Reports what the run is actually for: how much of the world is known.

    Episode reward is NOT the headline number in QA mode. A run can hold a
    perfectly healthy reward curve while discovering nothing new, because
    once the frontier is exhausted the drought and time penalties settle into
    a stable, unremarkable-looking equilibrium. Coverage is the number that
    says whether the run is doing its job.
    """

    def __init__(self, coverage: SpatialCoverage | None, every: int = 10_000,
                 session_start_covered: int | None = None,
                 audit_path: str | None = None, trail_path: str | None = None,
                 map_path: str | None = None,
                 checkpoints: ExactMilestoneCheckpointCallback | None = None,
                 verbose: int = 0) -> None:
        super().__init__(verbose)
        self.coverage = coverage
        self.every = every
        self._next = 0
        self._last_total: int | None = None
        self._last_step = 0
        # What this run/session found, as opposed to the campaign's total.
        self.session_start_covered = session_start_covered
        # Late in a campaign: where the remaining pixels are and whether
        # discovery has stalled, for a later audit (Phase 4F) - as JSON and as
        # a picture of the level. Never deletes or reclassifies a pixel.
        self.audit_path = audit_path
        self.map_path = map_path
        self.plateau = lc.PlateauTracker()
        # THE AUDIT TRAIL: one JSON line per report, appended - never
        # rewritten - across every run of the campaign, so the whole history
        # of coverage growth survives restarts (the console does not).
        self.trail_path = trail_path
        self.checkpoints = checkpoints
        self._episodes = 0

    def _append_trail(self, entry: dict[str, Any]) -> None:
        path = cast(str, self.trail_path)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(entry) + '\n')

    def _on_step(self) -> bool:
        self._episodes += int(sum(bool(d) for d in self.locals.get("dones", ())))
        if self.coverage is None or self.num_timesteps < self._next:
            return True
        self._next = self.num_timesteps + self.every
        cov = self.coverage

        covered = cov.covered_testable()
        if self.session_start_covered is None:
            self.session_start_covered = covered
        rate = 0.0
        if self._last_total is not None and self.num_timesteps > self._last_step:
            gained = covered - self._last_total
            rate = gained * 10_000.0 / (self.num_timesteps - self._last_step)
        self._last_total, self._last_step = covered, self.num_timesteps

        remaining = cov.remaining()
        pct = cov.coverage_pct()
        cov.assert_consistent()
        # noncoverage is mostly normal (jump arcs, pit deaths, collision
        # tolerance); anomalous is the genuinely impossible subset and is the
        # one worth watching - if it leaves zero mid-run, a glitch was found.
        noncov = cov.noncoverage_px()
        anomalous = cov.anomalous_px()
        session_new = covered - self.session_start_covered
        testable_total = cov.testable_total
        # pct_text truncates: 4,013,722 covered prints 99.9999%, never 100%.
        shown = (lc.pct_text(covered, testable_total) if testable_total
                 else f"{pct:.2f}%")
        tail = f" | remaining {remaining:,}" if remaining is not None else ""
        log.info("[COVERAGE] step %s | covered %s / %s testable (%s)%s | +%s this session "
                 "| %s new px per 10k | noncoverage %s (anomalous %s)",
                 f"{self.num_timesteps:,}", f"{covered:,}", f"{testable_total or 0:,}",
                 shown, tail, f"{session_new:,}", f"{rate:,.0f}", f"{noncov:,}",
                 f"{anomalous:,}")

        plateau = self.plateau.update(self.num_timesteps, covered)
        if remaining is not None and 0 < remaining < xconfig.STAGNATION_REMAINING_MIN:
            self._report_plateau(remaining, plateau)
        if self.trail_path and testable_total and remaining is not None:
            origin = lc.provenance(cov)
            self._append_trail({
                'global_timestep': int(self.num_timesteps),
                'covered_testable_px': int(covered),
                'testable_total': int(testable_total),
                'coverage_pct_text': shown,
                'remaining_testable_px': int(remaining),
                'new_px_per_10k': round(rate, 1),
                'session_new_px': int(session_new),
                'episodes_this_session': self._episodes,
                'last_checkpoint_this_session': (self.checkpoints.last_saved
                                                 if self.checkpoints is not None else None),
                'bootstrap_known_px': origin.get('bootstrap_known_testable_px'),
                'qa_discovered_px': origin.get('qa_discovered_testable_px'),
                'stalled': bool(plateau['stalled']),
                'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
            })
        if self.logger is not None:
            self.logger.record("coverage/covered_testable_px", covered)
            self.logger.record("coverage/noncoverage_px", noncov)
            self.logger.record("coverage/anomalous_px", anomalous)
            self.logger.record("coverage/pct_testable", pct)
            self.logger.record("coverage/new_px_per_10k", rate)
            self.logger.record("coverage/session_new_px", session_new)
            self.logger.record("coverage/stalled", float(plateau['stalled']))
            self.logger.record("coverage/novelty_mult", cov.novelty_mult)
            if remaining is not None:
                self.logger.record("coverage/remaining_px", remaining)
        return True

    def _report_plateau(self, remaining: int, plateau: dict[str, Any]) -> None:
        """Late in a campaign: where the leftover pixels are, as a log line,
        a JSON audit and (optionally) a picture of the level."""
        cov = cast(SpatialCoverage, self.coverage)
        report = lc.plateau_report(cov, self.num_timesteps, plateau)
        regions = report['remaining_regions']
        big = regions['largest'][0] if regions['largest'] else None
        where = (f"; largest {big['pixels']:,} px at x {big['world_bbox'][0]}-"
                 f"{big['world_bbox'][2]}, y {big['world_bbox'][1]}-{big['world_bbox'][3]}"
                 if big else "")
        log.info("[PLATEAU] %s testable px remain in %s region(s)%s | no new pixel for %s "
                 "steps%s", f"{remaining:,}", regions['regions'], where,
                 f"{plateau['steps_since_last_gain']:,}",
                 ' -> STALLED' if plateau['stalled'] else '')
        if self.map_path:
            try:
                report['map_png'] = lc.write_remaining_map(cov, self.map_path, regions)
            except Exception as exc:          # a picture must never stop training
                report['map_error'] = str(exc)
        if self.audit_path:
            lc.write_json_atomic(report, self.audit_path)


class Level1CompletionCallback(BaseCallback):
    """Ends QA training the moment Level 1 is fully covered (Phase 4F).

    THE CONDITION is lc.is_level_complete: covered testable pixels ==
    4,013,723, exactly. A timestep never ends QA training as a success.

    WHEN IT CHECKS. The exact count is a pass over the ~10M-pixel bitmap
    (~13 ms), too slow for every step, so it runs every `check_every` vector
    steps - and ALWAYS on the last collection step of a rollout. That second
    rule is the guarantee: the policy only changes in PPO's train(), which
    runs after a rollout is collected. Checking on the last collection step
    means completion is always seen before the next update, so the snapshot
    holds exactly the policy that finished the level, and no spatial-novelty
    update is ever made after it. (Returning False from on_step makes SB3's
    learn() break out BEFORE train().)

    ON COMPLETION: the policy, the exact coverage and the proof are written
    at once into snapshot_dir - even between periodic checkpoints - and
    training stops.
    """

    def __init__(self, coverage: SpatialCoverage, snapshot_dir: str, name_prefix: str,
                 session_start: dict[str, int], check_every: int = 256,
                 verbose: int = 0) -> None:
        super().__init__(verbose)
        self.coverage = coverage
        self.snapshot_dir = snapshot_dir
        self.name_prefix = name_prefix
        self.session_start = dict(session_start)
        self.check_every = check_every
        self.completed = False
        self.snapshot: dict[str, Any] | None = None
        self._since_check = 0
        self._last_check: tuple[int, int] | None = None

    def _last_collection_step(self) -> bool:
        n, n_total = self.locals.get("n_steps"), self.locals.get("n_rollout_steps")
        return n is not None and n_total is not None and n == n_total - 1

    def _on_step(self) -> bool:
        self._since_check += 1
        if self._since_check < self.check_every and not self._last_collection_step():
            return True
        self._since_check = 0
        covered = self.coverage.covered_testable()
        previous, self._last_check = self._last_check, (self.num_timesteps, covered)
        if not lc.is_level_complete(covered, cast(int, self.coverage.testable_total)):
            return True
        self.snapshot = lc.write_completion_snapshot(
            model=self.model, coverage=self.coverage,
            snapshot_dir=self.snapshot_dir, name_prefix=self.name_prefix,
            timesteps=self.num_timesteps, session_start=self.session_start,
            previous_check=previous)
        self.completed = True
        paths = self.snapshot['_paths']
        rule = '=' * 68
        log.info(rule)
        log.info("[LEVEL 1 COMPLETE] %s / %s testable px covered at global step %s.",
                 f"{covered:,}", f"{self.coverage.testable_total:,}",
                 f"{self.num_timesteps:,}")
        log.info("[LEVEL 1 COMPLETE] policy   -> %s", paths['model'])
        log.info("[LEVEL 1 COMPLETE] coverage -> %s", paths['coverage'])
        log.info("[LEVEL 1 COMPLETE] proof    -> %s", paths['metadata'])
        log.info("[LEVEL 1 COMPLETE] Stopping QA training; no further updates.")
        log.info(rule)
        return False


class LifecycleStatsCallback(BaseCallback):
    """How episodes move through EXPLORE -> COMPLETE, and how they end.

    Counts per-episode outcomes from the final info of every finished
    episode, across all workers, and reports them on the same cadence as the
    coverage line. Two numbers here are diagnostic alarms rather than stats:

      * T4 (the explore backstop) should be RARE. If it is the usual
        transition, the adaptive criteria are mis-set - the brief says report
        it rather than tune around it.
      * safety_reset should be rare too. It is the last-resort ending, after
        level completion and death. Each is reported with the evidence it
        fired on (stuck / unproductive_loop).

    Also reported: the longest episode, and how many outlived the old QA clock
    (9,781 agent steps) - under the respawn rule those are the episodes the
    engine timer used to kill.
    """

    def __init__(self, every: int = 10_000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.every = every
        self._next = 0
        self._transitions: dict[str, int] = {}
        self._ends: dict[str, int] = {}
        self._episodes = 0
        self._outlived_clock = 0
        self._longest = 0

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get("dones", []),
                              self.locals.get("infos", []), strict=True):
            if not done:
                continue
            self._episodes += 1
            t = info.get("phase_transition_reason") or "none"
            self._transitions[t] = self._transitions.get(t, 0) + 1
            end = info.get("episode_end_reason") or "unknown"
            # TimeLimit sits above the wrapper; SB3 flags its truncation here.
            if info.get("TimeLimit.truncated"):
                end = "time_limit"
            if end == "safety_reset" and info.get("safety_reset_reason"):
                end = f"safety_reset:{info['safety_reset_reason']}"
            self._ends[end] = self._ends.get(end, 0) + 1
            self._outlived_clock += bool(info.get("clock_extensions"))
            self._longest = max(self._longest, int(info.get("lifecycle_agent_steps") or 0))

        if self.num_timesteps < self._next or not self._episodes:
            return True
        self._next = self.num_timesteps + self.every

        def fmt(d: dict[str, int]) -> str:
            return ", ".join(f"{k} {v}" for k, v in sorted(d.items()))
        log.info("[LIFECYCLE] %d episodes | transitions: %s | ends: %s | longest %s steps, "
                 "%d outlived the old %s-step clock", self._episodes,
                 fmt(self._transitions), fmt(self._ends), f"{self._longest:,}",
                 self._outlived_clock, f"{xconfig.QA_EPISODE_CAP_AGENT_STEPS_MEASURED:,}")
        if self.logger is not None:
            for k, v in self._transitions.items():
                self.logger.record(f"lifecycle/transition_{k}", v / self._episodes)
            for k, v in self._ends.items():
                self.logger.record(f"lifecycle/end_{k}", v / self._episodes)
            self.logger.record("lifecycle/longest_episode_steps", self._longest)
            self.logger.record("lifecycle/outlived_old_clock", self._outlived_clock / self._episodes)
        self._transitions, self._ends, self._episodes = {}, {}, 0
        self._outlived_clock, self._longest = 0, 0
        return True


class StagnationCallback(BaseCallback):
    """Escalates exploration pressure when discovery genuinely stalls.

    Fires only when ALL of these hold:
      * the new-pixel rate stayed under STAGNATION_RATE_THRESHOLD per 10k
        steps for STAGNATION_WINDOW consecutive windows, and
      * there is still meaningful unexplored space left
        (at least STAGNATION_REMAINING_MIN px).

    That second condition is the important one. Late in a campaign the rate
    falls toward zero because the world is nearly exhausted, which is SUCCESS,
    not stagnation - escalating there would only push a converged policy
    around for nothing.

    The novelty multiplier is written through SHARED MEMORY rather than to
    config. Workers are separate processes; a module-level assignment here
    would change only the parent's copy, and this callback would appear to
    work while having no effect whatsoever on the rollouts.
    """

    NOVELTY_STEP = 1.25
    ENTROPY_STEP = 1.15

    def __init__(self, coverage: SpatialCoverage | None, every: int = 10_000,
                 verbose: int = 0) -> None:
        super().__init__(verbose)
        self.coverage = coverage
        self.every = every
        self._next = 0
        self._last_total: int | None = None
        self._last_step = 0
        self._lean_windows = 0

    def _on_step(self) -> bool:
        if self.coverage is None or self.num_timesteps < self._next:
            return True
        self._next = self.num_timesteps + self.every

        total = self.coverage.covered_testable()
        if self._last_total is None:
            self._last_total, self._last_step = total, self.num_timesteps
            return True
        span = max(1, self.num_timesteps - self._last_step)
        rate = (total - self._last_total) * 10_000.0 / span
        self._last_total, self._last_step = total, self.num_timesteps

        remaining = self.coverage.remaining()
        if remaining is not None and remaining < xconfig.STAGNATION_REMAINING_MIN:
            self._lean_windows = 0
            return True

        if rate >= xconfig.STAGNATION_RATE_THRESHOLD:
            self._lean_windows = 0
            return True

        self._lean_windows += 1
        if self._lean_windows < xconfig.STAGNATION_WINDOW:
            return True
        self._lean_windows = 0

        model = cast(PPO, self.model)
        mult = min(xconfig.NOVELTY_MULT_MAX, self.coverage.novelty_mult * self.NOVELTY_STEP)
        ent = min(xconfig.ENT_COEF_MAX, float(model.ent_coef) * self.ENTROPY_STEP)
        self.coverage.set_novelty_mult(mult)
        model.ent_coef = ent
        log.info("[STAGNATION] %s new px per 10k for %d windows with %s px still unexplored.",
                 f"{rate:,.0f}", xconfig.STAGNATION_WINDOW,
                 f"{remaining:,}" if remaining is not None else "?")
        log.info("             novelty_mult -> %.2f (max %s), ent_coef -> %.4f (max %s)",
                 self.coverage.novelty_mult, xconfig.NOVELTY_MULT_MAX, ent,
                 xconfig.ENT_COEF_MAX)
        return True
