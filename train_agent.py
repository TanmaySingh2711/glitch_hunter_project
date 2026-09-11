import argparse
import datetime
import json
import os
import sys
import zipfile
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import get_schedule_fn
import numpy as np
from collections import deque

from exploration import config as xconfig
from exploration import coverage as coverage_mod
from exploration import level_completion as lc

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
    def __init__(self, targets=None, save_path=None, name_prefix=None, coverage=None,
                 every=None, verbose=0):
        super().__init__(verbose)
        if (targets is None) == (every is None):
            raise ValueError("give exactly one of targets= or every=")
        self.targets = sorted(targets) if targets is not None else None
        self.every = every
        self._start = None
        self.save_path = save_path
        self.name_prefix = name_prefix
        # When present, the exploration bitmap is saved as a MATCHED PAIR with
        # every model zip. A checkpoint without its coverage is not resumable:
        # reloading it against the wrong map would either re-pay the agent for
        # ground it already explored or starve it of reward for ground it has
        # not, and both look like ordinary training rather than like a bug.
        self.coverage = coverage
        self._saved = set()
        self.last_saved = None          # the newest checkpoint this run wrote

    def _sentinel_path(self, target):
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

    def _due(self):
        if self.targets is not None:
            return [t for t in self.targets
                    if self.num_timesteps >= t and t not in self._saved]
        target = (self.num_timesteps // self.every) * self.every
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
            open(self._sentinel_path(target), "w").close()
            self._saved.add(target)
            self.last_saved = path
            print(f"\n[CHECKPOINT] Saved exact milestone: {path} (at {self.num_timesteps:,} steps)")
            if self.coverage is not None and self.coverage.testable_total:
                covered = self.coverage.covered_testable()
                print(f"[CHECKPOINT] Paired coverage: {covered:,} / "
                      f"{self.coverage.testable_total:,} testable px "
                      f"({lc.pct_text(covered, self.coverage.testable_total)})")
            elif self.coverage is not None:
                print(f"[CHECKPOINT] Paired coverage: "
                      f"{self.coverage.total_unique():,} unique world px")
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
    near-zero within one or two iterations (see the target_kl comment below
    for the incident this was written from).
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.reward_history = deque(maxlen=100)  # Rolling window of episode rewards
        self.peak_reward = -float('inf')
        self.last_alert_step = 0

    def _on_step(self) -> bool:
        # ─── LIVE PROGRESS INDICATOR ───
        if self.num_timesteps % 1000 == 0:
            print(f"Crunching frames... {self.num_timesteps:,} steps collected", end="\r", flush=True)

        # ─── BLIND CHECK (every 5000 steps) ───
        if self.num_timesteps % 5000 == 0:
            obs = self.locals.get("new_obs")
            if obs is not None:
                mean_pixel = np.mean(obs)
                if mean_pixel == 0.0:
                    print("CRITICAL: AI IS BLIND (Mean pixel = 0.0). Stopping training.")
                    return False

        # ─── EPISODE ANALYTICS ───
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                self.reward_history.append(ep_reward)

                # Track peak reward
                avg_reward = np.mean(self.reward_history)
                if avg_reward > self.peak_reward:
                    self.peak_reward = avg_reward

                # ─── COLLAPSE DETECTION ───
                # If we have enough data and reward has dropped >50% from peak
                if len(self.reward_history) >= 50 and self.peak_reward > 0:
                    if avg_reward < self.peak_reward * 0.5:
                        # Only alert once per 50k steps to avoid spam
                        if self.num_timesteps - self.last_alert_step > 50000:
                            self.last_alert_step = self.num_timesteps
                            print("\n[WARNING] WATCHDOG ALERT: Possible brain collapse detected!")
                            print(f"    Peak reward: {self.peak_reward:.1f}")
                            print(f"    Current avg: {avg_reward:.1f}")
                            print(f"    Step: {self.num_timesteps:,}")
                            print("    Continuing training (entropy should help recovery)...\n")

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

    def __init__(self, warmup_until, warmup_lr, normal_lr, verbose=0):
        super().__init__(verbose)
        self.warmup_until = warmup_until
        self.warmup_lr = warmup_lr
        self.normal_lr = normal_lr
        self._restored = False

    def _apply(self, lr):
        self.model.learning_rate = lr
        self.model.lr_schedule = get_schedule_fn(lr)

    def _on_training_start(self) -> None:
        if self.num_timesteps >= self.warmup_until:
            self._restored = True
            self._apply(self.normal_lr)
            print(f"[WARMUP] Past {self.warmup_until:,}; running at "
                  f"lr={self.normal_lr:g}.")
        else:
            self._apply(self.warmup_lr)
            print(f"[WARMUP] Value-function warm-up active: lr="
                  f"{self.warmup_lr:g} until {self.warmup_until:,} steps, "
                  f"then {self.normal_lr:g}.")

    def _on_step(self) -> bool:
        if not self._restored and self.num_timesteps >= self.warmup_until:
            self._restored = True
            self._apply(self.normal_lr)
            print(f"{os.linesep}[WARMUP] Warm-up complete at "
                  f"{self.num_timesteps:,} steps; learning rate restored to "
                  f"{self.normal_lr:g}.")
        return True


class CoverageStatsCallback(BaseCallback):
    """Reports what the run is actually for: how much of the world is known.

    Episode reward is NOT the headline number in QA mode. A run can hold a
    perfectly healthy reward curve while discovering nothing new, because
    once the frontier is exhausted the drought and time penalties settle into
    a stable, unremarkable-looking equilibrium. Coverage is the number that
    says whether the run is doing its job.
    """

    def __init__(self, coverage, every=10_000, session_start_covered=None,
                 audit_path=None, trail_path=None, map_path=None, checkpoints=None,
                 verbose=0):
        super().__init__(verbose)
        self.coverage = coverage
        self.every = every
        self._next = 0
        self._last_total = None
        self._last_step = None
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
        self.checkpoints = checkpoints          # ExactMilestoneCheckpointCallback
        self._episodes = 0

    def _append_trail(self, entry):
        os.makedirs(os.path.dirname(self.trail_path) or '.', exist_ok=True)
        with open(self.trail_path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(entry) + '\n')

    def _on_step(self) -> bool:
        self._episodes += int(sum(bool(d) for d in self.locals.get("dones", ())))
        if self.coverage is None or self.num_timesteps < self._next:
            return True
        self._next = self.num_timesteps + self.every

        covered = self.coverage.covered_testable()
        if self.session_start_covered is None:
            self.session_start_covered = covered
        total = covered
        rate = 0.0
        if self._last_total is not None and self.num_timesteps > self._last_step:
            gained = total - self._last_total
            rate = gained * 10_000.0 / (self.num_timesteps - self._last_step)
        self._last_total, self._last_step = total, self.num_timesteps

        remaining = self.coverage.remaining()
        pct = self.coverage.coverage_pct()
        self.coverage.assert_consistent()
        # noncoverage is mostly normal (jump arcs, pit deaths, collision
        # tolerance); anomalous is the genuinely impossible subset and is the
        # one worth watching - if it leaves zero mid-run, a glitch was found.
        noncov = self.coverage.noncoverage_px()
        anomalous = self.coverage.anomalous_px()
        session_new = covered - self.session_start_covered
        testable_total = self.coverage.testable_total
        # pct_text truncates: 4,013,722 covered prints 99.9999%, never 100%.
        shown = (lc.pct_text(covered, testable_total) if testable_total
                 else f"{pct:.2f}%")
        tail = f" | remaining {remaining:,}" if remaining is not None else ""
        print(f"{os.linesep}[COVERAGE] step {self.num_timesteps:,} | covered "
              f"{covered:,} / {testable_total or 0:,} testable ({shown}){tail} | "
              f"+{session_new:,} this session | {rate:,.0f} new px per 10k | "
              f"noncoverage {noncov:,} (anomalous {anomalous:,})")

        plateau = self.plateau.update(self.num_timesteps, covered)
        if (remaining is not None and 0 < remaining < xconfig.STAGNATION_REMAINING_MIN):
            report = lc.plateau_report(self.coverage, self.num_timesteps, plateau)
            regions = report['remaining_regions']
            big = regions['largest'][0] if regions['largest'] else None
            where = (f"; largest {big['pixels']:,} px at x {big['world_bbox'][0]}-"
                     f"{big['world_bbox'][2]}, y {big['world_bbox'][1]}-{big['world_bbox'][3]}"
                     if big else "")
            print(f"[PLATEAU] {remaining:,} testable px remain in "
                  f"{regions['regions']} region(s){where} | no new pixel for "
                  f"{plateau['steps_since_last_gain']:,} steps"
                  f"{' -> STALLED' if plateau['stalled'] else ''}")
            if self.map_path:
                try:
                    report['map_png'] = lc.write_remaining_map(self.coverage, self.map_path,
                                                               regions)
                except Exception as exc:          # a picture must never stop training
                    report['map_error'] = str(exc)
            if self.audit_path:
                lc.write_json_atomic(report, self.audit_path)
        if self.trail_path and testable_total:
            origin = lc.provenance(self.coverage)
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
            self.logger.record("coverage/novelty_mult",
                               self.coverage.novelty_mult)
            if remaining is not None:
                self.logger.record("coverage/remaining_px", remaining)
        return True


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

    def __init__(self, coverage, snapshot_dir, name_prefix, session_start,
                 check_every=256, verbose=0):
        super().__init__(verbose)
        self.coverage = coverage
        self.snapshot_dir = snapshot_dir
        self.name_prefix = name_prefix
        self.session_start = dict(session_start)
        self.check_every = check_every
        self.completed = False
        self.snapshot = None
        self._since_check = 0
        self._last_check = None

    def _last_collection_step(self):
        n, n_total = self.locals.get("n_steps"), self.locals.get("n_rollout_steps")
        return n is not None and n_total is not None and n == n_total - 1

    def _on_step(self) -> bool:
        self._since_check += 1
        if self._since_check < self.check_every and not self._last_collection_step():
            return True
        self._since_check = 0
        covered = self.coverage.covered_testable()
        previous, self._last_check = self._last_check, (self.num_timesteps, covered)
        if not lc.is_level_complete(covered, self.coverage.testable_total):
            return True
        self.snapshot = lc.write_completion_snapshot(
            model=self.model, coverage=self.coverage,
            snapshot_dir=self.snapshot_dir, name_prefix=self.name_prefix,
            timesteps=self.num_timesteps, session_start=self.session_start,
            previous_check=previous)
        self.completed = True
        paths = self.snapshot['_paths']
        print(f"{os.linesep}{'=' * 68}")
        print(f"[LEVEL 1 COMPLETE] {covered:,} / {self.coverage.testable_total:,} "
              f"testable px covered at global step {self.num_timesteps:,}.")
        print(f"[LEVEL 1 COMPLETE] policy   -> {paths['model']}")
        print(f"[LEVEL 1 COMPLETE] coverage -> {paths['coverage']}")
        print(f"[LEVEL 1 COMPLETE] proof    -> {paths['metadata']}")
        print("[LEVEL 1 COMPLETE] Stopping QA training; no further updates.")
        print("=" * 68)
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

    def __init__(self, every=10_000, verbose=0):
        super().__init__(verbose)
        self.every = every
        self._next = 0
        self._transitions = {}
        self._ends = {}
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

        def fmt(d):
            return ", ".join(f"{k} {v}" for k, v in sorted(d.items()))
        print(f"[LIFECYCLE] {self._episodes} episodes | "
              f"transitions: {fmt(self._transitions)} | ends: {fmt(self._ends)} | "
              f"longest {self._longest:,} steps, {self._outlived_clock} outlived "
              f"the old {xconfig.QA_EPISODE_CAP_AGENT_STEPS_MEASURED:,}-step clock")
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

    def __init__(self, coverage, every=10_000, verbose=0):
        super().__init__(verbose)
        self.coverage = coverage
        self.every = every
        self._next = 0
        self._last_total = None
        self._last_step = None
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

        mult = min(xconfig.NOVELTY_MULT_MAX, self.coverage.novelty_mult * 1.25)
        ent = min(xconfig.ENT_COEF_MAX, self.model.ent_coef * 1.15)
        self.coverage.set_novelty_mult(mult)
        self.model.ent_coef = ent
        print(f"{os.linesep}[STAGNATION] {rate:,.0f} new px per 10k for "
              f"{xconfig.STAGNATION_WINDOW} windows with {remaining:,} px "
              f"still unexplored.")
        print(f"             novelty_mult -> {self.coverage.novelty_mult:.2f} "
              f"(max {xconfig.NOVELTY_MULT_MAX}), "
              f"ent_coef -> {ent:.4f} (max {xconfig.ENT_COEF_MAX})")
        return True


def make_env(rank, shm_names=None, reward_mode=None):
    """Returns a function that creates a single wrapped env instance.

    `shm_names` is a dict of multiprocessing.shared_memory block NAMES, not
    objects. That distinction is the whole reason this signature changed: a
    SharedMemory object cannot be cloudpickled into a SubprocVecEnv worker,
    but its name is just a string, and the worker re-attaches by name on the
    other side. All 8 workers therefore write into ONE bitmap.

    Without that, each worker would hold a private copy of "explored", every
    one of them would be paid full novelty for the same opening stretch of
    the level, and coverage would be counted eight times over forever.
    """
    # Resolved HERE, in the parent, and closed over as plain ints - so the
    # worker receives numbers rather than having to re-derive the mode.
    # LEGACY: TimeLimit is a BACKSTOP strictly above the engine timer's
    # measured cap (the engine clock is authoritative - see exploration/
    # config.py EPISODE LENGTH). It used to be 4000 in both modes, which was
    # dead code: the engine always timed out first, at 2,451 agent steps.
    # QA: none. A step count alone may not end a QA episode (the respawn
    # rule, config QA_TIMEOUT_ENDS_EPISODE); the lifecycle's evidence-based
    # safety reset is what ends an unproductive one.
    mode = reward_mode or xconfig.REWARD_MODE
    max_steps = (None if mode == "qa_exploration"
                 else xconfig.LEGACY_EPISODE_MAX_STEPS)
    skip = xconfig.SUBSTEPS_PER_AGENT_STEP

    def _init():
        import time
        time.sleep(rank * 0.5)  # Stagger window creation to prevent Windows DWM race conditions
        from custom_mario_env import CustomMarioEnv
        from agent_logic import GlitchHunterWrapper
        from gymnasium.wrappers import (
            MaxAndSkipObservation, GrayscaleObservation,
            ResizeObservation, FrameStackObservation, TimeLimit
        )
        from stable_baselines3.common.monitor import Monitor
        env = CustomMarioEnv()
        env = GlitchHunterWrapper(env, reward_mode=reward_mode,
                                  shm_names=shm_names)
        env = MaxAndSkipObservation(env, skip=skip)
        env = GrayscaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, 4)
        if max_steps is not None:
            env = TimeLimit(env, max_episode_steps=max_steps)
        env = Monitor(env)
        return env
    return _init

# ═══════════════════════════════════════════════════════════════════════
# PHASE SELECTION
#
# QA_PHASE follows exploration/config.py REWARD_MODE, so there is exactly ONE
# switch for the whole retrofit and no way for the reward and the checkpoint
# naming to disagree with each other.
#
# ─── THE 6M MASTER IS NEVER WRITTEN IN QA MODE ───
# mario_brain_checkpoint.zip and checkpoints/ belong to the completion-phase
# brain. QA training reads the master once, to seed itself, and from then on
# writes only to checkpoints_qa/ and glitch_hunter_qa.zip. Backing the
# retrofit out is therefore always possible: flip REWARD_MODE back and the
# original files are exactly where they were, untouched, byte for byte.
# ═══════════════════════════════════════════════════════════════════════
QA_PHASE = (xconfig.REWARD_MODE == "qa_exploration")

LEGACY_CHECKPOINT_DIR = "./checkpoints/"
LEGACY_CHECKPOINT_NAME = "mario_brain_checkpoint"

if QA_PHASE:
    CHECKPOINT_DIR = f"./{xconfig.CHECKPOINT_DIR_QA}/"
    CHECKPOINT_NAME = xconfig.CHECKPOINT_NAME_QA
else:
    CHECKPOINT_DIR = LEGACY_CHECKPOINT_DIR
    # Master checkpoint written on clean finish and on Ctrl+C. The checkpoint
    # discovery below prefers this file over the numbered milestones, so it is
    # always the resume point unless it is deleted or moved aside.
    CHECKPOINT_NAME = LEGACY_CHECKPOINT_NAME

FINAL_MODEL_PATH = f"./{CHECKPOINT_NAME}"
COVERAGE_FINAL_PATH = f"./{CHECKPOINT_NAME}_coverage.npz"

# ═══════════════════════════════════════════════════════════════════════
# FRESH_START — set True to force a brand-new model from step 0, ignoring
# ANY checkpoint on disk. This is the explicit, unambiguous way to start
# over, rather than relying on remembering to delete/move files out of the
# way. Set it back to False to resume normally; reset_num_timesteps is then
# computed automatically from whether a checkpoint was actually found, so
# that never needs hand-toggling either.
# ═══════════════════════════════════════════════════════════════════════
FRESH_START = False

# ═══════════════════════════════════════════════════════════════════════
# RESET_VALUE_HEAD — applies only on the FIRST QA run (the one seeded from
# the completion brain), and only in QA mode. See reset_value_head() below
# for the full reasoning; in short, the 6M critic is confidently wrong about
# a reward it has never seen, and its error would backpropagate through the
# shared CNN into the locomotion features the retrofit is trying to keep.
#
# Set False only if you want to observe that failure deliberately.
# ═══════════════════════════════════════════════════════════════════════
RESET_VALUE_HEAD = True

# ═══════════════════════════════════════════════════════════════════════
# WHEN TRAINING ENDS
#
# LEGACY: a step budget, unchanged. The completion phase finished at exactly
# 6,000,000, so its remaining budget is 0 and it refuses to train again.
#
# QA (Phase 4F): there is NO step target. QA training succeeds when, and only
# when, every verified testable pixel of Level 1 is covered:
#
#     covered_testable_pixels == 4,013,723      (exploration/level_completion.py)
#
# Level1CompletionCallback stops training the moment that holds and writes
# the immutable completion snapshot. Timesteps keep numbering checkpoints and
# logs and nothing else. (This replaced a fixed 12,000,000 lifetime target,
# which said nothing about whether the level had actually been explored.)
#
# QA_SAFETY_CAP_TIMESTEPS is a SAFETY CAP ONLY - an absolute global step at
# which a run is cut off whether or not the level is done (e.g. to bound a
# short validation run). Reaching it is never Level-1 completion and is
# reported as a stop with pixels still remaining. None = no cap.
# `--safety-cap-timesteps N` sets it for one launch without editing this file.
#
# THE LAUNCH GATE. A QA launch with no cap at all is an unrestricted long
# campaign, and one of those only starts with `--unrestricted`: before it,
# the controlled validation (python train_agent.py --safety-cap-timesteps
# 6020000, i.e. 6.0M -> 6.02M) has to be run and its coverage growth, reward
# balance, lifecycle behaviour, resume integrity and completion retention
# reviewed. The gate cannot check that the review happened - only a person
# can - but it makes sure an unbounded run is never started by default.
# ═══════════════════════════════════════════════════════════════════════
TOTAL_TIMESTEPS_LEGACY = 6_000_000
QA_SAFETY_CAP_TIMESTEPS = None
# SB3's learn() must be given SOME integer; with no cap this is it. It is not
# a target and no message ever presents it as one.
_SB3_OPEN_ENDED = 2 ** 62
NUM_ENVS = 8                          # Parallel environments

# ═══════════════════════════════════════════════════════════════════════
# EXACT CHECKPOINT MILESTONES — you get exactly one .zip per number below,
# named "mario_brain_checkpoint_{N}_steps.zip", no more and no less,
# regardless of how many times training is stopped and resumed in between.
# See ExactMilestoneCheckpointCallback above for why this is reliable where
# a frequency-based approach isn't.
# ═══════════════════════════════════════════════════════════════════════
LEGACY_CHECKPOINT_MILESTONES = [400_000 * i for i in range(1, 16)]  # 400k .. 6.0M
# QA: every multiple of 400k GLOBAL timesteps - 6.4M, 6.8M, 7.2M, ... - with
# no last one, since QA has no step target. Absolute lifetime counts, in
# checkpoints_qa/ under the QA name, so they never collide with the
# completion phase's 400k-6.0M files.
QA_CHECKPOINT_EVERY = 400_000
CHECKPOINT_MILESTONES = None if QA_PHASE else LEGACY_CHECKPOINT_MILESTONES

# The Level-1 completion snapshot: its own directory beside the periodic
# checkpoints, written once, read-only (exploration/level_completion.py). The
# remaining-pixel audit is a live file, rewritten late in a campaign.
LEVEL1_SNAPSHOT_DIR = os.path.join(CHECKPOINT_DIR, lc.SNAPSHOT_DIRNAME)
REMAINING_AUDIT_PATH = os.path.join(CHECKPOINT_DIR, "level1_remaining_audit.json")
REMAINING_MAP_PATH = os.path.join(CHECKPOINT_DIR, "level1_remaining_map.png")
# Append-only history of coverage growth, one JSON line per report, across
# every run of the campaign.
COVERAGE_TRAIL_PATH = os.path.join(CHECKPOINT_DIR, "coverage_audit_trail.jsonl")

# ═══════════════════════════════════════════════════════════════════════
# OPTIONAL TENSORBOARD LOGGING
# Stable-Baselines3 hard-fails ("tensorboard is not installed") the moment
# tensorboard_log is set to a path without the tensorboard package present.
# tensorboard is a large dependency that is only useful for inspecting
# training curves, so requirements.txt does NOT force everyone who just
# wants to watch the agent play to install it. Detecting it here means
# training works either way: you get the curves if you have it, and a plain
# console run if you don't, instead of a crash on startup.
#   To enable:  pip install tensorboard
#   Then view:  tensorboard --logdir ./logs/
# ═══════════════════════════════════════════════════════════════════════
try:
    import tensorboard  # noqa: F401  (imported for its presence, not its API)
    TENSORBOARD_LOG = "./logs/"
except ImportError:
    TENSORBOARD_LOG = None
    print("[INFO] tensorboard not installed - training will run without "
          "logging curves. Install it with `pip install tensorboard` if you "
          "want them.")


def _milestone_steps(filename):
    """Step count encoded in 'mario_brain_checkpoint_{N}_steps.zip', else -1.

    Returning -1 for unparseable names keeps them sorted below every real
    milestone, so a stray file can never be picked as "latest" — the old
    version fell back to lexicographic order on a parse failure, which
    silently ranks '800000' above '6000000'.
    """
    parts = filename[:-len(".zip")].split("_")
    if len(parts) >= 2 and parts[-1] == "steps":
        try:
            return int(parts[-2])
        except ValueError:
            return -1
    return -1


def checkpoint_timesteps(path):
    """The global timestep saved inside an SB3 checkpoint zip, read from its
    'data' record without loading any weights (~1 ms) - so the coverage that
    belongs to it can be verified before a single worker starts."""
    with zipfile.ZipFile(path) as z:
        return int(json.loads(z.read("data"))["num_timesteps"])


def steps_to_run(qa, num_timesteps, reset_num_timesteps, safety_cap=None):
    """What to hand SB3's learn().

    LEGACY: the remaining lifetime budget - the formula this project already
    used, unchanged (SB3 adds num_timesteps on top when resuming, so passing
    the raw total would re-aim for that many MORE steps each restart).

    QA: open-ended; Level1CompletionCallback ends it. With a safety cap, the
    steps left until the cap - which is a cut-off, never a success.
    """
    if not qa:
        return (TOTAL_TIMESTEPS_LEGACY if reset_num_timesteps
                else max(0, TOTAL_TIMESTEPS_LEGACY - num_timesteps))
    if safety_cap is None:
        return _SB3_OPEN_ENDED
    return max(0, int(safety_cap) - int(num_timesteps))


def campaign_coverage_path(latest_checkpoint, seeded_from_legacy):
    """The coverage file that belongs to the checkpoint being resumed."""
    if latest_checkpoint is None:
        raise SystemExit(
            "QA mode needs a checkpoint to resume and the coverage that belongs "
            "to it; there is none (FRESH_START?). Refusing to start from an "
            "empty map.")
    if seeded_from_legacy:
        cov_path = xconfig.BOOTSTRAP_COVERAGE
        if not os.path.exists(cov_path):
            raise SystemExit(
                f"{cov_path} not found.\n"
                f"The first QA run starts from the bootstrap map. Run "
                f"`python tools/bootstrap_coverage.py` first - starting "
                f"from an empty map would pay the agent full novelty "
                f"for re-walking everywhere it already knows.")
        return cov_path
    paired = f"{os.path.splitext(latest_checkpoint)[0]}_coverage.npz"
    if os.path.exists(paired):
        return paired
    if os.path.exists(COVERAGE_FINAL_PATH):
        return COVERAGE_FINAL_PATH
    raise SystemExit(
        f"Resuming {latest_checkpoint} but no matching "
        f"coverage file was found (looked for {paired} and "
        f"{COVERAGE_FINAL_PATH}).\n"
        f"Refusing to continue with an empty map - that would "
        f"silently re-reward everywhere already explored.")


def prepare_qa_coverage(coverage, latest_checkpoint, seeded_from_legacy):
    """Loads and VERIFIES the campaign's cumulative coverage into `coverage`
    before any worker starts. Returns (path, checkpoint timesteps).

    Anything incompatible or corrupted - wrong mask fingerprint, wrong
    denominator, a bitmap that does not re-count to its own totals, a file
    paired with a different checkpoint - stops the launch with the reason.
    It is never reset, never repaired, and nothing is written.
    """
    cov_path = campaign_coverage_path(latest_checkpoint, seeded_from_legacy)
    try:
        expected = checkpoint_timesteps(latest_checkpoint)
        coverage.load_verified(cov_path, expected_timesteps=expected)
    except (coverage_mod.CoverageFormatMismatch, coverage_mod.CoverageCorrupted,
            coverage_mod.CoverageCheckpointMismatch, FileNotFoundError,
            zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise SystemExit(
            f"[COVERAGE] Refusing to resume: {cov_path} is not a usable "
            f"campaign coverage state for {latest_checkpoint}.\n"
            f"  {type(exc).__name__}: {exc}\n"
            f"Nothing was trained and no file was written. Restore the "
            f"matching coverage file rather than starting from an empty map.") from exc
    return cov_path, expected


VALIDATION_CAP_TIMESTEPS = 6_020_000        # the controlled +20k run from the 6M seed


def parse_args(argv):
    ap = argparse.ArgumentParser(description="Train the Glitch Hunter agent.")
    ap.add_argument("--safety-cap-timesteps", type=int, metavar="N", default=None,
                    help="QA: stop at global step N whether or not Level 1 is complete "
                         f"(a cut-off, never success). The validation run is "
                         f"--safety-cap-timesteps {VALIDATION_CAP_TIMESTEPS}.")
    ap.add_argument("--unrestricted", action="store_true",
                    help="QA: explicitly approve a campaign with no safety cap. Only "
                         "after the controlled validation run has been reviewed.")
    return ap.parse_args(list(argv))


def check_launch_gate(qa, safety_cap, unrestricted):
    """No QA campaign runs unbounded unless someone said so explicitly."""
    if not qa or safety_cap is not None or unrestricted:
        return
    raise SystemExit(
        "[LAUNCH] Refusing an unrestricted QA campaign: no safety cap was given "
        "and it was not explicitly approved.\n"
        "  1. Run the controlled validation from a clean state and review it:\n"
        f"       python train_agent.py --safety-cap-timesteps {VALIDATION_CAP_TIMESTEPS}\n"
        "  2. Only after reviewing its coverage growth, reward balance, lifecycle\n"
        "     behaviour, resume integrity and completion retention:\n"
        "       python train_agent.py --unrestricted\n"
        "Nothing was trained and no file was written.")


def report_level_complete(coverage):
    """Launch-time: the campaign is already done. Says so, points at the
    proof if there is one, and trains nothing."""
    print("=" * 68)
    print(f"LEVEL 1 IS ALREADY COMPLETE: {coverage.covered_testable():,} / "
          f"{coverage.testable_total:,} testable px covered (100%).")
    found = lc.committed_snapshots(LEVEL1_SNAPSHOT_DIR)
    if found:
        path, meta = found[0]
        print(f"Completed at global step {meta['global_timestep']:,}; proof: {path}")
    else:
        print(f"(no completion snapshot found in {LEVEL1_SNAPSHOT_DIR})")
    final = lc.final_level1_brain(LEVEL1_SNAPSHOT_DIR)
    if final:
        print(f"Final Level-1 brain (VERIFIED): {final[1]['final_level1_brain']['path']}")
    else:
        print("Not yet the final Level-1 brain: run `python tools/verify_level1.py`.")
    print("Not training further, and not re-saving anything. No other level is started.")
    print("=" * 68)


def build_model(latest_checkpoint, vec_env, device):
    """Loads or constructs the PPO model.

    Lifted verbatim out of the __main__ block so the QA phase can reuse it
    unchanged - every hyperparameter note below was written from a real
    incident in this project's training history and none of it should drift
    between the two phases.
    """
    # Resume or fresh start
    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=vec_env, device=device)
        # PPO.load() restores hyperparameters from the checkpoint itself, so
        # anything set in the fresh-start branch below never reaches a
        # resumed run unless it's applied here too.
        #
        # ─── target_kl 0.03 -> 0.05 (post-5.2M tuning) ───
        # Around 4.6M-5.2M steps, "Early stopping at step 0" was firing on
        # most iterations with the triggering per-minibatch KL sitting at
        # 0.05-0.07 - well above the 1.5*target_kl=0.045 break threshold,
        # but nowhere near the 11.37 that caused the real collapse at
        # ~3.22M. With batch_size=256, each rollout (16384 samples) splits
        # into 64 tiny minibatches, and a single noisy one can trip the
        # ceiling even when the overall update is fine. 0.05 keeps a large
        # (~220x) safety margin below the value that actually caused
        # instability, while no longer treating this routine minibatch
        # noise as a violation.
        model.target_kl = 0.05
        # ─── batch_size 256 -> 512 (post-5.2M tuning) ───
        # Larger minibatches average the KL estimate over more samples,
        # directly reducing the per-minibatch noise described above -
        # addresses the actual noise source rather than just raising the
        # ceiling to tolerate it.
        model.batch_size = 512
        # Same trap for learning_rate: PPO.load() also restores an internal
        # lr_schedule closure built from the checkpoint's saved rate.
        # Setting model.learning_rate alone does NOT change what the
        # optimizer actually uses each update - lr_schedule has to be
        # rebuilt explicitly too, or this silently has zero effect.
        # Left at 1e-4 (not lowered further to 1e-5): the early-stopping
        # pattern above was present from the very first iteration after the
        # 3.2M resume, not something that emerged as the policy "got more
        # advanced" - so it's minibatch noise, not a step-size problem. A
        # 10x cut this late (~800k steps left of the 6M budget) risked
        # stalling real progress in the final stretch for a problem it
        # wasn't actually fixing.
        model.learning_rate = 1.0e-4
        model.lr_schedule = get_schedule_fn(1.0e-4)
    else:
        print("Starting fresh training...")
        model = PPO(
            "CnnPolicy", vec_env,
            learning_rate=1.0e-4,      # Lowered from 2.5e-4 for the same reason
                                       # noted in the resume branch above: too-
                                       # large per-epoch updates were hitting
                                       # target_kl on every iteration.
            n_steps=2048,             # Longer rollouts = more stable learning
            batch_size=512,           # Raised from 256 for the same reason
                                       # noted in the resume branch above:
                                       # larger minibatches average out KL
                                       # noise instead of letting a single
                                       # noisy one trip target_kl early.
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.03,            # Raised slightly from 0.02: with 2 new
                                       # actions (Left+Sprint, Left+Jump) and
                                       # reshaped rewards, the policy needs a
                                       # bit more entropy to actually sample
                                       # and discover the new momentum-based
                                       # play patterns instead of collapsing
                                       # onto the old 8-action habits.
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.05,           # Hard stop: if a rollout's mean KL
                                       # divergence blows past this, SB3 cuts
                                       # the remaining epochs for that update
                                       # instead of continuing to push the
                                       # policy further. clip_range alone
                                       # doesn't guarantee this - a bad batch
                                       # can still overwhelm the clipping
                                       # (this is what caused the collapse at
                                       # ~3.22M steps: approx_kl hit 11.37,
                                       # clip_fraction 0.834, and the policy
                                       # collapsed to near-zero entropy).
                                       # Raised from 0.03 to 0.05 after
                                       # ~4.6M-5.2M steps showed routine
                                       # minibatch noise (0.05-0.07) tripping
                                       # early stopping almost every
                                       # iteration - still a ~220x margin
                                       # below the value that actually caused
                                       # the real collapse.
            verbose=1,
            device=device,
            tensorboard_log=TENSORBOARD_LOG,
        )
    return model


def reset_value_head(model):
    """Reinitialises the critic's output layer, leaving the policy untouched.

    ═══════════════════════════════════════════════════════════════════════
    WHY THIS IS NECESSARY, AND WHY IT IS SAFE
    ═══════════════════════════════════════════════════════════════════════
    Calibration measured the problem exactly. The legacy reward's typical
    episode returned ~625 (median; the mean of ~1748 is skewed by level
    completions), and roughly 2,180 points of a successful episode came from
    terms that are monotone in max-x - precisely the terms the QA reward
    deletes. The QA reward has no equivalent, so no safe novelty weight
    brings the two scales together: closing the gap would need a weight two
    orders of magnitude above the point where a single substep outweighs an
    entire legacy episode.

    That leaves a critic which confidently predicts returns in the hundreds
    for a reward that now produces returns near zero. Two consequences:

      1. Every advantage is computed against a badly wrong baseline. SB3
         normalises advantages per minibatch, so the ABSOLUTE error largely
         washes out - but the RELATIVE ordering across states is still driven
         by a value function fitted to a different objective, so the policy
         gradient points somewhere meaningless.

      2. Far worse: the value loss is enormous, on the order of the squared
         error between ~625 and ~0. With vf_coef=0.5 that gradient flows back
         through the SHARED CNN feature extractor. max_grad_norm=0.5 bounds
         its size but not its direction, so the first updates would be
         dominated by value error tearing at exactly the convolutional
         features that encode the locomotion this retrofit is trying to
         preserve.

    Reinitialising the value head to predict ~0 removes both. What is touched:
    ONLY policy.value_net, a single Linear(features_dim -> 1). What is NOT
    touched: the CNN feature extractor, the shared MLP extractor, and
    policy.action_net - i.e. every parameter that encodes how to run, jump,
    build momentum and avoid enemies. The agent wakes up able to play exactly
    as well as it did, but with no opinion about what states are worth.

    Small weights and a zero bias rather than a full random reinit: it starts
    the critic at "everything is worth about nothing", which is much closer to
    the truth for the QA reward than anything it currently believes, and it
    lets the value warm-up fit upward from a neutral prior instead of
    unlearning a confident wrong one.
    """
    import torch

    head = getattr(model.policy, "value_net", None)
    if head is None:
        print("[VALUE HEAD] policy has no value_net attribute - skipping "
              "reset. Check the SB3 version before trusting the first "
              "iterations.")
        return False

    with torch.no_grad():
        before = float(head.weight.abs().mean())
        torch.nn.init.orthogonal_(head.weight, gain=0.01)
        if head.bias is not None:
            head.bias.zero_()
        after = float(head.weight.abs().mean())

    # The optimizer carries Adam moment estimates fitted to the old value
    # scale. Leaving them attached to freshly initialised weights would apply
    # months of accumulated momentum to parameters that no longer mean what
    # those moments were measuring.
    cleared = 0
    for group in model.policy.optimizer.param_groups:
        for param in group["params"]:
            if param is head.weight or param is head.bias:
                model.policy.optimizer.state.pop(param, None)
                cleared += 1

    print("[VALUE HEAD] Reinitialised the critic output layer "
          f"(mean |w| {before:.4f} -> {after:.4f}, bias zeroed, "
          f"{cleared} optimizer states cleared).")
    print("[VALUE HEAD] The CNN features and the action head are UNTOUCHED - "
          "locomotion is preserved; only the value estimate is reset.")
    return True


def save_pair(model, coverage):
    """Saves the model and, in QA mode, its matching coverage map.

    Order matters for the same reason it does at milestones: the model is
    written first, then the coverage. A coverage file is only ever useful
    beside the model it was recorded with, and load() enforces that pairing.
    """
    model.save(FINAL_MODEL_PATH)
    if coverage is not None:
        coverage.save(COVERAGE_FINAL_PATH, model_timesteps=model.num_timesteps)
        print(f"[COVERAGE] Saved {COVERAGE_FINAL_PATH}: "
              f"{coverage.total_unique():,} unique world px")


def main(argv=()):
    args = parse_args(argv)
    # The command line wins over the constant; either is a cut-off only.
    safety_cap = (args.safety_cap_timesteps if args.safety_cap_timesteps is not None
                  else QA_SAFETY_CAP_TIMESTEPS)
    # Look for the latest checkpoint (skipped entirely if FRESH_START).
    # The master file wins over numbered milestones when it exists.
    latest_checkpoint = None
    seeded_from_legacy = False
    if not FRESH_START:
        if os.path.exists(f"{CHECKPOINT_NAME}.zip"):
            latest_checkpoint = f"{CHECKPOINT_NAME}.zip"
        elif os.path.exists(CHECKPOINT_DIR):
            candidates = [f for f in os.listdir(CHECKPOINT_DIR)
                          if f.endswith(".zip") and f.startswith(CHECKPOINT_NAME)
                          and _milestone_steps(f) >= 0]
            if candidates:
                newest = max(candidates, key=_milestone_steps)
                latest_checkpoint = os.path.join(CHECKPOINT_DIR, newest)

        # ─── QA SEEDING ───
        # First QA run only: there is no QA checkpoint yet, so the completion
        # phase's master is READ to seed the weights. It is never written
        # back - from here on every save goes to checkpoints_qa/ and
        # glitch_hunter_qa.zip. num_timesteps rides along inside the zip, so
        # training continues from 6,000,001 rather than appearing to restart.
        if latest_checkpoint is None and QA_PHASE:
            if os.path.exists(f"{LEGACY_CHECKPOINT_NAME}.zip"):
                latest_checkpoint = f"{LEGACY_CHECKPOINT_NAME}.zip"
                seeded_from_legacy = True
            else:
                raise SystemExit(
                    f"QA phase needs a brain to continue from, but neither a "
                    f"QA checkpoint nor {LEGACY_CHECKPOINT_NAME}.zip was "
                    f"found.\n"
                    f"Restore the 6M master (backup_6M/ holds a copy) before "
                    f"starting the QA phase.")
    else:
        print("FRESH_START is True — ignoring any existing checkpoints, training from step 0.")

    # ═══════════════════════════════════════════════════════════════════
    # SHARED COVERAGE (QA phase only)
    #
    # Created BEFORE SubprocVecEnv, because the workers attach to it by name
    # as they start. The parent keeps its own handle for two reasons: it is
    # the authoritative reader for every reported metric (total_unique() is
    # recomputed from the bitmap, so it is exact even though the per-worker
    # novelty claims race), and it is the only process allowed to unlink.
    # On Windows the blocks are refcounted and vanish when the last handle
    # closes, so the parent's handle is also what keeps them alive.
    # ═══════════════════════════════════════════════════════════════════
    coverage = None
    shm_names = None
    if QA_PHASE:
        reachable = coverage_mod.load_testable()
        if reachable is None:
            raise SystemExit(
                "exploration_data/reachable_mask.npz not found.\n"
                "Run `python tools/build_reachability.py` first - without it "
                "there is no frontier and no coverage denominator.")
        coverage, shm_names = coverage_mod.create_shared(testable_mask=reachable)
        print(f"[COVERAGE] Shared grid allocated: {shm_names['visited']}")

    vec_env = None
    try:
        # ═══════════════════════════════════════════════════════════════
        # LOAD THE COVERAGE THAT BELONGS TO THIS CHECKPOINT - before any
        # worker starts (Phase 4F)
        #
        # The model and its coverage are a MATCHED PAIR. Pairing coverage with
        # the wrong brain corrupts every downstream number silently: the agent
        # gets re-paid for ground it already explored, or starved of reward
        # for ground it has not, and either way it looks like ordinary
        # training rather than a bug. load_verified also checks the mask
        # fingerprint, the denominator and the bitmap's own counts, and
        # prepare_qa_coverage turns any refusal into a clean exit.
        #
        # Loading first also means a finished campaign never spawns a worker.
        # ═══════════════════════════════════════════════════════════════
        session_start = None
        if QA_PHASE:
            cov_path, expected_timesteps = prepare_qa_coverage(
                coverage, latest_checkpoint, seeded_from_legacy)
            covered = coverage.covered_testable()
            session_start = {'global_timestep': expected_timesteps,
                             'covered_testable_px': covered}
            print(f"[COVERAGE] Loaded {cov_path} (verified: mask fingerprint "
                  f"{xconfig.TESTABLE_FINGERPRINT[:12]}..., paired at step "
                  f"{expected_timesteps:,})")
            print(f"           world raster    {xconfig.WORLD_RASTER_PX:>12,}"
                  f"   informational only")
            print(f"           testable        {coverage.testable_total:>12,}"
                  f"   method {xconfig.ADOPTED_METHOD}")
            print(f"           covered         {covered:>12,}"
                  f"   {lc.pct_text(covered, coverage.testable_total)}")
            print(f"           remaining       {coverage.remaining():>12,}")
            nb = coverage.noncoverage_breakdown()
            print(f"           noncoverage     {nb['total']:>12,}"
                  f"   NOT coverage")
            print(f"             expected      {nb['expected_total']:>12,}"
                  f"   normal engine behaviour, not glitches")
            print(f"             model gap     {nb['model_gap_total']:>12,}"
                  f"   reachability model, not the game")
            print(f"             anomalous     {nb['anomalous_total']:>12,}"
                  f"   -> glitch system")
            if lc.is_level_complete(covered, coverage.testable_total):
                report_level_complete(coverage)
                return
            # After the coverage checks (which write nothing), before any
            # worker, model or file.
            check_launch_gate(QA_PHASE, safety_cap, args.unrestricted)

        # Create 8 parallel environments
        vec_env = SubprocVecEnv([
            make_env(i, shm_names=shm_names,
                     reward_mode=xconfig.REWARD_MODE)
            for i in range(NUM_ENVS)
        ])

        # Use CUDA if it's actually available; fall back to CPU instead of
        # crashing outright. Training on CPU is dramatically slower, so this
        # prints a loud warning rather than failing silently — if you see this
        # warning, something is wrong with your CUDA/driver setup and training
        # will be much slower than expected until it's fixed.
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            print("=" * 60)
            print("WARNING: CUDA is not available. Training on CPU will be")
            print("MUCH slower. Check your PyTorch/CUDA installation if this")
            print("is unexpected.")
            print("=" * 60)

        model = build_model(latest_checkpoint, vec_env, device)
        if QA_PHASE and model.num_timesteps != session_start['global_timestep']:
            raise SystemExit(
                f"{latest_checkpoint} loaded at {model.num_timesteps:,} steps but "
                f"its zip record said {session_start['global_timestep']:,}; "
                f"refusing to pair it with coverage verified for the latter.")

        # ─── FIRST QA RUN ONLY ───
        # Done exactly once, when the QA phase is seeded from the completion
        # brain, because that is the only moment the critic is fitted to the
        # wrong objective. Resuming a QA checkpoint must NOT reset it again -
        # by then it is fitted to the reward actually in use, and throwing it
        # away would discard real progress.
        if QA_PHASE and seeded_from_legacy and RESET_VALUE_HEAD:
            reset_value_head(model)

        # Auto-save at exact milestones (see ExactMilestoneCheckpointCallback
        # above for why this replaces SB3's built-in CheckpointCallback).
        # QA: every 400k global steps, open-ended; legacy: its fixed list.
        if QA_PHASE:
            checkpoint_callback = ExactMilestoneCheckpointCallback(
                every=QA_CHECKPOINT_EVERY, save_path=CHECKPOINT_DIR,
                name_prefix=CHECKPOINT_NAME, coverage=coverage)
        else:
            checkpoint_callback = ExactMilestoneCheckpointCallback(
                targets=CHECKPOINT_MILESTONES, save_path=CHECKPOINT_DIR,
                name_prefix=CHECKPOINT_NAME, coverage=coverage)

        callbacks = [checkpoint_callback, WatchdogCallback()]
        completion = None
        if QA_PHASE:
            completion = Level1CompletionCallback(
                coverage, snapshot_dir=LEVEL1_SNAPSHOT_DIR,
                name_prefix=CHECKPOINT_NAME, session_start=session_start)
            callbacks.append(completion)
            callbacks.append(ValueWarmupCallback(
                warmup_until=model.num_timesteps + xconfig.VF_WARMUP_STEPS,
                warmup_lr=xconfig.VF_WARMUP_LR,
                normal_lr=xconfig.NORMAL_LR,
            ))
            callbacks.append(CoverageStatsCallback(
                coverage, session_start_covered=session_start['covered_testable_px'],
                audit_path=REMAINING_AUDIT_PATH, map_path=REMAINING_MAP_PATH,
                trail_path=COVERAGE_TRAIL_PATH, checkpoints=checkpoint_callback))
            callbacks.append(LifecycleStatsCallback())
            callbacks.append(StagnationCallback(coverage))
        callback_list = CallbackList(callbacks)

        # Computed automatically: True only when we're actually starting a
        # brand-new model (no checkpoint found, or FRESH_START forced it).
        # False whenever resuming an existing checkpoint, so the step counter
        # keeps climbing correctly instead of restarting at 0. Seeding the QA
        # phase from the 6M master counts as resuming, which is what makes
        # training continue at 6,000,001 rather than at 1.
        reset_num_timesteps = latest_checkpoint is None

        # See steps_to_run(): legacy passes its remaining budget (SB3 adds
        # num_timesteps on top when resuming); QA is open-ended and ends on
        # Level-1 coverage, or at the safety cap if one is set.
        n_steps = steps_to_run(QA_PHASE, model.num_timesteps, reset_num_timesteps,
                               safety_cap)

        print("=" * 68)
        print(f"PHASE            : {'QA EXPLORATION' if QA_PHASE else 'LEGACY COMPLETION'}")
        print(f"REWARD_MODE      : {xconfig.REWARD_MODE}")
        print(f"resuming from    : {latest_checkpoint or 'scratch'}"
              f"{'  (SEED - read only, never written)' if seeded_from_legacy else ''}")
        print(f"current step     : {model.num_timesteps:,}")
        if QA_PHASE:
            covered = coverage.covered_testable()
            print(f"success          : Level 1 covered = {coverage.testable_total:,} "
                  f"testable px (NOT a step count)")
            print(f"covered          : {covered:,}  "
                  f"({lc.pct_text(covered, coverage.testable_total)}), "
                  f"{coverage.remaining():,} remaining")
            print(f"safety cap       : "
                  f"{'none (approved with --unrestricted)' if safety_cap is None else f'{safety_cap:,} (a cut-off, never completion)'}")
            print(f"coverage trail   : {COVERAGE_TRAIL_PATH} (appended every 10,000 steps)")
            nxt = (model.num_timesteps // QA_CHECKPOINT_EVERY + 1) * QA_CHECKPOINT_EVERY
            print(f"checkpoints      : every {QA_CHECKPOINT_EVERY:,} global steps "
                  f"(next {nxt:,}) -> {CHECKPOINT_DIR}")
            print(f"on completion    : immutable snapshot -> {LEVEL1_SNAPSHOT_DIR}")
        else:
            print(f"lifetime target  : {TOTAL_TIMESTEPS_LEGACY:,}")
            print(f"remaining        : {n_steps:,}")
        print(f"writes model to  : {FINAL_MODEL_PATH}.zip and {CHECKPOINT_DIR}")
        if QA_PHASE:
            print(f"master preserved : {LEGACY_CHECKPOINT_NAME}.zip and "
                  f"{LEGACY_CHECKPOINT_DIR} are NOT written in this phase")
        print("=" * 68)

        try:
            if n_steps == 0:
                # Nothing to run. Return WITHOUT training and WITHOUT saving:
                # learn(0) still collects a rollout, and re-saving here would
                # overwrite the master with a model stepped past its point for
                # no reason.
                if QA_PHASE:
                    print(f"Safety cap {safety_cap:,} already reached at step "
                          f"{model.num_timesteps:,}. Level 1 is NOT complete "
                          f"({coverage.remaining():,} px remain). Raise the cap "
                          f"(--safety-cap-timesteps) to continue.")
                else:
                    print(f"Already at TOTAL_TIMESTEPS ({TOTAL_TIMESTEPS_LEGACY:,}); current step: "
                          f"{model.num_timesteps:,}. Nothing to train — raise TOTAL_TIMESTEPS to continue.")
            else:
                if QA_PHASE:
                    print("Training until Level 1 is fully covered. Ctrl+C to stop "
                          "and resume later.")
                else:
                    print(f"Training to {TOTAL_TIMESTEPS_LEGACY:,} total steps "
                          f"({n_steps:,} remaining). Ctrl+C to stop and resume later.")
                model.learn(
                    total_timesteps=n_steps,
                    callback=callback_list,
                    reset_num_timesteps=reset_num_timesteps,
                )
                save_pair(model, coverage)
                if not QA_PHASE:
                    print(f"Training complete! Model saved to {FINAL_MODEL_PATH}.zip")
                elif completion.completed:
                    print(f"Level 1 complete. Master {FINAL_MODEL_PATH}.zip holds the "
                          f"same policy as the snapshot; relaunching will not train.")
                    print("It is NOT yet the final Level-1 brain: verify it first with "
                          "`python tools/verify_level1.py`. No other level is started.")
                elif (safety_cap is not None
                      and model.num_timesteps >= safety_cap):
                    print(f"Stopped at the SAFETY CAP ({safety_cap:,}). "
                          f"Level 1 is NOT complete: {coverage.remaining():,} px remain "
                          f"({lc.pct_text(coverage.covered_testable(), coverage.testable_total)}).")
                else:
                    print(f"Training stopped at step {model.num_timesteps:,} before "
                          f"Level 1 was complete ({coverage.remaining():,} px remain). "
                          f"Saved; run again to resume.")
        except KeyboardInterrupt:
            print("\nTraining paused by user. Saving current brain state...")
            save_pair(model, coverage)
            print("Saved successfully! Run this script again to resume.")
    finally:
        if vec_env is not None:
            vec_env.close()
        if coverage is not None:
            # Workers close their own handles as they shut down; the parent is
            # the only one allowed to destroy the blocks, and only after every
            # worker is gone.
            coverage.close()
            coverage.unlink()


if __name__ == "__main__":
    main(sys.argv[1:])
