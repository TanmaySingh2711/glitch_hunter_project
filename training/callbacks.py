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
from exploration.lifecycle import EpisodePhase
from rewards.qa import QA_CHANNELS

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


class AnchorConsolidationCallback(BaseCallback):
    """Holds completion retention by bounding drift from a healthy policy.

    ─── WHY ───
    Official completion retention across six checkpoints is predicted almost
    exactly by the mean KL(healthy || current) on the states the healthy policy
    visits when it plays Level 1-1 (Pearson r = -0.962; see config ANCHOR
    CONSOLIDATION for the table and the fit). Every attempt to remove the CAUSE
    of the drift left the drift in place, so this bounds the drift itself.

    ─── WHAT ───
    After every PPO update (i.e. at the start of each rollout but the first)
    it measures that KL on a fixed ANCHOR set recorded by
    tools/build_anchor_set.py. Over budget, it takes gradient steps minimising
    KL(reference || policy) on anchor minibatches until the measured KL is back
    under the budget or the step cap is reached. It then records what it did.

    It is a separate phase, not a term inside PPO's loss, so the PPO update is
    untouched. It uses its OWN optimiser, so PPO's Adam moments never see its
    gradients. It trains only what the action distribution depends on - the
    shared features and the action head - and never the value head. Under
    budget it does nothing at all. The reference is frozen and never trained.
    """

    def __init__(self, reference_path: str, anchor_path: str, kl_target: float,
                 batch: int = 256, max_steps: int = 64, lr: float = 1.0e-4,
                 measure_n: int = 2048, seed: int = 0, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.reference_path = reference_path
        self.anchor_path = anchor_path
        self.kl_target = float(kl_target)
        self.batch = int(batch)
        self.max_steps = int(max_steps)
        self.lr = float(lr)
        self.measure_n = int(measure_n)
        self._rng = np.random.default_rng(seed)
        self._first = True
        self.history: list[dict[str, float]] = []

    def _on_training_start(self) -> None:
        import torch
        from stable_baselines3 import PPO as _PPO

        from common.fileio import canonical_sha256
        with np.load(self.anchor_path, allow_pickle=False) as d:
            self.anchors = d['states']
            recorded_for = str(d['reference_sha256'])
        actual = canonical_sha256(self.reference_path)
        if recorded_for != actual:
            raise RuntimeError(
                f"{self.anchor_path} was recorded for reference {recorded_for[:12]}..., "
                f"not {self.reference_path} ({actual[:12]}...). Rebuild it with "
                f"tools/build_anchor_set.py --reference {self.reference_path}.")
        policy = self.model.policy
        self.reference = _PPO.load(self.reference_path, device=policy.device).policy
        self.reference.set_training_mode(False)
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self._params = (list(policy.features_extractor.parameters())
                        + list(policy.mlp_extractor.policy_net.parameters())
                        + list(policy.action_net.parameters()))
        self._opt = torch.optim.Adam(self._params, lr=self.lr)
        log.info("[ANCHOR] Holding KL(healthy || policy) <= %.3f on %s anchor states "
                 "(reference %s).", self.kl_target, f"{len(self.anchors):,}",
                 self.reference_path)

    def _kl(self, idx: np.ndarray, grad: bool) -> Any:
        import torch
        obs, _ = self.model.policy.obs_to_tensor(self.anchors[idx])
        # A Categorical for Discrete(10); SB3 types .distribution as a union
        # that also admits MultiCategorical's list, so it is read through Any.
        with torch.no_grad():
            ref_dist: Any = self.reference.get_distribution(obs)
            ref = ref_dist.distribution.probs
        with torch.set_grad_enabled(grad):
            cur_dist: Any = self.model.policy.get_distribution(obs)
            cur = cur_dist.distribution.probs
            return (ref * (torch.log(ref + 1e-12) - torch.log(cur + 1e-12))).sum(-1).mean()

    def measure(self) -> float:
        n = min(self.measure_n, len(self.anchors))
        idx = self._rng.choice(len(self.anchors), n, replace=False)
        return float(self._kl(idx, grad=False))

    def consolidate(self) -> dict[str, float]:
        """Pull the policy back under budget. Never makes anchor KL worse.

        Minimising KL is itself an optimisation and can overshoot: measured in
        a test rig with a deliberately large step size, 200 steps took anchor
        KL from 1.42 to 22.55. So the actor is snapshotted first and restored
        if the result is worse than the start - consolidation can only help or
        do nothing, and a divergence is reported rather than saved.
        """
        import torch
        before = self.measure()
        snapshot = [p.detach().clone() for p in self._params]
        steps = 0
        after = before
        while after > self.kl_target and steps < self.max_steps:
            idx = self._rng.choice(len(self.anchors), min(self.batch, len(self.anchors)),
                                   replace=False)
            loss = self._kl(idx, grad=True)
            self._opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._params, 0.5)
            self._opt.step()
            steps += 1
            if steps % 8 == 0 or steps == self.max_steps:
                after = self.measure()
        reverted = after > before
        if reverted:
            with torch.no_grad():
                for p, saved in zip(self._params, snapshot, strict=True):
                    p.copy_(saved)
            after = self.measure()
            log.error("[ANCHOR] Consolidation diverged over %d steps; the actor was "
                      "restored (anchor KL %.4f). Lower ANCHOR_LR.", steps, after)
        rec = {'timestep': float(self.num_timesteps), 'kl_before': before,
               'kl_after': after, 'steps': float(steps), 'reverted': float(reverted)}
        self.history.append(rec)
        self.logger.record("anchor/kl_before", before)
        self.logger.record("anchor/kl_after", after)
        self.logger.record("anchor/steps", steps)
        if after > self.kl_target:
            log.warning("[ANCHOR] KL %.4f still over the %.3f budget after %d steps "
                        "(was %.4f).", after, self.kl_target, steps, before)
        return rec

    def _on_rollout_start(self) -> None:
        if self._first:                     # nothing has been trained yet
            self._first = False
            return
        self.consolidate()

    def _on_training_end(self) -> None:
        # ─── THE LAST UPDATE WOULD OTHERWISE BE SAVED UNCONSOLIDATED ───
        # Consolidation runs at the START of each rollout, i.e. after the
        # previous update - but no rollout follows the final one, and
        # train_agent saves the model as soon as learn() returns. Without
        # this the saved checkpoint carries one update that was never checked
        # against the budget. (Milestone saves are safe: they happen during a
        # rollout, after that rollout's consolidation.) Measured on the first
        # run: the final update took anchor KL 0.079 -> 0.116, inside the
        # 0.13 budget by luck rather than by design.
        if not self._first:
            self.consolidate()

    def _on_step(self) -> bool:
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

    ─── A LOWER LEARNING RATE DID NOT DO WHAT THE PARAGRAPH ABOVE CLAIMS ───
    Lowering the rate slows the critic and the actor equally, so the actor
    still walks on the critic's wrong estimates - just more slowly. Measured
    on the 6,032,768-step seed, in two independent 10-update runs, the policy
    began drifting immediately: episode length 334 -> 1,010 and episode
    return 14.9 -> 6.5 within 8 updates, with the first minibatch of the
    first update already at approx_kl 0.08. And once NORMAL_LR was set equal
    to VF_WARMUP_LR the warm-up became a literal no-op.

    So `freeze_actor` does what the docstring always intended: the shared
    features and the action head are frozen (requires_grad False), leaving
    only the critic's value head to fit. The policy is then EXACTLY stationary
    while the critic re-fits - the weights cannot move, so neither can the
    action distribution - and the drift the actor accumulated on wrong
    advantages cannot happen at all. It is released when the critic's
    explained variance reaches `ev_release` on two consecutive updates, or
    when `warmup_until` is reached, whichever comes first.
    """

    def __init__(self, warmup_until: int, warmup_lr: float, normal_lr: float,
                 freeze_actor: bool = False, ev_release: float | None = None,
                 verbose: int = 0) -> None:
        super().__init__(verbose)
        self.warmup_until = warmup_until
        self.warmup_lr = warmup_lr
        self.normal_lr = normal_lr
        self.freeze_actor = freeze_actor
        self.ev_release = ev_release
        self._restored = False
        self._frozen = False
        self._ev_hits = 0

    def _apply(self, lr: float) -> None:
        self.model.learning_rate = lr
        self.model.lr_schedule = FloatSchedule(lr)

    def _actor_params(self) -> list[Any]:
        """Everything the ACTION distribution depends on. With shared features
        that includes the feature extractor: training the critic through it
        would move the logits, so it is frozen along with the action head."""
        policy = self.model.policy
        params = list(policy.features_extractor.parameters())
        params += list(policy.action_net.parameters())
        params += list(policy.mlp_extractor.policy_net.parameters())
        return params

    def _freeze(self) -> None:
        for p in self._actor_params():
            p.requires_grad_(False)
        self._frozen = True
        trainable = sum(p.numel() for p in self.model.policy.parameters() if p.requires_grad)
        log.info("[WARMUP] Actor FROZEN: only the critic head trains (%s parameters). "
                 "The policy cannot move until release.", f"{trainable:,}")

    def _release(self, why: str) -> None:
        for p in self.model.policy.parameters():
            p.requires_grad_(True)
        self._frozen = False
        log.info("[WARMUP] Actor RELEASED at %s steps: %s.", f"{self.num_timesteps:,}", why)

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
            if self.freeze_actor:
                self._freeze()

    def _on_rollout_start(self) -> None:
        # The previous update's statistics are in the logger by now: the order
        # is collect -> train -> next rollout, so this is the first moment the
        # critic's explained variance for the update just finished is readable.
        if not self._frozen:
            return
        ev = self.model.logger.name_to_value.get("train/explained_variance")
        if ev is None:
            return
        self._ev_hits = self._ev_hits + 1 if (
            self.ev_release is not None and float(ev) >= self.ev_release) else 0
        if self._ev_hits >= 2:
            self._release(f"explained variance {float(ev):.3f} >= {self.ev_release} twice running")

    def _on_step(self) -> bool:
        if not self._restored and self.num_timesteps >= self.warmup_until:
            self._restored = True
            self._apply(self.normal_lr)
            if self._frozen:
                self._release("warm-up step budget spent")
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
        self._table_rows_printed = 0

    # A plain, column-aligned table on stdout, separate from the [COVERAGE]
    # log line above: that line is the persisted record (console + file,
    # tested), this is purely a live-readable view of the same numbers,
    # appended (not overwritten) so the run's whole coverage history stays
    # scrollable rather than collapsing to one gauge line.
    _TABLE_COLS = (("Step", 11), ("Covered", 12), ("Testable", 12),
                   ("Coverage%", 10), ("Remaining", 12), ("New(sess)", 11),
                   ("New/10k", 9))
    _TABLE_HEADER_EVERY = 10

    def _print_table_row(self, step: int, covered: int, testable_total: int,
                          pct_text: str, remaining: int | None, session_new: int,
                          rate: float) -> None:
        lines = []
        if self._table_rows_printed % self._TABLE_HEADER_EVERY == 0:
            header = "  ".join(name.rjust(w) for name, w in self._TABLE_COLS)
            lines += [header, "-" * len(header)]
        values = (f"{step:,}", f"{covered:,}", f"{testable_total:,}", pct_text,
                  f"{remaining:,}" if remaining is not None else "-",
                  f"{session_new:,}", f"{rate:,.0f}")
        lines.append("  ".join(v.rjust(w) for v, (_, w)
                                in zip(values, self._TABLE_COLS, strict=True)))
        # log.info, not print: T201 bans raw print in this codebase, and
        # routing through logging means the console handler correctly ends
        # any write_progress() line first instead of colliding with it.
        # _ConsoleFormatter shows INFO bare, so the table reaches the
        # terminal exactly as built; the file gets one extra timestamped
        # entry per row, which is a cheap, permanent copy of the same table.
        log.info("\n".join(lines))
        self._table_rows_printed += 1

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
        self._print_table_row(self.num_timesteps, covered, testable_total or 0,
                               shown, remaining, session_new, rate)

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
    coverage line. One number here is a diagnostic alarm rather than a stat:

      * safety_reset should be RARE. It is the last-resort ending, after
        level completion and death, and it is reported with the evidence it
        fired on (stuck / unproductive_loop).

    Transitions are counted per criterion (T1/T2/T3, and "none" for an
    episode that stayed in EXPLORE throughout - which is now a legitimate
    outcome at any episode length, since the time-only T4 backstop is
    retired).

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


class RewardTelemetryCallback(BaseCallback):
    """The QA reward's own books, appended to a JSONL file as the run happens.

    ─── WHY THIS EXISTS ───
    rewards/qa.py already accounts for every term it pays as a separate
    CHANNEL (QA_CHANNELS), accumulated per episode AND per phase, and leaves
    the finished books in the final info of each episode. Until now nothing
    read them during training: the only consumer was
    tools/calibrate_phase_reward.py, offline, on replayed trajectories. So
    "is the reward balanced?" could be asked of a calibration run but not of
    the run that actually trained the policy. This writes them down as they
    happen, which is what the controlled validation needs.

    ─── IT CANNOT CHANGE WHAT PPO LEARNS ───
    It reads the info dicts SB3 already hands every callback, copies what it
    needs and appends to a file. It never touches the wrapper, the reward,
    the observation or the model, and it always returns True, so the rewards
    PPO sees are bit-identical whether or not it is attached
    (tests/test_reward_telemetry.py pins both the rewards and the resulting
    weights). QA only: an info without 'qa_channels' - every legacy episode -
    is ignored.

    ─── WHAT IS WRITTEN, one JSON object per line ───
      session   once per run, at training start: where the run resumed from,
                and the constants these numbers should be read against.
      episode   one per finished episode, from its final info: every channel
                summed per phase, the RECONCILIATION below, how the episode
                ended, and the lifecycle context.
      interval  every `every` timesteps: the same channels summed over the
                episodes that closed in that interval, so a long campaign can
                be read without walking every episode line.

    ─── RECONCILIATION: the record accounts for the reward PPO was given ───
    rewards/qa.py pays every term into a named channel and nothing outside
    them, so per substep the channels sum EXACTLY to the reward the wrapper
    returns; MaxAndSkipObservation then sums four substeps and Monitor sums
    the episode. Each record therefore carries the chain end to end:

      env_reward            the engine's own reward for the episode. It is
                            the 'death' channel: custom_mario_env pays 0.0
                            every substep except -5.0 on a death, which is
                            what that channel is named for.
      preclip_total         every channel except 'clip' - the reward before
                            the backstop clamp.
      clip_adjustment       the 'clip' channel: what QA_REWARD_CLIP removed.
                            Zero unless a substep exceeded the clamp.
      final_reward          preclip_total + clip_adjustment = the sum of ALL
                            channels = what the wrapper actually returned.
      monitor_reward        SB3 Monitor's own total for the same episode,
                            measured independently of the books.
      unaccounted_residual  monitor_reward - final_reward. Float-summation
                            noise only (measured ~3e-7 on episode totals in
                            the hundreds, from summing the same values in a
                            different order). Anything larger means a reward
                            term is reaching PPO without being booked, so it
                            is logged as a warning rather than left in a file
                            nobody reads.

    One line per EPISODE, not per substep: an episode is hundreds to
    thousands of agent steps, so this is a few dozen lines for the +20k
    validation and one small append per episode thereafter.

    Append-only, never rewritten, so a resumed run continues the same file
    rather than truncating it; every record carries its global_timestep and
    the session's own id, so two runs in one file stay apart.
    """

    KINDS = ('session', 'episode', 'interval')
    # Above this, the residual is no longer float noise and someone must look.
    RESIDUAL_TOLERANCE = 1e-3

    def __init__(self, path: str, every: int = 10_000,
                 session_start: dict[str, int] | None = None,
                 verbose: int = 0) -> None:
        super().__init__(verbose)
        self.path = path
        self.every = every
        self.session_start = dict(session_start or {})
        self.session_id = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        self._next = 0
        self._episodes = 0
        self._interval_episodes = 0
        self._acc: dict[str, dict[str, float]] = {}
        self._ends: dict[str, int] = {}
        self._transitions: dict[str, int] = {}
        self._interval_clips = 0
        self._interval_agent_steps = 0
        self._interval_started_at = 0
        self._interval_rewards: list[float] = []
        self._interval_residual = 0.0
        self._interval_max_residual = 0.0
        self.worst_residual = 0.0        # the run's worst, for the final say
        # qa_clip_events is CUMULATIVE INSIDE EACH WORKER, and workers are
        # separate processes, so the only way to count them once is per-env
        # deltas. A value that went DOWN means that worker restarted, so the
        # new value is itself the delta.
        self._clip_seen: dict[int, int] = {}

    # ── writing ───────────────────────────────────────────────────────────
    def _append(self, record: dict[str, Any]) -> None:
        """One JSON line. A telemetry failure must never stop training."""
        try:
            os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
            with open(self.path, 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(record) + '\n')
        except Exception as exc:
            log.warning("[REWARD] could not append telemetry to %s: %s", self.path, exc)

    def _stamp(self, kind: str) -> dict[str, Any]:
        return {'kind': kind, 'session_id': self.session_id,
                'global_timestep': int(self.num_timesteps),
                'written_at': datetime.datetime.now().isoformat(timespec='seconds')}

    def _on_training_start(self) -> None:
        self._next = self.num_timesteps + self.every
        self._interval_started_at = int(self.num_timesteps)
        record = self._stamp('session')
        record.update({
            'reward_mode': xconfig.REWARD_MODE,
            'resumed_at': self.session_start.get('global_timestep'),
            'channels': list(QA_CHANNELS),
            'phases': [p.value for p in EpisodePhase],
            # The constants the channel sums below are produced by, so a
            # record can be read years later without guessing which tuning
            # was live. Reward values themselves are never set here.
            'constants': {
                'novelty_weight': xconfig.NOVELTY_WEIGHT,
                'novelty_cap': xconfig.NOVELTY_CAP,
                'n_ref': xconfig.N_REF,
                'frontier_weight': xconfig.FRONTIER_WEIGHT,
                'drought_max': xconfig.DROUGHT_MAX,
                'drought_episode_cap': xconfig.DROUGHT_EPISODE_CAP,
                'explore_time_penalty': xconfig.EXPLORE_TIME_PENALTY,
                'complete_time_penalty': xconfig.COMPLETE_TIME_PENALTY,
                'complete_progress_per_px': xconfig.COMPLETE_PROGRESS_PER_PX,
                'complete_novelty_mult': xconfig.COMPLETE_NOVELTY_MULT,
                'explore_flag_reward': xconfig.EXPLORE_FLAG_REWARD,
                'complete_flag_reward': xconfig.COMPLETE_FLAG_REWARD,
                'interaction_episode_cap': xconfig.QA_INTERACTION_EPISODE_CAP,
                'locomotion_episode_cap': xconfig.QA_LOCOMOTION_EPISODE_CAP,
                'shortfall_penalty': xconfig.SHORTFALL_PENALTY,
                'reward_clip': xconfig.QA_REWARD_CLIP,
            },
        })
        self._append(record)

    # ── reconciliation ────────────────────────────────────────────────────
    def _reconcile(self, totals: dict[str, float],
                   monitor: dict[str, Any]) -> dict[str, Any]:
        """The reward chain end to end, from the engine to what PPO was given.

        Nothing here recomputes a reward: every number is read out of the
        books rewards/qa.py already kept, and `monitor_reward` is SB3's
        independent total for the same episode. Their difference is the
        whole point - see RECONCILIATION in the class docstring.
        """
        clip_adjustment = totals['clip']
        final_reward = sum(totals.values())
        monitor_reward = float(monitor['r']) if 'r' in monitor else None
        residual = (None if monitor_reward is None
                    else monitor_reward - final_reward)
        return {
            'env_reward': round(totals['death'], 6),
            'preclip_total': round(final_reward - clip_adjustment, 6),
            'clip_adjustment': round(clip_adjustment, 6),
            'final_reward': round(final_reward, 6),
            'monitor_reward': None if monitor_reward is None else round(monitor_reward, 6),
            'unaccounted_residual': None if residual is None else residual,
        }

    # ── per episode ───────────────────────────────────────────────────────
    def _episode_record(self, info: dict[str, Any], channels: dict[str, Any],
                        clips: int) -> dict[str, Any]:
        by_phase = {phase: {ch: float(vals.get(ch, 0.0)) for ch in QA_CHANNELS}
                    for phase, vals in channels.items()}
        totals = {ch: sum(p[ch] for p in by_phase.values()) for ch in QA_CHANNELS}
        monitor = info.get('episode') or {}
        reconciliation = self._reconcile(totals, monitor)
        record = self._stamp('episode')
        record.update({
            'episode_index': self._episodes,
            'agent_steps': info.get('lifecycle_agent_steps'),
            'end_reason': ('time_limit' if info.get('TimeLimit.truncated')
                           else info.get('episode_end_reason')),
            'safety_reset_reason': info.get('safety_reset_reason'),
            'phase_at_end': info.get('episode_phase'),
            'transition_reason': info.get('phase_transition_reason'),
            'transition_step': info.get('phase_transition_step'),
            'completion_credit': info.get('completion_credit'),
            'flag_get': bool(info.get('flag_get')),
            'clock_extensions': info.get('clock_extensions'),
            'rehearsal': info.get('qa_rehearsal'),
            'max_x': info.get('qa_max_x'),
            'max_x_at_transition': info.get('qa_max_x_at_transition'),
            'channels_by_phase': by_phase,
            'channels_total': {k: round(v, 6) for k, v in totals.items()},
            # The engine reward, the clamp and the two independent totals,
            # end to end - see RECONCILIATION in the class docstring.
            'reconciliation': reconciliation,
            'clip_events': clips,
            'novelty_shape': info.get('qa_novelty_shape'),
            'coverage_episode_new_px': info.get('coverage_episode_new'),
            'coverage_episode_target_px': info.get('coverage_episode_target'),
            'coverage_total_px': info.get('coverage_total'),
        })
        return record

    def _accumulate(self, record: dict[str, Any]) -> None:
        for phase, vals in record['channels_by_phase'].items():
            into = self._acc.setdefault(phase, dict.fromkeys(QA_CHANNELS, 0.0))
            for ch, v in vals.items():
                into[ch] += v
        end = record['end_reason'] or 'unknown'
        self._ends[end] = self._ends.get(end, 0) + 1
        t = record['transition_reason'] or 'none'
        self._transitions[t] = self._transitions.get(t, 0) + 1
        self._interval_episodes += 1
        self._interval_clips += record['clip_events']
        self._interval_agent_steps += int(record['agent_steps'] or 0)
        self._interval_rewards.append(record['reconciliation']['final_reward'])
        residual = record['reconciliation']['unaccounted_residual']
        if residual is None:
            return
        self._interval_residual += residual
        self._interval_max_residual = max(self._interval_max_residual, abs(residual))
        self.worst_residual = max(self.worst_residual, abs(residual))
        if abs(residual) > self.RESIDUAL_TOLERANCE:
            # Past float noise: a reward term is reaching PPO without being
            # booked, which is exactly what this file exists to catch.
            log.warning("[REWARD] episode %d: %.6f of reward is UNACCOUNTED "
                        "(Monitor %.6f vs channels %.6f). A term is missing "
                        "from the books.", record['episode_index'], residual,
                        record['reconciliation']['monitor_reward'],
                        record['reconciliation']['final_reward'])

    def _flush_interval(self) -> None:
        if not self._interval_episodes:
            return
        steps = self._interval_agent_steps
        record = self._stamp('interval')
        record.update({
            'from_global_timestep': self._interval_started_at,
            'episodes': self._interval_episodes,
            'agent_steps': steps,
            'channels_by_phase': {p: {k: round(v, 6) for k, v in c.items()}
                                  for p, c in self._acc.items()},
            'channels_mean_per_episode': {
                p: {k: round(v / self._interval_episodes, 6) for k, v in c.items()}
                for p, c in self._acc.items()},
            'clip_events': self._interval_clips,
            'clip_events_per_1k_agent_steps': (round(1000.0 * self._interval_clips / steps, 4)
                                               if steps else None),
            'episode_reward_mean': round(float(np.mean(self._interval_rewards)), 6),
            'episode_reward_median': round(float(np.median(self._interval_rewards)), 6),
            'unaccounted_residual_sum': self._interval_residual,
            'unaccounted_residual_max_abs': self._interval_max_residual,
            'ends': dict(self._ends),
            'transitions': dict(self._transitions),
        })
        self._append(record)
        self._acc, self._ends, self._transitions = {}, {}, {}
        self._interval_episodes = self._interval_clips = self._interval_agent_steps = 0
        self._interval_rewards = []
        self._interval_residual = self._interval_max_residual = 0.0
        self._interval_started_at = int(self.num_timesteps)

    def _on_step(self) -> bool:
        for i, (done, info) in enumerate(zip(self.locals.get("dones", []),
                                             self.locals.get("infos", []), strict=True)):
            channels = info.get('qa_channels') if done else None
            if not channels:                  # not done, or a legacy episode
                continue
            clips = self._clip_delta(i, int(info.get('qa_clip_events') or 0))
            record = self._episode_record(info, channels, clips)
            self._episodes += 1
            self._append(record)
            self._accumulate(record)
        if self.num_timesteps >= self._next:
            self._next = self.num_timesteps + self.every
            self._flush_interval()
        return True

    def _clip_delta(self, idx: int, seen: int) -> int:
        last = self._clip_seen.get(idx)
        self._clip_seen[idx] = seen
        if last is None or seen < last:       # first sight, or a restarted worker
            return seen
        return seen - last

    def _on_training_end(self) -> None:
        """Whatever the last interval had, so nothing is lost to a stop."""
        self._flush_interval()


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
        self.detections = 0

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

        if not xconfig.STAGNATION_ESCALATE:
            self.detections += 1
            log.warning("[STAGNATION] %s new px per 10k for %d windows with %s px still "
                        "unexplored. Detected only - the objective and optimiser are "
                        "left as validated (config.STAGNATION_ESCALATE is False).",
                        f"{rate:,.0f}", xconfig.STAGNATION_WINDOW,
                        f"{remaining:,}" if remaining is not None else "?")
            return True

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
