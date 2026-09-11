"""GlitchHunterWrapper: coverage recording, the episode lifecycle, and reward.

The wrapper sits directly on CustomMarioEnv, BELOW MaxAndSkipObservation(4),
so it sees every engine substep. Per substep it:

  1. keeps the engine clock alive in QA mode (the respawn rule - time alone
     never ends a QA episode; see config QA_TIMEOUT_ENDS_EPISODE),
  2. records the collider's swept footprint into the persistent coverage
     bitmap - in BOTH modes, whenever coverage is attached,
  3. folds the substep into the EXPLORE -> COMPLETE lifecycle (QA only),
  4. computes the reward for the selected objective:

        "qa_exploration"      rewards/qa.py      the current objective
        "legacy_completion"   rewards/legacy.py  the 6M brain's, preserved exactly

Coverage is RECORDED whenever it is attached; only the REWARD branches on
the mode. That separation is what lets tools/bootstrap_coverage.py replay the
6M brain under its own native legacy reward - so the routes recorded are the
ones it actually learned - while still recording where it went.

The dashboard runtime that used to live at the bottom of this file is now
dashboard_backend.py.
"""
from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import numpy as np

from custom_mario_env import CustomMarioEnv
from exploration import config
from exploration import coverage as coverage_mod
from exploration import lifecycle as lifecycle_mod
from rewards.legacy import LegacyCompletionReward
from rewards.qa import QA_CHANNELS, QAExplorationReward
from rewards.shared import Info

__all__ = ['ACTION_NAMES', 'QA_CHANNELS', 'REWARD_MODES', 'GlitchHunterWrapper']

# Keep in sync with the action space in custom_mario_env.py and the info modal
# in templates/index.html (tests/test_env.py checks the first two agree).
ACTION_NAMES = {
    0: "Stand Still",
    1: "Walk Right",
    2: "Walk Right + Jump",
    3: "Run Right",
    4: "Run Right + Jump",
    5: "Jump",
    6: "Walk Left",
    7: "Crouch",
    8: "Run Left (momentum retreat)",
    9: "Left + Jump",
}

REWARD_MODES = ("qa_exploration", "legacy_completion")


class GlitchHunterWrapper(QAExplorationReward, LegacyCompletionReward,
                          gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """Reward shaping for the Mario PPO agent, in two selectable modes.

    See the module docstring for what happens on each substep, rewards/qa.py
    for the QA objective and rewards/legacy.py for the legacy one.
    """

    def __init__(self, env: gym.Env[np.ndarray, int], reward_mode: str | None = None,
                 coverage: coverage_mod.SpatialCoverage | None = None,
                 shm_names: dict[str, str] | None = None,
                 testable_mask: np.ndarray | None = None,
                 attach_coverage: bool | None = None) -> None:
        super().__init__(env)
        self.reward_mode = reward_mode or config.REWARD_MODE
        if self.reward_mode not in REWARD_MODES:
            raise ValueError(
                f"unknown REWARD_MODE {self.reward_mode!r}; expected "
                f"'qa_exploration' or 'legacy_completion'")
        qa = self.reward_mode == "qa_exploration"

        # ─── COVERAGE ATTACHMENT ───
        if attach_coverage is None:
            attach_coverage = coverage is not None or shm_names is not None or qa
        self.coverage = coverage
        if self.coverage is None and attach_coverage:
            mask = testable_mask if testable_mask is not None else coverage_mod.load_testable()
            if shm_names:
                self.coverage = coverage_mod.open_shared(shm_names, testable_mask=mask)
            else:
                self.coverage = coverage_mod.SpatialCoverage(testable_mask=mask)

        if qa:
            if self.coverage is None:
                raise ValueError(
                    "qa_exploration mode requires a coverage channel; pass "
                    "coverage= or shm_names=, or leave attach_coverage as None")
            # Fails at construction rather than 200k steps into a run.
            config.assert_reward_balance()
            config.assert_phase_reward_balance()

        # ─── EPISODE LIFECYCLE ───
        # The engine timer is the one authoritative episode clock; QA mode
        # raises its budget and ends at the castle door (measurements in
        # exploration/config.py, EPISODE LENGTH). Set in BOTH modes, not only
        # QA: an env is configured by whichever wrapper wraps it, so a legacy
        # wrapper can never inherit a QA budget left behind on a shared env.
        base = self._base_env()
        base.episode_time_units = config.QA_EPISODE_TIME_UNITS if qa else None
        base.end_on_level_complete = config.QA_END_ON_LEVEL_COMPLETE if qa else False
        # EXPLORE -> COMPLETE and the safety reset are part of the QA
        # objective only. Legacy keeps its own stuck termination, untouched.
        self.lifecycle = lifecycle_mod.EpisodeLifecycle(self.coverage) if qa else None
        # The respawn rule: in QA, time alone never ends an episode, so the
        # engine clock is kept above zero (CustomMarioEnv.hold_clock). Legacy
        # keeps the timeout the 6M brain was trained with.
        self.holds_engine_clock = qa and not config.QA_TIMEOUT_ENDS_EPISODE

        # Cumulative across episodes: the clamp counter and the death memory.
        self.qa_clip_events = 0
        self._init_death_memory()
        self._reset_episode_state()

    def _base_env(self) -> CustomMarioEnv:
        """The engine at the bottom of the chain. Duck-typed on purpose: tests
        stand a lightweight fake in for it."""
        return cast(CustomMarioEnv, self.env.unwrapped)

    def _reset_episode_state(self) -> None:
        """Everything that starts over with each episode, in both modes."""
        self._reset_shared_state()
        self._reset_legacy_state()
        self._reset_qa_state()
        self.ep_reward_total = 0.0
        self.ep_substeps = 0
        self.ep_clock_extensions = 0
        self.last_n_new = 0

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self._reset_episode_state()

        # Coverage itself is NEVER cleared here - that is the entire point of
        # the retrofit. begin_episode() only rolls the per-episode counters
        # and drops prev_rect, so the spawn teleport is not swept as a false
        # corridor across the level.
        if self.coverage is not None:
            self.coverage.begin_episode()
        # After coverage.begin_episode(), which is what rolls the history the
        # adaptive target is computed from.
        if self.lifecycle is not None:
            self.lifecycle.begin_episode()
        self._decay_danger_zones()

        return self.env.reset(seed=seed, options=options)

    # ══════════════════════════════════════════════════════════════════════
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.holds_engine_clock:
            hold = getattr(self.env.unwrapped, 'hold_clock', None)
            if hold is not None and hold():
                self.ep_clock_extensions += 1
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        if self.holds_engine_clock:
            # How many times this episode has outlived a full QA clock.
            info['clock_extensions'] = self.ep_clock_extensions

        n_new = self._record_coverage(info)
        # Lifecycle sees the substep BEFORE the reward, so the safety-reset
        # decision inside _qa_reward is made on up-to-date state - and so the
        # transition substep is already scored in the phase it moved into.
        if self.lifecycle is not None:
            self.lifecycle.observe(info, n_new)

        reward = float(env_reward)
        if self.reward_mode == "qa_exploration":
            reward, done = self._qa_reward(reward, done, info, n_new)
        else:
            reward, done = self._legacy_reward(reward, done, info)

        if self.lifecycle is not None:
            self.lifecycle.annotate(info, done)

        self.ep_reward_total += float(reward)
        self.ep_substeps += 1
        return obs, float(reward), done, False, info

    # ── coverage recording (both modes) ───────────────────────────────────
    def _record_coverage(self, info: Info) -> int:
        """Marks the swept collider region and returns how many px were new.

        Uses info['mario_rect'], the live collider in WORLD coordinates at
        its real per-form size. A None rect means the env hit its
        AttributeError fallback and genuinely does not know where Mario is;
        recording a guess there would fabricate coverage, so it records
        nothing.
        """
        self.last_n_new = 0
        if self.coverage is None:
            return 0
        rect = info.get('mario_rect')
        if not rect:
            return 0
        n_new = self.coverage.record({
            'cur_rect': tuple(rect),
            'viewport_x': info.get('viewport_x', 0),
            'x_vel': info.get('x_vel', 0.0),
            'on_ground': info.get('on_ground', True),
        })
        self.last_n_new = n_new
        info['coverage_new_px'] = n_new
        info['coverage_episode_new'] = self.coverage.episode_new
        info['coverage_drought'] = self.coverage.steps_since_new_pixel
        return n_new
