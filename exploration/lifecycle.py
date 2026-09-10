"""The episode lifecycle: EXPLORE -> COMPLETE, and how an episode ends.

Two questions this module keeps strictly apart:

  WHAT PHASE IS THE EPISODE IN?     EXPLORE, then (once, one way) COMPLETE.
                                    A phase change is a LIFECYCLE event. It
                                    never ends the episode.

  WHY DID THE EPISODE END?          level_complete / death / timeout /
                                    safety_reset / engine_done (and
                                    time_limit, which only TimeLimit above the
                                    wrapper can know about).

Before this existed, "the episode is over" was the only lifecycle signal
there was, and a timeout came back dressed as a death (death_cause
'timeout'). Nothing could say "the exploration objective is met, go finish
the level" without also saying "stop".

The reward reads the phase (Phase 4A - see GlitchHunterWrapper._qa_reward and
config PHASE-GATED REWARD), but the TRANSITION itself earns nothing: entering
COMPLETE changes which terms apply from then on, never lands a bonus. (The
frontier potential closes its telescoping sum on the transition substep - at
most FRONTIER_WEIGHT - and that closure is what makes WHERE the transition
happens irrelevant to the shaping total, rather than a reward for it.)

Everything is driven per SUBSTEP (the wrapper sits below
MaxAndSkipObservation), and every threshold in config is in AGENT steps, so
the conversion happens here and nowhere else.
"""

from __future__ import annotations

import math
from enum import Enum

from . import config


class EpisodePhase(Enum):
    EXPLORE = "explore"
    COMPLETE = "complete"


class Transition:
    """Which criterion moved the episode into COMPLETE (brief section 5.2)."""
    TARGET_MET = "T1_target_met"
    YIELD_EXHAUSTED = "T2_yield_exhausted"
    NOTHING_AHEAD = "T3_nothing_ahead"
    EXPLORE_BACKSTOP = "T4_explore_backstop"


class EndReason:
    LEVEL_COMPLETE = "level_complete"
    DEATH = "death"
    TIMEOUT = "timeout"
    SAFETY_RESET = "safety_reset"
    ENGINE_DONE = "engine_done"
    # Set by TimeLimit, which wraps outside this layer; the trainer's
    # callback reads it from SB3's "TimeLimit.truncated" info key.
    TIME_LIMIT = "time_limit"


def classify_end(info, safety_fired):
    """Why the episode ended, from the final substep's info.

    Engine reasons outrank the safety reset: if Mario died on the same
    substep the reset would have fired, he died - and the reset only fires
    on substeps the engine has not already ended (see the wrapper).
    """
    if info.get('flag_get'):
        return EndReason.LEVEL_COMPLETE
    cause = info.get('death_cause')
    if cause == 'timeout':
        return EndReason.TIMEOUT
    if cause or info.get('is_dead'):
        return EndReason.DEATH
    if safety_fired:
        return EndReason.SAFETY_RESET
    return EndReason.ENGINE_DONE


class EpisodeLifecycle:
    """Per-worker, per-episode lifecycle state. Never shared across workers.

    `coverage` is optional: without it T1, T3 and the frontier half of the
    transit test are simply unavailable (straightness still works), which is
    what lets the classifier be unit-tested on synthetic paths.
    """

    def __init__(self, coverage=None):
        self.coverage = coverage
        self.sps = config.SUBSTEPS_PER_AGENT_STEP
        self.window_substeps = config.LIFECYCLE_WINDOW * self.sps
        self.begin_episode()

    # ── episode boundaries ────────────────────────────────────────────────
    def begin_episode(self):
        self.phase = EpisodePhase.EXPLORE
        self.transition_reason = None
        self.transition_step = None
        self.completion_credit = 0.0
        self.substeps = 0
        self.last_discovery_substep = 0
        self.safety_fired = False
        self.end_reason = None

        self.consecutive_exhausted = 0
        self.consecutive_stuck = 0
        self.last_window = None          # the most recent window's verdict
        self._reset_window()

        # T1's target, frozen for the episode. coverage.episode_target() calls
        # remaining(), which is a full-grid popcount every time a new pixel
        # invalidates its cache - evaluating it on every discovering substep
        # would cost several times a whole env.step(). The target only moves
        # with episode HISTORY anyway, which changes between episodes.
        self.target = (self.coverage.episode_target()
                       if self.coverage is not None else None)

    def _reset_window(self):
        self._w_start = None
        self._w_prev = None
        self._w_path = 0.0
        self._w_min = [math.inf, math.inf]
        self._w_max = [-math.inf, -math.inf]
        self._w_new = 0
        self._w_substeps = 0

    @property
    def agent_steps(self):
        return self.substeps // self.sps

    @property
    def in_coherent_transit(self):
        """Did the most recently COMPLETED window read as transit?

        Judged on the last full window, so it lags by up to one window, and it
        is False before the first window closes: there is no evidence either
        way yet, and "no evidence" must not waive a penalty.
        """
        return bool(self.last_window and self.last_window['in_transit'])

    @property
    def is_complete(self):
        return self.phase is EpisodePhase.COMPLETE

    def drought_agent_steps(self):
        return (self.substeps - self.last_discovery_substep) // self.sps

    # ── per substep ───────────────────────────────────────────────────────
    def observe(self, info, n_new):
        """Folds one substep in. Call BEFORE the reward is computed."""
        self.substeps += 1
        self._w_substeps += 1
        if n_new >= config.SAFETY_MEANINGFUL_NEW_PX:
            self.last_discovery_substep = self.substeps
        self._w_new += int(n_new)

        rect = info.get('mario_rect')
        if rect:
            p = (rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0)
            if self._w_start is None:
                self._w_start = p
            else:
                self._w_path += math.hypot(p[0] - self._w_prev[0],
                                           p[1] - self._w_prev[1])
            self._w_prev = p
            for i in (0, 1):
                self._w_min[i] = min(self._w_min[i], p[i])
                self._w_max[i] = max(self._w_max[i], p[i])

        if self._w_substeps >= self.window_substeps:
            self._close_window(info.get('viewport_x', 0))

        if self.phase is EpisodePhase.EXPLORE:
            self._check_transition(n_new)

    def _close_window(self, viewport_x):
        """Classifies the window just finished as transit / stuck / neither."""
        v = {'new_px': self._w_new, 'straightness': 0.0,
             'frontier_gain': 0.0, 'bbox_area': 0.0, 'cells_ahead': None}
        if self._w_start is not None:
            s, e = self._w_start, self._w_prev
            net = math.hypot(e[0] - s[0], e[1] - s[1])
            v['straightness'] = net / max(self._w_path, 1e-6)
            v['bbox_area'] = ((self._w_max[0] - self._w_min[0])
                              * (self._w_max[1] - self._w_min[1]))
            if self.coverage is not None and self.coverage.testable is not None:
                # Both ends measured against the SAME, current frontier, so a
                # map that moved mid-window (another worker exploring) is not
                # mistaken for Mario moving.
                n_ahead, d = self.coverage.frontier_query(viewport_x, [s, e])
                v['cells_ahead'] = n_ahead
                if n_ahead:
                    v['frontier_gain'] = float(d[0] - d[1])

        in_transit = (v['straightness'] > config.TRANSIT_STRAIGHTNESS
                      or v['frontier_gain'] > config.TRANSIT_FRONTIER_GAIN_PX)
        is_stuck = (not in_transit and self._w_start is not None
                    and v['bbox_area'] < config.STUCK_BBOX_AREA
                    and v['new_px'] == 0)
        exhausted = v['new_px'] < config.YIELD_FLOOR and not in_transit
        v.update(in_transit=in_transit, is_stuck=is_stuck, exhausted=exhausted)

        self.consecutive_stuck = self.consecutive_stuck + 1 if is_stuck else 0
        self.consecutive_exhausted = (self.consecutive_exhausted + 1
                                      if exhausted else 0)
        self.last_window = v
        self._reset_window()

    def _check_transition(self, n_new):
        reason = None
        if (n_new and self.target is not None
                and self.coverage.episode_new >= self.target):
            reason = Transition.TARGET_MET
        elif self.consecutive_exhausted >= config.YIELD_WINDOWS:
            reason = Transition.YIELD_EXHAUSTED
        elif self.last_window is not None and self.last_window['cells_ahead'] == 0:
            reason = Transition.NOTHING_AHEAD
        elif self.agent_steps >= config.MAX_EXPLORE_STEPS:
            reason = Transition.EXPLORE_BACKSTOP
        if reason is not None:
            # One way, once per episode. Flip-flopping would make the reward
            # non-stationary within an episode as soon as reward reads phase.
            self.phase = EpisodePhase.COMPLETE
            self.transition_reason = reason
            self.transition_step = self.agent_steps
            self.completion_credit = self._credit_for(reason)

    def _credit_for(self, reason):
        """How much of the COMPLETE-phase payout this episode has earned.

        T1 and T3 mean exploration is genuinely done - the target was met, or
        nothing reachable is left ahead - so completion is worth its full
        value. T2 and T4 fire on exhaustion or elapsed time, and T2 in
        particular can be REACHED ON PURPOSE by idling off the frontier for
        three windows. Paying full completion value there would make "idle,
        then sprint for the flag" a strategy - the speedrunner rebuilt with a
        720-step wait bolted on the front. So there the credit is the fraction
        of the episode's own target actually discovered.

        Fixed at the transition and never revised, like the phase itself.
        """
        if reason in (Transition.TARGET_MET, Transition.NOTHING_AHEAD):
            return 1.0
        if self.coverage is None or not self.target:
            return 0.0
        return min(1.0, self.coverage.episode_new / self.target)

    # ── safety reset ──────────────────────────────────────────────────────
    def safety_reset_due(self):
        """All three, or nothing. Elapsed steps alone can never fire it, and
        neither can a drought while Mario is coherently in transit - transit
        windows are never stuck windows, so they break the stuck streak."""
        return (self.agent_steps > config.SAFETY_MIN_EPISODE_STEPS
                and self.drought_agent_steps() > config.SAFETY_DROUGHT_STEPS
                and self.consecutive_stuck >= config.SAFETY_STUCK_WINDOWS)

    def fire_safety_reset(self):
        self.safety_fired = True

    # ── info, every substep ───────────────────────────────────────────────
    def annotate(self, info, done):
        """Writes the lifecycle into info on EVERY substep.

        MaxAndSkipObservation keeps only the last of its four info dicts, so
        anything written on a single substep would be lost three times in
        four. Writing every substep is what makes it survive.
        """
        info['episode_phase'] = self.phase.value
        info['phase_transition_reason'] = self.transition_reason
        info['phase_transition_step'] = self.transition_step
        info['completion_credit'] = self.completion_credit
        info['lifecycle_agent_steps'] = self.agent_steps
        info['lifecycle_stuck_windows'] = self.consecutive_stuck
        info['lifecycle_in_transit'] = (self.last_window['in_transit']
                                        if self.last_window else None)
        if done:
            self.end_reason = classify_end(info, self.safety_fired)
            info['episode_end_reason'] = self.end_reason
