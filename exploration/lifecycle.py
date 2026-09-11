"""The episode lifecycle: EXPLORE -> COMPLETE, and how an episode ends.

Two questions this module keeps strictly apart:

  WHAT PHASE IS THE EPISODE IN?     EXPLORE, then (once, one way) COMPLETE.
                                    A phase change is a LIFECYCLE event. It
                                    never ends the episode.

  WHY DID THE EPISODE END?          level_complete / death / timeout /
                                    safety_reset / engine_done (and
                                    time_limit, which only TimeLimit above the
                                    wrapper can know about). In QA mode time
                                    alone never ends an episode, so timeout and
                                    time_limit are legacy-only there.

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
from collections import deque
from enum import Enum
from itertools import pairwise
from typing import TYPE_CHECKING, Any, TypedDict

from . import config

if TYPE_CHECKING:
    from .coverage import SpatialCoverage

Point = tuple[float, float]
Info = dict[str, Any]


class WindowVerdict(TypedDict):
    """What one closed LIFECYCLE_WINDOW of play amounted to (_close_window)."""
    new_px: int                     # testable pixels discovered in the window
    straightness: float             # net displacement / path length
    frontier_gain: float            # how much closer to the frontier it ended
    bbox_area: float                # area of the box Mario's centre stayed in
    cells_ahead: int | None         # unexplored frontier cells ahead (None: no mask)
    start: Point | None             # Mario's centre at the first / last substep;
    end: Point | None               # None when Mario was never observed
    path: float                     # distance walked by that centre
    centre: Point | None            # middle of the box
    in_transit: bool
    is_stuck: bool
    exhausted: bool
    unproductive: bool


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


class SafetyReason:
    """Which stagnation evidence a safety reset fired on (config SAFETY RESET).
    There is deliberately no reason made of elapsed time or drought alone."""
    STUCK = "stuck"
    LOOP = "unproductive_loop"


def _seen(point: Point | None) -> Point:
    """A window's position, which every window in a stagnation span has: an
    unproductive window is by definition one in which Mario was observed."""
    if point is None:
        raise ValueError("a window with no observed position inside a stagnation span")
    return point


def classify_end(info: Info, safety_fired: bool) -> str:
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

    def __init__(self, coverage: SpatialCoverage | None = None) -> None:
        self.coverage = coverage
        self.sps = config.SUBSTEPS_PER_AGENT_STEP
        self.window_substeps = config.LIFECYCLE_WINDOW * self.sps
        # False pins the phase: the natural T1-T4 checks never run, and only
        # force_complete() moves it. For calibration arms and tests that need
        # an episode held in one phase; training never turns it off.
        self.auto_transition = True
        self.begin_episode()

    # ── episode boundaries ────────────────────────────────────────────────
    def begin_episode(self) -> None:
        self.phase = EpisodePhase.EXPLORE
        self.transition_reason: str | None = None
        self.transition_step: int | None = None
        self.completion_credit = 0.0
        self.substeps = 0
        self.last_discovery_substep = 0
        self.safety_fired = False
        self.safety_reason: str | None = None
        self.end_reason: str | None = None

        self.consecutive_exhausted = 0
        self.consecutive_stuck = 0
        self.consecutive_unproductive = 0
        self.last_window: WindowVerdict | None = None  # the most recent window's verdict
        # Recent verdicts, for judging a streak of windows as ONE span (the
        # LOOP arm). Bounded: only the last SAFETY_STUCK_WINDOWS are read.
        self.recent_windows: deque[WindowVerdict] = deque(maxlen=64)
        self._reset_window()

        # T1's target, frozen for the episode. coverage.episode_target() calls
        # remaining(), which is a full-grid popcount every time a new pixel
        # invalidates its cache - evaluating it on every discovering substep
        # would cost several times a whole env.step(). The target only moves
        # with episode HISTORY anyway, which changes between episodes.
        self.target: int | None = (self.coverage.episode_target()
                                   if self.coverage is not None else None)
        self.target_informed = (self.coverage is not None
                                and self.coverage.target_informed())

    def _reset_window(self) -> None:
        self._w_start: Point | None = None
        self._w_prev: Point | None = None
        self._w_path = 0.0
        self._w_min = [math.inf, math.inf]
        self._w_max = [-math.inf, -math.inf]
        self._w_new = 0
        self._w_substeps = 0

    @property
    def agent_steps(self) -> int:
        return self.substeps // self.sps

    @property
    def in_coherent_transit(self) -> bool:
        """Did the most recently COMPLETED window read as transit?

        Judged on the last full window, so it lags by up to one window, and it
        is False before the first window closes: there is no evidence either
        way yet, and "no evidence" must not waive a penalty.
        """
        return bool(self.last_window and self.last_window['in_transit'])

    @property
    def is_complete(self) -> bool:
        return self.phase is EpisodePhase.COMPLETE

    def drought_agent_steps(self) -> int:
        return (self.substeps - self.last_discovery_substep) // self.sps

    # ── per substep ───────────────────────────────────────────────────────
    def observe(self, info: Info, n_new: int) -> None:
        """Folds one substep in. Call BEFORE the reward is computed."""
        self.substeps += 1
        self._w_substeps += 1
        if n_new >= config.SAFETY_MEANINGFUL_NEW_PX:
            self.last_discovery_substep = self.substeps
        self._w_new += int(n_new)

        rect = info.get('mario_rect')
        if rect:
            p = (rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0)
            if self._w_start is None or self._w_prev is None:
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

        if self.phase is EpisodePhase.EXPLORE and self.auto_transition:
            self._check_transition()

    def _close_window(self, viewport_x: int) -> None:
        """Classifies the window just finished as transit / stuck / neither."""
        start, end = self._w_start, self._w_prev
        straightness = frontier_gain = bbox_area = 0.0
        cells_ahead: int | None = None
        centre: Point | None = None
        if start is not None and end is not None:
            centre = ((self._w_min[0] + self._w_max[0]) / 2.0,
                      (self._w_min[1] + self._w_max[1]) / 2.0)
            net = math.hypot(end[0] - start[0], end[1] - start[1])
            straightness = net / max(self._w_path, 1e-6)
            bbox_area = ((self._w_max[0] - self._w_min[0])
                         * (self._w_max[1] - self._w_min[1]))
            if self.coverage is not None and self.coverage.testable is not None:
                # Both ends measured against the SAME, current frontier, so a
                # map that moved mid-window (another worker exploring) is not
                # mistaken for Mario moving.
                n_ahead, d = self.coverage.frontier_query(viewport_x, [start, end])
                cells_ahead = n_ahead
                if n_ahead:
                    frontier_gain = float(d[0] - d[1])

        new_px = self._w_new
        observed = start is not None
        in_transit = (straightness > config.TRANSIT_STRAIGHTNESS
                      or frontier_gain > config.TRANSIT_FRONTIER_GAIN_PX)
        is_stuck = (not in_transit and observed
                    and bbox_area < config.STUCK_BBOX_AREA
                    and new_px == 0)
        exhausted = new_px < config.YIELD_FLOOR and not in_transit
        # The LOOP arm's window: Mario was somewhere, found nothing at all, and
        # was NOT transit. is_stuck is the same thing confined to a small box.
        unproductive = new_px == 0 and not in_transit and observed
        v: WindowVerdict = {
            'new_px': new_px, 'straightness': straightness,
            'frontier_gain': frontier_gain, 'bbox_area': bbox_area,
            'cells_ahead': cells_ahead, 'start': start, 'end': end,
            'path': self._w_path, 'centre': centre, 'in_transit': in_transit,
            'is_stuck': is_stuck, 'exhausted': exhausted, 'unproductive': unproductive,
        }
        self.recent_windows.append(v)

        self.consecutive_stuck = self.consecutive_stuck + 1 if is_stuck else 0
        self.consecutive_unproductive = (self.consecutive_unproductive + 1
                                         if unproductive else 0)
        self.consecutive_exhausted = (self.consecutive_exhausted + 1
                                      if exhausted else 0)
        self.last_window = v
        self._reset_window()

    def _check_transition(self) -> None:
        # T1 needs an INFORMED target. With fewer than TARGET_MIN_HISTORY
        # episodes behind it - every worker's first episodes of every run -
        # the target is the bare 500 px floor, which says nothing about this
        # policy on this map. Phase 4B measured what firing on it does: on a
        # virgin map the 6M policy met it on the FIRST substep, so an episode
        # that found ~630,000 px earned 0.00 EXPLORE novelty (every pixel paid
        # at the COMPLETE tie-breaker rate) against 1,157-1,516 for the same
        # policy once the target was informed - and on the bootstrap map it
        # handed out full completion credit after 69-233 steps. No evidence,
        # no T1; T2-T4 still apply, exactly as in_coherent_transit refuses to
        # waive a penalty before any window has closed.
        #
        # T1 ALSO needs the target's yield to have genuinely dried up (Phase
        # 4C). Meeting the target alone used to fire T1 on the very substep
        # the count crossed it - which on real 6M trajectories meant 55-64%
        # of an episode's total discovery happened AFTER the switch, at the
        # reduced COMPLETE novelty rate, while EXPLORE was still finding
        # plenty: median 2,202 px in the 60 agent steps right before
        # switching. drought_agent_steps() is the same "steps since a
        # meaningfully new pixel" signal the safety reset already tracks;
        # reusing it here (rather than a windowed rate, which most episodes
        # are too short to ever close even one LIFECYCLE_WINDOW of - median
        # 400 agent steps, one window is 240) let T1 wait for a real pause in
        # discovery. Measured on the same trajectories: post-switch discovery
        # 59.3% -> 1.4%, premature switches (>=50% of the episode's total
        # discovery still ahead) 20/90 -> 0/90. T2/T3/T4 are untouched: on
        # the same data they never overlapped with T1's old or new firing
        # point.
        #
        # Transit is exempt for the same reason it is everywhere else: a
        # temporary dip while crossing old ground toward a real frontier is
        # not exhaustion, so it must not be read as one here either. That
        # exemption is WHY T1_DECLINE_DROUGHT_STEPS == LIFECYCLE_WINDOW, not
        # merely "large enough by feel" - see config.py for the guarantee
        # this equality buys (the drought threshold can never be crossed
        # before the first window has closed to give the exemption evidence
        # to act on).
        reason = None
        if (self.target is not None and self.target_informed
                and self.coverage is not None
                and self.coverage.episode_new >= self.target
                and self.drought_agent_steps() >= config.T1_DECLINE_DROUGHT_STEPS
                and not self.in_coherent_transit):
            reason = Transition.TARGET_MET
        elif self.consecutive_exhausted >= config.YIELD_WINDOWS:
            reason = Transition.YIELD_EXHAUSTED
        elif self.last_window is not None and self.last_window['cells_ahead'] == 0:
            reason = Transition.NOTHING_AHEAD
        elif self.agent_steps >= config.MAX_EXPLORE_STEPS:
            reason = Transition.EXPLORE_BACKSTOP
        if reason is not None:
            self.force_complete(reason)

    def force_complete(self, reason: str, credit: float | None = None) -> None:
        """Moves the episode into COMPLETE, recording why and when.

        The natural criteria end here; calibration and tests call it directly
        to put an episode in COMPLETE at a chosen step. `credit` overrides the
        completion credit the lifecycle would compute for `reason`.

        One way, once per episode. Flip-flopping would make the reward
        non-stationary within an episode as soon as reward reads phase.
        """
        self.phase = EpisodePhase.COMPLETE
        self.transition_reason = reason
        self.transition_step = self.agent_steps
        self.completion_credit = self._credit_for(reason) if credit is None else credit

    def _credit_for(self, reason: str) -> float:
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
    def safety_reset_due(self) -> bool:
        return self.safety_evidence() is not None

    def safety_evidence(self) -> str | None:
        """The SafetyReason the episode has earned a reset for, or None.

        Every reset needs STAGNATION EVIDENCE; nothing made of elapsed steps
        or of a drought alone can fire it. Both arms need ALL of:

          * a drought: no meaningful new pixel for SAFETY_DROUGHT_STEPS;
          * the floor SAFETY_MIN_EPISODE_STEPS (a floor - it only ever delays
            a reset, it can never cause one);
          * the last SAFETY_STUCK_WINDOWS closed windows each found nothing
            and each was NOT coherent transit. A transit window breaks every
            streak, so Mario crossing old ground is never reset, however long
            the crossing or the drought;
          * across those windows, no meaningful progress: where Mario is (the
            centre of each window's box) moved no more than
            TRANSIT_FRONTIER_GAIN_PX, and the frontier came no closer than
            that. A slow zigzag that keeps gaining ground is progress.

        and then, what KIND of stagnation:

          STUCK  each of those windows stayed inside a small box.
          LOOP   the windows, taken as one path, doubled back on themselves:
                 net displacement is no more than TRANSIT_STRAIGHTNESS of the
                 path walked - the single-window transit test, on the span.
        """
        if (self.drought_agent_steps() <= config.SAFETY_DROUGHT_STEPS
                or self.agent_steps <= config.SAFETY_MIN_EPISODE_STEPS):
            return None
        k = config.SAFETY_STUCK_WINDOWS
        span = list(self.recent_windows)[-k:]
        if (self.consecutive_unproductive < k or len(span) < k
                or self._span_progressed(span)):
            return None
        if self.consecutive_stuck >= k:
            return SafetyReason.STUCK
        if self._span_straightness(span) <= config.TRANSIT_STRAIGHTNESS:
            return SafetyReason.LOOP
        return None

    @staticmethod
    def _span_progressed(span: list[WindowVerdict]) -> bool:
        """Did Mario get meaningfully anywhere across these windows?"""
        a, b = _seen(span[0]['centre']), _seen(span[-1]['centre'])
        moved = math.hypot(b[0] - a[0], b[1] - a[1])
        closed_in = sum(w['frontier_gain'] for w in span)
        return (moved > config.TRANSIT_FRONTIER_GAIN_PX
                or closed_in > config.TRANSIT_FRONTIER_GAIN_PX)

    @staticmethod
    def _span_straightness(span: list[WindowVerdict]) -> float:
        walked = sum(w['path'] for w in span)
        gaps = sum(math.hypot(_seen(b['start'])[0] - _seen(a['end'])[0],
                              _seen(b['start'])[1] - _seen(a['end'])[1])
                   for a, b in pairwise(span))
        path = walked + gaps
        first, last = _seen(span[0]['start']), _seen(span[-1]['end'])
        return math.hypot(last[0] - first[0], last[1] - first[1]) / max(path, 1e-6)

    def fire_safety_reset(self) -> None:
        self.safety_fired = True
        self.safety_reason = self.safety_evidence()

    # ── info, every substep ───────────────────────────────────────────────
    def annotate(self, info: Info, done: bool) -> None:
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
            info['safety_reset_reason'] = (self.safety_reason
                                           if self.end_reason == EndReason.SAFETY_RESET
                                           else None)
