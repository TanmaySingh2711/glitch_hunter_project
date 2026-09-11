import os
import threading
import time
import cv2
import gymnasium as gym

from stable_baselines3 import PPO
from collections import deque
from custom_mario_env import CustomMarioEnv

from exploration import config
from exploration import coverage as coverage_mod
from exploration import lifecycle as lifecycle_mod

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

# ═══════════════════════════════════════════════════════════════════════════
# TILE SIZE for discretizing X position.
# The Mario clone uses tiles of roughly 43 pixels (BRICK_SIZE_MULTIPLIER=2.69,
# original 16px tiles → 16*2.69 ≈ 43). We use 40px for clean math.
# This means the agent gets +1.0 reward for every NEW 40-pixel chunk it visits,
# instead of +1.0 per pixel (which caused reward explosion).
# ═══════════════════════════════════════════════════════════════════════════
TILE_SIZE = 40
MILESTONE_STEP_PX = 400         # Grant a milestone bonus every 400px of NEW max-x
SPRINT_VEL_THRESHOLD = 4.5      # Matches the engine's own "fast jump" x_vel cutoff
BACKWARD_WINDOW = 45            # ~0.75s at 60fps: net-displacement window for
                                 # the backward penalty (see below)
STUCK_WINDOW = 120              # ~2s at 60fps
STUCK_SPREAD_PX = 80            # If x-position spread in the window is below
                                 # this, Mario is considered "not really moving"
STUCK_HARD_LIMIT = 260          # Consecutive stuck checks before hard episode end

POWERUP_PROXIMITY_CAP_PX = 300  # Distance beyond which proximity shaping saturates
POWERUP_PULL_SCALE_NEEDED = 0.05    # Shaping strength while Mario is small
POWERUP_PULL_SCALE_OPTIONAL = 0.015  # Shaping strength while already powered up

# ═══════════════════════════════════════════════════════════════════════════
# ADAPTIVE DEATH MEMORY — "if he dies the same way in the same place 10 times
# in a row, he should change how he's playing there."
#
# This is implemented as a per-location, per-cause failure tracker that
# lives on the WRAPPER instance (i.e. it persists across episodes within one
# training process — it is intentionally NOT cleared in reset()). When the
# same (location, cause) combination causes death 10 CONSECUTIVE times in a
# row, that spot is marked as a temporary "danger zone" for the next 15
# episodes, during which extra shaping kicks in specifically there:
#   - pit deaths    -> momentum/clean-jump rewards are doubled in that zone,
#                       and the stuck-penalty is relaxed there (more patience
#                       to actually line up a good running jump instead of
#                       being time-pressured into repeating the same mistake)
#   - enemy deaths  -> an extra "attempt a jump here" bonus is added in that
#                       zone, nudging toward stomping instead of walking in
#   - timeout       -> the time penalty is halved in that zone
#
# This is genuine PPO-compatible shaping (nothing here needs access to the
# training loop or hyperparameters) and it decays automatically — if the
# agent starts clearing the spot, the zone still expires on schedule rather
# than lingering forever.
#
# Why shaping rather than touching the training loop: the alternatives
# considered were (a) raising PPO's ent_coef when the agent looks stuck and
# (b) curriculum-style level restarts near the failure point. Both require
# reaching into the optimizer or the training loop mid-run, which is fragile
# and cannot be done from inside an environment wrapper. This approach is
# pure reward shaping, so it needs nothing from PPO and works unchanged
# whether the agent is training or just being played back in the dashboard.
# ═══════════════════════════════════════════════════════════════════════════
DEATH_STREAK_TRIGGER = 10        # consecutive same-cause deaths at the same spot
DANGER_ZONE_BUCKET_PX = 200      # spatial resolution for "the same spot"
DANGER_ZONE_BOOST_EPISODES = 15  # how many episodes the boost stays active

# Every QA reward term, as a separately accounted channel. The channels of a
# substep sum to exactly the reward it returns - 'clip' is the backstop
# clamp's own adjustment, so nothing is left unattributed - and they are
# accumulated per episode AND per phase (GlitchHunterWrapper.ep_channels).
# 'death' is the env's own reward, which is -5.0 on any engine death. (The
# timer running out was one, until the respawn rule: in QA mode the engine
# clock is held above zero - see config QA_TIMEOUT_ENDS_EPISODE.)
QA_CHANNELS = ('death', 'novelty', 'frontier', 'drought', 'safety_reset',
               'locomotion', 'time', 'progress', 'interaction', 'flag',
               'shortfall', 'clip')


def _blank_channels():
    return {phase.value: dict.fromkeys(QA_CHANNELS, 0.0)
            for phase in lifecycle_mod.EpisodePhase}


class GlitchHunterWrapper(gym.Wrapper):
    """
    Reward shaping for the Mario PPO agent, in two selectable modes.

    ═══════════════════════════════════════════════════════════════════════
    MODE: "legacy_completion"   (exploration/config.py REWARD_MODE)
    ═══════════════════════════════════════════════════════════════════════
    The original objective, preserved EXACTLY - not approximated, not
    "mostly the same". This is what the 6M-step brain was trained under, so
    it is the only honest baseline to calibrate against and the only safe
    thing to fall back to. Its rules:

      1. Reward THOROUGH exploration of the whole level, not just running
         right as fast as possible ("no speedrunning").
      2. Never let camping/idling near an obstacle be more attractive than
         attempting it.
      3. Make tactical backward movement cheap, aimless backtracking not.
      4. Explicitly reward the mechanics needed to clear pipes/gaps.
      5. Don't double-count coins.
      6. Reward reaching the flagpole.

    ═══════════════════════════════════════════════════════════════════════
    MODE: "qa_exploration"
    ═══════════════════════════════════════════════════════════════════════
    A different objective entirely: find as much of the world as possible,
    persistently, across episodes and across processes.

        previously explored territory = TRANSIT SPACE
        new global world-space coverage = REWARD SPACE

    Why the legacy reward had to go rather than just be added to: nearly
    every dominant legacy term was a monotone function of max-x. The +1.0
    per `visited_tiles` entry is keyed on the x-COLUMN alone - it is named
    exploration but it is forward progress under another name, and it pays
    the same whether Mario walks the ground or sails over everything. Add
    +25 per 400px milestone, +0.1 per new max-x, and the completion bonus,
    and roughly 2,180 of a ~2,200-point successful episode was "how far
    right did you get". Layering a coverage bonus on top of that would have
    been a rounding error against it.

    So in QA mode those terms are DELETED, not down-weighted:
        visited_tiles, visited_altitude_tiles, milestones, new-max-x,
        the windowed backward penalty, and the x-spread stuck detector.

    One comes back, in one place. QA episodes run EXPLORE then COMPLETE
    (exploration/lifecycle.py), and new-max-x is paid again ONLY in COMPLETE,
    whose objective genuinely is finishing - scaled by how much exploration
    the episode actually did before it got there. In EXPLORE it is exactly
    zero. See config PHASE-GATED REWARD for every phase-specific term.

    The x-spread stuck detector matters most of those. It punished exactly
    the behaviours a QA explorer needs - probing a wall, trying fifteen jump
    variants from one spot, working a single suspicious corner - because it
    could not tell "not moving right" from "not exploring". Its replacement
    (drought, below) keys on FAILURE TO FIND NEW PIXELS, which is the thing
    actually worth punishing.

    What is KEPT, rescaled: the locomotion skills. Clean running jumps,
    momentum, powerups, combat. Those are capabilities, not objectives -
    the agent needs them to reach anywhere new - so they survive at roughly
    a tenth of their legacy weight and, importantly, become
    DIRECTION-AGNOSTIC. A hard leftward gap is exactly as much of a jump as
    a rightward one, and a QA explorer needs both.

    Reward is per SUBSTEP. This wrapper sits BELOW
    MaxAndSkipObservation(skip=4), which SUMS four substeps, so what PPO
    sees per decision is about 4x the numbers here.
    """

    def __init__(self, env, reward_mode=None, coverage=None, shm_names=None,
                 testable_mask=None, attach_coverage=None):
        super().__init__(env)
        self.reward_mode = reward_mode or config.REWARD_MODE
        if self.reward_mode not in ("qa_exploration", "legacy_completion"):
            raise ValueError(
                f"unknown REWARD_MODE {self.reward_mode!r}; expected "
                f"'qa_exploration' or 'legacy_completion'")

        # ─── COVERAGE ATTACHMENT ───
        # Coverage is RECORDED whenever it is attached, in BOTH modes. Only
        # the REWARD branches on the mode. That separation is what lets
        # tools/bootstrap_coverage.py replay the 6M brain under its own
        # native legacy reward - so the routes recorded are the ones it
        # actually learned - while still recording where it went.
        if attach_coverage is None:
            attach_coverage = (coverage is not None or shm_names is not None
                               or self.reward_mode == "qa_exploration")
        self.coverage = coverage
        if self.coverage is None and attach_coverage:
            mask = testable_mask
            if mask is None:
                mask = coverage_mod.load_testable()
            if shm_names:
                self.coverage = coverage_mod.open_shared(shm_names,
                                                         testable_mask=mask)
            else:
                self.coverage = coverage_mod.SpatialCoverage(testable_mask=mask)

        if self.reward_mode == "qa_exploration":
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
        base = self.env.unwrapped
        qa = self.reward_mode == "qa_exploration"
        base.episode_time_units = config.QA_EPISODE_TIME_UNITS if qa else None
        base.end_on_level_complete = config.QA_END_ON_LEVEL_COMPLETE if qa else False
        # EXPLORE -> COMPLETE and the safety reset are part of the QA
        # objective only. Legacy keeps its own stuck termination, untouched.
        self.lifecycle = lifecycle_mod.EpisodeLifecycle(self.coverage) if qa else None
        # The respawn rule: in QA, time alone never ends an episode, so the
        # engine clock is kept above zero (CustomMarioEnv.hold_clock). Legacy
        # keeps the timeout the 6M brain was trained with.
        self.holds_engine_clock = qa and not config.QA_TIMEOUT_ENDS_EPISODE
        self.ep_clock_extensions = 0

        # ─── LEGACY STATE ───
        self.visited_tiles = set()
        self.visited_altitude_tiles = set()
        self.milestones_hit = set()
        self.stuck_counter = 0
        self.last_x_pos = None
        self.x_history = deque(maxlen=STUCK_WINDOW)
        self.recent_x = deque(maxlen=BACKWARD_WINDOW)  # for windowed backward penalty
        self.sprint_frames = 0          # consecutive frames spent sprinting on ground
        self.jump_start_x = None        # x_pos when the current airborne phase began
        self.jump_start_had_momentum = False
        self.was_on_ground = True

        self.last_score = 0
        self.last_coins = 0
        self.last_status = 'small'
        self.last_flag_get = False
        self.max_x_reached = 0
        self.last_powerup_count = 0
        self.last_powerup_phi = 0.0

        # ─── QA STATE ───
        self.last_frontier_phi = 0.0
        self.last_frontier_version = -1
        self.ep_novelty_shape = 0.0     # unweighted; calibration solves on this
        self.ep_reward_total = 0.0
        self.ep_substeps = 0
        self.ep_drought_paid = 0.0
        self.ep_interaction_paid = 0.0
        self.ep_locomotion_paid = 0.0
        self.ep_max_x = 0               # episode max-x: the progress baseline
        self.ep_progress_paid = 0.0     # COMPLETE-only forward progress paid
        self.ep_complete_time_paid = 0.0
        self.ep_channels = _blank_channels()
        self.last_channels = None
        self.qa_clip_events = 0
        self.last_n_new = 0

        # ─── Adaptive death memory — persists ACROSS episodes on purpose ───
        self.death_streaks = {}       # (x_bucket, cause) -> consecutive count
        self.last_death_key = None    # the (x_bucket, cause) of the previous death
        self.danger_zones = {}        # x_bucket -> {'cause': str, 'episodes_left': int}

    def reset(self, **kwargs):
        # Clear state variables back to init state
        self.visited_tiles.clear()
        self.visited_altitude_tiles.clear()
        self.milestones_hit.clear()
        self.stuck_counter = 0
        self.last_x_pos = None
        self.x_history.clear()
        self.recent_x.clear()
        self.sprint_frames = 0
        self.jump_start_x = None
        self.jump_start_had_momentum = False
        self.was_on_ground = True

        self.last_score = 0
        self.last_coins = 0
        self.last_status = 'small'
        self.last_flag_get = False
        self.max_x_reached = 0
        self.last_powerup_count = 0
        self.last_powerup_phi = 0.0

        self.last_frontier_phi = 0.0
        self.last_frontier_version = -1
        self.ep_novelty_shape = 0.0
        self.ep_reward_total = 0.0
        self.ep_substeps = 0
        self.ep_drought_paid = 0.0
        self.ep_interaction_paid = 0.0
        self.ep_locomotion_paid = 0.0
        self.ep_max_x = 0               # episode max-x: the progress baseline
        self.ep_progress_paid = 0.0     # COMPLETE-only forward progress paid
        self.ep_complete_time_paid = 0.0
        # A NEW dict, not cleared in place: the previous episode's final info
        # carries a snapshot of the old one.
        self.ep_channels = _blank_channels()
        self.last_channels = None
        self.last_n_new = 0
        self.ep_clock_extensions = 0

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

        # Decay active danger-zone boosts by one episode; drop expired ones.
        # (death_streaks and danger_zones themselves are NOT cleared here —
        # they're meant to persist across episodes, see the class docstring
        # constants above.)
        expired = [b for b, z in self.danger_zones.items() if z['episodes_left'] <= 1]
        for b in expired:
            del self.danger_zones[b]
        for b in self.danger_zones:
            self.danger_zones[b]['episodes_left'] -= 1

        return self.env.reset(**kwargs)

    # ══════════════════════════════════════════════════════════════════════
    def step(self, action):
        if self.holds_engine_clock:
            hold = getattr(self.env.unwrapped, 'hold_clock', None)
            if hold is not None and hold():
                self.ep_clock_extensions += 1
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        if self.holds_engine_clock:
            # How many times this episode has outlived a full QA clock.
            info['clock_extensions'] = self.ep_clock_extensions

        n_new = self._record_coverage(info)
        # Lifecycle sees the substep BEFORE the reward, so the safety-reset
        # decision inside _qa_reward is made on up-to-date state - and so the
        # transition substep is already scored in the phase it moved into.
        if self.lifecycle is not None:
            self.lifecycle.observe(info, n_new)

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
    def _record_coverage(self, info):
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

    # ══════════════════════════════════════════════════════════════════════
    # QA EXPLORATION REWARD
    # ══════════════════════════════════════════════════════════════════════
    def _qa_reward(self, reward, done, info, n_new):
        cov = self.coverage
        x_pos = info.get('x_pos', self.last_x_pos or 0)
        x_vel = info.get('x_vel', 0.0)
        on_ground = info.get('on_ground', True)
        rect = info.get('mario_rect')

        current_bucket = int(x_pos) // DANGER_ZONE_BUCKET_PX
        active_zone = self.danger_zones.get(current_bucket)

        # ─── PHASE ───
        # Read after lifecycle.observe() has run for this substep, so the
        # transition substep is already scored as COMPLETE. Every term below
        # that differs by phase says so; everything else is shared.
        complete = self.lifecycle.is_complete
        credit = self.lifecycle.completion_credit

        # Per-channel accounting. Every term below is added to `reward` and
        # recorded in `ch` as the same value, in the same order, so the
        # reward itself is unchanged by keeping the books.
        ch = dict.fromkeys(QA_CHANNELS, 0.0)
        ch['death'] = float(reward)

        # ─── 1. NOVELTY — the primary signal ───
        # Paid ONLY for world pixels no worker in this run has ever occupied.
        # sqrt-scaled and capped, so a freak 2288px sweep cannot pay 60x what
        # an ordinary stride pays; a bounded reward is the whole answer to
        # "don't let magnitude explode". A revisit pays exactly
        # REVISIT_REWARD (0.0) - free, never negative, because charging for
        # traversal would make backtracking to reach unexplored space
        # self-defeating, and the agent would learn to refuse to cross its
        # own history.
        #
        # PHASE: full weight in EXPLORE. In COMPLETE it drops to a tie-breaker
        # (config.COMPLETE_NOVELTY_MULT) so a detour onto virgin ground cannot
        # out-earn heading for the finish - while the pixels themselves are
        # still RECORDED at full fidelity, in _record_coverage(), before this
        # method ever runs. Only what they pay changes.
        shape = cov.novelty_shape(n_new)
        self.ep_novelty_shape += shape
        if complete:
            novelty = config.COMPLETE_NOVELTY_MULT * config.NOVELTY_WEIGHT * shape
        else:
            novelty = cov.novelty_reward(n_new)
        reward += novelty
        ch['novelty'] = novelty

        # ─── 2. FRONTIER GUIDANCE (potential-based) ───
        # Ng/Harada/Russell shaping toward the nearest unexplored cell AHEAD
        # of the one-way camera. Deliberately weak: the entire potential
        # range is worth less than one agent-step of real discovery (checked
        # by config.assert_reward_balance at construction), so approaching
        # the frontier can never out-earn actually crossing into it.
        #
        # The delta is SKIPPED on the step a refresh lands. Phi jumps then
        # because the map moved - possibly because another worker explored
        # something - and paying for that would be paying this agent for
        # someone else's discovery.
        #
        # PHASE: EXPLORE only. In COMPLETE the potential is defined as 0, and
        # the same formula keeps running - so on the transition substep it
        # pays FRONTIER_WEIGHT * (0 - Phi_t), closing the telescoping sum,
        # and 0 ever after. Simply STOPPING mid-sum would leave the shaping
        # total depending on how close to the frontier Mario happened to be
        # when the phase changed - a small reward for WHERE he transitioned.
        # Closing it makes that irrelevant. Bounded by FRONTIER_WEIGHT (0.1).
        if rect:
            if complete:
                phi_now = 0.0
            else:
                cov.refresh_frontier(viewport_x=info.get('viewport_x', 0))
                cx = rect[0] + rect[2] // 2
                cy = rect[1] + rect[3] // 2
                phi_now = cov.phi(cx, cy)
            if cov.frontier_version == self.last_frontier_version:
                frontier = config.FRONTIER_WEIGHT * (
                    config.GAMMA * phi_now - self.last_frontier_phi)
                reward += frontier
                ch['frontier'] = frontier
            self.last_frontier_phi = phi_now
            self.last_frontier_version = cov.frontier_version

        # ─── 3. EXPLORATION DROUGHT ───
        # Pressure for FAILING TO EXPLORE - not for standing on an old pixel.
        # That distinction is the whole reason the legacy x-spread detector
        # had to go: vertical probing, wall testing and trying many jump
        # variants from one spot are exactly the QA behaviours it punished.
        # Here they are free for as long as they keep finding pixels, and
        # only a genuinely unproductive stretch costs anything.
        # The cumulative cap is as important as the per-substep one. Without
        # it the penalty is bounded per substep but unbounded per EPISODE, and
        # the first calibration run measured exactly that: episodes ending at
        # -685, with the drought outweighing everything the agent discovered
        # by three orders of magnitude. Ending the episode is what the drought
        # is for; the ramp is only the gradient that leads there.
        #
        # PHASE: never in COMPLETE - crossing covered ground toward the finish
        # is that phase's whole job. In EXPLORE it is waived while the last
        # completed lifecycle window read as COHERENT TRANSIT: a bare drought
        # is not being stuck, and old territory has to stay usable as transit
        # space. The ramp itself is unchanged, and so is its timing for
        # anything that is not demonstrably transit - including the first
        # window of an episode, where there is not yet any evidence either way.
        drought = cov.steps_since_new_pixel
        if (drought > config.DROUGHT_GRACE and not complete
                and not self.lifecycle.in_coherent_transit):
            over = drought - config.DROUGHT_GRACE
            notch = 1 + over // config.DROUGHT_STEP
            due = min(config.DROUGHT_MAX, config.DROUGHT_NOTCH * notch)
            due = min(due, config.DROUGHT_EPISODE_CAP - self.ep_drought_paid)
            if due > 0:
                self.ep_drought_paid += due
                reward -= due
                ch['drought'] = -due

        # ─── 4. SAFETY RESET (replaces the old bare-drought hard limit) ───
        # No infinite punishment loops - but a drought alone is not being
        # stuck. The old rule ended the episode after 1600 substeps without a
        # new pixel, which killed agents crossing already-covered ground to
        # reach new ground, and kept every episode far below the 5,000-step
        # floor. The lifecycle now requires a long episode AND a long drought
        # AND sustained genuine stagnation - stuck in a small box, or a
        # non-transit loop that goes nowhere (exploration/lifecycle.py). Never
        # elapsed steps or a drought alone. The terminal charge is the same
        # term and magnitude as before. Only evaluated on substeps the engine
        # has not already ended, so a death is never charged twice.
        if not done and self.lifecycle.safety_reset_due():
            self.lifecycle.fire_safety_reset()
            reward += config.DROUGHT_TERMINAL_PENALTY
            ch['safety_reset'] = config.DROUGHT_TERMINAL_PENALTY
            done = True

        # ─── 5. RETAINED LOCOMOTION SKILLS (rescaled, direction-agnostic) ───
        # abs(x_vel) rather than x_vel: building speed is a capability the
        # agent already has and needs, but "speed to the RIGHT" is precisely
        # the x-monotone bias being removed. A leftward runway into a
        # leftward gap is the same skill.
        momentum_multiplier = 2.0 if (active_zone and active_zone['cause'] == 'pit') else 1.0
        if on_ground and abs(x_vel) > SPRINT_VEL_THRESHOLD:
            self.sprint_frames = min(self.sprint_frames + 1, 30)
            momentum = self._locomotion_allowance(
                momentum_multiplier * config.QA_MOMENTUM_SCALE
                * 0.01 * self.sprint_frames / 30.0)
            reward += momentum
            ch['locomotion'] += momentum
        else:
            self.sprint_frames = 0

        # Clean jump, in either direction, for the same reason.
        if self.was_on_ground and not on_ground:
            self.jump_start_x = x_pos
            self.jump_start_had_momentum = abs(x_vel) > SPRINT_VEL_THRESHOLD
        elif (not self.was_on_ground) and on_ground:
            if self.jump_start_had_momentum and self.jump_start_x is not None:
                if abs(x_pos - self.jump_start_x) > 60:
                    jump = self._locomotion_allowance(
                        config.QA_CLEAN_JUMP_REWARD * momentum_multiplier)
                    reward += jump
                    ch['locomotion'] += jump
            self.jump_start_x = None
            self.jump_start_had_momentum = False
        self.was_on_ground = on_ground

        if active_zone and active_zone['cause'] in ('goomba', 'koopa', 'koopa_shell') and not on_ground:
            attempt = self._locomotion_allowance(0.05 * config.QA_POWERUP_SCALE)
            reward += attempt
            ch['locomotion'] += attempt

        # ─── 6. TIME ───
        # PHASE: EXPLORE_TIME_PENALTY (0.0) - at the 9,781-step QA cap a flat
        # per-substep charge would reach -196 an episode and make ending early
        # the best available move; see config for the arithmetic. COMPLETE
        # keeps mild urgency, capped per episode below the death penalty so
        # dying can never be a way to stop paying it.
        if complete:
            due = min(config.COMPLETE_TIME_PENALTY,
                      config.COMPLETE_TIME_PENALTY_EPISODE_CAP
                      - self.ep_complete_time_paid)
            if due > 0:
                self.ep_complete_time_paid += due
                reward -= due
                ch['time'] = -due
        else:
            reward -= config.EXPLORE_TIME_PENALTY
            ch['time'] = -config.EXPLORE_TIME_PENALTY

        # ─── 6b. FORWARD PROGRESS — COMPLETE ONLY ───
        # The x-monotone signal the retrofit deleted, back in exactly one
        # place: the phase whose objective IS finishing. The episode max-x
        # baseline advances in EXPLORE as well, so nothing covered earlier in
        # the episode is paid for twice, and a per-substep gain above
        # MAX_FRAME_DX is a teleport - a glitch - so it moves the baseline
        # without paying. Scaled by completion credit (lifecycle), which is
        # what stops "idle into T2, then sprint" from being worth anything.
        gain = max(0, int(x_pos) - self.ep_max_x)
        if complete and gain:
            paid = (config.COMPLETE_PROGRESS_PER_PX * credit
                    * min(gain, config.MAX_FRAME_DX))
            self.ep_progress_paid += paid
            reward += paid
            ch['progress'] = paid
        self.ep_max_x = max(self.ep_max_x, int(x_pos))

        # ─── 7. WORLD INTERACTION (kept, rescaled, and CAPPED PER EPISODE) ───
        # Coins, score and powerups still mean "you interacted with real game
        # content", which is genuinely QA-relevant. But the cap is what makes
        # that safe, and rescaling alone was not enough - see
        # config.QA_INTERACTION_EPISODE_CAP for the measurement. In short:
        # this clone pays a large end-of-level time bonus straight into
        # `score`, so a completing QA episode was still returning ~350 against
        # ~4 of novelty. Completion had not stopped being the objective, it
        # had just moved into a different variable.
        #
        # Everything in this block is accumulated into one number and charged
        # against a single per-episode ceiling. Penalties are exempt: capping
        # the downside would be a loophole rather than a safeguard.
        interaction = 0.0
        score = info.get('score', 0)
        coins = info.get('coins', 0)
        status = info.get('status', 'small')
        score_delta = score - self.last_score
        coins_delta = coins - self.last_coins
        if score_delta > 0 or coins_delta > 0:
            # Same double-count fix as legacy: the game awards +200 score AND
            # +1 coin for the same pickup.
            score_delta_adjusted = max(0, score_delta - 200 * max(0, coins_delta))
            interaction += score_delta_adjusted * config.QA_SCORE_SCALE
            interaction += coins_delta * config.QA_COIN_SCALE

        powerup_count = info.get('powerup_active_count', 0)
        if powerup_count > self.last_powerup_count:
            interaction += 8.0 * config.QA_POWERUP_SCALE
        self.last_powerup_count = powerup_count

        nearest_dx = info.get('nearest_powerup_dx', None)
        scale = (POWERUP_PULL_SCALE_NEEDED if status == 'small'
                 else POWERUP_PULL_SCALE_OPTIONAL) * config.QA_POWERUP_SCALE
        phi_pow = (-min(abs(nearest_dx), POWERUP_PROXIMITY_CAP_PX)
                   / POWERUP_PROXIMITY_CAP_PX) if nearest_dx is not None else 0.0
        interaction += scale * (config.GAMMA * phi_pow - self.last_powerup_phi)
        self.last_powerup_phi = phi_pow

        if status != self.last_status:
            if status in ['tall', 'fireball'] and self.last_status == 'small':
                interaction += 20.0 * config.QA_POWERUP_SCALE
            elif status == 'fireball' and self.last_status == 'tall':
                interaction += 10.0 * config.QA_POWERUP_SCALE
            elif status == 'small' and self.last_status in ['tall', 'fireball']:
                interaction -= 10.0 * config.QA_POWERUP_SCALE

        if interaction > 0.0:
            allowed = min(interaction,
                          config.QA_INTERACTION_EPISODE_CAP
                          - self.ep_interaction_paid)
            allowed = max(0.0, allowed)
            self.ep_interaction_paid += allowed
            reward += allowed
            ch['interaction'] = allowed
        else:
            reward += interaction
            ch['interaction'] = interaction

        # ─── 8. THE FLAG — worth what the phase says ───
        # EXPLORE: 5.0, no more than the shortfall a premature finish
        # triggers, so rushing to it can at best break even. COMPLETE: the
        # objective, rising to COMPLETE_FLAG_REWARD with completion credit -
        # and never below the EXPLORE value. The legacy +500 is not restored:
        # it would only be clipped, and it is not the point.
        flag_get = info.get('flag_get', False)
        if flag_get and not self.last_flag_get:
            if complete:
                flag = config.EXPLORE_FLAG_REWARD + credit * (
                    config.COMPLETE_FLAG_REWARD - config.EXPLORE_FLAG_REWARD)
            else:
                flag = config.EXPLORE_FLAG_REWARD
            reward += flag
            ch['flag'] = flag
        self.last_flag_get = flag_get

        self.last_score = score
        self.last_coins = coins
        self.last_status = status
        self.last_x_pos = x_pos

        # ─── 9. EPISODE SHORTFALL ───
        # Bounded at the same magnitude as the death penalty so it can never
        # dominate. The target tracks the agent's OWN recent median (measured
        # decay: 72352 -> 10955 -> 16576 -> 11972 -> 1787 -> 48), because any
        # fixed target becomes impossible within about five episodes. This
        # exerts pressure; it is not a promise that RL will hit a number.
        if done:
            target = max(1, cov.episode_target())
            if cov.episode_new < target:
                shortfall = 1.0 - (cov.episode_new / target)
                reward -= config.SHORTFALL_PENALTY * shortfall
                ch['shortfall'] = -config.SHORTFALL_PENALTY * shortfall
            info['coverage_total'] = cov.total_unique()
            info['coverage_episode_target'] = target

        self._track_death_memory(done, info, current_bucket)

        # ─── 10. BACKSTOP CLAMP ───
        # Every term above is individually bounded, so under correct
        # operation this never fires. It exists so that if one ever stops
        # being bounded, PPO's advantage estimator sees a large number rather
        # than an unbounded one - and the counter makes that visible instead
        # of silent.
        if reward > config.QA_REWARD_CLIP or reward < -config.QA_REWARD_CLIP:
            self.qa_clip_events += 1
            unclipped = reward
            reward = max(-config.QA_REWARD_CLIP,
                         min(config.QA_REWARD_CLIP, reward))
            ch['clip'] = reward - unclipped

        acc = self.ep_channels[self.lifecycle.phase.value]
        for k, v in ch.items():
            acc[k] += v
        self.last_channels = ch
        if done:
            # A snapshot, so the final info survives the next reset().
            info['qa_channels'] = {p: dict(c) for p, c in self.ep_channels.items()}

        info['qa_novelty_shape'] = self.ep_novelty_shape
        info['qa_clip_events'] = self.qa_clip_events
        info['qa_progress_paid'] = self.ep_progress_paid
        info['qa_complete_time_paid'] = self.ep_complete_time_paid
        return reward, done

    def _locomotion_allowance(self, amount):
        """What is left of this episode's locomotion budget, up to `amount`.

        Below the cap this returns `amount` itself, so every ordinary
        locomotion payment is unchanged to the bit. See
        config.QA_LOCOMOTION_EPISODE_CAP for the farm that made it necessary.
        """
        allowed = max(0.0, min(amount, config.QA_LOCOMOTION_EPISODE_CAP
                               - self.ep_locomotion_paid))
        self.ep_locomotion_paid += allowed
        return allowed

    def _track_death_memory(self, done, info, current_bucket):
        """Shared by both modes - see ADAPTIVE DEATH MEMORY at top of file."""
        death_cause = info.get('death_cause', None)
        if done and death_cause:
            key = (current_bucket, death_cause)
            if self.last_death_key == key:
                self.death_streaks[key] = self.death_streaks.get(key, 0) + 1
            else:
                self.death_streaks[key] = 1
            self.last_death_key = key

            if self.death_streaks[key] >= DEATH_STREAK_TRIGGER:
                self.danger_zones[current_bucket] = {
                    'cause': death_cause,
                    'episodes_left': DANGER_ZONE_BOOST_EPISODES,
                }
                self.death_streaks[key] = 0

    # ══════════════════════════════════════════════════════════════════════
    # LEGACY COMPLETION REWARD — unchanged from the 6M training run.
    # Do not "clean up" anything below: this is the baseline the QA reward is
    # calibrated against and the mode to fall back to, so it has to stay
    # exactly what the existing checkpoint was trained under.
    # ══════════════════════════════════════════════════════════════════════
    def _legacy_reward(self, reward, done, info):
        x_pos = info.get('x_pos', self.last_x_pos or 0)
        x_vel = info.get('x_vel', 0.0)
        on_ground = info.get('on_ground', True)

        # ─── TILE-DISCRETIZED EXPLORATION REWARD ───
        # Instead of rewarding every unique pixel (which gave +200 for just
        # running right), we reward every unique 40px tile chunk. This is
        # the main "explore every pixel" driver and it fires the same
        # whether Mario got there by walking, running, or jumping — so
        # there's no bias toward one single path through the level.
        tile_x = int(x_pos) // TILE_SIZE
        tile_y = int(info.get('y_pos', 0)) // TILE_SIZE
        if tile_x not in self.visited_tiles:
            self.visited_tiles.add(tile_x)
            reward += 1.0

        # ─── ALTITUDE EXPLORATION REWARD (BUGFIX: was per-column, not global) ───
        # The previous version paired tile_y with tile_x — (tile_x, tile_y) —
        # and rewarded +0.5 for every unique pair. That sounds reasonable but
        # during ANY ordinary rightward jump, tile_x and tile_y both change
        # nearly every single frame of the arc, so almost every frame of
        # every jump was a "new" pair and paid out +0.5. Reviewed training
        # footage confirmed this in practice: the agent learned to fire
        # straight up as high as possible while drifting right, sailing over
        # entire rows of "?" blocks and Goombas/Koopas without touching any
        # of them, because a single tall leap paid out more reward (many
        # frames * 0.5) than actually engaging with anything on the ground.
        #
        # Fixed by decoupling altitude reward from x entirely: it now only
        # pays out once per episode for each NEW global height band ever
        # reached (regardless of which column), e.g. discovering a high
        # secret platform for the first time. Routine jump arcs mostly
        # revisit height bands already seen earlier in the episode and no
        # longer pay anything, so there's no reward for jumping high just
        # for its own sake, and ground-level content stops being something
        # to fly over on the way to a bigger number.
        if tile_y not in self.visited_altitude_tiles:
            self.visited_altitude_tiles.add(tile_y)
            reward += 0.3

        # ─── HIDDEN ITEM REVEAL BONUS ───
        # Fires the instant a "?" block or brick is bumped and releases a
        # mushroom / fire flower / star / 1-up (coins are handled separately
        # below, via the score/coin delta — this is specifically for actual
        # power-ups). Before this existed, bumping one of these blocks had
        # ZERO immediate reward — the only payoff was +20 for successfully
        # walking into the resulting mushroom later, which is a much less
        # certain outcome (it can bounce off a ledge, fall in a pit, or get
        # missed). With no immediate reward for the bump itself and an
        # uncertain payoff afterward, there was little incentive to ever
        # trigger these blocks at all, which matches what the training video
        # showed: whole rows of "?" blocks left untouched. Now the bump
        # itself is worth something right away, on top of the (still much
        # larger) reward for actually collecting the item.
        powerup_count = info.get('powerup_active_count', 0)
        if powerup_count > self.last_powerup_count:
            reward += 8.0
        self.last_powerup_count = powerup_count

        # ─── POTENTIAL-BASED SHAPING TOWARD AN ACTIVE POWERUP ───
        # While a mushroom/fire flower/star/1-up is out and moving around the
        # level, nudge the agent toward it using proper potential-based
        # reward shaping (Ng, Harada & Russell 1999): reward = gamma*Phi(s')
        # - Phi(s), where Phi is based on (negative, capped) distance to the
        # nearest active power-up. This form is provably policy-invariant —
        # the reward from any round trip back to a starting position always
        # sums to exactly zero — so it CANNOT be farmed by pacing back and
        # forth near an item, and it automatically stops helping (rather
        # than punishing) once the item is gone, whether that's because
        # Mario grabbed it, it fell in a pit, or it despawned. This is what
        # makes "go get it if it's practical, but don't obsess over it if
        # it's not" possible as a single mechanism.
        #
        # The pull is intentionally stronger while Mario is small (the
        # mushroom/fire flower is more "necessary" then) and weaker once
        # already powered up, matching "get it if it is necessary."
        nearest_dx = info.get('nearest_powerup_dx', None)
        status_now = info.get('status', 'small')
        scale = POWERUP_PULL_SCALE_NEEDED if status_now == 'small' else POWERUP_PULL_SCALE_OPTIONAL
        if nearest_dx is not None:
            phi_now = -min(abs(nearest_dx), POWERUP_PROXIMITY_CAP_PX) / POWERUP_PROXIMITY_CAP_PX
        else:
            phi_now = 0.0
        reward += scale * (0.99 * phi_now - self.last_powerup_phi)
        self.last_powerup_phi = phi_now

        # ─── GENERALIZED MILESTONE BONUS ───
        # The old code gave a single hardcoded +50 for x_pos > 3000 (one
        # specific pit). That meant every obstacle AFTER that one pit had
        # zero extra incentive beyond +1.0/tile, so the agent had no strong
        # reason to push deep into the level once it farmed the first pit
        # once during training. Now every NEW 400px of progress (~11
        # milestones across the whole level) grants a chunky bonus, so
        # every pipe/pit/gap in the level gets its own "worth the risk"
        # reward, not just the first one.
        milestone = int(x_pos) // MILESTONE_STEP_PX
        if milestone > 0 and milestone not in self.milestones_hit:
            self.milestones_hit.add(milestone)
            reward += 25.0

        # ─── FORWARD PROGRESS BONUS ───
        # Small bonus for reaching new maximum X. Encourages steady forward movement.
        if x_pos > self.max_x_reached:
            reward += 0.1
            self.max_x_reached = x_pos

        # ─── WINDOWED BACKWARD PENALTY (was: instant per-frame penalty) ───
        # The old version subtracted 0.1 on literally every single frame
        # where x_pos ticked down even 1px. That punished the EXACT
        # behavior needed to clear a hard jump — backing up a few steps to
        # get a running start — as harshly as aimless backtracking. Now we
        # only penalize *net* backward drift measured over a short rolling
        # window (~0.75s). A quick "back up 3 tiles, then sprint past your
        # old position" nets out to a forward gain over that window and
        # costs nothing; genuinely wandering backward for a long stretch
        # still gets penalized.
        self.recent_x.append(x_pos)
        if len(self.recent_x) == self.recent_x.maxlen:
            net_drift = self.recent_x[-1] - self.recent_x[0]
            if net_drift < -20:
                reward -= 0.03

        # ─── DANGER ZONE LOOKUP (see ADAPTIVE DEATH MEMORY at top of file) ───
        current_bucket = int(x_pos) // DANGER_ZONE_BUCKET_PX
        active_zone = self.danger_zones.get(current_bucket)

        # ─── MOMENTUM / SPRINT-BUILDING REWARD ───
        # Directly rewards holding a rightward sprint (x_vel above the
        # engine's own "fast jump" threshold of 4.5) while on the ground.
        # This is what teaches the agent that charging up speed BEFORE a
        # jump is valuable, rather than just tapping jump the instant it's
        # adjacent to an obstacle.
        momentum_multiplier = 2.0 if (active_zone and active_zone['cause'] == 'pit') else 1.0
        if on_ground and x_vel > SPRINT_VEL_THRESHOLD:
            self.sprint_frames = min(self.sprint_frames + 1, 30)
            reward += momentum_multiplier * 0.01 * self.sprint_frames / 30.0
        else:
            self.sprint_frames = 0

        # ─── CLEAN RUNNING-JUMP BONUS ───
        # Tracks each airborne phase (ground -> not ground) and remembers
        # whether Mario had real sprint momentum (x_vel > threshold) the
        # instant he left the ground. If he lands having advanced a good
        # distance forward without dying, that's a "clean clear" of a
        # pipe/gap using proper momentum, so it gets an explicit bonus on
        # top of the tile/milestone rewards it already earned along the way.
        #
        # Doubled inside an active "pit" danger zone (see ADAPTIVE DEATH
        # MEMORY): if Mario has died at this exact spot to a pit 10 times in
        # a row, a successful clean clear here is worth extra specifically
        # to pull the policy toward repeating THAT instead of the failure.
        if self.was_on_ground and not on_ground:
            self.jump_start_x = x_pos
            self.jump_start_had_momentum = x_vel > SPRINT_VEL_THRESHOLD
        elif (not self.was_on_ground) and on_ground:
            if self.jump_start_had_momentum and self.jump_start_x is not None:
                cleared = x_pos - self.jump_start_x
                if cleared > 60:
                    reward += 3.0 * momentum_multiplier
            self.jump_start_x = None
            self.jump_start_had_momentum = False
        self.was_on_ground = on_ground

        # ─── "TRY A JUMP HERE" NUDGE (enemy danger zones only) ───
        # If this spot has killed Mario via a Goomba/Koopa 10 times in a row,
        # give a small extra nudge toward attempting a jump while passing
        # through it — a stomp attempt instead of repeating whatever ground-
        # level contact killed him last time. Deliberately small: this is a
        # nudge, not a command, and the agent still has to time it correctly
        # to get the (much larger) real combat reward from an actual kill.
        if active_zone and active_zone['cause'] in ('goomba', 'koopa', 'koopa_shell') and not on_ground:
            reward += 0.05

        # ─── STUCK DETECTION (SMOOTH GRADIENT) ───
        # Uses a 120-frame window (~2 seconds). If Mario's X spread is <= 80px
        # in that window, he's not really going anywhere (jumping in place,
        # running into a wall, pacing back and forth, etc).
        #
        # The old version did nothing until the hard 200-step cutoff, which
        # meant "camp near the pit and jump in place" felt just as safe as
        # actually playing right up until the wall — no gradient pushed the
        # agent to bail out early. Now a small escalating penalty kicks in
        # well before the hard limit, so standing/jumping in place is
        # visibly worse than even a failed, risky attempt to cross.
        #
        # Relaxed (threshold raised from 40 to 100) inside an active "pit"
        # danger zone: the agent needs room to deliberately back up and set
        # up a proper running jump there without being time-pressured into
        # repeating the same rushed attempt that's been killing it.
        self.x_history.append(x_pos)
        if len(self.x_history) == self.x_history.maxlen:
            spread = max(self.x_history) - min(self.x_history)
            if spread <= STUCK_SPREAD_PX:
                self.stuck_counter += 1
                stuck_threshold = 100 if (active_zone and active_zone['cause'] == 'pit') else 40
                if self.stuck_counter > stuck_threshold:
                    reward -= 0.02 * min((self.stuck_counter - stuck_threshold) / 20.0, 5.0)
            else:
                self.stuck_counter = 0

        # ─── TIME PENALTY (REDUCED) ───
        # The project explicitly does NOT want a speedrunning agent — it
        # wants thorough exploration. A flat -0.1/step (-6.0/sec) creates
        # strong urgency that competes with careful play and exploration
        # bonuses. Reduced to a light -0.02/step: enough to stop an episode
        # from stalling forever doing nothing productive, but no longer the
        # dominant term pushing the agent to rush.
        #
        # Halved inside an active "timeout" danger zone: if the clock has
        # run out here 10 times in a row, the agent needs more breathing
        # room, not more time pressure.
        reward -= 0.01 if (active_zone and active_zone['cause'] == 'timeout') else 0.02

        # ─── STUCK TERMINATION ───
        # If stuck for 260+ consecutive checks (~4.3s real-time), end the
        # episode. No infinite punishment loops.
        if self.stuck_counter > STUCK_HARD_LIMIT:
            reward -= 5.0
            done = True

        # ─── SCORE AND COIN REWARDS (double-count fixed) ───
        score = info.get('score', 0)
        coins = info.get('coins', 0)
        status = info.get('status', 'small')

        score_delta = score - self.last_score
        coins_delta = coins - self.last_coins

        if score_delta > 0 or coins_delta > 0:
            # BUG FIX: in the underlying game, every coin picked up from a
            # brick/box ALSO adds exactly +200 to `score` in the same frame
            # (see level1.py adjust_mario_for_y_*_collisions). The old code
            # rewarded both the score delta (*0.1 => +20) AND the coin delta
            # (*3.0 => +3) for the SAME coin, handing out +23 for one block
            # bump — about 23 tiles' worth of forward-progress reward from
            # standing still under one brick. That made camping near a
            # stack of coin bricks (exactly what's happening in the "stuck
            # near the pit" screenshot, which has bricks right overhead)
            # far more attractive than risking the pit. We now subtract the
            # coin-attributable portion out of the score delta before
            # applying the score multiplier, so each coin is only rewarded
            # once, via the dedicated coin bonus.
            score_delta_adjusted = max(0, score_delta - 200 * max(0, coins_delta))
            # Raised from 0.1 -> 0.15: makes stomping/fireballing enemies and
            # the flagpole-height bonus more salient rewards on their own,
            # not just a side effect of moving forward. Per the request that
            # the agent should "kill them too whenever it gets the chance"
            # (not avoid combat entirely, just not be forced into it).
            reward += score_delta_adjusted * 0.15    # Combat/points reward
            reward += coins_delta * 3.0               # Coin collection reward

        # ─── POWERUP REWARDS ───
        if status != self.last_status:
            if status in ['tall', 'fireball'] and self.last_status == 'small':
                reward += 20.0  # Got a mushroom!
            elif status == 'fireball' and self.last_status == 'tall':
                reward += 10.0  # Got a fire flower!
            elif status == 'small' and self.last_status in ['tall', 'fireball']:
                reward -= 10.0  # Got hit and shrunk

        # ─── LEVEL COMPLETION BONUS ───
        # The agent has never finished a level once. Reward the moment
        # Mario reaches the flagpole/castle sequence (flag_get rising
        # edge), rather than waiting for the multi-second end-of-level
        # animation to fully play out and `done` to fire naturally.
        flag_get = info.get('flag_get', False)
        if flag_get and not self.last_flag_get:
            reward += 500.0
        self.last_flag_get = flag_get

        self.last_score = score
        self.last_coins = coins
        self.last_status = status
        self.last_x_pos = x_pos

        # ─── ADAPTIVE DEATH MEMORY: record this death, escalate if repeated ───
        # Runs whenever the episode is ending because Mario actually died
        # (death_cause is set — this excludes the "stuck" hard-termination
        # above, which isn't a death and shouldn't count toward this).
        death_cause = info.get('death_cause', None)
        if done and death_cause:
            key = (current_bucket, death_cause)
            if self.last_death_key == key:
                self.death_streaks[key] = self.death_streaks.get(key, 0) + 1
            else:
                self.death_streaks[key] = 1
            self.last_death_key = key

            if self.death_streaks[key] >= DEATH_STREAK_TRIGGER:
                self.danger_zones[current_bucket] = {
                    'cause': death_cause,
                    'episodes_left': DANGER_ZONE_BOOST_EPISODES,
                }
                # Reset the streak so it takes another 10 in a row before
                # re-triggering (rather than re-arming the zone every death).
                self.death_streaks[key] = 0
        return reward, done

from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation, ResizeObservation, MaxAndSkipObservation

# Must match CHECKPOINT_NAME / FINAL_MODEL_PATH in train_agent.py — this is
# the master checkpoint that training writes on finish and on Ctrl+C.
CHECKPOINT_NAME = "mario_brain_checkpoint"

_global_env = None
_global_model = None

# ═══════════════════════════════════════════════════════════════════════
# ENV_LOCK — serialises every touch of the environment and its pygame window
#
# app.py runs Flask-SocketIO in threading mode, so a click handler ("Reset",
# or a browser disconnect) executes on a DIFFERENT OS thread from the frame
# loop. Without this lock, close_window() can destroy the SDL display while
# the frame loop is part-way through env.step() or render_scaled().
#
# That is not hypothetical. A stress test firing start/stop/reset at random
# intervals for 20 seconds produced 9 crashes in the frame loop:
#     pygame.error: video system not initialized
#     pygame.error: cannot convert without pygame.display initialized
# The old eventlet mode could not hit this - one thread, switching only at
# explicit yield points - so it arrived with the move to real threads.
#
# Rule: hold this for the duration of any single env operation, and NEVER
# across a sleep. Teardown then lands cleanly between two steps instead of
# in the middle of one.
#
# The dashboard itself no longer relies on it: every window and env call now
# runs on dashboard_service's single game thread (the only thread Windows
# lets drive the window), so there is nothing left to race. It stays as a
# guard for the module-level helpers, which remain callable from anywhere.
# ═══════════════════════════════════════════════════════════════════════
env_lock = threading.RLock()

# Streamed frames are resized down from the native 800x600 window before
# JPEG encoding. The dashboard displays them scaled into a much smaller
# panel anyway (CSS `object-fit: contain`), so sending full resolution only
# costs encode time and bandwidth for pixels nobody sees larger. 480x360
# keeps the exact 4:3 aspect ratio at 36% of the original pixel count.
STREAM_SIZE = (480, 360)
STREAM_JPEG_QUALITY = 80  # default cv2 quality (~95) costs real encode time
                          # for no visible difference at this display size


def _ensure_global_env_and_model():
    """Creates the global env + loads the model on first use. Shared by
    run_mario_agent() and open_agent_window() so a window can be popped up
    (on the very first 'Start Testing' click) without duplicating this
    setup logic in two places."""
    global _global_env, _global_model
    if _global_env is not None:
        return

    # ─── THE REWARD MODE FOLLOWS THE CHECKPOINT, NOT THE CONFIG ───
    # The dashboard displays whatever brain is on disk, so it has to display
    # that brain's OWN reward. Reading config.REWARD_MODE here would show QA
    # rewards for a completion-phase policy - numbers that describe an
    # objective this checkpoint was never trained on - and would additionally
    # let the drought terminate episodes early during a demo, which looks
    # exactly like a bug.
    #
    # The QA brain wins when it exists, because by then it is the current one.
    qa_path = f"{config.CHECKPOINT_NAME_QA}.zip"
    if os.path.exists(qa_path):
        model_path, mode = qa_path, "qa_exploration"
    else:
        model_path, mode = f"{CHECKPOINT_NAME}.zip", "legacy_completion"

    # In QA mode, show coverage against the map that brain actually built.
    # A fresh empty map would credit it with rediscovering the whole level
    # every time the dashboard is opened.
    coverage = None
    if mode == "qa_exploration":
        coverage = coverage_mod.SpatialCoverage(
            testable_mask=coverage_mod.load_testable())
        paired = f"{config.CHECKPOINT_NAME_QA}_coverage.npz"
        if os.path.exists(paired):
            try:
                coverage.load(paired)
            except Exception as exc:      # never block playback on telemetry
                print(f"[WARNING] could not load {paired}: {exc}")

    _global_env = CustomMarioEnv()
    _global_env = GlitchHunterWrapper(_global_env, reward_mode=mode,
                                      coverage=coverage)
    _global_env = MaxAndSkipObservation(_global_env, skip=4)
    _global_env = GrayscaleObservation(_global_env, keep_dim=False)
    _global_env = ResizeObservation(_global_env, (84, 84))
    _global_env = FrameStackObservation(_global_env, 4)

    # Initialize model. Falls back to an untrained policy so the
    # dashboard still runs (badly) rather than crashing outright when no
    # checkpoint is present.
    if os.path.exists(model_path):
        print(f"[DASHBOARD] {model_path} ({mode})")
        _global_model = PPO.load(model_path, env=_global_env, device="auto")
    else:
        print(f"[WARNING] {model_path} not found — running an UNTRAINED policy.")
        _global_model = PPO('CnnPolicy', _global_env, verbose=0)


def open_agent_window():
    """Ensures the env/model exist and the game window is visible and
    focused. Called on every 'Start Testing' click (not just the first),
    so the window reliably comes to the front even if it's buried behind
    other windows from earlier in the session."""
    with env_lock:
        _ensure_global_env_and_model()
        _global_env.unwrapped.open_window()


def close_agent_window():
    """Closes the game window if one exists. Called on dashboard reset. The
    model and env stay loaded in memory - only the OS window closes - so the
    next open_agent_window() call is fast, not a full reload."""
    with env_lock:
        if _global_env is not None:
            _global_env.unwrapped.close_window()


class DashboardBackend:
    """The real work behind dashboard_service.GameWindowService.

    Every method runs on the service's one game thread - the thread that
    creates the window, and so the only one Windows lets drive it. env_lock
    is still taken, so the module-level helpers above stay safe to call.
    """

    def preload(self):
        """Loads the model and builds the env - which creates the window, on
        this thread - then hides the window until the first Start."""
        with env_lock:
            _ensure_global_env_and_model()
            _global_env.unwrapped.hide_window()

    def open_window(self):
        with env_lock:
            _ensure_global_env_and_model()
            return _global_env.unwrapped.open_window()

    def hide_window(self):
        with env_lock:
            if _global_env is not None:
                _global_env.unwrapped.hide_window()

    def close_window(self):
        close_agent_window()

    def poll_close_request(self):
        with env_lock:
            return (_global_env is not None
                    and _global_env.unwrapped.poll_close_request())

    def new_session(self):
        return run_mario_agent()

    def stop_audio(self):
        # Best-effort: the mixer may not be initialized at all (audio is
        # forced to the "dummy" driver in custom_mario_env.py).
        try:
            import pygame as pg
            if pg.mixer.get_init():
                pg.mixer.music.stop()
                pg.mixer.stop()
        except Exception:
            pass


def run_mario_agent():
    _ensure_global_env_and_model()
    env = _global_env
    model = _global_model

    # Initial reset
    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, _ = reset_result
    else:
        obs = reset_result
    obs = obs.copy()

    step_count = 0
    fps_window_start = time.time()
    fps_window_frames = 0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        action_val = int(action.item()) if hasattr(action, 'item') else int(action)

        # Step environment
        step_result = env.step(action_val)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated

        obs = obs.copy()
        step_count += 1

        # Render already-downscaled to STREAM_SIZE via render_scaled() -
        # see custom_mario_env.py for why this avoids a full-resolution
        # array3d() call (the same technique _fast_obs() uses for the
        # model's observation, just at a different target size for display).
        frame = env.unwrapped.render_scaled(STREAM_SIZE)

        if frame is not None:
            # BGR for cv2, then JPEG at a quality that doesn't waste CPU on
            # precision nobody sees at this display size.
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
            # Raw bytes, not base64: flask-socketio/python-socketio send
            # `bytes` values as a native binary WebSocket frame automatically
            # (the client's socket.io library reassembles it transparently).
            # Base64 was costing ~33% more payload plus real encode/decode
            # CPU time on both ends for no benefit, since nothing here
            # actually needs a text-safe representation.
            frame_bytes = buffer.tobytes()
        else:
            frame_bytes = b""

        # ─── SERVER-SIDE FPS INSTRUMENTATION ───
        # Prints the actual measured frame rate every ~2 seconds, so "is it
        # really running at 60fps" is something you can read from the
        # terminal instead of guessing from how smooth the browser looks.
        fps_window_frames += 1
        now = time.time()
        elapsed = now - fps_window_start
        if elapsed >= 2.0:
            print(f"[STREAM FPS] {fps_window_frames / elapsed:.1f}")
            fps_window_start = now
            fps_window_frames = 0

        log_message = None
        if info.get('glitch_alert'):
            log_message = '🚨 BUG FOUND: ' + info['glitch_alert']
        else:
            action_name = ACTION_NAMES.get(int(action_val), "Unknown")
            log_message = f"Step {step_count}: Action: {action_name} ({action_val}) | Reward: {float(reward):.2f}"

        yield {
            'frame': frame_bytes,
            'action': action_val,
            'step': step_count,
            'reward': float(reward),
            'log': log_message
        }

        if done:
            reset_result = env.reset()
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs, _ = reset_result
            else:
                obs = reset_result
            obs = obs.copy()
