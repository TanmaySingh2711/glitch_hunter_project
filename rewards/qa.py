"""QA EXPLORATION REWARD — find as much of the world as possible, persistently.

        previously explored territory = TRANSIT SPACE
        new global world-space coverage = REWARD SPACE

Why the legacy reward had to go rather than just be added to: nearly every
dominant legacy term was a monotone function of max-x. The +1.0 per
`visited_tiles` entry is keyed on the x-COLUMN alone - it is named
exploration but it is forward progress under another name, and it pays the
same whether Mario walks the ground or sails over everything. Add +25 per
400px milestone, +0.1 per new max-x, and the completion bonus, and roughly
2,180 of a ~2,200-point successful episode was "how far right did you get".
Layering a coverage bonus on top of that would have been a rounding error
against it.

So in QA mode those terms are DELETED, not down-weighted:
    visited_tiles, visited_altitude_tiles, milestones, new-max-x,
    the windowed backward penalty, and the x-spread stuck detector.

One comes back, in one place. QA episodes run EXPLORE then COMPLETE
(exploration/lifecycle.py), and new-max-x is paid again ONLY in COMPLETE,
whose objective genuinely is finishing - scaled by how much exploration the
episode actually did before it got there. In EXPLORE it is exactly zero. See
config PHASE-GATED REWARD for every phase-specific term.

The x-spread stuck detector matters most of those. It punished exactly the
behaviours a QA explorer needs - probing a wall, trying fifteen jump
variants from one spot, working a single suspicious corner - because it
could not tell "not moving right" from "not exploring". Its replacement
(drought, below) keys on FAILURE TO FIND NEW PIXELS, which is the thing
actually worth punishing.

What is KEPT, rescaled: the locomotion skills. Clean running jumps,
momentum, powerups, combat. Those are capabilities, not objectives - the
agent needs them to reach anywhere new - so they survive at roughly a tenth
of their legacy weight and, importantly, become DIRECTION-AGNOSTIC. A hard
leftward gap is exactly as much of a jump as a rightward one, and a QA
explorer needs both.
"""
from __future__ import annotations

from exploration import config
from exploration import lifecycle as lifecycle_mod
from exploration.coverage import SpatialCoverage

from .shared import (
    CLEAN_JUMP_MIN_PX,
    ENEMY_CAUSES,
    POWERUP_PULL_SCALE_NEEDED,
    POWERUP_PULL_SCALE_OPTIONAL,
    SPRINT_RAMP_FRAMES,
    SPRINT_VEL_THRESHOLD,
    Info,
    RewardState,
    danger_bucket,
    powerup_potential,
)

# Every QA reward term, as a separately accounted channel. The channels of a
# substep sum to exactly the reward it returns - 'clip' is the backstop
# clamp's own adjustment, so nothing is left unattributed - and they are
# accumulated per episode AND per phase (QAExplorationReward.ep_channels).
# 'death' is the env's own reward, which is -5.0 on any engine death. (The
# timer running out was one, until the respawn rule: in QA mode the engine
# clock is held above zero - see config QA_TIMEOUT_ENDS_EPISODE.)
QA_CHANNELS = ('death', 'novelty', 'frontier', 'drought', 'safety_reset',
               'locomotion', 'time', 'progress', 'interaction', 'flag',
               'shortfall', 'clip')

Channels = dict[str, float]


def blank_channels() -> dict[str, Channels]:
    """Zeroed per-phase channel accumulators: {phase: {channel: 0.0}}."""
    return {phase.value: dict.fromkeys(QA_CHANNELS, 0.0)
            for phase in lifecycle_mod.EpisodePhase}


class QAExplorationReward(RewardState):
    """Mixin: the "qa_exploration" objective. See the module docstring.

    Reward is per SUBSTEP. The wrapper sits BELOW MaxAndSkipObservation(skip=4),
    which SUMS four substeps, so what PPO sees per decision is about 4x the
    numbers here.
    """

    # Provided by the wrapper; a QA wrapper always has both.
    coverage: SpatialCoverage | None
    lifecycle: lifecycle_mod.EpisodeLifecycle | None

    last_frontier_phi: float
    last_frontier_version: int
    ep_novelty_shape: float         # unweighted; calibration solves on this
    ep_drought_paid: float
    ep_interaction_paid: float
    ep_locomotion_paid: float
    ep_max_x: int                   # episode max-x: the progress baseline
    ep_progress_paid: float         # COMPLETE-only forward progress paid
    ep_complete_time_paid: float
    ep_channels: dict[str, Channels]
    last_channels: Channels | None
    qa_clip_events: int             # cumulative across episodes, like the clamp itself

    def _reset_qa_state(self) -> None:
        self.last_frontier_phi = 0.0
        self.last_frontier_version = -1
        self.ep_novelty_shape = 0.0
        self.ep_drought_paid = 0.0
        self.ep_interaction_paid = 0.0
        self.ep_locomotion_paid = 0.0
        self.ep_max_x = 0
        self.ep_progress_paid = 0.0
        self.ep_complete_time_paid = 0.0
        # A NEW dict, not cleared in place: the previous episode's final info
        # carries a snapshot of the old one.
        self.ep_channels = blank_channels()
        self.last_channels = None

    def _qa_reward(self, reward: float, done: bool, info: Info,
                   n_new: int) -> tuple[float, bool]:
        cov = self.coverage
        lifecycle = self.lifecycle
        if cov is None or lifecycle is None:
            raise RuntimeError("the QA reward needs a coverage channel and a lifecycle")
        x_pos = info.get('x_pos', self.last_x_pos or 0)
        x_vel = info.get('x_vel', 0.0)
        on_ground = info.get('on_ground', True)
        rect = info.get('mario_rect')

        current_bucket = danger_bucket(x_pos)
        active_zone = self.danger_zones.get(current_bucket)

        # ─── PHASE ───
        # Read after lifecycle.observe() has run for this substep, so the
        # transition substep is already scored as COMPLETE. Every term below
        # that differs by phase says so; everything else is shared.
        complete = lifecycle.is_complete
        credit = lifecycle.completion_credit

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
                and not lifecycle.in_coherent_transit):
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
        if not done and lifecycle.safety_reset_due():
            lifecycle.fire_safety_reset()
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
            self.sprint_frames = min(self.sprint_frames + 1, SPRINT_RAMP_FRAMES)
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
                if abs(x_pos - self.jump_start_x) > CLEAN_JUMP_MIN_PX:
                    jump = self._locomotion_allowance(
                        config.QA_CLEAN_JUMP_REWARD * momentum_multiplier)
                    reward += jump
                    ch['locomotion'] += jump
            self.jump_start_x = None
            self.jump_start_had_momentum = False
        self.was_on_ground = on_ground

        if active_zone and active_zone['cause'] in ENEMY_CAUSES and not on_ground:
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

        scale = (POWERUP_PULL_SCALE_NEEDED if status == 'small'
                 else POWERUP_PULL_SCALE_OPTIONAL) * config.QA_POWERUP_SCALE
        phi_pow = powerup_potential(info.get('nearest_powerup_dx', None))
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
            # ─── THROTTLED WHILE THE EPISODE IS FAILING TO EXPLORE ───
            # The per-episode ceiling is absolute but novelty decays with
            # coverage, so the margin that made the ceiling safe inverts on
            # its own (config.QA_INTERACTION_DROUGHT_SCALE has the measured
            # numbers and what it cost). The condition is deliberately the
            # SAME one the drought penalty uses a few blocks above, not a new
            # signal: it already means "unproductive lingering, and not
            # merely standing on an old pixel", and it already exempts
            # COMPLETE and coherent transit. Sharing it means the two can
            # never disagree about whether this substep counts as exploring.
            if (drought > config.DROUGHT_GRACE and not complete
                    and not lifecycle.in_coherent_transit):
                interaction *= config.QA_INTERACTION_DROUGHT_SCALE
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

        acc = self.ep_channels[lifecycle.phase.value]
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

    def _locomotion_allowance(self, amount: float) -> float:
        """What is left of this episode's locomotion budget, up to `amount`.

        Below the cap this returns `amount` itself, so every ordinary
        locomotion payment is unchanged to the bit. See
        config.QA_LOCOMOTION_EPISODE_CAP for the farm that made it necessary.
        """
        allowed = max(0.0, min(amount, config.QA_LOCOMOTION_EPISODE_CAP
                               - self.ep_locomotion_paid))
        self.ep_locomotion_paid += allowed
        return allowed
