"""LEGACY COMPLETION REWARD — unchanged from the 6M training run.

Do not "clean up" the arithmetic below: this is the baseline the QA reward is
calibrated against and the mode to fall back to, so it has to stay exactly
what the existing checkpoint was trained under - including the ORDER of the
additions, which decides the last bits of every float it returns
(tests/test_reward_wrapper.py pins it). It moved here from agent_logic.py
verbatim; the only change is that the death-memory block, which was a copy of
the shared RewardState._track_death_memory, now calls it.

Its rules:
  1. Reward THOROUGH exploration of the whole level, not just running
     right as fast as possible ("no speedrunning").
  2. Never let camping/idling near an obstacle be more attractive than
     attempting it.
  3. Make tactical backward movement cheap, aimless backtracking not.
  4. Explicitly reward the mechanics needed to clear pipes/gaps.
  5. Don't double-count coins.
  6. Reward reaching the flagpole.
"""
from __future__ import annotations

from collections import deque

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

# ═══════════════════════════════════════════════════════════════════════════
# TILE SIZE for discretizing X position.
# The Mario clone uses tiles of roughly 43 pixels (BRICK_SIZE_MULTIPLIER=2.69,
# original 16px tiles → 16*2.69 ≈ 43). We use 40px for clean math.
# This means the agent gets +1.0 reward for every NEW 40-pixel chunk it visits,
# instead of +1.0 per pixel (which caused reward explosion).
# ═══════════════════════════════════════════════════════════════════════════
TILE_SIZE = 40
MILESTONE_STEP_PX = 400         # Grant a milestone bonus every 400px of NEW max-x
BACKWARD_WINDOW = 45            # ~0.75s at 60fps: net-displacement window for
                                 # the backward penalty (see below)
STUCK_WINDOW = 120              # ~2s at 60fps
STUCK_SPREAD_PX = 80            # If x-position spread in the window is below
                                 # this, Mario is considered "not really moving"
STUCK_HARD_LIMIT = 260          # Consecutive stuck checks before hard episode end


class LegacyCompletionReward(RewardState):
    """Mixin: the "legacy_completion" objective. See the module docstring."""

    visited_tiles: set[int]
    visited_altitude_tiles: set[int]
    milestones_hit: set[int]
    stuck_counter: int
    x_history: deque[float]
    recent_x: deque[float]          # for the windowed backward penalty
    max_x_reached: float

    def _reset_legacy_state(self) -> None:
        self.visited_tiles = set()
        self.visited_altitude_tiles = set()
        self.milestones_hit = set()
        self.stuck_counter = 0
        self.x_history = deque(maxlen=STUCK_WINDOW)
        self.recent_x = deque(maxlen=BACKWARD_WINDOW)
        self.max_x_reached = 0

    def _legacy_reward(self, reward: float, done: bool, info: Info) -> tuple[float, bool]:
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
        status_now = info.get('status', 'small')
        scale = POWERUP_PULL_SCALE_NEEDED if status_now == 'small' else POWERUP_PULL_SCALE_OPTIONAL
        phi_now = powerup_potential(info.get('nearest_powerup_dx', None))
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

        # ─── DANGER ZONE LOOKUP (see ADAPTIVE DEATH MEMORY, rewards/shared.py) ───
        current_bucket = danger_bucket(x_pos)
        active_zone = self.danger_zones.get(current_bucket)

        # ─── MOMENTUM / SPRINT-BUILDING REWARD ───
        # Directly rewards holding a rightward sprint (x_vel above the
        # engine's own "fast jump" threshold of 4.5) while on the ground.
        # This is what teaches the agent that charging up speed BEFORE a
        # jump is valuable, rather than just tapping jump the instant it's
        # adjacent to an obstacle.
        momentum_multiplier = 2.0 if (active_zone and active_zone['cause'] == 'pit') else 1.0
        if on_ground and x_vel > SPRINT_VEL_THRESHOLD:
            self.sprint_frames = min(self.sprint_frames + 1, SPRINT_RAMP_FRAMES)
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
                if cleared > CLEAN_JUMP_MIN_PX:
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
        if active_zone and active_zone['cause'] in ENEMY_CAUSES and not on_ground:
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
        self._track_death_memory(done, info, current_bucket)
        return reward, done
