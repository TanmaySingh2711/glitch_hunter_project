import os
import cv2
import gymnasium as gym

from stable_baselines3 import PPO
import numpy as np
from collections import deque
import base64
from custom_mario_env import CustomMarioEnv

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
LEVEL_LENGTH_PX = 8800          # Approx. full length of world 1-1 in this clone
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
# than lingering forever. See IMPLEMENTATION_3.md Part C for the full
# reasoning and the other options that were considered.
# ═══════════════════════════════════════════════════════════════════════════
DEATH_STREAK_TRIGGER = 10        # consecutive same-cause deaths at the same spot
DANGER_ZONE_BUCKET_PX = 200      # spatial resolution for "the same spot"
DANGER_ZONE_BOOST_EPISODES = 15  # how many episodes the boost stays active


class GlitchHunterWrapper(gym.Wrapper):
    """
    Reward shaping for the Mario PPO agent.

    Design goals (see IMPLEMENTATION.md for the full rationale):
      1. Reward THOROUGH exploration of the whole level, not just running
         right as fast as possible ("no speedrunning").
      2. Never let camping/idling near an obstacle be more attractive than
         attempting it — remove reward sources that can be farmed in place,
         and make "stuck" pressure escalate smoothly instead of a hard wall.
      3. Make tactical backward movement (backing up to get a running start)
         cheap, while still discouraging aimless backtracking.
      4. Explicitly reward the mechanics needed to clear pipes/gaps: sustained
         sprint velocity, and a clean running jump.
      5. Don't double-count coins (the underlying game awards both
         COIN_TOTAL +1 *and* SCORE +200 for the same coin — see custom
         reward calc below).
      6. Reward reaching the flagpole/level completion, since the agent
         has never once finished a level.
    """

    def __init__(self, env):
        super().__init__(env)
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
        self.total_steps = 0

        self.last_score = 0
        self.last_coins = 0
        self.last_status = 'small'
        self.last_flag_get = False
        self.max_x_reached = 0
        self.last_powerup_count = 0
        self.last_powerup_phi = 0.0

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

    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
        else:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated

        self.total_steps += 1

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

        return obs, float(reward), done, False, info

from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation, ResizeObservation, MaxAndSkipObservation

_global_env = None
_global_model = None

def run_mario_agent():
    global _global_env, _global_model
    
    if _global_env is None:
        
        _global_env = CustomMarioEnv()
        _global_env = GlitchHunterWrapper(_global_env)
        _global_env = MaxAndSkipObservation(_global_env, skip=4)
        _global_env = GrayscaleObservation(_global_env, keep_dim=False)
        _global_env = ResizeObservation(_global_env, (84, 84))
        _global_env = FrameStackObservation(_global_env, 4)
        
        # Initialize model
        model_path = "mario_brain_v2_checkpoint.zip"
        if os.path.exists(model_path):
            _global_model = PPO.load(model_path, env=_global_env, device="auto")
        else:
            _global_model = PPO('CnnPolicy', _global_env, verbose=0)
        
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
        
        # Render frame
        try:
            frame = env.render(mode="rgb_array")
        except TypeError:
            frame = env.render()
            
        if frame is not None:
            # Convert RGB to BGR for cv2
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Encode BGR to JPEG
            _, buffer = cv2.imencode('.jpg', frame_bgr)
            b64_string = base64.b64encode(buffer).decode('utf-8')
        else:
            b64_string = ""
            

        log_message = None
        if info.get('glitch_alert'):
            log_message = '🚨 BUG FOUND: ' + info['glitch_alert']
        else:
            action_name = ACTION_NAMES.get(int(action_val), "Unknown")
            log_message = f"Step {step_count}: Action: {action_name} ({action_val}) | Reward: {float(reward):.2f}"
            
        yield {
            'frame': b64_string,
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
