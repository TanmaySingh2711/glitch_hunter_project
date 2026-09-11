"""State and shaping constants both reward modes share."""
from __future__ import annotations

from typing import Any, TypedDict

# The per-substep info dict CustomMarioEnv.step() returns and the wrapper
# annotates. Heterogeneous by nature (ints, rects, strings, None), so it is
# typed as a plain mapping rather than a TypedDict every producer must match.
Info = dict[str, Any]

SPRINT_VEL_THRESHOLD = 4.5      # Matches the engine's own "fast jump" x_vel cutoff
CLEAN_JUMP_MIN_PX = 60          # horizontal distance that makes a landing a "clean clear"
SPRINT_RAMP_FRAMES = 30         # consecutive sprint frames to reach full momentum reward

POWERUP_PROXIMITY_CAP_PX = 300  # Distance beyond which proximity shaping saturates
POWERUP_PULL_SCALE_NEEDED = 0.05    # Shaping strength while Mario is small
POWERUP_PULL_SCALE_OPTIONAL = 0.015  # Shaping strength while already powered up

ENEMY_CAUSES = ('goomba', 'koopa', 'koopa_shell')

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
#                       and (legacy) the stuck-penalty is relaxed there (more
#                       patience to actually line up a good running jump
#                       instead of being time-pressured into repeating the
#                       same mistake)
#   - enemy deaths  -> an extra "attempt a jump here" bonus is added in that
#                       zone, nudging toward stomping instead of walking in
#   - timeout       -> (legacy) the time penalty is halved in that zone
#
# Both modes track deaths the same way; the boosts marked (legacy) exist only
# in the legacy reward, because the QA reward has no stuck penalty, no flat
# time penalty in EXPLORE, and no engine timeout (the respawn rule).
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


class DangerZone(TypedDict):
    cause: str
    episodes_left: int


def danger_bucket(x_pos: float) -> int:
    """The DANGER_ZONE_BUCKET_PX-wide column of the level `x_pos` falls in."""
    return int(x_pos) // DANGER_ZONE_BUCKET_PX


def powerup_potential(nearest_dx: float | None) -> float:
    """Phi for the potential-based pull toward the nearest active powerup:
    negative, capped distance, 0 when nothing is out (see the legacy reward's
    POTENTIAL-BASED SHAPING note for why this form cannot be farmed)."""
    if nearest_dx is None:
        return 0.0
    return -min(abs(nearest_dx), POWERUP_PROXIMITY_CAP_PX) / POWERUP_PROXIMITY_CAP_PX


class RewardState:
    """State both reward modes read and write. Declared here, set by
    _reset_shared_state (per episode) and _init_death_memory (once)."""

    # locomotion
    sprint_frames: int                  # consecutive frames spent sprinting on ground
    jump_start_x: float | None          # x_pos when the current airborne phase began
    jump_start_had_momentum: bool
    was_on_ground: bool
    # world interaction
    last_score: int
    last_coins: int
    last_status: str
    last_flag_get: bool
    last_x_pos: float | None
    last_powerup_count: int
    last_powerup_phi: float
    # adaptive death memory - persists ACROSS episodes on purpose
    death_streaks: dict[tuple[int, str], int]   # (x_bucket, cause) -> consecutive count
    last_death_key: tuple[int, str] | None      # the (x_bucket, cause) of the previous death
    danger_zones: dict[int, DangerZone]         # x_bucket -> the active boost there

    def _reset_shared_state(self) -> None:
        self.sprint_frames = 0
        self.jump_start_x = None
        self.jump_start_had_momentum = False
        self.was_on_ground = True
        self.last_score = 0
        self.last_coins = 0
        self.last_status = 'small'
        self.last_flag_get = False
        self.last_x_pos = None
        self.last_powerup_count = 0
        self.last_powerup_phi = 0.0

    def _init_death_memory(self) -> None:
        self.death_streaks = {}
        self.last_death_key = None
        self.danger_zones = {}

    def _decay_danger_zones(self) -> None:
        """Called once per episode reset: every active boost loses an
        episode, and expired ones are dropped. (death_streaks and danger_zones
        themselves are NOT cleared - they persist across episodes.)"""
        expired = [b for b, z in self.danger_zones.items() if z['episodes_left'] <= 1]
        for b in expired:
            del self.danger_zones[b]
        for zone in self.danger_zones.values():
            zone['episodes_left'] -= 1

    def _track_death_memory(self, done: bool, info: Info, current_bucket: int) -> None:
        """Records a death and escalates a repeated one into a danger zone.

        Runs whenever the episode is ending because Mario actually died
        (death_cause is set — this excludes a stuck or safety-reset ending,
        which isn't a death and shouldn't count toward this). Shared by both
        modes - see ADAPTIVE DEATH MEMORY above.
        """
        death_cause = info.get('death_cause', None)
        if not (done and death_cause):
            return
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
