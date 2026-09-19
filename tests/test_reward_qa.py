"""GlitchHunterWrapper shaping rules - QA EXPLORATION MODE.

The behaviours pinned here are the ones the retrofit exists to produce, and
every one of them is silent when it breaks. A reward bug does not raise; it
shows up eight hours into a training run as "the agent stopped moving", and
by then the 6M brain has already been overwritten by whatever the broken
signal taught it.

Most of these drive the wrapper with SYNTHETIC info dicts rather than the
live game. That is deliberate: it makes "walk over exactly the same pixels
twice" an exact, repeatable statement instead of an approximate one, so the
central claim - old ground is free, new ground pays - can be asserted rather
than estimated.
"""

import numpy as np
import pytest

from agent_logic import GlitchHunterWrapper
from exploration import config
from exploration.coverage import SpatialCoverage
from exploration.lifecycle import EpisodePhase

BASE = {
    'x_pos': 500, 'y_pos': 400, 'x_vel': 0.0, 'on_ground': True,
    'status': 'small', 'flag_get': False, 'powerup_active_count': 0,
    'nearest_powerup_dx': None, 'death_cause': None, 'is_dead': False,
    'score': 0, 'coins': 0, 'viewport_x': 0,
    'mario_rect': (500, 400, 30, 40),
}


def _make_qa(env, mask, patch_step):
    cov = SpatialCoverage(testable_mask=mask)
    w = GlitchHunterWrapper(env, reward_mode="qa_exploration", coverage=cov)
    w.reset()
    # These tests pin EXPLORATION-phase reward, and predate the lifecycle. On
    # a blank map T1 fires on the very first substep (a whole collider is
    # 1,200 new px against a 500 px floor target), and on the empty-mask
    # fixture T3 fires at the first window - either would silently move them
    # into COMPLETE. Phase-specific behaviour is tested, both phases, in
    # tests/test_phase_reward.py.
    w.lifecycle.auto_transition = False

    def drive(**overrides):
        info = dict(BASE, **overrides)
        patch_step(lambda a, _i=info: (None, 0.0, False, False, dict(_i)))
        return w.step(0)[1]

    w.drive = drive
    return w


@pytest.fixture
def qa(env, patch_step):
    """QA-mode wrapper over a permissive reachable mask.

    An all-True mask keeps the frontier and the remaining() denominator well
    defined without needing exploration_data/ to have been built, so these
    tests run on a fresh clone.
    """
    return _make_qa(env, np.ones((config.GRID_H, config.GRID_W), dtype=bool),
                    patch_step)


@pytest.fixture
def qa_flat(env, patch_step):
    """QA-mode wrapper with NO frontier, isolating the coverage terms.

    With an empty reachable mask there is no frontier, so Phi is identically
    zero and the frontier shaping contributes nothing. That matters for the
    tests below that assert an EXACT per-substep cost: with a live frontier,
    every step carries a small potential delta, and (because PBRS is
    policy-invariant under the DISCOUNTED objective, not the undiscounted
    one) even a stationary agent sees a residue of (gamma - 1) * Phi. That
    residue is real and is characterised in its own test below - but mixing
    it into these assertions would turn an exact claim about coverage into an
    approximate one.
    """
    return _make_qa(env, np.zeros((config.GRID_H, config.GRID_W), dtype=bool),
                    patch_step)


def _at(x, y=400, w=30, h=40):
    return {'x_pos': x, 'y_pos': y, 'mario_rect': (x, y, w, h)}


# ── Test 3 (full): old territory is transit space ─────────────────────────
def test_walking_new_ground_pays_and_rewalking_it_does_not(qa):
    """The single claim the whole retrofit rests on.

    previously explored territory = transit space
    new global world-space coverage = reward space
    """
    first = sum(qa.drive(**_at(1000 + i * 10)) for i in range(20))

    # Same pixels, same order, second time around.
    qa.coverage.begin_episode()
    qa.last_frontier_version = -1
    second = sum(qa.drive(**_at(1000 + i * 10)) for i in range(20))

    assert first > 0, "virgin ground earned nothing"
    assert second < first, "re-walking old ground paid the same as discovery"
    assert qa.coverage.episode_new == 0, "old pixels were counted as new"


def test_revisits_are_free_never_penalised(qa_flat):
    """Rule 7: zero novelty, NOT a negative one.

    A per-step cost for old ground would make backtracking to reach
    unexplored space self-defeating - the agent would learn to refuse to
    cross its own history, which is the exact opposite of the goal. In
    EXPLORE there is no time penalty either (config EXPLORE_TIME_PENALTY), so
    crossing explored ground is exactly free.
    """
    for i in range(20):
        qa_flat.drive(**_at(2000 + i * 10))
    qa_flat.coverage.begin_episode()

    # Well inside DROUGHT_GRACE, so nothing but the time penalty applies.
    rewards = [qa_flat.drive(**_at(2000 + i * 10)) for i in range(20)]
    for r in rewards:
        assert r == pytest.approx(-config.EXPLORE_TIME_PENALTY, abs=1e-6), (
            f"traversing explored ground cost {r:.4f}; it should be free")


# ── Test 4: the drought replaces the x-spread stuck detector ──────────────
def test_no_drought_pressure_inside_the_grace_window(qa_flat):
    """Standing still is free for a while - probing takes time."""
    qa_flat.drive(**_at(3000))
    for _ in range(config.DROUGHT_GRACE - 5):
        r = qa_flat.drive(**_at(3000))
    assert r == pytest.approx(-config.EXPLORE_TIME_PENALTY, abs=1e-6)


def test_drought_pressure_escalates_and_is_capped_per_substep(qa_flat):
    """Pressure builds only after the grace window, and stops at DROUGHT_MAX."""
    qa_flat.drive(**_at(3500))
    seen = [qa_flat.drive(**_at(3500))
            for _ in range(config.DROUGHT_GRACE + config.DROUGHT_STEP * 4)]

    inside_grace = seen[config.DROUGHT_GRACE // 2]
    first_notch = seen[config.DROUGHT_GRACE + 10]
    later_notch = seen[config.DROUGHT_GRACE + config.DROUGHT_STEP * 2 + 10]
    assert inside_grace > first_notch, "pressure started inside the grace window"
    assert first_notch > later_notch, "drought pressure never escalated"
    floor = -(config.DROUGHT_MAX + config.EXPLORE_TIME_PENALTY)
    assert min(seen) >= floor - 1e-6, "per-substep drought exceeded its ceiling"


def test_drought_is_bounded_per_episode_not_just_per_substep(qa_flat):
    """The defect the first calibration run exposed.

    The per-substep penalty was always bounded, but it was charged on every
    substep of a drought, so across an episode it was not bounded at all -
    measured QA returns of -685, against roughly +6 earned from discovery in
    the same episode. A term that can outweigh the entire objective by two
    orders of magnitude is the objective.
    """
    # Long enough for the cumulative cap to bind (~870 substeps), and well
    # short of anything that could end the episode.
    long_drought = 1595
    qa_flat.drive(**_at(3700))
    total = sum(qa_flat.drive(**_at(3700)) for _ in range(long_drought))
    drought_part = -(total + config.EXPLORE_TIME_PENALTY * long_drought)
    assert drought_part <= config.DROUGHT_EPISODE_CAP + 1e-6, (
        f"an episode paid {drought_part:.1f} in drought against a cap of "
        f"{config.DROUGHT_EPISODE_CAP}")
    assert qa_flat.ep_drought_paid == pytest.approx(drought_part, abs=1e-6)


def test_drought_ramp_completes_before_the_episode_cap_binds(qa_flat):
    """The ramp has to be able to express itself, or it is decoration.

    An earlier cap of 5.0 was exhausted after ~50 substeps - before the
    second notch of a ramp designed to reach full pressure over 320. The
    gradient existed in the code and never once appeared in the reward.
    """
    ramp_cost = config.DROUGHT_STEP * config.DROUGHT_NOTCH * (1 + 2 + 3 + 4)
    assert ramp_cost < config.DROUGHT_EPISODE_CAP, (
        f"reaching full drought pressure costs {ramp_cost:.1f}, but the "
        f"episode cap is {config.DROUGHT_EPISODE_CAP} - the ramp is dead code")


def test_drought_alone_no_longer_ends_the_episode(qa):
    """The retired DROUGHT_HARD_LIMIT used to end the episode here, at 1600
    substeps without a new pixel. A drought is not being stuck - see
    tests/test_episode_lifecycle.py for what does end an unproductive
    episode now, and why it needs more than elapsed time."""
    qa.drive(**_at(4000))
    for _ in range(1700):
        _obs, _r, done, _t, info = qa.step(0)
        assert not done, (
            f"a bare drought of {qa.coverage.steps_since_new_pixel} substeps "
            f"ended the episode")
    assert info['episode_phase'] in ('explore', 'complete')


def test_vertical_probing_is_not_punished(qa):
    """The behaviour the legacy x-spread detector got wrong.

    Jumping repeatedly in one column has almost no x-spread, so the legacy
    stuck detector punished it. It is also exactly how a QA explorer probes a
    suspicious ceiling. Here it keeps finding new pixels, so it stays free.
    """
    total = 0.0
    for i in range(30):
        total += qa.drive(**_at(5000, y=400 - i * 8))
    assert total > 0, "vertical probing that found new ground was punished"
    assert qa.coverage.steps_since_new_pixel == 0


# ── Test 12: magnitudes, and what is no longer the objective ──────────────
def test_completion_is_no_longer_the_objective(qa):
    """500.0 -> 5.0. The flagpole becomes ordinary."""
    qa.drive(**_at(6000))
    r = qa.drive(**dict(_at(6010), flag_get=True))
    assert r < 10.0, f"flag_get paid {r:.1f}; completion is still dominant"
    assert r > config.EXPLORE_FLAG_REWARD * 0.5, "flag_get paid nothing at all"


def test_completion_score_windfall_cannot_dominate(qa_flat):
    """The failure the second calibration run caught.

    This clone pays a large end-of-level time bonus straight into `score`.
    With the score term merely rescaled, a completing QA episode still
    returned ~350 against ~4 of novelty - so completion was still the
    objective, just hiding in a different variable. The per-episode
    interaction cap is what actually closes that.
    """
    qa_flat.drive(**_at(6500))
    total = 0.0
    # A 20,000-point end-of-level windfall, arriving over several frames.
    for i in range(10):
        total += qa_flat.drive(**dict(_at(6500), score=2000 * (i + 1)))
    assert total <= config.QA_INTERACTION_EPISODE_CAP + 1e-6, (
        f"a score windfall paid {total:.1f}, above the "
        f"{config.QA_INTERACTION_EPISODE_CAP} interaction ceiling")


def test_interaction_cap_does_not_cap_penalties(qa_flat):
    """Capping the downside would be a loophole, not a safeguard."""
    qa_flat.drive(**dict(_at(6600), status='tall'))
    for i in range(30):   # exhaust the positive budget
        qa_flat.drive(**dict(_at(6600), score=1000 * (i + 1), status='tall'))
    assert qa_flat.ep_interaction_paid == pytest.approx(
        config.QA_INTERACTION_EPISODE_CAP, abs=1e-6)
    # Shrinking back to small must still cost, budget or no budget.
    r = qa_flat.drive(**dict(_at(6600), status='small'))
    assert r < -1.0, "losing a powerup became free once the cap was reached"


def test_x_monotone_terms_are_gone(qa):
    """No +1.0/tile, no +25 milestone, no +0.1 per new max-x.

    Running right across ALREADY-EXPLORED ground must earn nothing. Under the
    legacy reward the same 40 substeps would have paid roughly +10 in tile
    bonuses alone, plus a +25 milestone whenever a 400px boundary was crossed.
    """
    for i in range(60):
        qa.drive(**_at(100 + i * 20))
    qa.coverage.begin_episode()

    total = sum(qa.drive(**_at(100 + i * 20)) for i in range(60))
    # Not literally <= 0 any more. This used to pass because the flat time
    # penalty buried the frontier-potential residue; EXPLORE has no time
    # penalty now (config PHASE-GATED REWARD), so the residue is visible. It
    # is bounded by FRONTIER_WEIGHT * (1 - GAMMA) per substep - 0.06 across
    # these 60 - where a surviving tile/milestone term would pay ~+10.
    residue_bound = 60 * config.FRONTIER_WEIGHT * (1.0 - config.GAMMA)
    assert total <= residue_bound, (
        f"crossing explored ground still pays {total:.4f} - an x-monotone "
        f"term survived the retrofit")


def test_frontier_residue_is_a_rounding_error_against_the_drought(qa):
    """Characterises the one place PBRS is not literally free.

    Phi is <= 0 and gamma < 1, so a stationary agent collects
    (gamma - 1) * Phi * FRONTIER_WEIGHT per substep - a small POSITIVE
    residue. It is not zero and this test does not pretend otherwise. What
    matters is that it is negligible against the pressure not to idle, so
    pacing can never become a better strategy than exploring.
    """
    qa.drive(**_at(9000))
    idle = qa.drive(**_at(9000))
    residue = idle + config.EXPLORE_TIME_PENALTY      # strip the flat time cost
    assert residue >= 0
    theoretical_max = config.FRONTIER_WEIGHT * (1.0 - config.GAMMA) * 1.0
    assert residue <= theoretical_max + 1e-9
    assert residue < config.DROUGHT_MAX / 50, (
        f"idling earns {residue:.5f}/substep against a {config.DROUGHT_MAX} "
        f"drought penalty - too close to farmable")


def test_novelty_dominates_frontier_guidance(qa):
    """Rule 10: approaching the frontier must never out-earn crossing it."""
    config.assert_reward_balance()
    whole_map_frontier = config.FRONTIER_WEIGHT * 1.0
    one_agent_step = 4 * qa.coverage.novelty_reward(int(config.N_REF))
    assert whole_map_frontier < one_agent_step


def test_reward_is_bounded_every_substep(qa):
    """Backstop: nothing reaches PPO's advantage estimator unbounded."""
    extremes = [
        dict(_at(7000), score=99999, coins=999),
        dict(_at(7020), flag_get=True, status='fireball'),
        dict(_at(7040), powerup_active_count=50),
        dict(_at(-200, y=-300)),
    ]
    for e in extremes:
        r = qa.drive(**e)
        assert abs(r) <= config.QA_REWARD_CLIP + 1e-9, f"unbounded reward {r}"


def test_out_of_world_position_is_not_credited_as_coverage(qa):
    """A glitch that puts Mario outside the padded grid must not pay novelty."""
    before = qa.coverage.total_unique()
    qa.drive(**_at(config.GRID_X0 - 9000))
    assert qa.coverage.oob_events >= 1
    assert qa.coverage.total_unique() == before


# ── mode plumbing ─────────────────────────────────────────────────────────
def test_unknown_mode_is_rejected_loudly(env):
    with pytest.raises(ValueError, match="unknown REWARD_MODE"):
        GlitchHunterWrapper(env, reward_mode="explore_everything")


def test_legacy_mode_attaches_no_coverage_by_default(env):
    """Falling back must not drag the retrofit's cost along with it."""
    w = GlitchHunterWrapper(env, reward_mode="legacy_completion")
    assert w.coverage is None


def test_legacy_mode_can_still_record_coverage(env, patch_step):
    """Which is what lets bootstrap replay the 6M brain under its own reward."""
    cov = SpatialCoverage()
    w = GlitchHunterWrapper(env, reward_mode="legacy_completion", coverage=cov)
    w.reset()
    info = dict(BASE, **_at(8000))
    patch_step(lambda a: (None, 0.0, False, False, dict(info)))
    w.step(0)
    assert cov.total_unique() == 30 * 40


def test_qa_mode_without_coverage_is_refused(env):
    with pytest.raises(ValueError, match="requires a coverage channel"):
        GlitchHunterWrapper(env, reward_mode="qa_exploration",
                            attach_coverage=False)


# ══════════════════════════════════════════════════════════════════════════
# REWARD INVERSION: interaction must not become a substitute for exploring
#
# The per-episode interaction ceiling is an absolute 10.0, but the novelty it
# was calibrated against ("roughly 36 per episode" - config) shrinks as the
# map fills. Between 6,032,768 and 6,400,000 steps the ratio inverted:
# novelty fell 12.32 -> 1.51 per episode while interaction held 6.19 -> 7.49
# and hit its ceiling in ~35% of episodes. Half of those episodes discovered
# nothing at all. The policy learned to linger next to interactable objects,
# median max-x fell 5,971 -> 1,172, and completion retention went from 43.8%
# to 6.0% (REGRESSED).
#
# The gate throttles interaction on exactly the condition the drought penalty
# already uses, so the two can never disagree about what "exploring" means.
# ══════════════════════════════════════════════════════════════════════════
def _drought(rig, substeps, score=0):
    """Stand still long enough to be past DROUGHT_GRACE on an old pixel.

    `score` is held at whatever the caller last used, because `score` in the
    info dict is ABSOLUTE and the reward reads its DELTA - letting it fall
    back to 0 here would hand the next call a much bigger delta than
    intended and quietly change what the test measures.
    """
    for _ in range(substeps):
        rig.drive(**dict(_at(6800), score=score))


def test_locomotion_is_throttled_once_the_episode_stops_discovering(qa_flat):
    """The other half of the farm, and the one that actually won.

    Locomotion is direction-agnostic on purpose, so jump-left-jump-right
    earns momentum credit forever while going nowhere. Measured on-policy
    over 10 bare-engine episodes, the 6.4M policy spent 66.6% of its actions
    on Jump (no direction) and 21.3% on Left+Jump, against 10.8% rightward,
    and its median max_x was 858 where the healthy 6.03M policy reached
    7,058. The episode cap of 10.0 did not stop that, because 10.0 is
    enormous against a median episode novelty of 1.00.
    """
    from rewards.qa import SPRINT_VEL_THRESHOLD
    fast = SPRINT_VEL_THRESHOLD + 1.0

    # The momentum payment ramps with sprint_frames, and standing still in
    # _drought() resets that counter - so both measurements are taken on the
    # FIRST fast substep after a standstill, or the ramp, not the gate, is
    # what the comparison would be measuring.
    qa_flat.drive(**_at(6800))
    before_fresh = qa_flat.ep_locomotion_paid
    qa_flat.drive(**dict(_at(6800), x_vel=fast))
    fresh = qa_flat.ep_locomotion_paid - before_fresh

    _drought(qa_flat, config.DROUGHT_GRACE + config.DROUGHT_STEP)
    before = qa_flat.ep_locomotion_paid
    qa_flat.drive(**dict(_at(6800), x_vel=fast))
    gated = qa_flat.ep_locomotion_paid - before

    assert fresh > 0.0, "momentum paid nothing before any drought"
    assert gated == pytest.approx(fresh * config.QA_SECONDARY_DROUGHT_SCALE,
                                  rel=1e-6), (
        f"a drought-bound sprint substep paid {gated:.8f}; the same substep "
        f"while discovering paid {fresh:.8f}, so the gate must pay "
        f"{fresh * config.QA_SECONDARY_DROUGHT_SCALE:.8f}")


def test_both_secondary_channels_read_one_shared_gate(qa_flat):
    """Interaction and locomotion must never disagree about 'exploring'."""
    qa_flat.drive(**_at(6800))
    assert qa_flat.secondary_gated is False, \
        "the gate was already active on a fresh pixel"
    _drought(qa_flat, config.DROUGHT_GRACE + config.DROUGHT_STEP)
    assert qa_flat.secondary_gated is True, \
        "the gate did not engage after the drought grace elapsed"


def test_a_fresh_episode_does_not_inherit_the_previous_gate(qa_flat):
    """reset() must clear the flag, or substep 1 is scored by episode 0."""
    _drought(qa_flat, config.DROUGHT_GRACE + config.DROUGHT_STEP)
    assert qa_flat.secondary_gated is True
    qa_flat.reset()
    assert qa_flat.secondary_gated is False


def test_interaction_is_throttled_once_the_episode_stops_discovering(qa_flat):
    """Lingering next to a scoring object must stop being worth much."""
    # Small deltas on purpose: big ones hit the 10.0 episode ceiling, and
    # then the CAP is what bound the payment, not the gate under test.
    qa_flat.drive(**_at(6800))
    fresh = qa_flat.drive(**dict(_at(6800), score=100))
    _drought(qa_flat, config.DROUGHT_GRACE + config.DROUGHT_STEP, score=100)
    before = qa_flat.ep_interaction_paid
    qa_flat.drive(**dict(_at(6800), score=200))     # the same +100 delta
    gated = qa_flat.ep_interaction_paid - before

    full_pay = 100 * config.QA_SCORE_SCALE
    assert fresh == pytest.approx(full_pay, abs=1e-6), \
        "the first interaction, before any drought, was not paid in full"
    assert gated == pytest.approx(full_pay * config.QA_SECONDARY_DROUGHT_SCALE,
                                  abs=1e-6), (
        f"a drought-bound interaction paid {gated:.4f}; it must be scaled by "
        f"{config.QA_SECONDARY_DROUGHT_SCALE}")


def test_interaction_still_pays_in_full_while_discovering(qa):
    """Genuinely useful interaction learning is preserved.

    The gate is about lingering, not about interacting. An agent picking up
    a powerup while covering new ground is doing both things right and must
    be paid for both.
    """
    qa.drive(**_at(1000))
    before = qa.ep_interaction_paid
    qa.drive(**dict(_at(1100), score=100))     # new ground AND a score event
    paid = qa.ep_interaction_paid - before
    assert qa.coverage.steps_since_new_pixel == 0, "this step found no new ground"
    assert paid == pytest.approx(100 * config.QA_SCORE_SCALE, abs=1e-6), \
        "interaction was throttled on a step that discovered new pixels"


def test_the_gate_never_scales_a_penalty(qa_flat):
    """Same rule the cap follows: the downside is never softened."""
    qa_flat.drive(**dict(_at(6900), status='tall'))
    # status is held across the drought, so the tall -> small transition
    # happens ONCE, on the measured step, and not silently inside the loop.
    for _ in range(config.DROUGHT_GRACE + config.DROUGHT_STEP):
        qa_flat.drive(**dict(_at(6900), status='tall'))
    r = qa_flat.drive(**dict(_at(6900), status='small'))
    assert r < -1.0, (
        "losing a powerup during a drought became cheap - the gate must only "
        "ever throttle positive interaction")


def test_the_gate_is_off_in_the_complete_phase(qa_flat):
    """Crossing old ground is COMPLETE's whole job, so it is not a drought."""
    qa_flat.lifecycle.phase = EpisodePhase.COMPLETE
    qa_flat.lifecycle.completion_credit = 1.0
    _drought(qa_flat, config.DROUGHT_GRACE + config.DROUGHT_STEP)
    before = qa_flat.ep_interaction_paid
    qa_flat.drive(**dict(_at(6800), score=100))
    paid = qa_flat.ep_interaction_paid - before
    assert paid == pytest.approx(100 * config.QA_SCORE_SCALE, abs=1e-6), \
        "interaction was throttled in COMPLETE, where old ground is the route"


def test_the_gate_is_off_during_coherent_transit(qa_flat):
    """Transit across old territory toward a frontier is not lingering."""
    _drought(qa_flat, config.DROUGHT_GRACE + config.DROUGHT_STEP)
    # in_coherent_transit is derived from the last COMPLETED window, so it is
    # set the way the lifecycle sets it rather than assigned to.
    qa_flat.lifecycle.last_window = {'in_transit': True, 'cells_ahead': 1,
                                     'new_px': 0}
    assert qa_flat.lifecycle.in_coherent_transit
    before = qa_flat.ep_interaction_paid
    qa_flat.drive(**dict(_at(6800), score=100))
    paid = qa_flat.ep_interaction_paid - before
    assert paid == pytest.approx(100 * config.QA_SCORE_SCALE, abs=1e-6), \
        "interaction was throttled during coherent transit"


def test_a_zero_yield_episode_cannot_be_farmed_into_profit(qa_flat):
    """The exploit that cost the policy the level, end to end.

    Stand on one already-covered pixel and take the biggest score windfall
    the engine can produce. 36 of the 129 recorded zero-yield episodes ended
    net POSITIVE before the gate; replaying them through it leaves 1.

    Note what is NOT claimed: the gate is a scale, not a second ceiling, so a
    large enough windfall still reaches the +10 interaction cap eventually -
    it just needs 10x the score to get there. The property that matters is
    that the drought outruns it, so the EPISODE is a loss. Turning the cap
    itself down during a drought was considered and rejected as the larger
    change: it would also retroactively shrink interaction an episode had
    already legitimately earned before the drought began.
    """
    # Long enough for the per-episode drought cap to bind - the same 1,595
    # substeps the drought-cap test above uses. A shorter run only measures
    # the ramp, not the steady state a real stuck episode reaches.
    total = qa_flat.drive(**_at(6800))
    for i in range(1595):
        total += qa_flat.drive(**dict(_at(6800), score=2000 * (i + 1)))
    assert qa_flat.coverage.episode_new == 0, "this episode discovered pixels"
    assert total < 0.0, (
        f"standing still and farming score returned {total:+.2f} - it has to "
        f"be a losing strategy while unexplored pixels remain")
    assert qa_flat.ep_drought_paid > qa_flat.ep_interaction_paid, (
        f"drought {qa_flat.ep_drought_paid:.2f} did not outrun interaction "
        f"{qa_flat.ep_interaction_paid:.2f} - lingering is still profitable")
