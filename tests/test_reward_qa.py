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
    cross its own history, which is the exact opposite of the goal. What
    little cost there is here must come only from the time penalty, which is
    the same everywhere and is not a function of coverage.
    """
    for i in range(20):
        qa_flat.drive(**_at(2000 + i * 10))
    qa_flat.coverage.begin_episode()

    # Well inside DROUGHT_GRACE, so nothing but the time penalty applies.
    rewards = [qa_flat.drive(**_at(2000 + i * 10)) for i in range(20)]
    for r in rewards:
        assert r == pytest.approx(-config.QA_TIME_PENALTY, abs=1e-6), (
            f"traversing explored ground cost {r:.4f}, expected only the "
            f"flat time penalty")


# ── Test 4: the drought replaces the x-spread stuck detector ──────────────
def test_no_drought_pressure_inside_the_grace_window(qa_flat):
    """Standing still is free for a while - probing takes time."""
    qa_flat.drive(**_at(3000))
    for _ in range(config.DROUGHT_GRACE - 5):
        r = qa_flat.drive(**_at(3000))
    assert r == pytest.approx(-config.QA_TIME_PENALTY, abs=1e-6)


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
    floor = -(config.DROUGHT_MAX + config.QA_TIME_PENALTY)
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
    drought_part = -(total + config.QA_TIME_PENALTY * long_drought)
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
    assert r > config.QA_FLAG_GET_REWARD * 0.5, "flag_get paid nothing at all"


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
    assert total <= 0, (
        f"crossing explored ground still pays {total:.2f} - an x-monotone "
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
    residue = idle + config.QA_TIME_PENALTY      # strip the flat time cost
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
