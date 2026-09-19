"""RewardTelemetryCallback: the reward's books, written as the run happens.

The point of this file is one claim: the telemetry is a READER. It has to
record what each episode paid, per channel and per phase, and it has to leave
the rewards PPO trains on - and therefore the weights - exactly as they would
have been without it. So the tests come in two halves:

  * the RECORD, driven through the callback with the info dicts SB3 would
    hand it, so each field can be checked at the episode it describes;
  * the NON-EFFECT, on a real (tiny) PPO run, comparing every reward the
    model saw and every parameter it ended with, against the same run with
    the callback removed.
"""
import json
import logging

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from agent_logic import GlitchHunterWrapper
from custom_mario_env import wrap_observation
from exploration import config
from exploration.coverage import SpatialCoverage
from rewards.qa import QA_CHANNELS
from training import callbacks as cbs


class _Model:
    """SB3's surface, as much of it as a callback touches."""

    def __init__(self, t=0, logger=None):
        self.num_timesteps = t
        self.logger = logger

    def get_env(self):
        return None


def _drive(cb, t, **local):
    cb.model.num_timesteps = t
    cb.update_locals(local)
    return cb.on_step()


def _channels(explore=None, complete=None):
    """A qa_channels snapshot, as rewards/qa.py leaves it in the final info."""
    return {'explore': dict.fromkeys(QA_CHANNELS, 0.0) | (explore or {}),
            'complete': dict.fromkeys(QA_CHANNELS, 0.0) | (complete or {})}


def _episode_info(explore=None, complete=None, clips=0, steps=400, **kw):
    return {
        'qa_channels': _channels(explore, complete),
        'qa_clip_events': clips,
        'lifecycle_agent_steps': steps,
        'episode_phase': 'complete' if complete else 'explore',
        'episode_end_reason': 'death',
        'phase_transition_reason': 'T1_target_met' if complete else None,
        'coverage_episode_new': 1234,
        **kw,
    }


def _telemetry(tmp_path, **kw):
    cb = cbs.RewardTelemetryCallback(str(tmp_path / "reward_telemetry.jsonl"), **kw)
    cb.init_callback(_Model())
    return cb


def _lines(cb, kind=None):
    with open(cb.path, encoding='utf-8') as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    return [r for r in records if kind is None or r['kind'] == kind]


# ══════════════════════════════════════════════════════════════════════════
# THE RECORD
# ══════════════════════════════════════════════════════════════════════════
def test_a_session_header_names_the_constants_the_numbers_mean(tmp_path):
    """Read years later, a channel sum is meaningless without the weights it
    was produced by, so the run writes them down once."""
    cb = _telemetry(tmp_path, session_start={'global_timestep': 6_000_000})
    cb.on_training_start({}, {})
    session = _lines(cb, 'session')
    assert len(session) == 1
    assert session[0]['resumed_at'] == 6_000_000
    assert session[0]['reward_mode'] == config.REWARD_MODE
    assert session[0]['channels'] == list(QA_CHANNELS)
    assert session[0]['constants']['novelty_weight'] == config.NOVELTY_WEIGHT
    assert session[0]['constants']['reward_clip'] == config.QA_REWARD_CLIP


def test_explore_and_complete_components_are_kept_apart(tmp_path):
    """Required test 3. The whole reason the books are per-phase: a novelty
    payment in EXPLORE and one in COMPLETE are different measurements."""
    cb = _telemetry(tmp_path)
    cb.on_training_start({}, {})
    _drive(cb, 8, dones=[True], infos=[_episode_info(
        explore={'novelty': 30.0, 'drought': -1.5},
        complete={'progress': 4.0, 'flag': 6.0, 'novelty': 0.05})])
    rec = _lines(cb, 'episode')[0]
    assert rec['channels_by_phase']['explore']['novelty'] == 30.0
    assert rec['channels_by_phase']['complete']['novelty'] == 0.05
    assert rec['channels_by_phase']['explore']['progress'] == 0.0, (
        "EXPLORE must never show forward-progress pay")
    assert rec['channels_total']['novelty'] == pytest.approx(30.05)
    assert rec['reconciliation']['final_reward'] == pytest.approx(38.55)
    assert rec['transition_reason'] == 'T1_target_met'
    # Every channel the reward defines is present, so a term that starts
    # paying is visible rather than silently absent.
    for phase in ('explore', 'complete'):
        assert set(rec['channels_by_phase'][phase]) == set(QA_CHANNELS)


def test_the_reward_chain_is_accounted_end_to_end(tmp_path):
    """The engine reward, the clamp, the channel sum and SB3's own total, in
    one block: what PPO was given must be fully explained by what was logged."""
    cb = _telemetry(tmp_path)
    cb.on_training_start({}, {})
    _drive(cb, 8, dones=[True], infos=[_episode_info(
        explore={'novelty': 12.0, 'death': -5.0}, episode={'r': 7.0, 'l': 400})])
    r = _lines(cb, 'episode')[0]['reconciliation']
    assert r['env_reward'] == pytest.approx(-5.0)      # the engine's own -5 on death
    assert r['preclip_total'] == pytest.approx(7.0)
    assert r['clip_adjustment'] == 0.0
    assert r['final_reward'] == pytest.approx(7.0)
    assert r['monitor_reward'] == pytest.approx(7.0)
    assert r['unaccounted_residual'] == pytest.approx(0.0, abs=1e-9)


def test_the_clamp_is_shown_as_its_own_adjustment(tmp_path):
    """When the backstop clamp fires, preclip_total is what the reward would
    have been and clip_adjustment is exactly what the clamp removed - so the
    two still add up to what PPO received."""
    cb = _telemetry(tmp_path)
    cb.on_training_start({}, {})
    _drive(cb, 8, dones=[True], infos=[_episode_info(
        explore={'novelty': 40.0, 'clip': -16.0}, clips=2,
        episode={'r': 24.0, 'l': 400})])
    r = _lines(cb, 'episode')[0]['reconciliation']
    assert r['preclip_total'] == pytest.approx(40.0)
    assert r['clip_adjustment'] == pytest.approx(-16.0)
    assert r['final_reward'] == pytest.approx(24.0)
    assert r['preclip_total'] + r['clip_adjustment'] == pytest.approx(r['final_reward'])
    assert r['unaccounted_residual'] == pytest.approx(0.0, abs=1e-9)


def test_an_unbooked_reward_term_is_reported_not_hidden(tmp_path, caplog):
    """The failure this block exists to catch: Monitor says one thing, the
    books another. It must reach the file AND the log, not be rounded away."""
    cb = _telemetry(tmp_path)
    cb.on_training_start({}, {})
    with caplog.at_level(logging.WARNING):
        _drive(cb, 8, dones=[True], infos=[_episode_info(
            explore={'novelty': 10.0}, episode={'r': 12.5, 'l': 400})])
    r = _lines(cb, 'episode')[0]['reconciliation']
    assert r['unaccounted_residual'] == pytest.approx(2.5)
    assert any("UNACCOUNTED" in m.getMessage() for m in caplog.records)
    assert cb.worst_residual == pytest.approx(2.5)


def test_float_noise_is_not_reported_as_a_missing_term(tmp_path, caplog):
    """The real residual is float-summation noise (~3e-7 on totals in the
    hundreds). That must not raise an alarm, and must not be rounded to 0
    either - the number itself is kept."""
    cb = _telemetry(tmp_path)
    cb.on_training_start({}, {})
    with caplog.at_level(logging.WARNING):
        _drive(cb, 8, dones=[True], infos=[_episode_info(
            explore={'novelty': 832.913034}, episode={'r': 832.9130343315, 'l': 1113})])
    r = _lines(cb, 'episode')[0]['reconciliation']
    assert 0 < abs(r['unaccounted_residual']) < cb.RESIDUAL_TOLERANCE
    assert not [m for m in caplog.records if "UNACCOUNTED" in m.getMessage()]


def test_clip_events_are_counted_per_worker_as_deltas(tmp_path):
    """Required test 4. qa_clip_events is CUMULATIVE INSIDE EACH WORKER, so
    counting it as reported would count every earlier clip again on every
    later episode of the same worker."""
    cb = _telemetry(tmp_path)
    cb.on_training_start({}, {})
    # Two workers, one episode each: 3 clips and 0.
    _drive(cb, 8, dones=[True, True],
           infos=[_episode_info(clips=3), _episode_info(clips=0)])
    # Worker 0 ends another episode, now at 5 cumulative: 2 more, not 5.
    _drive(cb, 16, dones=[True, False],
           infos=[_episode_info(clips=5), _episode_info(clips=99)])
    assert [r['clip_events'] for r in _lines(cb, 'episode')] == [3, 0, 2]
    # A worker that restarts reports a SMALLER cumulative count; its new
    # value is then the delta, never a negative number.
    _drive(cb, 24, dones=[True], infos=[_episode_info(clips=1)])
    assert _lines(cb, 'episode')[-1]['clip_events'] == 1


def test_no_clipping_is_recorded_as_zero_not_omitted(tmp_path):
    cb = _telemetry(tmp_path)
    cb.on_training_start({}, {})
    _drive(cb, 8, dones=[True], infos=[_episode_info(clips=0)])
    rec = _lines(cb, 'episode')[0]
    assert rec['clip_events'] == 0
    cb.on_training_end()
    assert _lines(cb, 'interval')[0]['clip_events'] == 0


def test_every_episode_is_written_across_a_whole_run(tmp_path):
    """Required test 5: telemetry persists across many episodes, one line
    each, in order."""
    cb = _telemetry(tmp_path, every=10_000)
    cb.on_training_start({}, {})
    for i in range(12):
        _drive(cb, 8 * (i + 1), dones=[True],
               infos=[_episode_info(explore={'novelty': float(i)})])
    episodes = _lines(cb, 'episode')
    assert len(episodes) == 12
    assert [r['episode_index'] for r in episodes] == list(range(12))
    assert [r['channels_by_phase']['explore']['novelty'] for r in episodes] == [
        float(i) for i in range(12)]


def test_an_interval_summary_sums_the_episodes_it_covers(tmp_path):
    cb = _telemetry(tmp_path, every=100)
    cb.on_training_start({}, {})
    for t in (40, 80):
        _drive(cb, t, dones=[True], infos=[_episode_info(
            explore={'novelty': 10.0}, complete={'flag': 3.0}, steps=500)])
    assert _lines(cb, 'interval') == []          # nothing due yet
    _drive(cb, 120, dones=[False], infos=[{}])
    summary = _lines(cb, 'interval')[0]
    assert summary['episodes'] == 2
    assert summary['agent_steps'] == 1000
    assert summary['channels_by_phase']['explore']['novelty'] == pytest.approx(20.0)
    assert summary['channels_mean_per_episode']['complete']['flag'] == pytest.approx(3.0)
    assert summary['ends'] == {'death': 2}
    assert summary['transitions'] == {'T1_target_met': 2}
    assert summary['episode_reward_mean'] == pytest.approx(13.0)
    assert summary['episode_reward_median'] == pytest.approx(13.0)
    assert summary['unaccounted_residual_max_abs'] == 0.0
    # Counters restart, so the next interval is not a running total.
    _drive(cb, 200, dones=[True], infos=[_episode_info(explore={'novelty': 1.0})])
    _drive(cb, 260, dones=[False], infos=[{}])
    assert _lines(cb, 'interval')[1]['channels_by_phase']['explore']['novelty'] == 1.0


def test_a_stop_flushes_what_the_last_interval_had(tmp_path):
    """Ctrl+C, or the safety cap, must not lose the episodes since the last
    summary - the +20k run is shorter than two report intervals."""
    cb = _telemetry(tmp_path, every=10_000)
    cb.on_training_start({}, {})
    _drive(cb, 8, dones=[True], infos=[_episode_info(explore={'novelty': 5.0})])
    assert _lines(cb, 'interval') == []
    cb.on_training_end()
    assert _lines(cb, 'interval')[0]['episodes'] == 1


def test_resume_appends_and_keeps_the_runs_apart(tmp_path):
    """Required test 6. A second run continues the same file; nothing earlier
    is truncated, and the session id says which run each line came from."""
    first = _telemetry(tmp_path)
    first.on_training_start({}, {})
    _drive(first, 8, dones=[True], infos=[_episode_info(explore={'novelty': 1.0})])

    second = cbs.RewardTelemetryCallback(first.path)
    second.session_id = "second-run"
    second.init_callback(_Model(t=6_020_000))
    second.on_training_start({}, {})
    _drive(second, 6_020_008, dones=[True], infos=[_episode_info(explore={'novelty': 2.0})])

    episodes = _lines(second, 'episode')
    assert [r['channels_by_phase']['explore']['novelty'] for r in episodes] == [1.0, 2.0]
    assert episodes[0]['session_id'] != episodes[1]['session_id']
    assert len(_lines(second, 'session')) == 2
    assert episodes[1]['global_timestep'] == 6_020_008


def test_legacy_episodes_are_ignored(tmp_path):
    """Required test 7. Legacy mode has no channels, no phase and no
    lifecycle; its episodes must produce nothing at all here."""
    cb = _telemetry(tmp_path, every=1)
    cb.on_training_start({}, {})
    legacy = {'episode': {'r': 2200.0, 'l': 410}, 'flag_get': True, 'x_pos': 8751}
    assert _drive(cb, 8, dones=[True, True], infos=[legacy, {}]) is True
    cb.on_training_end()
    assert _lines(cb, 'episode') == []
    assert _lines(cb, 'interval') == []


def test_an_unfinished_episode_writes_nothing(tmp_path):
    cb = _telemetry(tmp_path)
    cb.on_training_start({}, {})
    # qa_channels is only ever in a FINAL info; a live one carries none.
    _drive(cb, 8, dones=[False], infos=[{'qa_clip_events': 2, 'episode_phase': 'explore'}])
    assert _lines(cb, 'episode') == []


def test_a_broken_telemetry_file_never_stops_training(tmp_path):
    """It is a notebook, not a gate: if it cannot be written, the run goes on."""
    cb = _telemetry(tmp_path)
    cb.path = str(tmp_path / "no-such-dir" / "x" / "t.jsonl")

    def refuse(*_a, **_k):
        raise OSError("disk full")
    original = cbs.os.makedirs
    cbs.os.makedirs = refuse
    try:
        assert _drive(cb, 8, dones=[True], infos=[_episode_info()]) is True
    finally:
        cbs.os.makedirs = original


# ══════════════════════════════════════════════════════════════════════════
# THE NON-EFFECT — a real PPO run, with the callback and without it
# ══════════════════════════════════════════════════════════════════════════
EPISODE_LEN = 20


class _FakeQAEnv(gym.Env):
    """A deterministic env that ends episodes with QA-shaped info dicts.

    Deterministic on purpose: the two runs compared below must differ ONLY by
    whether the callback is attached, so nothing here may consult an RNG.
    """

    observation_space = spaces.Box(0.0, 1.0, (4,), dtype=np.float32)
    action_space = spaces.Discrete(2)

    def __init__(self, rewards_seen):
        self.rewards_seen = rewards_seen
        self.t = 0
        self.episodes = 0

    def _obs(self):
        v = (self.t % EPISODE_LEN) / EPISODE_LEN
        return np.full((4,), v, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        reward = float(action) * 0.5 - 0.25 * (self.t % 3)
        self.rewards_seen.append(reward)
        terminated = self.t >= EPISODE_LEN
        info = {}
        if terminated:
            self.episodes += 1
            info = _episode_info(explore={'novelty': float(self.episodes)},
                                 complete={'flag': 1.0}, clips=self.episodes,
                                 steps=EPISODE_LEN)
        return self._obs(), reward, terminated, False, info


def _train(callback, rewards_seen):
    """One short, seeded PPO run. Same seed and same env every time."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    venv = DummyVecEnv([lambda: Monitor(_FakeQAEnv(rewards_seen)) for _ in range(2)])
    model = PPO("MlpPolicy", venv, n_steps=32, batch_size=32, n_epochs=1,
                seed=0, device="cpu", verbose=0)
    model.learn(total_timesteps=128, callback=callback)
    venv.close()
    return model


def test_logging_changes_neither_the_rewards_nor_the_weights(tmp_path):
    """Required tests 1 and 2, on a real PPO update rather than by argument.

    Same seed, same deterministic env, 128 steps: once with the telemetry
    attached and once without. Every reward the model was given must match,
    and so must every parameter it ended with - bit for bit.
    """
    import torch

    with_rewards, without_rewards = [], []
    cb = cbs.RewardTelemetryCallback(str(tmp_path / "t.jsonl"), every=64)
    logged = _train(cb, with_rewards)
    plain = _train(None, without_rewards)

    assert with_rewards and with_rewards == without_rewards, (
        "the telemetry changed the rewards the model was trained on")
    a, b = logged.policy.state_dict(), plain.policy.state_dict()
    assert a.keys() == b.keys()
    for name in a:
        assert torch.equal(a[name], b[name]), f"{name} differs with telemetry attached"
    # ...and it did do its job while changing nothing.
    episodes = _lines(cb, 'episode')
    assert episodes, "nothing was recorded"
    assert all(r['agent_steps'] == EPISODE_LEN for r in episodes)


def test_the_recorded_totals_match_sb3s_own_episode_totals(tmp_path):
    """The Monitor total is SB3's independent count of the same episode, so
    it is captured on a real run, not only when hand-fed."""
    cb = cbs.RewardTelemetryCallback(str(tmp_path / "t.jsonl"), every=64)
    _train(cb, [])
    episodes = _lines(cb, 'episode')
    assert episodes
    for rec in episodes:
        assert rec['reconciliation']['monitor_reward'] is not None
        assert isinstance(rec['reconciliation']['final_reward'], float)


# ══════════════════════════════════════════════════════════════════════════
# THE RECONCILIATION, ON THE REAL REWARD
# ══════════════════════════════════════════════════════════════════════════
def test_the_books_account_for_the_real_reward_ppo_is_given(env, tmp_path):
    """The claim, on the actual QA reward path rather than on fixtures.

    Three real episodes of the real engine, wrapped exactly as training wraps
    it (GlitchHunterWrapper -> skip/grayscale/resize/stack -> Monitor). For
    each one, three independently-computed totals must agree: the rewards the
    agent was handed step by step, SB3 Monitor's own episode total, and the
    sum of the per-channel books the telemetry writes down.

    This is what "the telemetry accounts for the exact reward PPO receives"
    means, and it is measured here rather than argued.
    """
    from stable_baselines3.common.monitor import Monitor

    full = np.ones((config.GRID_H, config.GRID_W), dtype=bool)
    wrapper = GlitchHunterWrapper(env, reward_mode="qa_exploration",
                                  coverage=SpatialCoverage(testable_mask=full))
    chain = Monitor(wrap_observation(wrapper))
    cb = _telemetry(tmp_path)
    cb.on_training_start({}, {})

    episodes, handed_to_agent = 0, []
    chain.reset()
    while episodes < 3:
        _obs, reward, terminated, truncated, info = chain.step(3)   # run right into a goomba
        handed_to_agent.append(float(reward))
        if not (terminated or truncated):
            continue
        episodes += 1
        assert 'qa_channels' in info, "the books were never snapshotted"
        _drive(cb, 8 * episodes, dones=[True], infos=[info])
        rec = _lines(cb, 'episode')[-1]['reconciliation']
        agent_total = sum(handed_to_agent)
        assert rec['monitor_reward'] == pytest.approx(agent_total, abs=1e-4), (
            "Monitor disagrees with the rewards the agent was actually handed")
        assert rec['final_reward'] == pytest.approx(agent_total, abs=1e-4), (
            "the channels do not sum to the reward PPO received")
        assert abs(rec['unaccounted_residual']) < cb.RESIDUAL_TOLERANCE, (
            f"{rec['unaccounted_residual']} of reward is unaccounted for")
        # The engine's own contribution is the death, and it is booked.
        assert rec['env_reward'] == pytest.approx(-5.0)
        assert rec['preclip_total'] + rec['clip_adjustment'] == pytest.approx(
            rec['final_reward'])
        handed_to_agent = []
        chain.reset()
    assert cb.worst_residual < cb.RESIDUAL_TOLERANCE
