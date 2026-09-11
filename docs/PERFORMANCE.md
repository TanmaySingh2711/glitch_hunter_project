# Performance

Where time and memory go in one training worker, what was done about it,
and how to re-measure. All numbers: this machine (16 logical cores,
Windows 11, Python 3.12), one environment, headless SDL, random actions.

## Time per agent step

`python tools/benchmark_step.py --steps 1500`:

| stack | ms / agent step | agent steps / s | vs bare engine |
|---|---|---|---|
| bare `CustomMarioEnv` + observation chain | 7.43 | 134.6 | 1.00x |
| + `GlitchHunterWrapper`, legacy reward | 7.07 | 141.4 | 0.95x |
| + `GlitchHunterWrapper`, QA reward (coverage, lifecycle) | 7.59 | 131.7 | 1.02x |

The legacy row beats the bare engine only because its stuck rule ends
episodes early, and a level near the spawn is cheaper to update. The
project's whole per-step layer - coverage, lifecycle and QA reward - costs
about 2% on top of the engine.

`--profile` shows where the rest goes (8,000 engine frames, QA stack):

| | share |
|---|---|
| engine `Level1.update` (vendored, not modified) | ~60% |
| of which `Surface.blit` - the engine redrawing the level | ~32% |
| frame capture + downscale (`_fast_obs`) | ~16% before, ~8% after the change below |
| Gymnasium wrappers (max-pool, grayscale, resize, stack) | ~15% |
| `GlitchHunterWrapper` (coverage write, lifecycle, reward) | ~5% |

### Changes, and what pins them

| Change | Where | Effect | Guard |
|---|---|---|---|
| Render only the two frames the max-pool keeps | `custom_mario_env.SkipObservation` | 8.85 -> 7.68 ms per agent step: -13% time, +15% rollout throughput | `tests/test_env.py`: identical to `MaxAndSkipObservation`, frame for frame |
| Downscale before copying to numpy | `CustomMarioEnv._fast_obs` | 5.6 -> 0.47 ms per capture; byte-identical to the old cv2 path (checked once, when made) | observation shape: `tests/test_env.py` |
| No `chdir` in the engine step | `CustomMarioEnv.step` | ~1.9% of a step | `tests/test_env.py` |
| Downscaled JPEG frames, raw bytes over the socket | `dashboard_backend` | 36% of the pixels, no base64 | `tests/test_dashboard_backend.py` |
| Frozen T1 target per episode | `EpisodeLifecycle.begin_episode` | avoids a full-grid count on every discovering substep | - |

## Memory per worker

| Measurement | before | after | change |
|---|---|---|---|
| private memory committed at start-up | 512 MB | 30 MB | one BLAS thread |
| peak private memory over 30,000 agent steps | 688 MB | 489 MB | collect on reset |
| mean private memory over 30,000 agent steps | 449 MB | 331 MB | collect on reset |

(The two "collect on reset" rows were both measured with one BLAS thread, so
they show that change alone.)

* **One BLAS thread per env worker** (`train_agent.limit_worker_blas_threads`,
  and the same line at the top of `app.py`). Env workers do no linear algebra,
  yet OpenBLAS committed a buffer per core the moment numpy loaded: ~480 MB
  per process, **~3.8 GB across the 8 workers**. The learner's own BLAS is
  untouched.
* **Collect the previous level on reset** (`CustomMarioEnv.reset`). The
  engine's sprites and groups reference each other, so a finished level is
  cyclic garbage that only a full collection frees, and Python runs those
  rarely. One `gc.collect()` per episode (episodes last thousands of steps)
  frees it immediately.
* **Bounded state everywhere else**: the coverage bitmap is fixed-size shared
  memory (one copy for all workers); the lifecycle keeps at most 64 window
  verdicts; the adaptive target keeps 20 episodes of history; the 9.8 MB
  noncoverage class map is loaded only by the process that reports it.

What remains is the engine's own per-frame churn (sprite groups rebuilt every
frame), collected in ordinary garbage collections: a sawtooth with no upward
trend over 30,000 steps.

## Re-measuring

```bash
python tools/benchmark_step.py --steps 1500           # time per step, three stacks
python tools/benchmark_step.py --steps 1500 --profile # + where it goes
```

Measure before and after any change to `custom_mario_env.py`,
`agent_logic.py`, `rewards/` or `exploration/coverage.py`, and record the
numbers next to the change, as the ones above are.
