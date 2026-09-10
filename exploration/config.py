"""Every tunable for the exploration retrofit, in one place.

Numbers here were measured against this repository, not guessed. Where a value
came from a measurement, the measurement is recorded next to it - if you change
one, you are overriding evidence and should say why.
"""

# ═══════════════════════════════════════════════════════════════════════
# FORMAT VERSION
# Bump when the on-disk coverage layout changes in a way older files cannot
# be read as. Loading refuses on mismatch rather than guessing.
# ═══════════════════════════════════════════════════════════════════════
# v2 retires the "coverable" denominator of 7,606,986, which exceeded the
# entire 9087x600 world raster by 2,154,786 px because it counted sky above
# the world and applied no jump-height limit. Any coverage state written
# against it is incompatible by definition - its percentages meant nothing -
# so the version bump makes loading one a hard error rather than a silent
# reinterpretation.
COVERAGE_FORMAT_VERSION = 2

# ═══════════════════════════════════════════════════════════════════════
# WORLD GRID
# The level is 9087 x 600 (measured: level_rect == (0,0,9087,600)).
# The grid is padded so legal-but-outside positions never need clamping -
# clamping would fabricate coverage at the boundary. Padding absorbs
# negative y from jump arcs (observed to -29, glitch threshold at -200) and
# below-death-plane y > 600.
# ═══════════════════════════════════════════════════════════════════════
GRID_X0 = -256
GRID_Y0 = -320
GRID_W = 9600
GRID_H = 1024

LEVEL_W = 9087
LEVEL_H = 600

# The authoritative world. Verified live: level_rect == (0, 0, 9087, 600),
# from level1.setup_background() scaling level_1.png (3392x224) by
# BACKGROUND_MULTIPLER = 2.679. The viewport is 800x600 - the CAMERA, not
# the world - and GRID_W x GRID_H (9600x1024) is the padded data structure.
# Neither is ever a coverage denominator.
WORLD_RASTER_PX = LEVEL_W * LEVEL_H          # 5,452,200

# Jump envelope, used by reachability Methods B and C.
#
# JUMP_RISE_PX: mario.py:584 sets y_vel = JUMP_VEL - 0.5 = -10.5 for a
# running jump, under JUMP_GRAVITY = 0.31, giving an analytic rise of
# 10.5^2 / (2 * 0.31) = 177.8 px. Measured in-level maxima: 173 px (this
# repo, 32 clean ground->air->ground phases; capped below the analytic value
# because real launch points sit under ceilings). 183 is adopted as a
# SUPERSET bound - the mask must never exclude somewhere Mario can reach, so
# erring high is the safe direction.
#
# JUMP_REACH_PX: measured max horizontal air distance 413 px in-level; 480
# adopted for the same superset reason.
#
# A naive rise measurement returns 339 px. That is TWO CHAINED JUMPS - Mario
# lands on a coin box mid-arc and jumps again. Per-airborne-phase accounting
# is what avoids counting them as one.
JUMP_RISE_PX = 183
JUMP_REACH_PX = 480

# ─── METHOD C LATTICE — settled by a convergence study, not chosen ───
# Method C's BFS runs on an anchor lattice of this pitch. A coarse cell is
# open if ANY fine anchor inside it is open, so a coarser pitch rounds
# OUTWARD and can place reachable space a few px above Method B's exact
# per-column ceiling. The adopted denominator originally came from a 4 px
# lattice, which left an open question: was 4,013,839 a property of the
# level, or of the lattice? Measured, at three pitches, on identical geometry:
#
#     pitch   raw C        overshoot vs B   trimmed (= adopted)   runtime
#       4     4,025,096        11,257           4,013,839           1.5 s
#       2     4,019,660         5,821           4,013,839          36.5 s
#       1     4,013,723             0           4,013,723         284.8 s
#
# The overshoot halves with the pitch and reaches exactly zero at 1 px, which
# is the signature of a discretisation artifact rather than a modelling
# error. At 1 px there is no lattice at all, so Method C is a strict subset
# of Method B by construction - the overshoot of 0 is a measurement of that,
# not an assumption.
#
# 4 px and 1 px differ by 116 px, 0.0029% of the denominator, which moves the
# coverage percentage by less than 0.003 points. So the coarse answer was
# sound. It is nonetheless retired in favour of 1 px: five minutes of build
# time, paid once whenever the level geometry changes, buys a denominator
# with no discretisation term left in it to caveat.
REACHABILITY_LATTICE = 1

# Mario's collider, measured live from the running game.
MARIO_SMALL_W = 30
MARIO_SMALL_H = 40
MARIO_BIG_W = 40
MARIO_BIG_H = 80

# Measured maximum per-frame displacement: |dx| <= 14 (sprint), |dy| <= 12
# (MAX_Y_VEL = 11, observed peak y_vel 11.83). Both are strictly less than
# the corresponding MINIMUM collider dimension (30 wide, 40 tall), so
# consecutive collider rects ALWAYS overlap on both axes. The swept region
# is therefore exactly their AABB union - there are no tunnelling gaps.
#
# Verified directly: over 6000 substeps of random play, zero frames exceeded
# dy = 12. An earlier measurement appeared to show dy = 40; that was an
# artefact of measuring ACROSS a death/respawn teleport. This is why
# SpatialCoverage clears prev_rect on reset() - see its record() docstring.
MAX_FRAME_DX = 14
MAX_FRAME_DY = 12

# ═══════════════════════════════════════════════════════════════════════
# COVERAGE RECORDING
# ═══════════════════════════════════════════════════════════════════════
# Visit counts are a heatmap/diagnostic only and are NEVER a reward input,
# so losing an increment to a cross-worker race is acceptable.
TRACK_VISIT_COUNTS = True

# Lock-free by default. The bitmap write is `|= 1`, which is idempotent, so
# concurrent writes cannot corrupt it, and uint8 element writes are
# single-byte stores with no torn-write hazard. The only race is novelty
# double-credit - two workers both claiming the same virgin pixel - which is
# bounded by one swept AABB and can only ever slightly OVER-reward. It can
# never under-report or corrupt state. Authoritative totals are always
# computed parent-side as visited.sum(), which is race-free by construction,
# so the METRICS are exact even though the REWARD is best-effort.
#
# Setting this True uses a Manager().Lock() - measured 0.025 ms per
# acquire+release, roughly 2x the cost of the write it guards.
STRICT_COVERAGE_LOCK = False

# ═══════════════════════════════════════════════════════════════════════
# FRONTIER INDEX
# A coarse summary over `visited`, used only to point the agent at
# unexplored space. Coverage itself is always literal 1x1 world pixels -
# measured at 0.0073 ms per write against a ~1.8 ms env.step(), i.e. ~0.6%
# overhead, so there is no reason to coarsen it.
# ═══════════════════════════════════════════════════════════════════════
FRONTIER_CELL = 40
FRONTIER_REFRESH_STEPS = 2048

# ═══════════════════════════════════════════════════════════════════════
# REWARD
# All values are PER SUBSTEP. GlitchHunterWrapper sits BELOW
# MaxAndSkipObservation(skip=4), which SUMS the four substep rewards, so the
# reward PPO actually sees per agent step is 4x what is written here.
# ═══════════════════════════════════════════════════════════════════════
REWARD_MODE = "qa_exploration"      # "qa_exploration" | "legacy_completion"

# Novelty - the primary signal. sqrt-scaled so it stays strictly increasing
# in new pixels while compressing the range: a typical new-ground substep
# (~280 px) pays ~0.35, and the worst-case 2288-px sweep is capped at 1.0.
# N_REF = 14 px max stride * 40 px min collider height = one full stride of
# virgin ground. Bounded above and never per-pixel-linear, which is the
# whole answer to "don't let reward magnitude explode".
N_REF = 560.0
NOVELTY_CAP = 2.0

# NOVELTY_WEIGHT is solved empirically by tools/calibrate_reward.py.
#
# The original intent was to land QA episode returns within 0.5x-2.0x of
# legacy returns, so the 6M critic would not face a reward of an unfamiliar
# size. MEASUREMENT SHOWED THAT IS NOT ACHIEVABLE, and the note is left here
# rather than quietly deleted because the reasoning still matters: the legacy
# return was dominated by terms that are monotone in max-x, and a coverage
# reward has no equivalent of them. Matching the scale would need a weight
# roughly 70x above the point where a single substep outweighs an entire
# legacy episode.
#
# The risk that motivated the target is real - this project collapsed once at
# ~3.22M steps (approx_kl 11.37, clip_fraction 0.834, entropy -> 0) - so it is
# addressed directly instead: the value head is reinitialised on the first QA
# run and the learning rate is held down for VF_WARMUP_STEPS while the critic
# re-fits. See train_agent.py reset_value_head().
# ─── SOLVED BY tools/calibrate_reward.py ───
# Measured 2026-09-07 against the 6,000,000-step baseline,
# 30 episodes per arm, QA episodes starting from the bootstrap state.
#   legacy median return = 2393.55   (mean 2558.27; the distribution is
#                          bimodal, so the median is the target, not the mean)
#   QA non-novelty terms = -6.05 per episode
#   QA novelty shape     = 27.33 per episode (unweighted)
#   solved               = (2393.55 - -6.05) / 27.33 = 87.7985
#   safety ceiling       = 1.2000   (novelty alone must never reach half the reward clip)
#   ADOPTED              = 1.2000
#
# The solved value is 73x the ceiling, so the ceiling is what binds.
# Scale matching is NOT achievable here: the legacy return was dominated by the
# x-monotone terms this retrofit deletes, so QA returns land at 0.011x the
# legacy median and no SAFE weight closes that gap. The residual mismatch is
# handled by resetting the value head on the first QA run (train_agent.py
# RESET_VALUE_HEAD), not by forcing this weight upward.
NOVELTY_WEIGHT = 1.2000
NOVELTY_WEIGHT_MULT = 1.0           # StagnationCallback escalates this
NOVELTY_MULT_MAX = 2.5

# Revisiting old ground is exactly free. Not rewarded, and NOT penalised -
# traversal must never cost anything, or backtracking to reach unexplored
# space becomes self-defeating.
REVISIT_REWARD = 0.0

# Frontier potential (Ng/Harada/Russell 1999). Policy-invariant under the
# DISCOUNTED objective, so it points the agent at unexplored space without
# changing what the optimal policy is, and it vanishes on its own once the
# frontier is reached.
#
# Lowered 0.3 -> 0.1 alongside the drought rescale above. PBRS is invariant
# under discounting but NOT literally zero-sum undiscounted: a closed loop
# nets (gamma - 1) * sum(Phi), a small POSITIVE residue, so pacing does pay
# something. At weight 0.3 against the old 0.5 drought that residue was 83x
# smaller than the pressure not to idle; against the corrected 0.02 drought
# it would have been only 7x smaller, which is too close to farmable. At 0.1
# the worst-case residue is 0.001 per substep against a 0.02 drought ceiling,
# restoring a 20x margin. The whole potential range is also now worth 0.1,
# against ~5 for a single agent step on new ground - so guidance can never
# out-earn discovery (enforced by assert_reward_balance below).
FRONTIER_WEIGHT = 0.1
FRONTIER_DIST_CAP = 1200.0
GAMMA = 0.99                        # must match model.gamma

# Exploration drought. This targets FAILURE TO EXPLORE, not "being on an old
# pixel" - the distinction is the point. Vertical probing, wall-testing and
# trying many jump variants from one spot are exactly the QA behaviours the
# old x-spread stuck detector punished.
DROUGHT_GRACE = 240                 # substeps with n_new == 0 before pressure
DROUGHT_STEP = 80
# ─── RESCALED after the first calibration run ───
# This was 0.5 per substep, i.e. 2.0 per agent step - as much as a full
# agent-step of discovering virgin ground pays. That inverted the whole
# design: not-exploring was punished as hard as exploring was rewarded, so
# the drought, not novelty, was the dominant term. Measured consequence:
# QA episodes ending at -685 while earning ~6 from discovery.
#
# Now 0.02 per substep (0.08 per agent step) at full ramp. Idling still
# costs, but discovery outpays it by roughly 25x, which is the ordering the
# brief actually asks for: novelty is the primary signal and the drought is
# only pressure against camping.
DROUGHT_MAX = 0.02
# ─── DROUGHT_HARD_LIMIT IS RETIRED (Phase 3) ───
# It used to end the episode after 1600 substeps (400 agent steps) with no new
# pixel. That is a BARE drought, and a bare drought is not being stuck: with
# the bootstrap map loaded, the whole opening of the level is already covered,
# so an agent crossing it to reach new ground was being killed for doing
# exactly the right thing. It also made the 5,000-step safety floor
# unreachable a second time over, independently of the engine timer.
# Unproductive episodes are now ended by the SAFETY RESET below, which needs
# a long episode AND a long drought AND sustained genuine stuckness. The
# per-substep drought ramp above is reward logic and is unchanged.
#
# Charged when the safety reset fires - the same term, and the same magnitude
# as the legacy stuck penalty, that the retired hard limit used to charge.
DROUGHT_TERMINAL_PENALTY = -5.0

# ─── CUMULATIVE CAP (added after the first calibration run) ───
# The per-substep penalty above is bounded, but it was applied on EVERY
# substep of a drought, so over an episode it was not bounded at all. The
# first calibration run measured the consequence directly: QA episode returns
# of -685 and -607, driven almost entirely by a drought that ran for most of
# the episode at up to 0.5 per substep. That is ~1300x the novelty a whole
# episode earned, which makes the drought - not discovery - the dominant term
# in the objective, and it also gave QA returns a variance the value function
# would have had to absorb for no good reason.
#
# The drought exists to end unproductive episodes, and the -5.0 termination
# above is what should carry that weight. The per-substep ramp is only a
# gradient pointing at it. So the cumulative drought an episode can pay is
# capped here at the same magnitude as a death.
# Sized so the ramp above completes first: reaching full pressure costs 4.0,
# and the cap binds only after roughly 870 substeps of continuous drought.
# 15.0 is about three deaths - meaningful, and impossible to confuse with the
# 600+ an uncapped drought was charging.
DROUGHT_EPISODE_CAP = 15.0

# Episode shortfall, bounded at the same magnitude as the death penalty so it
# can never dominate.
SHORTFALL_PENALTY = 5.0

# ═══════════════════════════════════════════════════════════════════════
# ADAPTIVE PER-EPISODE TARGET
# Measured decay for a scripted run-right policy with coverage persisted
# across resets: 72352, 10955, 16576, 11972, 1787, 48 new px over 6
# episodes. Any FIXED target becomes impossible within ~5 episodes, so the
# target tracks the agent's own recent median and is clamped by what
# actually remains.
#
# This exerts PRESSURE. RL cannot guarantee a fixed number of new pixels per
# episode, and nothing here should be read as claiming otherwise.
# ═══════════════════════════════════════════════════════════════════════
TARGET_BETA = 1.0
TARGET_FLOOR = 500
TARGET_REMAIN_FRAC = 0.25
TARGET_HISTORY_LEN = 20
# Episodes of history before the target stops being the bare floor. Until
# then the target is not a measurement of anything, and T1 does not fire on
# it (see lifecycle._check_transition for the calibration that forced this).
TARGET_MIN_HISTORY = 5

# ═══════════════════════════════════════════════════════════════════════
# LEGACY REWARD RESCALING (applied only in "qa_exploration" mode)
#
# Why: every dominant legacy term was monotone in max-x. Roughly 2180 of
# ~2200 points of a successful episode were a strictly increasing function
# of how far right Mario got - including `visited_tiles`, which is named
# "exploration" but is keyed on the x-column alone, making it forward
# progress under another name. That is what is being replaced.
# ═══════════════════════════════════════════════════════════════════════
# The flag reward and time penalty are PHASE-SPECIFIC now - see PHASE-GATED
# REWARD below. QA_FLAG_GET_REWARD and QA_TIME_PENALTY were removed rather
# than kept as aliases, so nothing can silently read a phase-blind value.
QA_CLEAN_JUMP_REWARD = 0.3          # was 3.0 - keep the skill, drop the pull
QA_MOMENTUM_SCALE = 0.1
QA_COIN_SCALE = 0.5
QA_SCORE_SCALE = 0.02               # was 0.15
QA_POWERUP_SCALE = 0.25

# ─── CUMULATIVE INTERACTION CAP (added after the second calibration run) ───
# Measured, not anticipated. With the score term merely rescaled 0.15 -> 0.02,
# a QA episode that COMPLETED the level still returned ~350, against ~4 of
# novelty in the same episode - because this clone awards a large end-of-level
# time bonus straight into `score`, and 0.02 x that bonus dwarfed everything
# the retrofit was trying to reward. Completion was still the objective; it
# had merely changed which variable it was hiding in.
#
# That is exactly the failure mode the brief exists to prevent, and rescaling
# alone could not fix it: any scale small enough to tame the end-of-level
# windfall makes ordinary combat and coins worth nothing at all.
#
# So the whole world-interaction channel - score, coins, powerup reveals and
# powerup status changes together - is capped per episode. Interacting with
# real game content stays worth something (it is genuinely QA-relevant), but
# it can never out-earn discovery. At the calibrated weight a typical episode
# earns roughly 36 from novelty against this ceiling of 10.
#
# Only the POSITIVE side is capped. Losing a powerup still costs full price -
# a cap on penalties would be a loophole, not a safeguard.
QA_INTERACTION_EPISODE_CAP = 10.0

# ─── CUMULATIVE LOCOMOTION CAP (Phase 4B) ───
# Momentum and clean running jumps are direction-agnostic on purpose (see
# the wrapper), which also makes them farmable in place. Measured with a
# scripted controller (run right, running jump, run back left, hop) that
# never leaves the spawn screen and is never "stuck" - its box is wider than
# STUCK_BBOX_AREA, so the safety reset cannot end it: it collected 35.7 over
# a full 9,782-step episode, climbing linearly, against a natural maximum of
# 8.14 across 276 calibration episodes of the 6M policy (median 5.74, p99
# 7.91). At the calibrated COMPLETE scale that was enough to make farming
# out the timer pay about as much as a whole full-credit run to the
# castle - Requirement B inverted. Capped like interaction: above every
# natural episode measured, so ordinary play never meets it; a farm stops
# paying at the cap. Asserted against the COMPLETE payout in
# assert_phase_reward_balance.
QA_LOCOMOTION_EPISODE_CAP = 10.0

# Hard clamp on the reward PPO can receive from any single SUBSTEP. This is a
# backstop, not a tuning knob: every term above is individually bounded, so
# under correct operation this never fires. If it starts firing (the wrapper
# counts it and exports qa_clip_events) something has gone wrong upstream and
# you want to know before the advantage estimator does. The value is set just
# above the worst legitimate substep: a death (-5) landing on the same frame
# as a drought termination (-5), or a flag_get (5) plus a powerup (5).
QA_REWARD_CLIP = 12.0

# Drought pressure escalates one notch per DROUGHT_STEP substeps without a
# new pixel, each notch worth this much, up to DROUGHT_MAX. Four notches
# reach the ceiling, so the ramp plays out over 320 substeps and is fully
# expressed well before DROUGHT_EPISODE_CAP starts binding.
DROUGHT_NOTCH = 0.005


# ═══════════════════════════════════════════════════════════════════════
# PHASE-GATED REWARD (Phase 4A)
#
# The lifecycle (exploration/lifecycle.py) defines the priority, not a
# blend: every episode is EXPLORE, then - once, one way - COMPLETE. There is
# no fixed split between the two objectives; each phase pays for its own.
#
#   EXPLORE   maximise new testable coverage. Nothing pays for moving right,
#             nothing charges for time, and finishing is worth no more than
#             the shortfall it triggers.
#   COMPLETE  finish Level 1-1. Forward progress pays, the flag pays more,
#             novelty drops to a tie-breaker, exploration-only pressure stops.
#
# ─── CALIBRATED IN PHASE 4B (tools/calibrate_phase_reward.py) ───
# The 4A values were sized against PER-SUBSTEP dominance only, and measured
# at EPISODE scale they were badly off. The 6M policy, run with inference
# only from the bootstrap map, earns this much EXPLORE novelty per episode
# (EXPLORE pinned for the whole run, 20 episodes):
#     median 8.92, mean 24.53 (0 .. 128; it is heavily skewed)
# while ONE full-credit COMPLETE run to the castle paid 259 progress + 10 flag
# at 4A's 0.03/px - thirty median exploration episodes. The policy cannot
# see the phase (same 4x84x84 frames in both), so that pay leaks into
# EXPLORE behaviour wherever the two phases share the screen. Measured on
# the natural lifecycle (ppo_boot + eps_boot + campaign, 90 episodes,
# steady state = after the target-history warm-up):
#
#                                         4A values     4B values
#     completion / exploration, summed       3.05          0.24
#     ... discounted-return spread           4.40          0.39
#     COMPLETE > EXPLORE pay at x >= 3072    every bucket  flag area only
#
# The spread is the number that matters for PPO: it is the part of each
# group's discounted return the critic cannot absorb, i.e. what reaches the
# advantage. 4B holds it under 0.5 - completion is a real but minority
# signal - rather than at a fixed split. See the report in the Phase 4B
# commit for every arm's per-channel breakdown.
# ═══════════════════════════════════════════════════════════════════════
# The env's own death reward (custom_mario_env.step: `reward = -5.0`), named
# so the bounds below can be stated against it.
ENGINE_DEATH_PENALTY = 5.0
# mario_clone constants MAX_WALK_SPEED = 6 px/frame. Walking, not sprinting,
# is the pace the dominance check below is held to: the stricter case.
WALK_PX_PER_SUBSTEP = 6
# Spawn x = 110 to the castle door: every completed episode in calibration
# ended at max-x 8,745-8,751. The whole of what forward progress can pay.
LEVEL_COMPLETE_SPAN_PX = 8641
# The measured reference that bounds a whole COMPLETE run (see above): the
# mean EXPLORE novelty of one 6M episode from the bootstrap map.
CALIB_EXPLORE_EPISODE_NOVELTY_MEAN = 24.53
CALIB_EXPLORE_EPISODE_NOVELTY_MEDIAN = 8.92

# ─── TIME ───
# EXPLORE: zero. The old 0.005/substep was harmless at a 2,451-step cap
# (-49 at most) but at the 9,781-step QA cap it reaches -196 per episode -
# several times what an episode earns from discovery, and 39x the -5 cost of
# dying. That makes ending the episode early (by death or by the flag) the
# single best move available, which is exactly "pressure to finish early".
# Idling is still not free: stuck-gated drought, the safety reset and the
# end-of-episode shortfall each cost something, and each is bounded. None of
# them scales with how long a PRODUCTIVE episode runs.
EXPLORE_TIME_PENALTY = 0.0
# COMPLETE: mild urgency, capped per episode at HALF the death penalty, so
# dying can never be a way to stop paying it - and a stuck agent can save at
# most 2.0 by dying instead of timing out, not 4.0 as under 4A. The rate is
# set so the cap binds at 1,600 substeps = 400 agent steps, about one whole
# 6M completion run (median 410 steps): 4A's 0.005 hit its cap after 200
# steps and gave no urgency for the second half of a run. Per substep it is
# 1/7 of what walking toward the castle pays, so it adds urgency without
# ever competing with progress.
COMPLETE_TIME_PENALTY = 0.00125
COMPLETE_TIME_PENALTY_EPISODE_CAP = 2.0

# ─── FLAG ───
# EXPLORE: unchanged from the previous QA value, and no larger than the
# shortfall a premature finish triggers - so reaching the flag having
# explored nothing can at best break even, while forfeiting every pixel the
# rest of the episode could have found.
EXPLORE_FLAG_REWARD = 5.0
# COMPLETE, at full completion credit. Interpolated from EXPLORE_FLAG_REWARD
# by the credit (see lifecycle.completion_credit), so finishing in COMPLETE is
# never worth less than finishing in EXPLORE. 4A's 10.0 was a sparse jackpot
# 25x a median exploration episode's worth; 6.0 is under half the dense
# progress budget (13.0), so the route pays more than the finish line, and
# it keeps the flag substep under QA_REWARD_CLIP even if a powerup lands on
# the same substep - which 10.0 did not (asserted below). The legacy +500
# would simply be clipped, and restoring it is not the point.
COMPLETE_FLAG_REWARD = 6.0

# ─── FORWARD PROGRESS (COMPLETE only) ───
# Paid per px of NEW episode max-x while in COMPLETE, scaled by completion
# credit. The baseline advances in EXPLORE too, so ground already covered
# this episode is never paid for twice. Per-substep gain is capped at
# MAX_FRAME_DX: anything larger is a teleport, i.e. a glitch, not progress.
#
# 0.03 -> 0.0015 (Phase 4B). A whole level of progress is now 13.0, and with
# the flag a full-credit run from the spawn pays 19.0 - under the MEAN
# exploration episode (24.53), about two MEDIAN ones. 0.002 was measured too:
# it passed at the start of training but pushed completion past exploration
# (ratio 1.2) in the persistent campaign once the map began to fill, which
# is the direction training moves in.
COMPLETE_PROGRESS_PER_PX = 0.0015

# ─── NOVELTY IN COMPLETE ───
# Reduced to a tie-breaker, not zeroed. Coverage is still RECORDED at full
# fidelity either way; this is only what discovery PAYS. It must be small
# enough that walking toward the finish out-earns a detour onto virgin ground
# (asserted below, with a 2x margin), and it deliberately ignores the
# StagnationCallback multiplier, which escalates EXPLORATION pressure and has
# no business growing a distraction in the phase whose job is to finish.
# It scales with COMPLETE_PROGRESS_PER_PX, so 4B's 20x cut in progress takes
# it from 0.1 to 0.003: a full virgin stride pays 0.0036 against 0.009 for a
# walking substep toward the castle.
COMPLETE_NOVELTY_MULT = 0.003


def assert_phase_reward_balance():
    """The inequalities that make the phase gating mean what it says.

    Called at wrapper construction next to assert_reward_balance(), so a
    retune that breaks one fails at startup rather than as a silently
    speedrunning agent.
    """
    problems = []
    qa_cap_substeps = QA_EPISODE_CAP_AGENT_STEPS_MEASURED * SUBSTEPS_PER_AGENT_STEP
    walk = WALK_PX_PER_SUBSTEP * COMPLETE_PROGRESS_PER_PX
    sprint = MAX_FRAME_DX * COMPLETE_PROGRESS_PER_PX
    progress_budget = COMPLETE_PROGRESS_PER_PX * LEVEL_COMPLETE_SPAN_PX
    if EXPLORE_TIME_PENALTY * qa_cap_substeps >= ENGINE_DEATH_PENALTY:
        problems.append(
            f"EXPLORE time penalty totals "
            f"{EXPLORE_TIME_PENALTY * qa_cap_substeps:.1f} over a full QA "
            f"episode - more than a death, so dying early would pay")
    # Half, not merely below: the most a stuck agent can save by dying
    # instead of timing out is the unpaid part of this cap.
    if COMPLETE_TIME_PENALTY_EPISODE_CAP > 0.5 * ENGINE_DEATH_PENALTY:
        problems.append("COMPLETE time penalty cap exceeds half the death "
                        "penalty - dying would save too much of it")
    if walk < 4 * COMPLETE_TIME_PENALTY:
        problems.append(
            f"COMPLETE time ({COMPLETE_TIME_PENALTY}/substep) is more than a "
            f"quarter of walking progress ({walk:.4f}) - urgency would compete "
            f"with the direction it is meant to hurry")
    full_stride = COMPLETE_NOVELTY_MULT * NOVELTY_WEIGHT * 1.0
    capped_sweep = COMPLETE_NOVELTY_MULT * NOVELTY_WEIGHT * NOVELTY_CAP
    if 2 * full_stride > walk:
        problems.append(
            f"in COMPLETE, walking toward the finish pays {walk:.4f}/substep "
            f"but a stride of virgin ground pays {full_stride:.4f} - less than "
            f"a 2x margin, so novelty could pull Mario off the route")
    if 2 * capped_sweep > sprint:
        problems.append("in COMPLETE, sprinting toward the finish does not "
                        "out-earn a worst-case novelty sweep by 2x")
    # The flag substep can also carry a powerup (small -> tall = 20 x scale),
    # which is the worst case QA_REWARD_CLIP was sized for.
    worst_flag_substep = (COMPLETE_FLAG_REWARD + capped_sweep + sprint
                          + 2 * QA_CLEAN_JUMP_REWARD + 20.0 * QA_POWERUP_SCALE)
    if worst_flag_substep >= QA_REWARD_CLIP:
        problems.append(
            f"the COMPLETE flag substep can reach {worst_flag_substep:.2f}, "
            f"at or above the {QA_REWARD_CLIP} clip")
    if EXPLORE_FLAG_REWARD > SHORTFALL_PENALTY:
        problems.append("the EXPLORE flag reward exceeds the shortfall it "
                        "triggers - rushing to the flag would pay")
    if not EXPLORE_FLAG_REWARD <= COMPLETE_FLAG_REWARD:
        problems.append("finishing in COMPLETE pays less than in EXPLORE")
    if progress_budget < COMPLETE_FLAG_REWARD:
        problems.append(
            f"the COMPLETE flag ({COMPLETE_FLAG_REWARD}) is worth more than "
            f"the whole level of progress ({progress_budget:.2f}) - a sparse "
            f"jackpot, not a finish line")
    # Episode scale, which 4A never checked. The phase is invisible to the
    # policy, so one full-credit run to the castle must not be worth more
    # than one measured exploration episode.
    full_run = progress_budget + COMPLETE_FLAG_REWARD
    if full_run > CALIB_EXPLORE_EPISODE_NOVELTY_MEAN:
        problems.append(
            f"a full-credit COMPLETE run pays {full_run:.1f}, more than the "
            f"measured mean exploration episode "
            f"({CALIB_EXPLORE_EPISODE_NOVELTY_MEAN}) - completion would "
            f"dominate what the policy learns")
    # A COMPLETE agent that farms locomotion until the timer ends it earns
    # at most the cap, less the time cost and the timer's own death charge.
    # Finishing - even at zero credit - pays at least the EXPLORE flag.
    farm = (QA_LOCOMOTION_EPISODE_CAP - COMPLETE_TIME_PENALTY_EPISODE_CAP
            - ENGINE_DEATH_PENALTY)
    if farm >= EXPLORE_FLAG_REWARD:
        problems.append(
            f"farming locomotion out the timer in COMPLETE can net {farm:.1f}, "
            f"not less than the {EXPLORE_FLAG_REWARD} a finish pays")
    if not 0.0 <= COMPLETE_NOVELTY_MULT <= 1.0:
        problems.append("COMPLETE_NOVELTY_MULT outside [0, 1]")
    if problems:
        raise ValueError("phase reward balance violated: " + "; ".join(problems))


def assert_reward_balance():
    """Guards the ordering the whole design depends on.

    Called at wrapper construction, so a bad retune fails at the start of a
    training run rather than 200,000 steps in.

    The invariant: the ENTIRE frontier potential range must be worth less
    than a single agent step of genuine discovery. Otherwise the agent is
    being paid more to approach unexplored space than to actually explore it,
    and "walk toward the frontier and stop just short" becomes a strategy.
    """
    # Phi is bounded in [-1, 0], so the total shaping available from crossing
    # the whole map to the frontier is at most FRONTIER_WEIGHT * 1.0.
    frontier_total = FRONTIER_WEIGHT * 1.0
    # One agent step = 4 substeps (MaxAndSkipObservation sums them). A substep
    # covering one full stride of virgin ground (N_REF px) scores exactly 1.0
    # before weighting.
    one_step_novelty = 4.0 * NOVELTY_WEIGHT * 1.0
    if frontier_total >= one_step_novelty:
        raise ValueError(
            f"frontier guidance ({frontier_total:.3f} for the whole map) is "
            f"not clearly weaker than discovery ({one_step_novelty:.3f} for a "
            f"single agent step on new ground). Lower FRONTIER_WEIGHT or "
            f"raise NOVELTY_WEIGHT - as configured the agent is paid more to "
            f"approach the frontier than to cross it.")
    # Revisits must be free. A negative value here would make backtracking to
    # reach unexplored space self-defeating, which is the opposite of the goal.
    if REVISIT_REWARD < 0.0:
        raise ValueError(
            f"REVISIT_REWARD={REVISIT_REWARD} penalises traversal of already "
            f"explored ground. Old territory is transit space, not a cost.")

# ═══════════════════════════════════════════════════════════════════════
# GLITCH ORACLES (purely additive - the existing 5 are untouched)
# ═══════════════════════════════════════════════════════════════════════
# px of overlap into a solid's INTERIOR before it counts as a glitch.
#
# Raised from 2 after measuring it. Over the 40-episode bootstrap run
# (1,882,128 recorded px of ordinary play by the 6M policy), the collider
# was found inside a solid 679 times, at chessboard depths of
# 1:203  2:174  3:118  4:100  5:84  and never at 6 or beyond.
# So the engine's collision resolution routinely lets the rect sink up to
# 5px into a block for a frame before pushing it back out. A tolerance of 2
# would have fired on 302 of those - a detector that cries wolf during
# normal play is worse than no detector at all. 6 sits just outside the
# entire observed envelope.
PENETRATION_TOL = 6

# ═══════════════════════════════════════════════════════════════════════
# EPISODE LENGTH — ROOT CAUSE OF THE 2,451-STEP CAP, AND THE FIX
#
# The authoritative episode clock is the ENGINE's HUD timer, nothing else.
# mario_clone/data/components/info.py:
#
#     self.time = 401                                           (line 23)
#     elif (CURRENT_TIME - self.current_time) > 400:            (line 302)
#         self.current_time = CURRENT_TIME
#         self.time -= 1
#
# and level1.check_if_time_out() kills Mario with death_cause 'timeout' the
# frame it reaches 0. CustomMarioEnv advances CURRENT_TIME by 1000/60 ms per
# substep, so a unit needs 24 substeps of exactly 400.000 ms - but the test is
# STRICTLY greater-than, and float accumulation of 1000/60 lands just above or
# just below 400 on alternate units. Measured on the raw engine, NOOP held:
#
#     401 units:  218 x 24 + 182 x 25 substeps   ->  timeout at substep 9806
#                 9806 / 4 (MaxAndSkipObservation skip)  =  2,451 agent steps
#
# The env ends the episode the same substep (it skips the death animation),
# so there is no tail. That is the whole cap: 401 units x 24.45 substeps / 4.
#
# THE LIMIT HIERARCHY, after this fix. Exactly one clock is authoritative;
# everything else is either a deliberate backstop or a lifecycle rule:
#
#   1. Engine timer   AUTHORITATIVE. QA: QA_EPISODE_TIME_UNITS. Legacy: 401,
#                     untouched, so the 6M brain still sees the episode it was
#                     trained in. The TIME box draws a legacy-equivalent
#                     view of it in QA; see QA_EPISODE_TIME_UNITS.
#   2. TimeLimit      BACKSTOP only, strictly above the measured timer cap, so
#                     it can never pre-empt the timer. It exists for the one
#                     case the timer cannot handle: a glitch that freezes the
#                     HUD clock. In a QA tool that is a realistic failure, not
#                     a hypothetical one.
#   3. Safety reset   LIFECYCLE rule (exploration/lifecycle.py). Needs a floor
#                     AND a drought AND sustained stuckness; elapsed steps
#                     alone can never fire it.
#
# The retired DROUGHT_HARD_LIMIT was a fourth, conflicting limit; see above.
# ═══════════════════════════════════════════════════════════════════════
SUBSTEPS_PER_AGENT_STEP = 4         # MaxAndSkipObservation(skip=...) - one source

ENGINE_TIME_UNITS_DEFAULT = 401
ENGINE_CAP_AGENT_STEPS_MEASURED = 2451          # 9806 substeps, NOOP, raw engine

# QA mode only. Set on the engine's HUD counter at reset, so the game's own
# clock stays the one authority rather than being second-guessed by a wrapper.
# Measured with the same NOOP hold: 1600 units -> timeout at substep 39126
# = 9,781 agent steps (873 x 24 + 726 x 25 substeps per unit), 3.99x the
# default.
#
# What the TIME box DRAWS is decoupled from this clock (Phase 4D): it leaves
# the extra units out and shows the clock a legacy episode would show at the
# same point, holding at 001 once legacy would have timed out, and reading
# 000 on the frame the real clock runs out (OverheadInfo.display_time, set up
# in CustomMarioEnv.reset). Drawing only; every rule still reads the clock.
# Measured on identical action sequences through the real 84x84 pipeline:
#   drawing the clock itself ("1600")  34.8 px differ from legacy per frame,
#                                      on 100% of frames (4-digit layout)
#   legacy-equivalent clock            0 px on every frame up to legacy's
#                                      own timeout; every later frame is a
#                                      frame legacy also drew ("001")
# The 6M policy on real states with only the timer swapped: "001" vs "401"
# agree on the argmax action 99.2%, closer than two adjacent legacy frames
# ("200" vs "199": 96.4%), so the hold is not read as imminent death.
QA_EPISODE_TIME_UNITS = 1600
QA_EPISODE_CAP_AGENT_STEPS_MEASURED = 9781

# TimeLimit backstops, in agent steps. Each is strictly above its mode's
# measured timer cap (asserted by tests/test_episode_lifecycle.py). The legacy
# value is unchanged from what the 6M brain was trained with.
QA_EPISODE_MAX_STEPS = 12000
LEGACY_EPISODE_MAX_STEPS = 4000

# End QA episodes at the castle door instead of after the engine's victory
# sequence. At the door the HUD switches to FAST_COUNT_DOWN and burns the
# REMAINING time one unit per frame, converting it to score, before `done`.
# Measured tail = remaining units + ~121 substeps: 130 agent steps at the
# default budget, but 429 at 1600 - hundreds of steps per completion where no
# action has any effect, a length set by the timer budget rather than by the
# agent. The env already ends on death for the same reason (it skips the
# death animation). Legacy mode keeps the full sequence it was trained with.
QA_END_ON_LEVEL_COMPLETE = True

# ═══════════════════════════════════════════════════════════════════════
# EPISODE LIFECYCLE — EXPLORE -> COMPLETE  (exploration/lifecycle.py)
#
# Every episode starts in EXPLORE. It moves to COMPLETE, once and one way,
# when any transition criterion fires. The transition is a LIFECYCLE event,
# not a termination: it never ends the episode. What each phase pays is in
# PHASE-GATED REWARD above.
#
# All counts below are AGENT steps; the lifecycle converts via
# SUBSTEPS_PER_AGENT_STEP because the wrapper sees substeps.
# ═══════════════════════════════════════════════════════════════════════
# One tumbling window serves both the transit/stuck classifier and the
# novelty-yield check. The brief sets TRANSIT_WINDOW and YIELD_WINDOW both to
# 240; making them ONE window means "exhausted and not in transit" (T2) is
# judged over the same span of play, not two windows that drift apart.
LIFECYCLE_WINDOW = 240

# T1 - target met is necessary but not sufficient (Phase 4C). It must ALSO
# have been at least this many agent steps since the last meaningfully-new
# pixel (EpisodeLifecycle.drought_agent_steps(), the same signal the safety
# reset below already uses) - reusing an existing, continuously-updated
# per-substep counter rather than a windowed one, because a LIFECYCLE_WINDOW
# is 240 agent steps and the natural mix's median episode is ~400: most
# episodes never close even one window, so a decline check gated on closed
# windows almost never gets evidence before the episode ends (measured: a
# window-based version converged to the SAME outcome across ratios from 0.15
# to 1.0 - it was really just deferring everything to T2).
#
# MUST be >= LIFECYCLE_WINDOW, not just "large enough" by feel. The transit
# exemption (in_coherent_transit, below) only has evidence once a window has
# closed; before that it reads as "no evidence", same as everywhere else in
# this file (see test_transit_with_target_met_does_not_transition, which
# fails at any value below LIFECYCLE_WINDOW). Worst case for how EARLY the
# drought clock can reach its threshold is discovery on literally the first
# substep of the episode, making drought_agent_steps() grow from substep 0 -
# so a threshold below LIFECYCLE_WINDOW*SPS substeps can cross before window
# 1 has ever closed, and the exemption has nothing to exempt with yet.
# Set equal to LIFECYCLE_WINDOW, the tightest value with that guarantee:
# _close_window() always runs (unconditionally, every observe()) before
# _check_transition() on the SAME substep, so by the time drought can first
# reach LIFECYCLE_WINDOW agent steps, window 1 has already closed or is
# closing on that exact substep - T1 can never fire ahead of the first
# window's own verdict.
#
# Measured on ppo_boot+eps_boot+campaign, 90 episodes of the 6M policy from
# the bootstrap map: post-switch discovery share 59.3% -> 1.4%, premature
# switches (half or more of the episode's total discovery still ahead)
# 20/90 -> 0/90, switches attributable to T1 31/90 -> 8/90 (T2: 11 in both -
# the same episodes reach genuine exhaustion either way). 180-239 measured
# the same aggregate outcome on this data but do not carry the guarantee
# above; below 180 the outcome visibly degrades (more premature switches) as
# the threshold shrinks toward DROUGHT_GRACE.
T1_DECLINE_DROUGHT_STEPS = LIFECYCLE_WINDOW

# T2 - novelty yield exhausted and NOT in transit, sustained.
YIELD_FLOOR = 200                   # new testable px per window
YIELD_WINDOWS = 3                   # consecutive exhausted windows
# T4 - backstop only. 0.75 x the measured QA cap, per the brief. If T4 is
# the criterion that usually fires, the other thresholds are wrong.
MAX_EXPLORE_STEPS = int(0.75 * QA_EPISODE_CAP_AGENT_STEPS_MEASURED)    # 7335

# Transit vs stuck. "No new pixels for a while" is NEVER stuck on its own:
# crossing covered ground to reach new ground is the job.
TRANSIT_STRAIGHTNESS = 0.35         # net displacement / path length
TRANSIT_FRONTIER_GAIN_PX = 200      # closed this much distance on the frontier
STUCK_BBOX_AREA = 120 * 120         # px^2 of the window's position bbox

# ═══════════════════════════════════════════════════════════════════════
# SAFETY RESET — requires ALL THREE. Fires in either phase.
#
# Reachable now that the QA cap is 9,781: 5,000 < 9,781. At the old 2,451
# cap it was dead code, and so was anything above 2,451.
# ═══════════════════════════════════════════════════════════════════════
SAFETY_MIN_EPISODE_STEPS = 5000     # a FLOOR, never a trigger by itself
SAFETY_DROUGHT_STEPS = 1500         # agent steps since meaningful discovery
SAFETY_STUCK_WINDOWS = 4            # consecutive IS_STUCK windows
# What counts as "meaningful". 1 testable px - the most conservative choice,
# since it makes the fallback as hard as possible to fire. n_new already
# counts only TESTABLE pixels, so an out-of-world clip cannot reset it.
SAFETY_MEANINGFUL_NEW_PX = 1

# ═══════════════════════════════════════════════════════════════════════
# TRAINING (QA PHASE)
# ═══════════════════════════════════════════════════════════════════════
VF_WARMUP_STEPS = 200_000
VF_WARMUP_LR = 2.5e-5
NORMAL_LR = 1.0e-4

STAGNATION_WINDOW = 3
STAGNATION_RATE_THRESHOLD = 500     # new px per 10k steps
STAGNATION_REMAINING_MIN = 100_000
ENT_COEF_BASE = 0.03
ENT_COEF_MAX = 0.06

# ═══════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════
EXPLORATION_DATA_DIR = "exploration_data"
REACHABLE_MASK_PATH = "exploration_data/reachable_mask.npz"
CHECKPOINT_DIR_QA = "checkpoints_qa"
CHECKPOINT_NAME_QA = "glitch_hunter_qa"
BASELINE_MODEL = "backup_6M/mario_brain_checkpoint.zip"
BOOTSTRAP_COVERAGE = "exploration_data/coverage_bootstrap_6000000.npz"
BOOTSTRAP_EPISODES = 40

# ═══════════════════════════════════════════════════════════════════════
# EXPECTED GEOMETRY — asserted by build_reachability.py so a level change
# fails loudly instead of silently moving the coverage denominator.
#
# EXPECTED_COVERABLE_PX (7,606,986) IS GONE AND MUST NOT COME BACK. It was
# larger than the world it described. See exploration/reachability.py for
# the full account and the measurement that proved it.
# ═══════════════════════════════════════════════════════════════════════
EXPECTED_SOLID_RECTS = 80
EXPECTED_SOLID_PX = 733_176

# Filled in by tools/build_reachability.py after the three methods
# reconcile. Left as None until then rather than carrying a guess.
# ─── ADOPTED BY tools/build_reachability.py ───
#   world raster        = 5,452,200   (9087 x 600, informational only - NEVER a denominator)
#   solid px            = 733,176   (80 collider rects)
#   valid anchors       = 4,117,339
#   standable anchors   = 11,613
#
#   Method A (geometric)  = 4,699,146   informational; no gravity, no jump limit
#   Method B (jump env.)  = 4,097,097
#   Method C (BFS)        = 4,013,723   <- ADOPTED
#   B vs C delta          = 2.03%
#
# Coverage percentage is ALWAYS covered_testable / TESTABLE_TOTAL.
# Pixels outside this mask are noncoverage_px, never coverage. Most
# of that is NORMAL (jump arcs, pit deaths, collision tolerance);
# only the genuinely impossible subset is anomalous_px.
TESTABLE_TOTAL = 4013723
TESTABLE_FINGERPRINT = "9a4d666eba4751adb38f53ddfc3cf8ac79cafca68857489d343635fa9092e71e"
ADOPTED_METHOD = "C"
