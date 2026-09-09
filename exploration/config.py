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
DROUGHT_HARD_LIMIT = 1600           # substeps -> terminate the episode
DROUGHT_TERMINAL_PENALTY = -5.0     # same magnitude as the legacy stuck penalty

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

# ═══════════════════════════════════════════════════════════════════════
# LEGACY REWARD RESCALING (applied only in "qa_exploration" mode)
#
# Why: every dominant legacy term was monotone in max-x. Roughly 2180 of
# ~2200 points of a successful episode were a strictly increasing function
# of how far right Mario got - including `visited_tiles`, which is named
# "exploration" but is keyed on the x-column alone, making it forward
# progress under another name. That is what is being replaced.
# ═══════════════════════════════════════════════════════════════════════
QA_FLAG_GET_REWARD = 5.0            # was 500.0 - completion becomes ordinary
QA_CLEAN_JUMP_REWARD = 0.3          # was 3.0 - keep the skill, drop the pull
QA_MOMENTUM_SCALE = 0.1
QA_TIME_PENALTY = 0.005             # was 0.02 - thorough QA takes time
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
# BLOCKING DESIGN FACT — EPISODE LENGTH IS CAPPED BY THE ENGINE
#
# Recorded here in Phase 2, to be ACTED ON in the episode-lifecycle phase.
# Nothing here changes the engine yet.
#
# info.py sets self.time = 401 and decrements it once per 400 ms of game
# time. Game time advances 16.667 ms per substep, so one time-unit is 24
# substeps = 6 agent steps. Measured directly by holding NOOP from reset:
#
#     DONE at substep 9806  =>  2,451 agent steps
#     time left 0, death_cause 'timeout'
#
# Consequences that block the Explore->Complete design as drafted:
#   * TimeLimit(max_episode_steps=4000) in train_agent.make_env() NEVER
#     FIRES. The in-game timer always wins first.
#   * A 5,000-agent-step safety-reset floor is UNREACHABLE. Any threshold
#     above 2,451 is dead code.
#   * "Explore for a long time, then finish the level" does not fit in one
#     episode at the current cap.
#
# The fix (raising the timer for QA mode only, additively and gated) belongs
# to the episode-lifecycle phase, not to the pixel-count correction.
EPISODE_CAP_AGENT_STEPS_MEASURED = 2451
EPISODE_CAP_TIME_UNITS_DEFAULT = 401

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
