# Nonlinear Set Membership (NSM) comparator arm for the Part-1 data-only
# benchmark (Diona vs NN vs GP vs NSM/RBF/IDW, 5 synthetic functions, 3-8 dims).
#
# ==============================================================================
# PROVENANCE
#
# No Julia package implements Nonlinear Set Membership identification
# (Milanese & Novara, "Set membership identification of nonlinear systems",
# Automatica 40(6):957-975, 2004), so this arm is hand-written. It is a
# SELF-CONTAINED port of the noise (`feps`) and Lipschitz (`fgam`) estimators
# and of the upper/lower cone bounds used by SMGO-Delta:
#
#     /home/ls122/code/smgo-delta/opt_smgo_d.m
#
# Every ported block cites its MATLAB line numbers inline. The three ported
# regions are:
#
#   * `nsm_eps`     <- opt_smgo_d.m:257-279   (`feps`, the noise-level estimate)
#   * `nsm_gamma`   <- opt_smgo_d.m:295-306   (`fgam`, the Lipschitz estimate)
#   * `nsm_predict` <- opt_smgo_d.m:482-483   (`sbl_cdpt(SBL_ROW_FUB/FLB, .)`),
#                                              plus the midpoint, which the
#                                              MATLAB optimiser does not form
#                                              (it uses a `delta`-weighted blend
#                                              at line 449 for acquisition, NOT
#                                              for regression).
#
# The estimator is
#
#     ybar(x) = min_i ( y_i + eps + gamma*||x - x_i|| )      (upper bound)
#     ylow(x) = max_i ( y_i - eps - gamma*||x - x_i|| )      (lower bound)
#     fhat(x) = 0.5 * ( ybar(x) + ylow(x) )                  (central estimate)
#
# over ALL training samples, with no ball restriction (opt_smgo_d.m:482-483),
# with `gamma = s.gamma_factor * fgam` (see SAFETY FACTOR below).
#
# ==============================================================================
# THE INTERVAL IS NOT CERTIFIED -- `interval_kind == :set_membership_estimated`
#
# An earlier revision of this file labelled the band `:certified_lipschitz`.
# That label was wrong and is retired; see `NSM_INTERVAL_SEMANTICS` below for
# the string the harness reports. The band is the set-membership interval
# implied by the (eps, gamma) pair, and BOTH are estimated from the training
# data by the two heuristics above -- the benchmark protocol hands no arm an a
# priori noise bound or Lipschitz constant -- so nothing about it is certified.
# Three measurements, all on the 36 Part-1 cells (2026-09-20), say so directly:
#
#   * eps UNDERSHOOTS the injected noise half-width on every noisy cell:
#     ratio 0.0057 to 0.648, mean 0.339, never reaching 1. The driver is ball
#     OCCUPANCY (see `nsm_eps` on why feps shrinks like 1/density), not n.
#   * consequently the band excludes the TRUE function at 51.1-99.35% of
#     TRAINING anchors on every noisy cell. B2/B3 (see the gate) pin the band at
#     a training point inside `y_i +/- eps`, so an eps below the noise level is
#     exactly an exclusion rate at the anchors.
#   * even at noise 0 the QUERY-point coverage bottoms at 0.97925, not 1.000.
#
# The last one is structural, not a tuning artefact. `fgam` is a max over the
# FINITELY MANY observed secant slopes |y_j - y_i| / ||x_j - x_i||, further
# DEFLATED by 2*eps in the numerator (opt_smgo_d.m:300). A max over finitely
# many observed secants is at best a lower bound on the true Lipschitz constant
# of f over the box, and the deflation pushes it strictly lower still, so the
# cone under-covers BETWEEN samples. (Under noise the deflation is a correction
# rather than a slack, and the estimate is then not a bound in either
# direction.)
#
# The corollary that matters for reporting: query-point coverage, which is what
# the harness measures, stays 0.955-1.000 on all 36 cells. The failure is
# invisible where it is measured. Do not read a high coverage column on this
# arm as evidence that the band is sound.
#
# ==============================================================================
# SAFETY FACTOR ON GAMMA (slate symmetry, not fidelity)
#
# Diona multiplies its estimated Hessian-Lipschitz constant by
# `L_H_factor = 2.5`. `NSMSettings.gamma_factor` gives NSM the same multiplier
# so that neither arm carries a cushion the other does not, and it defaults to
# 2.5 for that reason. It is a WRAPPER, applied in `fit_nsm` on the value
# `nsm_gamma` returns; the ported estimator itself is untouched, and
# `gamma_factor = 1.0` reproduces the source bit-for-bit (this is what
# `nsm_equivalence_tests.jl` runs against `nsm_reference.jl`). `NSMModel`
# carries `gamma_raw` (the port's own `fgam`) and `gamma` (the value the cone
# actually uses) separately so the two can be reported paired.
#
# eps is NOT multiplied, because Diona does not scale its noise bound.
#
# WHAT THE FACTOR CAN AND CANNOT DO. The phase brief predicted that it "cannot
# change training-point containment at all", on the grounds that the distance
# at a training point is zero so the gamma term vanishes. That reasoning is
# incomplete and the conclusion is true only conditionally. `upper` is a MIN
# over ALL n cone terms, not just the i-th: the i-th term is `(y_i + eps) +
# gamma*0 = y_i + eps` exactly, but a NEIGHBOUR's term can be smaller and bind
# instead. So the band at x_i sits INSIDE `[y_i - eps, y_i + eps]` (that is
# B2/B3, and it is exact in floating point), and raising gamma pushes it OUT
# toward that cap without ever passing it. Training-point containment is
# therefore monotone non-decreasing in the factor and SATURATES at the cap --
# not invariant by theorem.
#
# Which of the two it is in practice is a measurement, and it splits by noise
# level (12 of the 36 Part-1 cells, 2026-09-20, shipped path, 2000-anchor
# subsample, containment of the TRUE function at training anchors):
#
#   noise 5%:  0.3040 -> 0.3040   0.0910 -> 0.0910   0.4885 -> 0.4885
#              0.4750 -> 0.4750   0.0785 -> 0.0785   0.0065 -> 0.0065
#   noise 0:   0.9940 -> 1.0000   0.9995 -> 1.0000   0.9990 -> 1.0000
#              0.9950 -> 1.0000   0.9995 -> 1.0000   1.0000 -> 1.0000
#
# On the NOISY cells the factor changes containment by exactly nothing, to four
# decimals, because the band was already at the cap at 99.7-100% of anchors --
# the brief's prediction is right where the failure lives, for a reason one
# step removed from the one it gave. On the NOISELESS cells it does move, from
# ~0.999 to 1.000, closing the small residue where a neighbour's term was
# binding. Either way it saturates: the eps undershoot documented above is a
# ceiling on training-anchor containment that no gamma can lift.
#
# Away from the samples the band simply widens -- mean QUERY width x2.8 to x3.0
# on the same cells -- and the harness's coverage metric moves up with it
# (worst cell 0.9553 -> 0.9960, another 0.9853 -> 1.0000, most to exactly
# 1.0000). That is the whole shape of the thing: the factor buys coverage where
# coverage is measured and buys nothing where the band is actually broken.
#
# `src/noise.jl` (`estimate_noise_bound`) is deliberately NOT used and
# `PreparedSmokeCell.noise_bound` is deliberately NOT read: NSM estimates its
# own `eps` per opt_smgo_d.m:257-279. This module imports nothing from
# `src/` at all. Its ONLY dependency is NearestNeighbors.jl (already a
# `test/ml/Project.toml` dependency); see the note above `module NSMTrainer`
# for why `LinearAlgebra` in particular is not used.
#
# ==============================================================================
# COORDINATE FRAME (port decision, not inherited from the source)
#
# SMGO-Delta generates its own samples inside the unit hypercube: `x0 = rand(D,1)`
# (opt_smgo_d.m:103), the black box is called as `f(normd2real(x_n, bounds))`
# (lines 236/241), so `X_n(1:D,:)` holds NORMALISED coordinates, and
# `diam = sqrt(D)` (line 145) is the exact corner-to-corner diameter of
# `[0,1]^D` -- a constant of D, never an empirical data diameter.
#
# Two frames are supported and `NSMSettings.frame` picks between them.
#
#   `:pm1`     `[-1,1]^D`, `diam = 2*sqrt(D)`, `gamma_floor = 5e-7`. THE
#              DEFAULT, and what this arm ships in. It is the slate's
#              convention: the NN and GP arms both fit on
#              `BenchmarkDriver.normalize_inputs_pm1(bf, X)`, so every arm sees
#              bit-identical input coordinates and no comparison can turn on a
#              remap.
#   `:unit01`  `[0,1]^D`, `diam = sqrt(D)` (the literal opt_smgo_d.m:145),
#              `gamma_floor = 1e-6` (the literal line 148/303). The
#              LITERAL-PORT frame, kept selectable and used by the equivalence
#              gate, which compares against `nsm_reference.jl` in the frame the
#              two ports agree on. Reach it with
#              `nsm_normalize_unit01(X, bf.lb, bf.ub)` (identical arithmetic to
#              `BenchmarkDriver.normalize_inputs`).
#
# The frame choice is a slate-wide convention, NOT a fidelity question: both
# are faithful, because `diam` enters only through `ball_rad = rad*diam/2`
# (line 260) and is rescaled with the box. `:unit01` is merely the one whose
# constants can be read off the MATLAB unchanged.
#
# Queries are NOT clamped, so a query outside the training box simply gets a
# normalised coordinate outside the frame's box and a distance that may exceed
# `diam`; that is harmless for the estimator, which never uses `diam` at
# predict time. Under this benchmark's samplers pool and queries share
# `bf.lb`/`bf.ub`, so the situation does not arise in practice.
#
# `NSMSettings.frame` records the choice so it lands in the settings hash,
# `NSMModel.frame` records it so a fitted model says which coordinates it
# lives in, and `fit_nsm` CHECKS the training coordinates against the declared
# frame's box (see below).
#
# WHAT THE CHOICE COSTS, MEASURED. With `diam` AND `gamma_floor` both rescaled,
# the two frames are mathematically equivalent: `eps` is identical, `gamma` is
# exactly halved, and the cone product `gamma*d` is unchanged (both factors
# scale by exact powers of two). Numerically the equivalence is only as good as
# the `x -> 2x-1` map, which is exact when `2x-1` is representable and lossy
# near the lower face of the box. Over 15 datasets x 400 queries x 3 quantities
# (VERIFIED): dyadic coordinates (`rand()`) agree BIT-FOR-BIT, 13500/13500;
# `[-pi,pi]` coordinates normalised to `[0,1]` agree bit-for-bit on 8299/13500
# and to 2.7e-14 relative otherwise; coordinates clustered in `[0,1e-3]` agree
# on 2831/13500 and only to 1.5e-10 relative. So shipping `:pm1` is a
# defensible choice but not a free one: min-max normalisation already lands in
# `[0,1]`, so `:unit01` would apply no remap at all, whereas `:pm1` pays the
# `x -> 2x-1` rounding above. Slate symmetry is what buys that cost. This is
# NOT asserted in the gate (it would be false on non-dyadic data); probe G
# asserts the exact relations -- same eps, RAW gamma ratio exactly 0.5, floor
# 1e-6 vs 5e-7 -- which do hold unconditionally.
#
# THE FRAME IS LOAD-BEARING AND WAS PREVIOUSLY UNENFORCED. `diam` sets the
# whole ball ladder, so fitting on coordinates that are not in the declared box
# runs without error and returns numbers that look entirely reasonable while
# being badly wrong. Measured on ishigami (n = 500, z-scored y): the correct
# `[0,1]^3` fit gives eps = 0.110, gamma = 18.0; the SAME call on raw
# `[-pi,pi]^3` coordinates gives eps = 0.0053, gamma = 3.05, and on `[-1,1]^3`
# coordinates eps = 0.069, gamma = 9.22. `fit_nsm` therefore rejects training
# coordinates that lie outside the declared frame's box (`check_frame=false`
# opts out for a deliberately custom `diam`). In SMGO-Delta this failure mode
# cannot exist -- `[0,1]^D` is structural there -- so the guard is a port
# addition, not a transcription of anything.
#
# `gamma` consequently carries units of (fitted y) per (unit length IN THE
# FITTED FRAME) and `eps` units of (fitted y). Predict-time distances MUST be
# evaluated in the same frame the model was fitted in or the cone width is
# silently wrong -- `nsm_predict` reads geometry only from `m.X`, and the
# caller is responsible for normalising query points the same way.
#
# `gamma_floor` (opt_smgo_d.m:148/303) is DIMENSIONAL in both x and y: it is a
# raw-objective value per unit `[0,1]^D` length. It is therefore rescaled with
# the frame -- `nsm_frame_gamma_floor` returns `1e-6` for `:unit01` and `5e-7`
# for `:pm1`, so that a `:pm1` fit is the exact image of the `:unit01` fit.
# Without that rescaling the two frames are NOT equivalent: on the n = 2
# fixture where every anchor falls back to the floor, both frames returned
# gamma = 1e-6 (ratio 1.0, where frame-equivalence demands 0.5) and the cone
# width at a query differed by exactly 2x. It is NOT rescaled in y; see below.
#
# ==============================================================================
# y IS NOT TRANSFORMED IN THIS MODULE -- the caller does it, as for every arm
#
# This module fits on whatever `y` it is handed and returns `pred`/`lower`/
# `upper` in those same units. It does NOT z-score and it does NOT invert a
# transform. That is not an omission: it is the slate's trainer contract.
# `GPTrainer.train_gp` and the NN trainer are the same -- `test/smoke_data.jl`
# computes `yt = BenchmarkDriver.target_zscore(prep.y_train_noisy)`, fits on
# `yt.normalized`, and inverts with `yt.mean + yt.std * value` on the
# prediction AND on both bounds. An earlier revision of this header announced
# that NSM itself would do this. It never did; the sentence is deleted rather
# than left describing code that is not here.
#
# WHAT THE Z-SCORING DOES TO `gamma_floor`, since it is the one non-equivariant
# constant. Scaling `y` by `1/sigma` scales `eps` and every ratio at line 300
# by `1/sigma` as well, so NSM is exactly scale-equivariant in y -- EXCEPT that
# `gamma_floor` is an absolute constant in the source's RAW-objective units and
# does not move. Measured on the n = 2 fixture (y scaled by `a`, prediction
# divided back by `a`): 1.49999995 at a = 1, 1.49995097 at a = 1e-3,
# 1.4999999999510 at a = 1e3 -- not constant, as exact equivariance would
# require.
#
# The floor binds only when SOME anchor has no eligible partner, i.e. when the
# y-spread around it is <= 2*eps. On every cell of this benchmark eps came out
# 0.05-0.7 in z-units against a y-range of several, so it does not bind and the
# z-scoring is harmless HERE. A caller who needs the source's raw-y semantics
# under a y-rescaling passes `y_scale` to `nsm_settings` (it divides
# `gamma_floor` by the same factor y was divided by), and the rescaled floor
# lands in the settings hash like any other field.
#
# ==============================================================================
# SANCTIONED DEVIATION -- KD-tree neighbour search in `nsm_eps`
#
# opt_smgo_d.m:261-265 materialises a dense `iter x iter` ball-membership
# matrix. At this benchmark's largest cell (ishigami 50^3 grid, n_train =
# 125,000) that is 1.56e10 entries: ~15.6 GB as `Bool`, ~125 GB as the
# `double` the MATLAB actually allocates. It is not portable.
#
# The neighbour search is therefore reimplemented with a KD-tree
# (NearestNeighbors.jl, already a `test/ml` dependency). It yields NUMERICALLY
# IDENTICAL per-point neighbour counts and per-point max |dy|:
#
#   * the KD-tree is used only as a CANDIDATE GENERATOR, queried at a radius
#     inflated by one part in 1e12 (the tree's own test compares SQUARED
#     distances, `d^2 <= r^2`, which can disagree with MATLAB's
#     `vecnorm(...) <= ball_rad` in the last ulp at an exact tie); every
#     candidate is then re-tested with the literal `sqrt(sum of squares) <=
#     ball_rad` predicate of line 263, so membership matches the dense port;
#   * self-membership is dropped by index (`idx == k`), reproducing
#     `ball_idx(k,k) = 0` at line 264 and, crucially, NOT dropping a distinct
#     sample that happens to sit at the same coordinates (MATLAB zeroes only
#     the diagonal);
#   * the radius ladder (line 259) is resolved from the minimum pairwise
#     distance, computed once (`knn` with k = 2, then the same literal distance
#     formula) and compared against each rung's `ball_rad`. That is the
#     `sum(ball_idx,'all') > 0` break test of lines 266-268 without
#     materialising any matrix. `knn` ranks candidates by the TREE's metric,
#     which can disagree with `_pairdist` in the last ulp, so the knn distance
#     is only an upper bound on the true minimum; when it lands within
#     `_LADDER_ULP` ABOVE a rung, `_any_pair_within` re-tests that rung
#     exhaustively with the literal line-263 predicate. Without that rescue the
#     break test was the one place the KD-tree deviation was not self-
#     correcting (the membership test has always been ulp-guarded);
#   * the numerator is accumulated in ascending point index and the ordered
#     pair count is summed the same way. That makes the result independent of
#     the block size and the thread count; it is NOT a claim that it matches
#     MATLAB's `sum`, which is blocked/vectorised rather than a left-to-right
#     scalar accumulation.
#
# WHAT IS ACTUALLY VERIFIED, AND WHAT IS NOT. No MATLAB was executed at any
# point: nothing here is established against `vecnorm`, `sum`, `min` or `max`
# as MathWorks implements them, only against their DOCUMENTED semantics
# (NaN-omitting min/max, inclusive `<=`, the colon's `base + k*step`).
# `nsm_equivalence_tests.jl` checks this module against `nsm_reference.jl`, a
# dense literal transcription of opt_smgo_d.m:257-306 written independently,
# with its own reductions and its own distance expression. Across 29 cells,
# 4554 evaluation points and 6032 band checks, `eps`, `gamma`, `upper`,
# `lower` and `pred` all agree BIT-FOR-BIT (worst relative difference exactly
# 0, against a 1e-12 gate).
#
# That is an agreement between two Julia ports, NOT a measurement against
# MATLAB, and one caveat is worth stating precisely because an earlier revision
# of this file got it backwards. The reference measures distance with
# `LinearAlgebra.norm`, which was described here as "rescaling by maxabs" and
# blamed for a ~2e-16 residual. Both halves were wrong: Julia's `generic_norm2`
# takes an UNSCALED fast path whenever `dim*maxabs^2` is finite and nonzero, so
# for any data in this benchmark's magnitude range it returns exactly the same
# double as the explicit loop below (VERIFIED: 0 differences in 200,000 random
# vectors at each of D = 2,3,5,8; they diverge only at magnitudes ~1e160, where
# the rescaling path engages). The real source of the old residual was `@simd`
# licensing FMA contraction in this module's own kernel -- see `_sqdist`.
#
# The consequence for the gate is honest but limited: the two ports agree
# exactly, AND their distance kernels coincide bit-for-bit in this regime, so
# the gate does not independently corroborate the distance kernel against
# `vecnorm`. That corroboration is unavailable without running MATLAB and is
# not claimed anywhere in this file.
#
# Ball MEMBERSHIP is the one place where last-ulp distance differences are
# amplified rather than absorbed (a tie at `ball_rad` flips a pair in or out of
# the ball, and `P` can be as small as 2 at the breaking rung, so `eps` can move
# by tens of percent). That is inherent to the source's `<=` test, not to this
# port; it needs an exact tie to trigger, which this benchmark's Sobol and grid
# pools do not produce.
#
# `nsm_gamma` is O(n^2) BY CONSTRUCTION (line 298 has no ball restriction --
# gamma is a GLOBAL Lipschitz estimate, unlike the LOCAL eps; that asymmetry is
# deliberate in the source and is preserved here). It is computed streaming in
# index blocks with no n x n matrix ever allocated, exploiting the exact
# symmetry of the ratio in (i,j) to visit each unordered pair once. Both
# estimators are per-cell constants, computed once in `fit_nsm`.
#
# Public API: `nsm_settings`, `nsm_normalize_unit01`, `nsm_normalize_pm1`,
# `nsm_eps`, `nsm_gamma`, `fit_nsm`, `nsm_predict`, `nsm_predict_batch`,
# `nsm_status`, and the three-piece arm contract surface
# `nsm_settings_string` / `nsm_model_type` / `nsm_interval_kind` (plus
# `NSM_INTERVAL_SEMANTICS`), which mirrors `SurrogateArms`'
# `surrogate_settings_string` / `surrogate_model_type` /
# `surrogate_interval_kind` so that `test/smoke_data.jl` needs no per-arm
# special case when this arm is wired in.

module NSMTrainer

# NearestNeighbors.jl is the only dependency (already in test/ml/Project.toml).
# No LinearAlgebra, no Statistics: every distance is written out as an explicit
# sqrt-of-sum-of-squares, accumulated left to right with no fused multiply-add,
# so that the accumulation order really is fixed, visible, and the same on
# every machine (see `_sqdist` on why `@simd` was removed).
#
# `LinearAlgebra.norm` is NOT avoided because it computes something different:
# for any data at this benchmark's magnitudes it returns exactly the same
# double (VERIFIED -- Julia's `generic_norm2` only engages its maxabs rescaling
# when `dim*maxabs^2` would overflow, around 1e160). It is avoided because the
# loop below allocates nothing on a `dim x n` column pair and because the
# arithmetic is then visible in this file rather than delegated. WHICH of the
# two matches MATLAB's `vecnorm` bit-for-bit is NOT established here: no MATLAB
# was run, and MathWorks documents `vecnorm`'s value, not its summation.
using NearestNeighbors

export NSMModel, NSMSettings, fit_nsm, nsm_predict, nsm_predict_batch,
       nsm_eps, nsm_gamma, nsm_settings, nsm_normalize_unit01, nsm_normalize_pm1,
       nsm_frame_diam, nsm_frame_box, nsm_frame_gamma_floor, nsm_radius_ladder,
       nsm_status, nsm_settings_string, nsm_model_type, nsm_interval_kind,
       NSM_INTERVAL_SEMANTICS, NSM_SOURCE_PROVENANCE, NSM_DEFAULT_GAMMA_FACTOR, set_gamma_factor!

# ------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------

"""
    NSM_DEFAULT_GAMMA_FACTOR

`2.5`, the multiplier applied to the ported `fgam` estimate. It is Diona's
`L_H_factor`, adopted here so the two arms carry the SAME safety cushion on
their estimated smoothness constant; before this, NSM used its raw estimate
while Diona inflated its own by 2.5. Symmetry is the fairness argument, not
any property of NSM.

It is NOT part of `opt_smgo_d.m`. `gamma_factor = 1.0` is the source.
"""
# Per-campaign knob (solver.gamma_factor), mirroring Diona's L_H_factor so the
# two methods carry the SAME safety multiplier. A Ref, set once before any cell
# runs; it enters nsm_settings_string and therefore the :nsm settings hash, so
# campaigns at different factors key to different fragments.
const NSM_GAMMA_FACTOR_DEFAULT = 2.5
const NSM_GAMMA_FACTOR = Ref(NSM_GAMMA_FACTOR_DEFAULT)
NSM_DEFAULT_GAMMA_FACTOR() = NSM_GAMMA_FACTOR[]

"""
    set_gamma_factor!(v::Real)

Set the NSM Lipschitz safety multiplier for this process. Must be called before
any cell is fitted or any settings hash is computed.
"""
function set_gamma_factor!(v::Real)
    v > 0 || throw(ArgumentError("set_gamma_factor!: gamma_factor must be > 0, got $v"))
    NSM_GAMMA_FACTOR[] = Float64(v)
end

"""
    NSMSettings(seed, radii, gamma_floor, frame, gamma_factor)

Fixed configuration for the NSM arm. Everything here except `gamma_factor` is
lifted from `opt_smgo_d.m`, and none of it is tuned per cell or per dimension
(the comparator slate's fairness argument is that every arm runs ONE fixed
config).

- `seed`: carried only so the arm can fill `ArmCellResult.model_seed`. NSM is
  fully deterministic -- there is no sampling, no initialisation and no
  stochastic optimisation anywhere in it.
- `radii`: the ball-radius ladder, `0.1:0.1:1.0` (opt_smgo_d.m:259). Exactly
  ten values; the effective radius is `rad * diam / 2` (line 260), so the
  LARGEST ball in the ladder is `diam/2`, i.e. half the box diameter. Do not
  extend the ladder -- see `nsm_eps` on what happens when it is exhausted.
  Built by `nsm_radius_ladder`, NOT by `collect(0.1:0.1:1.0)`; see there.
- `gamma_floor`: `1e-6` in the `:unit01` frame (opt_smgo_d.m:303, matching
  `fgam_0` at line 148). A PER-ANCHOR fallback, not a global clamp; see
  `nsm_gamma`. It is a DIMENSIONAL constant (raw objective per unit
  `[0,1]^D` length), so it is rescaled with the frame by
  `nsm_frame_gamma_floor` and would also have to be rescaled by any rescaling
  of y -- see the coordinate-frame note at the top of this file.
- `frame`: the coordinate frame the model is fitted and queried in. `:pm1`
  (the DEFAULT) is the slate's `normalize_inputs_pm1` convention, shared with
  the NN and GP arms: `[-1,1]^D`, `diam = 2*sqrt(D)`, `gamma_floor = 5e-7`.
  `:unit01` is the literal-port frame (`[0,1]^D`, `diam = sqrt(D)`,
  `gamma_floor = 1e-6`) and is what the equivalence gate runs in. `fit_nsm`
  checks the training coordinates against it.
- `gamma_factor`: the multiplier applied to the ported `fgam` (see
  `NSM_DEFAULT_GAMMA_FACTOR`). `1.0` is the source. It is applied by `fit_nsm`
  as a WRAPPER -- `nsm_gamma` never sees it -- so that the estimator stays
  bit-exact against `nsm_reference.jl` at 1.0. `eps` is not multiplied.
"""
struct NSMSettings
    seed::Int
    radii::Vector{Float64}
    gamma_floor::Float64
    frame::Symbol
    gamma_factor::Float64
end

# Pre-`gamma_factor` positional constructor. Defaults to 1.0, i.e. the raw
# ported estimate, so any caller written against the 4-field struct keeps the
# source's semantics rather than silently acquiring the 2.5 cushion.
NSMSettings(seed, radii, gamma_floor, frame) =
    NSMSettings(seed, radii, gamma_floor, frame, 1.0)

"""
    nsm_radius_ladder() -> Vector{Float64}

The ten ball-ladder radii of `for rad = 0.1:0.1:1.0` (opt_smgo_d.m:259), with
MATLAB's colon arithmetic.

This is NOT `collect(0.1:0.1:1.0)`. MATLAB's colon materialises `base + k*step`
in double precision; Julia's range arithmetic is `TwicePrecision`, so it
returns the correctly-rounded double of the EXACT decimal instead. The two
disagree by one ulp at two of the ten rungs (VERIFIED in this session):

    rung 3   Julia 0.29999999999999999   MATLAB 0.30000000000000004
    rung 7   Julia 0.69999999999999996   MATLAB 0.70000000000000007

which moves `ball_rad = rad*diam/2` by an ulp at those rungs and can therefore
flip an exact tie in the INCLUSIVE membership test at line 263. Immaterial on
any real dataset, corrected because it is free to correct. Note the form
matters: `0.1*k` is a third answer again, differing from both at rung 6.
"""
nsm_radius_ladder() = [0.1 + (k - 1) * 0.1 for k in 1:10]   # opt_smgo_d.m:259

"""
    nsm_settings(dim::Int; frame::Symbol=:pm1,
                 gamma_factor::Float64=NSM_DEFAULT_GAMMA_FACTOR(),
                 y_scale::Float64=1.0) -> NSMSettings

The one fixed NSM configuration. `dim`-indexed purely for signature symmetry
with the other arms' settings constructors (`ComparatorRegistry.settings_for`,
`gp_settings`, ...); the returned value is identical for every dimension --
nothing in NSM is dimension-scheduled.

`frame` defaults to `:pm1`, the slate's `normalize_inputs_pm1` convention, and
carries the frame-consistent `diam` AND `gamma_floor` together. `:unit01`, the
literal-port frame, stays selectable and is what the equivalence gate uses.
Passing data in one frame with the other frame's settings is rejected by
`fit_nsm` rather than silently mis-scaled.

`gamma_factor` defaults to `NSM_DEFAULT_GAMMA_FACTOR` (2.5, Diona's
`L_H_factor`). Pass `1.0` for the literal source behaviour.

`y_scale` is the factor the caller DIVIDED `y` by before fitting (e.g.
`target_zscore(...).std`). `gamma_floor` is dimensional in y and the source's
value is in RAW-objective units, so it is divided by the same factor to keep
the source's semantics under a rescaled target. The default `1.0` leaves it
exactly as `nsm_frame_gamma_floor` returns it; see the y-transform note in
this file's header for why it does not bind on any cell of this benchmark
either way.
"""
function nsm_settings(dim::Int; frame::Symbol=:pm1,
                      gamma_factor::Float64=NSM_DEFAULT_GAMMA_FACTOR(),
                      y_scale::Float64=1.0)
    dim >= 1 || throw(ArgumentError("nsm_settings: dim must be >= 1, got $dim"))
    isfinite(gamma_factor) && gamma_factor > 0 ||
        throw(ArgumentError("nsm_settings: gamma_factor must be finite and positive, got $gamma_factor"))
    isfinite(y_scale) && y_scale > 0 ||
        throw(ArgumentError("nsm_settings: y_scale must be finite and positive, got $y_scale"))
    floor_ = nsm_frame_gamma_floor(frame)
    y_scale == 1.0 || (floor_ = floor_ / y_scale)
    return NSMSettings(0, nsm_radius_ladder(), floor_, frame, gamma_factor)
end

# ------------------------------------------------------------------------------
# Arm contract surface
#
# `test/smoke_data.jl` asks each arm for exactly three things per cell:
# a canonical settings string (which it hashes into `settings_hash` and the
# checkpoint identity), a `model_type` label, and an `interval_kind` symbol.
# `SurrogateArms` exposes them as `surrogate_settings_string` /
# `surrogate_model_type` / `surrogate_interval_kind`; these are the same three,
# named for this arm, so wiring NSM in needs no new shape.
# ------------------------------------------------------------------------------

"""
    NSM_SOURCE_PROVENANCE

What this arm is a port OF, pinned into the settings hash so that a change of
source or of ported region invalidates every recorded hash instead of passing
unnoticed. There is no package version to record -- this arm has no upstream
package, which is why it exists in the first place.
"""
const NSM_SOURCE_PROVENANCE =
    "Nonlinear Set Membership (Milanese & Novara, Automatica 40(6):957-975, 2004). " *
    "No Julia package implements it; this arm is a hand-written port of " *
    "smgo-delta/opt_smgo_d.m -- feps (lines 257-279), fgam (lines 295-306) and the " *
    "upper/lower cone bounds (lines 482-483). Sanctioned deviations: KD-tree " *
    "candidate generation in feps (the dense iter x iter ball matrix is ~125 GB at " *
    "n = 125,000), MATLAB colon arithmetic for the radius ladder, a coordinate-frame " *
    "guard, and gamma_factor, which is not in the source at all."

"""
    NSM_INTERVAL_SEMANTICS

What `[lower, upper]` on this arm does and does not mean, in the style of
`ComparatorRegistry.GP_INTERVAL_SEMANTICS`. Reported alongside the arm's rows
so a reader of the coverage column cannot mistake the band for a certificate.

This string replaced an `interval_kind` of `:certified_lipschitz`. That label
was refuted by this benchmark's own measurements; see the header of this file
and `nsm_interval_kind`.
"""
const NSM_INTERVAL_SEMANTICS =
    "NSM interval = [max_i(y_i - eps - gamma*||x-x_i||), min_i(y_i + eps + gamma*||x-x_i||)], " *
    "the set-membership band implied by a noise level eps and a Lipschitz constant gamma " *
    "that are BOTH ESTIMATED FROM THE TRAINING DATA (opt_smgo_d.m:257-279 and 295-306); " *
    "no a priori noise bound or Lipschitz constant is supplied to this arm, so the band " *
    "carries NO GUARANTEE of containing the true function. gamma is a maximum over the " *
    "finitely many OBSERVED secant slopes, deflated by 2*eps, hence at best a LOWER BOUND " *
    "on the true Lipschitz constant -- the cone therefore under-covers between samples " *
    "even at zero noise (measured: query coverage bottoms at 0.97925 across the 36 Part-1 " *
    "cells at noise 0). eps likewise undershoots the injected noise half-width on every " *
    "noisy cell (ratio 0.0057-0.648, mean 0.339), and the band consequently excludes the " *
    "true function at 51.1-99.35% of TRAINING anchors there, while query-point coverage " *
    "still reads 0.955-1.000. gamma is multiplied by gamma_factor (default 2.5, matching " *
    "Diona's L_H_factor); eps is not scaled. See test/ml/baselines/nsm.jl."

"""
    nsm_interval_kind() -> Symbol

`:set_membership_estimated` -- the `ArmCellResult.interval_kind` label for this
arm.

NOT `:certified_lipschitz`, which is what this file used to declare. Under this
benchmark's protocol every arm estimates its own constants from the training
data and none is handed an oracle, so no arm's interval is certified; and the
measurements in this file's header refute the label directly (the band excludes
the true function at up to 99.35% of training anchors, and query coverage at
zero noise bottoms at 0.97925, not 1.000). The symbol says what the band IS --
a set-membership band built from estimated constants -- and
`NSM_INTERVAL_SEMANTICS` carries the rest.
"""
nsm_interval_kind() = :set_membership_estimated

"""
    nsm_model_type() -> String

`ArmCellResult.model_type` label. There is no package type to name here: the
estimator is this file. See `NSM_SOURCE_PROVENANCE` for what it is a port of.
"""
nsm_model_type() = "NSMTrainer.NSMModel (hand-written NSM port of opt_smgo_d.m)"

# Canonical serialisation, byte-identical in FORM to
# `ComparatorRegistry.canonical_serialize` and `SurrogateArms`' `_ser`, so the
# three arms' hash styles match and a hash is comparable across them.
_ser(x::Bool) = x ? "true" : "false"
_ser(x::Integer) = string(Int(x))
_ser(x::Float64) = "f64:0x" * string(reinterpret(UInt64, x); base=16, pad=16)
_ser(x::AbstractFloat) = _ser(Float64(x))
_ser(x::AbstractString) = "s:" * repr(String(x))
_ser(x::Symbol) = "sym:" * String(x)
_ser(v::AbstractVector) = "[" * join((_ser(x) for x in v), ",") * "]"

"""
    nsm_settings_string(s::NSMSettings) -> String

Deterministic canonical string for `bytes2hex(sha256(...))`, mirroring
`SurrogateArms.surrogate_settings_string`.

Floats are serialised by their exact bit pattern, so the one-ulp radius-ladder
detail that `nsm_radius_ladder` exists to preserve is inside the hash: swapping
in `collect(0.1:0.1:1.0)` changes it. `gamma_factor` and the frame-rescaled
`gamma_floor` are in there too, which is the point of putting the factor in the
settings rather than in the estimator.

No package version is recorded because there is no package; `NSM_SOURCE_PROVENANCE`
takes its place, so a change of ported region invalidates the hash.
"""
function nsm_settings_string(s::NSMSettings)
    return "NSMTrainer.NSMSettings{" *
           "arm=" * _ser(:nsm) *
           ",seed=" * _ser(s.seed) *
           ",frame=" * _ser(s.frame) *
           ",radii=" * _ser(s.radii) *
           ",gamma_floor=" * _ser(s.gamma_floor) *
           ",gamma_factor=" * _ser(s.gamma_factor) *
           ",interval_kind=" * _ser(nsm_interval_kind()) *
           ",provenance=" * _ser(NSM_SOURCE_PROVENANCE) * "}"
end

"""
    nsm_frame_diam(frame::Symbol, dim::Int) -> Float64

Box diameter for a coordinate frame. `opt_smgo_d.m:145` hard-codes
`diam = sqrt(D)`, the corner-to-corner diameter of `[0,1]^D`; `diam` enters the
estimator only through `ball_rad = rad * diam/2` (line 260), so the source
semantics are preserved iff `diam` is the TRUE diameter of whatever box the
model is fitted in.
"""
function nsm_frame_diam(frame::Symbol, dim::Int)
    frame === :unit01 && return sqrt(dim)          # opt_smgo_d.m:145
    frame === :pm1 && return 2 * sqrt(dim)         # same constant re-derived for [-1,1]^D
    throw(ArgumentError("nsm_frame_diam: unsupported frame $(frame) (expected :unit01 or :pm1)"))
end

"""
    nsm_frame_box(frame::Symbol) -> (lo::Float64, hi::Float64)

The per-coordinate extent of a frame's box: `(0.0, 1.0)` for `:unit01`,
`(-1.0, 1.0)` for `:pm1`. Used by `fit_nsm` to check that the training
coordinates really are in the frame the settings declare.
"""
function nsm_frame_box(frame::Symbol)
    frame === :unit01 && return (0.0, 1.0)
    frame === :pm1 && return (-1.0, 1.0)
    throw(ArgumentError("nsm_frame_box: unsupported frame $(frame) (expected :unit01 or :pm1)"))
end

"""
    nsm_frame_gamma_floor(frame::Symbol) -> Float64

The per-anchor Lipschitz fallback of opt_smgo_d.m:303 (`1e-6`, matching
`fgam_0` at line 148), expressed in the units of `frame`.

`gamma` has units of y per unit LENGTH, so the floor is not frame-invariant:
`[-1,1]^D` doubles every distance and therefore halves every ratio at line 300,
and a floor carried over verbatim would sit twice as high relative to the
ratios it competes with in the `max` at line 306. `5e-7` is the `[-1,1]^D`
image of the source's `1e-6`, so a `:pm1` fit is the exact image of the
`:unit01` fit. VERIFIED this session that the un-rescaled floor breaks that
equivalence on the n = 2 fixture (both frames returned exactly `1e-6`, i.e. a
ratio of 1.0 where 0.5 was required, and the cone widths differed by 2x).

The floor is dimensional in y as well; see the header. Nothing rescales it
there, because a model is not told what transform produced its `y`.
"""
function nsm_frame_gamma_floor(frame::Symbol)
    frame === :unit01 && return 1.0e-6             # opt_smgo_d.m:148/303, verbatim
    frame === :pm1 && return 5.0e-7                # 1e-6 re-expressed per unit [-1,1]^D length
    throw(ArgumentError("nsm_frame_gamma_floor: unsupported frame $(frame) (expected :unit01 or :pm1)"))
end

"""
    nsm_normalize_unit01(X, lb, ub) -> Matrix{Float64}

Min-max map `dim x n` PHYSICAL-domain inputs to `[0,1]^dim`. Algorithmically
identical to `BenchmarkDriver.normalize_inputs`, reimplemented here so this
module stays self-contained (it must not depend on `src/`, and depending on the
driver would drag the whole benchmark environment into unit tests of the
estimators).

Points outside `[lb, ub]` map outside `[0,1]` and are NOT clamped -- see the
coordinate-frame note at the top of this file.
"""
function nsm_normalize_unit01(
    X::AbstractMatrix{Float64}, lb::AbstractVector{Float64}, ub::AbstractVector{Float64}
)
    size(X, 1) == length(lb) == length(ub) ||
        throw(DimensionMismatch("nsm_normalize_unit01: X has $(size(X,1)) rows, lb/ub have $(length(lb))/$(length(ub))"))
    return (X .- lb) ./ (ub .- lb)
end

"""
    nsm_normalize_pm1(X, lb, ub) -> Matrix{Float64}

Map `dim x n` PHYSICAL-domain inputs to `[-1,1]^dim`, the `:pm1` frame this arm
defaults to. Written as `2 .* (X .- lb) ./ (ub .- lb) .- 1`, which is the
EXACT expression of `BenchmarkDriver.normalize_inputs_pm1` -- the arithmetic is
duplicated deliberately, character for character, so that the coordinates this
arm fits on are bit-identical to the ones the NN and GP arms fit on. (Writing
it as `2 * nsm_normalize_unit01(...) - 1` would be algebraically the same and
is NOT guaranteed to be bit-identical.) Reimplemented here rather than imported
so this module stays free of `src/` and of the benchmark driver.

Not clamped; see `nsm_normalize_unit01`.
"""
function nsm_normalize_pm1(
    X::AbstractMatrix{Float64}, lb::AbstractVector{Float64}, ub::AbstractVector{Float64}
)
    size(X, 1) == length(lb) == length(ub) ||
        throw(DimensionMismatch("nsm_normalize_pm1: X has $(size(X,1)) rows, lb/ub have $(length(lb))/$(length(ub))"))
    return 2 .* (X .- lb) ./ (ub .- lb) .- 1
end

# ------------------------------------------------------------------------------
# Distance kernel
# ------------------------------------------------------------------------------
#
# `vecnorm` over columns of `X_n(1:D, :)` (opt_smgo_d.m:263, 301) is
# sqrt(sum of squared coordinate differences). Written out explicitly (rather
# than via `norm(view(X,:,i) - view(X,:,j))`) so the hot loops allocate
# nothing; `X` is `dim x n` so both columns are contiguous.
#
# NO `@simd` HERE, DELIBERATELY. `@simd` licenses the compiler both to
# reassociate the reduction and to fuse `s += delta*delta` into an FMA, which
# skips the rounding of `delta*delta`. VERIFIED this session: with `@simd` the
# result is `fma(d2, d2, d1*d1)` rather than `d1*d1 + d2*d2` on 164 of 1000
# random D=2 pairs, and at D in 2..8 it disagrees with the sequential loop on
# 8-15% of random pairs by up to 2.2e-16 relative. That made the kernel depend
# on the host's FMA support and falsified this file's own claim that the
# accumulation order is fixed. Removing it costs ~13% of this kernel at D=3 and
# nothing at D=8 (measured), against a `sqrt` and the memory traffic.

@inline function _sqdist(X::AbstractMatrix{Float64}, i::Int, j::Int)
    s = 0.0
    @inbounds for d in axes(X, 1)
        delta = X[d, i] - X[d, j]
        s += delta * delta
    end
    return s
end

@inline _pairdist(X::AbstractMatrix{Float64}, i::Int, j::Int) = sqrt(_sqdist(X, i, j))

"""
    _sqdist_point(X, i, x) -> Float64

Squared distance between training column `i` and a free query point `x`. Same
accumulation as `_sqdist` (left to right, no `@simd`, no FMA -- see there),
shared by `nsm_predict` so that predict-time and fit-time geometry are computed
by one kernel (a second, subtly different distance expression would move the
cone width in the last ulp for no reason).
"""
@inline function _sqdist_point(X::AbstractMatrix{Float64}, i::Int, x::AbstractVector{Float64})
    s = 0.0
    @inbounds for d in axes(X, 1)
        delta = X[d, i] - x[d]
        s += delta * delta
    end
    return s
end

"""
    _index_blocks(n, block) -> Vector{UnitRange{Int}}

Split `1:n` into contiguous blocks of at most `block` indices. Used to bound
the working set of the threaded loops (and, in `nsm_gamma`, to give the dynamic
scheduler enough chunks to balance the triangular workload).
"""
function _index_blocks(n::Int, block::Int)
    block >= 1 || throw(ArgumentError("_index_blocks: block must be >= 1, got $block"))
    return [((b - 1) * block + 1):min(b * block, n) for b in 1:cld(n, block)]
end

# ------------------------------------------------------------------------------
# eps -- the noise-level estimate (opt_smgo_d.m:257-279)
# ------------------------------------------------------------------------------

# The KD-tree ranks candidates by its OWN metric (squared, Distances.jl), which
# can disagree with `_pairdist` in the last ulp. Every use of the tree in this
# file is therefore a CANDIDATE GENERATOR queried at a radius inflated by this
# slack and then filtered with the literal opt_smgo_d.m:263 predicate.
# 1e-12 is ~1000x the worst plausible disagreement (O(dim) roundings at
# ~1.1e-16 each) and is far below any radius gap in the ladder, so it can only
# add candidates, never drop one.
const _TREE_ULP_SLACK = 1.0e-12

"""
    _min_pairwise_distance(tree, X) -> Float64

An UPPER BOUND on the minimum distance between two DISTINCT samples (by index
-- coincident samples at distance 0 count, matching the source, which only
excludes the diagonal), exact except at a tree-metric near-tie.

`knn` with k = 2 gives each point's nearest other sample and the distance is
then RECOMPUTED with `_pairdist`, so the returned value is always an ACHIEVED
`_pairdist` of some off-diagonal pair. It can exceed the true minimum only if
the tree ranks a near-tied candidate the other way round, which is bounded by
`_TREE_ULP_SLACK`; `_ladder_ball_rad` closes that gap.
"""
function _min_pairwise_distance(tree::KDTree, X::AbstractMatrix{Float64})
    n = size(X, 2)
    idxs, _ = knn(tree, X, 2, true)
    dmin = Inf
    @inbounds for k in 1:n
        for idx in idxs[k]
            idx == k && continue
            d = _pairdist(X, idx, k)
            d < dmin && (dmin = d)
        end
    end
    return dmin
end

"""
    _any_pair_within(tree, X, r) -> Bool

Is there an off-diagonal pair at distance `<= r`? The literal
`sum(ball_idx,'all') > 0` test of opt_smgo_d.m:266-268, evaluated with the
line-263 predicate, early-exiting on the first hit and never materialising the
`iter x iter` matrix. Single-threaded on purpose: it runs only in the rescue
path of `_ladder_ball_rad`, where it usually exits on the first point.
"""
function _any_pair_within(tree::KDTree, X::AbstractMatrix{Float64}, r::Float64)
    dim, n = size(X)
    buf = Int[]
    point = Vector{Float64}(undef, dim)
    r_query = r * (1 + _TREE_ULP_SLACK)
    @inbounds for k in 1:n
        for d in 1:dim
            point[d] = X[d, k]
        end
        empty!(buf)
        inrange!(buf, tree, point, r_query)
        for idx in buf
            idx == k && continue                 # opt_smgo_d.m:264, diagonal only
            _pairdist(X, idx, k) <= r && return true   # INCLUSIVE, opt_smgo_d.m:263
        end
    end
    return false
end

"""
    _ladder_ball_rad(tree, X, radii, diam) -> Float64

The radius at which opt_smgo_d.m:259-268 breaks, or `NaN` if the ladder is
exhausted. `ball_rad = rad * diam/2` (line 260) grows over `radii` until some
off-diagonal pair lies within it (INCLUSIVE `<=`, line 263).

`dmin` decides the rung outright whenever it is clearly on one side of it. In
the narrow window where `dmin` sits ABOVE the rung but within
`_TREE_ULP_SLACK` of it -- the only regime where the tree's ranking could have
hidden a genuinely closer pair -- the rung is re-tested exhaustively with
`_any_pair_within`. The membership test in `nsm_eps` has always carried this
guard; the break test did not, which left it as the one place where the
sanctioned KD-tree deviation was not self-correcting.
"""
function _ladder_ball_rad(
    tree::KDTree, X::AbstractMatrix{Float64}, radii::AbstractVector{Float64}, diam::Float64
)
    dmin = _min_pairwise_distance(tree, X)
    @inbounds for rad in radii
        r = rad * diam / 2                       # opt_smgo_d.m:260
        dmin <= r && return r                    # INCLUSIVE, opt_smgo_d.m:263
        if dmin <= r * (1 + _TREE_ULP_SLACK) && _any_pair_within(tree, X, r)
            return r
        end
    end
    return NaN                                   # ladder exhausted; see `nsm_eps`
end

"""
    nsm_eps(X, y; diam=nsm_frame_diam(:unit01, size(X,1)), radii=nsm_radius_ladder(),
            tree=nothing, block=1024) -> Float64

`feps`, the NSM noise-level estimate -- a literal port of opt_smgo_d.m:257-279
with the dense `iter x iter` ball matrix replaced by a KD-tree (see the
sanctioned-deviation note at the top of this file).

`X` is `dim x n` in the model's own coordinate frame; `y` is the length-`n`
vector of targets the model is fitted on.

`diam` must be the diameter of the box THAT frame's coordinates live in -- it
sets the entire ball ladder, and the default is the `:unit01` value
`sqrt(dim)`, so a direct call on `[-1,1]^dim` coordinates must pass
`diam = nsm_frame_diam(:pm1, dim)` or it silently runs a half-size ladder.
`fit_nsm` always passes the value its settings imply and additionally checks
the coordinates; this function, being an exported estimator, checks nothing.

The algorithm, with source line numbers:

1. grow `ball_rad = rad * diam/2` over `radii` (lines 259-260) until some
   off-diagonal pair is within `ball_rad` (INCLUSIVE `<=`, line 263; the
   diagonal is dropped at line 264);
2. for each point `j` that has at least one in-ball neighbour (line 275), take
   `M_j = max |y_i - y_j|` over that point's in-ball neighbours (line 276);
3. return `sum_j M_j / (2 * P)` where `P = sum(ball_idx,'all')` is the count of
   ORDERED in-ball pairs (line 279).

The ordered/unordered asymmetry is real: the numerator has one term per active
POINT, the denominator counts ordered PAIRS, which reduces exactly to
`feps = mean_j(M_j) / (2 * nbar)` with `nbar` the mean ball occupancy. So the
estimate shrinks like `1/density` -- adding samples inside a fixed ball drives
`feps` toward zero even at unchanged noise. This is what the source computes
and it is ported verbatim; it is NOT the Milanese-Novara textbook estimator and
must not be described as "half the maximum local discrepancy" in any write-up
(that reading is correct only in the single-in-ball-pair case, where
`feps = |y_1 - y_2| / 2`).

Returns `0.0` for `n < 2` (`feps_0`, opt_smgo_d.m:153-154 -- the whole block is
guarded by `if iter > 1` at line 252).

Returns `NaN` when the radius ladder is EXHAUSTED, i.e. no two samples are
within `diam/2` of each other. MATLAB falls out of the `for` loop with an
all-zero `ball_idx`, an empty filtered numerator and a zero denominator, and
silently yields `0/0 = NaN` with no error and no warning; that NaN then
poisons `fgam` (every eligibility test at line 298 is false, so `fgam = 1e-6`)
and every prediction. This port reproduces the NaN faithfully rather than
substituting a repair value, so a cell where it triggers is VISIBLE downstream
as `:NSM_PREDICT_NONFINITE` on every query instead of silently reporting
numbers the reference would not have produced. It is reachable only for small
`n` at low `D` (e.g. D=2 with samples only at the four corners: min pairwise
distance 1.0 > sqrt(2)/2); by a packing bound it is impossible for large `n`,
and for D >= 5 even the 2^D hypercube corners sit inside the largest ball.
"""
function nsm_eps(
    X::AbstractMatrix{Float64},
    y::AbstractVector{Float64};
    diam::Float64=nsm_frame_diam(:unit01, size(X, 1)),
    radii::AbstractVector{Float64}=nsm_radius_ladder(),
    tree::Union{Nothing,KDTree}=nothing,
    block::Int=1024,
)
    dim, n = size(X)
    length(y) == n ||
        throw(DimensionMismatch("nsm_eps: X has $n columns but y has $(length(y)) entries"))
    dim >= 1 || throw(ArgumentError("nsm_eps: X must have at least one row"))
    # opt_smgo_d.m:252 -- the noise block is never entered on the first sample,
    # so feps keeps feps_0 = 0 (lines 153-154).
    n < 2 && return 0.0

    kdt = tree === nothing ? KDTree(X) : tree

    # opt_smgo_d.m:259-268 -- grow the ball until it contains an off-diagonal pair.
    ball_rad = _ladder_ball_rad(kdt, X, radii, diam)
    # Ladder exhausted: MATLAB falls through with an all-zero ball_idx and
    # divides 0 by 0 (lines 270-279). Reproduced, deliberately, as NaN.
    isnan(ball_rad) && return NaN

    counts = zeros(Int, n)                       # per-point |N(j)|
    maxdy = zeros(Float64, n)                    # per-point M_j; 0.0 where inactive
    # The KD-tree compares SQUARED distances, so query one part in 1e12 wide and
    # re-test every candidate with the literal line-263 predicate. Inflating can
    # only ADD candidates, never drop one, so the filtered set is exact.
    r_query = ball_rad * (1 + _TREE_ULP_SLACK)

    blocks = _index_blocks(n, block)
    Threads.@threads :dynamic for blk in blocks
        buf = Int[]
        point = Vector{Float64}(undef, dim)
        @inbounds for k in blk
            for d in 1:dim
                point[d] = X[d, k]
            end
            empty!(buf)
            inrange!(buf, kdt, point, r_query)
            cnt = 0
            mx = 0.0
            for idx in buf
                idx == k && continue             # opt_smgo_d.m:264, diagonal only
                _pairdist(X, idx, k) <= ball_rad || continue   # opt_smgo_d.m:263
                cnt += 1
                dy = abs(y[idx] - y[k])          # opt_smgo_d.m:276
                # `dy > mx` (rather than `max`) also reproduces MATLAB's
                # NaN-omitting `max`: a NaN comparison is false, so a NaN
                # target is skipped instead of poisoning M_j the way Julia's
                # `maximum` would.
                dy > mx && (mx = dy)
            end
            counts[k] = cnt
            maxdy[k] = mx
        end
    end

    # opt_smgo_d.m:279. Numerator: one term per ACTIVE point (line 275 guard;
    # the `ns_lkup > 0` filter drops isolated points, and dropping an exact
    # zero contributes nothing). Denominator: 2 * ORDERED in-ball pair count.
    # Accumulated in ascending index, which makes the result independent of the
    # block size and the thread count. It is NOT claimed to match MATLAB's
    # `sum`, which is blocked/vectorised rather than a left-to-right scalar
    # accumulation; that difference is one of the ~1e-16 effects the gate
    # measures rather than assumes away.
    num = 0.0
    P = 0
    @inbounds for j in 1:n
        counts[j] == 0 && continue
        P += counts[j]
        maxdy[j] > 0 && (num += maxdy[j])
    end
    return num / (2 * P)
end

# ------------------------------------------------------------------------------
# gamma -- the Lipschitz-constant estimate (opt_smgo_d.m:295-306)
# ------------------------------------------------------------------------------

"""
    nsm_gamma(X, y, eps; gamma_floor=1e-6, block=64) -> Float64

`fgam`, the NSM Lipschitz-constant estimate -- a literal port of
opt_smgo_d.m:295-306:

    gamma = max_i [ if E_i != {} then max_{j in E_i} (|y_j - y_i| - 2*eps) / ||x_j - x_i||
                    else gamma_floor ]
    E_i   = { j : |y_j - y_i| > 2*eps }        # STRICT >, opt_smgo_d.m:298

Note what is and is not restricted:

- eligibility is evaluated over ALL `n` samples with NO ball restriction --
  gamma is a GLOBAL estimate, while `eps` is LOCAL. The asymmetry is deliberate
  in the source; do not "improve" it by adding a locality window.
- `j = i` is self-excluded automatically, since `|y_i - y_i| = 0` and the test
  is strict.
- the numerator is the y-gap DEFLATED by `2*eps` (one `eps` per endpoint) and
  is strictly positive on `E_i`, so the ratio is never `0/0`; only the
  denominator can vanish (coincident samples), giving `+Inf`.
- `gamma_floor` is a PER-ANCHOR fallback (line 303), not a global clamp. If
  every anchor has at least one eligible partner it never enters `fgam_db` and
  gamma may legitimately come out SMALLER than `1e-6` (a tiny deflated gap over
  a long distance). That non-uniformity is real; it is replicated.
- `gamma_floor` is DIMENSIONAL (y per unit length in the working frame), so
  the default `1e-6` is the `:unit01` value; pass
  `nsm_frame_gamma_floor(frame)` for any other frame, as `fit_nsm` does.

Implementation: the ratio is exactly symmetric in `(i,j)`, so `max_i max_{j in E_i}`
equals the max over unordered eligible pairs, and each pair is visited once
(`j > i`). "Anchor `i` has no eligible partner" is equivalent to
`max(|y_max - y_i|, |y_min - y_i|) <= 2*eps` over the NaN-omitting extremes of
`y` (plus: an anchor whose own target is NaN is always empty), decided in O(n)
from `y` alone, so the `gamma_floor` contribution is recovered without a second
pass. Nothing `n x n` is ever allocated; the triangular loop is blocked and
threaded, and `max` is exactly order-independent in floating point, so the
result does not depend on the thread count.

Returns `gamma_floor` for `n < 2` (`fgam_0 = 1e-6`, opt_smgo_d.m:148-149).

`eps === NaN` (radius-ladder exhaustion) also returns `gamma_floor`: every
`|dy| > 2*NaN` test at line 298 is false, so every anchor takes the fallback.

NaN targets: line 298 excludes them from every eligible set (any comparison
against NaN is false), so they contribute no ratio, and an anchor whose own
target is NaN falls back to `gamma_floor` at line 303. Both are reproduced.
"""
function nsm_gamma(
    X::AbstractMatrix{Float64},
    y::AbstractVector{Float64},
    eps::Float64;
    gamma_floor::Float64=1.0e-6,
    block::Int=64,
)
    dim, n = size(X)
    length(y) == n ||
        throw(DimensionMismatch("nsm_gamma: X has $n columns but y has $(length(y)) entries"))
    # opt_smgo_d.m:252 guard -- block not entered, fgam stays fgam_0 (lines 148-149).
    n < 2 && return gamma_floor
    # Faithful NaN propagation of opt_smgo_d.m:298 (see docstring).
    isnan(eps) && return gamma_floor

    two_eps = 2 * eps

    # opt_smgo_d.m:299/303 -- does ANY anchor have an empty eligible set? Since
    # max_j |y_j - y_i| is attained at one of the two extremes of y, this is an
    # O(n) test, exactly equivalent to the per-anchor `sum(elg_idx) > 0` check.
    #
    # The extremes are taken with MATLAB's NaN-OMITTING max/min, not Julia's
    # `extrema`: line 298 compares `abs(y_j - y_i) > 2*feps`, and any comparison
    # against NaN is false in both languages, so a NaN target simply never
    # enters anyone's eligible set. An anchor whose OWN target is NaN therefore
    # has an empty set and takes the line-303 fallback. `extrema` propagates
    # NaN, which made every anchor look non-empty and silently dropped the
    # floor: on X = [0 1 0.5], y = [0, 1e-9, NaN], eps = 0 this returned 1e-9
    # where the dense reference returns 1e-6 (VERIFIED, now fixed). `fit_nsm`
    # rejects non-finite y, but `nsm_gamma` is exported and `nsm_eps` already
    # reproduces the same NaN-omission at line 276, so the two must agree.
    ylo = Inf
    yhi = -Inf
    any_nan_anchor = false
    @inbounds for v in y
        if isnan(v)
            any_nan_anchor = true                # empty eligible set, line 299 -> 303
        else
            v < ylo && (ylo = v)
            v > yhi && (yhi = v)
        end
    end
    any_empty = any_nan_anchor
    if !any_empty
        @inbounds for i in 1:n
            if max(abs(yhi - y[i]), abs(ylo - y[i])) <= two_eps
                any_empty = true
                break
            end
        end
    end

    blocks = _index_blocks(n, block)
    best = fill(-Inf, length(blocks))
    Threads.@threads :dynamic for b in eachindex(blocks)
        m = -Inf
        @inbounds for i in blocks[b]
            yi = y[i]
            for j in (i + 1):n                   # symmetric ratio -> each pair once
                dy = abs(y[j] - yi)
                dy > two_eps || continue         # opt_smgo_d.m:298, STRICT >
                # opt_smgo_d.m:300-301
                ratio = (dy - two_eps) / _pairdist(X, i, j)
                ratio > m && (m = ratio)
            end
        end
        best[b] = m
    end

    gam = maximum(best)                          # opt_smgo_d.m:306
    # The per-anchor fallback acts as a floor exactly when some anchor is empty.
    any_empty && (gam = max(gam, gamma_floor))
    # Unreachable in principle (no eligible pair implies every anchor is empty),
    # kept so a degenerate input can never leak -Inf into the cone width.
    gam == -Inf && (gam = gamma_floor)
    return gam
end

# ------------------------------------------------------------------------------
# Model + fit
# ------------------------------------------------------------------------------

"""
    NSMModel

A fitted NSM model: the training set in the model's own coordinate frame plus
the two scalars the estimator needs.

- `X`: `dim x n_train`, in `frame` coordinates (`[0,1]^dim` by default). Query
  points MUST be mapped to the same frame -- `gamma` has units of fitted-y per
  unit length IN THIS FRAME.
- `y`: the targets fitted on, in whatever units the caller supplied (z-scored
  by the harness convention -- this module does not transform y; see the
  header).
- `eps`: `feps`, see `nsm_eps`. NEVER scaled by `gamma_factor`.
- `gamma`: the Lipschitz constant the cone ACTUALLY USES,
  `gamma_factor * gamma_raw`. `nsm_predict` reads this one.
- `gamma_raw`: what `nsm_gamma` returned, i.e. the literal `fgam` of
  opt_smgo_d.m:295-306. Kept so the estimate and the inflated value can be
  reported paired and so `gamma == gamma_raw` is a checkable statement of
  "this fit is the source".
- `gamma_factor`: the multiplier that relates them (`1.0` = the source).
- `diam`: the diameter of the working box (`sqrt(dim)` for `[0,1]^dim`). Kept
  for provenance/inspection; it is used only during fitting, never at predict
  time.
- `frame`: which box those coordinates live in (`:unit01` or `:pm1`). Recorded
  so that a fitted model states its own contract instead of leaving it to the
  caller's memory: `eps` and `gamma` are meaningless outside it, and a query
  normalised into the wrong frame produces plausible, wrong numbers with no
  error. The 6-argument constructor defaults it to `:unit01`, the literal-port
  frame; `fit_nsm` always sets it explicitly from its settings.

`gamma_raw` and `gamma_factor` are APPENDED fields: the 6- and 7-argument
constructors below predate them and are still in use (the diagnostic scripts
under `gate/` build models directly), so they fill in `gamma_raw = gamma` and
`gamma_factor = 1.0` -- an un-inflated model, which is what those callers
constructed before the factor existed.
"""
struct NSMModel
    X::Matrix{Float64}
    y::Vector{Float64}
    eps::Float64
    gamma::Float64
    diam::Float64
    n_train::Int
    frame::Symbol
    gamma_raw::Float64
    gamma_factor::Float64
end

NSMModel(X, y, eps, gamma, diam, n_train) =
    NSMModel(X, y, eps, gamma, diam, n_train, :unit01, gamma, 1.0)
NSMModel(X, y, eps, gamma, diam, n_train, frame) =
    NSMModel(X, y, eps, gamma, diam, n_train, frame, gamma, 1.0)

"""
    fit_nsm(X, y, s::NSMSettings; diam=nsm_frame_diam(s.frame, size(X,1))) -> NSMModel

Fit the NSM arm: estimate `eps` (opt_smgo_d.m:257-279) then `gamma`
(lines 295-306, which consumes `eps`), multiply the latter by
`s.gamma_factor`, and keep the training set for the predict-time cone
evaluation (lines 482-483). There is no optimisation and no randomness;
`s.seed` is carried for reporting only.

THE FACTOR IS APPLIED HERE, not inside `nsm_gamma` -- see
`_apply_gamma_factor`. `m.gamma_raw` is the estimator's own output and
`m.gamma` is what the cone uses; at `s.gamma_factor == 1.0` they are the same
Float64 and the whole fit is the source's. `eps` is never scaled.

`X` is `dim x n` and must ALREADY be in `s.frame` coordinates (use
`nsm_normalize_pm1` for the default `:pm1`, `nsm_normalize_unit01` for
`:unit01`); `y` is what the model is fitted on (the noisy, z-scored training
targets under the harness convention -- never query truth, never the clean
targets).

Non-finite entries in `X` or `y` are rejected up front. The reference has no
such guard, but its behaviour on non-finite input is silently divergent between
MATLAB and Julia (MATLAB's `max` OMITS NaN, Julia's `maximum` PROPAGATES it)
and a KD-tree cannot be built on non-finite coordinates at all, so an explicit
error is preferable to a number nobody can reproduce. Under this benchmark's
fixtures the guard never fires.

THE FRAME IS CHECKED, not assumed. `diam` sets the entire ball ladder, so
coordinates outside `nsm_frame_box(s.frame)` produce a silently wrong `eps` and
`gamma` -- measured at 6x on `gamma` for raw `[-pi,pi]^3` input, with `rel_l2`
and coverage moving in the third decimal, i.e. invisible in every reported
metric. A coordinate more than `frame_atol` (an absolute slack in frame units,
default 1e-9) outside the box is an error naming the offending dimension.
Pass `check_frame=false` when deliberately fitting in a custom box with an
explicit `diam`; there is then no contract left to check and `m.frame` records
only what the settings claimed. Queries are NOT checked anywhere: a query
outside the training box is legitimate (port decision P1) and the estimator
never uses `diam` at predict time.
"""
function fit_nsm(
    X::AbstractMatrix{Float64},
    y::AbstractVector{Float64},
    s::NSMSettings;
    diam::Float64=nsm_frame_diam(s.frame, size(X, 1)),
    check_frame::Bool=true,
    frame_atol::Float64=1.0e-9,
)
    dim, n = size(X)
    n >= 1 || throw(ArgumentError("fit_nsm: need at least one training sample"))
    length(y) == n ||
        throw(DimensionMismatch("fit_nsm: X has $n columns but y has $(length(y)) entries"))
    all(isfinite, X) || throw(ArgumentError("fit_nsm: X contains non-finite entries"))
    all(isfinite, y) || throw(ArgumentError("fit_nsm: y contains non-finite entries"))
    isfinite(diam) && diam > 0 ||
        throw(ArgumentError("fit_nsm: diam must be finite and positive, got $diam"))
    check_frame && _check_frame(X, s.frame, frame_atol)

    Xm = Matrix{Float64}(X)
    yv = Vector{Float64}(y)

    # One tree, shared by the radius-ladder scan and the neighbour search.
    tree = n >= 2 ? KDTree(Xm) : nothing
    eps = nsm_eps(Xm, yv; diam=diam, radii=s.radii, tree=tree)
    gamma_raw = nsm_gamma(Xm, yv, eps; gamma_floor=s.gamma_floor)
    gamma = _apply_gamma_factor(gamma_raw, s.gamma_factor)
    return NSMModel(Xm, yv, eps, gamma, diam, n, s.frame, gamma_raw, s.gamma_factor)
end

"""
    _apply_gamma_factor(gamma_raw, factor) -> Float64

The safety-factor WRAPPER (see `NSM_DEFAULT_GAMMA_FACTOR`). It is deliberately
the only place the factor is applied: `nsm_gamma` itself is the transliteration
of opt_smgo_d.m:295-306 and must keep returning what MATLAB returns, or the
equivalence gate against `nsm_reference.jl` stops meaning anything.

`factor == 1.0` returns `gamma_raw` by identity rather than by multiplication.
`1.0 * x === x` for every Float64 anyway, including `Inf` and `-0.0`, so this
is not a correctness fix -- it is a statement that the source path contains no
arithmetic that could be got wrong.

`gamma_raw = +Inf` (coincident samples with differing targets) stays `+Inf`,
so the `:NSM_PREDICT_NONFINITE` path is unchanged by the factor.
"""
@inline function _apply_gamma_factor(gamma_raw::Float64, factor::Float64)
    factor == 1.0 && return gamma_raw
    return factor * gamma_raw
end

"""
    _check_frame(X, frame, atol)

Throw unless every training coordinate lies in `nsm_frame_box(frame)`, up to an
absolute slack of `atol` in frame units. See `fit_nsm` on why this is an error
rather than a warning. The message names the dimension and the observed extent
so the caller can see at once whether it handed over physical, `[-1,1]` or
`[0,1]` coordinates.
"""
function _check_frame(X::AbstractMatrix{Float64}, frame::Symbol, atol::Float64)
    lo, hi = nsm_frame_box(frame)
    dim = size(X, 1)
    @inbounds for d in 1:dim
        mn = Inf
        mx = -Inf
        for j in axes(X, 2)
            v = X[d, j]
            v < mn && (mn = v)
            v > mx && (mx = v)
        end
        if mn < lo - atol || mx > hi + atol
            throw(ArgumentError(
                "fit_nsm: training coordinates are not in the declared frame $(frame) " *
                "(box [$lo, $hi]): dimension $d spans [$mn, $mx]. NSM's ball ladder is " *
                "scaled by diam = nsm_frame_diam($(frame), $dim), so fitting out-of-frame " *
                "coordinates silently mis-scales eps and gamma. Normalise the inputs " *
                "(nsm_normalize_unit01 for :unit01), or choose the matching frame with " *
                "nsm_settings(dim; frame=...), or pass check_frame=false with an explicit diam."))
        end
    end
    return nothing
end

# ------------------------------------------------------------------------------
# Predict (opt_smgo_d.m:482-483)
# ------------------------------------------------------------------------------

"""
    nsm_predict(m::NSMModel, x) -> (pred, lower, upper)

Evaluate the set-membership cones at a single query point `x` (length `dim`, in
the model's coordinate frame) and return the central estimate and the two
bounds, in the SAME 3-tuple order as `GPTrainer.gp_predict`:

    upper = min_i ( y_i + eps + gamma*||x - x_i|| )     # opt_smgo_d.m:482
    lower = max_i ( y_i - eps - gamma*||x - x_i|| )     # opt_smgo_d.m:483
    pred  = 0.5 * (lower + upper)

taken over ALL training samples, with no ball restriction, with `eps = m.eps`
and `gamma = m.gamma` -- the INFLATED constant `gamma_factor * m.gamma_raw`.
The additions are
grouped EXACTLY as the source groups them -- `(y_i + eps) + gamma*d` and
`(y_i - eps) - gamma*d`, matching the elementwise
`X_n(FVAL,:) + feps + fgam*dist` of line 482 -- rather than hoisting `eps` out
of the `min`/`max`. Hoisting would be algebraically identical but not
bit-identical (it regroups two roundings), and the whole point of this arm is
that its numbers are the reference's numbers.

The `+ 2*feps` that opt_smgo_d.m:440-441 adds to `lambda` is the ACQUISITION
uncertainty width, not the estimator, and is deliberately not applied here.

The two reductions are MATLAB's `min`/`max`, which OMIT NaN -- and return NaN
when EVERY element is NaN. Both halves matter here:

- `gamma = +Inf` (coincident samples with different targets) makes the term at
  a sample `Inf*0 = NaN` while the others are finite; MATLAB skips it, so a
  comparison-based reduction with `+/-Inf` sentinels is already right;
- `eps = NaN` (radius-ladder exhaustion) makes EVERY term NaN, and MATLAB then
  returns NaN for both bounds. A comparison-based reduction leaves the
  sentinels untouched instead, which is why this loop tracks whether either
  reduction ever saw a non-NaN candidate. Before that fix this function
  returned `(NaN, -Inf, +Inf)` where the dense reference returns
  `(NaN, NaN, NaN)` -- an INFINITE exported interval width instead of a NaN
  one, which propagates differently through any mean-width or coverage
  aggregation downstream (VERIFIED, now fixed). `nsm_status` rejects the query
  either way, so the status column was never affected.

O(n_train * dim) per query with no allocation. The caller times this call to
fill `ArmQueryRow.elapsed_s`.
"""
function nsm_predict(m::NSMModel, x::AbstractVector{Float64})
    length(x) == size(m.X, 1) ||
        throw(DimensionMismatch("nsm_predict: query has $(length(x)) coordinates, model has $(size(m.X,1))"))
    X = m.X
    y = m.y
    gamma = m.gamma
    e = m.eps
    upper = Inf
    lower = -Inf
    seen_u = false                               # did MATLAB's min see a non-NaN?
    seen_l = false                               # ditto max
    @inbounds for i in 1:m.n_train
        cone = gamma * sqrt(_sqdist_point(X, i, x))
        u = (y[i] + e) + cone                    # opt_smgo_d.m:482
        seen_u |= !isnan(u)
        u < upper && (upper = u)
        l = (y[i] - e) - cone                    # opt_smgo_d.m:483
        seen_l |= !isnan(l)
        l > lower && (lower = l)
    end
    seen_u || (upper = NaN)                      # all-NaN: MATLAB min -> NaN
    seen_l || (lower = NaN)                      # all-NaN: MATLAB max -> NaN
    return 0.5 * (lower + upper), lower, upper
end

"""
    nsm_status(pred, lower, upper) -> Symbol

Per-query status in the harness's vocabulary: `:COMPLETED` when all three
quantities are finite, `:NSM_PREDICT_NONFINITE` otherwise. The non-finite case
is reachable two ways, both inherited from the reference and both deliberately
left un-repaired so they stay visible: `eps = NaN` from radius-ladder
exhaustion (see `nsm_eps`), and `gamma = +Inf` from coincident samples with
different targets (opt_smgo_d.m:300 divides by a zero distance, unguarded).

`(:NSM_TRAIN_FAILED` is emitted by the caller, not here, when `fit_nsm`
throws.)
"""
function nsm_status(pred::Float64, lower::Float64, upper::Float64)
    return (isfinite(pred) && isfinite(lower) && isfinite(upper)) ? :COMPLETED :
           :NSM_PREDICT_NONFINITE
end

"""
    nsm_predict_batch(m::NSMModel, X_query; block=256)
        -> (pred, lower, upper, status, elapsed_s)

Vectorised, memory-bounded sweep over a `dim x n_query` block of query points
(in the model's coordinate frame). Returns four length-`n_query` vectors plus
`elapsed_s`, the AMORTISED per-query wall time (total sweep time divided by
`n_query`) -- the batch path is there for speed, so it cannot report a genuine
per-query time; a caller that needs true per-query timing calls `nsm_predict`
in its own timed loop, exactly as `run_gp_arm_cell` does.

Queries are processed in blocks of `block` so the working set stays bounded
regardless of `n_query`; the per-query results are independent, so the values
do not depend on `block` or on the thread count.
"""
function nsm_predict_batch(m::NSMModel, X_query::AbstractMatrix{Float64}; block::Int=256)
    size(X_query, 1) == size(m.X, 1) ||
        throw(DimensionMismatch("nsm_predict_batch: queries have $(size(X_query,1)) coordinates, model has $(size(m.X,1))"))
    nq = size(X_query, 2)
    pred = Vector{Float64}(undef, nq)
    lower = Vector{Float64}(undef, nq)
    upper = Vector{Float64}(undef, nq)
    status = Vector{Symbol}(undef, nq)
    nq == 0 && return (pred, lower, upper, status, 0.0)

    dim = size(m.X, 1)
    blocks = _index_blocks(nq, block)
    t0 = time()
    Threads.@threads :dynamic for blk in blocks
        x = Vector{Float64}(undef, dim)
        @inbounds for q in blk
            for d in 1:dim
                x[d] = X_query[d, q]
            end
            p, l, u = nsm_predict(m, x)
            pred[q] = p
            lower[q] = l
            upper[q] = u
            status[q] = nsm_status(p, l, u)
        end
    end
    elapsed = (time() - t0) / nq
    return (pred, lower, upper, status, elapsed)
end

end # module NSMTrainer
