# =============================================================================
# nsm_equivalence_tests.jl -- Phase-2 gate for the NSM comparator arm.
#
# WHAT THIS GATES
#   1. EQUIVALENCE.  The fast production port (`nsm.jl`, module NSMTrainer,
#      KD-tree candidate generation + blocked/threaded triangular loops) must
#      reproduce the literal dense transliteration of opt_smgo_d.m
#      (`nsm_reference.jl`, module NSMReference) on the same data:
#
#        eps    agree to <= 1e-12 RELATIVE      (EPS_RTOL)
#        gamma  agree to <= 1e-12 RELATIVE      (GAMMA_RTOL)
#        upper / lower / pred at every evaluated point, to <= 1e-12 relative to
#        the local y-scale                      (PRED_RTOL)
#
#      The tolerance is 1e-12 rather than 0 because the two ports are written
#      independently, with different reductions and different distance
#      expressions, and nothing requires them to round identically.  As of
#      2026-09-20 they nevertheless agree BIT-FOR-BIT on all five quantities
#      (worst relative difference exactly 0 on 29 cells / 4554 points).  That
#      is MEASURED and printed, not asserted, and it should not be promoted to
#      an assertion: it is a property of this Julia version on this hardware.
#
#      THE GATE RUNS AT gamma_factor = 1.0 AND frame = :unit01, WHICH IS NOT
#      WHAT THE ARM SHIPS.  The arm ships `:pm1` with `gamma_factor = 2.5`
#      (Diona's L_H_factor, adopted for slate symmetry).  Neither is in
#      opt_smgo_d.m, so neither is a thing the reference could agree with:
#      1.0 is the source's gamma and `:unit01` is the source's box.  Running
#      the gate there is what makes "the port is exact" a statement about the
#      PORT.  The two shipped choices are gated separately and by construction
#      rather than by transliteration -- probe G for the frame (eps identical,
#      RAW gamma ratio exactly 0.5, gamma_floor rescaled) and probe I for the
#      factor (wrapper identity, eps unscaled, band monotone and capped).
#      `fit_nsm` asserts `gamma === gamma_raw` at factor 1.0 inside the main
#      loop, so a factor applied in the wrong place -- inside `nsm_gamma`
#      instead of on top of it -- fails here rather than passing quietly.
#
#      An earlier revision of this header explained a ~5.6e-16 residual as the
#      reference's `LinearAlgebra.norm` "rescaling by the max |coordinate|".
#      That explanation was wrong in both halves.  Julia's `generic_norm2` only
#      engages its rescaling path when `dim*maxabs^2` would overflow (~1e160),
#      so on normalised coordinates it returns exactly the same double as an
#      explicit sqrt-of-sum-of-squares (verified: 0 differences in 200,000
#      random vectors at each of D = 2,3,5,8).  The residual was in fact `@simd`
#      in the production port's own distance kernel, which licenses FMA
#      contraction; removing it took every equivalence figure to 0.  Re-adding
#      it restores the old numbers exactly (gamma 2.204e-16, lower 5.585e-16,
#      same cells, same values) -- and does NOT trip this gate, which is why
#      that mutant is listed below as uncaught-by-design.
#
#      1e-12 relative remains ~4 orders of magnitude looser than the last-ulp
#      noise floor and ~4 orders tighter than the smallest algorithmic
#      divergence measured here (1.6e-10, see MUTATION EVIDENCE), so it
#      discriminates.
#
#   2. THE NSM BAND PROPERTY, asserted directly on the PRODUCTION model (it is
#      a claim about the arm we ship, not about the oracle).  At every training
#      point x_i, with the model's own (eps, gamma):
#
#        (B1) lower(x_i) <= upper(x_i)
#        (B2) upper(x_i) <= y_i + eps        -- exact; the i-th cone term is
#                                               (y_i + eps) + gamma*0
#        (B3) lower(x_i) >= y_i - eps        -- exact, same reason
#        (B4) |pred(x_i) - y_i| <= eps       -- follows from B2+B3
#        (B5) y_i in [lower(x_i) - eps, upper(x_i) + eps]
#
#      B2+B3 say the band at a training point is CONTAINED in [y_i-eps,
#      y_i+eps]: the estimator never places a training point's band further
#      than eps from the observed (noisy) value.  B5 is the containment
#      statement in the direction the phase brief asks for.
#
#      TOLERANCES, and why they are not uniform.  B2 and B3 hold EXACTLY in
#      floating point and are asserted with tolerance 0: the i-th cone term at
#      x_i is (y_i +/- eps) + gamma*sqrt(0), the square root of an exactly-zero
#      squared distance, so y_i+eps is literally one of the values the `min`
#      ranges over and the result cannot exceed it by even one ulp.  (Measured:
#      0 violations in 6032 training points.)  B1, B4 and B5 are ALGEBRAIC
#      CONSEQUENCES of B2/B3 rather than direct consequences of a single cone
#      term -- B1 compares two independent reductions, B4 and B5 add the
#      midpoint's own rounding -- so each can be violated by a fraction of an
#      ulp of the local y-scale.  They are asserted at BAND_RTOL = 1e-14
#      relative to that scale (~45 ulps).  This is NOT a slackened gate: a real
#      violation of any of them is of order eps/y-scale, which on these cells
#      is ~3e-2, i.e. twelve orders of magnitude above the bound.  The measured
#      worst excursions are 1.4e-16, 1.7e-16 and 1.4e-16 respectively, all
#      one-ulp effects on the 0.5*(lower+upper) and lower-vs-upper comparisons.
#
#      NOTE what is deliberately NOT asserted: y_i in [lower(x_i), upper(x_i)]
#      unconditionally.  That is NOT a theorem of this estimator.  gamma is
#      calibrated so that gamma*d_ij >= |y_i-y_j| - 2*eps (line 300 deflates
#      the gap by one eps PER ENDPOINT), which bounds lower(x_i) - y_i by +eps,
#      not by 0.  The gate MEASURES how often strict containment holds and
#      reports it, rather than asserting something the port would be right to
#      violate.  See the STRICT-CONTAINMENT column of the output.
#
# COVERAGE
#   N in {50, 97, 200, 500} x D in {2, 3, 5} x {clean, noisy}  = 24 cells,
#   plus five structured cells:
#     dup-same-y      exact coincident inputs, equal targets   (gamma finite)
#     dup-diff-y      exact coincident inputs, differing y     (gamma = +Inf)
#     near-dup        inputs 1e-9 apart                        (gamma huge)
#     wide-range      y spanning ~4.1e6 (rosenbrock-scale)
#     wide-range-noisy same, with 5% relative noise
#   Each cell is evaluated at 60 interior queries, 8 queries OUTSIDE the
#   training box (the ports do not clip -- port decision P1), and up to 100
#   training points (distance exactly 0, the cone-degenerate case).
#
#   Probe I adds one more cell of its own (D=3, n=300, 400 queries, 5% noise,
#   fitted three times at gamma_factor 1.0 / 2.5 / 10.0).
#
#   Randomised cells CANNOT reach the two boundary predicates the source makes
#   load-bearing -- the INCLUSIVE ball test at opt_smgo_d.m:263 and the STRICT
#   eligibility test at line 298 -- because random data never lands exactly on
#   a boundary.  (Verified by mutation: flipping `<=` to `<` and `>` to `>=` in
#   the production port leaves all 29 randomised cells bit-identical.)  Nine
#   hand-constructed PROBES close that gap; see `run_probes`.  Probes Cii, E, F,
#   G and H were added on 2026-09-20 with the repairs they check: they cover the
#   bounds at ladder exhaustion, NaN targets in `nsm_gamma`, the colon
#   arithmetic of the radius ladder, the coordinate-frame guard together with
#   the frame-rescaled `gamma_floor`, and the ladder break test against an
#   O(n^2) brute force.  Probe I was added on the same date with the
#   `gamma_factor` wrapper.  None of those five regions was reachable by the
#   randomised cells, which is why four real divergences survived the first
#   version of this gate.
#
# HOW THE TWO PORTS ARE PUT ON THE SAME FOOTING
#   `NSMReference.fit` normalises raw inputs with the TRAINING POOL's
#   per-dimension min/max; `NSMTrainer.fit_nsm` takes coordinates that are
#   ALREADY in frame.  The gate therefore normalises once, with
#   `NSMReference.normalise_inputs` and the reference's own bounds, and feeds
#   the result to `fit_nsm` with the `:unit01` settings (diam = sqrt(D),
#   opt_smgo_d.m:145).  Queries are normalised with the SAME call, so the two
#   ports see bit-identical coordinates and the comparison isolates the
#   estimators and the cone evaluation.  Normalisation itself is not under
#   test here.
#
#   Consequence: the gate compares the ports in the frame they AGREE on.  The
#   production arm's separate decision to normalise benchmark inputs with the
#   benchmark function's DECLARED bf.lb/bf.ub rather than the pool's empirical
#   min/max is a property of the (not yet written) arm wiring, not of these
#   two modules, and is out of scope here.
#
# MUTATION EVIDENCE (how this gate was validated; re-runnable by hand)
#   Eleven single-edit defects were injected into a COPY of nsm.jl and the gate
#   re-run against the unmodified reference.  Nine were caught:
#     ball test `<=` -> `<`                  -> probe A          (eps -> NaN)
#     denominator 2*P -> P                   -> eps, 29/29 cells, worst 0.5
#     drop the 2*eps deflation in gamma      -> gamma, worst 0.68
#     hoist eps out of the predict min/max   -> pred, worst 1.6e-10
#     ladder starting at 0.2 / ball_rad*2    -> eps, worst 0.68
#     predict drops eps from the lower cone  -> lower, worst 0.074, and B1/B4
#     gamma floor made a GLOBAL clamp        -> probe Bii
#   Two were NOT caught, and both are provably EQUIVALENT mutants rather than
#   gate blind spots:
#     gamma's `for j in (i+1):n` -> `i:n` adds the self pair, whose |dy| is 0
#       and so is never strictly > 2*eps for eps >= 0.
#     eligibility `>` -> `>=` ALONE admits pairs whose deflated numerator is
#       exactly 0, and a ratio of 0 can never raise a max that is already >= 0;
#       `any_empty`'s `<=` -> `<` ALONE is absorbed by the `gam == -Inf` guard.
#       Changed TOGETHER -- i.e. a genuine switch to inclusive eligibility --
#       they ARE caught, by probe Bi (gamma 0 vs the 1e-6 floor).
#   The 1.6e-10 figure is the useful one for tolerance-setting: a pure
#   REGROUPING of two roundings in the cone sum already shows up 160x above
#   PRED_RTOL, because the midpoint cancels.  1e-12 is not a loose bound.
#
#   SECOND ROUND (2026-09-20), validating the five probes added with the
#   repairs.  Each repair was reverted one at a time in a COPY of nsm.jl and the
#   gate re-run against the unmodified reference.  Five of seven caught:
#     revert the NaN-omitting predict reduction -> probe Cii  ((-Inf,Inf) vs (NaN,NaN))
#     revert nsm_gamma's NaN handling (extrema) -> probe E    (1e-9 vs the 1e-6 floor)
#     revert the ladder to collect(0.1:0.1:1.0) -> probe F    (rungs 3 and 7)
#     revert the fit_nsm frame guard            -> probe G    (threw=false)
#     revert the frame-rescaled gamma_floor     -> probe G    (floor 1e-6 vs 5e-7)
#   Two were NOT caught, and the reasons differ:
#     re-adding `@simd` to the distance kernel moves the ports apart by at most
#       5.6e-16 relative, four orders inside PRED_RTOL.  This gate cannot see it
#       and is not meant to: the case against `@simd` is that it licenses FMA
#       contraction, so the kernel stops being reproducible across hosts.  It is
#       argued in nsm.jl, not gated here.
#     removing the ladder's ulp rescue scan changes nothing on any dataset the
#       gate can construct, because it only fires when the KD-tree mis-ranks a
#       near-tied neighbour -- which cannot be forced deliberately.  Probe H
#       therefore checks the machinery (`_ladder_ball_rad` and
#       `_any_pair_within` against an O(n^2) brute force over 160 datasets,
#       including planted exact-rung ties and ladder exhaustions) rather than
#       the trigger.  Stated plainly: the rescue is defensive code whose trigger
#       condition this gate does not reach.
#
# RUN
#   julia --project=/home/ls122/code/Diona/test/ml \
#         /home/ls122/code/Diona/test/ml/baselines/nsm_equivalence_tests.jl
#
#   Exit status 0 = PASS, 1 = FAIL.  On failure the worst discrepancy and the
#   cell that produced it are printed for every check family.
# =============================================================================

using Random
using Printf

include(joinpath(@__DIR__, "nsm_reference.jl"))
include(joinpath(@__DIR__, "nsm.jl"))

const EPS_RTOL = 1.0e-12
const GAMMA_RTOL = 1.0e-12
const PRED_RTOL = 1.0e-12
# Band checks.  B2/B3 are exact in floating point (see the header) and are
# asserted at 0.  B1/B4/B5 are derived from them and carry up to a fraction of
# an ulp of the local y-scale; 1e-14 is ~45 ulps, and twelve orders of
# magnitude below the ~eps/y-scale size of a genuine violation.
const BAND_EXACT = 0.0
const BAND_RTOL = 1.0e-14

# -----------------------------------------------------------------------------
# Discrepancy bookkeeping
# -----------------------------------------------------------------------------

"""
Relative discrepancy between two scalars, with exact special-value handling.

Returns 0.0 when the two agree exactly (including NaN/NaN and +Inf/+Inf, which
are meaningful agreements here: NaN is radius-ladder exhaustion and +Inf is a
coincident-sample gamma, and the ports must reproduce BOTH), and +Inf when one
is special and the other is not.  Otherwise |a-b| / scale.
"""
function reldiff(a::Float64, b::Float64, scale::Float64)
    (isnan(a) || isnan(b)) && return (isnan(a) && isnan(b)) ? 0.0 : Inf
    (isinf(a) || isinf(b)) && return (a == b) ? 0.0 : Inf
    a == b && return 0.0
    s = max(scale, abs(a), abs(b))
    s == 0 && return 0.0
    return abs(a - b) / s
end

mutable struct Worst
    name::String
    tol::Float64
    value::Float64        # worst RELATIVE discrepancy seen
    absval::Float64       # the absolute discrepancy that produced it
    where_::String
    nchecks::Int
    nviol::Int
end
Worst(name, tol) = Worst(name, tol, 0.0, 0.0, "(none)", 0, 0)

function record!(w::Worst, v::Float64, where_::String; absval::Float64=NaN)
    w.nchecks += 1
    if v > w.value || (isnan(v) && !isnan(w.value))
        w.value = v
        w.absval = absval
        w.where_ = where_
    end
    (isnan(v) || v > w.tol) && (w.nviol += 1)
    return nothing
end

passed(w::Worst) = w.nviol == 0

# The worst case is printed whether or not the check passes: on a PASS it is
# the evidence for HOW MUCH headroom the tolerance has, which is the number
# that tells a reader whether the gate discriminates.
function report(w::Worst)
    tag = passed(w) ? "PASS" : "FAIL"
    @printf("  [%s] %-34s worst = %-11.4g (tol %.0e)  %d/%d violations\n",
            tag, w.name, w.value, w.tol, w.nviol, w.nchecks)
    if w.value > 0 || !passed(w)
        absnote = isnan(w.absval) ? "" : @sprintf(" [abs %.4g]", w.absval)
        @printf("         worst at%s: %s\n", absnote, w.where_)
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Datasets.  Each returns (Xraw :: D x n, y :: n, label).
# Columns are points, matching X_n's layout in both ports.
# -----------------------------------------------------------------------------

"A smooth, genuinely nonlinear test field with cross terms in every dimension."
function field(x::AbstractVector{Float64})
    D = length(x)
    v = 0.0
    for d in 1:D
        v += sin(2.3 * x[d] + 0.4 * d) + 0.5 * x[d]^2
    end
    for d in 1:(D - 1)
        v += 0.35 * x[d] * x[d + 1]
    end
    return v
end

"Rosenbrock, used only to get a genuinely wide output range (~4e6 here)."
function rosenbrock(x::AbstractVector{Float64})
    s = 0.0
    for d in 1:(length(x) - 1)
        s += 100.0 * (x[d + 1] - x[d]^2)^2 + (1.0 - x[d])^2
    end
    return s
end

struct Case
    label::String
    Xraw::Matrix{Float64}
    y::Vector{Float64}
    note::String
end

"Sample `n` points in the box [lo,hi]^D and evaluate `f`, optionally noised."
function make_case(label::String, D::Int, n::Int, f::Function, lo::Float64, hi::Float64,
                   noise_frac::Float64, seed::Int; note::String="")
    rng = MersenneTwister(seed)
    X = lo .+ (hi - lo) .* rand(rng, D, n)
    y = [f(X[:, j]) for j in 1:n]
    if noise_frac > 0
        span = maximum(y) - minimum(y)
        y = y .+ noise_frac * span .* randn(rng, n)
    end
    return Case(label, X, y, note)
end

function build_cases()
    cases = Case[]
    seed = 1000
    for D in (2, 3, 5), n in (50, 97, 200, 500)
        for (tag, nf) in (("clean", 0.0), ("noisy", 0.05))
            seed += 1
            push!(cases, make_case(@sprintf("grid D=%d n=%-3d %s", D, n, tag),
                                   D, n, field, -2.0, 2.0, nf, seed))
        end
    end

    # --- exact duplicates, EQUAL targets: gamma stays finite, the band
    #     property is still meaningful, and the ball test sees a 0 distance.
    let D = 3, n = 200
        rng = MersenneTwister(7001)
        X = -2.0 .+ 4.0 .* rand(rng, D, n)
        y = [field(X[:, j]) for j in 1:n]
        for k in 1:10                       # 10 exact coincident pairs
            X[:, 2k] = X[:, 2k - 1]
            y[2k] = y[2k - 1]
        end
        push!(cases, Case("dup-same-y   D=3 n=200", X, y,
                          "10 exact coincident input pairs with equal targets"))
    end

    # --- exact duplicates, DIFFERENT targets: opt_smgo_d.m:300 divides by a
    #     zero distance, so gamma = +Inf (port decision P4).  Both ports must
    #     reproduce that, and MATLAB's NaN-OMITTING min/max must be reproduced
    #     at a query sitting exactly on a duplicated sample (Inf*0 = NaN there).
    let D = 3, n = 120
        rng = MersenneTwister(7002)
        X = -2.0 .+ 4.0 .* rand(rng, D, n)
        y = [field(X[:, j]) for j in 1:n]
        for k in 1:5
            X[:, 2k] = X[:, 2k - 1]
            y[2k] = y[2k - 1] + 3.0        # gap large enough to be eligible
        end
        push!(cases, Case("dup-diff-y   D=3 n=120", X, y,
                          "5 coincident input pairs with differing targets -> gamma = +Inf"))
    end

    # --- near-duplicates: distinct but 1e-9 apart, so gamma is huge but finite.
    let D = 2, n = 150
        rng = MersenneTwister(7003)
        X = -2.0 .+ 4.0 .* rand(rng, D, n)
        y = [field(X[:, j]) for j in 1:n]
        for k in 1:8
            X[:, 2k] = X[:, 2k - 1] .+ 1.0e-9
            y[2k] = field(X[:, 2k])
        end
        y .+= 0.02 * (maximum(y) - minimum(y)) .* randn(rng, n)
        push!(cases, Case("near-dup     D=2 n=150", X, y,
                          "8 pairs 1e-9 apart, 2% noise"))
    end

    # --- wide output range (~4.1e6): checks that nothing in either port
    #     assumes an O(1) y-scale, and that the 1e-12 relative tolerance is a
    #     genuine relative tolerance.
    push!(cases, make_case("wide-range   D=5 n=300 clean", 5, 300, rosenbrock,
                           -5.0, 5.0, 0.0, 7004;
                           note="rosenbrock on [-5,5]^5"))
    push!(cases, make_case("wide-range   D=5 n=300 noisy", 5, 300, rosenbrock,
                           -5.0, 5.0, 0.05, 7005;
                           note="rosenbrock on [-5,5]^5, 5% noise"))
    return cases
end

# -----------------------------------------------------------------------------
# Boundary probes.
#
# These are hand-constructed so that a floating-point comparison lands EXACTLY
# on a boundary, which randomised data never does.  Each probe checks the two
# ports against each other AND against the value the MATLAB semantics dictate,
# so it pins the semantics even if both ports were changed together.
# -----------------------------------------------------------------------------

struct Probe
    name::String
    ok::Bool
    detail::String
end

"Euclidean distance between two columns, written exactly as both ports write it."
function probe_dist(X::Matrix{Float64}, i::Int, j::Int)
    s = 0.0
    for d in axes(X, 1)
        delta = X[d, i] - X[d, j]
        s += delta * delta
    end
    return sqrt(s)
end

"Exact agreement, treating NaN as equal to NaN (both ports must reproduce it)."
exacteq(a::Float64, b::Float64) = (a === b) || (isnan(a) && isnan(b))

function run_probes()
    P = Probe[]

    # --- PROBE A: INCLUSIVE ball boundary, opt_smgo_d.m:263 ------------------
    # Twelve points in four well-separated clusters of three.  Within a cluster
    # A-B is EXACTLY ball_rad at the first ladder rung (rad = 0.1, ball_rad =
    # 0.1*sqrt(2)/2), while A-C is 0.10 and B-C is 0.1225 -- both strictly
    # between rung 1 and rung 2.  So under the source's INCLUSIVE `<=` the
    # ladder breaks at rung 1 with only the A-B pairs in the ball; under `<` it
    # falls through to rung 2 and admits all three pairs, giving a different
    # eps.  The two are distinguishable ONLY at this exact tie.
    #
    # Every cluster uses the SAME x-arithmetic (all anchors share x = 0.05) and
    # is separated along y, because `anchor + ball_rad` is only an exact tie for
    # some anchors: at x = 0.45 the sum rounds and the distance comes out one
    # ulp ABOVE ball_rad, which would silently un-tie that cluster.  The probe
    # therefore VERIFIES the tie count rather than assuming it: it requires all
    # 8 ordered A-B pairs to be at distance exactly ball_rad and NO other pair
    # to be within it, so a future change in floating-point details reports
    # `ties=...` instead of quietly testing nothing.
    let
        rad1 = collect(0.1:0.1:1.0)[1]
        diam = sqrt(2)
        ball1 = rad1 * diam / 2
        anchors = [(0.05, 0.05), (0.05, 0.35), (0.05, 0.65), (0.05, 0.95)]
        X = Matrix{Float64}(undef, 2, 12)
        y = Vector{Float64}(undef, 12)
        k = 0
        for (ci, (a, b)) in enumerate(anchors)
            X[:, k + 1] = [a, b]                 # A
            X[:, k + 2] = [a + ball1, b]         # B -- exactly ball_rad from A
            X[:, k + 3] = [a, b + 0.10]          # C -- 0.10 from A
            y[k + 1] = 1.0 * ci
            y[k + 2] = 1.0 * ci + 0.7 * ci
            y[k + 3] = 1.0 * ci - 0.3 * ci
            k += 3
        end
        ties = 0
        inball = 0
        for i in 1:12, j in 1:12
            i == j && continue
            dij = probe_dist(X, i, j)
            dij === ball1 && (ties += 1)
            dij <= ball1 && (inball += 1)
        end
        tie_exact = (rad1 === 0.1) && ties == 8 && inball == 8
        e_prod = NSMTrainer.nsm_eps(X, y)
        e_ref = NSMReference.estimate_eps(X, y)
        # Dictated by opt_smgo_d.m:279: 8 ORDERED in-ball pairs in the
        # denominator, and 8 active points in the numerator each contributing
        # |y_B - y_A| = 0.7*ci.
        want = 2 * sum(0.7 * ci for ci in 1:4) / (2 * 8)
        push!(P, Probe("A  inclusive ball boundary (line 263)",
                       tie_exact && exacteq(e_prod, e_ref) && abs(e_prod - want) <= 1e-15 * want,
                       @sprintf("ties=%d inball=%d prod=%.17g ref=%.17g expected=%.17g",
                                ties, inball, e_prod, e_ref, want)))
    end

    # --- PROBE B: STRICT eligibility boundary, opt_smgo_d.m:298 --------------
    # B-i: every |y_j - y_i| is EXACTLY 2*eps, so the strict test excludes every
    #      pair, every anchor falls back, and gamma must be exactly the floor.
    #      Under `>=` every pair would be eligible with a numerator of exactly
    #      0.  B-ii: one ulp above 2*eps makes them all eligible, so NO anchor
    #      takes the fallback and gamma comes out ~5e-16 -- BELOW the 1e-6
    #      floor.  That is the source's per-anchor (not global) fallback
    #      semantics, the thing most likely to be "fixed" into a global clamp.
    let
        X = [0.10 0.35 0.60 0.85 0.20 0.70;
             0.15 0.40 0.65 0.90 0.80 0.25]
        eps_b = 0.25
        y_i = [0.0, 0.5, 0.0, 0.5, 0.0, 0.5]            # |dy| in {0, 0.5}; 2*eps = 0.5
        boundary_exact = (2 * eps_b === 0.5) && (abs(y_i[2] - y_i[1]) === 2 * eps_b)
        g_prod = NSMTrainer.nsm_gamma(X, y_i, eps_b)
        g_ref = NSMReference.estimate_gamma(X, y_i, eps_b)
        push!(P, Probe("Bi strict eligibility, |dy| == 2*eps",
                       boundary_exact && exacteq(g_prod, g_ref) && g_prod === 1.0e-6,
                       @sprintf("boundary_exact=%s prod=%.17g ref=%.17g expected=1e-6",
                                boundary_exact, g_prod, g_ref)))

        hi = nextfloat(0.5)
        y_ii = [0.0, hi, 0.0, hi, 0.0, hi]              # one ulp above 2*eps
        g_prod2 = NSMTrainer.nsm_gamma(X, y_ii, eps_b)
        g_ref2 = NSMReference.estimate_gamma(X, y_ii, eps_b)
        push!(P, Probe("Bii per-anchor fallback is NOT a clamp",
                       exacteq(g_prod2, g_ref2) && 0.0 < g_prod2 < 1.0e-6,
                       @sprintf("prod=%.17g ref=%.17g (must be >0 and BELOW the 1e-6 floor)",
                                g_prod2, g_ref2)))
    end

    # --- PROBE C: radius-ladder exhaustion (port decision P3) ----------------
    # The four corners of [0,1]^2 are 1.0 apart; the largest ball in the ladder
    # has radius sqrt(2)/2 = 0.707.  MATLAB falls out of the loop and divides
    # 0 by 0.  Both ports must return NaN, silently, and gamma must collapse to
    # the floor because every line-298 comparison against NaN is false.
    let
        X = [0.0 1.0 0.0 1.0; 0.0 0.0 1.0 1.0]
        y = [1.0, 2.0, 3.0, 4.0]
        e_prod = NSMTrainer.nsm_eps(X, y)
        e_ref = NSMReference.estimate_eps(X, y)
        g_prod = NSMTrainer.nsm_gamma(X, y, e_prod)
        g_ref = NSMReference.estimate_gamma(X, y, e_ref)
        p, l, u = NSMTrainer.nsm_predict(
            NSMTrainer.NSMModel(X, y, e_prod, g_prod, sqrt(2), 4), [0.5, 0.5])
        push!(P, Probe("C  ladder exhaustion -> NaN, not repaired",
                       isnan(e_prod) && isnan(e_ref) && exacteq(g_prod, g_ref) &&
                       g_prod === 1.0e-6 &&
                       NSMTrainer.nsm_status(p, l, u) === :NSM_PREDICT_NONFINITE,
                       @sprintf("eps prod=%s ref=%s; gamma prod=%.17g ref=%.17g; status=%s",
                                e_prod, e_ref, g_prod, g_ref,
                                NSMTrainer.nsm_status(p, l, u))))

        # C-ii: the BOUNDS at exhaustion, which probe C never compared.  With
        # eps = NaN every candidate at lines 482-483 is NaN; MATLAB's min/max
        # omit NaN but return NaN when ALL elements are NaN, so both bounds are
        # NaN and the exported width is NaN.  A comparison-based reduction with
        # +/-Inf sentinels leaves them untouched and reports an INFINITE width
        # instead -- which is what the production port did until 2026-09-20.
        # The status was :NSM_PREDICT_NONFINITE either way, so probe C passed
        # throughout; only a direct comparison of l/u against the reference
        # sees it.
        rl, ru = NSMReference.bounds(NSMReference.fit(X, y), [0.5, 0.5])
        push!(P, Probe("Cii exhaustion bounds are NaN, not +/-Inf",
                       exacteq(l, rl) && exacteq(u, ru) && isnan(l) && isnan(u) &&
                       isnan(u - l),
                       @sprintf("prod (l,u)=(%s,%s) ref (l,u)=(%s,%s) width=%s", l, u, rl, ru, u - l)))
    end

    # --- PROBE E: NaN targets, opt_smgo_d.m:298-304 --------------------------
    # An anchor whose own target is NaN has an empty eligible set (every
    # comparison against NaN is false), so line 303 gives it the 1e-6 fallback
    # and line 306's max picks that up.  Julia's `extrema` propagates NaN, which
    # made the O(n) empty-set shortcut miss the fallback entirely and return the
    # raw pair maximum.  fit_nsm rejects non-finite y, but nsm_gamma is exported
    # and nsm_eps already reproduces the same NaN-omission at line 276.
    let
        X = [0.0 1.0 0.5]
        y = [0.0, 1.0e-9, NaN]
        g_prod = NSMTrainer.nsm_gamma(X, y, 0.0)
        g_ref = NSMReference.estimate_gamma(X, y, 0.0)
        y_all = [NaN, NaN, NaN]
        g_prod_all = NSMTrainer.nsm_gamma(X, y_all, 0.0)
        g_ref_all = NSMReference.estimate_gamma(X, y_all, 0.0)
        push!(P, Probe("E  NaN target keeps the line-303 fallback",
                       exacteq(g_prod, g_ref) && g_prod === 1.0e-6 &&
                       exacteq(g_prod_all, g_ref_all) && g_prod_all === 1.0e-6,
                       @sprintf("one NaN: prod=%.17g ref=%.17g; all NaN: prod=%.17g ref=%.17g",
                                g_prod, g_ref, g_prod_all, g_ref_all)))
    end

    # --- PROBE F: the radius ladder is MATLAB's colon, not Julia's range -----
    # MATLAB's colon materialises base + k*step in double precision; Julia's
    # range is TwicePrecision and returns the correctly-rounded double of the
    # exact decimal.  They differ by one ulp at rungs 3 and 7, which moves
    # ball_rad and can flip an exact tie in the INCLUSIVE test at line 263.
    let
        ladder = NSMTrainer.nsm_radius_ladder()
        colon = [0.1 + (k - 1) * 0.1 for k in 1:10]
        jrange = collect(0.1:0.1:1.0)
        differs = count(k -> ladder[k] !== jrange[k], 1:10)
        push!(P, Probe("F  ladder uses MATLAB colon arithmetic",
                       length(ladder) == 10 && all(k -> ladder[k] === colon[k], 1:10) &&
                       differs == 2 && ladder[3] === 0.30000000000000004 &&
                       ladder[7] === 0.70000000000000007,
                       @sprintf("rung3=%.17g rung7=%.17g; differs from collect(0.1:0.1:1.0) at %d/10 rungs",
                                ladder[3], ladder[7], differs)))
    end

    # --- PROBE G: the coordinate frame is enforced, and :pm1 is the exact -----
    #     image of :unit01 (diam AND gamma_floor both rescaled).
    # diam sets the whole ball ladder, so out-of-frame coordinates silently
    # mis-scale eps and gamma.  G-i: fitting [-1,1] data with :unit01 settings
    # must throw.  G-ii: the same data in :pm1 settings must give the SAME eps
    # and exactly HALF the gamma -- including in the floor-bound regime, which
    # an un-rescaled 1e-6 breaks (it pins both frames at 1e-6, ratio 1.0).
    let
        rng = MersenneTwister(2718)
        D, n = 3, 60
        X01 = rand(rng, D, n)
        y = [sin(3 * X01[1, j]) + X01[2, j]^2 - X01[3, j] for j in 1:n]
        Xpm = 2 .* X01 .- 1
        threw = false
        try
            # [-1,1] data declared as :unit01.  NOTE the explicit frame: since
            # the default became :pm1 this call would otherwise be CORRECT and
            # the probe would silently test nothing.
            NSMTrainer.fit_nsm(Xpm, y, NSMTrainer.nsm_settings(D; frame=:unit01))
        catch err
            threw = err isa ArgumentError
        end
        m01 = NSMTrainer.fit_nsm(X01, y, NSMTrainer.nsm_settings(D; frame=:unit01, gamma_factor=1.0))
        mpm = NSMTrainer.fit_nsm(Xpm, y, NSMTrainer.nsm_settings(D; frame=:pm1, gamma_factor=1.0))
        # floor-bound regime: two points, no eligible pair, every anchor falls back
        Xf = [0.20 0.30; 0.40 0.50]
        yf = [1.0, 2.0]
        f01 = NSMTrainer.fit_nsm(Xf, yf, NSMTrainer.nsm_settings(2; frame=:unit01, gamma_factor=1.0))
        fpm = NSMTrainer.fit_nsm(2 .* Xf .- 1, yf, NSMTrainer.nsm_settings(2; frame=:pm1, gamma_factor=1.0))
        # G-iii: the SHIPPED default really is :pm1 (frame) with the 2.5 factor,
        # and the frame relations are a property of the RAW estimate, so they
        # must survive the factor unchanged.
        d01 = NSMTrainer.nsm_settings(D; frame=:unit01)
        dpm = NSMTrainer.nsm_settings(D)
        s01 = NSMTrainer.fit_nsm(X01, y, d01)
        spm = NSMTrainer.fit_nsm(Xpm, y, dpm)
        defaults_ok = dpm.frame === :pm1 && d01.frame === :unit01 &&
                      dpm.gamma_factor === NSMTrainer.NSM_DEFAULT_GAMMA_FACTOR() &&
                      dpm.gamma_floor === 5.0e-7 && d01.gamma_floor === 1.0e-6 &&
                      exacteq(s01.eps, m01.eps) && exacteq(spm.eps, mpm.eps) &&
                      exacteq(s01.gamma_raw, m01.gamma) && exacteq(spm.gamma_raw, mpm.gamma) &&
                      spm.gamma_raw === 0.5 * s01.gamma_raw
        push!(P, Probe("G  frame enforced; :pm1 is the exact image",
                       threw && m01.frame === :unit01 && mpm.frame === :pm1 &&
                       exacteq(m01.eps, mpm.eps) && mpm.gamma === 0.5 * m01.gamma &&
                       f01.gamma === 1.0e-6 && fpm.gamma === 5.0e-7 && defaults_ok,
                       @sprintf("pm1-into-unit01 threw=%s; eps %.17g/%.17g; gamma ratio %.17g; floor %.3g/%.3g; defaults(:pm1,%.3g) raw-ratio-under-default %.17g -> %s",
                                threw, m01.eps, mpm.eps, mpm.gamma / m01.gamma,
                                f01.gamma, fpm.gamma, dpm.gamma_factor,
                                spm.gamma_raw / s01.gamma_raw, defaults_ok ? "ok" : "MISMATCH")))
    end

    # --- PROBE H: the ladder break test against a dense brute force ----------
    # opt_smgo_d.m:266-268 breaks at the first radius whose ball holds an
    # off-diagonal pair.  The port resolves that from a knn-selected neighbour
    # plus an ulp-window rescue scan (`_any_pair_within`), because knn ranks by
    # the TREE's metric and can mis-rank a near-tie.  The mis-ranking itself is
    # not constructible on demand, so what is checked here is the machinery:
    # `_ladder_ball_rad` must equal an O(n^2) brute force over the same literal
    # predicate on every dataset, and `_any_pair_within` must equal the dense
    # `sum(ball_idx,'all') > 0` test at EVERY rung -- including datasets whose
    # minimum pair sits EXACTLY on a rung (the inclusive `<=` boundary) and
    # datasets that exhaust the ladder entirely.  The probe counts the exact
    # ties it actually achieved rather than assuming them (see probe A).
    let
        ladder = NSMTrainer.nsm_radius_ladder()
        rng = MersenneTwister(31415)
        bad_break = 0
        bad_any = 0
        ties = 0
        exhausted = 0
        cases = 0
        for D in (1, 2, 3, 5), trial in 1:40
            diam = sqrt(D)
            n = rand(rng, 2:25)
            X = Matrix{Float64}(undef, D, n)
            mode = trial % 4
            for j in 1:n, d in 1:D
                # mode 0: spread over the box; 1: tight cluster; 2: two clusters;
                # 3: corners only, which exhausts the ladder at low D.
                X[d, j] = mode == 0 ? rand(rng) :
                          mode == 1 ? 0.5 + 0.01 * rand(rng) :
                          mode == 2 ? (j <= n ÷ 2 ? 0.1 * rand(rng) : 0.9 + 0.1 * rand(rng)) :
                          Float64(rand(rng, 0:1))
            end
            if mode == 1 && D >= 2          # plant an EXACT tie at a rung
                r = ladder[rand(rng, 1:10)] * diam / 2
                X[:, 1] .= 0.05
                X[:, 2] .= 0.05
                X[1, 2] = 0.05 + r
            end
            tree = NSMTrainer.KDTree(X)
            dense_break = NaN
            for r0 in ladder
                r = r0 * diam / 2
                hit = false
                for i in 1:n, j in 1:n
                    i == j && continue
                    if NSMTrainer._pairdist(X, i, j) <= r
                        hit = true
                        break
                    end
                end
                if NSMTrainer._any_pair_within(tree, X, r) != hit
                    bad_any += 1
                end
                if hit && isnan(dense_break)
                    dense_break = r
                end
            end
            got = NSMTrainer._ladder_ball_rad(tree, X, ladder, diam)
            exacteq(got, dense_break) || (bad_break += 1)
            isnan(dense_break) && (exhausted += 1)
            for i in 1:n, j in (i + 1):n
                d = NSMTrainer._pairdist(X, i, j)
                any(r0 -> d === r0 * diam / 2, ladder) && (ties += 1)
            end
            cases += 1
        end
        push!(P, Probe("H  ladder break == dense brute force",
                       bad_break == 0 && bad_any == 0 && ties >= 10 && exhausted >= 1,
                       @sprintf("%d datasets: %d break mismatches, %d rung-membership mismatches, %d exact rung ties, %d ladder exhaustions",
                                cases, bad_break, bad_any, ties, exhausted)))
    end

    # --- PROBE D: n == 1 (opt_smgo_d.m:252, the `iter > 1` guard) ------------
    # feps keeps feps_0 = 0 and fgam keeps fgam_0 = 1e-6 (lines 148-154), so the
    # cone terms cancel and the prediction at the sample is exactly y_1.
    let
        Xraw = reshape([0.3, 0.7, 0.1], 3, 1)
        y = [7.0]
        refm = NSMReference.fit(Xraw, y)
        Xn = NSMReference.normalise_inputs(Xraw, NSMReference.normalisation_bounds(Xraw)...)
        prodm = NSMTrainer.fit_nsm(Xn, y, NSMTrainer.nsm_settings(3; frame=:unit01, gamma_factor=1.0))
        pp, _, _ = NSMTrainer.nsm_predict(prodm, Xn[:, 1])
        rp = NSMReference.predict(refm, Xraw[:, 1])
        push!(P, Probe("D  n == 1 (iter > 1 guard, line 252)",
                       exacteq(prodm.eps, refm.eps) && prodm.eps === 0.0 &&
                       exacteq(prodm.gamma, refm.gamma) && prodm.gamma === 1.0e-6 &&
                       pp === 7.0 && rp === 7.0,
                       @sprintf("eps=%.17g gamma=%.17g pred prod=%.17g ref=%.17g",
                                prodm.eps, prodm.gamma, pp, rp)))
    end

    # --- PROBE I: gamma_factor is a WRAPPER, and what it can/cannot move -----
    # The arm ships with gamma_factor = 2.5 (Diona's L_H_factor, adopted for
    # slate symmetry).  This probe pins the three things that make it a wrapper
    # rather than an edit to the estimator, and MEASURES the claim that it
    # cannot rescue training-point containment.
    #
    #   I-1  eps is bit-identical across factors            (eps is NOT scaled)
    #   I-2  gamma_raw is bit-identical across factors, and
    #        gamma === factor * gamma_raw exactly            (the wrapper is all
    #                                                         the factor does)
    #   I-3  B2/B3 hold AT EVERY FACTOR, with tolerance 0: the i-th cone term at
    #        x_i is (y_i +/- eps) + gamma*sqrt(0), so the band at a training
    #        point is capped at [y_i - eps, y_i + eps] however large gamma gets
    #   I-4  the band is monotone in the factor at every training point and
    #        every query: upper non-decreasing, lower non-increasing
    #
    # I-3 + I-4 together are the real statement, and it is NOT the one the phase
    # brief guessed.  The brief predicted the factor "CANNOT change
    # training-point containment at all" because the distance is zero there.
    # That is true only of the i-th TERM.  `upper` is a MIN over all n terms, so
    # a NEIGHBOUR's term can be the binding one and the band can sit strictly
    # inside the cap -- in which case a larger gamma pushes it OUT toward the
    # cap and containment can IMPROVE.  It is monotone and saturating, never
    # unchanged by theorem.  The probe therefore reports the containment of the
    # TRUE (noiseless) function at each factor next to `cap`, the count of
    # anchors with |f_i - y_i| <= eps, which is the ceiling no factor can pass.
    let
        rng = MersenneTwister(31415)
        D, n, nq = 3, 300, 400
        X = 2 .* rand(rng, D, n) .- 1                # :pm1, the shipped frame
        fI(v) = sin(2.3 * v[1]) + 0.5 * v[2]^2 - 0.8 * v[3] + 0.35 * v[1] * v[2]
        ytrue = [fI(view(X, :, j)) for j in 1:n]
        y = ytrue .+ 0.05 * (maximum(ytrue) - minimum(ytrue)) .* randn(rng, n)
        Q = 2 .* rand(rng, D, nq) .- 1

        factors = (1.0, 2.5, 10.0)
        ms = [NSMTrainer.fit_nsm(X, y, NSMTrainer.nsm_settings(D; gamma_factor=fc))
              for fc in factors]

        eps_same = all(k -> exacteq(ms[k].eps, ms[1].eps), 2:3)
        raw_same = all(k -> exacteq(ms[k].gamma_raw, ms[1].gamma_raw), 2:3)
        scale_exact = all(k -> ms[k].gamma === factors[k] * ms[k].gamma_raw, 1:3)
        shipped_default = ms[2].gamma_factor === NSMTrainer.NSM_DEFAULT_GAMMA_FACTOR()

        b23_viol = 0
        mono_viol = 0
        e = ms[1].eps
        cap = count(i -> abs(ytrue[i] - y[i]) <= e, 1:n)
        cover = zeros(Int, 3)
        atcap = zeros(Int, 3)
        prev_l = fill(Inf, n)
        prev_u = fill(-Inf, n)
        for k in 1:3
            m = ms[k]
            for i in 1:n
                _, pl, pu = NSMTrainer.nsm_predict(m, view(X, :, i))
                (pu > y[i] + m.eps) && (b23_viol += 1)          # B2, tolerance 0
                (pl < y[i] - m.eps) && (b23_viol += 1)          # B3, tolerance 0
                (k > 1 && (pu < prev_u[i] || pl > prev_l[i])) && (mono_viol += 1)
                prev_u[i] = pu
                prev_l[i] = pl
                (pl <= ytrue[i] <= pu) && (cover[k] += 1)
                (pu === y[i] + m.eps && pl === y[i] - m.eps) && (atcap[k] += 1)
            end
        end

        qw = zeros(Float64, 3)
        qmono = 0
        qprev_l = fill(Inf, nq)
        qprev_u = fill(-Inf, nq)
        for k in 1:3
            for j in 1:nq
                _, pl, pu = NSMTrainer.nsm_predict(ms[k], view(Q, :, j))
                (k > 1 && (pu < qprev_u[j] || pl > qprev_l[j])) && (qmono += 1)
                qprev_u[j] = pu
                qprev_l[j] = pl
                qw[k] += pu - pl
            end
            qw[k] /= nq
        end

        cover_monotone = cover[1] <= cover[2] <= cover[3] && cover[3] <= cap
        push!(P, Probe("I  gamma_factor is a wrapper (eps unscaled)",
                       eps_same && raw_same && scale_exact && shipped_default &&
                       b23_viol == 0 && mono_viol == 0 && qmono == 0 && cover_monotone,
                       @sprintf("eps=%.6g (identical x3) gamma_raw=%.6g; B2/B3 viol %d, band-monotone viol %d train / %d query; TRAIN containment of true f at factor 1.0/2.5/10: %d/%d/%d of %d (ceiling %d = |f-y|<=eps); band already AT the y+/-eps cap: %d/%d/%d; mean QUERY width %.4g/%.4g/%.4g",
                                ms[1].eps, ms[1].gamma_raw, b23_viol, mono_viol, qmono,
                                cover[1], cover[2], cover[3], n, cap,
                                atcap[1], atcap[2], atcap[3],
                                qw[1], qw[2], qw[3])))
    end

    return P
end

# -----------------------------------------------------------------------------
# The gate
# -----------------------------------------------------------------------------

function run_gate()
    cases = build_cases()

    w_eps    = Worst("eps (relative)", EPS_RTOL)
    w_gam    = Worst("gamma (relative)", GAMMA_RTOL)
    w_upper  = Worst("upper bound (relative)", PRED_RTOL)
    w_lower  = Worst("lower bound (relative)", PRED_RTOL)
    w_pred   = Worst("pred/midpoint (relative)", PRED_RTOL)
    w_b1     = Worst("B1 lower <= upper", BAND_RTOL)
    w_b2     = Worst("B2 upper <= y_i + eps", BAND_EXACT)
    w_b3     = Worst("B3 lower >= y_i - eps", BAND_EXACT)
    w_b4     = Worst("B4 |pred - y_i| <= eps", BAND_RTOL)
    w_b5     = Worst("B5 y_i in [low-eps, up+eps]", BAND_RTOL)
    w_batch  = Worst("batch == single (exact)", 0.0)

    strict_ok = 0            # how often y_i really lies inside [lower, upper]
    strict_total = 0
    nonfinite_cells = String[]

    println("=" ^ 78)
    println("NSM equivalence gate: NSMTrainer (production) vs NSMReference (oracle)")
    println("=" ^ 78)
    @printf("%-28s %12s %12s %10s %7s\n", "case", "eps", "gamma", "y-span", "queries")
    println("-" ^ 78)

    for c in cases
        D, n = size(c.Xraw)

        # --- both ports onto identical normalised coordinates -----------------
        refmodel = NSMReference.fit(c.Xraw, c.y)
        lo, hi = NSMReference.normalisation_bounds(c.Xraw)
        Xn = NSMReference.normalise_inputs(c.Xraw, lo, hi)

        # EQUIVALENCE IS TESTED AT gamma_factor = 1.0, DELIBERATELY.  The arm
        # SHIPS at 2.5 (Diona's L_H_factor, adopted for slate symmetry), but
        # that factor is not in opt_smgo_d.m, so the reference has no such
        # thing to agree with.  1.0 is the source; the assertion below is what
        # makes "the port is exact" a statement about the port rather than
        # about the factor.  The factor itself is gated by probe I.
        s = NSMTrainer.nsm_settings(D; frame=:unit01, gamma_factor=1.0)
        @assert s.frame === :unit01
        @assert s.gamma_factor === 1.0
        @assert NSMTrainer.nsm_frame_diam(s.frame, D) == sqrt(D)
        prod = NSMTrainer.fit_nsm(Xn, c.y, s)
        # At factor 1.0 the wrapper must be the identity, bit-for-bit: the
        # cone's gamma IS the estimator's gamma.
        @assert prod.gamma === prod.gamma_raw

        yscale = max(maximum(abs, c.y), maximum(c.y) - minimum(c.y))

        record!(w_eps, reldiff(prod.eps, refmodel.eps, abs(refmodel.eps)),
                @sprintf("%s: prod eps=%.17g ref eps=%.17g", c.label, prod.eps, refmodel.eps);
                absval=abs(prod.eps - refmodel.eps))
        record!(w_gam, reldiff(prod.gamma, refmodel.gamma, abs(refmodel.gamma)),
                @sprintf("%s: prod gamma=%.17g ref gamma=%.17g", c.label, prod.gamma, refmodel.gamma);
                absval=(isfinite(prod.gamma) && isfinite(refmodel.gamma)) ?
                       abs(prod.gamma - refmodel.gamma) : NaN)

        # --- evaluation points -----------------------------------------------
        rng = MersenneTwister(hash(c.label) % 100000)
        nq_in, nq_out = 60, 8
        Qraw = Matrix{Float64}(undef, D, nq_in + nq_out)
        for j in 1:nq_in                      # inside the training box
            for d in 1:D
                Qraw[d, j] = lo[d] + (hi[d] - lo[d]) * rand(rng)
            end
        end
        for j in (nq_in + 1):(nq_in + nq_out) # OUTSIDE it -- neither port clips
            for d in 1:D
                span = hi[d] - lo[d]
                Qraw[d, j] = lo[d] - 0.3 * span + 1.6 * span * rand(rng)
            end
        end
        # plus training points themselves (distance exactly 0)
        tp = collect(1:max(1, n ÷ 100):n)[1:min(100, n)]
        Eraw = hcat(Qraw, c.Xraw[:, tp])
        En = NSMReference.normalise_inputs(Eraw, lo, hi)
        m = size(Eraw, 2)

        for j in 1:m
            rl, ru = NSMReference.bounds(refmodel, Eraw[:, j])
            rp = 0.5 * (ru + rl)
            pp, pl, pu = NSMTrainer.nsm_predict(prod, En[:, j])
            tagged(q) = @sprintf("%s point %d/%d: prod=%.17g ref=%.17g", c.label, j, m, q[1], q[2])
            absd(a, b) = (isfinite(a) && isfinite(b)) ? abs(a - b) : NaN
            record!(w_upper, reldiff(pu, ru, yscale), tagged((pu, ru)); absval=absd(pu, ru))
            record!(w_lower, reldiff(pl, rl, yscale), tagged((pl, rl)); absval=absd(pl, rl))
            record!(w_pred,  reldiff(pp, rp, yscale), tagged((pp, rp)); absval=absd(pp, rp))
        end

        # --- batch path must reproduce the single-query path exactly ----------
        bp, bl, bu, bstat, _ = NSMTrainer.nsm_predict_batch(prod, En)
        for j in 1:m
            pp, pl, pu = NSMTrainer.nsm_predict(prod, En[:, j])
            agree = (bp[j] === pp || (isnan(bp[j]) && isnan(pp))) &&
                    (bl[j] === pl || (isnan(bl[j]) && isnan(pl))) &&
                    (bu[j] === pu || (isnan(bu[j]) && isnan(pu))) &&
                    bstat[j] === NSMTrainer.nsm_status(pp, pl, pu)
            record!(w_batch, agree ? 0.0 : 1.0,
                    @sprintf("%s point %d: batch=(%.17g,%.17g,%.17g) single=(%.17g,%.17g,%.17g)",
                             c.label, j, bp[j], bl[j], bu[j], pp, pl, pu))
        end

        # --- THE BAND PROPERTY, on the production model, at EVERY training pt --
        # Skipped only where the ports themselves are non-finite by design
        # (gamma = +Inf from coincident samples, or eps = NaN from ladder
        # exhaustion); those cells are named in the summary instead.
        band_applicable = isfinite(prod.eps) && isfinite(prod.gamma)
        if band_applicable
            for i in 1:n
                pp, pl, pu = NSMTrainer.nsm_predict(prod, Xn[:, i])
                yi = c.y[i]
                sc = max(yscale, abs(yi))
                at = @sprintf("%s training point %d/%d (y=%.17g, eps=%.17g)", c.label, i, n, yi, prod.eps)
                record!(w_b1, max(0.0, (pl - pu)) / sc, at)                     # B1
                record!(w_b2, max(0.0, pu - (yi + prod.eps)) / sc, at)          # B2
                record!(w_b3, max(0.0, (yi - prod.eps) - pl) / sc, at)          # B3
                record!(w_b4, max(0.0, abs(pp - yi) - prod.eps) / sc, at)       # B4
                record!(w_b5, max(0.0, max(pl - prod.eps - yi, yi - (pu + prod.eps))) / sc, at)  # B5
                strict_total += 1
                (pl <= yi <= pu) && (strict_ok += 1)
            end
        else
            push!(nonfinite_cells, c.label)
        end

        @printf("%-28s %12.6g %12.6g %10.4g %7d\n", c.label, prod.eps, prod.gamma, yscale, m)
    end

    println("-" ^ 78)
    println()
    println("EQUIVALENCE (production vs literal dense reference)")
    for w in (w_eps, w_gam, w_upper, w_lower, w_pred, w_batch)
        report(w)
    end
    println()
    println("NSM BAND PROPERTY (production model, every training point)")
    for w in (w_b1, w_b2, w_b3, w_b4, w_b5)
        report(w)
    end
    println()
    @printf("  strict containment y_i in [lower(x_i), upper(x_i)]: %d/%d (%.4f) -- MEASURED, NOT ASSERTED\n",
            strict_ok, strict_total, strict_ok / max(1, strict_total))
    if !isempty(nonfinite_cells)
        println("  band property not applicable (non-finite eps or gamma BY DESIGN) in: ",
                join(nonfinite_cells, ", "))
        println("  those cells are still fully covered by the equivalence checks above.")
    end
    println()

    println("BOUNDARY PROBES (exact ties; randomised cells cannot reach these)")
    probes = run_probes()
    for pr in probes
        @printf("  [%s] %-40s %s\n", pr.ok ? "PASS" : "FAIL", pr.name, pr.detail)
    end
    println()

    all_w = (w_eps, w_gam, w_upper, w_lower, w_pred, w_batch, w_b1, w_b2, w_b3, w_b4, w_b5)
    ok = all(passed, all_w) && all(pr -> pr.ok, probes)
    println("=" ^ 78)
    println(ok ? "GATE: PASS" : "GATE: FAIL")
    println("=" ^ 78)
    return ok
end

ok = run_gate()
exit(ok ? 0 : 1)
