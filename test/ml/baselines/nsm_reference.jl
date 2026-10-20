# =============================================================================
# nsm_reference.jl -- LITERAL reference port of the SMGO-Delta noise (eps) and
#                     Lipschitz (gamma) estimators, and the Nonlinear
#                     Set-Membership (NSM) estimator built on them.
#
# SOURCE OF AUTHORITY
#   /home/ls122/code/smgo-delta/opt_smgo_d.m
#     line 145        diam = sqrt(D)
#     lines 148-154   fgam_0 = 1e-6, feps_0 = 0
#     lines 252-279   noise estimate feps
#     lines 295-306   Lipschitz estimate fgam
#     lines 482-483   the set-membership upper/lower bounds
#   /home/ls122/code/smgo-delta/util/normd2real.m  (coordinate convention)
#
# PURPOSE
#   This file exists ONLY as an ORACLE against which the fast production NSM
#   port is validated.  It is a deliberately naive, line-for-line
#   transliteration of the MATLAB: dense n x n ball matrix, the same loop
#   structure, the same NaN sentinel, the same filters, the same silent
#   division by zero.  NOTHING here is optimised.  Fidelity and clarity beat
#   speed.  It is only ever expected to run at n = 50..500.
#
#   Every block carries the MATLAB line number it transliterates.  Where the
#   two languages disagree semantically (MATLAB `max`/`min` OMIT NaN, Julia
#   `maximum`/`minimum` PROPAGATE it; MATLAB for-loops fall through on
#   exhaustion) the MATLAB behaviour is reproduced explicitly and flagged in a
#   comment.
#
# ESTIMATOR (lines 482-483)
#   ybar(x) = min_i ( y_i + eps + gamma * ||x - x_i|| )
#   ylow(x) = max_i ( y_i - eps - gamma * ||x - x_i|| )
#   fhat(x) = 0.5 * ( ybar(x) + ylow(x) )
#   over ALL sampled points, with no ball restriction.  Note that the
#   `+ 2*feps` at lines 440-441 belongs to the acquisition function's lambda,
#   NOT to the estimator, and is deliberately absent here.
#
# WHAT THIS IS NOT
#   feps reduces exactly to  Mbar / (2 * nbar)  -- the mean per-point
#   neighbour maximum divided by twice the mean ball occupancy -- because the
#   numerator counts active POINTS while the denominator counts ORDERED PAIRS
#   (line 279 divides by 2 * sum(ball_idx,'all'), and ball_idx is symmetric
#   with a zero diagonal).  It therefore shrinks like 1/density: adding points
#   inside a fixed ball drives feps toward 0 even at unchanged noise.  It is
#   NOT the Milanese & Novara (2004) textbook estimator and must never be
#   written up as "half the maximum local discrepancy".  Ported verbatim
#   regardless -- the MATLAB is the authority.
#
#   Likewise, the asymmetry is deliberate and must be preserved: eps is LOCAL
#   (ball-restricted, line 263) while gamma is GLOBAL (all pairs, line 298, no
#   ball restriction).  Do not "improve" either one.
#
# PORT DECISIONS (points the MATLAB source does not settle, because it
# generates its own samples in [0,1]^D and never sees a fixed dataset)
#
#   P1. INPUT NORMALISATION.  `diam = sqrt(D)` (line 145) is the exact
#       corner-to-corner diameter of the unit hypercube, and X_n(1:D,:) holds
#       NORMALISED coordinates (the black box is called as
#       f(normd2real(x, bounds)), lines 236/241).  A benchmark dataset arrives
#       in arbitrary real coordinates, so this port min-max normalises each
#       input dimension using the TRAINING pool's per-dimension min/max only.
#       Queries are NOT clipped: a query outside the training box gets a
#       normalised coordinate outside [0,1] and may sit further than diam from
#       every sample.  `diam` stays the literal sqrt(D), never the empirical
#       pairwise maximum distance.  y is NOT normalised anywhere (eps is in
#       raw-y units; gamma is raw-y per unit-normalised-x).
#
#   P2. DEGENERATE INPUT DIMENSION (hi == lo).  Not reachable in the source.
#       This port maps such a dimension to the constant 0.0, so it contributes
#       zero to every distance.
#
#   P3. eps == NaN (radius-loop exhaustion, see below).  Reproduced
#       FAITHFULLY, silently, exactly as MATLAB does -- including the
#       downstream collapse to gamma == 1e-6 and NaN predictions.  This file is
#       the oracle; it does not repair the source.  If the production arm
#       chooses to raise or to fall back to eps = 0 instead, that divergence is
#       a property of the production port and must be gated by its own test.
#
#   P4. DUPLICATE x WITH DIFFERENT y.  Reproduced faithfully: the gamma ratio
#       (line 301) divides by a zero distance and yields +Inf, so gamma = +Inf.
#       No deduplication, no distance floor.  The source has no guard.
#
#   P5. n == 0.  Not reachable in the source (iter >= 1 always).  `fit` throws.
#
# API CONTRACT (shared with the production port, so an equivalence test can
# call both by module qualification: NSMReference.fit vs NSM.fit, etc.)
# Nothing is exported; callers qualify.
#
#   estimate_eps(X, y)                -> Float64        # X is D x n, NORMALISED
#   estimate_gamma(X, y, eps)         -> Float64        # X is D x n, NORMALISED
#   normalisation_bounds(Xraw)        -> (lo, hi)
#   normalise_inputs(Xraw, lo, hi)    -> Matrix{Float64}
#   fit(Xraw, y)                      -> NSMReferenceModel
#   bounds(model, xq::AbstractVector) -> (ylow, ybar)
#   bounds(model, Xq::AbstractMatrix) -> (Vector, Vector)
#   predict(model, xq::AbstractVector)-> Float64
#   predict(model, Xq::AbstractMatrix)-> Vector{Float64}
#
#   Matrices are D x n / D x m (columns are points), matching X_n's layout.
# =============================================================================

module NSMReference

using LinearAlgebra: norm

"""
    NSMReferenceModel

A fitted NSM model.  `X` holds the NORMALISED training inputs (D x n), `lo`
and `hi` the per-dimension normalisation bounds taken from the training pool
(port decision P1), `eps` the noise estimate (line 279) and `gamma` the
Lipschitz estimate (line 306).
"""
struct NSMReferenceModel
    X::Matrix{Float64}     # D x n, normalised coordinates (MATLAB X_n(1:D,:))
    y::Vector{Float64}     # MATLAB X_n(XN_ROW_FVAL, :)
    lo::Vector{Float64}
    hi::Vector{Float64}
    eps::Float64           # MATLAB feps
    gamma::Float64         # MATLAB fgam
end

# -----------------------------------------------------------------------------
# MATLAB semantic helpers.
#
# MATLAB's max/min OMIT NaN by default; Julia's maximum/minimum PROPAGATE it.
# Every reduction in the source (lines 276, 300-301, 306, 482, 483) is a MATLAB
# max/min, so all of them go through these helpers.  On NaN-free input they are
# identical to maximum/minimum.
#
# Both return NaN for an all-NaN (or empty) input.  MATLAB returns [] for an
# empty input, but every call site in the source is guarded to be non-empty
# (lines 275 and 299), so the distinction never arises.
# -----------------------------------------------------------------------------

"MATLAB `max` over a Float64 iterable: NaN entries are OMITTED, not propagated."
function matlab_maximum(values)
    best = NaN
    for v in values
        isnan(v) && continue          # MATLAB max omits NaN
        if isnan(best) || v > best
            best = v
        end
    end
    return best
end

"MATLAB `min` over a Float64 iterable: NaN entries are OMITTED, not propagated."
function matlab_minimum(values)
    best = NaN
    for v in values
        isnan(v) && continue          # MATLAB min omits NaN
        if isnan(best) || v < best
            best = v
        end
    end
    return best
end

"""
    column_distances(X, x0) -> Vector{Float64}

MATLAB `vecnorm(X(1:D, :) - repmat(x0, 1, n))` (lines 263, 301, 480).

Euclidean distance from `x0` to each COLUMN of `X`, in normalised coordinates.
Deliberately written as a dense per-column loop.

D == 1 note: MATLAB's `vecnorm` on a 1 x n row vector defaults to dim 1 and so
returns the elementwise absolute value, which is the correct 1-D distance.  A
column-wise Julia norm over a 1 x n matrix gives exactly the same thing, so no
special case is needed.
"""
function column_distances(X::AbstractMatrix{<:Real}, x0::AbstractVector{<:Real})
    n = size(X, 2)
    d = Vector{Float64}(undef, n)
    for j in 1:n
        d[j] = norm(Float64.(X[:, j]) .- Float64.(x0))
    end
    return d
end

# -----------------------------------------------------------------------------
# Noise estimate.  MATLAB lines 252-279.
# -----------------------------------------------------------------------------

"""
    estimate_eps(X, y) -> Float64

Literal port of MATLAB lines 252-279 (`feps`).  `X` is D x n in NORMALISED
coordinates, `y` is the length-n vector of raw objective values.  Only sampled
points enter; queries and candidate points never do.

Because the MATLAB block re-derives everything from scratch from all `iter`
columns on every iteration, evaluating this once in batch with n = iter
reproduces the final MATLAB value exactly.

Exact formula (line 279):

    feps = ( sum over points j with >=1 in-ball neighbour of max_{i in N(j)} |y_i - y_j| )
           / ( 2 * sum(ball_idx, 'all') )

where N(j) = { i != j : ||x_i - x_j|| <= ball_rad* } and ball_rad* is the
radius at which the schedule broke.  `sum(ball_idx,'all')` counts ORDERED
pairs; do not "fix" that against the per-point numerator (see the header).

Returns NaN, silently, on radius-loop exhaustion -- see the comment at the
fall-through below and port decision P3.
"""
function estimate_eps(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})
    D, n = size(X)
    length(y) == n || throw(DimensionMismatch("estimate_eps: size(X,2) = $n but length(y) = $(length(y))"))

    # line 252: `if iter > 1`.  At iter == 1 the whole block is skipped and feps
    # keeps feps_0 = 0 (lines 153-154).
    n < 2 && return 0.0

    diam = sqrt(D)                             # line 145, a CONSTANT from D only

    # ---- radius schedule, lines 259-268 -------------------------------------
    # "increment 'rad' when there are no points inside the ball of any other"
    #
    # CORRECTION TO THIS REFERENCE (2026-09-20).  This loop previously read
    # `for rad in 0.1:0.1:1.0`, which is NOT line 259's schedule: MATLAB's colon
    # materialises `base + k*step` in double precision, while Julia's range is
    # TwicePrecision and returns the correctly-rounded double of the exact
    # decimal instead.  They differ by one ulp at rungs 3 and 7
    # (0.29999999999999999 vs 0.30000000000000004; 0.69999999999999996 vs
    # 0.70000000000000007), which shifts `ball_rad` by an ulp and can flip an
    # exact tie in the INCLUSIVE test at line 263.  The reference, not the
    # production port, was the one diverging from the MATLAB here.  Written out
    # as the colon's own arithmetic.  No other line of this file was touched.
    ball = falses(n, n)
    for rad in (0.1 + (k - 1) * 0.1 for k in 1:10)   # line 259, exactly 10 values
        ball_rad = rad * diam / 2              # line 260; caps at diam/2 = sqrt(D)/2
        ball = falses(n, n)                    # line 261, REBUILT from scratch each rad
        for k in 1:n
            dk = column_distances(X, X[:, k])
            for j in 1:n
                ball[k, j] = dk[j] <= ball_rad # line 263, INCLUSIVE <=
            end
            ball[k, k] = false                 # line 264, diagonal zeroed
        end
        if sum(ball) > 0                       # line 266
            break                              # line 267
        end
    end
    # MATLAB for-loops do NOT error on exhaustion (no pair within sqrt(D)/2 of
    # another at any radius).  The loop falls through with rad = 1.0 and `ball`
    # left as the all-false matrix rebuilt at that last radius, and the code
    # below then divides 0 by 0 and returns NaN silently.  Reproduced exactly:
    # no error, no warning, no repair.  See port decision P3.
    # (Line 270's `logical(ball_idx)` is a no-op here: `ball` is already a
    # BitMatrix.)

    # ---- collecting the noise values, lines 273-276 -------------------------
    # ns_lkup is NaN-initialised (line 273); entry j is written ONLY if row j
    # has at least one in-ball neighbour, so isolated points keep NaN.
    ns_lkup = fill(NaN, n)                     # line 273
    for j in 1:n
        if sum(view(ball, j, :)) > 0           # line 275
            # line 276: MATLAB max OMITS NaN, so a NaN y among the neighbours
            # is skipped rather than poisoning the maximum.  matlab_maximum
            # reproduces that; a naive Julia `maximum` would not.
            ns_lkup[j] = matlab_maximum(abs(y[i] - y[j]) for i in 1:n if ball[j, i])
        end
    end

    # ---- line 279 -----------------------------------------------------------
    # `sum(ns_lkup(ns_lkup > 0))`.  NaN > 0 is false in BOTH languages, so the
    # filter drops isolated points; it also drops entries that are exactly 0,
    # which contribute nothing to a sum anyway.  MATLAB's sum of an empty array
    # is 0, which `numerator = 0.0` reproduces.  The denominator is built
    # independently from `ball`, not from the filtered set.
    numerator = 0.0
    for j in 1:n
        if ns_lkup[j] > 0                      # line 279 filter
            numerator += ns_lkup[j]
        end
    end
    ordered_pairs = sum(ball)                  # sum(ball_idx,'all'), ORDERED pairs

    # 0.0 / 0 is NaN in Julia exactly as 0/0 is NaN in MATLAB.
    return numerator / (2 * ordered_pairs)     # line 279
end

# -----------------------------------------------------------------------------
# Lipschitz estimate.  MATLAB lines 295-306.
# -----------------------------------------------------------------------------

"""
    estimate_gamma(X, y, eps) -> Float64

Literal port of MATLAB lines 295-306 (`fgam`).  `X` is D x n in NORMALISED
coordinates.

    gamma = max_i [ if E_i nonempty then max_{j in E_i} (|y_j - y_i| - 2*eps) / ||x_j - x_i||
                    else 1e-6 ]
    E_i   = { j : |y_j - y_i| > 2*eps }      # STRICT >, over ALL j, no ball restriction

Notes reproduced verbatim:

  * The eligibility test (line 298) runs over all n columns INCLUDING j == i;
    self is excluded automatically because |y_i - y_i| = 0 and the test is
    strict.  There is no locality restriction here -- gamma is GLOBAL where
    eps is local.
  * 1e-6 (line 303) is a PER-ANCHOR fallback, not a global clamp.  If every
    anchor has an eligible partner it never enters gam_db and gamma may
    legitimately come out smaller than 1e-6.  That non-uniformity is real.
  * gamma is never exactly 0 (every gam_db entry is 1e-6 or a max of strictly
    positive ratios), but it can be +Inf when two samples share coordinates and
    differ in y by more than 2*eps (port decision P4).
  * If `eps` is NaN, every comparison at line 298 is false, so every anchor
    takes the 1e-6 fallback and gamma = 1e-6 while eps stays NaN.  Faithful.
  * `fgam_prev` (line 295) exists only for cone-height rescaling in the
    optimiser (line 348) and has no role in the estimator, so it is absent.
"""
function estimate_gamma(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real}, eps::Real)
    D, n = size(X)
    length(y) == n || throw(DimensionMismatch("estimate_gamma: size(X,2) = $n but length(y) = $(length(y))"))

    # line 252 guard again: at iter == 1 the block is skipped and fgam keeps
    # fgam_0 = 1e-6 (lines 148-149).
    n < 2 && return 1e-6

    gam_db = zeros(Float64, n)                 # line 296; inert, every entry is overwritten
    for i in 1:n
        # line 298: STRICT >, over ALL j, no ball restriction.
        elg_idx = [abs(y[j] - y[i]) > 2 * eps for j in 1:n]

        if sum(elg_idx) > 0                    # line 299
            dist = column_distances(X, X[:, i])
            # lines 300-301: numerator is the y-gap DEFLATED by 2*eps (one eps
            # per endpoint); denominator is the full Euclidean distance in
            # normalised coordinates.  A duplicate x gives 0 in the denominator
            # and hence +Inf, exactly as MATLAB.  0/0 cannot occur: the
            # numerator is strictly positive on E_i by construction.
            # MATLAB max again omits NaN.
            gam_db[i] = matlab_maximum(
                (abs(y[j] - y[i]) - 2 * eps) / dist[j] for j in 1:n if elg_idx[j]
            )
        else
            gam_db[i] = 1e-6                   # line 303, per-anchor fallback
        end
    end

    return matlab_maximum(gam_db)              # line 306
end

# -----------------------------------------------------------------------------
# Coordinate normalisation.  Port decisions P1/P2; no MATLAB counterpart,
# because the source samples directly in [0,1]^D (line 103, lines 236/241,
# util/normd2real.m).
# -----------------------------------------------------------------------------

"""
    normalisation_bounds(Xraw) -> (lo, hi)

Per-dimension min/max of the TRAINING pool (D x n).  Port decision P1: the
query set never contributes to these bounds.
"""
function normalisation_bounds(Xraw::AbstractMatrix{<:Real})
    D = size(Xraw, 1)
    lo = Vector{Float64}(undef, D)
    hi = Vector{Float64}(undef, D)
    for d in 1:D
        lo[d] = Float64(minimum(view(Xraw, d, :)))
        hi[d] = Float64(maximum(view(Xraw, d, :)))
    end
    return lo, hi
end

"""
    normalise_inputs(Xraw, lo, hi) -> Matrix{Float64}

Min-max map each dimension of a D x n matrix into [0,1] using the training
bounds.  This is the inverse of `normd2real` (util/normd2real.m).

Queries are NOT clipped: a coordinate outside [lo, hi] maps outside [0,1] and
may sit further than diam = sqrt(D) from every sample (port decision P1).
A degenerate dimension (hi == lo) maps to the constant 0.0 (port decision P2).
"""
function normalise_inputs(Xraw::AbstractMatrix{<:Real}, lo::AbstractVector{<:Real}, hi::AbstractVector{<:Real})
    D, n = size(Xraw)
    (length(lo) == D && length(hi) == D) ||
        throw(DimensionMismatch("normalise_inputs: bounds have length $(length(lo))/$(length(hi)) for D = $D"))
    Z = Matrix{Float64}(undef, D, n)
    for d in 1:D
        span = Float64(hi[d]) - Float64(lo[d])
        for j in 1:n
            Z[d, j] = span == 0 ? 0.0 : (Float64(Xraw[d, j]) - Float64(lo[d])) / span
        end
    end
    return Z
end

# -----------------------------------------------------------------------------
# Fit / predict.  MATLAB lines 482-483 for the bounds.
# -----------------------------------------------------------------------------

"""
    fit(Xraw, y) -> NSMReferenceModel

Normalise the inputs (port decision P1), then run `estimate_eps` (lines
252-279) followed by `estimate_gamma` (lines 295-306) on the normalised
coordinates, in that order -- gamma consumes the eps just computed, exactly as
the source does within one iteration.

`Xraw` is D x n with columns as points; `y` is length n in RAW units.
"""
function fit(Xraw::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})
    D, n = size(Xraw)
    n >= 1 || throw(ArgumentError("fit: need at least one sample (port decision P5)"))
    length(y) == n || throw(DimensionMismatch("fit: size(Xraw,2) = $n but length(y) = $(length(y))"))

    lo, hi = normalisation_bounds(Xraw)
    X = normalise_inputs(Xraw, lo, hi)
    yv = Vector{Float64}(undef, n)
    for j in 1:n
        yv[j] = Float64(y[j])
    end

    eps = estimate_eps(X, yv)
    gamma = estimate_gamma(X, yv, eps)
    return NSMReferenceModel(X, yv, lo, hi, eps, gamma)
end

"""
    bounds(model, xq) -> (ylow, ybar)

Literal port of MATLAB lines 482-483:

    ybar = min_i ( y_i + eps + gamma * ||x - x_i|| )
    ylow = max_i ( y_i - eps - gamma * ||x - x_i|| )

over ALL sampled points, with no ball restriction.  The extra `+ 2*feps` at
lines 440-441 belongs to the acquisition lambda and is deliberately NOT here.

`xq` is a raw-coordinate point (length D); it is normalised with the training
bounds and is not clipped.

The reductions go through `matlab_minimum`/`matlab_maximum` so that MATLAB's
NaN-omission is reproduced.  This matters in exactly one reachable case: when
gamma == +Inf (duplicate x, port decision P4) and the query coincides with a
sample, the term Inf * 0 is NaN, which MATLAB skips and a naive Julia
`minimum` would propagate.
"""
function bounds(model::NSMReferenceModel, xq::AbstractVector{<:Real})
    D = size(model.X, 1)
    length(xq) == D || throw(DimensionMismatch("bounds: query has length $(length(xq)) but D = $D"))
    xn = normalise_inputs(reshape(Float64.(collect(xq)), D, 1), model.lo, model.hi)[:, 1]
    dist = column_distances(model.X, xn)       # line 480

    n = length(model.y)
    ybar = matlab_minimum(model.y[i] + model.eps + model.gamma * dist[i] for i in 1:n)  # line 482
    ylow = matlab_maximum(model.y[i] - model.eps - model.gamma * dist[i] for i in 1:n)  # line 483
    return (ylow, ybar)
end

"""
    bounds(model, Xq::AbstractMatrix) -> (Vector, Vector)

Column-wise `bounds`, returning `(ylow, ybar)` vectors of length size(Xq, 2).
"""
function bounds(model::NSMReferenceModel, Xq::AbstractMatrix{<:Real})
    m = size(Xq, 2)
    ylow = Vector{Float64}(undef, m)
    ybar = Vector{Float64}(undef, m)
    for j in 1:m
        ylow[j], ybar[j] = bounds(model, Xq[:, j])
    end
    return (ylow, ybar)
end

"""
    predict(model, xq) -> Float64

The NSM point estimate: the midpoint of the set-membership interval,

    fhat(x) = 0.5 * ( ybar(x) + ylow(x) ).

At n == 1 the eps and gamma terms cancel and this is exactly y_1.
"""
function predict(model::NSMReferenceModel, xq::AbstractVector{<:Real})
    ylow, ybar = bounds(model, xq)
    return 0.5 * (ybar + ylow)
end

"""
    predict(model, Xq::AbstractMatrix) -> Vector{Float64}

Column-wise `predict`.
"""
function predict(model::NSMReferenceModel, Xq::AbstractMatrix{<:Real})
    m = size(Xq, 2)
    out = Vector{Float64}(undef, m)
    for j in 1:m
        out[j] = predict(model, Xq[:, j])
    end
    return out
end

end # module NSMReference
