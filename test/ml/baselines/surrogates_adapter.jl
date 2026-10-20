# Surrogates.jl comparator arms for the Part-1 data-only benchmark: `:rbf`
# (RadialBasis) and `:idw` (InverseDistanceSurrogate).
#
# ONE shim serves both arms. Both are classical, non-parametric, zero-tuning
# interpolators from the same package, they take the same inputs, they produce
# a point prediction and no interval, and they share every container/bounds
# convention. Splitting them into two files would duplicate the conversion
# layer and the failure handling verbatim.
#
# ============================================================================
# FAIRNESS CONTRACT -- PACKAGE DEFAULTS, VERBATIM.
#
# Neither arm is tuned, regularised, cross-validated or otherwise adapted to
# the benchmark. `fit_rbf` and `fit_idw` call the package constructors with
# NO keyword arguments at all, so whatever Surrogates.jl ships as its default
# IS the configuration. That is the whole point: the comparison is against the
# library as an ordinary user would reach for it, and a single fixed
# configuration is applied to all 36 cells (the same symmetry argument the
# NN/PINN arms use). `RBFSettings`/`IDWSettings` therefore carry no model
# hyperparameters -- only the seed (unused; both arms are deterministic) and,
# for `:rbf`, a harness-level memory guard that never changes a fit that
# proceeds (see FAILURE AT SCALE below).
#
# The defaults being inherited rather than chosen are recorded verbatim in
# `SURROGATES_DEFAULTS_PROVENANCE`, which goes into the settings hash.
# ============================================================================
#
# ============================================================================
# WHAT THESE ARMS DO AND DO NOT PROVIDE.
#
# * NO INTERVAL. Both are deterministic interpolators with no uncertainty
#   model. `lower`/`upper` are always NaN, and `interval_kind` is `:none`.
#   Any coverage/width column for these arms is vacuous by construction, not
#   by failure. Do NOT let a downstream table read a missing interval as a
#   zero-width one.
#
# * EXACT INTERPOLATION OF NOISE. RadialBasis at defaults fits the n x n
#   kernel system alone -- `regularization = 0.0`, and no polynomial tail --
#   so it reproduces the *noisy* training targets to machine precision
#   (measured: max |fit - noisy y| = 2.0e-15 for RadialBasis, exactly 0.0 for
#   InverseDistanceSurrogate, in both Surrogates v6.10.0 and v8.0.0). On the
#   5%-noise cells these arms are fitting the noise, deliberately and by the
#   package's own design. This is expected behaviour to report, not a bug to
#   patch by adding regularisation -- adding it would break the
#   defaults-verbatim contract above.
#
# * NO POLYNOMIAL TAIL, HENCE NO ANCHORED EXTRAPOLATION. Surrogates v8's own
#   `RadialBasis` docstring: "No polynomial tail is fitted. `_calc_coeffs`
#   solves the kernel system alone, so this field [`dim_poly`] records the
#   kernel's requirement and does not describe the model." With `linearRadial`
#   there is no constant term, so predictions outside the training hull are
#   unanchored. The benchmark queries the same box it trains in, so this
#   mostly bites near the boundary.
#
# * `scale_factor = 0.5` IS A NO-OP AT THE DEFAULT KERNEL. `linearRadial` is
#   positively homogeneous, so the scale cancels between the interpolation
#   matrix and the basis evaluation (measured identical to all printed digits
#   at 0.5 / 1.0 / 10.0). Do not present it as a meaningful or tuned default.
#
# * IDW does NOT take the package default `p = 1.0`. That exponent is
#   degenerate above one dimension -- the far field outvotes the near field
#   and the surrogate collapses to a global mean. `p = dim + 2` is used
#   instead, by one dimensional rule on every cell. See `idw_exponent`.
# ============================================================================
#
# ============================================================================
# FAILURE AT SCALE -- EXPECTED, RECORDED, NEVER PAPERED OVER.
#
# `:rbf` builds a DENSE n x n interpolation matrix and factorises it. Cost was
# MEASURED (2026-09-21, one process per n, peak VmHWM), not modelled: 12.1
# bytes/n^2 at Float64 and 6.04 at Float32, stable to 4% over n = 8,000..48,000.
# An earlier 16*n^2 guess was ~32% high.
#
# MEMORY IS THE BINDING CONSTRAINT, NOT TIME. The numbers below are this
# file's own gate (`surrogates_adapter_tests.jl` testset 7), not estimates
# carried over from an earlier draft:
#
#     cell                     n_train   est. peak    fit time       status
#     ishigami  sobol 500D       1500     0.04 GiB    0.15 s (m)    fits
#     hartmann6 sobol 2000D     12000     2.1  GiB    ~8.6 s (e)    fits
#     rosenbrock sobol 2000D    16000     3.8  GiB   17.4 s  (m)    fits
#     ishigami  grid 50^3      125000   176.1  GiB      --         Float64
#                                                                  does NOT fit
#     ishigami  grid 50^3      125000    87.8  GiB   31.7 min (m)   FITS, f32
#
# (m) measured on this machine.
#
# Read the last two rows together. At Float64 the 50^3 system is 176.1 GiB
# against 96 GiB of budget (and 157 GiB of machine), so it does not fit. At
# Float32 it is 87.8 GiB and fits, and the whole fit takes 31.7 minutes and
# reaches rel-L2 0.00093 -- better than the n = 8,000 rung's 0.00854, i.e. the
# accuracy trend continues and conditioning does not bite.
#
# MEMORY, NOT TIME, is what selects the precision. Time was never the obstacle:
# even at Float64 the solve would be about an hour. Two earlier versions of this
# table were wrong in opposite directions -- one claimed "~10 h" and called the
# cell infeasible on time; the next claimed 232.8 GiB from a 16*n^2 guess and
# called it excluded outright. Both were replaced by measurement.
#
# Cells 35/36 (the 50^3 Ishigami grid, n_train = 125000) cannot be fitted on
# any machine this benchmark is expected to run on -- for want of memory. That
# is a real, reportable property of the method -- dense RBF interpolation does
# not scale -- so it is surfaced as an explicit per-cell failure and NOT
# worked around:
#
#   * NO SUBSAMPLING, NO CAPPING, NO SWITCHING TO `sparse = true`. A silently
#     subsampled fit would be a different (and flattering) method wearing the
#     `:rbf` label.
#   * The whole cell is recorded as `train_ok = false`, `fit_time_s = NaN`,
#     and every query row of that cell gets `pred = lower = upper = NaN` with
#     status `:RBF_TRAIN_FAILED` -- exactly how the GP and NN arms record a
#     training failure. The cell still produces a fragment, with
#     `n_finite_pred == 0`.
#   * `RBFSettings.max_fit_bytes` (default 16 GiB) exists ONLY to turn an
#     allocation that is certain to exhaust memory into that recorded failure,
#     instead of an OS OOM kill that would lose the whole worker and every row
#     with it. It is NOT a model knob: no fit that proceeds is altered by it,
#     and it is chosen so that every cell that could actually succeed is
#     attempted (the largest feasible cell needs 3.8 GiB; the infeasible one
#     needs 232 GiB). Set it to `Inf` to disable the pre-flight entirely and
#     let the allocation be attempted for real.
#   * A genuine `OutOfMemoryError`, a singular/failed factorisation, or any
#     other exception during the fit lands in the same recorded-failure path.
#
# `:idw` has no fit at all (the constructor just stores the samples), so it
# has no fitting failure mode. Its cost is O(n) per query -- 125000 x 3000
# distance evaluations on cells 35/36, which is slow but finite.
# ============================================================================
#
# ============================================================================
# ENVIRONMENT.
#
# Surrogates.jl is NOT in `test/ml/Project.toml`. This file is therefore
# include-safe in an environment that lacks it: the import is guarded exactly
# the way `test/smoke_data.jl` guards `_GP_TRAINER_AVAILABLE` /
# `_NN_TRAINER_AVAILABLE`, so including it is always harmless and only the
# fit/predict entry points raise an actionable error. Check
# `surrogates_available()` before dispatching a cell here.
#
# VERSION WARNING. `Pkg.add("Surrogates")` against the existing
# `test/ml/Manifest.toml` silently resolves to Surrogates v6.10.0 (held back
# by the manifest's pins on Zygote v0.6.77 and QuasiMonteCarlo v0.3.11), not
# v8.0.0. v6's `InverseDistanceSurrogate` returns NaN for a query that
# coincides with a training point whenever the query container type differs
# from the storage container type. This shim is immune by construction -- it
# stores AND queries `Vector{Float64}` (see `_sample_vectors` /
# `_query_vector`), which v6 compares correctly -- but the version is still
# recorded in the provenance string, and `require_surrogates_version` can
# assert a minimum if the campaign wants one.
#
# CONTAINER RULES (both versions):
#   * samples as `Vector{Vector{Float64}}`; a `dim x n` Matrix is REJECTED by
#     the package (`DimensionMismatch`). Converting the harness's `dim x n`
#     layout is this module's job, not the caller's.
#   * `lb`/`ub` as `dim`-length Vectors, never scalars -- v6's IDW selects its
#     N-D branch on `length(lb) != 1`.
# ============================================================================
#
# Public API: `surrogates_available`, `rbf_settings`, `idw_settings`,
# `fit_rbf`, `fit_idw`, `surrogate_predict`, `train_surrogate_arm`,
# `predict_surrogate_arm`, `surrogate_settings_string`,
# `surrogate_model_type`, `surrogate_interval_kind`.
#
# This module deliberately does NOT construct `ArmQueryRow` / `ArmCellResult`:
# those live in `test/smoke_data.jl`, which includes the trainers, so the
# dependency only runs one way. Mirrors the `gp/trainer.jl` split exactly.

module SurrogateArms

export SURROGATE_ARMS, RBFSettings, IDWSettings, rbf_settings, idw_settings, idw_exponent,
       fit_rbf, fit_idw, surrogate_predict,
       train_surrogate_arm, train_surrogate_arm_detailed,
       predict_surrogate_arm, predict_surrogate_arm_timed,
       train_failed_status, predict_nonfinite_status,
       surrogate_settings_string, surrogate_model_type, surrogate_interval_kind,
       surrogates_available, surrogates_version, require_surrogates_version,
       rbf_dense_system_bytes, rbf_precision_for, RBF_PRECISION,
       RBF_BYTES_PER_N2_F64, RBF_BYTES_PER_N2_F32, SURROGATES_DEFAULTS_PROVENANCE,
       SURROGATES_DEFAULTS_SIGNATURES

# --------------------------------------------------------------------------
# Guarded import -- mirrors the `_GP_TRAINER_AVAILABLE` / `_NN_TRAINER_AVAILABLE`
# pattern at test/smoke_data.jl:27-44. Including this file in an environment
# without Surrogates.jl must not throw; only using an arm must.
# --------------------------------------------------------------------------
const _SURROGATES_AVAILABLE = try
    import Surrogates
    true
catch e
    @debug "Surrogates.jl unavailable in this Julia project (expected under " *
           "--project=test/ml, which does not depend on it); the :rbf and :idw " *
           "arms will error with an actionable message if invoked here" exception=e
    false
end

"""
    surrogates_available() -> Bool

Whether Surrogates.jl loaded in this process. `false` means the `:rbf` and
`:idw` arms cannot run here and must be dispatched to an environment that has
the package; including this file is still safe.
"""
surrogates_available() = _SURROGATES_AVAILABLE

const _SURROGATES_VERSION = _SURROGATES_AVAILABLE ?
    string(pkgversion(Surrogates)) : "unavailable"

"""
    surrogates_version() -> String

The loaded Surrogates.jl version, or `"unavailable"`. Recorded in the settings
hash: v6 and v8 differ on sample-coincident IDW queries and on whether
`RadialBasis` accepts a `regularization` keyword at all.
"""
surrogates_version() = _SURROGATES_VERSION

"""
    require_surrogates_version(minimum::VersionNumber)

Throw unless a Surrogates.jl at least `minimum` is loaded. Optional guard for a
campaign that wants to refuse v6.10.0 outright; nothing in this module calls it.
"""
function require_surrogates_version(minimum::VersionNumber)
    _SURROGATES_AVAILABLE || error("Surrogates.jl is not loaded in this process.")
    v = pkgversion(Surrogates)
    v >= minimum || error("Surrogates.jl $v is loaded but the campaign requires >= $minimum.")
    return v
end

const SURROGATE_ARMS = (:rbf, :idw)

"""
    SURROGATES_DEFAULTS_SIGNATURES

The two package constructor signatures both arms inherit, recorded VERBATIM as
Surrogates.jl declares them, read from the installed source rather than from
documentation. This is the provenance record for the fairness contract: it
states exactly which keyword values are in force even though the call sites in
`fit_rbf`/`fit_idw` pass no keywords at all.

    RadialBasis(x, y, lb, ub; rad = linearRadial(), scale_factor = 0.5,
                sparse = false, regularization = 0.0)
    InverseDistanceSurrogate(x, y, lb, ub; p = dim + 2)

Two of those values decide how the arms behave on this benchmark, and both
look like defects until you read them as defaults:

* `regularization = 0.0` means RadialBasis INTERPOLATES THE NOISE. There is
  no ridge and no polynomial tail, so the coefficients solve the n x n kernel
  system exactly and the fitted surface passes through every NOISY training
  target to machine precision (measured: max |fit - noisy y| = 2.0e-15). On
  the 5%-noise cells the arm is therefore fitting the noise realisation, in
  full, by the package's own design.
* IDW is the ONE documented departure from defaults-verbatim. `p = 1.0`, the
  package default, is not a tuning choice that happens to be poor -- it is
  degenerate for dim >= 2, collapsing the surrogate to a global weighted mean
  (measured: 5-17% of the truth's variance retained, within 3-16% of a
  constant predictor). An arm with no capacity for the function class is a
  strawman, not a baseline. `p = dim + 2` is applied instead, as ONE
  DIMENSIONAL RULE across all 36 cells -- no cross-validation, no per-cell or
  per-function search. See `idw_exponent` for the r^(dim-1-p) argument.

RBF IS NOT PATCHED, AND THAT IS THE POINT: no regularisation is added and
nothing is cross-validated, so it interpolates the noise by the package's own
design. IDW departs from the default on `p` alone, and only because the
default is structurally degenerate rather than merely untuned (above).

THE FAIRNESS ARGUMENT IS SYMMETRY OF RULE, NOT SYMMETRY OF PROVENANCE. Each
arm gets ONE fixed configuration applied to all 36 cells -- `p = dim + 2` is a
formula in the dimension, not a fitted value -- exactly as the NN/PINN arms
apply one fixed configuration to theirs. What would destroy the comparison is
per-cell or per-function tuning, or tuning one arm and not another to make a
point. Neither is done here.
"""
const SURROGATES_DEFAULTS_SIGNATURES =
    "RadialBasis(x, y, lb, ub; rad = linearRadial(), scale_factor = 0.5, " *
    "sparse = false, regularization = 0.0); " *
    "InverseDistanceSurrogate(x, y, lb, ub; p = dim + 2)"

"""
Verbatim record of the package defaults both arms inherit, read from the
installed source rather than from documentation. Enters the settings hash, so
a version change or a knob slipped in later is visible as a hash change. See
`SURROGATES_DEFAULTS_SIGNATURES` for the signatures and the fairness argument.
"""
const SURROGATES_DEFAULTS_PROVENANCE =
    "Surrogates.jl v" * _SURROGATES_VERSION * " package defaults, verbatim, no keywords passed: " *
    SURROGATES_DEFAULTS_SIGNATURES * ". " *
    "rad=linearRadial() is RadialFunction(0, z->norm(z)); scale_factor=0.5 is a no-op at this " *
    "positively-homogeneous kernel; regularization=0.0 (v8 only; the keyword does not exist in " *
    "v6) plus no polynomial tail (coeffs solve the n x n kernel system alone) means RadialBasis " *
    "INTERPOLATES THE NOISE at the 5%-noise cells. IDW departs from the default on p ALONE: " *
    "p=dim+2 rather than p=1.0, because p=1 is degenerate for dim>=2 (shell weight mass ~ " *
    "r^(dim-1-p) is non-decreasing, so the far field outvotes the near field and the surrogate " *
    "collapses to a global mean); IDW also interpolates the noisy targets exactly. Neither arm " *
    "is cross-validated or tuned per cell or per function: one fixed rule across all 36 cells, " *
    "and that symmetry is the fairness argument. Neither produces an interval."

# --------------------------------------------------------------------------
# Settings. No model hyperparameters by construction -- see FAIRNESS CONTRACT.
# --------------------------------------------------------------------------

"""
    DEFAULT_MAX_FIT_BYTES

Pre-flight ceiling on `:rbf`'s dense system (16 GiB). Harness plumbing, not a
model knob: it never alters a fit that proceeds, and it is set so that every
benchmark cell that could actually succeed is attempted (largest feasible cell
needs ~3.8 GiB; the 50^3 Ishigami grid needs ~233 GiB). `Inf` disables it.
"""
# MEASURED 2026-09-21 on the real Ishigami grid cells (one process per n, peak
# VmHWM). The constant is stable to 4% over n = 8,000..48,000 and Float32 is
# exactly half of Float64:
#
#     Float64   12.1 bytes/n^2      n=125,000 -> 176.1 GiB
#     Float32    6.04 bytes/n^2     n=125,000 ->  87.8 GiB  (measured 87.83)
#
# The previous 16*n^2 model was a guess and was ~32% high.
const RBF_BYTES_PER_N2_F64 = 12.1
const RBF_BYTES_PER_N2_F32 = 6.04

# The ONE precision `:rbf` fits at, on every cell. See rbf_precision_for.
const RBF_PRECISION = :f32

# 96 GiB. The machine has 157 GiB (WSL, .wslconfig memory=160GB); the measured
# n=125,000 Float32 peak is 87.8 GiB, so this admits it with ~8 GiB of margin
# while leaving ~60 GiB for shards running concurrently. Raise it only against a
# measurement, not a hope.
const DEFAULT_MAX_FIT_BYTES = 96.0 * 1024^3

"""
    RBFSettings(seed = 0; max_fit_bytes = DEFAULT_MAX_FIT_BYTES)

`:rbf` configuration. `seed` is carried only to fill `ArmCellResult.model_seed`
-- RadialBasis is deterministic and consumes no randomness. `max_fit_bytes` is
the memory pre-flight described above. There are deliberately no model
hyperparameters here.
"""
struct RBFSettings
    seed::Int
    max_fit_bytes::Float64
    precision::Symbol          # :f64, or :f32 where Float64 does not fit
end
function RBFSettings(seed::Int = 0; max_fit_bytes::Real = DEFAULT_MAX_FIT_BYTES,
                     precision::Symbol = RBF_PRECISION)
    precision in (:f64, :f32) ||
        throw(ArgumentError("RBFSettings: precision must be :f64 or :f32, got $precision"))
    return RBFSettings(seed, Float64(max_fit_bytes), precision)
end

"""
    IDWSettings(seed = 0)

`:idw` configuration. `seed` is carried only to fill
`ArmCellResult.model_seed`; InverseDistanceSurrogate is deterministic and has
no fitting step. There are deliberately no model hyperparameters here.
"""
struct IDWSettings
    seed::Int
    p::Float64
end
IDWSettings(; seed::Int = 0, p::Float64 = 2.0) = IDWSettings(seed, p)
# Two-argument positional form keeps the (seed, p) order explicit at call sites.
IDWSettings(seed::Int) = IDWSettings(seed, 2.0)

"""
    rbf_settings(dim::Int) -> RBFSettings

Dim-indexed for symmetry with `ComparatorRegistry.settings_for`; returns the
same fixed configuration for every dimension, which is the fairness argument.
"""
function rbf_settings(dim::Int; max_fit_bytes::Real = DEFAULT_MAX_FIT_BYTES)
    dim >= 1 || throw(ArgumentError("rbf_settings: dim must be >= 1, got $dim"))
    # No n_train: the precision is the same on every cell, so the settings -- and
    # therefore the settings hash -- do not depend on cell size.
    return RBFSettings(0; max_fit_bytes=max_fit_bytes, precision=RBF_PRECISION)
end

"""
    idw_settings(dim::Int) -> IDWSettings

Dim-indexed for symmetry with `ComparatorRegistry.settings_for`; returns the
same fixed configuration for every dimension.
"""
function idw_settings(dim::Int)
    dim >= 1 || throw(ArgumentError("idw_settings: dim must be >= 1, got $dim"))
    return IDWSettings(0, idw_exponent(dim))
end

"""
    idw_exponent(dim::Int) -> Float64

The inverse-distance exponent, `p = dim + 2`. ONE RULE, applied to every cell.

WHY NOT THE PACKAGE DEFAULT. Shepard weighting puts weight `r^-p` on a sample
at distance `r`. For roughly uniform data in `dim` dimensions the number of
samples in a shell at radius `r` grows as `r^(dim-1)`, so the weight MASS from
that shell goes as `r^(dim-1-p)`. At the package default `p = 1` this exponent
is `dim - 2`, i.e. non-decreasing for every dim >= 2: the far field outvotes
the near field and the estimator degenerates to a global weighted mean of all
`y`. Measured on this benchmark at p = 1, the predictions carry only 5-17% of
the truth's variance and land within 3-16% of the trivial "predict the mean"
baseline.

The package docstring's own example uses `p = 2.0`, but that only buys
differentiability at the sample points; it does NOT restore locality above
dim = 1, and measured here it still collapses (variance ratio 0.11-0.39).

`p = dim + 2` makes the shell exponent `-3`: decaying with margin rather than
marginally (decay needs only `p > dim - 1`). It is a DIMENSIONAL RULE, not a
per-function fit -- the same formula on all 36 cells, no cross-validation, no
per-cell search -- so the symmetry that makes the comparison fair is intact.
"""
idw_exponent(dim::Int) = Float64(dim + 2)

"""
    surrogate_model_type(arm::Symbol) -> String

`ArmCellResult.model_type` label: the concrete package type being benchmarked.
"""
function surrogate_model_type(arm::Symbol)
    arm === :rbf && return "Surrogates.RadialBasis"
    arm === :idw && return "Surrogates.InverseDistanceSurrogate"
    throw(ArgumentError("surrogate_model_type: unknown arm $arm (expected one of $SURROGATE_ARMS)"))
end

"""
    surrogate_interval_kind(arm::Symbol) -> Symbol

`:none` for both arms: neither produces an uncertainty interval, so `lower`
and `upper` are NaN on every row. Not a failure -- an absence.
"""
function surrogate_interval_kind(arm::Symbol)
    arm in SURROGATE_ARMS && return :none
    throw(ArgumentError("surrogate_interval_kind: unknown arm $arm (expected one of $SURROGATE_ARMS)"))
end

# Canonical serialisation, byte-identical in form to
# `ComparatorRegistry.canonical_serialize` so the two hash styles match.
_ser(x::Bool) = x ? "true" : "false"
_ser(x::Integer) = string(Int(x))
_ser(x::Float64) = "f64:0x" * string(reinterpret(UInt64, x); base=16, pad=16)
_ser(x::AbstractFloat) = _ser(Float64(x))
_ser(x::AbstractString) = "s:" * repr(String(x))
_ser(x::Symbol) = "sym:" * String(x)

"""
    surrogate_settings_string(s) -> String

Deterministic canonical string for `bytes2hex(sha256(...))`. Includes the
Surrogates.jl version and the defaults provenance, so a package upgrade or a
silently-introduced keyword changes the settings hash rather than passing
unnoticed.

WIRING HAZARD: the version is whatever loaded in THIS process, so a settings
hash computed in a process that lacks Surrogates.jl records
`"unavailable"` and will not match the one computed in the worker that
actually fits. Compute the hash where the fit happens (or assert
`surrogates_available()` before hashing).
"""
function surrogate_settings_string(s::RBFSettings)
    return "SurrogateArms.RBFSettings{" *
           "arm=" * _ser(:rbf) *
           ",seed=" * _ser(s.seed) *
           ",max_fit_bytes=" * _ser(s.max_fit_bytes) *
           ",precision=" * _ser(s.precision) *
           ",pkg_version=" * _ser(_SURROGATES_VERSION) *
           ",defaults=" * _ser(SURROGATES_DEFAULTS_PROVENANCE) * "}"
end

function surrogate_settings_string(s::IDWSettings)
    return "SurrogateArms.IDWSettings{" *
           "arm=" * _ser(:idw) *
           ",seed=" * _ser(s.seed) *
           ",pkg_version=" * _ser(_SURROGATES_VERSION) *
           ",defaults=" * _ser(SURROGATES_DEFAULTS_PROVENANCE) * "}"
end

# --------------------------------------------------------------------------
# Container conversion. The harness is `dim x n` column-major throughout;
# Surrogates.jl wants one container element per sample and REJECTS a matrix.
# One container type -- Vector{Float64} -- is used for both the stored samples
# and the query point, which is also what keeps the v6 IDW coincident-query
# NaN bug unreachable.
# --------------------------------------------------------------------------

function _sample_vectors(X::AbstractMatrix{<:Real}, ::Type{T} = Float64) where {T<:AbstractFloat}
    n = size(X, 2)
    return [T[X[d, i] for d in 1:size(X, 1)] for i in 1:n]
end

_query_vector(x::AbstractVector{<:Real}) = Float64[xi for xi in x]

function _check_fit_inputs(name::AbstractString, X::AbstractMatrix{<:Real},
                           y::AbstractVector{<:Real},
                           lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real})
    _SURROGATES_AVAILABLE || error("$name: Surrogates.jl is not loaded in this process. " *
        "test/ml/Project.toml does not depend on it; run the :rbf/:idw arms under an " *
        "environment that does, or check surrogates_available() before dispatching.")
    d, n = size(X)
    n >= 1 || throw(ArgumentError("$name: no training points"))
    d >= 1 || throw(ArgumentError("$name: zero-dimensional inputs"))
    length(y) == n || throw(DimensionMismatch(
        "$name: X has $n columns but y has $(length(y)) entries"))
    length(lb) == d || throw(DimensionMismatch(
        "$name: X has $d rows but lb has $(length(lb)) entries"))
    length(ub) == d || throw(DimensionMismatch(
        "$name: ub has $(length(ub)) entries, expected $d"))
    all(isfinite, X) || throw(ArgumentError("$name: non-finite training input"))
    all(isfinite, y) || throw(ArgumentError("$name: non-finite training target"))
    return (d, n)
end

"""
    rbf_dense_system_bytes(n::Integer) -> Float64

Estimated peak bytes for `:rbf`'s dense fit at `n` training points: `16*n^2`,
i.e. the `n x n` Float64 interpolation matrix plus the copy `D \\ y`'s
factorisation makes. Used by the memory pre-flight and for reporting; it does
not include the O(n) sample storage, which is negligible beside it.
"""
function rbf_dense_system_bytes(n::Integer, precision::Symbol = :f64)
    per = precision === :f32 ? RBF_BYTES_PER_N2_F32 :
          precision === :f64 ? RBF_BYTES_PER_N2_F64 :
          throw(ArgumentError("rbf_dense_system_bytes: precision must be :f64 or :f32, got $precision"))
    return per * float(n)^2
end

"""
    rbf_precision_for(n, max_fit_bytes = DEFAULT_MAX_FIT_BYTES) -> RBF_PRECISION | nothing

`RBF_PRECISION` when the dense system at `n` training points fits the budget,
`nothing` when it does not and the arm is therefore inapplicable.

`:rbf` runs at ONE precision on every cell. An earlier revision picked Float64
where it fit and Float32 only where it did not, which made the arm two
configurations wearing one name -- asymmetric WITHIN an arm, which is harder to
defend than any difference between arms. It is now uniformly Float32.

The cost of that is nil and was measured: at n = 8,000 Float64 and Float32 give
rel-L2 0.00854 and 0.00854, identical to five decimals, and at n = 125,000
Float32 reaches 0.00093 with the accuracy trend intact (0.051 -> 0.00854 ->
0.00093 as the grid refines), i.e. no sign of conditioning damage. Precision
here is a memory representation, not a tuning knob.

Other arms are NOT Float32, and deliberately so: precision is a numerical
stability requirement of each algorithm, not a comparator configuration.
Clarabel's 1e-6 tolerances in `:diona` and the 1e-6 Cholesky jitter in `:gp`
both sit within an order of magnitude of Float32 eps (1.19e-7) and would stop
meaning anything. What IS standard across arms is the Float64 fixture and the
Float64 reported values.
"""
function rbf_precision_for(n::Integer, max_fit_bytes::Real = DEFAULT_MAX_FIT_BYTES)
    rbf_dense_system_bytes(n, RBF_PRECISION) <= max_fit_bytes && return RBF_PRECISION
    return nothing
end

# Human-readable size for the failure message. `round(GiB; digits=1)` prints
# "0.0 GiB" for every small system, which makes a pre-flight refusal at a low
# limit unreadable.
function _fmt_bytes(bytes::Real)
    isfinite(bytes) || return "unlimited"
    for (unit, scale) in (("GiB", 1024^3), ("MiB", 1024^2), ("KiB", 1024))
        bytes >= scale && return string(round(bytes / scale; digits=2), " ", unit)
    end
    return string(round(bytes; digits=0), " B")
end

# --------------------------------------------------------------------------
# Fitting. NO KEYWORDS ARE PASSED TO EITHER CONSTRUCTOR. See FAIRNESS CONTRACT.
# --------------------------------------------------------------------------

"""
    fit_rbf(X, y, lb, ub, s::RBFSettings)

Fit `Surrogates.RadialBasis` at package defaults on `X` (`dim x n`) and `y`.
Throws on an over-large dense system (see `max_fit_bytes`), which the caller
records as a cell failure. Never subsamples.
"""
function fit_rbf(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
                 lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real},
                 s::RBFSettings)
    _, n = _check_fit_inputs("fit_rbf", X, y, lb, ub)
    T = s.precision === :f32 ? Float32 : Float64
    need = rbf_dense_system_bytes(n, s.precision)
    if need > s.max_fit_bytes
        error("fit_rbf: RadialBasis needs a dense $n x $n system, about " *
              "$(_fmt_bytes(need)) peak, over the " *
              "$(_fmt_bytes(s.max_fit_bytes)) pre-flight limit. " *
              "This cell is recorded as a training failure rather than subsampled " *
              "(dense RBF interpolation not scaling is the result, not an obstacle). " *
              "Raise RBFSettings.max_fit_bytes, or set it to Inf, to attempt it anyway.")
    end
    # Package defaults verbatim (RBF only): rad=linearRadial(), scale_factor=0.5,
    # sparse=false, regularization=0.0. Do not add keywords here.
    # Element type drives the interpolation matrix's type inside Surrogates.
    return Surrogates.RadialBasis(_sample_vectors(X, T), T.(y), T.(lb), T.(ub))
end

"""
    fit_idw(X, y, lb, ub, s::IDWSettings)

Fit `Surrogates.InverseDistanceSurrogate` at `p = dim + 2` (see `idw_exponent`) on
`X` (`dim x n`) and `y`. There is no actual fitting work: the constructor
stores the samples, and all cost is deferred to prediction, O(n) per query.
"""
function fit_idw(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
                 lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real},
                 s::IDWSettings)
    _check_fit_inputs("fit_idw", X, y, lb, ub)
    # p = dim + 2 (see `idw_exponent`), NOT the package default of 1.0, which
    # is degenerate above one dimension. Carried on the settings so it enters
    # the settings hash.
    return Surrogates.InverseDistanceSurrogate(_sample_vectors(X), Float64.(y),
                                               Float64.(lb), Float64.(ub); p = s.p)
end

"""
    surrogate_predict(model, x::AbstractVector{<:Real}) -> Float64

Point prediction at `x`, in whatever frame and target scaling the model was
fitted in. Neither arm produces an interval; see `predict_surrogate_arm` for
the harness-shaped `(pred, lower, upper, status)` form.
"""
surrogate_predict(model, x::AbstractVector{<:Real}) = Float64(model(_query_vector(x)))

# --------------------------------------------------------------------------
# Harness-shaped entry points. Shapes match `BaselineArms.train_baseline` /
# `predict_baseline`, so the facade is a forward and `run_rbf_arm_cell` /
# `run_idw_arm_cell` stay mechanically identical to `run_gp_arm_cell`.
# --------------------------------------------------------------------------

const _TRAIN_FAILED_STATUS = Dict(:rbf => :RBF_TRAIN_FAILED, :idw => :IDW_TRAIN_FAILED)
const _PREDICT_NONFINITE_STATUS = Dict(:rbf => :RBF_PREDICT_NONFINITE,
                                       :idw => :IDW_PREDICT_NONFINITE)

"""
    train_failed_status(arm) -> Symbol
    predict_nonfinite_status(arm) -> Symbol

The two per-row failure statuses for an arm, in the harness's existing
`<ARM>_TRAIN_FAILED` / `<ARM>_PREDICT_NONFINITE` vocabulary.
"""
function train_failed_status(arm::Symbol)
    haskey(_TRAIN_FAILED_STATUS, arm) ||
        throw(ArgumentError("train_failed_status: unknown arm $arm"))
    return _TRAIN_FAILED_STATUS[arm]
end

function predict_nonfinite_status(arm::Symbol)
    haskey(_PREDICT_NONFINITE_STATUS, arm) ||
        throw(ArgumentError("predict_nonfinite_status: unknown arm $arm"))
    return _PREDICT_NONFINITE_STATUS[arm]
end

"""
    train_surrogate_arm_detailed(arm, X_fit, y_fit, s; lb, ub)
        -> (model, fit_time_s, final_loss, train_ok, failure)

As `train_surrogate_arm`, plus `failure`: `nothing` on success, else a string
naming why the cell failed (over-limit dense system, `OutOfMemoryError`,
singular factorisation, ...). Intended for a diagnostics column or a log line;
the recorded per-row status stays `<ARM>_TRAIN_FAILED` either way.

`final_loss` is NaN for both arms: neither minimises a scalar training
objective, the same convention the Diona arm uses.
"""
function train_surrogate_arm_detailed(arm::Symbol, X_fit::AbstractMatrix{<:Real},
                                      y_fit::AbstractVector{<:Real}, s;
                                      lb::AbstractVector{<:Real},
                                      ub::AbstractVector{<:Real})
    arm in SURROGATE_ARMS ||
        throw(ArgumentError("train_surrogate_arm: unknown arm $arm (expected one of $SURROGATE_ARMS)"))
    arm === :rbf && !(s isa RBFSettings) &&
        throw(ArgumentError("train_surrogate_arm(:rbf, ...) needs RBFSettings, got $(typeof(s))"))
    arm === :idw && !(s isa IDWSettings) &&
        throw(ArgumentError("train_surrogate_arm(:idw, ...) needs IDWSettings, got $(typeof(s))"))

    local model
    fit_time = NaN
    try
        fit_time = @elapsed model = arm === :rbf ?
            fit_rbf(X_fit, y_fit, lb, ub, s) :
            fit_idw(X_fit, y_fit, lb, ub, s)
        return (model, fit_time, NaN, true, nothing)
    catch err
        # Every failure path -- pre-flight refusal, OutOfMemoryError, singular
        # system -- becomes one recorded cell failure. Nothing is retried at a
        # reduced size and nothing is silently degraded.
        @warn "$arm training failed (n_train=$(size(X_fit, 2)), dim=$(size(X_fit, 1)))" exception=(err, catch_backtrace())
        reason = err isa OutOfMemoryError ? "OutOfMemoryError" : sprint(showerror, err)
        return (nothing, NaN, NaN, false, reason)
    end
end

"""
    train_surrogate_arm(arm, X_fit, y_fit, s; lb, ub)
        -> (model, fit_time_s, final_loss, train_ok)

Fit one cell. `X_fit` is `dim x n` in the working frame (the comparator
convention is `normalize_inputs_pm1`, so `lb = fill(-1.0, dim)`,
`ub = fill(1.0, dim)`), `y_fit` is the z-scored noisy training target.

On failure returns `(nothing, NaN, NaN, false)`; the caller then writes NaN
predictions with `train_failed_status(arm)` on every row of the cell.
"""
function train_surrogate_arm(arm::Symbol, X_fit::AbstractMatrix{<:Real},
                             y_fit::AbstractVector{<:Real}, s;
                             lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real})
    model, fit_time, final_loss, train_ok, _ =
        train_surrogate_arm_detailed(arm, X_fit, y_fit, s; lb=lb, ub=ub)
    return (model, fit_time, final_loss, train_ok)
end

"""
    predict_surrogate_arm(arm, model, x) -> (pred, lower, upper, status)

One query. `lower` and `upper` are ALWAYS NaN -- neither arm produces an
interval. `status` is `:COMPLETED`, or `<ARM>_PREDICT_NONFINITE` when the
prediction is not finite or the package call threw (both mean "no usable
prediction at this point", and the harness's metrics count only finite rows).

Timing is the caller's, exactly as in `run_gp_arm_cell`, so that the recorded
`elapsed_s` measures the same thing for every arm; `predict_surrogate_arm_timed`
is available where that is inconvenient.
"""
function predict_surrogate_arm(arm::Symbol, model, x::AbstractVector{<:Real})
    status_bad = predict_nonfinite_status(arm)
    model === nothing && return (NaN, NaN, NaN, train_failed_status(arm))
    pred = try
        surrogate_predict(model, x)
    catch err
        @debug "$arm prediction threw; recorded as $status_bad" exception=err
        NaN
    end
    return isfinite(pred) ? (pred, NaN, NaN, :COMPLETED) : (NaN, NaN, NaN, status_bad)
end

"""
    predict_surrogate_arm_timed(arm, model, x)
        -> (pred, lower, upper, status, elapsed_s)

`predict_surrogate_arm` with the per-query wall time attached, for filling
`ArmQueryRow.elapsed_s` directly.
"""
function predict_surrogate_arm_timed(arm::Symbol, model, x::AbstractVector{<:Real})
    t0 = time()
    pred, lower, upper, status = predict_surrogate_arm(arm, model, x)
    return (pred, lower, upper, status, time() - t0)
end

end # module SurrogateArms
