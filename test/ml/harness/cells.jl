# Benchmark cells: enumeration, data loading, preparation, the Diona cell solve.
#
# Extracted verbatim from test/smoke_data.jl. This is a plain `include`d FRAGMENT,
# not a module: every name must stay top-level in the including scope, because
# test/run_full_benchmark.jl reaches into the harness via getproperty(harness, :name)
# and test/benchmark_data.jl mutates _SMOKE_OPTS directly. The include sits at the
# exact position the code occupied, so evaluation order is unchanged.
#
# NOTE: SCRIPT_PATH stays in smoke_data.jl -- it is abspath(@__FILE__) and is the
# target workers re-invoke, so it must resolve to the entry point, not to this file.

struct SmokeCell
    name::Symbol           # benchmark function, e.g. :ishigami
    dim::Int               # input dimension
    layout::Symbol         # :sobol or :grid
    tier_label::String     # human-readable tier (e.g. "500D", "10³")
    n_train::Int           # number of training points at this tier
    noise_level::Float64   # amplitude ratio 10^(-SNR/20): 0.0 clean, 0.1 = 20 dB
end
# Amplitude ratios: 0.0 is noiseless, 0.1 == snr_db_to_level(20) is 20 dB.
const NOISE_LEVELS = (0.0, 0.1)

# CELL IDS ARE POSITIONS IN THIS LIST AND ARE FROZEN. A fragment's filename
# carries its cell id, so renumbering orphans every fragment of every previous
# campaign. The emission order is therefore: the five ORIGINAL Sobol' functions
# (ids 1-30), then the Ishigami grid block (31-36), then any function added
# later, appended (37+). Appending a new function to BENCHMARK_FUNCTIONS alone
# would have pushed the grid block to 37-42 and orphaned 36 fragments,
# including the two n=125,000 :rbf cells that cost ~47 minutes each.
function _sobol_cells_for(bf)::Vector{SmokeCell}
    out = SmokeCell[]
    for mult in N_TIERS_PER_DIM
        n = mult * bf.dim
        for nl in NOISE_LEVELS
            push!(out, SmokeCell(bf.name, bf.dim, :sobol, "$(mult)D", n, nl))
        end
    end
    return out
end

function enumerate_cells()::Vector{SmokeCell}
    cells = SmokeCell[]
    for bf in _ORIGINAL_FUNCTIONS
        append!(cells, _sobol_cells_for(bf))
    end
    for k in ISHIGAMI_GRID_SIDES
        for nl in NOISE_LEVELS
            push!(cells, SmokeCell(:ishigami, 3, :grid,
                                   "$(k)³", k^3, nl))
        end
    end
    orig = Set(bf.name for bf in _ORIGINAL_FUNCTIONS)
    for bf in BENCHMARK_FUNCTIONS
        bf.name in orig && continue
        append!(cells, _sobol_cells_for(bf))
    end

    return cells
end

const _BENCH_FN = Dict{Symbol,Any}(bf.name => bf for bf in BENCHMARK_FUNCTIONS)

function load_training(cell::SmokeCell)
    bf = _BENCH_FN[cell.name]
    if cell.layout == :sobol
        return sample_training_points(bf, cell.n_train)
    else  # :grid — Ishigami only; recover k from n_train
        k = only(k for k in ISHIGAMI_GRID_SIDES if k^3 == cell.n_train)
        return ishigami_grid_points(k)
    end
end


const _QUERY_SET_CACHE = Dict{Tuple{Symbol,Int},Tuple{Matrix{Float64},Vector{Float64},Float64}}()

function query_set_for(cell::SmokeCell)
    return get!(_QUERY_SET_CACHE, (cell.name, QUERY_MULTIPLIER[])) do
        bf = _BENCH_FN[cell.name]
        X_query, y_truth = sample_query_set(bf)
        or_val = output_range(bf)
        (X_query, y_truth, or_val)
    end
end


struct CellResult
    cell::SmokeCell
    pred::Vector{Float64}      # nominal predictions (NaN if unavailable)
    lower::Vector{Float64}     # lower UQ bounds (NaN if unavailable)
    upper::Vector{Float64}     # upper UQ bounds (NaN if unavailable)
    status::Vector{Symbol}     # per-query status codes
    elapsed::Vector{Float64}   # per-query wall time (seconds)
    noise_bound_used::Float64  # estimated ε̃ passed to Diona.infer
    y_truth::Vector{Float64}   # noise-free truth at held-out query points
    n_query::Int               # number of held-out query points
    n_train_actual::Int        # ACTUAL loaded training row count (PRD §7.3:
    t_neighbor::Vector{Float64}    # Algorithm 1, three passes (KD-tree candidates, QR, SVD)
    t_lh::Vector{Float64}          # the L̃_H LP of manuscript §4.1 (HiGHS)
    t_assembly::Vector{Float64}    # assembling Ξ ψ ≤ b
    t_bounds::Vector{Float64}      # the two bound LPs (11a)–(11b) (HiGHS)
end
const _SINGLE_CELL_ANCHOR_NAME    = :ishigami
const _SINGLE_CELL_ANCHOR_LAYOUT  = :sobol
const _SINGLE_CELL_ANCHOR_NTRAIN  = 1500
const _SINGLE_CELL_ANCHOR_NOISE   = 0.1
const _SMOKE_NOISE_SEED = 42

const PREPARED_SMOKE_CELL_SCHEMA_VERSION = "3.0.0"

struct PreparedSmokeCell
    cell::SmokeCell                 # cell descriptor (identity)
    cell_id::Int                    # 1-based index into enumerate_cells(); 0 if unresolved
    X_train::Matrix{Float64}        # dim × n_train_actual
    y_train_clean::Vector{Float64}  # noise-free training targets
    y_train_noisy::Vector{Float64}  # noisy training targets (seed noise_seed)
    noise_level::Float64            # amplitude ratio (== cell.noise_level); 0.1 = 20 dB
    # PROTOCOL: ESTIMATED from the noisy training data by estimate_noise_bound,
    # never the injected sigma (noise_level * noise_sigma). prepare_smoke_cell
    # raises rather than substituting that oracle; see its body.
    noise_bound::Float64            # estimated noise half-width (src/noise.jl)
    noise_seed::Int                 # RNG seed used for the noise draw (_SMOKE_NOISE_SEED, 42)
    output_range::Float64           # max-min; denominator for RELATIVE WIDTHS, not noise
    query_ids::Vector{Int}          # ordered 1:n_query ids
    X_query::Matrix{Float64}        # dim × n_query dense held-out inputs
    y_query_truth::Vector{Float64}  # noise-free truth at held-out query points — EVAL-ONLY
    n_train_actual::Int             # actual loaded training row count (PRD §7.3)
    n_query::Int                    # number of held-out query points
    schema_version::String          # PREPARED_SMOKE_CELL_SCHEMA_VERSION at generation time
    generated_at::String            # ISO-8601 timestamp (provenance)
end

function prepare_smoke_cell(cell::SmokeCell; cell_id::Union{Int,Nothing}=nothing)::PreparedSmokeCell
    bf = _BENCH_FN[cell.name]
    X_train = load_training(cell)
    y_true_train = [bf.f(view(X_train, :, i)) for i in 1:cell.n_train]
    n_train_actual = size(X_train, 2)
    or_val     = output_range(bf)            # reporting scale for relative widths
    # ZERO-MEAN GAUSSIAN noise at a fixed SNR: sigma = level * sigma_f, with
    # level = 10^(-SNR_dB/20). NOT a fraction of output_range (max-min), which
    # is an unstable estimator here and does not hold SNR fixed across
    # functions -- see `noise_sigma` in driver.jl.
    sigma      = cell.noise_level * noise_sigma(bf)   # noise DRAW sigma -- never a bound
    y_noisy    = y_true_train .+ draw_noise(_SMOKE_NOISE_SEED, n_train_actual, sigma)
    # COORDINATES: estimate on the SAME normalized box every arm's model sees
    # (normalize_inputs_pm1, as at the diona solve below and in arms/nsm.jl:43).
    # estimate_noise_bound grows its ball radius as a FRACTION OF THE EUCLIDEAN
    # DOMAIN DIAMETER, so rho/diameter is scale-invariant on an isotropic box:
    # this is a byte-for-byte no-op for every [a,b]^d benchmark here (verified on
    # ishigami/park91a/friedman1/hartmann6/rosenbrock, all digits). On an
    # ANISOTROPIC box it is not. Piston spans [0.005,0.02] in S and [9e4,1.1e5]
    # in P0 -- a 2.5e6 ratio -- so the diameter is entirely P0 and the starting
    # rho = 0.01*diam ~ 204 is wider than the WHOLE DOMAIN in 5 of 7 dimensions.
    # The "ball" is then a slab: every point has neighbours (ballfrac 1.0), mean
    # occupancy grows with n (5.6 -> 11.1 -> 22.2), and the measured max-min is
    # the full swing of f over those 5 free dimensions, not a local one. That put
    # eps at 19-27% of output_range and made it GROW with n, which pinned L_H at
    # a floor and inflated the relative interval past 1.0
    # (1.04/1.28/1.46 -> 0.072/0.057/0.061 once normalized, coverage unchanged).
    _norm_train = BenchmarkDriver.normalize_inputs_pm1(bf, X_train)
    _noisy_samples = hcat(permutedims(_norm_train), y_noisy)
    _estimated_nb = estimate_noise_bound(
        _noisy_samples; x_cols=1:size(_norm_train, 1), z_col=size(_norm_train, 1) + 1)
    # PROTOCOL: the noise bound is ESTIMATED from the training data and nothing
    # else. There is deliberately NO fallback to the injected `sigma` here -- that is the
    # injected noise half-width, i.e. the ground truth the estimator exists to
    # approximate, and substituting it would hand the solver an oracle. A
    # non-finite estimate means estimate_noise_bound found no valid ball set
    # (src/noise.jl); a non-positive one turns the data band into a set of
    # equalities and is indistinguishable in the artifacts from a genuinely tight
    # estimate. Both are operator-visible failures, not defaults.
    isfinite(_estimated_nb) || error(
        "prepare_smoke_cell: estimate_noise_bound returned $(_estimated_nb) for cell " *
        "$(cell.name)/$(cell.layout)/$(cell.tier_label)/noise_level=$(cell.noise_level) " *
        "(n_train=$(n_train_actual)) -- the estimator found no valid ball set. Raise " *
        "min_ball_points / min_ball_coverage or widen the radius schedule. The injected " *
        "noise half-width is deliberately NOT substituted: that would be an oracle.")
    _estimated_nb > 0 || error(
        "prepare_smoke_cell: estimate_noise_bound returned the non-positive bound " *
        "$(_estimated_nb) for cell $(cell.name)/$(cell.layout)/$(cell.tier_label)/" *
        "noise_level=$(cell.noise_level) (n_train=$(n_train_actual)). A zero noise bound " *
        "makes the data band a set of equalities and is a silent source of " *
        ":INFEASIBLE; it is refused rather than passed on.")
    noise_bound = _estimated_nb
    X_query, y_truth, _ = query_set_for(cell)
    n_q = size(X_query, 2)

    resolved_id = cell_id === nothing ?
        something(findfirst(==(cell), enumerate_cells()), 0) : cell_id

    return PreparedSmokeCell(
        cell, resolved_id,
        X_train, y_true_train, y_noisy,
        cell.noise_level, noise_bound, _SMOKE_NOISE_SEED, or_val,
        collect(1:n_q), X_query, y_truth,
        n_train_actual, n_q,
        PREPARED_SMOKE_CELL_SCHEMA_VERSION, string(Dates.now()),
    )
end
# Diona's local-smoothness safety multiplier. A Ref rather than a const because
# it is a per-campaign knob (solver.L_H_factor), set once before any cell runs.
# It ENTERS the :diona settings hash, so two campaigns at different factors key
# to different fragments and can share one out dir without colliding.
const _SMOKE_L_H_FACTOR_DEFAULT = 2.5
const _SMOKE_L_H_FACTOR = Ref(_SMOKE_L_H_FACTOR_DEFAULT)

# Diona's poisedness threshold μ0 (manuscript, Definition 1 / Assumption 3),
# fixed for every cell and every dimension. Assumption 3 only requires
# 0 < μ0 < μ̄_D (the E-optimal ceiling), and a small value is the intended
# operating point: the selected set must span every quadratic coefficient
# (full rank), not approach the optimal design. It ENTERS the :diona settings
# hash, as do the three passes of Algorithm 1 and the LP solver.
const _SMOKE_DIONA_MU0 = 1.0e-6
const _SMOKE_DIONA_LAYERS = Diona.DEFAULT_LAYERS
_diona_settings_string() =
    "L_H=:estimate,L_H_factor=$(_SMOKE_L_H_FACTOR[]),mu0=$(_SMOKE_DIONA_MU0)," *
    "layers=$(_SMOKE_DIONA_LAYERS),solver=HiGHS"

"""
    set_lh_factor!(v::Real)

Set Diona's `L_H_factor` for this process. Must be called before any cell is
solved or any settings hash is computed. Raises on a non-positive value.
"""
function set_lh_factor!(v::Real)
    v > 0 || throw(ArgumentError("set_lh_factor!: L_H_factor must be > 0, got $v"))
    _SMOKE_L_H_FACTOR[] = Float64(v)
end
const _SINGLE_CELL_REF_REL_L2   = 0.1711
const _SINGLE_CELL_REF_COVERAGE = 1.000

"""
    solve_cell_diona(prep; verbose=false, lh_used=nothing) -> CellResult

`lh_used`, when given, must be a length-`prep.n_query` vector; it is filled with
`DionaResult.L_H_used` per query, i.e. `L̃_H × L_H_factor` (NaN where the solver
returned none, e.g. the `:NEIGHBOR_FAILED` early return). It is an OUT-PARAMETER rather than a new
`CellResult` field on purpose: the reporting-correctness gate asserts
`fieldnames(CellResult)` verbatim as the legacy Diona-only path, and widening
that tuple would weaken the check to make room for this one.
"""
function solve_cell_diona(prep::PreparedSmokeCell; verbose::Bool=false,
                          lh_used::Union{Nothing,Vector{Float64}}=nothing)::CellResult
    cell = prep.cell
    bf   = _BENCH_FN[cell.name]
    # INPUTS ARE NORMALISED TO [-1,1]^d, exactly as every other arm does via
    # BenchmarkDriver.normalize_inputs_pm1. :diona used to receive RAW
    # coordinates, which was invisible while every benchmark function had a
    # near-isotropic domain box and fatal as soon as one did not: on :piston,
    # whose box spans 0.008 in V0 against 20000 in P0, Euclidean distance is
    # dominated by two of the seven inputs, the neighbour stencil collapses
    # onto a degenerate subspace, and EVERY query returned :NEIGHBOR_FAILED
    # (measured: 0 of 40 OPTIMAL raw, 40 of 40 normalised).
    #
    # The box is experimental-design information, identical for all arms and
    # carrying nothing about the response, so normalising here does not leak
    # anything -- it removes an inconsistency in which one arm saw a different
    # geometry from the other five.
    X_train     = BenchmarkDriver.normalize_inputs_pm1(bf, prep.X_train)
    y_noisy     = prep.y_train_noisy
    noise_bound = prep.noise_bound
    samples_mat = hcat(permutedims(X_train), y_noisy)  # n_train × (dim+1)

    X_query = BenchmarkDriver.normalize_inputs_pm1(bf, prep.X_query)
    n_q     = prep.n_query
    build_kdtree_cache!(permutedims(X_train))   # n_train × dim
    pred    = fill(NaN, n_q)
    lo      = fill(NaN, n_q)
    hi      = fill(NaN, n_q)
    status  = Vector{Symbol}(undef, n_q)
    elapsed = zeros(n_q)
    t_neighbor = zeros(n_q)
    t_lh       = zeros(n_q)
    t_assembly = zeros(n_q)
    t_bounds   = zeros(n_q)
    lh_used === nothing || length(lh_used) == n_q ||
        throw(DimensionMismatch("solve_cell_diona: lh_used has $(length(lh_used)) entries, need $n_q"))

    for i in 1:n_q
        x_query = X_query[:, i]
        t0 = time()
        res = infer(samples_mat, x_query;
                    noise_bound = noise_bound,
                    L_H         = :estimate,
                    L_H_factor  = _SMOKE_L_H_FACTOR[],
                    mu0         = _SMOKE_DIONA_MU0,
                    layers      = _SMOKE_DIONA_LAYERS,
                    verbose     = verbose)
        elapsed[i] = time() - t0
        status[i]  = res.status
        lh_used === nothing ||
            (lh_used[i] = res.L_H_used === nothing ? NaN : Float64(res.L_H_used))
        res.estimate !== nothing && (pred[i] = res.estimate)
        if res.lower !== nothing && res.upper !== nothing
            lo[i] = res.lower
            hi[i] = res.upper
        end
        t_neighbor[i] = res.timings.neighbor_selection
        t_lh[i]       = res.timings.LH_estimation
        t_assembly[i] = res.timings.assembly
        t_bounds[i]   = res.timings.bounds
    end

    return CellResult(cell, pred, lo, hi, status, elapsed, noise_bound,
                       prep.y_query_truth, n_q, prep.n_train_actual,
                       t_neighbor, t_lh, t_assembly, t_bounds)
end

function run_cell(cell::SmokeCell; verbose::Bool=false)::CellResult
    prep = prepare_smoke_cell(cell)
    return solve_cell_diona(prep; verbose=verbose)
end

function _cell_rel_l2(cr::CellResult)
    mask = isfinite.(cr.pred)
    any(mask) || return NaN
    denom = norm(cr.y_truth[mask])
    denom == 0.0 && return NaN
    return norm(cr.pred[mask] .- cr.y_truth[mask]) / denom
end

function _cell_coverage(cr::CellResult)
    mask = isfinite.(cr.lower) .& isfinite.(cr.upper)
    count(mask) == 0 && return NaN
    n_covered = sum((cr.y_truth[mask] .>= cr.lower[mask]) .&
                    (cr.y_truth[mask] .<= cr.upper[mask]))
    return n_covered / count(mask)
end

function _cell_status_counts(cr::CellResult)
    d = Dict{Symbol,Int}()
    for s in cr.status
        d[s] = get(d, s, 0) + 1
    end
    return d
end


function compute_cell_metrics(cr::CellResult)
    uq_mask    = isfinite.(cr.lower) .& isfinite.(cr.upper)
    mean_width = count(uq_mask) > 0 ? mean(cr.upper[uq_mask] .- cr.lower[uq_mask]) : NaN
    return (
        rel_l2          = _cell_rel_l2(cr),
        coverage        = _cell_coverage(cr),
        mean_width      = mean_width,
        width_over_2eps = isfinite(mean_width) ? mean_width / (2 * cr.noise_bound_used) : NaN,
        t_mean          = mean(cr.elapsed),
        t_median        = median(cr.elapsed),
        t_max           = maximum(cr.elapsed),
        status_counts   = _cell_status_counts(cr),
        t_neighbor_mean     = mean(cr.t_neighbor),
        t_neighbor_median   = median(cr.t_neighbor),
        t_neighbor_max      = maximum(cr.t_neighbor),
        t_lh_mean           = mean(cr.t_lh),
        t_lh_median         = median(cr.t_lh),
        t_lh_max            = maximum(cr.t_lh),
        t_assembly_mean     = mean(cr.t_assembly),
        t_assembly_median   = median(cr.t_assembly),
        t_assembly_max      = maximum(cr.t_assembly),
        t_bounds_mean       = mean(cr.t_bounds),
        t_bounds_median     = median(cr.t_bounds),
        t_bounds_max        = maximum(cr.t_bounds),
    )
end
