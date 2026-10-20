# Direct Inference on Admissible Sets (manuscript, Section 4 and Algorithm 2).
#
#   1. ε̃: estimate the noise bound if ε̄ is not known a priori (noise.jl).
#   2. 𝒩 ← NeighbourSelection(x', X^⟨n⟩, μ0)           (Algorithm 1, three passes)
#   3. L̃_H ← min L_H  s.t.  Ξ_P [ψ; L_H] ≤ b_P           (Section 4.1)
#   4. 𝒜 = {ψ : Ξ ψ ≤ b} over P = {x'} ∪ 𝒩               (equation 10)
#   5. u̲' ← min_{ψ∈𝒜} u',  ū' ← max_{ψ∈𝒜} u'            (equations 11a–11b)
#   6. return ũ' = (u̲' + ū')/2 and [u̲', ū'],  w = ū' − u̲'
#
# Deviations from the manuscript are listed in docs/algorithm.md.

"""
    DionaTimings

Seconds spent in each stage of [`infer`](@ref): `neighbor_selection`,
`LH_estimation` (zero when `L_H` is supplied), `assembly`, `bounds` (the two
LPs), and `total`.
"""
struct DionaTimings
    neighbor_selection::Float64
    LH_estimation::Float64
    assembly::Float64
    bounds::Float64
    total::Float64
end

"""
    DionaResult

Output of [`infer`](@ref): the estimate `ũ'` (`estimate`), the bounds
`lower = u̲'` and `upper = ū'`, their difference `width = w`, the constant
`L_H_used` the LPs were solved at, the `noise_bound` used, the number of
neighbours `n_neighbors = N`, the neighbourhood `radius = r_𝒩`, a `status`,
`timings`, and `neighbor_diagnostics`.

`status` is `:OPTIMAL` on success; otherwise `:NEIGHBOR_FAILED` (Algorithm 1
found no poised set), `:INFEASIBLE` (𝒜 is empty: the noise and smoothness
constants are inconsistent with the data), `:UNBOUNDED` (𝒜 does not bound
`u'`: too few neighbours), or `:SOLVE_FAILED` (LP time limit or numerical
failure). On failure every scalar field is `nothing`; no partial interval is
exposed.
"""
struct DionaResult
    estimate::Union{Nothing,Float64}
    lower::Union{Nothing,Float64}
    upper::Union{Nothing,Float64}
    width::Union{Nothing,Float64}
    L_H_used::Union{Nothing,Float64}
    noise_bound::Float64
    n_neighbors::Int
    radius::Union{Nothing,Float64}
    status::Symbol
    timings::DionaTimings
    neighbor_diagnostics::Union{Nothing,NeighborDiagnostics}
end

const DIONA_STATUSES = (:OPTIMAL, :NEIGHBOR_FAILED, :INFEASIBLE, :UNBOUNDED, :SOLVE_FAILED)

is_accepted(r::DionaResult) = r.status == :OPTIMAL
has_interval(r::DionaResult) = r.lower !== nothing && r.upper !== nothing

function _failed_result(status::Symbol, noise_bound, n_neighbors, radius, L_H_used, timings, diagnostics)
    return DionaResult(nothing, nothing, nothing, nothing, L_H_used, Float64(noise_bound),
                       n_neighbors, radius, status, timings, diagnostics)
end

"""
    estimate_LH(P; noise_bound) -> (L̃_H, status)

Section 4.1: the smallest Hessian Lipschitz constant consistent with the active
points, `L̃_H = min L_H s.t. Ξ_P [ψ; L_H] ≤ b_P`, with `L_H` appended to the
decision vector and bounded below by zero. `status` is the LP status; `L̃_H`
is `nothing` unless it is `:OPTIMAL`.
"""
function estimate_LH(P::ActiveSet; noise_bound::Real)
    T, ρ = taylor_system(P)
    M, z_stack = measurement_system(P)
    n_psi = size(T, 2)
    # [T  −ρ/s; M  0] [ψ; s L_H] ≤ [0; z + ε̄], with s = max ρ so that the L_H
    # column has entries in (0, 1] (a change of units, not of the optimum).
    s = max(maximum(ρ), eps())
    Ξ_P = hvcat((2, 2), T, sparse(reshape(-ρ ./ s, :, 1)), M, spzeros(size(M, 1), 1))
    b_P = vcat(zeros(length(ρ)), z_stack .+ Float64(noise_bound))
    cost = zeros(n_psi + 1); cost[end] = 1.0
    col_lower = fill(-Inf, n_psi + 1); col_lower[end] = 0.0
    sol = solve_lp(Ξ_P, b_P, cost; col_lower=col_lower, time_limit=LH_LP_TIME_LIMIT_S)
    sol.status == :OPTIMAL || return nothing, sol.status
    return max(sol.x[end] / s, 0.0), :OPTIMAL
end

"""
    admissible_bounds(P; noise_bound, L_H) -> (lower, upper, status)

Equations (11a)–(11b): the minimum and maximum of `u'` over the admissible set
𝒜 assembled by [`admissible_set`](@ref).
"""
function admissible_bounds(P::ActiveSet; noise_bound::Real, L_H::Real)
    Ξ, b = admissible_set(P; noise_bound=noise_bound, L_H=L_H)
    u_index = layout(P).u                       # u' is the first entry of ψ
    return extremal_values(Ξ, b, u_index)
end

"""
    infer(samples, x_query; noise_bound, L_H=:estimate, L_H_factor=1.0,
          method=:poised, layers=nothing, mu0=DEFAULT_MU0,
          max_neighbors=nothing, pool_size=nothing, rank_tol=DEFAULT_RANK_TOL,
          pool_growth=DEFAULT_POOL_GROWTH, param=nothing, verbose=false) -> DionaResult

Algorithm 2 (Diona). `samples` is an `n × (D+1)` matrix whose rows are the
samples `(x_i, z_i)`; `x_query` is `x'`. `noise_bound` is ε̄ (or its estimate
ε̃ from [`estimate_noise_bound`](@ref)). `L_H=:estimate` solves the Section 4.1
LP for `L̃_H` on the selected neighbours; a number is used as given. Either is
multiplied by `L_H_factor` (the manuscript's benchmark uses 2.5) before the
bound LPs are solved. The remaining keywords are passed to
[`select_neighbors`](@ref).

The bounds are valid when `noise_bound` and the constant actually used satisfy
Assumptions 1–2 (Lemma 1). With estimated constants they are the admissible
set under those estimates, not a guarantee.
"""
function infer(samples::AbstractMatrix{<:Real}, x_query::AbstractVector{<:Real};
               noise_bound::Real,
               L_H::Union{Real,Symbol}=:estimate,
               L_H_factor::Real=1.0,
               method::Symbol=:poised,
               layers::Union{Nothing,Int}=nothing,
               mu0::Real=DEFAULT_MU0,
               max_neighbors::Union{Nothing,Int}=nothing,
               pool_size::Union{Nothing,Int}=nothing,
               rank_tol::Float64=DEFAULT_RANK_TOL,
               pool_growth::Int=DEFAULT_POOL_GROWTH,
               param::Union{Nothing,Int}=nothing,
               verbose::Bool=false)
    t_start = time()
    D = size(samples, 2) - 1
    D >= 1 || throw(ArgumentError("samples must have at least two columns: coordinates then z"))
    length(x_query) == D ||
        throw(DimensionMismatch("query has $(length(x_query)) coordinates, samples have $D"))
    noise_bound >= 0 || throw(ArgumentError("noise_bound must be nonnegative, got $noise_bound"))
    L_H_factor > 0 || throw(ArgumentError("L_H_factor must be positive, got $L_H_factor"))
    L_H isa Symbol && L_H != :estimate && throw(ArgumentError("L_H must be a number or :estimate"))
    L_H isa Real && L_H < 0 && throw(ArgumentError("L_H must be nonnegative, got $L_H"))

    samples_f = samples isa Matrix{Float64} ? samples : Matrix{Float64}(samples)
    X = @view samples_f[:, 1:D]              # leading columns are contiguous: no copy
    xq = Vector{Float64}(x_query)
    supplied_LH = L_H isa Real ? Float64(L_H) * Float64(L_H_factor) : nothing

    # Step 2: neighbour selection (Algorithm 1, `layers` passes)
    t0 = time()
    selection = try
        select_neighbors(X, xq; method=method, layers=layers, mu0=mu0,
                         max_neighbors=max_neighbors, pool_size=pool_size, rank_tol=rank_tol,
                         pool_growth=pool_growth, param=param, return_diagnostics=true)
    catch err
        err isa NeighborSelectionError || rethrow()
        t_sel = time() - t0
        timings = DionaTimings(t_sel, 0.0, 0.0, 0.0, time() - t_start)
        return _failed_result(:NEIGHBOR_FAILED, noise_bound, 0, nothing, supplied_LH, timings,
                              err.diagnostics)
    end
    t_sel = time() - t0
    diagnostics = NeighborDiagnostics(selection.diagnostics.method, selection.diagnostics.mu0,
                                      selection.diagnostics.mu, selection.diagnostics.radius,
                                      selection.diagnostics.n_selected, selection.diagnostics.pool_size,
                                      selection.diagnostics.pool_expansions, selection.indices)
    P = ActiveSet(samples_f, xq, selection.indices)
    N = n_neighbors(P)
    r_N = radius(P)
    verbose && println("  neighbours: N=$N, r_N=$(round(r_N, sigdigits=4)), " *
                       "μ=$(diagnostics.mu === nothing ? "-" : round(diagnostics.mu, sigdigits=4))")

    # Step 3: L̃_H (Section 4.1)
    t0 = time()
    if L_H === :estimate
        LH_hat, st = estimate_LH(P; noise_bound=noise_bound)
        t_lh = time() - t0
        if st != :OPTIMAL
            timings = DionaTimings(t_sel, t_lh, 0.0, 0.0, time() - t_start)
            return _failed_result(st, noise_bound, N, r_N, nothing, timings, diagnostics)
        end
        L_H_used = LH_hat * Float64(L_H_factor)
        verbose && println("  L̃_H=$(round(LH_hat, sigdigits=4)) × $(L_H_factor) → $(round(L_H_used, sigdigits=4))")
    else
        t_lh = 0.0
        L_H_used = supplied_LH
    end

    # Steps 4–5: admissible set and bounds (equations 10, 11a–11b)
    t0 = time()
    Ξ, b = admissible_set(P; noise_bound=noise_bound, L_H=L_H_used)
    t_asm = time() - t0
    t0 = time()
    lower, upper, st = extremal_values(Ξ, b, layout(P).u)
    t_bounds = time() - t0
    timings = DionaTimings(t_sel, t_lh, t_asm, t_bounds, time() - t_start)
    st == :OPTIMAL || return _failed_result(st, noise_bound, N, r_N, L_H_used, timings, diagnostics)

    # Step 6
    return DionaResult((lower + upper) / 2, lower, upper, upper - lower, L_H_used,
                       Float64(noise_bound), N, r_N, :OPTIMAL, timings, diagnostics)
end
