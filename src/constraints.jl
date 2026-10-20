# The admissible set of state triples.
#
# Manuscript, Section 2.2, equations (8a)–(8e). For the query x' and each
# neighbour x_i with displacement a_i = x' − x_i, the Taylor relations between
# the two triples θ' and θ_i, together with the measurement band, are
#
#   (8a)  |u' − (u_i + a_iᵀ g_i + λ_iᵀ h_i)|        ≤ L_H/6 ‖a_i‖³
#   (8b)  |u_i − (u' − a_iᵀ g' + λ_iᵀ h')|          ≤ L_H/6 ‖a_i‖³
#   (8c)  ‖g' − (g_i + C_iᵀ (h' + h_i))‖_∞          ≤ L_H/4 ‖a_i‖²
#   (8d)  ‖h' − h_i‖_∞                             ≤ L_H ‖a_i‖
#   (8e)  |u_i − z_i|                              ≤ ε̄
#
# with λ_i = λ(a_i) and C_i = C(a_i) from triple.jl. Stacking the unknowns as
# ψ = col(θ', θ_{i_1}, …, θ_{i_N}) (eq. 9) every row is affine in ψ, and the
# admissible set is the polytope 𝒜 = {ψ : Ξ ψ ≤ b} (eq. 10).

"""
    ActiveSet(samples, x_query, indices)

The active points P = {x'} ∪ 𝒩 of one query: the query point and the selected
neighbour samples (rows `indices` of `samples`, whose columns are the `D`
coordinates followed by the measurement `z`).
"""
struct ActiveSet
    x_query::Vector{Float64}
    X::Matrix{Float64}          # N × D neighbour coordinates
    z::Vector{Float64}          # N measurements z_i
    indices::Vector{Int}        # rows of `samples`
end

function ActiveSet(samples::AbstractMatrix{<:Real}, x_query::AbstractVector{<:Real},
                   indices::AbstractVector{<:Integer})
    D = size(samples, 2) - 1
    length(x_query) == D ||
        throw(DimensionMismatch("query has $(length(x_query)) coordinates, samples have $D"))
    isempty(indices) && throw(ArgumentError("an active set needs at least one neighbour"))
    X = Matrix{Float64}(samples[indices, 1:D])
    z = Vector{Float64}(samples[indices, D + 1])
    return ActiveSet(Vector{Float64}(x_query), X, z, Vector{Int}(indices))
end

n_neighbors(P::ActiveSet) = length(P.z)
dimension(P::ActiveSet) = length(P.x_query)
layout(P::ActiveSet) = TripleLayout(dimension(P))

"""Neighbourhood radius r_𝒩 = max_i ‖x' − x_i‖ (Lemma 2)."""
radius(P::ActiveSet) = maximum(norm(P.x_query .- @view(P.X[i, :])) for i in 1:n_neighbors(P))

"""Number of Taylor rows (8a)–(8d) per neighbour: 4 + 2D + 2D_h."""
taylor_rows_per_neighbor(L::TripleLayout) = 4 + 2 * L.D + 2 * L.D_h

"""
    taylor_pair_rows!(Q, S, ρ, a, L)

Fill the Taylor rows (8a)–(8d) for one neighbour with displacement `a = x' − x_i`:

    Q θ' + S θ_i ≤ L_H ρ

`Q` and `S` are `m × n` dense blocks (`m = taylor_rows_per_neighbor(L)`,
`n = L.n`) acting on the query triple and the neighbour triple; `ρ` holds the
remainder coefficients `‖a‖³/6`, `‖a‖²/4`, `‖a‖` that multiply `L_H`.
Each absolute-value or ∞-norm inequality contributes two rows (+ and −).
"""
function taylor_pair_rows!(Q::AbstractMatrix{Float64}, S::AbstractMatrix{Float64},
                           ρ::AbstractVector{Float64}, a::AbstractVector{<:Real},
                           L::TripleLayout)
    m = taylor_rows_per_neighbor(L)
    size(Q) == (m, L.n) && size(S) == (m, L.n) && length(ρ) == m ||
        throw(DimensionMismatch("taylor_pair_rows!: block sizes do not match the layout"))
    length(a) == L.D || throw(DimensionMismatch("displacement has the wrong length"))
    fill!(Q, 0.0); fill!(S, 0.0)

    r = norm(a)
    λ = lambda_coeffs(a)
    C = c_coeffs(a)

    # (8a): u' − u_i − aᵀ g_i − λᵀ h_i  ∈ ±L_H r³/6
    Q[1, L.u] = 1.0;  S[1, L.u] = -1.0
    S[1, L.g] .= -a;  S[1, L.h] .= -λ
    Q[2, :] .= -Q[1, :];  S[2, :] .= -S[1, :]
    ρ[1] = ρ[2] = r^3 / 6

    # (8b): u_i − u' + aᵀ g' − λᵀ h'  ∈ ±L_H r³/6
    S[3, L.u] = 1.0;  Q[3, L.u] = -1.0
    Q[3, L.g] .= a;   Q[3, L.h] .= -λ
    Q[4, :] .= -Q[3, :];  S[4, :] .= -S[3, :]
    ρ[3] = ρ[4] = r^3 / 6

    # (8c): g'_k − g_{i,k} − C[:,k]ᵀ (h' + h_i)  ∈ ±L_H r²/4,  k = 1..D
    row = 5
    @inbounds for k in 1:L.D
        Q[row, L.g[k]] = 1.0;  S[row, L.g[k]] = -1.0
        Q[row, L.h] .= -@view(C[:, k]);  S[row, L.h] .= -@view(C[:, k])
        Q[row + 1, :] .= -Q[row, :];  S[row + 1, :] .= -S[row, :]
        ρ[row] = ρ[row + 1] = r^2 / 4
        row += 2
    end

    # (8d): h'_p − h_{i,p}  ∈ ±L_H r,  p = 1..D_h
    @inbounds for p in 1:L.D_h
        Q[row, L.h[p]] = 1.0;  S[row, L.h[p]] = -1.0
        Q[row + 1, L.h[p]] = -1.0;  S[row + 1, L.h[p]] = 1.0
        ρ[row] = ρ[row + 1] = r
        row += 2
    end
    return Q, S, ρ
end

"""
    taylor_pair_rows(a, L) -> (Q, S, ρ)

Allocating form of [`taylor_pair_rows!`](@ref).
"""
function taylor_pair_rows(a::AbstractVector{<:Real}, L::TripleLayout)
    m = taylor_rows_per_neighbor(L)
    return taylor_pair_rows!(zeros(m, L.n), zeros(m, L.n), zeros(m), a, L)
end

"""Column range of θ_k in ψ: k = 0 is the query, k = 1..N the neighbours."""
triple_columns(L::TripleLayout, k::Integer) = (k * L.n + 1):((k + 1) * L.n)

function _emit_block!(I, J, V, row0::Int, cols::UnitRange{Int}, B::AbstractMatrix{Float64})
    @inbounds for c in axes(B, 2), r in axes(B, 1)
        v = B[r, c]
        v == 0.0 && continue
        push!(I, row0 + r); push!(J, cols[c]); push!(V, v)
    end
end

"""
    taylor_system(P) -> (T, ρ)

Stack the Taylor rows (8a)–(8d) of every neighbour of `P`, in neighbour order:
`T ψ ≤ L_H ρ`, with `T` sparse of size `(N m) × ((1+N) n)`.
"""
function taylor_system(P::ActiveSet)
    L = layout(P)
    N = n_neighbors(P)
    m = taylor_rows_per_neighbor(L)
    n_rows = N * m
    n_cols = (1 + N) * L.n
    I = Int[]; J = Int[]; V = Float64[]
    sizehint!(I, 2 * n_rows * L.n); sizehint!(J, 2 * n_rows * L.n); sizehint!(V, 2 * n_rows * L.n)
    ρ = Vector{Float64}(undef, n_rows)
    Q = zeros(m, L.n); S = zeros(m, L.n); ρ_i = zeros(m)
    query_cols = triple_columns(L, 0)
    for i in 1:N
        a = P.x_query .- @view(P.X[i, :])
        taylor_pair_rows!(Q, S, ρ_i, a, L)
        row0 = (i - 1) * m
        _emit_block!(I, J, V, row0, query_cols, Q)
        _emit_block!(I, J, V, row0, triple_columns(L, i), S)
        ρ[row0 + 1:row0 + m] .= ρ_i
    end
    return sparse(I, J, V, n_rows, n_cols), ρ
end

"""
    measurement_system(P) -> (M, z_stack)

The measurement rows (8e) of every neighbour, as `M ψ ≤ z_stack + ε̄ 1` where
`M` has the rows `+u_i` and `−u_i` and `z_stack = [z_i; −z_i]` for each
neighbour, in neighbour order.
"""
function measurement_system(P::ActiveSet)
    L = layout(P)
    N = n_neighbors(P)
    n_cols = (1 + N) * L.n
    I = Vector{Int}(undef, 2N); J = similar(I); V = Vector{Float64}(undef, 2N)
    z_stack = Vector{Float64}(undef, 2N)
    for i in 1:N
        u_col = first(triple_columns(L, i)) + L.u - 1
        I[2i - 1] = 2i - 1; J[2i - 1] = u_col; V[2i - 1] = 1.0;  z_stack[2i - 1] = P.z[i]
        I[2i]     = 2i;     J[2i]     = u_col; V[2i]     = -1.0; z_stack[2i]     = -P.z[i]
    end
    return sparse(I, J, V, 2N, n_cols), z_stack
end

"""
    admissible_set(P; noise_bound, L_H) -> (Ξ, b)

The polytope 𝒜 = {ψ : Ξ ψ ≤ b} of equation (10): Taylor rows (8a)–(8d) at the
Hessian Lipschitz constant `L_H`, then measurement rows (8e) with the noise
bound ε̄ = `noise_bound`.
"""
function admissible_set(P::ActiveSet; noise_bound::Real, L_H::Real)
    noise_bound >= 0 || throw(ArgumentError("noise_bound must be nonnegative"))
    L_H >= 0 || throw(ArgumentError("L_H must be nonnegative"))
    T, ρ = taylor_system(P)
    M, z_stack = measurement_system(P)
    Ξ = vcat(T, M)
    b = vcat(Float64(L_H) .* ρ, z_stack .+ Float64(noise_bound))
    return Ξ, b
end
