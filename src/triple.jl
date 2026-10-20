# State triples: the latent unknowns of Diona.
#
# Manuscript, "Notation" and Section 2.2. At any point x the *state triple* is
#
#     θ = col(u, g, h) ∈ R^{1 + D + D_h},    u = u(x),  g = ∇u(x),  h = vech(∇²u(x)),
#
# with D_h = D(D+1)/2. The Hessian enters only through its symmetric
# half-vectorisation h = vech(H), which stacks the lower triangle column by
# column: (1,1), (2,1), …, (D,1), (2,2), (3,2), …, (D,D). With the duplication
# matrix D_D (vec(H) = D_D vech(H)) the two coefficient maps of Section 2.2 are
#
#     λ(a) = ½ D_Dᵀ (a ⊗ a)      so that  λ(a)ᵀ h = ½ aᵀ H a,
#     C(a) = ½ D_Dᵀ (a ⊗ I_D)    so that  C(a)ᵀ h = ½ H a.
#
# Everything downstream (feature matrix Φ, the Taylor rows (8a)–(8d), the LPs)
# is written in terms of these three objects.

"""
    TripleLayout(D)

Index layout of one state triple θ = col(u, g, h) ∈ R^n, n = 1 + D + D_h.

Fields: `D`, `D_h = D(D+1)/2`, `n`, and the index ranges `u` (an `Int`),
`g::UnitRange`, `h::UnitRange` into θ.
"""
struct TripleLayout
    D::Int
    D_h::Int
    n::Int
    u::Int
    g::UnitRange{Int}
    h::UnitRange{Int}
end

function TripleLayout(D::Integer)
    D >= 1 || throw(ArgumentError("D must be at least 1, got $D"))
    D_h = div(D * (D + 1), 2)
    n = 1 + D + D_h
    return TripleLayout(D, D_h, n, 1, 2:(1 + D), (2 + D):n)
end

"""
    vech_pairs(D)

The `(r, c)` index pairs, `r ≥ c`, of `vech` in storage order: column by
column of the lower triangle.
"""
vech_pairs(D::Integer) = [(r, c) for c in 1:D for r in c:D]

"""
    vech(H)

Half-vectorise a symmetric `D × D` matrix: the lower triangle, column by column.
"""
function vech(H::AbstractMatrix)
    D = size(H, 1)
    size(H, 2) == D || throw(DimensionMismatch("vech needs a square matrix"))
    out = Vector{Float64}(undef, div(D * (D + 1), 2))
    p = 0
    @inbounds for c in 1:D, r in c:D
        p += 1
        out[p] = H[r, c]
    end
    return out
end

"""
    unvech(h, D)

Inverse of [`vech`](@ref): the symmetric `D × D` matrix with lower triangle `h`.
"""
function unvech(h::AbstractVector, D::Integer)
    length(h) == div(D * (D + 1), 2) ||
        throw(DimensionMismatch("h has length $(length(h)), expected $(div(D*(D+1),2))"))
    H = Matrix{Float64}(undef, D, D)
    p = 0
    @inbounds for c in 1:D, r in c:D
        p += 1
        H[r, c] = h[p]
        H[c, r] = h[p]
    end
    return H
end

"""
    duplication_matrix(D)

The duplication matrix `D_D` (D² × D_h) with `vec(H) = D_D * vech(H)` for
symmetric `H`. Provided for tests and documentation; the solver never forms it.
"""
function duplication_matrix(D::Integer)
    D_h = div(D * (D + 1), 2)
    M = zeros(D * D, D_h)
    p = 0
    for c in 1:D, r in c:D
        p += 1
        M[(c - 1) * D + r, p] = 1.0
        M[(r - 1) * D + c, p] = 1.0
    end
    return M
end

"""
    lambda_coeffs!(λ, a)

Fill `λ = ½ D_Dᵀ (a ⊗ a)` for the displacement `a`, so that `λ' * vech(H) == ½ a' H a`.
Entry `(r, c)` of `vech` receives `½ a_r²` on the diagonal and `a_r a_c` below it.
"""
function lambda_coeffs!(λ::AbstractVector{Float64}, a::AbstractVector{<:Real})
    D = length(a)
    length(λ) == div(D * (D + 1), 2) || throw(DimensionMismatch("λ has the wrong length"))
    p = 0
    @inbounds for c in 1:D, r in c:D
        p += 1
        λ[p] = r == c ? 0.5 * a[r] * a[r] : a[r] * a[c]
    end
    return λ
end

lambda_coeffs(a::AbstractVector{<:Real}) =
    lambda_coeffs!(Vector{Float64}(undef, div(length(a) * (length(a) + 1), 2)), a)

"""
    c_coeffs!(C, a)

Fill the `D_h × D` matrix `C = ½ D_Dᵀ (a ⊗ I_D)`, so that `C' * vech(H) == ½ H a`.
Column `k` holds the `vech` coefficients of `½ (H a)_k`.
"""
function c_coeffs!(C::AbstractMatrix{Float64}, a::AbstractVector{<:Real})
    D = length(a)
    size(C) == (div(D * (D + 1), 2), D) || throw(DimensionMismatch("C has the wrong size"))
    fill!(C, 0.0)
    p = 0
    @inbounds for c in 1:D, r in c:D
        p += 1
        if r == c
            C[p, r] += 0.5 * a[r]
        else
            C[p, r] += 0.5 * a[c]     # H_rc a_c contributes to (H a)_r
            C[p, c] += 0.5 * a[r]     # H_cr a_r contributes to (H a)_c
        end
    end
    return C
end

c_coeffs(a::AbstractVector{<:Real}) =
    c_coeffs!(Matrix{Float64}(undef, div(length(a) * (length(a) + 1), 2), length(a)), a)

"""
    state_triple(u, g, H)

Pack a value, gradient and symmetric Hessian into θ = col(u, g, vech(H)).
"""
state_triple(u::Real, g::AbstractVector, H::AbstractMatrix) = vcat(Float64(u), Float64.(g), vech(H))
