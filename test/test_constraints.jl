using Test, LinearAlgebra, Random, SparseArrays
using Diona

# Manuscript §2.2, rows (8a)–(8e) and the admissible set (10).
_quadratic(rng, D) = begin
    A = Matrix(Symmetric(randn(rng, D, D))); b = randn(rng, D); c = randn(rng)
    (u = x -> c + dot(b, x) + 0.5 * dot(x, A * x), g = x -> b + A * x, H = A)
end

@testset "Taylor pair rows" begin
    rng = MersenneTwister(7)
    for D in 1:5
        L = TripleLayout(D)
        m = Diona.taylor_rows_per_neighbor(L)
        @test m == 4 + 2D + 2L.D_h
        q = _quadratic(rng, D)
        xq, xi = randn(rng, D), randn(rng, D)
        θq = state_triple(q.u(xq), q.g(xq), q.H)
        θi = state_triple(q.u(xi), q.g(xi), q.H)
        Q, S, ρ = taylor_pair_rows(xq .- xi, L)
        @test size(Q) == (m, L.n) && size(S) == (m, L.n) && length(ρ) == m
        # A quadratic satisfies (8a)–(8d) with L_H = 0: every row is ≤ 0.
        @test maximum(Q * θq + S * θi) < 1e-10
        # Remainder coefficients are ‖a‖³/6 (four rows), ‖a‖²/4 (2D rows), ‖a‖ (2D_h rows).
        r = norm(xq .- xi)
        @test all(ρ[1:4] .≈ r^3 / 6)
        @test all(ρ[5:4+2D] .≈ r^2 / 4)
        @test all(ρ[5+2D:end] .≈ r)
        # Rows come in ± pairs.
        for k in 1:2:m
            @test Q[k, :] == -Q[k+1, :] && S[k, :] == -S[k+1, :] && ρ[k] == ρ[k+1]
        end
    end
end

@testset "containment of the true triples (Lemma 1)" begin
    # u(x) = x₁³ + x₂³ has Hessian diag(6x₁, 6x₂): ‖∇²u(x)−∇²u(y)‖_∞ ≤ 6‖x−y‖_∞ ≤ 6‖x−y‖₂.
    rng = MersenneTwister(11)
    u(x) = x[1]^3 + x[2]^3
    g(x) = [3x[1]^2, 3x[2]^2]
    H(x) = [6x[1] 0.0; 0.0 6x[2]]
    X = 2 .* rand(rng, 40, 2) .- 1
    ε̄ = 0.05
    z = [u(X[i, :]) + ε̄ * (2rand(rng) - 1) for i in 1:40]
    samples = hcat(X, z)
    xq = [0.1, -0.2]
    P = ActiveSet(samples, xq, 1:40)
    L = TripleLayout(2)
    ψ_true = vcat(state_triple(u(xq), g(xq), H(xq)),
                  (state_triple(u(X[i, :]), g(X[i, :]), H(X[i, :])) for i in 1:40)...)
    Ξ, b = admissible_set(P; noise_bound=ε̄, L_H=6.0)
    @test size(Ξ) == (40 * Diona.taylor_rows_per_neighbor(L) + 80, 41 * L.n)
    @test all(Ξ * ψ_true .<= b .+ 1e-12)
    # The assembled pieces agree with the per-neighbour rows.
    T, ρ = taylor_system(P)
    M, zs = measurement_system(P)
    @test Ξ == vcat(T, M) && b == vcat(6.0 .* ρ, zs .+ ε̄)
    Q, S, ρ1 = taylor_pair_rows(xq .- X[3, :], L)
    m = Diona.taylor_rows_per_neighbor(L)
    rows = (2m + 1):(3m)
    @test Matrix(T[rows, Diona.triple_columns(L, 0)]) == Q
    @test Matrix(T[rows, Diona.triple_columns(L, 3)]) == S
    @test ρ[rows] == ρ1
    @test n_neighbors(P) == 40
    @test radius(P) ≈ maximum(norm(xq .- X[i, :]) for i in 1:40)
    @test_throws ArgumentError admissible_set(P; noise_bound=-1.0, L_H=1.0)
    @test_throws ArgumentError ActiveSet(samples, xq, Int[])
    @test_throws DimensionMismatch ActiveSet(samples, [0.0], 1:3)
end
