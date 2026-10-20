using Test, LinearAlgebra, Random
using Diona

# Manuscript, Notation and §2.2: h = vech(H), λ(a) = ½ D_Dᵀ(a⊗a), C(a) = ½ D_Dᵀ(a⊗I_D)
@testset "state triples" begin
    rng = MersenneTwister(2026)
    for D in 1:6
        L = TripleLayout(D)
        @test L.D_h == div(D * (D + 1), 2)
        @test L.n == 1 + D + L.D_h
        @test L.u == 1 && L.g == 2:1+D && L.h == 2+D:L.n
        @test length(vech_pairs(D)) == L.D_h
        @test all(r >= c for (r, c) in vech_pairs(D))

        H = Matrix(Symmetric(randn(rng, D, D)))
        h = vech(H)
        @test unvech(h, D) == H
        @test duplication_matrix(D) * h == vec(H)

        a = randn(rng, D)
        λ = lambda_coeffs(a)
        C = c_coeffs(a)
        @test dot(λ, h) ≈ 0.5 * dot(a, H * a) atol=1e-12
        @test C' * h ≈ 0.5 * (H * a) atol=1e-12
        @test λ ≈ 0.5 * duplication_matrix(D)' * kron(a, a) atol=1e-12
        @test C ≈ 0.5 * duplication_matrix(D)' * kron(a, Matrix(I, D, D)) atol=1e-12

        θ = state_triple(1.5, a, H)
        @test θ[L.u] == 1.5 && θ[L.g] == a && θ[L.h] == h
    end
    @test_throws ArgumentError TripleLayout(0)
    @test_throws DimensionMismatch vech(zeros(2, 3))
end
