using Test, LinearAlgebra, Random
using Diona

@testset "Algorithm 2 on exact data" begin
    # 1-D quadratic with a known Hessian: zero-width interval at L_H = 0.
    xs = collect(-1.0:0.25:1.0)
    r = infer(hcat(xs, xs .^ 2), [0.1]; noise_bound=0.0, L_H=0.0)
    @test r.status == :OPTIMAL && is_accepted(r) && has_interval(r)
    @test r.estimate ≈ 0.01 atol=1e-7
    @test r.width < 1e-7
    @test r.L_H_used == 0.0 && r.noise_bound == 0.0
    @test r.n_neighbors == r.neighbor_diagnostics.n_selected == length(r.neighbor_diagnostics.indices)
    @test r.radius ≈ maximum(abs.(xs[r.neighbor_diagnostics.indices] .- 0.1))
    @test r.timings.LH_estimation == 0.0 && r.timings.total >= r.timings.neighbor_selection + r.timings.bounds

    # 3-D quadratic
    f(x) = x[1]^2 + x[2]^2 + 0.4x[3]
    pts = [[x, y, t] for x in -1.0:0.5:1.0 for y in -1.0:0.5:1.0 for t in 0.0:0.5:1.0]
    S = reduce(vcat, [reshape([p; f(p)], 1, :) for p in pts])
    q = [0.1, -0.1, 0.5]
    r = infer(S, q; noise_bound=1e-9, L_H=0.0)
    @test r.status == :OPTIMAL && r.lower <= f(q) + 1e-6 && r.upper >= f(q) - 1e-6
    @test r.estimate ≈ (r.lower + r.upper) / 2 && r.width ≈ r.upper - r.lower
    @test size(S, 2) == 4 && r.n_neighbors == 3 * 10   # three passes of the 10-point 3-D stencil
end

@testset "containment with the true constants (Lemma 1)" begin
    rng = MersenneTwister(99)
    u(x) = x[1]^3 - 2x[2]^3 + x[1] * x[2]          # Hessian entries are 6x₁, −12x₂, 1 → L_H = 12
    X = 2 .* rand(rng, 300, 2) .- 1
    ε̄ = 0.02
    z = [u(X[i, :]) + ε̄ * (2rand(rng) - 1) for i in 1:300]
    samples = hcat(X, z)
    for _ in 1:20
        q = 1.6 .* rand(rng, 2) .- 0.8
        r = infer(samples, q; noise_bound=ε̄, L_H=12.0)
        @test r.status == :OPTIMAL
        @test r.lower - 1e-8 <= u(q) <= r.upper + 1e-8
    end
end

@testset "L_H estimation (§4.1) and the safety factor" begin
    rng = MersenneTwister(5)
    u(x) = sin(3x[1]) + 0.4cos(2x[2]) + 0.2x[1] * x[2]^2
    X = 2 .* rand(rng, 250, 2) .- 1
    samples = hcat(X, [u(X[i, :]) for i in 1:250])
    q = [0.13, -0.21]
    single = infer(samples, q; noise_bound=0.0, layers=1)
    three = infer(samples, q; noise_bound=0.0)
    # One pass of Algorithm 1 is interpolated by a quadratic (L̃_H ≈ 0); three passes are not.
    @test single.status == :OPTIMAL && single.L_H_used <= 1e-6
    @test three.status == :OPTIMAL && three.L_H_used > 1.0
    P = ActiveSet(samples, q, three.neighbor_diagnostics.indices)
    LH, st = estimate_LH(P; noise_bound=0.0)
    @test st == :OPTIMAL && three.L_H_used == LH
    lo, hi, st2 = admissible_bounds(P; noise_bound=0.0, L_H=LH)
    @test st2 == :OPTIMAL
    @test lo ≈ three.lower atol=1e-9
    @test hi ≈ three.upper atol=1e-9
    scaled = infer(samples, q; noise_bound=0.0, L_H_factor=2.5)
    @test scaled.L_H_used ≈ 2.5 * LH && scaled.width >= three.width
    fixed = infer(samples, q; noise_bound=0.0, L_H=2.0, L_H_factor=2.5)
    @test fixed.L_H_used == 5.0 && fixed.timings.LH_estimation == 0.0
    @test three.timings.LH_estimation > 0.0
end

@testset "failure statuses expose nothing" begin
    @test DIONA_STATUSES == (:OPTIMAL, :NEIGHBOR_FAILED, :INFEASIBLE, :UNBOUNDED, :SOLVE_FAILED)
    conflicting = hcat([0.0, 0.0, 1.0, -1.0], [100.0, -100.0, 0.0, 0.0])
    r = infer(conflicting, [0.1]; noise_bound=1e-6, L_H=0.0, method=:knn, param=4)
    @test r.status == :INFEASIBLE && !is_accepted(r) && !has_interval(r)
    @test r.estimate === nothing && r.width === nothing && r.L_H_used == 0.0
    r = infer(hcat([0.0], [0.0]), [0.1]; noise_bound=0.05, L_H=2.0, method=:knn, param=1)
    @test r.status == :UNBOUNDED && r.estimate === nothing
    r = infer(hcat([-1.0, 1.0], [1.0, 1.0]), [0.0]; noise_bound=0.1, L_H=1.0)
    @test r.status == :NEIGHBOR_FAILED && r.n_neighbors == 0 && r.radius === nothing
    @test r.neighbor_diagnostics.method == :poised && r.L_H_used == 1.0
    @test r.timings.bounds == 0.0
end

@testset "argument validation" begin
    xs = collect(-1.0:0.25:1.0); S = hcat(xs, xs .^ 2)
    @test_throws DimensionMismatch infer(S, [0.0, 0.0]; noise_bound=0.0)
    @test_throws ArgumentError infer(S, [0.0]; noise_bound=-1.0)
    @test_throws ArgumentError infer(S, [0.0]; noise_bound=0.0, L_H_factor=0.0)
    @test_throws ArgumentError infer(S, [0.0]; noise_bound=0.0, L_H=:local)
    @test_throws ArgumentError infer(S, [0.0]; noise_bound=0.0, L_H=-1.0)
    @test_throws ArgumentError infer(reshape(xs, :, 1), [0.0]; noise_bound=0.0)
    # integer inputs are accepted
    r = infer(hcat(-2:2, (-2:2) .^ 2), [0]; noise_bound=0, L_H=0)
    @test r.status == :OPTIMAL
    @test r.estimate ≈ 0.0 atol=1e-7
end
