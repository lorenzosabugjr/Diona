using Test, Random
using Diona

@testset "noise-bound estimate (§4.1)" begin
    rng = MersenneTwister(3)
    X = 2 .* rand(rng, 400, 2) .- 1
    f = [sin(2X[i, 1]) + X[i, 2]^2 for i in 1:400]
    clean = hcat(X, f)
    ε_clean = estimate_noise_bound(clean; x_cols=1:2, z_col=3)
    @test isfinite(ε_clean) && ε_clean >= 0
    noisy = hcat(X, f .+ 0.1 .* (2 .* rand(rng, 400) .- 1))
    ε_noisy = estimate_noise_bound(noisy; x_cols=1:2, z_col=3)
    @test ε_noisy > ε_clean
    @test ε_noisy <= 0.1                      # biased low (see docstring), never above the truth here
    @test estimate_noise_bound(hcat(zeros(5), rand(rng, 5)); x_cols=1:1, z_col=2) == 0.0
    @test_throws ArgumentError estimate_noise_bound(noisy; x_cols=1:2, z_col=2)
    @test_throws ArgumentError estimate_noise_bound(noisy[1:1, :]; x_cols=1:2, z_col=3)
end
