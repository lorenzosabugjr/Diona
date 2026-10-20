using Test, SparseArrays
using Diona

@testset "HiGHS LP wrapper" begin
    # 0 ≤ x ≤ 1 written as A x ≤ b
    A = sparse([-1.0; 1.0][:, :]); b = [0.0, 1.0]
    lo, hi, st = Diona.extremal_values(A, b, 1)
    @test st == :OPTIMAL
    @test lo ≈ 0.0 atol=1e-9
    @test hi ≈ 1.0 atol=1e-9

    # contradictory: x ≤ 0 and x ≥ 1
    lo, hi, st = Diona.extremal_values(sparse([1.0; -1.0][:, :]), [0.0, -1.0], 1)
    @test st == :INFEASIBLE && lo === nothing && hi === nothing

    # no constraints: unbounded
    lo, hi, st = Diona.extremal_values(spzeros(0, 1), Float64[], 1)
    @test st == :UNBOUNDED && lo === nothing && hi === nothing

    # bounded below only: the maximum is unbounded, nothing is exposed
    lo, hi, st = Diona.extremal_values(sparse([-1.0][:, :]), [0.0], 1)
    @test st == :UNBOUNDED && lo === nothing && hi === nothing

    # solve_lp with column bounds: min x₁ + x₂ s.t. x₁ + x₂ ≥ 1, x ≥ 0
    sol = Diona.solve_lp(sparse([-1.0 -1.0]), [-1.0], [1.0, 1.0]; col_lower=[0.0, 0.0])
    @test sol.status == :OPTIMAL
    @test sol.objective ≈ 1.0 atol=1e-9
    @test_throws BoundsError Diona.extremal_values(A, b, 2)
end
