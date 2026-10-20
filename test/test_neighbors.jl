using Test, LinearAlgebra, Random
using Diona

function _cartesian_stencil(D::Int)
    values = -1.0:0.5:1.0
    pts = Iterators.product(ntuple(_ -> values, D)...)
    return reduce(vcat, [permutedims(collect(p)) for p in pts if any(!=(0.0), p)])
end

@testset "feature matrix Φ and poisedness (Definition 1)" begin
    rng = MersenneTwister(5)
    for D in 1:4
        X = randn(rng, 30, D); xq = zeros(D)
        idx = collect(1:12)
        Φ, r = feature_matrix(X, xq, idx)
        @test r ≈ maximum(norm(X[i, :]) for i in idx)
        D_h = div(D * (D + 1), 2)
        @test size(Φ) == (12, 1 + D + D_h)
        for (row, i) in enumerate(idx)
            â = (xq .- X[i, :]) ./ r
            @test Φ[row, 1] == 1.0
            @test Φ[row, 2:1+D] == -â
            @test Φ[row, 2+D:end] ≈ lambda_coeffs(â)          # λ̂ = λ(a)/r² = λ(â)
            @test Φ[row, 2+D:end] ≈ lambda_coeffs(xq .- X[i, :]) ./ r^2
        end
        μ, r2, σ = poisedness(X, xq, idx)
        @test r2 == r && μ ≈ σ / sqrt(12) && σ ≈ minimum(svdvals(Φ))
    end
end

@testset "Algorithm 1: poised selection" begin
    for D in 1:3
        X = _cartesian_stencil(D); xq = zeros(D)
        sel = select_poised_neighbors(X, xq)
        d = sel.diagnostics
        Φ, r = feature_matrix(X, xq, sel.indices)
        m = 1 + D + div(D * (D + 1), 2)
        @test d.method == :poised && d.mu0 == DEFAULT_MU0
        @test length(sel.indices) >= m && issorted(sel.indices) && allunique(sel.indices)
        @test d.mu ≈ minimum(svdvals(Φ)) / sqrt(length(sel.indices))
        @test d.mu >= d.mu0 && d.radius ≈ r && d.n_selected == length(sel.indices)
    end

    @testset "greedy growth before pool expansion" begin
        X = reshape([-1.0, -0.5, 0.5, 1.0], :, 1)
        # μ of the four-point set is 0.1787 and of the best three-point sets 0.1580 (paper scaling).
        sel = select_poised_neighbors(X, [0.0]; mu0=0.17, pool_size=4, max_neighbors=4)
        @test length(sel.indices) == 4 && sel.diagnostics.pool_expansions == 0
        @test sel.diagnostics.mu >= 0.17
        sel = select_poised_neighbors(X, [0.0]; mu0=0.15, pool_size=3, max_neighbors=3)
        @test length(sel.indices) == 3 && sel.diagnostics.pool_expansions == 1
        @test sel.diagnostics.pool_size == 4 && sel.diagnostics.mu >= 0.15
    end

    @testset "duplicates, one-sided and insufficient data" begin
        dup = reshape([0.0, 0.0, -1.0, -1.0, -0.5, 0.5, 1.0, 1.0], :, 1)
        sel = select_poised_neighbors(dup, [0.0]).indices
        @test allunique(dup[sel, 1]) && all(!=(0.0), dup[sel, 1])
        boundary = reshape([0.1, 0.2, 0.4, 0.7, 1.0], :, 1)
        @test select_poised_neighbors(boundary, [0.0]; mu0=0.005).diagnostics.mu >= 0.005
        @test_throws NeighborSelectionError select_poised_neighbors(reshape([-1.0, 1.0], :, 1), [0.0])
        collinear = reduce(vcat, [permutedims([x, 0.0]) for x in -2.0:0.5:2.0 if x != 0.0])
        err = try select_poised_neighbors(collinear, [0.0, 0.0]); nothing catch e; e end
        @test err isa NeighborSelectionError && err.diagnostics.method == :poised
        @test sprint(showerror, err) isa String
    end

    @testset "argument checks" begin
        X = _cartesian_stencil(2)
        @test_throws ArgumentError select_poised_neighbors(X, zeros(2); mu0=0.0)
        @test_throws ArgumentError select_poised_neighbors(X, zeros(2); rank_tol=1.0)
        @test_throws ArgumentError select_poised_neighbors(X, zeros(2); max_neighbors=3)
        @test_throws ArgumentError select_poised_neighbors(X, zeros(2); pool_growth=1)
        @test_throws DimensionMismatch select_poised_neighbors(X, zeros(3))
    end
end

@testset "three passes (§4.2) and the unified selector" begin
    rng = MersenneTwister(20260813)
    for D in 1:3
        X = 2 .* rand(rng, 180, D) .- 1
        xq = collect(range(-0.13; step=0.15, length=D))
        direct = select_poised_neighbors(X, xq)
        one = select_neighbors(X, xq; layers=1, return_diagnostics=true)
        @test one.indices == direct.indices && one.diagnostics == direct.diagnostics
        three = select_neighbors(X, xq; return_diagnostics=true)      # layers = DEFAULT_LAYERS = 3
        @test DEFAULT_LAYERS == 3
        @test issubset(direct.indices, three.indices)
        @test length(three.indices) == 3 * length(direct.indices)
        @test allunique(three.indices) && issorted(three.indices)
        @test three.diagnostics.n_selected == length(three.indices)
        @test three.diagnostics.indices === nothing
        μ, r, _ = poisedness(X, xq, three.indices)
        @test three.diagnostics.mu == μ && three.diagnostics.radius == r
        @test select_neighbors(X, xq) == three.indices
    end
    # later passes stop cleanly when the data run out: the 1-D stencil has four
    # points and a pass needs three, so a second pass cannot be completed.
    X = _cartesian_stencil(1)
    many = select_neighbors(X, [0.0]; layers=10)
    @test many == select_neighbors(X, [0.0]; layers=1)
    @test length(many) == 3
end

@testset "legacy selectors (not in the manuscript)" begin
    X = _cartesian_stencil(2); xq = [0.1, -0.1]
    @test NEIGHBOR_METHODS == (:poised, :knn, :info, :all)
    knn = select_neighbors(X, xq; method=:knn, param=5)
    @test knn == k_nearest_neighbors(X, xq, 5) && length(knn) == 5
    info = select_neighbors(X, xq; method=:info, param=2)
    @test info == k_informative_neighbors(X, xq; layers=2)
    @test length(k_informative_neighbors(X, xq; layers=1)) == 3        # D_h = 3 in 2-D
    all_idx = select_neighbors(X, xq; method=:all, return_diagnostics=true)
    @test length(all_idx.indices) == size(X, 1) && all_idx.diagnostics.mu === nothing
    @test_throws ArgumentError select_neighbors(X, xq; method=:info, layers=2)
    @test_throws ArgumentError select_neighbors(X, xq; method=:nope)
    # 1-D :info takes `layers` samples on each side
    X1 = reshape(collect(-1.0:0.25:1.0), :, 1)
    @test k_informative_neighbors(X1, [0.1]; layers=2) == [4, 5, 6, 7]
end

@testset "KD-tree cache" begin
    clear_kdtree_cache!()
    a = reshape([0.1, 10.0, 20.0], :, 1); b = reshape([10.0, 20.0, 0.1], :, 1)
    stale = build_kdtree_cache!(a)
    @test stale.trusted == false
    @test k_nearest_neighbors(a, [0.0], 1; kdtree_cache=stale) == [1]
    @test k_nearest_neighbors(b, [0.0], 1; kdtree_cache=stale) == [3]        # content change detected
    @test get_kdtree_cache().coordinate_fingerprint == coordinate_fingerprint(b)
    clear_kdtree_cache!()
    trusted = build_kdtree_cache!(a; trusted=true)
    @test k_nearest_neighbors(b, [0.0], 1; kdtree_cache=trusted) == [1]      # by design: not re-hashed
    @test get_kdtree_cache() === trusted
    c = reshape([0.1, 10.0, 20.0, 30.0], :, 1)
    @test k_nearest_neighbors(c, [100.0], 1; kdtree_cache=trusted) == [4]    # shape still checked
    clear_kdtree_cache!()
    pts = _cartesian_stencil(2)
    want = select_neighbors(pts, [0.15, -0.05])
    clear_kdtree_cache!(); build_kdtree_cache!(pts; trusted=true)
    @test select_neighbors(pts, [0.15, -0.05]) == want
    clear_kdtree_cache!()
    @test get_kdtree_cache() === nothing
end

@testset "candidate pool is nearest-first" begin
    for D in 1:3
        X = _cartesian_stencil(D); xq = fill(0.05, D)
        cache = build_kdtree_cache!(X)
        dists = [norm(X[i, :] .- xq) for i in axes(X, 1)]
        for requested in unique(min.((4, 9, 15), size(X, 1)))
            pool = Diona._candidate_pool(X, xq, requested, cache)
            @test length(pool) == requested && dists[pool] == sort(dists)[1:requested]
        end
    end
    clear_kdtree_cache!()
end
