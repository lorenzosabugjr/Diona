# Neighbour selection.
#
# Manuscript, Section 2.3 (poised neighbour sets, Definition 1, Assumption 3)
# and Section 4.2 / Algorithm 1 (the practical greedy selector). For a
# candidate neighbour x_i at trial radius r the normalised feature row is
#
#     φ_i = [1, −â_iᵀ, λ̂_iᵀ],   â_i = a_i / r,   λ̂_i = λ(a_i) / r² = λ(â_i),
#
# and a set 𝒩 of N ≥ 1 + D + D_h neighbours is μ0-poised when
#
#     σ_min(Φ) ≥ μ0 √N,   Φ = [φ_{i_1}ᵀ; …; φ_{i_N}ᵀ].
#
# The practical selector (Algorithm 1) fetches a nearest-first candidate pool
# from a KD-tree, proposes a full-rank m-point set by column-pivoted QR of Φ_Cᵀ,
# re-measures μ(𝒩) = σ_min(Φ)/√N at the exact selected radius, and greedily adds
# the candidate that raises μ the most until μ ≥ μ0, the cap N̄ is reached, or
# the pool is exhausted (in which case the pool is expanded).
#
# Section 4.2 then runs three passes of Algorithm 1, each excluding earlier
# picks, and uses the union (`layers=3` below).
#
# The `:knn`, `:info` and `:all` selectors at the end of this file are NOT in
# the manuscript. They are kept for experimentation and reproduction of older
# runs; the benchmark uses `:poised` only.

const DEFAULT_MU0 = 1e-6               # poisedness threshold μ0 (see Assumption 3)
const DEFAULT_RANK_TOL = 1e-9          # relative σ_min/σ_max test for the QR proposal
const DEFAULT_MAX_NEIGHBORS_FACTOR = 2 # N̄ = factor × (1 + D + D_h)
const DEFAULT_POOL_GROWTH = 2
const DEFAULT_LAYERS = 3               # Section 4.2: three passes of Algorithm 1
const DEFAULT_KNN = 8                  # legacy :knn
const DEFAULT_INFO_LAYERS = 6          # legacy :info
const NEIGHBOR_METHODS = (:poised, :knn, :info, :all)

# ---------------------------------------------------------------------------
# KD-tree cache (implementation detail: Algorithm 1 only needs nearest-first
# candidates; a cached tree makes repeated queries against one data set cheap).
# ---------------------------------------------------------------------------

struct KDTreeCache{T}
    tree::T
    n_samples::Int
    D::Int
    coordinate_fingerprint::UInt
    trusted::Bool   # skip the O(N D) content check on every hit
end

const KDTREE_CACHE = Ref{Union{Nothing,KDTreeCache}}(nothing)
const KDTREE_CACHE_LOCK = ReentrantLock()

"""
    coordinate_fingerprint(X)

Content hash of a coordinate matrix; changes whenever any coordinate changes.
"""
function coordinate_fingerprint(X::AbstractMatrix{<:Real})
    h = hash(size(X), UInt(0x9e3779b97f4a7c15))
    @inbounds for v in X
        h = hash(Float64(v), h)
    end
    return h
end

_cache_matches(cache::KDTreeCache, X::AbstractMatrix{<:Real}) =
    cache.n_samples == size(X, 1) && cache.D == size(X, 2) &&
    (cache.trusted || cache.coordinate_fingerprint == coordinate_fingerprint(X))

"""
    build_kdtree_cache!(X; trusted=false)

Build and cache a KD-tree over the `N × D` coordinate matrix `X`. With
`trusted=false` every later hit re-hashes the coordinates to detect in-place
edits; with `trusted=true` only the shape is checked, which is what a sweep of
many queries over one fixed data set wants. Thread-safe.
"""
function build_kdtree_cache!(X::AbstractMatrix{Float64}; trusted::Bool=false)
    return lock(KDTREE_CACHE_LOCK) do
        tree = KDTree(permutedims(X); leafsize=10)
        cache = KDTreeCache(tree, size(X, 1), size(X, 2), coordinate_fingerprint(X), trusted)
        KDTREE_CACHE[] = cache
        cache
    end
end

get_kdtree_cache() = KDTREE_CACHE[]

function clear_kdtree_cache!()
    lock(KDTREE_CACHE_LOCK) do
        KDTREE_CACHE[] = nothing
    end
end

function _matching_kdtree_cache(X::AbstractMatrix{Float64}, supplied::Union{Nothing,KDTreeCache})
    supplied !== nothing && _cache_matches(supplied, X) && return supplied
    cache = get_kdtree_cache()
    cache !== nothing && _cache_matches(cache, X) && return cache
    return build_kdtree_cache!(X)
end

# ---------------------------------------------------------------------------
# Diagnostics and failure
# ---------------------------------------------------------------------------

"""
    NeighborDiagnostics

What the selector did: `method`, the threshold `mu0` and achieved poisedness
`mu = σ_min(Φ)/√N` (both `nothing` for the legacy selectors), the radius
`r_N`, the number selected, the candidate-pool size and the number of pool
expansions, and the selected `indices`.
"""
struct NeighborDiagnostics
    method::Symbol
    mu0::Union{Nothing,Float64}
    mu::Union{Nothing,Float64}
    radius::Union{Nothing,Float64}
    n_selected::Int
    pool_size::Int
    pool_expansions::Int
    indices::Union{Nothing,Vector{Int}}
end

NeighborDiagnostics(method, mu0, mu, radius, n_selected::Int, pool_size::Int, pool_expansions::Int) =
    NeighborDiagnostics(method, mu0, mu, radius, n_selected, pool_size, pool_expansions, nothing)

Base.:(==)(a::NeighborDiagnostics, b::NeighborDiagnostics) =
    all(getfield(a, f) == getfield(b, f) for f in fieldnames(NeighborDiagnostics))
Base.hash(d::NeighborDiagnostics, h::UInt) =
    foldl((acc, f) -> hash(getfield(d, f), acc), fieldnames(NeighborDiagnostics); init=h)

"""A successful selection: the indices and their diagnostics."""
struct NeighborSelection
    indices::Vector{Int}
    diagnostics::NeighborDiagnostics
end

"""Raised when no candidate set meets the poisedness test (Algorithm 1, NeighbourFailure)."""
struct NeighborSelectionError <: Exception
    message::String
    diagnostics::NeighborDiagnostics
end
Base.showerror(io::IO, err::NeighborSelectionError) = print(io, err.message)

# ---------------------------------------------------------------------------
# Feature matrix Φ and poisedness μ (Section 2.3)
# ---------------------------------------------------------------------------

"""
    feature_matrix(X, x_query, indices; radius=nothing) -> (Φ, r)

Rows `φ_i = [1, −â_iᵀ, λ̂_iᵀ]` for the samples `indices` of `X`, normalised by
the radius `r` (the exact radius of the set unless `radius` is given).
"""
function feature_matrix(X::AbstractMatrix{Float64}, x_query::AbstractVector{Float64},
                        indices::AbstractVector{Int}; radius::Union{Nothing,Float64}=nothing)
    isempty(indices) && throw(ArgumentError("a feature matrix needs at least one point"))
    D = size(X, 2)
    length(x_query) == D || throw(DimensionMismatch("query dimension does not match samples"))
    D_h = div(D * (D + 1), 2)
    displacements = permutedims(x_query) .- X[indices, :]     # a_i = x' − x_i, one per row
    distances = vec(sqrt.(sum(abs2, displacements; dims=2)))
    r = radius === nothing ? maximum(distances) : radius
    isfinite(r) && r > 0 || throw(ArgumentError("selected radius must be finite and positive"))
    Φ = Matrix{Float64}(undef, length(indices), 1 + D + D_h)
    λ = Vector{Float64}(undef, D_h)
    @inbounds for row in axes(Φ, 1)
        â = @view(displacements[row, :]) ./ r
        lambda_coeffs!(λ, â)
        Φ[row, 1] = 1.0
        Φ[row, 2:1 + D] .= -â
        Φ[row, 2 + D:end] .= λ
    end
    return Φ, r
end

"""
    poisedness(X, x_query, indices) -> (μ, r, σ_min)

`μ(𝒩) = σ_min(Φ)/√N` at the exact radius of the set (Definition 1).
"""
function poisedness(X::AbstractMatrix{Float64}, x_query::AbstractVector{Float64},
                    indices::AbstractVector{Int})
    Φ, r = feature_matrix(X, x_query, indices)
    σ_min = minimum(svdvals(Φ))
    return σ_min / sqrt(length(indices)), r, σ_min
end

# ---------------------------------------------------------------------------
# Algorithm 1: practical μ0-poised neighbour selection
# ---------------------------------------------------------------------------

"""
    _candidate_pool(X, x_query, requested, cache; exclude=nothing)

Nearest-first unique, non-coincident candidates (at most `requested`), skipping
indices in `exclude`. The KD query grows until enough candidates are found or
the data set is exhausted.
"""
function _candidate_pool(X::AbstractMatrix{Float64}, x_query::AbstractVector{Float64},
                         requested::Int, cache::KDTreeCache;
                         exclude::Union{Nothing,AbstractSet{Int}}=nothing)
    n_samples = size(X, 1)
    query_count = min(n_samples, max(requested + 1, 1))
    unique_indices = Int[]; unique_dists = Float64[]
    while true
        idxs, dists = knn(cache.tree, x_query, query_count, true)
        empty!(unique_indices); empty!(unique_dists)
        seen = Set{Tuple}()
        for k in eachindex(idxs)
            dists[k] > 0 || continue
            idx = idxs[k]
            exclude !== nothing && idx in exclude && continue
            key = Tuple(@view X[idx, :])
            key in seen && continue
            push!(seen, key); push!(unique_indices, idx); push!(unique_dists, dists[k])
        end
        length(unique_indices) >= requested && break
        query_count == n_samples && break
        query_count = min(n_samples, max(query_count + 1, 2 * query_count))
    end
    order = sortperm(unique_dists; alg=MergeSort)
    resize!(order, min(requested, length(order)))
    return unique_indices[order]
end

function _poised_failure(message, mu0, best_mu, best_radius, best_size, pool_size, expansions)
    diagnostics = NeighborDiagnostics(:poised, mu0, best_mu, best_radius, best_size, pool_size, expansions)
    throw(NeighborSelectionError(message, diagnostics))
end

"""
    select_poised_neighbors(X, x_query; mu0=DEFAULT_MU0, max_neighbors=nothing,
                            pool_size=nothing, rank_tol=DEFAULT_RANK_TOL,
                            pool_growth=DEFAULT_POOL_GROWTH, kdtree_cache=nothing,
                            exclude=nothing) -> NeighborSelection

Algorithm 1 of the manuscript. `X` is `N × D`. `max_neighbors` is N̄ (default
`2 (1 + D + D_h)`), `pool_size` the initial candidate pool |𝒞| (default N̄).
Pivoted QR on Φ_𝒞ᵀ proposes the minimal full-rank set; every acceptance is an
SVD of Φ rebuilt at the exact selected radius; greedy growth adds the candidate
that maximises μ until μ ≥ `mu0`; the pool expands by `pool_growth` when the
cap or the pool is exhausted. Throws `NeighborSelectionError` when every
candidate set fails.
"""
function select_poised_neighbors(X::AbstractMatrix{Float64}, x_query::AbstractVector{Float64};
                                 mu0::Real=DEFAULT_MU0,
                                 max_neighbors::Union{Nothing,Int}=nothing,
                                 pool_size::Union{Nothing,Int}=nothing,
                                 rank_tol::Float64=DEFAULT_RANK_TOL,
                                 pool_growth::Int=DEFAULT_POOL_GROWTH,
                                 kdtree_cache::Union{Nothing,KDTreeCache}=nothing,
                                 exclude::Union{Nothing,AbstractSet{Int}}=nothing)
    n_samples, D = size(X)
    length(x_query) == D || throw(DimensionMismatch("query dimension does not match samples"))
    threshold = Float64(mu0)
    isfinite(threshold) && threshold > 0 || throw(ArgumentError("mu0 must be finite and positive"))
    isfinite(rank_tol) && 0 < rank_tol < 1 ||
        throw(ArgumentError("rank_tol must lie strictly between zero and one"))
    pool_growth >= 2 || throw(ArgumentError("pool_growth must be at least 2"))

    m = 1 + D + div(D * (D + 1), 2)
    cap = max_neighbors === nothing ? DEFAULT_MAX_NEIGHBORS_FACTOR * m : max_neighbors
    cap >= m || throw(ArgumentError("max_neighbors=$cap is smaller than the feature dimension m=$m"))
    initial_pool = pool_size === nothing ? cap : pool_size
    initial_pool >= m || throw(ArgumentError("pool_size must be at least m=$m"))

    n_available = exclude === nothing ? n_samples : n_samples - length(exclude)
    cache = _matching_kdtree_cache(X, kdtree_cache)
    requested = min(initial_pool, n_available)
    expansions = 0
    best_mu::Union{Nothing,Float64} = nothing
    best_radius::Union{Nothing,Float64} = nothing
    best_size = 0
    current_pool = 0

    while true
        candidates = _candidate_pool(X, x_query, requested, cache; exclude=exclude)
        current_pool = length(candidates)
        current_pool < m && _poised_failure(
            "no poised neighbour set: need at least $m distinct non-coincident samples, found $current_pool",
            threshold, best_mu, best_radius, best_size, current_pool, expansions)

        # Proposal: column-pivoted QR of Φ_𝒞ᵀ at the pool radius.
        pool_radius = maximum(norm(x_query .- @view(X[idx, :])) for idx in candidates)
        Φ_C, _ = feature_matrix(X, x_query, candidates; radius=pool_radius)
        pivots = qr(permutedims(Φ_C), ColumnNorm()).p
        selected = collect(candidates[pivots[1:m]])

        σ = svdvals(first(feature_matrix(X, x_query, selected)))
        if minimum(σ) > rank_tol * maximum(σ)
            while true
                mu, r, _ = poisedness(X, x_query, selected)
                if best_mu === nothing || mu > best_mu
                    best_mu, best_radius, best_size = mu, r, length(selected)
                end
                if mu >= threshold
                    diagnostics = NeighborDiagnostics(:poised, threshold, mu, r,
                                                      length(selected), current_pool, expansions)
                    return NeighborSelection(sort(selected), diagnostics)
                end
                length(selected) >= min(cap, current_pool) && break
                remaining = setdiff(candidates, selected)
                isempty(remaining) && break
                best_candidate = first(remaining); best_candidate_mu = -Inf
                for candidate in remaining
                    trial_mu, _, _ = poisedness(X, x_query, [selected; candidate])
                    if trial_mu > best_candidate_mu
                        best_candidate, best_candidate_mu = candidate, trial_mu
                    end
                end
                push!(selected, best_candidate)
            end
        end

        requested == n_available && break
        next_requested = min(n_available, max(requested + 1, pool_growth * requested))
        length(_candidate_pool(X, x_query, next_requested, cache; exclude=exclude)) == current_pool && break
        requested = next_requested
        expansions += 1
    end

    mu_text = best_mu === nothing ? "no full-rank proposal" : "best μ=$(best_mu)"
    _poised_failure("no neighbour set satisfies mu0=$threshold ($mu_text)",
                    threshold, best_mu, best_radius, best_size, current_pool, expansions)
end

"""
    _layered_poised_selections(X, x_query; layers, kwargs...) -> (indices, per_layer)

Section 4.2: `layers` passes of Algorithm 1, each excluding the picks of the
earlier passes. A failure in the first pass propagates; a failure in a later
pass (data exhausted) ends the layering cleanly.
"""
function _layered_poised_selections(X::AbstractMatrix{Float64}, x_query::AbstractVector{Float64};
                                    layers::Int, kdtree_cache=nothing, kwargs...)
    layers >= 1 || throw(ArgumentError("layers must be at least 1"))
    n_samples = size(X, 1)
    selected = Int[]
    excluded = Set{Int}()
    per_layer = NeighborSelection[]
    for _ in 1:layers
        sel = try
            select_poised_neighbors(X, x_query; kdtree_cache=kdtree_cache,
                                    exclude=isempty(excluded) ? nothing : excluded, kwargs...)
        catch err
            (!isempty(selected) && err isa NeighborSelectionError) || rethrow()
            break
        end
        append!(selected, sel.indices)
        push!(per_layer, sel)
        union!(excluded, sel.indices)
        length(excluded) >= n_samples && break
    end
    return sort(selected), per_layer
end

# ---------------------------------------------------------------------------
# Legacy selectors (not in the manuscript)
# ---------------------------------------------------------------------------

"""
    k_nearest_neighbors(X, x_query, k; kdtree_cache=nothing)

Indices of the `k` nearest non-coincident samples. Not in the manuscript.
"""
function k_nearest_neighbors(X::AbstractMatrix{Float64}, x_query::AbstractVector{Float64}, k::Int;
                             kdtree_cache::Union{Nothing,KDTreeCache}=nothing)
    n_samples = size(X, 1)
    length(x_query) == size(X, 2) || throw(DimensionMismatch("query dimension does not match samples"))
    k >= 1 || throw(ArgumentError("k must be at least 1"))
    cache = _matching_kdtree_cache(X, kdtree_cache)
    query_count = min(n_samples, k + 1)
    neighbors = Int[]
    while true
        idxs, dists = knn(cache.tree, x_query, query_count, true)
        neighbors = idxs[dists .> 0]
        (length(neighbors) >= k || query_count == n_samples) && break
        query_count = min(n_samples, max(query_count + 1, 2 * query_count))
    end
    isempty(neighbors) && throw(ArgumentError("no valid neighbours (all samples coincide with the query)"))
    resize!(neighbors, min(k, length(neighbors)))
    return sort(neighbors)
end

"""
    k_informative_neighbors(X, x_query; layers=1, rtol=1e-10, kdtree_cache=nothing)

Greedy nearest-first selection that adds a sample whenever its curvature
feature λ(a_i) raises the rank of the collected features, until the D_h
second-derivative directions are spanned; `layers > 1` repeats the pass
excluding earlier picks. In 1-D it takes `layers` nearest samples on each
side of the query. Not in the manuscript.
"""
function k_informative_neighbors(X::AbstractMatrix{Float64}, x_query::AbstractVector{Float64};
                                 layers::Int=1, rtol::Float64=1e-10,
                                 kdtree_cache::Union{Nothing,KDTreeCache}=nothing)
    n_samples, D = size(X)
    length(x_query) == D || throw(DimensionMismatch("query dimension does not match samples"))
    layers >= 1 || throw(ArgumentError("layers must be at least 1"))

    if D == 1
        dx = vec(X[:, 1]) .- x_query[1]
        left = findall(<(0), dx); right = findall(>(0), dx)
        left = left[sortperm(abs.(dx[left]))][1:min(layers, length(left))]
        right = right[sortperm(abs.(dx[right]))][1:min(layers, length(right))]
        result = sort(vcat(left, right))
        isempty(result) && throw(ArgumentError("no valid neighbours (all samples coincide with the query)"))
        return result
    end

    cache = kdtree_cache === nothing ? get_kdtree_cache() : kdtree_cache
    if cache !== nothing && _cache_matches(cache, X)
        idxs, dists = knn(cache.tree, x_query, n_samples, true)
        sorted = idxs[dists .> 0]
    else
        d2 = vec(sum(abs2, X .- permutedims(x_query); dims=2))
        valid = findall(>(0), d2)
        sorted = valid[sortperm(d2[valid])]
    end

    D_h = div(D * (D + 1), 2)
    Q = zeros(D_h, D_h); v = zeros(D_h)
    selected_mask = falses(n_samples)
    all_selected = Int[]
    for _ in 1:layers
        rank = 0; fill!(Q, 0.0)
        layer_selected = Int[]
        for idx in sorted
            selected_mask[idx] && continue
            lambda_coeffs!(v, x_query .- @view(X[idx, :]))
            norm_v = norm(v)
            norm_v == 0.0 && continue
            for k in 1:rank
                qk = @view Q[:, k]
                v .-= dot(v, qk) .* qk
            end
            residual = norm(v)
            if residual > rtol * norm_v
                rank += 1
                Q[:, rank] .= v ./ residual
                push!(layer_selected, idx); selected_mask[idx] = true
                rank >= D_h && break
            end
        end
        append!(all_selected, layer_selected)
        isempty(layer_selected) && break
    end
    return sort(all_selected)
end

# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

"""
    select_neighbors(X, x_query; method=:poised, layers=nothing, mu0=DEFAULT_MU0,
                     max_neighbors=nothing, pool_size=nothing, rank_tol=DEFAULT_RANK_TOL,
                     pool_growth=DEFAULT_POOL_GROWTH, param=nothing,
                     return_diagnostics=false, kdtree_cache=nothing)

Select the neighbour set 𝒩 of `x_query` from the `N × D` coordinates `X`.

`method=:poised` is Algorithm 1 run `layers` times (default `DEFAULT_LAYERS = 3`,
Section 4.2; the union of the passes is returned, and the diagnostics report μ
and r_𝒩 of that union). The legacy methods `:knn` (`param` nearest), `:info`
(`param` informative passes) and `:all` (every sample) are not in the
manuscript; they ignore the poisedness controls and take `layers = 1` only.
"""
function select_neighbors(X::AbstractMatrix{Float64}, x_query::AbstractVector{Float64};
                          method::Symbol=:poised,
                          layers::Union{Nothing,Int}=nothing,
                          mu0::Real=DEFAULT_MU0,
                          max_neighbors::Union{Nothing,Int}=nothing,
                          pool_size::Union{Nothing,Int}=nothing,
                          rank_tol::Float64=DEFAULT_RANK_TOL,
                          pool_growth::Int=DEFAULT_POOL_GROWTH,
                          param::Union{Nothing,Int}=nothing,
                          return_diagnostics::Bool=false,
                          kdtree_cache::Union{Nothing,KDTreeCache}=nothing)
    method in NEIGHBOR_METHODS || throw(ArgumentError("method must be one of $(NEIGHBOR_METHODS)"))
    length(x_query) == size(X, 2) || throw(DimensionMismatch("query dimension does not match samples"))
    layers = layers === nothing ? (method == :poised ? DEFAULT_LAYERS : 1) : layers
    layers >= 1 || throw(ArgumentError("layers must be at least 1"))

    if method == :poised
        result = if layers == 1
            select_poised_neighbors(X, x_query; mu0=mu0, max_neighbors=max_neighbors,
                                    pool_size=pool_size, rank_tol=rank_tol,
                                    pool_growth=pool_growth, kdtree_cache=kdtree_cache)
        else
            indices, per_layer = _layered_poised_selections(X, x_query; layers=layers,
                mu0=mu0, max_neighbors=max_neighbors, pool_size=pool_size,
                rank_tol=rank_tol, pool_growth=pool_growth, kdtree_cache=kdtree_cache)
            mu, r, _ = poisedness(X, x_query, indices)
            diagnostics = NeighborDiagnostics(:poised, Float64(mu0), mu, r, length(indices),
                                              sum(s.diagnostics.pool_size for s in per_layer),
                                              sum(s.diagnostics.pool_expansions for s in per_layer))
            NeighborSelection(indices, diagnostics)
        end
        return return_diagnostics ? result : result.indices
    end

    layers == 1 || throw(ArgumentError("layers is only supported for method=:poised"))
    indices = if method == :all
        d2 = vec(sum(abs2, X .- permutedims(x_query); dims=2))
        idx = findall(>(0), d2)
        isempty(idx) && throw(ArgumentError("no valid neighbours"))
        idx
    elseif method == :knn
        k_nearest_neighbors(X, x_query, param === nothing ? DEFAULT_KNN : param; kdtree_cache=kdtree_cache)
    else
        k_informative_neighbors(X, x_query; layers=param === nothing ? DEFAULT_INFO_LAYERS : param,
                                kdtree_cache=kdtree_cache)
    end
    r = maximum(norm(x_query .- @view(X[i, :])) for i in indices)
    diagnostics = NeighborDiagnostics(method, nothing, nothing, r, length(indices), length(indices), 0)
    return return_diagnostics ? NeighborSelection(indices, diagnostics) : indices
end
