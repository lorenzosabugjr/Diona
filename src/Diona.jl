"""
    Diona

Direct Inference on Admissible Sets: data-driven scalar regression with
admissible-set bounds from Taylor relations and a Hessian Lipschitz assumption.

Companion code to Sabug Jr. and Kerrigan, *Direct Inference on Admissible
Sets (Diona)*. Section numbers in docstrings refer to that manuscript.

    result = infer(samples, x_query; noise_bound=ε̃)
    result.estimate, result.lower, result.upper, result.width
"""
module Diona

using LinearAlgebra
using SparseArrays
using Statistics: mean
using NearestNeighbors
import HiGHS

include("triple.jl")        # θ = col(u, g, h), λ(a), C(a)          (Notation, §2.2)
include("neighbors.jl")     # Φ, μ0-poisedness, Algorithm 1           (§2.3, §4.2)
include("constraints.jl")   # rows (8a)–(8e), 𝒜 = {ψ : Ξψ ≤ b}        (§2.2)
include("noise.jl")         # ε̃                                       (§4.1)
include("lp.jl")            # HiGHS                                   (Table 1)
include("inference.jl")     # L̃_H, bounds, Algorithm 2                (§4, §5)

# Main entry point and result types
export infer, DionaResult, DionaTimings, DIONA_STATUSES, is_accepted, has_interval

# Constants and admissible set
export estimate_noise_bound, estimate_LH, admissible_bounds
export ActiveSet, admissible_set, taylor_system, measurement_system, taylor_pair_rows
export n_neighbors, radius

# State triples
export TripleLayout, vech, unvech, vech_pairs, duplication_matrix
export lambda_coeffs, lambda_coeffs!, c_coeffs, c_coeffs!, state_triple

# Neighbour selection
export select_neighbors, select_poised_neighbors, feature_matrix, poisedness
export k_nearest_neighbors, k_informative_neighbors
export NeighborSelection, NeighborDiagnostics, NeighborSelectionError
export DEFAULT_MU0, DEFAULT_LAYERS, DEFAULT_RANK_TOL, NEIGHBOR_METHODS
export build_kdtree_cache!, get_kdtree_cache, clear_kdtree_cache!, KDTreeCache, coordinate_fingerprint

end
