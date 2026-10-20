# Diona API

```julia
using Diona
```

## Inference

```julia
infer(samples, x_query; noise_bound, L_H=:estimate, L_H_factor=1.0,
      method=:poised, layers=nothing, mu0=DEFAULT_MU0,
      max_neighbors=nothing, pool_size=nothing, rank_tol=DEFAULT_RANK_TOL,
      pool_growth=2, param=nothing, verbose=false) -> DionaResult
```

`samples` is an `n × (D+1)` matrix: `D` coordinates then the measurement `z`,
one sample per row. `x_query` has `D` entries.

| Keyword | Default | Manuscript | Meaning |
|---|---|---|---|
| `noise_bound` | required | `ε̄` (or `ε̃`) | half-width of the measurement band (8e) |
| `L_H` | `:estimate` | `L_H` / `L̃_H` | a number, or `:estimate` for the §4.1 LP on the selected neighbours |
| `L_H_factor` | `1.0` | "scaled by 2.5" | multiplier applied to `L_H` or `L̃_H` before the bound LPs |
| `method` | `:poised` | Algorithm 1 | `:poised`; legacy `:knn`, `:info`, `:all` |
| `layers` | `3` for `:poised`, else `1` | §4.2 three passes | passes of Algorithm 1 whose union is used |
| `mu0` | `1e-6` | `μ0` | poisedness threshold |
| `max_neighbors` | `2(1 + D + D_h)` | `N̄` | cap on the selected set per pass |
| `pool_size` | `max_neighbors` | `|𝒞|` | initial candidate pool |
| `rank_tol` | `1e-9` | "full-rank" | relative `σ_min/σ_max` threshold for the QR proposal |
| `pool_growth` | `2` | "expand 𝒞" | pool growth factor |
| `param` | `nothing` | – | `k` for `:knn`, passes for `:info` |

`DionaResult` fields: `estimate` (`ũ'`), `lower` (`u̲'`), `upper` (`ū'`), `width`
(`w`), `L_H_used`, `noise_bound`, `n_neighbors` (`N`), `radius` (`r_𝒩`),
`status`, `timings::DionaTimings`, `neighbor_diagnostics::NeighborDiagnostics`.
On any failure the scalar fields are `nothing`.

| Status | Meaning |
|---|---|
| `:OPTIMAL` | both bound LPs solved |
| `:NEIGHBOR_FAILED` | Algorithm 1 found no μ0-poised set |
| `:INFEASIBLE` | `𝒜` is empty: the noise and smoothness constants contradict the data |
| `:UNBOUNDED` | `𝒜` does not bound `u'` |
| `:SOLVE_FAILED` | LP time limit or numerical failure |

`is_accepted(r)`, `has_interval(r)`. `DionaTimings`: `neighbor_selection`,
`LH_estimation`, `assembly`, `bounds`, `total` (seconds).

## Constants

```julia
estimate_noise_bound(samples; x_cols, z_col, rho_start_fraction=0.01,
                     rho_increment_fraction=0.005, min_ball_points=2,
                     min_ball_coverage=0.0) -> ε̃
estimate_LH(P::ActiveSet; noise_bound) -> (L̃_H, status)
```

## Admissible set

```julia
P = ActiveSet(samples, x_query, indices)        # P = {x'} ∪ 𝒩
Ξ, b = admissible_set(P; noise_bound, L_H)      # 𝒜 = {ψ : Ξψ ≤ b}
lower, upper, status = admissible_bounds(P; noise_bound, L_H)
T, ρ = taylor_system(P)                          # T ψ ≤ L_H ρ
M, zs = measurement_system(P)                    # M ψ ≤ zs + ε̄
Q, S, ρ_i = taylor_pair_rows(a, TripleLayout(D)) # one neighbour, Q θ' + S θ_i ≤ L_H ρ_i
n_neighbors(P), radius(P)
```

## State triples

```julia
L = TripleLayout(D)            # L.D, L.D_h, L.n, L.u, L.g, L.h
vech(H), unvech(h, D), vech_pairs(D), duplication_matrix(D)
lambda_coeffs(a), c_coeffs(a)  # λ(a) and C(a); in-place forms lambda_coeffs!, c_coeffs!
state_triple(u, g, H)
```

## Neighbour selection

```julia
select_neighbors(X, x_query; method=:poised, layers=nothing, mu0=DEFAULT_MU0, …,
                 return_diagnostics=false) -> indices or NeighborSelection
select_poised_neighbors(X, x_query; mu0, max_neighbors, pool_size, rank_tol,
                        pool_growth, kdtree_cache, exclude) -> NeighborSelection
feature_matrix(X, x_query, indices; radius=nothing) -> (Φ, r)
poisedness(X, x_query, indices) -> (μ, r, σ_min)
k_nearest_neighbors(X, x_query, k), k_informative_neighbors(X, x_query; layers=1)
```

`X` is the `n × D` coordinate block of `samples`. `NeighborDiagnostics` fields:
`method`, `mu0`, `mu`, `radius`, `n_selected`, `pool_size`, `pool_expansions`,
`indices`. `NeighborSelectionError` carries the diagnostics of the best
attempt.

KD-tree cache for repeated queries against one data set:

```julia
build_kdtree_cache!(X; trusted=false); get_kdtree_cache(); clear_kdtree_cache!()
```

`trusted=true` skips the per-query content hash (`coordinate_fingerprint`);
use it only while `X` is not mutated.
