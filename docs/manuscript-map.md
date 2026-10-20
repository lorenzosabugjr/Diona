# Manuscript-to-code map

The manuscript (*Direct Inference on Admissible Sets (Diona)*, Sabug Jr. and
Kerrigan) is the authority for every name below. Section and equation numbers
refer to it.

## Symbols

| Manuscript | Meaning | Code |
|---|---|---|
| `x'` | query point | `x_query` |
| `(x^(i), z^(i))` | sample location and (noisy) measurement | row `i` of `samples` (`D` coordinates, then `z`) |
| `a_i = x' − x^(i)` | displacement | `a` in `taylor_pair_rows!`, `displacements` in `feature_matrix` |
| `u, g, h` | value, gradient, `vech` of the Hessian | `TripleLayout` ranges `u`, `g`, `h` |
| `D_h = D(D+1)/2` | length of `h` | `TripleLayout.D_h` |
| `θ = col(u, g, h)` | state triple | `state_triple(u, g, H)`; `TripleLayout.n = 1 + D + D_h` |
| `λ_i = ½ D_Dᵀ (a_i ⊗ a_i)` | `λ_iᵀ h = ½ a_iᵀ H a_i` | `lambda_coeffs(a)` |
| `c_i = ½ D_Dᵀ (a_i ⊗ I_D)` | `c_iᵀ h = ½ H a_i` | `c_coeffs(a)` (a `D_h × D` matrix) |
| `D_D` | duplication matrix | `duplication_matrix(D)` (tests only) |
| `L_H` | Hessian Lipschitz constant (Assumption 1) | keyword `L_H` of `infer`; `L_H_used` in the result |
| `ε̄` | noise bound (Assumption 2) | keyword `noise_bound` |
| `𝒩`, `N` | neighbour set and its size | `ActiveSet.indices`, `n_neighbors` |
| `P = {x'} ∪ 𝒩` | active points | `ActiveSet` |
| `ψ = col(θ', θ^(i₁), …, θ^(i_N))` | decision vector | columns of `Ξ`; `triple_columns(L, k)` |
| `Ξ ψ ≤ b`, `𝒜` | admissible polytope (eq. 10) | `admissible_set(P; noise_bound, L_H)` |
| `u̲', ū'` | bounds (eqs. 11a, 11b) | `admissible_bounds`, `extremal_values` |
| `ũ' = (u̲' + ū')/2`, `w = ū' − u̲'` | estimate and uncertainty measure | `DionaResult.estimate`, `DionaResult.width` |
| `φ_i = [1, −â_iᵀ, λ̂_iᵀ]`, `Φ` | normalised feature rows (§2.3) | `feature_matrix(X, x_query, indices)` |
| `μ(𝒩) = σ_min(Φ)/√N`, `μ0` | poisedness and its threshold (Definition 1) | `poisedness`, keyword `mu0` (`DEFAULT_MU0 = 1e-6`) |
| `r_𝒩` | neighbourhood radius | `radius(P)`, `NeighborDiagnostics.radius` |
| `N̄` | maximum selected neighbours (Algorithm 1) | keyword `max_neighbors` (default `2(1 + D + D_h)`) |
| `𝒞` | candidate pool | `_candidate_pool`; keywords `pool_size`, `pool_growth` |
| `ε̃` | estimated noise bound (§4.1) | `estimate_noise_bound` |
| `L̃_H` | estimated Hessian Lipschitz constant (§4.1) | `estimate_LH`; `L_H=:estimate` |
| factor 2.5 (Table 1) | safety multiplier on `L̃_H` | keyword `L_H_factor` |

## Equations

| Manuscript | Code |
|---|---|
| (8a)–(8d), Taylor rows for one neighbour | `taylor_pair_rows!` in `src/constraints.jl` (`Q θ' + S θ_i ≤ L_H ρ`) |
| (8e), measurement band | `measurement_system` |
| (9), (10), stacking into `Ξ ψ ≤ b` | `taylor_system`, `admissible_set` |
| (11a), (11b) | `extremal_values` (`src/lp.jl`), `admissible_bounds` |
| §4.1, `min L_H s.t. Ξ_P [ψ; L_H] ≤ b_P` | `estimate_LH` |

## Algorithms

| Manuscript | Code |
|---|---|
| Algorithm 1, practical μ0-poised selection | `select_poised_neighbors` (`src/neighbors.jl`) |
| §4.2, three passes of Algorithm 1, union | `select_neighbors(…; layers=3)` (`DEFAULT_LAYERS`) |
| Algorithm 2, Diona | `infer` (`src/inference.jl`) |
| Table 1, "LPs are computed with HiGHS.jl" | `src/lp.jl` |

## Benchmark (Section 6)

| Manuscript | Code |
|---|---|
| six functions, `500D/1000D/2000D` Sobol' tiers, 20 dB Gaussian noise | `test/ml/common/driver.jl`, `test/ml/harness/cells.jl` |
| `5000D` Sobol' query points | `run.query_multiplier: 5000` in `test/configs/*.yaml` |
| `ε_L2`, `w_r`, `ρ` | `paper/extract_campaign.jl` |
| Table 1 settings of the comparators | `test/ml/common/registry.jl`, `test/ml/baselines/`, `test/ml/harness/arms/` |
| Figures 2–4 | `paper/matlab/make_diona_figs.m` |
