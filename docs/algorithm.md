# Algorithms 1 and 2 as implemented, and deviations from the manuscript

## Algorithm 1 (practical μ0-poised selection), `select_poised_neighbors`

| Line | Manuscript | Code |
|---|---|---|
| 1 | `m ← 1 + D + D_h`; fetch at least `m` closest samples | `_candidate_pool` (KD-tree, nearest-first, coincident samples skipped); initial pool `pool_size` (default `N̄`) |
| 3–4 | build `Φ_𝒞`; full-rank `m`-point proposal by column-pivoted QR on `Φ_𝒞ᵀ` | `feature_matrix` at the pool radius; `qr(Φ_𝒞ᵀ, ColumnNorm()).p[1:m]` |
| 5–6 | rank-deficient and samples exhausted → NeighbourFailure | `rank_tol` test on `σ_min/σ_max`; `NeighborSelectionError` |
| 7–9 | `r_𝒩`, rebuild `Φ_𝒩` at that radius, `μ ← σ_min/√|𝒩|` | `poisedness` |
| 10 | return when `μ ≥ μ0` | `NeighborSelection(indices, diagnostics)` |
| 12–14 | stop at `N̄` or pool exhausted; else add the point maximising `μ` | `max_neighbors`, greedy loop |
| 16–19 | expand `𝒞` or fail | `pool_growth`, `pool_expansions` in the diagnostics |

Section 4.2: `select_neighbors(…; layers=3)` runs Algorithm 1 three times,
each pass excluding earlier picks, and returns the union.

## Algorithm 2 (Diona), `infer(samples, x_query; noise_bound, …)`

| Line | Manuscript | Code |
|---|---|---|
| 1 | noise estimate `ε̃` if `ε̄` is unknown | the caller supplies `noise_bound`, typically `estimate_noise_bound(samples; …)` |
| 2 | `𝒩 ← NeighbourSelection(x', X, μ0)` (Algorithm 1) | `select_neighbors(X, x_query; method=:poised, layers=3, mu0, …)` |
| 3 | `L̃_H` by the LP of §4.1 | `estimate_LH(P; noise_bound)` when `L_H=:estimate`; then `L_H_used = L̃_H × L_H_factor` |
| 4 | assemble `𝒜 = {ψ : Ξψ ≤ b}` over `{x'} ∪ 𝒩` | `admissible_set(P; noise_bound, L_H=L_H_used)` |
| 5 | `u̲' ← min u'`, `ū' ← max u'` | `extremal_values(Ξ, b, 1)` |
| 6 | return `ũ' = (u̲' + ū')/2`, `[u̲', ū']` | `DionaResult(estimate, lower, upper, width, …)` |

Timings are reported per stage (`neighbor_selection`, `LH_estimation`,
`assembly`, `bounds`, `total`).

## Code implementation details

1. **μ0.** We use `mu0 = 1e-6` in every dimension. Assumption 3 only requires `0 < μ0 < μ̄_D`,
   and a small value is the intended operating point (full rank, not optimal
   design).
2. **Candidate retrieval.** A KD-tree gives the nearest-first pool; the
   manuscript does not say how candidates are fetched. Coincident samples
   (zero distance to the query) are excluded, since a zero displacement gives
   no Taylor information and a zero radius.
3. **Rank test.** The QR proposal is accepted as full rank when
   `σ_min(Φ) > rank_tol × σ_max(Φ)` with `rank_tol = 1e-9`.
4. **Default cap.** `N̄ = 2(1 + D + D_h)` and the initial pool equals `N̄`;
   the pool grows by a factor 2. The manuscript leaves `N̄` and the pool
   schedule to the implementation.
5. **L_H LP scaling.** The `L_H` column of the §4.1 LP is scaled by `max ρ` so
   its entries lie in `(0, 1]`; the optimum is unchanged, only its units.
6. **LP solver settings.** HiGHS dual simplex with presolve off, one thread, a
   60 s time limit per bound pair and 10 s for the `L̃_H` LP; the two bound LPs
   share one instance (the maximum restarts from the minimum's basis). Optimal
   values agree with presolve on to 1e-11.
7. **Noise-bound estimator defaults.** Radius schedule from 1 % of the domain
   diameter in steps of 0.5 %; stop at the first radius where any ball holds
   two samples (`min_ball_points = 2`, `min_ball_coverage = 0`). This is the
   Novara & Fagiano (2016) procedure the manuscript cites, with these
   particular constants.
8. **Legacy neighbour selectors.** `method=:knn`, `:info`, `:all` also exist as optional neighbour selection mechanisms and are not in the manuscript.
9. **Statuses.** The manuscript returns `NeighbourFailure` or a result; the code
   also distinguishes `:INFEASIBLE` (empty `𝒜`, i.e. inconsistent constants),
   `:UNBOUNDED` (`𝒜` does not bound `u'`) and `:SOLVE_FAILED`.

The benchmark harness adds its own protocol choices (input normalisation to
`[−1, 1]^D` for every arm, the noise bound estimated on the normalised box,
`L_H_factor = 2.5`); see `test/ml/README.md`.
