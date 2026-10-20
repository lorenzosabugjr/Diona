# Mathematical formulation

This page describes what the code computes, in accordance with the manuscript's notation. Proofs and theory (Lemmas 1–3, Theorem 1) are in the manuscript. The manuscript-to-code map is in [manuscript-map.md](manuscript-map.md).

## State triples (Notation, §2.2)

At any point `x`,

```
θ = col(u, g, h) ∈ R^{1 + D + D_h},   u = u(x),  g = ∇u(x),  h = vech(∇²u(x)),  D_h = D(D+1)/2.
```

`vech` stacks the lower triangle column by column: `(1,1), (2,1), …, (D,1), (2,2), …, (D,D)` (`vech_pairs(D)`). With the duplication matrix `D_D` (`vec(H) = D_D vech(H)`),

```
λ(a) = ½ D_Dᵀ (a ⊗ a)      ⇒  λ(a)ᵀ h = ½ aᵀ H a,
C(a) = ½ D_Dᵀ (a ⊗ I_D)    ⇒  C(a)ᵀ h = ½ H a.
```

Entry `(r, c)` of `λ(a)` is `½ a_r²` on the diagonal and `a_r a_c` below it;
`C(a)[(r,c), k] = ½(δ_{rk} a_c + δ_{ck} a_r)` for `r ≠ c` and `½ δ_{rk} a_r` for `r = c`
(`lambda_coeffs!`, `c_coeffs!` in `src/triple.jl`, verified against `D_D` in the tests).

## Taylor rows and the admissible set (§2.2)

For the query `x'` and a neighbour `x_i`, with `a_i = x' − x_i`, `r = ‖a_i‖`:

```
(8a)  |u' − (u_i + a_iᵀ g_i + λ_iᵀ h_i)|       ≤ L_H r³/6
(8b)  |u_i − (u' − a_iᵀ g' + λ_iᵀ h')|         ≤ L_H r³/6
(8c)  ‖g' − (g_i + C_iᵀ (h' + h_i))‖_∞         ≤ L_H r²/4
(8d)  ‖h' − h_i‖_∞                            ≤ L_H r
(8e)  |u_i − z_i|                             ≤ ε̄
```

`taylor_pair_rows!` writes (8a)–(8d) as `Q θ' + S θ_i ≤ L_H ρ`, two rows per
absolute value or ∞-norm component: `4 + 2D + 2D_h` rows with
`ρ = (r³/6 ×4, r²/4 ×2D, r ×2D_h)`. Stacking over the neighbours in order, with
`ψ = col(θ', θ_{i₁}, …, θ_{i_N})`, gives `T ψ ≤ L_H ρ` (`taylor_system`) and the
measurement rows `M ψ ≤ [z; −z] + ε̄` (`measurement_system`). The admissible set is

```
𝒜 = {ψ : Ξ ψ ≤ b},   Ξ = [T; M],   b = [L_H ρ; z + ε̄; −z + ε̄]        (eq. 10)
```

(`admissible_set`). `u'` is the first entry of `ψ`.

## Bounds and estimate (eqs. 11, Algorithm 2)

```
u̲' = min_{ψ ∈ 𝒜} u',   ū' = max_{ψ ∈ 𝒜} u',   ũ' = (u̲' + ū')/2,   w = ū' − u̲'.
```

`extremal_values` solves both LPs on one HiGHS instance (dual simplex, presolve
off; the maximum restarts from the basis of the minimum). If either LP is not
optimal the result carries no interval and no estimate.

## Poisedness and neighbour selection (§2.3, Definition 1, Algorithm 1)

For a candidate set at radius `r`, `â_i = a_i / r`, `λ̂_i = λ(a_i)/r² = λ(â_i)`, and

```
φ_i = [1, −â_iᵀ, λ̂_iᵀ],   Φ = [φ_{i₁}ᵀ; …; φ_{i_N}ᵀ],   μ(𝒩) = σ_min(Φ)/√N.
```

`𝒩` is μ0-poised when `μ(𝒩) ≥ μ0` (`feature_matrix`, `poisedness`). Algorithm 1
(`select_poised_neighbors`): nearest-first candidate pool `𝒞` from a KD-tree
(coincident samples excluded); column-pivoted QR of `Φ_𝒞ᵀ` proposes `m = 1 + D + D_h`
points; `μ` is recomputed at the exact radius of the proposal; the candidate that
raises `μ` the most is added until `μ ≥ μ0`, the cap `N̄` or the pool is reached;
the pool expands by `pool_growth` and the loop repeats; `NeighborSelectionError`
when the data are exhausted. Section 4.2: three passes, each excluding the
earlier picks, union returned (`select_neighbors`, `layers = 3`).

## Estimated constants (§4.1)

Noise bound: `estimate_noise_bound` is the set-membership estimate of Novara &
Fagiano (2016): half the mean spread `max z − min z` over balls of growing radius,
stopping at the first radius where balls hold at least two samples.

Hessian Lipschitz constant: with `L_H` appended to `ψ`,

```
L̃_H = min L_H   s.t.   T ψ − ρ L_H ≤ 0,   M ψ ≤ [z; −z] + ε̄,   L_H ≥ 0
```

(`estimate_LH`). The bounds are then solved at `L_H = L̃_H × L_H_factor`
(2.5 in the manuscript's benchmark).
