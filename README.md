# Diona

**Direct Inference on Admissible Sets**: data-driven scalar regression with
admissible-set bounds, in Julia.

Companion code to

> L. Sabug Jr. and E. Kerrigan, *Direct Inference on Admissible Sets (Diona)*, 2026 (under review).

Diona answers a query without training a model. For a query point `x'` it selects
a poised set of neighbouring samples, writes the second-order Taylor relations
between the unknown state triple `(u, ∇u, vech ∇²u)` at the query and at each
neighbour as linear inequalities (a Lipschitz-continuous Hessian bounds the
remainders), adds the measurement bands, and solves two small linear programs
for the smallest and largest admissible value of `u(x')`. The estimate is the
midpoint and the interval width is the uncertainty measure. Section numbers in
the docstrings refer to the manuscript; [docs/manuscript-map.md](docs/manuscript-map.md)
maps every symbol and algorithm line to the code.

## Installation

Diona is not yet registered. From a Julia session:

```julia
using Pkg
Pkg.add(url="https://github.com/lorenzosabugjr/Diona")
```

or clone the repository and `Pkg.develop(path=".")`. The dependencies are
[HiGHS.jl](https://github.com/jump-dev/HiGHS.jl) (linear programs),
NearestNeighbors.jl, and the standard library.

## Quick start

```julia
using Diona

# samples: one row per sample, D coordinates then the measurement z
xs = collect(-1.0:0.1:1.0)
samples = hcat(xs, sin.(2xs) .+ 0.3 .* xs .^ 2)

ε̃ = estimate_noise_bound(samples; x_cols=1:1, z_col=2)      # §4.1, Novara & Fagiano (2016)
r = infer(samples, [0.15]; noise_bound=ε̃, L_H_factor=2.5)    # Algorithm 2

r.status      # :OPTIMAL
r.estimate    # ũ' = (u̲' + ū')/2
r.lower, r.upper
r.width       # w = ū' − u̲'
r.L_H_used    # L̃_H × L_H_factor, the Hessian Lipschitz constant the bounds were solved at
```

`L_H` may be given as a number when the Hessian Lipschitz constant is known a
priori (Assumption 1); the default `L_H=:estimate` solves the LP of §4.1 on the
selected neighbours. `noise_bound` is the noise bound ε̄ of Assumption 2 or its
estimate. The bounds contain the true value whenever both constants are valid
(Lemma 1); with estimated constants they describe the admissible set under
those estimates and guarantee nothing.

Neighbour selection is Algorithm 1 (`select_poised_neighbors`) run three times
(§4.2) with poisedness threshold `mu0 = 1e-6`; see [docs/API.md](docs/API.md)
for every keyword and [docs/algorithm.md](docs/algorithm.md) for the steps and
the list of deviations from the manuscript.

## Tests

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

## Benchmark

`test/ml/` holds the comparative benchmark of the manuscript's Section 6:
Diona against a neural network, a sparse variational Gaussian process,
nonlinear set membership, radial basis functions and inverse distance weighting
on six functions of three to eight dimensions, with and without 20 dB noise.
The campaign configs are in `test/configs/`; `paper/` holds the metric extractor
and the figure scripts. See [test/ml/README.md](test/ml/README.md).

## Layout

- `src/triple.jl`: state triples, `vech`, and the coefficient maps λ(a) and C(a) (Notation, §2.2).
- `src/neighbors.jl`: feature matrix Φ, μ0-poisedness, Algorithm 1, the three-pass union (§2.3, §4.2).
- `src/constraints.jl`: rows (8a)–(8e) and the admissible polytope Ξψ ≤ b (§2.2).
- `src/noise.jl`: the noise-bound estimate ε̃ (§4.1).
- `src/lp.jl`: HiGHS wrapper for the three LPs.
- `src/inference.jl`: the L̃_H LP (§4.1), the bound LPs (11a)–(11b), Algorithm 2.
- `docs/`: API reference, algorithm and deviations, manuscript-to-code map.
- `test/`: unit tests (`Pkg.test()`), the benchmark harness (`test/ml/`) and its campaign configs (`test/configs/`).
- `paper/`: campaign metric extractor and MATLAB figure scripts.

## License

MIT, see [LICENSE](LICENSE). Please cite the manuscript ([CITATION.cff](CITATION.cff)).
