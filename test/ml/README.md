# Comparative benchmark: Diona, NN, GP, NSM, RBF, IDW

The campaign of the manuscript's Section 6. `test/smoke_data.jl` owns dataset
preparation, paired fixtures, arm dispatch, reporting and resumable workers;
`test/benchmark_data.jl` is the config-driven runner over that machinery;
`test/run_full_benchmark.jl` checks environments, runs the whole comparison and
writes a run manifest.

Six arms on six functions (Ishigami D=3, Park D=4, Friedman D=5, Hartmann D=6,
Piston D=7, Rosenbrock D=8), Sobol' data sets of `500D`, `1000D`, `2000D` samples,
noiseless and with 20 dB Gaussian noise, plus an Ishigami grid block:

| arm | model | interval kind |
| --- | --- | --- |
| `:diona` | Algorithm 2, `L_H_factor = 2.5`, `mu0 = 1e-6`, three passes of Algorithm 1, HiGHS | `:admissible_set` |
| `:nn` | tanh MLP (Lux), `ceil(D/2)+1` layers of 64 units, Adam 25 000 epochs, Float32, CPU or CUDA | `:none` |
| `:gp` | sparse variational GP, Matérn-5/2, `100D` inducing points, ±2σ | `:variational_approximate` |
| `:nsm` | nonlinear set membership (Milanese & Novara), ε̃ and γ̃ as in Novara & Fagiano, γ̃ × 2.5 | `:set_membership_estimated` |
| `:rbf` | Surrogates.jl `RadialBasis`, linear kernel, Float32 | `:none` |
| `:idw` | Surrogates.jl `InverseDistanceSurrogate`, `p = D + 2` | `:none` |

**Protocol.** Every arm sees the training data set and nothing else: no a priori
Lipschitz constant, no a priori noise bound, no oracle. Each method estimates its
own constants from the noisy training data. The single outside input is the
domain box, used identically by every arm to normalise inputs to `[−1, 1]^D`;
Diona's noise bound is estimated on that normalised box too. Estimated constants
certify nothing, so no arm's interval is a guarantee; the `interval_kind` column
is the whole claim.

**Cells.** 42 cells: 36 Sobol' (6 functions × 3 tiers × 2 noise levels) and 6
Ishigami grid (3 grid sides × 2 noise levels), 42 × 6 = 252 arm-cells. The
manuscript reports the 36 Sobol' cells. Query sets hold `5000D` Sobol' points
disjoint from the training data.

**Precision is per arm and deliberately not uniform.** `:nn` and `:rbf` fit in
Float32; the others in Float64. The fixture, the queries and every reported
value are Float64.

## Environments

```sh
julia --project=test/ml -e 'using Pkg; Pkg.instantiate()'
julia --project=test/ml/nn_env -e 'using Pkg; Pkg.instantiate()'
julia --project=test/ml/baselines/baseline_env -e 'using Pkg; Pkg.instantiate()'
```

`test/ml` holds Diona's dependencies (HiGHS, NearestNeighbors), the GP stack
and `:nsm`; `test/ml/nn_env` holds Lux, CUDA and LuxCUDA; `baseline_env` pins
Surrogates v8.0.0 for `:rbf` and `:idw` (an older version returns NaN at
sample-coincident queries). The harness launches each arm's workers under its
own environment. `--device` takes `cpu` (default) or `cuda` and reaches `:nn`
only.

## Checks that fit no model

```sh
julia --project=test/ml test/smoke_data.jl --gate=cell-enumeration
julia --project=test/ml test/smoke_data.jl --gate=reporting-correctness
julia --project=test/ml test/tests_benchmark_data_config.jl
julia --project=test/ml test/tests_full_benchmark_driver.jl
julia --project=test/ml test/ml/point_fragment_tests.jl
julia --project=test/ml test/ml/tuning/registry_tests.jl
julia --project=test/ml test/ml/model_checkpoint_tests.jl
julia --project=test/ml test/ml/data_only_tests.jl          # small real fits of every arm
julia --project=test/ml/nn_env test/ml/data_only_tests.jl   # the same under the NN environment
julia --project=test/ml test/ml/baselines/nsm_equivalence_tests.jl
julia --project=test/ml/baselines/baseline_env test/ml/baselines/surrogates_adapter_tests.jl
```

One real Diona cell (3 000 queries, about a minute):

```sh
julia --project=test/ml test/smoke_data.jl --gate=single-cell-proof
```

Small real fits of every arm: `--gate=comparative-smoke`;
`--gate=orchestration-resilience` also exercises failure and resume handling.

## Campaign runbook

`test/configs/*.yaml` carve the 252 arm-cells into shards that share one
`run.out`. Fragments are keyed by `(cell_id, arm, fixture_sha256, settings_hash)`
and written through an atomic temp+rename, so shards with disjoint `(cell, arm)`
sets can run concurrently.

```sh
# 1. Warm the fixtures ONCE (fixture_sha256 folds in a timestamp, so two cold
#    starts would write two fixtures and key their fragments differently).
julia --project=test/ml test/benchmark_data.jl --prepare-fixtures test/configs/report-all.yaml

# 2. Launch the shards, all into the same run.out. `jobs × blas_threads = 32`
#    on every shard, so every arm gets the same machine budget.
julia --project=test/ml test/benchmark_data.jl test/configs/diona-ishigami-sobol.yaml
julia --project=test/ml test/benchmark_data.jl test/configs/nsm-all.yaml
# ... and the rest of test/configs/, except report-all.yaml

# 3. Rebuild the canonical aggregate across all six arms.
julia --project=test/ml test/benchmark_data.jl --report test/configs/report-all.yaml
```

`report-all.yaml` is the campaign's claim: `tests_benchmark_data_config.jl`
asserts that the other configs cover exactly its arm-cells, with no gap and no
overlap, and `--report` hard-errors on any pair it claims and cannot read. The
report writes `comparative_summary.csv`, `comparative_problem_summary.csv/.md`
and `comparative_provenance.csv`.

**Code provenance.** Each fragment records a hash of the harness files and the
arm's own runner (`comparative_code_provenance`). A *resume* refuses a fragment
whose recorded hash differs from the current code. A *report* accepts the
recorded hash, requires all fragments of one arm to share it, and tabulates
recorded versus current per arm, so a finished campaign is reported as it was
run after later harness edits. Editing a shared harness file therefore never
invalidates finished arms, but a shard re-run after such an edit refits its
arm from scratch.

**Diona settings hash.** `L_H=:estimate, L_H_factor, mu0=1e-6, layers=3,
solver=HiGHS` (`_diona_settings_string` in `harness/cells.jl`). Any change
rotates every `:diona` fragment; the other arms are untouched.

## Metrics and figures

`paper/extract_campaign.jl` turns the fragments into `paper/data.json` with the
manuscript's metrics per arm-cell: relative `L2` error `ε_L2`, coverage `ρ`,
median relative width `w_r`, mean and median seconds per query, training time,
and Diona's per-stage mean times. `paper/matlab/make_diona_figs.m` draws the
figures from it.
