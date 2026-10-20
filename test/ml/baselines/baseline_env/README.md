# `baseline_env` -- the `:rbf` / `:idw` (Surrogates.jl) environment

Modelled on `test/ml/nn_env`: a separate Julia project whose only job is to
carry a dependency that cannot live in `test/ml/Project.toml`.

    julia --project=test/ml/baselines/baseline_env -e 'using Pkg; Pkg.instantiate()'
    julia --project=test/ml/baselines/baseline_env test/ml/baselines/surrogates_adapter_tests.jl

## Why a separate environment, and not `Pkg.add("Surrogates")` in `test/ml`

Two independent reasons, both measured:

1. **Version trap.** A fresh resolve of Surrogates gives **v8.0.0**, but
   `Pkg.add` against the *existing* `test/ml/Manifest.toml` resolves to
   **v6.10.0**, because that manifest pins `Zygote v0.6.77` and
   `QuasiMonteCarlo v0.3.11`, and Surrogates v8 excludes both. v6.10.0 carries
   a real bug that v8 fixed: `InverseDistanceSurrogate` tests `val in
   inverSurr.x` for an exact sample hit and returns `NaN` at sample-coincident
   query points on a container mismatch. The naive route therefore installs
   the broken version *silently*.
2. **Provenance.** Touching `test/ml/Manifest.toml` rotates
   `comparative_code_provenance` for `:diona` and `:gp` (both hash
   `WORKER_PROJECT/Manifest.toml`), invalidating every already-computed cell
   of arms that have nothing to do with these baselines. This environment
   keeps that manifest byte-identical.

`[compat] Surrogates = "8"` is pinned in `Project.toml` so a future
`Pkg.update` here cannot drift back to the v6 line without failing loudly.

## Resolved versions (2026-09-20, julia 1.12.5)

    Surrogates        v8.0.0     <- the point of this environment
    QuasiMonteCarlo   v0.4.4     (test/ml has v0.3.11)
    Clarabel          v0.11.1
    Sobol             v1.5.0
    NearestNeighbors  v0.4.29

Clarabel / NearestNeighbors / SparseArrays are here because
`surrogates_adapter_tests.jl` includes `test/ml/common/driver.jl`, which
includes `src/Diona.jl`. Sobol and QuasiMonteCarlo are the driver's own
sampling stack.

## The QuasiMonteCarlo bump does not move the data

`driver.jl` draws its designs with `Sobol.SobolSeq` + `QuasiMonteCarlo
.OwenScramble(base=2, pad=32, rng=MersenneTwister(seed))`. Since this
environment resolves QMC to v0.4.4 while `test/ml` stays at v0.3.11, the
scrambled designs were compared across the two environments directly:

    dim=3 n=600  sha256(design)[1:16] = d4b0c9f27525bdd2   (both)
    dim=6 n=1000 sha256(design)[1:16] = 4a4ac8809eaa4642   (both)

Bit-identical, so an `:rbf` / `:idw` cell run under this environment sees the
same training points as the `:diona` / `:gp` / `:nn` cells run under theirs.
Re-check this if either QMC version moves again -- pairing across arms depends
on it.

## Experimental protocol

Unchanged by this environment: each arm sees the training dataset and nothing
else. `:rbf` and `:idw` are shipped at package defaults verbatim, with no
keywords passed, and they take no constants from any oracle.

## The arm worker

`test/ml/baselines/baseline_arm_worker.jl` is the subprocess entry point the
comparative campaign dispatches to for both arms:

    julia --startup-file=no --project=test/ml/baselines/baseline_env \
          test/ml/baselines/baseline_arm_worker.jl \
          --arm=rbf --fixture=<fixture> --out=<prefix> [--checkpoint=<path>]

It is the structural twin of `test/ml/nn/nn_arm_worker.jl`: same CLI, same two
output files (`<prefix>.points.arrow`, `<prefix>.summary.csv`), same rule that
a refused or thrown fit is recorded as NaN predictions with an
`<ARM>_TRAIN_FAILED` status on every query row rather than as a crashed
process. The one CLI addition is the mandatory `--arm=`, because one worker
serves two arms. `--checkpoint` is optional here and defaults to *not*
checkpointing: `:idw` has no fitting step and `:rbf`'s fit is a single dense
solve, so there is nothing worth serialising unless the caller asks.

**`Arrow` was added to this environment for the worker.** `write_point_fragment`
(`test/ml/common/point_fragments.jl`) is the campaign's fragment format, so
any environment that hosts an arm worker needs it. This does not touch
`test/ml/Manifest.toml`, so the provenance argument above still holds.

The settings hash is computed *in the worker*. `surrogate_settings_string`
folds in the loaded Surrogates.jl version, which reads `"unavailable"` in any
process that lacks the package -- see the WIRING HAZARD note in
`surrogates_adapter.jl`.

### Measured behaviour (2026-09-20, cell 001, ishigami/sobol/500D, n=1500)

    :rbf  fit 0.48 s   rel-L2 0.0524   3000/3000 finite
    :idw  fit 3.1e-5 s rel-L2 0.6717   3000/3000 finite

On cell 035 (ishigami 50^3 grid, n=125,000) `:rbf` is fitted at FLOAT32: the
dense system is 176.1 GiB at Float64 but 87.8 GiB at Float32 (measured), against
a 96 GiB pre-flight budget. The worker picks the precision from `n_train` via
`rbf_settings(dim; n_train=...)`, which MUST match what the parent hashed --
precision is part of the settings string.

The refusal path still exists for a hypothetical tier past even the Float32
budget: the worker exits 0 and writes a complete `RBF_TRAIN_FAILED` fragment.
Such a cell should be excluded by `applicable_arms` rather than relying on that,
since `common_success_query_ids` intersects across expected arms.
