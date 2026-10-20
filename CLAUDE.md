# Repository guidance

Diona is the companion code of the manuscript *Direct Inference on Admissible
Sets (Diona)* (Sabug Jr. and Kerrigan; source in `~/code/manuscript-Diona/`).
The manuscript is the authority for names, symbols and algorithm steps; code
and docs cite its sections. Anything the code does that the manuscript does not
describe is listed under "Deviations" in `docs/algorithm.md` and must stay listed.

- `src/`: `triple.jl` (θ, vech, λ, C), `neighbors.jl` (Φ, μ0-poisedness, Algorithm 1,
  three passes), `constraints.jl` (rows 8a–8e, Ξψ ≤ b), `noise.jl` (ε̃), `lp.jl`
  (HiGHS), `inference.jl` (L̃_H LP, bound LPs, Algorithm 2 = `infer`).
- One algorithm only: a star of Taylor relations between the query and its
  neighbours, the L̃_H LP, two bound LPs, midpoint estimate. No graph depth, no
  nominal-only mode, no QP, no L_H escalation ladder. Do not reintroduce them.
- `:knn`, `:info`, `:all` neighbour selectors are legacy extras, not in the
  manuscript; the benchmark uses `:poised` only.
- Run `julia --project -e 'using Pkg; Pkg.test()'` before and after changes to `src/`.
- Benchmark harness: `test/ml/` (see its README), campaign configs in
  `test/configs/`. Editing `test/ml/harness/cells.jl`, `arms/common.jl`,
  `arms/dispatch.jl`, `campaign.jl`, `fixture.jl`, `registry.jl`,
  `test/smoke_data.jl` or `test/ml/Project.toml`/`Manifest.toml` rotates the
  code provenance of EVERY arm; `--report` accepts recorded provenance and
  tabulates it, but a resume does not.
- Estimated constants certify nothing: never describe an interval as certified,
  guaranteed or rigorous. "Admissible-set bounds under estimated constants."
- `results/`, `plans/`, `logs/`, `test/sandbox/` are ignored local research
  artefacts. Do not delete them as part of cleanup.
