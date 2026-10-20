# Paper artefacts

Everything needed to go from a finished campaign (`test/ml/README.md`) to the
figures of the manuscript's Section 6.

| file | role |
|---|---|
| `extract_campaign.jl` | reads `<run.out>/comparative_fragments/runs`, writes `data.json`: one row per arm-cell with `ε_L2`, `ρ`, median relative width (two denominators, see below), per-query and training times, and Diona's per-stage times |
| `data.json` | the extractor's output, written here by the command below; a result, not tracked |
| `matlab/make_diona_figs.m` | draws every panel from `data.json` (MATLAB R2019b+), see `matlab/README.md` |

```sh
julia --project=test/ml paper/extract_campaign.jl \
    results/<campaign>/comparative_fragments/runs paper/data.json
```

If one out dir holds two campaigns of an arm (two settings hashes), name the
one to keep: `diona=<hash prefix> nsm=<hash prefix>`.

**Relative width.** The manuscript defines `w_r(q) = w(q) / |u(q)|`; the figures
of the reported campaign used `w(q) / range(u)` with `range` the spread of `u`
over its domain. `data.json` carries both (`median_rel_width_abs_u` and
`median_rel_width`), so whichever the manuscript settles on can be plotted.

**Provenance.** `data.json` records each arm-cell's `settings_hash` and
`code_provenance`, so a figure can always be traced to the code that produced
its numbers.
