# MATLAB figures for the six-arm comparative campaign

`make_diona_figs.m` rebuilds every plot from the reporting page as `.fig` files,
one per panel.

```matlab
cd matlab
make_diona_figs                                  % 54 .fig into ./figs
make_diona_figs('SavePNG', true)                 % also 300-dpi .png
make_diona_figs('Visible', 'on')                 % watch them build
make_diona_figs('DataFile', '../other-data.json', ...
                'OutDir',   'figs-other')        % a different campaign
```

Requires **R2019b or newer** (`tiledlayout`, `jsondecode`). `subtitle` (R2020b) and
`exportgraphics` (R2020a) are used when present and fall back otherwise.

## What gets written

| prefix | n | content |
|---|---|---|
| `fig01_rel_l2` | 7 | relative L2, all six arms, log y |
| `fig02_interval_width` | 7 | median relative interval width, the three interval arms, log y |
| `fig03_coverage` | 7 | coverage, linear y from 50%, 95% reference line |
| `fig04_coverage_vs_width` | 7 | coverage against width, x reversed, 18 markers each |
| `fig05_inference_cost` | 7 | mean s/query, value labels |
| `fig06_diona_stage_split` | 7 | solver stages stacked to 100% of the mean |
| `fig07_stage_heatmaps` | 4 | stage cost by function × density, shared log shading |
| `fig08_stage_vs_dim` | 1 | stage cost against input dimension, 500D tier |
| `fig09_training_time` | 7 | training time, log y, value labels |

Named `<prefix>__<function>_<layout>.fig`, e.g. `fig04_coverage_vs_width__piston_sobol.fig`.

## Conventions carried over from the page

These are deliberate and worth knowing before reading a panel:

- **Bars are the noiseless cell; `+` is the same cell at 20 dB.** On `fig02` both are
  the *median* relative width. `fig05` and `fig09` carry no overlay at all, because
  mean query cost and training time are one number per cell, not a distribution.
- **Within an arm the three bars run left to right in increasing density tier**,
  shading light to dark.
- **Panels are ordered by input dimension, not cell id.** `piston` (d=7) holds frozen
  ids 37–42 and would otherwise sort after `rosenbrock` (d=8). The Ishigami grid block
  leads, being a layout variant of the d=3 function rather than a dimension of its own.
- **Axis decades are fixed per set, not fitted per panel**, so panels living in separate
  files stay comparable by eye — likewise the one log colour scale shared across all
  four heat panels. `fig04` is the deliberate exception: it widens per panel rather than
  clipping a marker, and its axis labels say which frame it is on.
- **`diona` is drawn as `0` on `fig09`.** It has no training phase at all, which a log
  axis cannot represent.
- **Read width next to coverage.** An interval wider than the output range covers
  everything by construction; where relative width approaches or exceeds 1, coverage
  carries no information.

## Data

`../data.json` is the extractor output for `results/data-6arm-5000D-20260923`
(252 arm-cells, 6,480,000 held-out predictions, factor 2.5). Regenerating it after a
new campaign is a matter of re-running the extractor and copying the result here; the
script reads it and needs no other change.

The script warns if any value falls outside its set's fixed decades, rather than
silently drawing the bar at the axis edge the way the HTML page does. On the current
data nothing is clamped.
