# Structural gates over cell enumeration, data loading and the single-cell proof.
#
# Extracted from test/smoke_data.jl. These are plain `include`d fragments, NOT a module:
# every name stays top-level in the including scope, because test/run_full_benchmark.jl
# reaches into the harness with getproperty(harness, :name). Included at the END of
# smoke_data.jl, so everything referenced here is already defined.

function check_gate_cell_enumeration(; verbose=true)
    cells = enumerate_cells()

    n_sobol    = count(c -> c.layout == :sobol, cells)
    n_grid     = count(c -> c.layout == :grid,  cells)
    total      = length(cells)
    # DERIVED from the function table, not a literal: every Sobol' function
    # contributes |N_TIERS_PER_DIM| x |NOISE_LEVELS| cells, and the grid block
    # contributes |ISHIGAMI_GRID_SIDES| x |NOISE_LEVELS|. A hardcoded 30 meant
    # that adding :piston failed this gate for counting correctly.
    per_fn      = length(N_TIERS_PER_DIM) * length(NOISE_LEVELS)
    want_sobol  = length(BENCHMARK_FUNCTIONS) * per_fn
    want_grid   = length(ISHIGAMI_GRID_SIDES) * length(NOISE_LEVELS)
    want_total  = want_sobol + want_grid

    if verbose
        println("Cell enumeration gate — Cell enumeration")
        println("  Sobol             : $n_sobol  (expected $want_sobol)")
        println("  Grid              : $n_grid   (expected $want_grid)")
        println("  ─────────────────────────────────")
        println("  Total             : $total   (expected $want_total)")
    end

    ok = (n_sobol == want_sobol) && (n_grid == want_grid) && (total == want_total)
    verbose && println(ok ? "  PASS ✓" : "  FAIL ✗ — subtotals above do not match expected values")
    return ok
end

function check_gate_data_loaders(; verbose=true)
    cells    = enumerate_cells()
    all_pass = true

    verbose && println("Data loaders gate — Data loaders")
    verbose && println("Checking X_train shape for all $(length(cells)) cells...")

    for cell in cells
        try
            X_train = load_training(cell)
            ok_dim    = size(X_train, 1) == cell.dim
            ok_ntrain = size(X_train, 2) == cell.n_train
            expected_str = string(cell.n_train)
            if !(ok_dim && ok_ntrain)
                verbose && @printf("  FAIL ✗  %-16s %-14s  got (%d×%d)  expected (%d×%s)\n",
                    cell.name, cell.tier_label,
                    size(X_train, 1), size(X_train, 2), cell.dim, expected_str)
                all_pass = false
            end
        catch e
            verbose && println("  ERROR ✗  $(cell.name)/$(cell.layout)/$(cell.tier_label): $e")
            all_pass = false
        end
    end

    all_pass && verbose && println("  All $(length(cells)) cells: X_train shape OK ✓")
    verbose && println("\nDense query-set spot-checks:")
    spot_cells = [
        findfirst(c -> c.name == :ishigami   && c.layout == :sobol, cells),
        findfirst(c -> c.name == :rosenbrock && c.layout == :sobol, cells),
    ]
    for idx in spot_cells
        idx === nothing && continue
        cell = cells[idx]
        try
            X_q, y_q, or_val = query_set_for(cell)
            n_q = length(y_q)
            expected_n = N_QUERY(_BENCH_FN[cell.name])
            ok = n_q == expected_n && size(X_q, 2) == expected_n
            all_pass &= ok
            verbose && @printf("  %-6s %-16s  output_range=%.4f  query_size=%d  %s\n",
                cell.layout, cell.name, or_val, n_q,
                ok ? "✓" : "✗ (expected $expected_n)")
        catch e
            verbose && println("  ERROR ✗  query_set_for($(cell.name)): $e")
            all_pass = false
        end
    end

    verbose && println()
    verbose && println("Data loaders gate overall: " * (all_pass ? "PASS ✓" : "FAIL ✗ — see above"))
    return all_pass
end

function check_gate_single_cell_proof(; verbose::Bool=true)
    cells = enumerate_cells()
    anchor_idx = findfirst(cells) do c
        c.name        == _SINGLE_CELL_ANCHOR_NAME    &&
        c.layout      == _SINGLE_CELL_ANCHOR_LAYOUT  &&
        c.n_train     == _SINGLE_CELL_ANCHOR_NTRAIN  &&
        c.noise_level == _SINGLE_CELL_ANCHOR_NOISE
    end
    anchor_idx === nothing && error("single-cell proof anchor cell not found in enumerate_cells()")
    cell = cells[anchor_idx]

    verbose && println("Single-cell proof gate — Single-cell end-to-end proof")
    verbose && @printf("  Anchor: %s / %s / n_train=%d / noise_level=%.2f\n",
                       cell.name, cell.layout, cell.n_train, cell.noise_level)
    verbose && println("  Running solve (may take ~1–5 min for 50 query points)...")
    flush(stdout)

    t_wall = @elapsed cr = run_cell(cell; verbose=false)

    rel_l2   = _cell_rel_l2(cr)
    coverage = _cell_coverage(cr)
    sc       = _cell_status_counts(cr)
    n_opt    = get(sc, :OPTIMAL, 0)
    uq_mask  = isfinite.(cr.lower) .& isfinite.(cr.upper)
    mean_w   = count(uq_mask) > 0 ? mean(cr.upper[uq_mask] .- cr.lower[uq_mask]) : NaN
    w_2eps   = isfinite(mean_w) ? mean_w / (2 * cr.noise_bound_used) : NaN
    true_hw  = _SINGLE_CELL_ANCHOR_NOISE * output_range(_BENCH_FN[_SINGLE_CELL_ANCHOR_NAME])

    if verbose
        println()
        println("  Results vs precedent artifact (2026-08-15, three poised passes):")
        @printf("  %-22s  this run: %-10s  precedent: %.4f\n",
                "rel-L2",
                isfinite(rel_l2)   ? @sprintf("%.4f", rel_l2)   : "NaN",
                _SINGLE_CELL_REF_REL_L2)
        @printf("  %-22s  this run: %-10s  precedent: %.3f\n",
                "coverage",
                isfinite(coverage) ? @sprintf("%.3f", coverage) : "NaN",
                _SINGLE_CELL_REF_COVERAGE)
        @printf("  %-22s  this run: %d/%d  precedent: 50/50\n",
                "OPTIMAL count", n_opt, cr.n_query)
        @printf("  %-22s  %.6f\n", "mean width", isnan(mean_w) ? NaN : mean_w)
        @printf("  %-22s  %.2f\n",  "width/2ε",  isnan(w_2eps) ? NaN : w_2eps)
        @printf("  %-22s  noise_bound=%.4e  true half_width=%.4e\n",
                "noise_bound_used", cr.noise_bound_used, true_hw)
        @printf("  %-22s  mean=%.1f ms  max=%.1f ms\n",
                "timing",
                1000 * mean(cr.elapsed), 1000 * maximum(cr.elapsed))
        println("  status counts: ",
                join(["$k=$v" for (k, v) in sort(collect(sc); by=x -> string(x[1]))], "  "))
        @printf("  total wall time: %.1f s\n", t_wall)
        println()
    end

    ok = true
    if !isfinite(rel_l2)
        verbose && println("  FAIL ✗  rel-L2 is non-finite")
        ok = false
    elseif !(0.01 <= rel_l2 <= 2.0)
        verbose && @printf("  FAIL ✗  rel-L2=%.4f outside [0.01, 2.0] — wrong order of magnitude\n", rel_l2)
        ok = false
    end
    if !isfinite(coverage) || coverage < 0.80
        verbose && @printf("  FAIL ✗  coverage=%.3f < 0.80 — unexpectedly low\n",
                           isfinite(coverage) ? coverage : NaN)
        ok = false
    end

    verbose && println(ok ? "  Single-cell proof gate: PASS ✓" : "  Single-cell proof gate: FAIL ✗")
    return ok, cr
end

function check_gate_metrics_aggregation(; cr::Union{CellResult,Nothing}=nothing, verbose::Bool=true)
    verbose && println("Metrics aggregation gate — Metrics aggregation (definition check)")

    if cr === nothing
        verbose && println("  No pre-computed CellResult supplied; re-running anchor cell...")
        flush(stdout)
        cells = enumerate_cells()
        anchor_idx = findfirst(cells) do c
            c.name == _SINGLE_CELL_ANCHOR_NAME &&
            c.layout == _SINGLE_CELL_ANCHOR_LAYOUT && c.n_train == _SINGLE_CELL_ANCHOR_NTRAIN &&
            c.noise_level == _SINGLE_CELL_ANCHOR_NOISE
        end
        anchor_idx === nothing && error("metrics aggregation anchor cell not found in enumerate_cells()")
        cr = run_cell(cells[anchor_idx]; verbose=false)
    end

    m            = compute_cell_metrics(cr)
    ref_rel_l2   = _cell_rel_l2(cr)
    ref_coverage = _cell_coverage(cr)

    ok = (m.rel_l2 === ref_rel_l2) && (m.coverage === ref_coverage)

    if verbose
        @printf("  %-22s  compute_cell_metrics=%-8s  private=%-8s  %s\n",
                "rel_l2",
                isfinite(m.rel_l2)   ? @sprintf("%.4f", m.rel_l2)   : "NaN",
                isfinite(ref_rel_l2) ? @sprintf("%.4f", ref_rel_l2) : "NaN",
                m.rel_l2 === ref_rel_l2 ? "✓" : "✗ MISMATCH")
        @printf("  %-22s  compute_cell_metrics=%-8s  private=%-8s  %s\n",
                "coverage",
                isfinite(m.coverage)   ? @sprintf("%.3f", m.coverage)   : "NaN",
                isfinite(ref_coverage) ? @sprintf("%.3f", ref_coverage) : "NaN",
                m.coverage === ref_coverage ? "✓" : "✗ MISMATCH")
        println()
        println("  Full metrics:")
        @printf("    rel_l2          = %s\n",   isfinite(m.rel_l2)        ? @sprintf("%.4f",  m.rel_l2)        : "NaN")
        @printf("    coverage        = %s\n",   isfinite(m.coverage)      ? @sprintf("%.3f",  m.coverage)      : "NaN")
        @printf("    mean_width      = %s\n",   isfinite(m.mean_width)    ? @sprintf("%.6f", m.mean_width)    : "NaN")
        @printf("    width_over_2eps = %s\n",   isfinite(m.width_over_2eps) ? @sprintf("%.2f", m.width_over_2eps) : "NaN")
        @printf("    t_mean          = %.1f ms\n", 1000 * m.t_mean)
        @printf("    t_median        = %.1f ms\n", 1000 * m.t_median)
        @printf("    t_max           = %.1f ms\n", 1000 * m.t_max)
        println("    status_counts   = ",
                join(["$k=$v" for (k, v) in sort(collect(m.status_counts); by=x -> string(x[1]))], "  "))
        println()
    end

    verbose && println(ok ? "  Metrics aggregation gate: PASS ✓" : "  Metrics aggregation gate: FAIL ✗ — metric definition mismatch, fix compute_cell_metrics")
    return ok
end
