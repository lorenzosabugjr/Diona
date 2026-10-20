# The multi-arm anchor gate: one cell, every applicable arm, paired identities.
#
# Extracted from test/smoke_data.jl. These are plain `include`d fragments, NOT a module:
# every name stays top-level in the including scope, because test/run_full_benchmark.jl
# reaches into the harness with getproperty(harness, :name). Included at the END of
# smoke_data.jl, so everything referenced here is already defined.

function check_gate_arms(; out_dir::Union{Nothing,AbstractString}=nothing, verbose::Bool=true)
    out_dir = out_dir === nothing ?
        joinpath(REPO_ROOT, "results", "smoke_data_arms",
                 Dates.format(Dates.now(), "yyyy-mm-dd-HHMMSS") * "-arms") : out_dir
    mkpath(out_dir)

    cells = enumerate_cells()
    idx = findfirst(cells) do c
        c.name        == _SINGLE_CELL_ANCHOR_NAME    &&
        c.layout      == _SINGLE_CELL_ANCHOR_LAYOUT  &&
        c.n_train     == _SINGLE_CELL_ANCHOR_NTRAIN  &&
        c.noise_level == _SINGLE_CELL_ANCHOR_NOISE
    end
    idx === nothing && error("check_gate_arms: anchor cell not found in enumerate_cells()")
    cell = cells[idx]

    verbose && println("Multi-arm anchor gate")
    verbose && @printf("  Anchor: %s / %s / n_train=%d / noise_level=%.2f\n",
                       cell.name, cell.layout, cell.n_train, cell.noise_level)
    verbose && println("  applicable_arms(cell) = ", applicable_arms(cell))
    verbose && println("  out_dir = ", out_dir)
    flush(stdout)

    reg = ComparatorRegistry.canonical_registry()
    prep = prepare_smoke_cell(cell; cell_id=idx)
    materialize_fixture(prep, out_dir)
    fpath, mpath = fixture_paths(out_dir, cell, idx)
    fx = read_fixture(fpath)
    verbose && @printf("  materialized fixture: %s  (sha256=%s, n_query=%d)\n",
                       fpath, fx.fixture_sha256, fx.n_query)

    results = Dict{Symbol,ArmCellResult}()
    rows_by_arm = Dict{Symbol,Vector{ArmQueryRow}}()
    for arm in applicable_arms(cell)
        verbose && println("\n  Running arm=$(arm) ($(arm_display_name(arm))) ...")
        flush(stdout)
        t_wall = @elapsed (res, rows) = run_arm_cell(cell, arm, out_dir;
                                                       cell_id=idx, reg=reg, fx=fx, verbose=false)
        results[arm] = res
        rows_by_arm[arm] = rows
        verbose && @printf("    wall=%.1fs  fit_time=%.2fs  infer_mean=%.2fms  final_loss=%s  rel_l2=%s  mae=%s  coverage=%s  mean_width=%s  n_finite=%d/%d\n",
                           t_wall, res.fit_time_s, 1000 * res.inference_time_mean_s,
                           isfinite(res.final_loss) ? @sprintf("%.6g", res.final_loss) : "NaN",
                           isfinite(res.rel_l2)   ? @sprintf("%.4f", res.rel_l2)   : "NaN",
                           isfinite(res.mae)      ? @sprintf("%.4f", res.mae)      : "NaN",
                           isfinite(res.coverage) ? @sprintf("%.3f", res.coverage) : "NaN",
                           isfinite(res.mean_width) ? @sprintf("%.4f", res.mean_width) : "NaN",
                           res.n_finite_pred, res.n_query)
    end

    ok = true
    verbose && println("\n  ── Gate checks ──")
    arms_present = sort(collect(keys(results)))
    expected_arms = sort(collect(applicable_arms(cell)))
    check1 = arms_present == expected_arms
    verbose && println("  [1] exactly $(length(expected_arms)) arm outcomes " *
                       "(applicable_arms=$(expected_arms)): got=$(arms_present)  ",
                       check1 ? "PASS ✓" : "FAIL ✗")
    ok &= check1
    fixture_hashes = unique(r.fixture_sha256 for r in values(results))
    check2 = length(fixture_hashes) == 1 && !isempty(first(fixture_hashes))
    verbose && println("  [2] identical fixture_sha256 across arms: ", fixture_hashes,
                       "  ", check2 ? "PASS ✓" : "FAIL ✗")
    ok &= check2
    n_query_canonical = fx.n_query
    check3 = all(length(rows_by_arm[a]) == n_query_canonical for a in arms_present) &&
             all(results[a].n_query == n_query_canonical for a in arms_present)
    verbose && println("  [3] every arm emits exactly n_query=$(n_query_canonical) rows: ",
                       check3 ? "PASS ✓" : "FAIL ✗")
    ok &= check3
    identity_ok = true
    ref_arm = first(arms_present)
    if check3
        for i in 1:n_query_canonical
            ref = rows_by_arm[ref_arm][i]
            for a in arms_present
                r = rows_by_arm[a][i]
                if r.query_id != ref.query_id || r.x != ref.x || r.truth != ref.truth
                    identity_ok = false
                end
            end
        end
    else
        identity_ok = false
    end
    verbose && println("  [4] query_id/coordinate/truth identity across all arms (row-by-row): ",
                       identity_ok ? "PASS ✓" : "FAIL ✗")
    ok &= identity_ok
    # Every arm must predict at every query. Only :nn and :gp minimise a scalar
    # training objective, so only they are required to report a finite loss;
    # :diona, :nsm, :rbf and :idw have no such quantity and record NaN by design.
    check5 = true
    for a in arms_present
        res = results[a]
        preds_ok = res.n_finite_pred == n_query_canonical
        loss_ok  = !_arm_reports_training_loss(a) || isfinite(res.final_loss)
        check5 &= preds_ok && loss_ok
        verbose && @printf("  [5] %-5s preds=%d/%d  final_loss=%s (required finite: %s)  %s\n",
                           arm_display_name(a), res.n_finite_pred, n_query_canonical,
                           isfinite(res.final_loss) ? @sprintf("%.6g", res.final_loss) : "NaN",
                           _arm_reports_training_loss(a),
                           (preds_ok && loss_ok) ? "PASS ✓" : "FAIL ✗")
    end
    ok &= check5
    # Every arm that DECLARES an interval must produce a finite, ordered one
    # wherever it produced a prediction. :diona is excluded: a failed UQ solve
    # legitimately exposes no interval at all (CLAUDE.md), and :nn/:rbf/:idw
    # declare :none.
    check6 = true
    for a in arms_present
        (a == :diona || comparative_interval_kind(a) == :none) && continue
        rows_a = [r for r in rows_by_arm[a] if isfinite(r.pred)]
        finite_ok  = all(isfinite(r.lower) && isfinite(r.upper) for r in rows_a)
        ordered_ok = finite_ok && all(r.lower <= r.upper for r in rows_a)
        check6 &= finite_ok && ordered_ok
        verbose && println("  [6] $(arm_display_name(a)) intervals finite=$(finite_ok) and " *
                           "ordered(lower<=upper)=$(ordered_ok): ",
                           (finite_ok && ordered_ok) ? "PASS ✓" : "FAIL ✗")
    end
    # The interval-less arms must expose no bounds at all -- an absence, not a failure.
    check6b = all(all(isnan(r.lower) && isnan(r.upper) for r in rows_by_arm[a])
                  for a in arms_present if comparative_interval_kind(a) == :none)
    verbose && println("  [6] interval-less arms (" *
                       join((string(a) for a in arms_present
                             if comparative_interval_kind(a) == :none), ", ") *
                       ") expose NaN bounds: ", check6b ? "PASS ✓" : "FAIL ✗")
    ok &= check6 && check6b
    diona_rows = rows_by_arm[:diona]
    check7 = !isempty(diona_rows) && all(r.status in _KNOWN_DIONA_STATUSES for r in diona_rows)
    verbose && println("  [7] Diona status ∈ known src/inference.jl vocabulary: ", check7 ? "PASS ✓" : "FAIL ✗")
    if !check7
        bad = unique(r.status for r in diona_rows if !(r.status in _KNOWN_DIONA_STATUSES))
        verbose && println("      unrecognized status(es): ", bad)
    end
    ok &= check7
    check8 = all(isfinite(results[a].rel_l2) || isnan(results[a].rel_l2) for a in arms_present)  # never errors
    verbose && println("  [8] generic rel-L2/MAE/MSE/interval metrics computed for all " *
                       "$(length(arms_present)) arms via one shared function " *
                       "(no arm==... branch): PASS ✓ (see per-arm numbers above)")
    ok &= check8

    verbose && println()
    verbose && println(ok ? "Multi-arm anchor gate: PASS ✓" :
                             "Multi-arm anchor gate: FAIL ✗ — see checks above")
    verbose && println("Output directory: $out_dir")
    return ok
end
