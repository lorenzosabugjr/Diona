# Campaign-level gates: resilience, fast smoke comparison, full audit.
#
# Extracted from test/smoke_data.jl. These are plain `include`d fragments, NOT a module:
# every name stays top-level in the including scope, because test/run_full_benchmark.jl
# reaches into the harness with getproperty(harness, :name). Included at the END of
# smoke_data.jl, so everything referenced here is already defined.

function _comparative_gate_ids(cells::Vector{SmokeCell})
    anchors = [id for id in eachindex(cells) if
               cells[id].name == :ishigami && cells[id].layout == :sobol &&
               cells[id].n_train == _SINGLE_CELL_ANCHOR_NTRAIN]
    extra = findfirst(c -> c.name == :ishigami && c.layout == :grid, cells)
    length(anchors) >= 2 || error("Orchestration resilience gate could not find two anchor cells")
    extra === nothing && error("Orchestration resilience gate could not find its third anchor cell")
    return Int[anchors[1], anchors[2], extra]
end

function check_gate_orchestration_resilience(; out_dir::Union{Nothing,AbstractString}=nothing,
                       verbose::Bool=true)
    cells = enumerate_cells()
    ids = _comparative_gate_ids(cells)
    out_dir = abspath(out_dir === nothing ?
        joinpath(REPO_ROOT, "results", "comparative_campaign",
            Dates.format(Dates.now(), "yyyy-mm-dd-HHMMSS") * "-gate-comparative") : String(out_dir))
    mkpath(out_dir)
    reg = ComparatorRegistry.canonical_registry()
    # The same --device the orchestrator below will read. The :nn settings hash
    # carries it, so expectations built at the default while the campaign ran on
    # the other device would point at fragment paths nothing ever writes.
    device = ComparatorRegistry.normalize_device(_smoke_opt("device", "cpu"))
    expectations, _ = _comparative_prepare_expectations(ids, cells, out_dir, reg; device=device)

    valid_key = (ids[1], :diona)
    stale_key = (ids[1], :nn)
    failure_key = (ids[2], :gp)
    valid_exp = only(exp for exp in expectations if (exp.cell_id, exp.arm) == valid_key)
    stale_exp = only(exp for exp in expectations if (exp.cell_id, exp.arm) == stale_key)
    pre = run_comparative_worker_stage(Any[valid_exp], out_dir, "gate-comparative-prewarm", 1,
                              DEFAULT_TIMEOUT_COMPARATIVE[:diona])
    get(pre.outcomes, valid_key, :MISSING) == :OK ||
        error("Orchestration resilience gate prewarm worker failed; see $(valid_exp.paths.log)")
    wrong_hash = repeat("0", 64)
    stale_result, stale_rows = _comparative_failure_payload(stale_exp.fx, :nn, wrong_hash, reg, :STALE)
    write_comparative_arm_fragments(out_dir, "gate-comparative-stale", stale_exp.fx,
        stale_result, stale_rows; terminal_state=:OK,
        code_provenance=stale_exp.code_provenance,
        worker_project=stale_exp.worker_project, paths=stale_exp.paths,
        blas_threads=DEFAULT_BLAS_THREADS_COMPARATIVE)

    # The gate now runs 3 cells x applicable_arms(cell) arm-cells rather than 9;
    # the caps keep it from oversubscribing the machine as it widens.
    gate_jobs = Dict(:diona => 2, :nn => 2, :gp => 1, :nsm => 2, :rbf => 1, :idw => 2)
    campaign = run_comparative_orchestrator(:representative;
        selected_ids_override=ids, out_dir_override=out_dir,
        inject_failure_override="$(failure_key[1]):$(failure_key[2])",
        jobs_override=gate_jobs)

    expected_keys = Set((exp.cell_id, exp.arm) for exp in campaign.expectations)
    actual_keys = Set(keys(campaign.payloads))
    failed_payload = campaign.payloads[failure_key]
    project_ok = all(normpath(payload.worker_project) == _comparative_project_file(key[2])
                     for (key, payload) in campaign.payloads)
    concurrency_ok = all(campaign.max_active[arm] <= campaign.jobs[arm]
                         for arm in _COMPARATIVE_ARM_ORDER)
    success_retained = all(isfile(exp.paths.summary) && isfile(exp.paths.points)
                           for exp in campaign.expectations
                           if campaign.payloads[(exp.cell_id, exp.arm)].terminal_state == :OK)
    failure_explicit = failed_payload.terminal_state == :ERROR &&
        length(failed_payload.rows) == failed_payload.result.n_query &&
        all(!isfinite(row.pred) && occursin("WORKER_ERROR", string(row.status))
            for row in failed_payload.rows)
    stale_reran = stale_key in campaign.stale && stale_key in keys(campaign.process_outcomes) &&
        campaign.payloads[stale_key].terminal_state == :OK &&
        campaign.payloads[stale_key].result.settings_hash == stale_exp.settings_hash
    valid_resumed = valid_key in campaign.resumed &&
        !haskey(campaign.process_outcomes, valid_key)
    exact_accounting = actual_keys == expected_keys &&
        length(campaign.payloads) == length(expected_keys)
    identity_on_resumed = try
        common_success_results(campaign.results, campaign.row_groups)
        valid_key in campaign.resumed
    catch
        false
    end

    checks = Pair{String,Bool}[
        "correct shared vs Lux worker project selected for every arm" => project_ok,
        "configured per-arm concurrency limits respected" => concurrency_ok,
        "successful worker fragments retained" => success_retained,
        "deliberate failure has explicit terminal fragment and query rows" => failure_explicit,
        "wrong-hash stale fragment detected and rerun" => stale_reran,
        "valid matching prior fragment resumed without worker relaunch" => valid_resumed,
        "exactly one terminal outcome per expected (cell_id,arm)" => exact_accounting,
        "reporting-layer query-identity validation runs over resumed outputs" => identity_on_resumed,
    ]
    overall = all(last, checks) && !campaign.success
    if verbose
        println("\nOrchestration resilience gate")
        println("  matrix cell ids=$(ids), arm-cells=$(length(expected_keys)); deliberate failure=$failure_key; stale=$stale_key; valid-resume=$valid_key")
        for (i, (label, passed)) in enumerate(checks)
            println("  [$i] $label: ", passed ? "PASS ✓" : "FAIL ✗")
        end
        println("  expected campaign state with injected failure: ",
                !campaign.success ? "PASS ✓ (NON-SUCCESS)" : "FAIL ✗ (unexpected SUCCESS)")
        println("  artifacts: $(campaign.out_dir)")
        println(overall ? "Orchestration resilience gate overall: PASS ✓" : "Orchestration resilience gate overall: FAIL ✗")
    end
    return overall, campaign
end

function _fast_smoke_cell_ids(cells::Vector{SmokeCell})
    p1 = findfirst(c -> c.name == :ishigami &&
                        c.layout == :sobol && c.tier_label == _FAST_SMOKE_TIER &&
                        c.noise_level == _FAST_SMOKE_NOISE_LEVEL, cells)
    p2 = findfirst(c -> c.name == :rosenbrock &&
                        c.layout == :sobol && c.tier_label == _FAST_SMOKE_TIER &&
                        c.noise_level == _FAST_SMOKE_NOISE_LEVEL, cells)
    (p1 === nothing || p2 === nothing) &&
        error("Fast comparative smoke gate representative middle-tier cells were not found")
    return Int[p1, p2]
end

function compare_fast_smoke_runs(reference_dir::AbstractString, campaign)
    ref_summary_path = joinpath(abspath(reference_dir), "comparative_summary.csv")
    ref_summary = _fast_smoke_read_csv(ref_summary_path)
    cur_summary = _fast_smoke_read_csv(campaign.artifacts.summary_path)
    ref_points = _fast_smoke_read_points(reference_dir, ref_summary)
    cur_points = _fast_smoke_read_points(campaign.out_dir, cur_summary)

    expected_keys = Set((exp.cell_id, exp.arm) for exp in campaign.expectations)
    summary_key(row) = (parse(Int, row[:cell_id]), Symbol(row[:arm]))
    ref_keys = summary_key.(ref_summary)
    cur_keys = summary_key.(cur_summary)
    exact_arms = length(ref_keys) == length(cur_keys) == length(expected_keys) &&
        length(unique(ref_keys)) == length(ref_keys) &&
        length(unique(cur_keys)) == length(cur_keys) &&
        Set(ref_keys) == Set(cur_keys) == expected_keys

    ref_summary_by_key = Dict(summary_key(row) => row for row in ref_summary)
    cur_summary_by_key = Dict(summary_key(row) => row for row in cur_summary)
    fixture_hashes = exact_arms && all(
        ref_summary_by_key[key][:fixture_sha256] == cur_summary_by_key[key][:fixture_sha256]
        for key in expected_keys)
    settings_hashes = exact_arms && all(
        ref_summary_by_key[key][:settings_hash] == cur_summary_by_key[key][:settings_hash]
        for key in expected_keys)

    point_key(row) = (_point_int(row[:cell_id]), _point_int(row[:query_id]),
                      Symbol(_point_string(row[:arm])))
    ref_point_by_key = Dict(point_key(row) => row for row in ref_points)
    cur_point_by_key = Dict(point_key(row) => row for row in cur_points)
    same_point_keys = length(ref_point_by_key) == length(ref_points) &&
        length(cur_point_by_key) == length(cur_points) &&
        Set(keys(ref_point_by_key)) == Set(keys(cur_point_by_key))
    query_identity = same_point_keys && exact_arms
    diona_exact = same_point_keys && exact_arms
    finite_predictions = same_point_keys && exact_arms
    if same_point_keys && exact_arms
        for key in keys(ref_point_by_key)
            ref = ref_point_by_key[key]
            cur = cur_point_by_key[key]
            dim = parse(Int, ref_summary_by_key[(key[1], key[3])][:dim])
            query_identity &= isequal(ref[:query_id], cur[:query_id]) &&
                isequal(ref[:truth], cur[:truth]) &&
                all(isequal(ref[Symbol("x_$d")], cur[Symbol("x_$d")]) for d in 1:dim)
            finite_predictions &= isfinite(_parse_float(ref[:pred])) &&
                                  isfinite(_parse_float(cur[:pred]))
            if key[3] == :diona
                diona_exact &= isequal(ref[:pred], cur[:pred]) &&
                               isequal(ref[:lower], cur[:lower]) &&
                               isequal(ref[:upper], cur[:upper]) &&
                               isequal(ref[:status], cur[:status])
            end
        end
    end
    return Pair{String,Bool}[
        "fixture hashes match the prior run exactly" => fixture_hashes,
        "settings hashes match the prior run exactly" => settings_hashes,
        "query IDs/coordinates/truth match the prior run exactly" => query_identity,
        "predictions are finite in both runs (ML predictions may differ)" => finite_predictions,
        "every expected arm appears exactly once in both runs; no out-of-scope arm" => exact_arms,
        "Diona predictions/bounds/status match the prior run exactly" => diona_exact,
    ]
end

function check_gate_fast_comparative_smoke(; out_dir::Union{Nothing,AbstractString}=nothing,
                       compare_to::Union{Nothing,AbstractString}=nothing, verbose::Bool=true)
    cells = enumerate_cells()
    ids = _fast_smoke_cell_ids(cells)
    campaign = run_comparative_orchestrator(:representative;
        selected_ids_override=ids, out_dir_override=out_dir,
        inject_failure_override="")
    expected = Set((id, arm) for id in ids for arm in applicable_arms(cells[id]))
    actual = [(r.cell_id, r.arm) for r in campaign.results]
    ok = campaign.success && length(actual) == length(expected) && Set(actual) == expected &&
        all(_fast_smoke_summary_row_populated, campaign.artifacts.summary_rows)
    if compare_to !== nothing
        comparison = compare_fast_smoke_runs(compare_to, campaign)
        ok &= all(last, comparison)
    end
    verbose && println("Fast comparative smoke: ", ok ? "PASS" : "FAIL")
    return ok, campaign
end

function check_gate_comparative_full_audit(campaign; problem_artifacts=nothing, verbose::Bool=true)
    cells = enumerate_cells()
    expected_keys = Set((id, arm) for id in eachindex(cells)
                        for arm in applicable_arms(cells[id]))
    actual_keys = [(result.cell_id, result.arm) for result in campaign.results]
    payload_keys = collect(keys(campaign.payloads))

    exact_terminal_count = length(campaign.payloads) == length(expected_keys) &&
        length(campaign.expectations) == length(expected_keys)
    exact_key_accounting = length(actual_keys) == length(expected_keys) &&
        length(unique(actual_keys)) == length(expected_keys) && Set(actual_keys) == expected_keys &&
        length(payload_keys) == length(expected_keys) && length(unique(payload_keys)) == length(expected_keys) &&
        Set(payload_keys) == expected_keys

    arms_by_cell_ok = all(begin
        got = Set(result.arm for result in campaign.results if result.cell_id == id)
        got == Set(applicable_arms(cells[id]))
    end for id in eachindex(cells))

    fragment_identity_ok = true
    for expected in campaign.expectations
        check = _comparative_try_read(expected; allow_failure=true)
        fragment_identity_ok &= check.valid
    end
    cross_arm_identity_ok = try
        _validated_comparative_index(campaign.results, campaign.row_groups)
        true
    catch err
        verbose && println("  reporting-layer whole-campaign identity error: ", sprint(showerror, err))
        false
    end
    identity_ok = fragment_identity_ok && cross_arm_identity_ok

    nonfinite_rows = [row for rows in campaign.row_groups for row in rows
                      if !isfinite(row.pred)]
    requested_rows = sum(result.n_query for result in campaign.results)
    explicit_failures_ok = all(begin
        payload = campaign.payloads[(expected.cell_id, expected.arm)]
        length(payload.rows) == expected.fx.n_query == payload.result.n_query &&
        sum(values(payload.result.status_counts)) == payload.result.n_query &&
        all(!isempty(string(row.status)) &&
            (isfinite(row.pred) || row.status != :COMPLETED) for row in payload.rows)
    end for expected in campaign.expectations)

    scope_ok = arms_by_cell_ok &&
        all(result.arm in _COMPARATIVE_ARM_ORDER for result in campaign.results)

    common_ok = try
        common = common_success_results(campaign.results, campaign.row_groups)
        length(common) == length(expected_keys) && all(0 <= row.n_common <= row.n_requested for row in common)
    catch
        false
    end

    problem_artifacts === nothing &&
        (problem_artifacts = write_problem_aggregate_artifacts(
            campaign.out_dir, campaign.results, campaign.row_groups))
    expected_problem_rows = length(Set((c.name, arm) for c in cells for arm in applicable_arms(c)))
    problem_aggregation_ok = length(problem_artifacts.rows) == expected_problem_rows &&
        isfile(problem_artifacts.csv_path) && isfile(problem_artifacts.markdown_path) &&
        all(row.n_common <= row.n_requested &&
            row.n_failed_pred == row.n_requested - row.n_finite_pred
            for row in problem_artifacts.rows)

    artifact_ok = isfile(campaign.artifacts.summary_path) &&
        all(isfile(expected.paths.points) for expected in campaign.expectations) &&
        !isfile(joinpath(campaign.out_dir, "comparative_points.csv")) &&
        isfile(campaign.outcomes_path) &&
        length(campaign.artifacts.summary_rows) == length(expected_keys) &&
        length(campaign.artifacts.point_rows) == requested_rows

    checks = Pair{String,Bool}[
        "exactly $(length(expected_keys)) terminal arm-cell outcomes are present" => exact_terminal_count,
        "every expected (cell_id,arm) appears exactly once (no gaps/duplicates)" => exact_key_accounting,
        "every cell has exactly its specified arm set" => arms_by_cell_ok,
        "every emitted row matches its fixture bitwise and reporting-layer validates all cross-arm identities" => identity_ok,
        "all missing/failed predictions remain explicit status-bearing rows" => explicit_failures_ok,
        "only the configured data-only arms entered the campaign" => scope_ok,
        "common-success accounting is complete for all $(length(expected_keys)) arm-cells" => common_ok,
        "$(expected_problem_rows) within-function/dataset arm summaries include explicit failure/common counts" => problem_aggregation_ok,
        "per-cell point/summary/outcome artifacts have complete row coverage" => artifact_ok,
    ]
    overall = all(last, checks)
    n_terminal_failed = count(payload -> payload.terminal_state != :OK,
                              values(campaign.payloads))
    n_nonfinite = length(nonfinite_rows)
    audit_path = joinpath(campaign.out_dir, "gate_comparative_full_audit.txt")
    lines = String["Comparative full-audit gate — Full comparative campaign complete-accounting audit"]
    for (i, (label, passed)) in enumerate(checks)
        push!(lines, "  [$i] $label: " * (passed ? "PASS" : "FAIL"))
    end
    push!(lines, "  terminal worker failures: $n_terminal_failed / $(length(expected_keys))")
    push!(lines, "  non-finite predictions: $n_nonfinite / $requested_rows")
    push!(lines, overall ? "Comparative full-audit gate overall: PASS" : "Comparative full-audit gate overall: FAIL")
    atomic_write(audit_path) do io
        foreach(line -> println(io, line), lines)
    end
    if verbose
        println()
        foreach(println, lines)
        println("  problem summaries: $(problem_artifacts.csv_path), $(problem_artifacts.markdown_path)")
        println("  audit artifact: $audit_path")
    end
    return (overall=overall, checks=checks, audit_path=audit_path,
            n_terminal_failed=n_terminal_failed, n_nonfinite_pred=n_nonfinite,
            n_requested_predictions=requested_rows,
            problem_artifacts=problem_artifacts)
end
