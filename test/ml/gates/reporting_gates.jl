# Reporting-layer gates and the synthetic fixtures they are built from.
#
# Extracted from test/smoke_data.jl. These are plain `include`d fragments, NOT a module:
# every name stays top-level in the including scope, because test/run_full_benchmark.jl
# reaches into the harness with getproperty(harness, :name). Included at the END of
# smoke_data.jl, so everything referenced here is already defined.

function _reporting_synthetic_result(cell_id::Int, cell::SmokeCell, arm::Symbol,
                              model_type::String, rows::Vector{ArmQueryRow};
                              fit_time_s::Float64, final_loss::Float64)
    return arm_cell_result_from_rows(rows;
        cell_id=cell_id, cell=cell, arm=arm, model_type=model_type,
        fixture_sha256="fixture-$(cell_id)", settings_hash="settings-$(arm)",
        model_seed=0, noise_seed=42, n_train=8,
        fit_time_s=fit_time_s, final_loss=final_loss)
end

function _reporting_make_rows(cell_id::Int, cell::SmokeCell, arm::Symbol,
                       truth::Vector{Float64}, pred::Vector{Float64},
                       status::Vector{Symbol}; lower=fill(NaN, length(truth)),
                       upper=fill(NaN, length(truth)), elapsed_base=0.01,
                       smoothness=fill(NaN, length(truth)))
    return ArmQueryRow[
        ArmQueryRow(cell_id, query_id, arm,
                    [Float64(query_id) + 0.1 * (d - 1) for d in 1:cell.dim],
                    truth[query_id], pred[query_id], lower[query_id], upper[query_id],
                    status[query_id], elapsed_base * query_id, smoothness[query_id])
        for query_id in eachindex(truth)
    ]
end

function _reporting_layer_is_pure()
    # The reporting layer now lives in its own fragment; read THAT, not this gate.
    source = read(joinpath(TEST_DIR, "ml", "harness", "reporting.jl"), String)
    begin_marker = "# REPORTING_LAYER_BEGIN"
    end_marker = "# REPORTING_LAYER_END"
    begin_range = findfirst(begin_marker, source)
    end_range = findfirst(end_marker, source)
    (begin_range === nothing || end_range === nothing) && return false
    block = source[last(begin_range) + 1:first(end_range) - 1]
    forbidden = ("prepare_smoke_cell(", "materialize_fixture(", "run_arm_cell(",
                 "solve_cell_diona(", "train_nn(", "train_gp(",
                 ".X_train", ".y_train_noisy")
    return all(!occursin(token, block) for token in forbidden)
end

function check_gate_reporting_correctness(; out_dir::Union{Nothing,AbstractString}=nothing,
                        verbose::Bool=true)
    out_dir = out_dir === nothing ? mktempdir() : String(out_dir)
    mkpath(out_dir)

    p1_id = 9001
    p1 = SmokeCell(:synthetic_a, 1, :sobol, "synthetic_4", 8, 0.0)
    p1_truth = [1.0, 2.0, 3.0, 4.0]
    completed4 = fill(:COMPLETED, 4)
    diona1 = _reporting_make_rows(p1_id, p1, :diona, p1_truth, [1.1, 1.9, 3.2, 3.8],
        completed4; lower=[0.5, 1.5, 2.5, 3.5], upper=[1.5, 2.5, 3.5, 4.5],
        smoothness=[2.5, 5.0, 10.0, 2.5])
    nn1 = _reporting_make_rows(p1_id, p1, :nn, p1_truth, [1.0, NaN, 2.5, NaN],
        [:COMPLETED, :NN_PREDICT_NONFINITE, :COMPLETED, :NN_PREDICT_NONFINITE];
        elapsed_base=0.02)
    gp1 = _reporting_make_rows(p1_id, p1, :gp, p1_truth, [0.9, 2.1, 3.1, 4.1],
        completed4; lower=[0.7, 1.7, 2.7, 3.7], upper=[1.1, 2.3, 3.3, 4.3],
        elapsed_base=0.03)
    nsm1 = _reporting_make_rows(p1_id, p1, :nsm, p1_truth, [1.05, 2.05, 2.95, 3.95],
        completed4; lower=[0.6, 1.6, 2.6, 3.6], upper=[1.4, 2.4, 3.4, 4.4],
        elapsed_base=0.04, smoothness=fill(7.5, 4))
    rbf1 = _reporting_make_rows(p1_id, p1, :rbf, p1_truth, [1.02, 2.02, 3.02, 4.02],
        completed4; elapsed_base=0.05)
    idw1 = _reporting_make_rows(p1_id, p1, :idw, p1_truth, [1.2, 2.2, 3.2, 4.2],
        completed4; elapsed_base=0.06)

    p2_id = 9002
    p2 = SmokeCell(:synthetic_b, 3, :grid, "synthetic_3", 8, 0.05)
    p2_truth = [0.5, 1.0, 1.5]
    completed3 = fill(:COMPLETED, 3)
    diona2 = _reporting_make_rows(p2_id, p2, :diona, p2_truth, [0.45, 1.05, 1.55],
        completed3; lower=[0.3, 0.8, 1.3], upper=[0.7, 1.2, 1.7])
    nn2 = _reporting_make_rows(p2_id, p2, :nn, p2_truth, [0.55, 0.95, 1.45],
        completed3; elapsed_base=0.02)
    gp2 = _reporting_make_rows(p2_id, p2, :gp, p2_truth, [0.52, 0.98, 1.52],
        completed3; lower=[0.4, 0.9, 1.4], upper=[0.6, 1.1, 1.6],
        elapsed_base=0.03)
    nsm2 = _reporting_make_rows(p2_id, p2, :nsm, p2_truth, [0.51, 1.01, 1.49],
        completed3; lower=[0.35, 0.85, 1.35], upper=[0.65, 1.15, 1.65],
        elapsed_base=0.04, smoothness=fill(3.75, 3))
    rbf2 = _reporting_make_rows(p2_id, p2, :rbf, p2_truth, [0.49, 1.02, 1.51],
        completed3; elapsed_base=0.05)
    idw2 = _reporting_make_rows(p2_id, p2, :idw, p2_truth, [0.6, 1.1, 1.6],
        completed3; elapsed_base=0.06)

    row_groups = Vector{ArmQueryRow}[diona1, nn1, gp1, nsm1, rbf1, idw1,
                                     diona2, nn2, gp2, nsm2, rbf2, idw2]
    results = ArmCellResult[
        _reporting_synthetic_result(p1_id, p1, :diona, "Diona", diona1;
                             fit_time_s=0.0, final_loss=NaN),
        _reporting_synthetic_result(p1_id, p1, :nn, "MLP", nn1;
                             fit_time_s=5.0, final_loss=0.25),
        _reporting_synthetic_result(p1_id, p1, :gp, "SVGP", gp1;
                             fit_time_s=7.0, final_loss=0.15),
        _reporting_synthetic_result(p1_id, p1, :nsm, "NSM", nsm1;
                             fit_time_s=1.0, final_loss=NaN),
        _reporting_synthetic_result(p1_id, p1, :rbf, "RadialBasis", rbf1;
                             fit_time_s=2.0, final_loss=NaN),
        _reporting_synthetic_result(p1_id, p1, :idw, "InverseDistance", idw1;
                             fit_time_s=0.0, final_loss=NaN),
        _reporting_synthetic_result(p2_id, p2, :diona, "Diona", diona2;
                             fit_time_s=0.0, final_loss=NaN),
        _reporting_synthetic_result(p2_id, p2, :nn, "MLP", nn2;
                             fit_time_s=4.0, final_loss=0.12),
        _reporting_synthetic_result(p2_id, p2, :gp, "SVGP", gp2;
                             fit_time_s=6.0, final_loss=0.10),
        _reporting_synthetic_result(p2_id, p2, :nsm, "NSM", nsm2;
                             fit_time_s=1.5, final_loss=NaN),
        _reporting_synthetic_result(p2_id, p2, :rbf, "RadialBasis", rbf2;
                             fit_time_s=2.5, final_loss=NaN),
        _reporting_synthetic_result(p2_id, p2, :idw, "InverseDistance", idw2;
                             fit_time_s=0.0, final_loss=NaN),
    ]

    # These cell ids are synthetic and are not enumerated benchmark cells, so the
    # expectation is supplied explicitly -- the same mechanism that lets a
    # partial (arm-sharded) campaign be reported without the check degrading
    # into "whatever arms turned up".
    gate_expected_arms = Dict(p1_id => Set((:diona, :nn, :gp, :nsm, :rbf, :idw)),
                              p2_id => Set((:diona, :nn, :gp, :nsm, :rbf, :idw)))

    artifacts = write_comparative_artifacts(out_dir, results, row_groups;
        prefix="reporting-correctness-gate", expected_arms=gate_expected_arms)
    point_rows = artifacts.point_rows
    summary_rows = artifacts.summary_rows
    common_ids = common_success_query_ids(results, row_groups;
                                          expected_arms=gate_expected_arms)

    checks = Pair{String,Bool}[]
    expected_keys = Set((row.cell_id, row.query_id, row.arm) for rows in row_groups for row in rows)
    actual_keys = [(row.cell_id, row.query_id, row.arm) for row in point_rows]
    n_expected_points = sum(length(rows) for rows in row_groups)
    push!(checks, "exact in-memory point key coverage and uniqueness ($(n_expected_points)/$(n_expected_points) rows)" =>
                  length(actual_keys) == n_expected_points &&
                  length(unique(actual_keys)) == n_expected_points &&
                  Set(actual_keys) == expected_keys)

    p1_common_rows = [row for row in summary_rows if row.cell_id == p1_id]
    push!(checks, "common-success intersection excludes NN failures (queries 2 and 4; 2/4 common)" =>
                  common_ids[p1_id] == [1, 3] &&
                  all(row.n_common == 2 && row.n_requested == 4 for row in p1_common_rows))

    nn_result = only(result for result in results if result.cell_id == p1_id && result.arm == :nn)
    expected_nn_rel_l2 = 0.5 / sqrt(10.0)
    push!(checks, "per-arm NN diagnostics use its two finite predictions only" =>
                  nn_result.n_finite_pred == 2 &&
                  isapprox(nn_result.rel_l2, expected_nn_rel_l2; atol=1e-14) &&
                  isapprox(nn_result.mae, 0.25; atol=1e-14) &&
                  isapprox(nn_result.mse, 0.125; atol=1e-14))

    failed_points = [row for row in point_rows if row.cell_id == p1_id && row.arm == :nn && !isfinite(row.pred)]
    push!(checks, "failed NN rows remain visible with actual failure status" =>
                  sort([row.query_id for row in failed_points]) == [2, 4] &&
                  all(row.status == :NN_PREDICT_NONFINITE for row in failed_points))

    diona_points = [row for row in point_rows if row.arm == :diona]
    gp_points = [row for row in point_rows if row.arm == :gp]
    nsm_points = [row for row in point_rows if row.arm == :nsm]
    no_interval_points = [row for row in point_rows if row.arm in (:nn, :rbf, :idw)]
    push!(checks, "interval semantics are distinct and NN/RBF/IDW bounds are unavailable" =>
                  all(row.interval_kind == :admissible_set for row in diona_points) &&
                  all(row.interval_kind == :variational_approximate for row in gp_points) &&
                  all(row.interval_kind == :set_membership_estimated for row in nsm_points) &&
                  !isempty(nsm_points) && !isempty(no_interval_points) &&
                  all(row.interval_kind == :none && isnan(row.lower) && isnan(row.upper) &&
                      !row.interval_available for row in no_interval_points))

    # The reporting layer's interval-kind vocabulary must agree with what each
    # arm's own module declares; a divergence here would mislabel the semantics
    # in every artifact. Skipped for a module that did not load in this process.
    arm_contract_ok = comparative_interval_kind(:nn) == :none
    if _NSM_AVAILABLE
        arm_contract_ok &= comparative_interval_kind(:nsm) == NSMTrainer.nsm_interval_kind()
    end
    if _SURROGATE_ADAPTER_AVAILABLE
        arm_contract_ok &= all(comparative_interval_kind(a) ==
                               SurrogateArms.surrogate_interval_kind(a) for a in (:rbf, :idw))
    end
    push!(checks, "interval_kind agrees with each arm module's own declaration" => arm_contract_ok)

    # The realised smoothness constant survives to the point rows for the arms
    # that estimate one, and is absent for the arms that do not.
    smoothness_ok =
        [row.smoothness_used for row in point_rows if row.arm == :diona && row.cell_id == p1_id] ==
            [2.5, 5.0, 10.0, 2.5] &&
        all(row.smoothness_used == 7.5 for row in nsm_points if row.cell_id == p1_id) &&
        all(isnan(row.smoothness_used) for row in point_rows if row.arm in (:nn, :gp, :rbf, :idw))
    push!(checks, "per-query realised smoothness constant survives to the point rows " *
                  "(Diona L_H_used escalates 2.5 -> 10.0; NaN where no constant is estimated)" =>
                  smoothness_ok)

    # The completeness check must still BITE: dropping one arm has to be an
    # error under the default expectation, and reportable only when the caller
    # states the partial expectation explicitly.
    partial_groups = [rows for rows in row_groups if first(rows).arm != :idw]
    partial_results = [r for r in results if r.arm != :idw]
    partial_arms = Set((:diona, :nn, :gp, :nsm, :rbf))
    partial_rejected = try
        comparative_summary_rows(partial_results, partial_groups;
                                 expected_arms=gate_expected_arms)
        false
    catch
        true
    end
    partial_reportable = try
        length(comparative_summary_rows(partial_results, partial_groups;
                                        expected_arms=partial_arms)) ==
            length(partial_results)
    catch
        false
    end
    push!(checks, "a missing arm is rejected against the full expectation and reportable " *
                  "only against an explicit partial one" =>
                  partial_rejected && partial_reportable)

    summary_header = split(first(readlines(artifacts.summary_path)), ',')
    p1_nn_summary = only(row for row in summary_rows if row.cell_id == p1_id && row.arm == :nn)
    push!(checks, "fit and inference times remain separate columns and retain separate values" =>
                  "fit_time_s" in summary_header &&
                  "inference_time_total_s" in summary_header &&
                  "inference_time_mean_s" in summary_header &&
                  p1_nn_summary.fit_time_s == 5.0 &&
                  p1_nn_summary.inference_time_total_s == sum(row.elapsed_s for row in nn1) &&
                  p1_nn_summary.inference_time_mean_s == mean(row.elapsed_s for row in nn1))

    p2_nn_summary = only(row for row in summary_rows if row.cell_id == p2_id && row.arm == :nn)
    all_arm_fields_written = all(string(field) in summary_header for field in fieldnames(ArmCellResult))
    push!(checks, "all arm fields survive end-to-end across both cells" =>
                  all_arm_fields_written && "interval_kind" in summary_header &&
                  "n_common" in summary_header && "n_requested" in summary_header &&
                  p2_nn_summary.fit_time_s == 4.0 && p2_nn_summary.final_loss == 0.12)

    push!(checks, "reporting layer is structurally isolated from preparation/training execution" =>
                  _reporting_layer_is_pure())

    legacy_fields = (:cell, :pred, :lower, :upper, :status, :elapsed,
                     :noise_bound_used, :y_truth, :n_query, :n_train_actual,
                     :t_neighbor, :t_lh, :t_assembly, :t_bounds)
    push!(checks, "legacy Diona-only CellResult/metrics/CSV path remains structurally unchanged" =>
                  fieldnames(CellResult) == legacy_fields &&
                  hasmethod(compute_cell_metrics, Tuple{CellResult}) &&
                  !occursin("arm", _FULL_SWEEP_SUMMARY_HEADER) && !occursin("arm", _FULL_SWEEP_POINTS_HEADER))

    overall = all(last, checks)
    if verbose
        println("Reporting correctness gate (synthetic arm outputs only)")
        for (index, (label, passed)) in enumerate(checks)
            println("  [$(index)] $(label): ", passed ? "PASS ✓" : "FAIL ✗")
        end
        println("  Artifacts: $(artifacts.summary_path) (point rows remain fragment-backed)")
        println(overall ? "Reporting correctness gate overall: PASS ✓" : "Reporting correctness gate overall: FAIL ✗")
    end
    return overall
end

function check_gate_full_sweep_coverage(rows, expected_ids::Vector{Int}; verbose::Bool=true)
    got_ids = sort(unique(r.cell_id for r in rows))
    want_ids = sort(expected_ids)
    missing_ids = setdiff(want_ids, got_ids)
    ok = (got_ids == want_ids)
    if verbose
        println()
        println("Full sweep gate — Full sweep coverage")
        println("  Rows: $(length(rows))  Expected: $(length(want_ids))")
        isempty(missing_ids) || println("  MISSING cell ids: ", missing_ids)
        println(ok ? "  PASS ✓" : "  FAIL ✗")
    end
    return ok
end
