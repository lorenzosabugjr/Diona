# The :nsm arm (set-membership, Fagiano eps/gamma).
#
# Plain `include`d FRAGMENT, not a module -- every name stays top-level in the
# including scope. Pulled in by ml/harness/arms.jl, which smoke_data.jl includes.
#
# Split per arm so comparative_code_provenance can hash ONE arm's runner: editing
# this file rotates only :nsm, not all six.

# ==========================================================================
# NSM (set-membership) arm -- in-process under --project=test/ml
#
# PROTOCOL: eps and gamma are estimated by the arm from the training data
# (a port of opt_smgo_d.m); nothing outside the dataset enters except the
# domain box, via the same normalize_inputs_pm1 every other arm uses. The band
# is a set-membership band built from ESTIMATED constants -- it certifies
# nothing, which is what nsm_interval_kind() == :set_membership_estimated says.
# ==========================================================================

function _nsm_settings_for(dim::Int)
    _NSM_AVAILABLE || error("the :nsm arm needs NSMTrainer (test/ml/baselines/nsm.jl), " *
        "which did not load in this process. Run it under --project=test/ml.")
    # frame=:pm1 and the default y_scale=1.0 reproduce the configuration of the
    # 36-cell eps/gamma campaign in test/ml/baselines/gate/ (gate.jl:97 FRAME=:pm1,
    # gate.jl:102 nsm_frame_gamma_floor(FRAME), gate.jl:104-107 normalize_inputs_pm1 +
    # target_zscore), so the recorded evidence describes the shipped arm.
    return NSMTrainer.nsm_settings(dim)
end

_nsm_settings_hash(dim::Int) =
    bytes2hex(sha256(NSMTrainer.nsm_settings_string(_nsm_settings_for(dim))))

function run_nsm_arm_cell(fx; verbose::Bool=false)
    _NSM_AVAILABLE || error("run_nsm_arm_cell: NSMTrainer is not loaded in this process " *
        "(test/ml/baselines/nsm.jl needs NearestNeighbors). Run the NSM arm under " *
        "--project=test/ml (this repo's documented default invocation).")
    prep = prepared_cell_from_fixture(fx)
    cell = prep.cell
    bf = _BENCH_FN[cell.name]

    s = _nsm_settings_for(cell.dim)
    settings_hash = bytes2hex(sha256(NSMTrainer.nsm_settings_string(s)))

    X_fit  = BenchmarkDriver.normalize_inputs_pm1(bf, prep.X_train)
    X_test = BenchmarkDriver.normalize_inputs_pm1(bf, prep.X_query)
    yt     = BenchmarkDriver.target_zscore(prep.y_train_noisy)

    local model, fit_time
    train_ok = true
    try
        fit_time = @elapsed model = NSMTrainer.fit_nsm(
            Matrix{Float64}(X_fit), Vector{Float64}(yt.normalized), s)
    catch err
        train_ok = false
        fit_time = NaN
        @warn "NSM fit failed for cell $(cell.name)/$(cell.tier_label)" exception=(err, catch_backtrace())
    end

    n_q = prep.n_query
    rows = Vector{ArmQueryRow}(undef, n_q)
    # Per-query timed loop rather than nsm_predict_batch: elapsed_s must measure
    # the same thing for every arm, and the batch path reports an AMORTISED time.
    gamma_used = train_ok ? model.gamma : NaN
    for i in 1:n_q
        if train_ok
            t0 = time()
            m, lo, hi = NSMTrainer.nsm_predict(model, Float64.(view(X_test, :, i)))
            el = time() - t0
            pred_i  = yt.mean + yt.std * m
            lower_i = yt.mean + yt.std * lo
            upper_i = yt.mean + yt.std * hi
            status_i = NSMTrainer.nsm_status(m, lo, hi)
        else
            pred_i, lower_i, upper_i, el, status_i = NaN, NaN, NaN, 0.0, :NSM_TRAIN_FAILED
        end
        rows[i] = ArmQueryRow(fx.cell_id, i, :nsm, prep.X_query[:, i], prep.y_query_truth[i],
                               pred_i, lower_i, upper_i, status_i, el, gamma_used)
    end

    result = arm_cell_result_from_rows(rows; cell_id=fx.cell_id, cell=cell, arm=:nsm,
        model_type=NSMTrainer.nsm_model_type(), fixture_sha256=fx.fixture_sha256,
        settings_hash=settings_hash, model_seed=s.seed, noise_seed=fx.noise_seed,
        n_train=size(X_fit, 2), fit_time_s=fit_time, final_loss=NaN)
    return result, rows
end
