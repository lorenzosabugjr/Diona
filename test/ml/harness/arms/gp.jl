# The :gp arm (sparse variational GP).
#
# Plain `include`d FRAGMENT, not a module -- every name stays top-level in the
# including scope. Pulled in by ml/harness/arms.jl, which smoke_data.jl includes.
#
# Split per arm so comparative_code_provenance can hash ONE arm's runner: editing
# this file rotates only :gp, not all six.

function run_gp_arm_cell(fx, reg::ComparatorRegistry.CanonicalRegistry=ComparatorRegistry.canonical_registry();
                          verbose::Bool=false, checkpoint_path=nothing)
    _GP_TRAINER_AVAILABLE || error("run_gp_arm_cell: GPTrainer is not loaded in this process " *
        "(this Julia process was likely started under --project=test/ml/nn_env). " *
        "Run the GP arm under --project=test/ml (this repo's documented default invocation).")
    prep = prepared_cell_from_fixture(fx)
    cell = prep.cell
    bf = _BENCH_FN[cell.name]
    s = ComparatorRegistry.settings_for(reg, cell.dim, :gp)
    settings_hash = bytes2hex(sha256(ComparatorRegistry.canonical_serialize(s)))
    identity = checkpoint_identity(fx.cell_id, :gp, s.seed,
                                   fx.fixture_sha256, settings_hash)

    X_fit  = BenchmarkDriver.normalize_inputs_pm1(bf, prep.X_train)
    X_test = BenchmarkDriver.normalize_inputs_pm1(bf, prep.X_query)
    yt     = BenchmarkDriver.target_zscore(prep.y_train_noisy)

    n_train = size(X_fit, 2)
    eff_batch = n_train > s.minibatch_threshold ? min(s.minibatch_batch_size, n_train) : 0

    local post, ps, loss_hist, fit_time
    train_ok = true
    checkpoint = checkpoint_path === nothing ? nothing :
        load_model_checkpoint(checkpoint_path, identity)
    if checkpoint !== nothing
        ps = checkpoint.ps
        post = GPTrainer.gp_posterior(ps; jitter=s.jitter)
        fit_time = Float64(checkpoint.fit_time_s)
        loss_hist = [Float64(checkpoint.final_loss)]
        verbose && println("  GP checkpoint reused: $checkpoint_path")
    else
        try
            fit_time = @elapsed (post, ps, loss_hist) = GPTrainer.train_gp(
                X_fit, yt.normalized;
                M=s.inducing_points, epochs=s.epochs, lr=s.learning_rate, seed=s.seed,
                init_variance=s.init_variance, init_lengthscale=s.init_lengthscale,
                init_noise_std=s.init_noise_std, jitter=s.jitter, batch_size=eff_batch,
                print_every=verbose ? 50 : 0)
            train_ok = isfinite(loss_hist[end])
            train_ok && checkpoint_path !== nothing && write_model_checkpoint(
                checkpoint_path, identity,
                (ps=ps, fit_time_s=fit_time, final_loss=Float64(loss_hist[end])))
        catch err
            train_ok = false
            loss_hist = [NaN]
            fit_time = NaN
            @warn "GP training failed for cell $(cell.name)/$(cell.tier_label)" exception=(err, catch_backtrace())
        end
    end
    final_loss = Float64(loss_hist[end])

    n_q = prep.n_query
    rows = Vector{ArmQueryRow}(undef, n_q)
    for i in 1:n_q
        if train_ok
            t0 = time()
            m, lo, hi = GPTrainer.gp_predict(post, Float64.(view(X_test, :, i)); z_score=s.z_score)
            el = time() - t0
            pred_i  = yt.mean + yt.std * m
            lower_i = yt.mean + yt.std * lo
            upper_i = yt.mean + yt.std * hi
            status_i = (isfinite(pred_i) && isfinite(lower_i) && isfinite(upper_i)) ?
                       :COMPLETED : :GP_PREDICT_NONFINITE
        else
            pred_i, lower_i, upper_i, el, status_i = NaN, NaN, NaN, 0.0, :GP_TRAIN_FAILED
        end
        rows[i] = ArmQueryRow(fx.cell_id, i, :gp, prep.X_query[:, i], prep.y_query_truth[i],
                               pred_i, lower_i, upper_i, status_i, el)
    end

    result = arm_cell_result_from_rows(rows; cell_id=fx.cell_id, cell=cell, arm=:gp,
        model_type="SVGP", fixture_sha256=fx.fixture_sha256, settings_hash=settings_hash,
        model_seed=s.seed, noise_seed=fx.noise_seed, n_train=n_train,
        fit_time_s=fit_time, final_loss=final_loss)
    return result, rows
end
