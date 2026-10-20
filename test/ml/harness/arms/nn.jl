# The :nn arm (tanh MLP), in-process and nn_env subprocess paths.
#
# Plain `include`d FRAGMENT, not a module -- every name stays top-level in the
# including scope. Pulled in by ml/harness/arms.jl, which smoke_data.jl includes.
#
# Split per arm so comparative_code_provenance can hash ONE arm's runner: editing
# this file rotates only :nn, not all six.
#
# DEVICE AND PRECISION. The device comes from --device (cpu or cuda) and the
# working precision from ComparatorRegistry's NNSettings.precision. Both are
# folded into the :nn settings hash (ComparatorRegistry.nn_settings_hash), so a
# CPU fragment and a CUDA fragment key to different fragment and checkpoint
# paths and can never be silently resumed from or compared against each other.
# Everything this file hands back to the harness is Float64: the working
# precision lives strictly inside NNTrainer.

function _run_nn_arm_inprocess(fx, cell::SmokeCell, s, settings_hash::String;
                               verbose::Bool=false, checkpoint_path=nothing,
                               device::String="cpu")
    prep = prepared_cell_from_fixture(fx)
    bf = _BENCH_FN[cell.name]
    X_fit  = BenchmarkDriver.normalize_inputs_pm1(bf, prep.X_train)
    X_test = BenchmarkDriver.normalize_inputs_pm1(bf, prep.X_query)
    yt     = BenchmarkDriver.target_zscore(prep.y_train_noisy)
    n_train = size(X_fit, 2)
    eff_batch = n_train > s.minibatch_threshold ? min(s.minibatch_batch_size, n_train) : 0

    identity = checkpoint_identity(fx.cell_id, :nn, s.seed,
                                   fx.fixture_sha256, settings_hash)
    checkpoint = checkpoint_path === nothing ? nothing :
        load_model_checkpoint(checkpoint_path, identity)
    local model, ps, st, loss_hist, fit_time
    train_ok = true
    if checkpoint !== nothing
        model = NNTrainer.build_mlp(cell.dim; hidden_width=s.width,
                                    n_hidden_layers=s.layers)
        ps, st = checkpoint.ps, checkpoint.st
        fit_time = Float64(checkpoint.fit_time_s)
        loss_hist = [Float64(checkpoint.final_loss)]
        verbose && println("  NN checkpoint reused: $checkpoint_path")
    else
        try
            fit_time = @elapsed (model, ps, st, loss_hist) = NNTrainer.train_nn(
                X_fit, yt.normalized;
                hidden_width=s.width, n_hidden_layers=s.layers,
                epochs=s.epochs, lr=s.learning_rate, seed=s.seed,
                precision=ComparatorRegistry.float_type(s), device=device,
                batch_size=eff_batch, print_every=verbose ? 500 : 0)
            train_ok = isfinite(loss_hist[end])
            train_ok && checkpoint_path !== nothing && write_model_checkpoint(
                checkpoint_path, identity,
                (ps=ps, st=st, fit_time_s=fit_time,
                 final_loss=Float64(loss_hist[end])))
        catch err
            train_ok = false
            loss_hist = [NaN]
            fit_time = NaN
            @warn "NN training failed for cell $(cell.name)/$(cell.tier_label)" exception=(err, catch_backtrace())
        end
    end
    final_loss = Float64(loss_hist[end])

    n_q = prep.n_query
    rows = Vector{ArmQueryRow}(undef, n_q)
    for i in 1:n_q
        if train_ok
            t0 = time()
            p = NNTrainer.nn_predict(model, ps, st, Float64.(view(X_test, :, i)))
            el = time() - t0
            pred_i = yt.mean + yt.std * p
            status_i = isfinite(pred_i) ? :COMPLETED : :NN_PREDICT_NONFINITE
        else
            pred_i, el, status_i = NaN, 0.0, :NN_TRAIN_FAILED
        end
        rows[i] = ArmQueryRow(fx.cell_id, i, :nn, prep.X_query[:, i], prep.y_query_truth[i],
                               pred_i, NaN, NaN, status_i, el)
    end
    result = arm_cell_result_from_rows(rows; cell_id=fx.cell_id, cell=cell, arm=:nn,
        model_type="MLP", fixture_sha256=fx.fixture_sha256, settings_hash=settings_hash,
        model_seed=s.seed, noise_seed=fx.noise_seed, n_train=n_train,
        fit_time_s=fit_time, final_loss=final_loss)
    return result, rows
end

function _run_nn_arm_subprocess(fx, cell::SmokeCell, fixture_path::AbstractString,
                                 out_dir::AbstractString, settings_hash::String, model_seed::Int;
                                 verbose::Bool=false, checkpoint_path=nothing,
                                 device::String="cpu")
    stem = cell_stem(cell, fx.cell_id)
    out_prefix = joinpath(out_dir, "arm_fragments", "nn_" * stem)
    mkpath(dirname(out_prefix))
    julia = Base.julia_cmd()
    checkpoint = checkpoint_path === nothing ? out_prefix * ".checkpoint.jls" : String(checkpoint_path)
    cmd = `$julia --startup-file=no --project=$(NN_ENV_PROJECT) $(NN_ARM_WORKER_PATH) --fixture=$(fixture_path) --out=$(out_prefix) --checkpoint=$(checkpoint) --device=$(device)`
    log_path = out_prefix * ".log"
    verbose && println("  [nn subprocess] ", cmd)
    exitcode = open(log_path, "w") do io
        proc = run(pipeline(ignorestatus(cmd); stdout=io, stderr=io); wait=true)
        proc.exitcode
    end
    exitcode == 0 || @warn "nn_arm_worker.jl subprocess exited nonzero (code=$exitcode); see $log_path"

    summary_path = out_prefix * ".summary.csv"
    points_path  = out_prefix * ".points.arrow"
    (isfile(summary_path) && isfile(points_path)) ||
        error("run_nn_arm_cell: NN worker subprocess did not produce expected output " *
              "($summary_path / $points_path, exitcode=$exitcode); see $log_path")

    slines = readlines(summary_path)
    shdr = split(slines[1], ',')
    sval = split(slines[2], ',')
    sd = Dict(Symbol(h) => v for (h, v) in zip(shdr, sval))
    fit_time   = _parse_float(sd[:fit_time_s])
    final_loss = _parse_float(sd[:final_loss])
    # The worker recomputes the settings hash from the registry and the --device
    # it was handed; this result is labelled with the hash computed HERE. If the
    # two ever disagreed -- a device that did not survive the command line, a
    # registry the subprocess resolved differently -- the fragment would carry a
    # label its own fit does not match. Check rather than assume.
    String(get(sd, :settings_hash, "")) == settings_hash ||
        error("run_nn_arm_cell: NN worker reported settings_hash=" *
              "$(get(sd, :settings_hash, "<absent>")) but this process expected " *
              "$settings_hash (device=$device); see $log_path")
    String(get(sd, :device, "")) == device ||
        error("run_nn_arm_cell: NN worker reported device=" *
              "$(get(sd, :device, "<absent>")) but was asked for $device; see $log_path")

    point_records = read_point_fragment(points_path)
    prep = prepared_cell_from_fixture(fx)
    n_q  = prep.n_query
    rows = Vector{ArmQueryRow}(undef, n_q)
    for pd in point_records
        qid    = _point_int(pd[:query_id])
        pred_i = _parse_float(pd[:pred])
        status_i = Symbol(_point_string(pd[:status]))
        el = _parse_float(pd[:elapsed_s])
        rows[qid] = ArmQueryRow(fx.cell_id, qid, :nn, prep.X_query[:, qid],
                                 prep.y_query_truth[qid], pred_i, NaN, NaN, status_i, el)
    end
    all(isassigned(rows, i) for i in 1:n_q) ||
        error("run_nn_arm_cell: NN worker points.arrow is missing one or more query_id rows " *
              "(expected exactly 1:$n_q); see $points_path")

    result = arm_cell_result_from_rows(rows; cell_id=fx.cell_id, cell=cell, arm=:nn,
        model_type="MLP", fixture_sha256=fx.fixture_sha256, settings_hash=settings_hash,
        model_seed=model_seed, noise_seed=fx.noise_seed, n_train=fx.n_train_actual,
        fit_time_s=fit_time, final_loss=final_loss)
    return result, rows
end

function run_nn_arm_cell(fx, out_dir::AbstractString,
                          reg::ComparatorRegistry.CanonicalRegistry=ComparatorRegistry.canonical_registry();
                          verbose::Bool=false, checkpoint_path=nothing, device::String="cpu")
    cell = smoke_cell_from_fixture(fx)
    dev = ComparatorRegistry.normalize_device(device)

    s = ComparatorRegistry.settings_for(reg, cell.dim, :nn)
    settings_hash = ComparatorRegistry.nn_settings_hash(s, dev)
    if _NN_TRAINER_AVAILABLE
        return _run_nn_arm_inprocess(fx, cell, s, settings_hash;
                                     verbose=verbose, checkpoint_path=checkpoint_path,
                                     device=dev)
    else
        fpath, _ = fixture_paths(out_dir, cell, fx.cell_id)
        isfile(fpath) ||
            error("run_nn_arm_cell: NN arm subprocess requires an already-materialized fixture " *
                  "at $fpath (call materialize_fixture first, e.g. via run_arm_cell)")
        return _run_nn_arm_subprocess(fx, cell, fpath, out_dir, settings_hash, s.seed;
                                      verbose=verbose, checkpoint_path=checkpoint_path,
                                      device=dev)
    end
end
