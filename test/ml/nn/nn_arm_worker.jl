
using Printf, Statistics, Random, SHA

include(joinpath(@__DIR__, "..", "common", "fixture.jl"))
using .SmokeFixture
include(joinpath(@__DIR__, "..", "common", "point_fragments.jl"))
using .PointFragments
include(joinpath(@__DIR__, "..", "common", "model_checkpoints.jl"))
using .ModelCheckpoints
include(joinpath(@__DIR__, "..", "common", "registry.jl"))
using .ComparatorRegistry
include(joinpath(@__DIR__, "..", "common", "driver.jl"))
using .BenchmarkDriver
include(joinpath(@__DIR__, "trainer.jl"))
using .NNTrainer

function _parse_opts(args)
    opts = Dict{String,String}()
    for a in args
        startswith(a, "--") || continue
        item = a[3:end]
        if occursin('=', item)
            k, v = split(item, '='; limit=2)
            opts[k] = v
        else
            opts[item] = "true"
        end
    end
    return opts
end

function _atomic_write(writer::Function, path::String)
    mkpath(dirname(path))
    tmp = path * ".tmp-$(getpid())"
    open(writer, tmp, "w")
    mv(tmp, path; force=true)
    return path
end

function main()
    opts = _parse_opts(ARGS)
    haskey(opts, "fixture") || error("nn_arm_worker.jl: missing required --fixture=<path>")
    haskey(opts, "out") || error("nn_arm_worker.jl: missing required --out=<path prefix>")
    fixture_path = opts["fixture"]
    out_prefix   = opts["out"]
    checkpoint_path = get(opts, "checkpoint", out_prefix * ".checkpoint.jls")
    device = ComparatorRegistry.normalize_device(get(opts, "device", "cpu"))

    fx = SmokeFixture.read_fixture(fixture_path)

    dim = fx.dim
    name = Symbol(fx.name)
    bf = only(b for b in BenchmarkDriver.BENCHMARK_FUNCTIONS if b.name == name)

    reg = ComparatorRegistry.canonical_registry()
    s = ComparatorRegistry.settings_for(reg, dim, :nn)
    model_seed = s.seed
    epochs = s.epochs
    precision = ComparatorRegistry.float_type(s)
    # Device-aware, and identical to the hash ml/harness/arms/nn.jl computes --
    # the campaign rejects a fragment whose settings hash it did not expect.
    settings_hash = ComparatorRegistry.nn_settings_hash(s, device)
    X_fit  = BenchmarkDriver.normalize_inputs_pm1(bf, fx.x_train)
    X_test = BenchmarkDriver.normalize_inputs_pm1(bf, fx.x_query)
    yt     = BenchmarkDriver.target_zscore(fx.y_train_noisy)

    n_train = size(X_fit, 2)
    eff_batch = n_train > s.minibatch_threshold ? min(s.minibatch_batch_size, n_train) : 0

    println("nn_arm_worker: cell=$(fx.name)/$(fx.layout)/$(fx.tier_label)  dim=$dim  " *
            "n_train=$n_train  width=$(s.width) layers=$(s.layers) epochs=$(epochs) " *
            "model_seed=$(model_seed) device=$(device) precision=$(s.precision) " *
            "batch_size=$(eff_batch == 0 ? "full" : eff_batch)")
    flush(stdout)

    identity = checkpoint_identity(fx.cell_id, :nn, model_seed,
                                   fx.fixture_sha256, settings_hash)
    checkpoint = load_model_checkpoint(checkpoint_path, identity)
    local model, ps, st, loss_history, fit_time
    init_hash = Ref("")
    train_ok = true
    if checkpoint !== nothing
        model = NNTrainer.build_mlp(dim; hidden_width=s.width,
                                    n_hidden_layers=s.layers)
        ps, st = checkpoint.ps, checkpoint.st
        fit_time = Float64(checkpoint.fit_time_s)
        loss_history = [Float64(checkpoint.final_loss)]
        init_hash[] = String(get(checkpoint, :init_hash, ""))
        println("nn_arm_worker: reused checkpoint $checkpoint_path")
    else
        try
            fit_time = @elapsed (model, ps, st, loss_history) = NNTrainer.train_nn(
                X_fit, yt.normalized;
                hidden_width=s.width, n_hidden_layers=s.layers,
                epochs=epochs, lr=s.learning_rate, seed=model_seed,
                precision=precision, device=device,
                batch_size=eff_batch, print_every=0,
                init_callback=ps0 -> (init_hash[] = NNTrainer.nn_params_hash(ps0)))
            train_ok = isfinite(loss_history[end])
            train_ok && write_model_checkpoint(checkpoint_path, identity,
                (ps=ps, st=st, fit_time_s=fit_time,
                 final_loss=Float64(loss_history[end]), init_hash=init_hash[]))
        catch err
            train_ok = false
            loss_history = [NaN]
            fit_time = NaN
            println(stderr, "nn_arm_worker: training FAILED: ", sprint(showerror, err))
            Base.show_backtrace(stderr, catch_backtrace())
        end
    end
    final_loss = Float64(loss_history[end])

    n_q = size(X_test, 2)
    pred    = fill(NaN, n_q)
    elapsed = zeros(n_q)
    status  = fill(train_ok ? :COMPLETED : :NN_TRAIN_FAILED, n_q)
    if train_ok
        for i in 1:n_q
            t0 = time()
            p = NNTrainer.nn_predict(model, ps, st, Float64.(view(X_test, :, i)))
            elapsed[i] = time() - t0
            pred[i] = yt.mean + yt.std * p
            isfinite(pred[i]) || (status[i] = :NN_PREDICT_NONFINITE)
        end
    end

    write_point_fragment(out_prefix * ".points.arrow", [
        (query_id=Int(fx.query_ids[i]), pred=pred[i], status=String(status[i]),
         elapsed_s=elapsed[i])
        for i in 1:n_q
    ])
    _atomic_write(out_prefix * ".summary.csv") do io
        println(io, "fit_time_s,final_loss,width,layers,epochs,seed,batch_size,train_ok,model_type,backend,precision,device,eta_max,eta_min,schedule,init_hash,nonfinite_gradient_count,fixture_sha256,settings_hash")
        println(io, "$(fit_time),$(final_loss),$(s.width),$(s.layers),$(epochs),$(model_seed)," *
                     "$(eff_batch),$(train_ok),MLP,Lux,$(s.precision),$(device)," *
                     "$(s.learning_rate)," *
                     "$(s.learning_rate)," *
                     "constant,$(init_hash[]),0,$(fx.fixture_sha256)," *
                     "$(settings_hash)")
    end

    @printf("nn_arm_worker: DONE  train_ok=%s  device=%s  precision=%s  final_loss=%.6g  fit_time=%.2fs  n_query=%d  n_finite_pred=%d/%d\n",
            train_ok, device, s.precision, final_loss, fit_time, n_q, count(isfinite, pred), n_q)
end

main()
