# Subprocess worker for the :rbf and :idw comparator arms.
#
#   julia --startup-file=no --project=test/ml/baselines/baseline_env \
#         test/ml/baselines/baseline_arm_worker.jl \
#         --arm=rbf --fixture=<path> --out=<prefix> [--checkpoint=<path>]
#
# Structural twin of test/ml/nn/nn_arm_worker.jl: same CLI shape, same
# fragment-writing contract (`<prefix>.points.arrow` + `<prefix>.summary.csv`),
# same failure reporting (a refused or thrown fit is recorded as NaN
# predictions carrying an `<ARM>_TRAIN_FAILED` status on every query row, never
# as a crashed process). The only deliberate CLI difference is the mandatory
# `--arm=` selector, because one worker serves two arms.
#
# DATA ONLY. The fixture is the sole source of training data. The only outside
# input is the benchmark function's domain box `bf.lb` / `bf.ub`, used exactly
# as the :nn and :gp arms use it, for input normalisation to [-1, 1].
#
# NO CERTIFICATE LANGUAGE. Both arms produce point predictions only; `lower`
# and `upper` are NaN on every row and the interval kind is `:none`. That is an
# absence of an interval, not a failed one.

using Printf, Statistics, Random, SHA

include(joinpath(@__DIR__, "..", "common", "fixture.jl"))
using .SmokeFixture
include(joinpath(@__DIR__, "..", "common", "point_fragments.jl"))
using .PointFragments
include(joinpath(@__DIR__, "..", "common", "model_checkpoints.jl"))
using .ModelCheckpoints
include(joinpath(@__DIR__, "..", "common", "driver.jl"))
using .BenchmarkDriver
include(joinpath(@__DIR__, "surrogates_adapter.jl"))
using .SurrogateArms

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

# A failure reason is free text from `showerror`; it has to survive one CSV
# cell. Flatten to a single comma-free line rather than quoting, so the
# two-line summary stays trivially parseable by `split(line, ',')` the way the
# NN worker's consumer already parses it.
function _csv_safe(text::AbstractString)
    cleaned = replace(String(text), '\r' => ' ', '\n' => "; ", ',' => ';', '"' => '\'')
    return length(cleaned) > 480 ? first(cleaned, 477) * "..." : cleaned
end

function _settings_for_arm(arm::Symbol, dim::Int)
    arm === :rbf && return SurrogateArms.rbf_settings(dim)
    arm === :idw && return SurrogateArms.idw_settings(dim)
    throw(ArgumentError("baseline_arm_worker: unknown --arm=$arm (expected one of $(SurrogateArms.SURROGATE_ARMS))"))
end

function main()
    opts = _parse_opts(ARGS)
    haskey(opts, "arm") || error("baseline_arm_worker.jl: missing required --arm=<rbf|idw>")
    haskey(opts, "fixture") || error("baseline_arm_worker.jl: missing required --fixture=<path>")
    haskey(opts, "out") || error("baseline_arm_worker.jl: missing required --out=<path prefix>")
    arm = Symbol(lowercase(opts["arm"]))
    arm in SurrogateArms.SURROGATE_ARMS ||
        error("baseline_arm_worker.jl: --arm=$(opts["arm"]) is not a surrogate arm " *
              "(expected one of $(SurrogateArms.SURROGATE_ARMS))")
    fixture_path = opts["fixture"]
    out_prefix   = opts["out"]
    # Unlike the NN worker this defaults to NO checkpoint: :idw has no fitting
    # step at all and :rbf's fit is a single dense solve, so a checkpoint buys
    # little and would serialise a package object for no gain. Honoured when
    # the caller passes one, so the CLI stays interchangeable with the NN
    # worker's.
    checkpoint_path = get(opts, "checkpoint", "")

    # The settings hash MUST be computed here. `surrogate_settings_string`
    # folds in the loaded Surrogates.jl version, which reads "unavailable" in
    # any process that lacks the package (see the WIRING HAZARD note in
    # surrogates_adapter.jl). This process has it; the orchestrator may not.
    SurrogateArms.surrogates_available() ||
        error("baseline_arm_worker.jl: Surrogates.jl did not load. Run this worker under " *
              "--project=test/ml/baselines/baseline_env.")

    fx = SmokeFixture.read_fixture(fixture_path)

    dim = fx.dim
    name = Symbol(fx.name)
    bf = only(b for b in BenchmarkDriver.BENCHMARK_FUNCTIONS if b.name == name)

    s = _settings_for_arm(arm, dim)
    model_seed = s.seed
    settings_hash = bytes2hex(sha256(SurrogateArms.surrogate_settings_string(s)))
    model_type = SurrogateArms.surrogate_model_type(arm)
    interval_kind = SurrogateArms.surrogate_interval_kind(arm)

    # Identical framing to the NN arm: inputs normalised to [-1, 1] with the
    # shared domain box, target z-scored on the noisy training values only.
    X_fit  = BenchmarkDriver.normalize_inputs_pm1(bf, fx.x_train)
    X_test = BenchmarkDriver.normalize_inputs_pm1(bf, fx.x_query)
    yt     = BenchmarkDriver.target_zscore(fx.y_train_noisy)
    lb = fill(-1.0, dim)
    ub = fill(1.0, dim)

    n_train = size(X_fit, 2)
    dense_bytes = arm === :rbf ? SurrogateArms.rbf_dense_system_bytes(n_train) : 0.0
    max_fit_bytes = arm === :rbf ? s.max_fit_bytes : Inf

    println("baseline_arm_worker: arm=$(arm) cell=$(fx.name)/$(fx.layout)/$(fx.tier_label)  " *
            "dim=$dim  n_train=$n_train  model=$(model_type)  " *
            "surrogates=$(SurrogateArms.surrogates_version())  seed=$(model_seed)" *
            (arm === :rbf ? "  dense_system=$(round(dense_bytes / 1024^3; digits=3)) GiB" : ""))
    flush(stdout)

    identity = checkpoint_identity(fx.cell_id, arm, model_seed,
                                   fx.fixture_sha256, settings_hash)
    # `load_model_checkpoint` already returns `nothing` for a missing, stale or
    # corrupt file; the extra `hasproperty` guard rejects a payload written by
    # a different arm's worker that happened to collide on the identity key.
    checkpoint = isempty(checkpoint_path) ? nothing :
        load_model_checkpoint(checkpoint_path, identity)
    (checkpoint !== nothing && hasproperty(checkpoint, :model) &&
     hasproperty(checkpoint, :fit_time_s)) || (checkpoint = nothing)

    local model
    fit_time = NaN
    train_ok = true
    failure_reason = ""
    if checkpoint !== nothing
        model = checkpoint.model
        fit_time = Float64(checkpoint.fit_time_s)
        println("baseline_arm_worker: reused checkpoint $checkpoint_path")
    else
        model, fit_time, _, train_ok, failure = SurrogateArms.train_surrogate_arm_detailed(
            arm, X_fit, yt.normalized, s; lb=lb, ub=ub)
        failure_reason = failure === nothing ? "" : String(failure)
        if !train_ok
            println(stderr, "baseline_arm_worker: $(arm) training FAILED: ", failure_reason)
        elseif !isempty(checkpoint_path)
            try
                write_model_checkpoint(checkpoint_path, identity,
                    (model=model, fit_time_s=fit_time))
            catch err
                # A checkpoint is an optimisation. Losing it must never turn a
                # successful cell into a failed one.
                @warn "baseline_arm_worker: could not write checkpoint $checkpoint_path" exception=err
            end
        end
    end

    n_q = size(X_test, 2)
    pred    = fill(NaN, n_q)
    elapsed = zeros(n_q)
    status  = fill(train_ok ? :COMPLETED : SurrogateArms.train_failed_status(arm), n_q)
    if train_ok
        nonfinite_status = SurrogateArms.predict_nonfinite_status(arm)
        for i in 1:n_q
            t0 = time()
            p, _, _, st = SurrogateArms.predict_surrogate_arm(
                arm, model, Float64.(view(X_test, :, i)))
            elapsed[i] = time() - t0
            if st === :COMPLETED
                pred[i] = yt.mean + yt.std * p
                isfinite(pred[i]) || (status[i] = nonfinite_status)
            else
                status[i] = st
            end
        end
    end

    write_point_fragment(out_prefix * ".points.arrow", [
        (query_id=Int(fx.query_ids[i]), pred=pred[i], status=String(status[i]),
         elapsed_s=elapsed[i])
        for i in 1:n_q
    ])
    # `final_loss` is NaN for both arms: neither minimises a scalar training
    # objective. Same convention the Diona arm uses.
    _atomic_write(out_prefix * ".summary.csv") do io
        println(io, "fit_time_s,final_loss,arm,model_type,interval_kind,seed,n_train,dim," *
                    "train_ok,failure_reason,surrogates_version,max_fit_bytes," *
                    "dense_system_bytes,fixture_sha256,settings_hash")
        println(io, "$(fit_time),$(NaN),$(arm),$(model_type),$(interval_kind),$(model_seed)," *
                    "$(n_train),$(dim),$(train_ok),$(_csv_safe(failure_reason))," *
                    "$(SurrogateArms.surrogates_version()),$(max_fit_bytes)," *
                    "$(dense_bytes),$(fx.fixture_sha256),$(settings_hash)")
    end

    @printf("baseline_arm_worker: DONE  arm=%s  train_ok=%s  fit_time=%.4gs  n_query=%d  n_finite_pred=%d/%d\n",
            arm, train_ok, fit_time, n_q, count(isfinite, pred), n_q)
end

main()
