# The :rbf and :idw arms (Surrogates.jl under baseline_env).
#
# Plain `include`d FRAGMENT, not a module -- every name stays top-level in the
# including scope. Pulled in by ml/harness/arms.jl, which smoke_data.jl includes.
#
# Split per arm so comparative_code_provenance can hash ONE arm's runner: editing
# this file rotates only :rbf and :idw, not all six.

# ==========================================================================
# Surrogates.jl arms (:rbf, :idw) -- fitted under
# --project=test/ml/baselines/baseline_env
#
# Neither arm produces an uncertainty interval, so lower/upper are NaN on every
# row and comparative_interval_kind is :none. That is an absence, not a failure.
# ==========================================================================

function _surrogate_settings_for(dim::Int, arm::Symbol)
    _SURROGATE_ADAPTER_AVAILABLE || error("the :$arm arm needs SurrogateArms " *
        "(test/ml/baselines/surrogates_adapter.jl), which did not load in this process.")
    # :rbf runs at ONE precision on every cell, so the settings do not depend on
    # cell size and the hash is the same for every cell of a given dimension.
    arm === :rbf && return SurrogateArms.rbf_settings(dim)
    arm === :idw && return SurrogateArms.idw_settings(dim)
    error("_surrogate_settings_for: unknown surrogate arm=$arm (expected :rbf or :idw)")
end

const _SURROGATE_SETTINGS_HASH_CACHE = Dict{Tuple{Int,Symbol},String}()

"""
    _surrogate_settings_hash(dim, arm) -> String

sha256 of `surrogate_settings_string`, which EMBEDS THE LOADED Surrogates.jl
VERSION (surrogates_adapter.jl:394, "WIRING HAZARD"). A process without the
package records `"unavailable"` and would produce a hash that can never match
the worker that actually fits, so when the package is absent here the hash is
obtained from a one-shot probe under `BASELINE_ENV_PROJECT` -- the same
environment the fit runs in -- and memoised for the campaign.
"""
function _surrogate_settings_hash(dim::Int, arm::Symbol)
    if _SURROGATE_ADAPTER_AVAILABLE && SurrogateArms.surrogates_available()
        return bytes2hex(sha256(
            SurrogateArms.surrogate_settings_string(
                _surrogate_settings_for(dim, arm))))
    end
    return get!(_SURROGATE_SETTINGS_HASH_CACHE, (dim, arm)) do
        adapter = joinpath(TEST_DIR, "ml", "baselines", "surrogates_adapter.jl")
        ctor = arm === :rbf ? "rbf_settings" :
               arm === :idw ? "idw_settings" :
               error("_surrogate_settings_hash: unknown surrogate arm=$arm")
        code = "using SHA\n" *
               "include(" * repr(adapter) * ")\n" *
               "using .SurrogateArms\n" *
               "SurrogateArms.surrogates_available() || error(\"Surrogates.jl is not " *
               "available under \" * string(Base.active_project()))\n" *
               "s = " * ctor * "(" * string(dim) * ")\n" *
               "println(\"SETTINGS_HASH=\", bytes2hex(sha256(surrogate_settings_string(s))))\n"
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(BASELINE_ENV_PROJECT) -e $code`
        output = try
            read(cmd, String)
        catch err
            error("could not obtain the :$arm settings hash from $(BASELINE_ENV_PROJECT): " *
                  sprint(showerror, err) * ". Instantiate it first: " *
                  "julia --project=test/ml/baselines/baseline_env -e 'using Pkg; Pkg.instantiate()'")
        end
        matched = match(r"SETTINGS_HASH=([0-9a-f]{64})", output)
        matched === nothing &&
            error("the :$arm settings-hash probe under $(BASELINE_ENV_PROJECT) printed no " *
                  "SETTINGS_HASH line; got: " * output)
        String(matched.captures[1])
    end
end

function _run_surrogate_arm_inprocess(fx, cell::SmokeCell, arm::Symbol, s,
                                      settings_hash::String; verbose::Bool=false)
    prep = prepared_cell_from_fixture(fx)
    bf = _BENCH_FN[cell.name]
    X_fit  = BenchmarkDriver.normalize_inputs_pm1(bf, prep.X_train)
    X_test = BenchmarkDriver.normalize_inputs_pm1(bf, prep.X_query)
    yt     = BenchmarkDriver.target_zscore(prep.y_train_noisy)
    dim = size(X_fit, 1)
    lb = fill(-1.0, dim)   # normalize_inputs_pm1 frame
    ub = fill(1.0, dim)

    model, fit_time, final_loss, train_ok, failure =
        SurrogateArms.train_surrogate_arm_detailed(arm, X_fit, yt.normalized, s; lb=lb, ub=ub)
    train_ok || @warn "$arm training failed for cell $(cell.name)/$(cell.tier_label)" failure

    n_q = prep.n_query
    rows = Vector{ArmQueryRow}(undef, n_q)
    for i in 1:n_q
        if train_ok
            t0 = time()
            p, _, _, st = SurrogateArms.predict_surrogate_arm(
                arm, model, Float64.(view(X_test, :, i)))
            el = time() - t0
            # Kept step-for-step identical to baseline_arm_worker.jl's loop, so
            # the arm's numbers do not depend on which of the two paths ran.
            if st === :COMPLETED
                pred_i = yt.mean + yt.std * p
                status_i = isfinite(pred_i) ? :COMPLETED :
                           SurrogateArms.predict_nonfinite_status(arm)
            else
                pred_i, status_i = NaN, st
            end
        else
            pred_i, el, status_i = NaN, 0.0, SurrogateArms.train_failed_status(arm)
        end
        rows[i] = ArmQueryRow(fx.cell_id, i, arm, prep.X_query[:, i], prep.y_query_truth[i],
                               pred_i, NaN, NaN, status_i, el)
    end

    result = arm_cell_result_from_rows(rows; cell_id=fx.cell_id, cell=cell, arm=arm,
        model_type=SurrogateArms.surrogate_model_type(arm), fixture_sha256=fx.fixture_sha256,
        settings_hash=settings_hash, model_seed=s.seed, noise_seed=fx.noise_seed,
        n_train=size(X_fit, 2), fit_time_s=fit_time, final_loss=final_loss)
    return result, rows
end

"""
    _run_surrogate_arm_subprocess(fx, cell, arm, fixture_path, out_dir, settings_hash, model_seed)

INTERFACE of `test/ml/baselines/baseline_arm_worker.jl` (owned by another
agent), mirroring `test/ml/nn/nn_arm_worker.jl`. Assumed when this was written
and since checked against the landed worker:

    julia --startup-file=no --project=<baseline_env> baseline_arm_worker.jl
        --fixture=<fixture path> --arm=<rbf|idw> --out=<prefix> --settings-hash=<hex>

writes exactly two files and exits 0:

  * `<prefix>.summary.csv`  -- header line + ONE data row, containing at least
    the columns `fit_time_s`, `final_loss` and `settings_hash`.
  * `<prefix>.points.arrow` -- one record per query with the columns
    `query_id`, `pred`, `status`, `elapsed_s`; `pred` is in the ORIGINAL target
    units (the worker owns the z-score round trip, as nn_arm_worker.jl does),
    `status` is `:COMPLETED` / `<ARM>_TRAIN_FAILED` / `<ARM>_PREDICT_NONFINITE`.

`settings_hash`, when present in the summary, is checked against the parent's:
`surrogate_settings_string` embeds the loaded Surrogates.jl version, so a
mismatch means the two processes disagree about the package and must not be
papered over.
"""
function _run_surrogate_arm_subprocess(fx, cell::SmokeCell, arm::Symbol,
                                       fixture_path::AbstractString, out_dir::AbstractString,
                                       settings_hash::String, model_seed::Int;
                                       verbose::Bool=false)
    isfile(BASELINE_ARM_WORKER_PATH) ||
        error("run_surrogate_arm_cell: the :$arm arm needs the baseline worker at " *
              "$BASELINE_ARM_WORKER_PATH, which does not exist. Either add it or run the " *
              "arm under --project=test/ml/baselines/baseline_env, where it runs in-process.")
    stem = cell_stem(cell, fx.cell_id)
    out_prefix = joinpath(out_dir, "arm_fragments", "$(arm)_" * stem)
    mkpath(dirname(out_prefix))
    julia = Base.julia_cmd()
    cmd = `$julia --startup-file=no --project=$(BASELINE_ENV_PROJECT) $(BASELINE_ARM_WORKER_PATH) --fixture=$(fixture_path) --arm=$(arm) --out=$(out_prefix) --settings-hash=$(settings_hash)`
    log_path = out_prefix * ".log"
    verbose && println("  [$arm subprocess] ", cmd)
    exitcode = open(log_path, "w") do io
        proc = run(pipeline(ignorestatus(cmd); stdout=io, stderr=io); wait=true)
        proc.exitcode
    end
    exitcode == 0 ||
        @warn "baseline_arm_worker.jl subprocess exited nonzero (code=$exitcode); see $log_path"

    summary_path = out_prefix * ".summary.csv"
    points_path  = out_prefix * ".points.arrow"
    (isfile(summary_path) && isfile(points_path)) ||
        error("run_surrogate_arm_cell: the $arm worker subprocess did not produce expected " *
              "output ($summary_path / $points_path, exitcode=$exitcode); see $log_path")

    slines = readlines(summary_path)
    shdr = split(slines[1], ',')
    sval = split(slines[2], ',')
    sd = Dict(Symbol(h) => v for (h, v) in zip(shdr, sval))
    fit_time   = _parse_float(sd[:fit_time_s])
    final_loss = _parse_float(sd[:final_loss])
    if haskey(sd, :settings_hash)
        String(sd[:settings_hash]) == settings_hash ||
            error("run_surrogate_arm_cell: the $arm worker reported settings_hash=" *
                  "$(sd[:settings_hash]) but the parent expected $settings_hash. " *
                  "surrogate_settings_string embeds the loaded Surrogates.jl version, so the " *
                  "two processes disagree about the package; see $log_path")
    end

    point_records = read_point_fragment(points_path)
    prep = prepared_cell_from_fixture(fx)
    n_q  = prep.n_query
    rows = Vector{ArmQueryRow}(undef, n_q)
    for pd in point_records
        qid = _point_int(pd[:query_id])
        rows[qid] = ArmQueryRow(fx.cell_id, qid, arm, prep.X_query[:, qid],
                                 prep.y_query_truth[qid], _parse_float(pd[:pred]), NaN, NaN,
                                 Symbol(_point_string(pd[:status])), _parse_float(pd[:elapsed_s]))
    end
    all(isassigned(rows, i) for i in 1:n_q) ||
        error("run_surrogate_arm_cell: the $arm worker points.arrow is missing one or more " *
              "query_id rows (expected exactly 1:$n_q); see $points_path")

    result = arm_cell_result_from_rows(rows; cell_id=fx.cell_id, cell=cell, arm=arm,
        model_type=SurrogateArms.surrogate_model_type(arm), fixture_sha256=fx.fixture_sha256,
        settings_hash=settings_hash, model_seed=model_seed, noise_seed=fx.noise_seed,
        n_train=fx.n_train_actual, fit_time_s=fit_time, final_loss=final_loss)
    return result, rows
end

function run_surrogate_arm_cell(fx, arm::Symbol, out_dir::AbstractString; verbose::Bool=false)
    arm in (:rbf, :idw) ||
        error("run_surrogate_arm_cell: arm=$arm is not a Surrogates.jl arm (expected :rbf or :idw)")
    cell = smoke_cell_from_fixture(fx)
    arm in applicable_arms(cell) ||
        error("run_surrogate_arm_cell: arm=$arm is excluded on cell $(cell.name)/" *
              "$(cell.tier_label) by applicable_arms (see its docstring)")
    s = _surrogate_settings_for(cell.dim, arm)
    if _SURROGATE_ADAPTER_AVAILABLE && SurrogateArms.surrogates_available()
        settings_hash = bytes2hex(sha256(SurrogateArms.surrogate_settings_string(s)))
        return _run_surrogate_arm_inprocess(fx, cell, arm, s, settings_hash; verbose=verbose)
    end
    settings_hash = _surrogate_settings_hash(cell.dim, arm)
    fpath, _ = fixture_paths(out_dir, cell, fx.cell_id)
    isfile(fpath) ||
        error("run_surrogate_arm_cell: the $arm subprocess requires an already-materialized " *
              "fixture at $fpath (call materialize_fixture first, e.g. via run_arm_cell)")
    return _run_surrogate_arm_subprocess(fx, cell, arm, fpath, out_dir, settings_hash, s.seed;
                                         verbose=verbose)
end
