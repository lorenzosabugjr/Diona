# Comparative campaign: provenance, fragments, workers, orchestrator, aggregates.
#
# Extracted verbatim from test/smoke_data.jl. This is a plain `include`d FRAGMENT,
# not a module: every name must stay top-level in the including scope, because
# test/run_full_benchmark.jl reaches into the harness via getproperty(harness, :name)
# and test/benchmark_data.jl mutates _SMOKE_OPTS directly. The include sits at the
# exact position the code occupied, so evaluation order is unchanged.
#
# NOTE: SCRIPT_PATH stays in smoke_data.jl -- it is abspath(@__FILE__) and is the
# target workers re-invoke, so it must resolve to the entry point, not to this file.

# v4: six arms, and the point fragment gained `smoothness_used`.
const COMPARATIVE_FRAGMENT_VERSION = "comparative-arm-cell-v4"
const COMPARATIVE_CODE_POLICY_VERSION = "comparative-code-runtime-v1"
const DEFAULT_JOBS_COMPARATIVE_AUTO = max(1, Sys.CPU_THREADS - 4)
const DEFAULT_JOBS_COMPARATIVE = Dict(
    :diona => DEFAULT_JOBS_COMPARATIVE_AUTO,
    :nn => DEFAULT_JOBS_COMPARATIVE_AUTO,
    :gp => DEFAULT_JOBS_COMPARATIVE_AUTO,
    :nsm => DEFAULT_JOBS_COMPARATIVE_AUTO,
    :rbf => max(1, DEFAULT_JOBS_COMPARATIVE_AUTO ÷ 4),
    :idw => DEFAULT_JOBS_COMPARATIVE_AUTO,
)

# NOT PROBED. The only measured number here is :diona's: the largest single
# Diona cell of the last campaign took 2.98 h against the previous 3.00 h
# default, i.e. 1.3% headroom, so it is doubled. :nsm/:rbf/:idw are order-of-
# magnitude allowances, not measurements -- the :rbf 16,000-point cell is a
# dense 16000x16000 solve (3.8 GiB, ~17 s fit measured by the adapter's own
# gate) and :nsm/:idw are O(n_train * n_query) with no fit phase, which on the
# 125,000-point grid cells is 1.25e5 * 3e3 distance evaluations. Replace each
# of these with a measured single-cell probe before a long campaign.
const DEFAULT_TIMEOUT_COMPARATIVE = Dict(
    :diona => 21600.0,   # measured: worst cell 2.98 h, was 10800.0 (1.3% headroom)
    :nn    => 1800.0,
    :gp    => 3600.0,
    :nsm   => 7200.0,    # UNPROBED
    :rbf   => 7200.0,    # UNPROBED
    :idw   => 7200.0,    # UNPROBED
)
const DEFAULT_BLAS_THREADS_COMPARATIVE = 1
# Staging order; an arm absent here silently never runs (it drives the jobs /
# timeouts loop, the per-arm stage loop and max_active).
const _COMPARATIVE_ARM_ORDER = (:diona, :nn, :gp, :nsm, :rbf, :idw)

const _COMPARATIVE_SUMMARY_FIELDS = (
    :fragment_version, :run_id, :cell_id, :arm, :terminal_state, :error,
    :fixture_schema, :fixture_sha256, :settings_hash, :code_provenance,
    :worker_project, :julia_version, :blas_threads,
    :name, :layout, :tier_label, :dim, :model_type,
    :model_seed, :noise_seed, :n_train, :n_query,
    :fit_time_s, :inference_time_total_s, :inference_time_mean_s,
    :final_loss, :n_finite_pred, :n_finite_interval,
    :rel_l2, :mae, :mse, :coverage, :mean_width, :status_counts,
)

# :nsm is pure Julia + NearestNeighbors and runs in the shared test/ml project.
# :rbf/:idw need Surrogates.jl, which only baseline_env resolves.
_comparative_project_dir(arm::Symbol) =
    arm in (:nn,) ? NN_ENV_PROJECT :
    arm in (:rbf, :idw) ? BASELINE_ENV_PROJECT : WORKER_PROJECT
_comparative_project_file(arm::Symbol) = normpath(joinpath(_comparative_project_dir(arm), "Project.toml"))

function _comparative_active_project()
    project = Base.active_project()
    return project === nothing ? "" : normpath(String(project))
end

function _comparative_model_type(arm::Symbol)
    arm == :diona && return "Diona"
    arm == :gp && return "SVGP"
    arm == :nn && return "MLP"
    if arm == :nsm
        _NSM_AVAILABLE ||
            error("_comparative_model_type(:nsm): NSMTrainer did not load in this process")
        return NSMTrainer.nsm_model_type()
    end
    if arm in (:rbf, :idw)
        _SURROGATE_ADAPTER_AVAILABLE ||
            error("_comparative_model_type(:$arm): SurrogateArms did not load in this process")
        return SurrogateArms.surrogate_model_type(arm)
    end
    error("unknown comparative arm=$arm")
end

# `device` is load-bearing for :nn ONLY. Every other arm is CPU-only, so its
# hash must NOT gain a device term -- that would rotate five settings hashes
# for a knob those arms do not have.
function _comparative_settings_hash(cell::SmokeCell, arm::Symbol,
                           reg::ComparatorRegistry.CanonicalRegistry;
                           device::String="cpu")
    if arm == :nn
        return ComparatorRegistry.nn_settings_hash(
            ComparatorRegistry.settings_for(reg, cell.dim, :nn), device)
    end
    if arm == :diona
        return bytes2hex(sha256(_diona_settings_string()))
    end
    # ComparatorRegistry.settings_for errors on an unknown arm (registry.jl:153),
    # so the baseline arms hash their own canonical settings string instead.
    arm == :nsm && return _nsm_settings_hash(cell.dim)
    arm in (:rbf, :idw) && return _surrogate_settings_hash(cell.dim, arm)
    settings = ComparatorRegistry.settings_for(reg, cell.dim, arm)
    return bytes2hex(sha256(ComparatorRegistry.canonical_serialize(settings)))
end

function _comparative_model_seed(cell::SmokeCell, arm::Symbol,
                        reg::ComparatorRegistry.CanonicalRegistry)
    arm == :diona && return _SMOKE_MODEL_SEED
    # :nsm/:rbf/:idw are fully deterministic -- no sampling, no initialisation,
    # no stochastic optimisation. The seed is carried for reporting symmetry only.
    arm in (:nsm, :rbf, :idw) && return 0
    return ComparatorRegistry.settings_for(reg, cell.dim, arm).seed
end

function comparative_code_provenance(arm::Symbol)
    # Hash exactly the files that decide what THIS arm's fragment contains.
    #
    # `ml/harness/arms/` is one file per arm precisely so this can be selective:
    # editing the NSM runner rotates :nsm alone and leaves :nn/:gp fragments
    # resumable. Shared files still rotate everything, which is correct -- they
    # really do change every arm's output.
    #
    # DELIBERATELY ABSENT, and each for a reason worth keeping:
    #   ml/harness/sweep.jl   Diona-only full sweep; writes different fragments.
    #   ml/gates/*            test code; never touches a campaign fragment.
    #   gate/gate3.jl         quarantined protocol-violating diagnostic.
    #
    # Under-listing is the dangerous direction: a behavioural edit that leaves the
    # hash unchanged means a stale fragment is silently accepted. When unsure,
    # add the file.
    arm_runner = Dict(
        :diona => "diona.jl", :gp => "gp.jl", :nn => "nn.jl",
        :nsm => "nsm.jl", :rbf => "surrogates.jl", :idw => "surrogates.jl",
    )
    haskey(arm_runner, arm) ||
        error("comparative_code_provenance: unrecognized arm=$arm")

    files = String[
        SCRIPT_PATH,
        # Shared harness: preparation, the dispatch, the arm-generic row/result
        # types, the interval-kind vocabulary, and this fragment writer.
        joinpath(TEST_DIR, "ml", "harness", "cells.jl"),
        joinpath(TEST_DIR, "ml", "harness", "arms.jl"),
        joinpath(TEST_DIR, "ml", "harness", "arms", "common.jl"),
        joinpath(TEST_DIR, "ml", "harness", "arms", "dispatch.jl"),
        joinpath(TEST_DIR, "ml", "harness", "reporting.jl"),
        joinpath(TEST_DIR, "ml", "harness", "campaign.jl"),
        joinpath(TEST_DIR, "ml", "common", "fixture.jl"),
        joinpath(TEST_DIR, "ml", "common", "point_fragments.jl"),
        joinpath(TEST_DIR, "ml", "common", "model_checkpoints.jl"),
        joinpath(TEST_DIR, "ml", "common", "registry.jl"),
        # This arm's runner ONLY.
        joinpath(TEST_DIR, "ml", "harness", "arms", arm_runner[arm]),
        joinpath(_comparative_project_dir(arm), "Project.toml"),
        joinpath(_comparative_project_dir(arm), "Manifest.toml"),
    ]
    if arm == :diona
        src_dir = joinpath(REPO_ROOT, "src")
        append!(files, sort(filter(path -> endswith(path, ".jl"),
                                   readdir(src_dir; join=true))))
    elseif arm == :gp
        push!(files, joinpath(TEST_DIR, "ml", "gp", "trainer.jl"))
        push!(files, joinpath(TEST_DIR, "ml", "common", "driver.jl"))
    elseif arm == :nsm
        push!(files, joinpath(TEST_DIR, "ml", "baselines", "nsm.jl"))
        push!(files, joinpath(TEST_DIR, "ml", "common", "driver.jl"))
    elseif arm in (:rbf, :idw)
        # gate/gate3.jl is deliberately NOT listed: it is a quarantined
        # protocol-violating diagnostic and is not part of any arm.
        append!(files, [
            joinpath(TEST_DIR, "ml", "baselines", "surrogates_adapter.jl"),
            BASELINE_ARM_WORKER_PATH,
            joinpath(TEST_DIR, "ml", "common", "driver.jl"),
        ])
    else
        append!(files, [
            joinpath(TEST_DIR, "ml", "nn", "trainer.jl"),
            joinpath(TEST_DIR, "ml", "nn", "nn_arm_worker.jl"),
            joinpath(TEST_DIR, "ml", "common", "driver.jl"),
        ])
    end
    parts = String[COMPARATIVE_CODE_POLICY_VERSION, "arm=$(arm)", "julia=$(VERSION)",
                   "machine=$(Sys.MACHINE)", "kernel=$(Sys.KERNEL)"]
    for path in sort(unique(normpath.(files)))
        isfile(path) || error("comparative provenance source is missing: $path")
        push!(parts, "$(relpath(path, REPO_ROOT))=$(bytes2hex(sha256(read(path))))")
    end
    return bytes2hex(sha256(join(parts, '\n')))
end

function comparative_arm_stem(cell::SmokeCell, id::Int, arm::Symbol,
                     fixture_sha256::AbstractString, settings_hash::AbstractString)
    return "$(cell_stem(cell, id))__$(arm)__fx$(first(String(fixture_sha256), 12))" *
           "__set$(first(String(settings_hash), 12))"
end

function comparative_fragment_paths(out_dir::AbstractString, cell::SmokeCell, id::Int,
                           arm::Symbol, fixture_sha256::AbstractString,
                           settings_hash::AbstractString)
    stem = comparative_arm_stem(cell, id, arm, fixture_sha256, settings_hash)
    directory = joinpath(out_dir, "comparative_fragments", "runs")
    return (
        summary=joinpath(directory, stem * ".summary.csv"),
        points=joinpath(directory, stem * ".points.arrow"),
        checkpoint=joinpath(directory, stem * ".checkpoint.jls"),
        log=joinpath(out_dir, "comparative_logs", stem * ".log"),
    )
end

function _comparative_status_counts_parse(text::AbstractString)
    counts = Dict{Symbol,Int}()
    isempty(text) && return counts
    for item in split(text, '|')
        isempty(item) && continue
        key, value = split(item, ':'; limit=2)
        counts[Symbol(key)] = parse(Int, value)
    end
    return counts
end

function _comparative_csv_dict(header_line::AbstractString, value_line::AbstractString)
    header = split(header_line, ',')
    values = split(value_line, ','; keepempty=true)
    length(header) == length(values) ||
        error("comparative fragment column mismatch: $(length(header)) headers, $(length(values)) values")
    return Dict(Symbol(h) => v for (h, v) in zip(header, values))
end

function write_comparative_arm_fragments(out_dir::AbstractString, run_id::AbstractString,
        fx, result::ArmCellResult, rows::Vector{ArmQueryRow};
        terminal_state::Symbol=:OK, error_message::AbstractString="",
        code_provenance::AbstractString=comparative_code_provenance(result.arm),
        worker_project::AbstractString=_comparative_active_project(),
        blas_threads::Int=LinearAlgebra.BLAS.get_num_threads(),
        paths=nothing)
    cell = smoke_cell_from_fixture(fx)
    length(rows) == result.n_query == fx.n_query ||
        error("write_comparative_arm_fragments: result/row/fixture query counts differ")
    paths === nothing && (paths = comparative_fragment_paths(
        out_dir, cell, fx.cell_id, result.arm, fx.fixture_sha256, result.settings_hash))

    summary_values = (
        COMPARATIVE_FRAGMENT_VERSION, run_id, result.cell_id, result.arm, terminal_state,
        error_message, fx.schema_version, result.fixture_sha256, result.settings_hash,
        code_provenance, normpath(String(worker_project)), string(VERSION), blas_threads,
        result.name, result.layout, result.tier_label, result.dim,
        result.model_type, result.model_seed, result.noise_seed, result.n_train,
        result.n_query, result.fit_time_s, result.inference_time_total_s,
        result.inference_time_mean_s, result.final_loss, result.n_finite_pred,
        result.n_finite_interval, result.rel_l2, result.mae, result.mse,
        result.coverage, result.mean_width, _status_counts_str(result.status_counts),
    )
    atomic_write(paths.summary) do io
        println(io, join(string.(_COMPARATIVE_SUMMARY_FIELDS), ','))
        println(io, join(_safe_csv_field.(summary_values), ','))
    end

    point_header = String[
        "fragment_version", "run_id", "cell_id", "query_id", "arm",
        "fixture_schema", "fixture_sha256", "settings_hash", "code_provenance",
    ]
    append!(point_header, ["x_$d" for d in 1:result.dim])
    append!(point_header, ["truth", "pred", "lower", "upper", "status", "elapsed_s",
                           "smoothness_used"])
    point_names = Tuple(Symbol.(point_header))
    point_type = NamedTuple{point_names}
    point_records = [begin
        values = Any[COMPARATIVE_FRAGMENT_VERSION, String(run_id), row.cell_id,
                     row.query_id, String(row.arm), fx.schema_version,
                     result.fixture_sha256, result.settings_hash, String(code_provenance)]
        append!(values, row.x)
        append!(values, Any[row.truth, row.pred, row.lower, row.upper,
                            String(row.status), row.elapsed_s, row.smoothness_used])
        point_type(Tuple(values))
    end for row in sort(rows; by=row -> row.query_id)]
    write_point_fragment(paths.points, point_records)
    return paths
end

function _comparative_failure_payload(fx, arm::Symbol, settings_hash::String,
                             reg::ComparatorRegistry.CanonicalRegistry,
                             state::Symbol)
    cell = smoke_cell_from_fixture(fx)
    status = Symbol(uppercase(string(arm)) * "_WORKER_" * uppercase(string(state)))
    rows = ArmQueryRow[
        ArmQueryRow(fx.cell_id, Int(fx.query_ids[i]), arm,
                    Vector{Float64}(fx.x_query[:, i]), fx.y_query_truth[i],
                    NaN, NaN, NaN, status, 0.0)
        for i in eachindex(fx.query_ids)
    ]
    result = arm_cell_result_from_rows(rows;
        cell_id=fx.cell_id, cell=cell, arm=arm, model_type=_comparative_model_type(arm),
        fixture_sha256=fx.fixture_sha256, settings_hash=settings_hash,
        model_seed=_comparative_model_seed(cell, arm, reg), noise_seed=fx.noise_seed,
        n_train=fx.n_train_actual, fit_time_s=NaN, final_loss=NaN)
    return result, rows
end

function write_comparative_failure_fragment(out_dir::AbstractString, run_id::AbstractString,
        fx, arm::Symbol, settings_hash::String,
        reg::ComparatorRegistry.CanonicalRegistry, state::Symbol,
        message::AbstractString; code_provenance::String=comparative_code_provenance(arm),
        worker_project::String=_comparative_project_file(arm), paths=nothing,
        blas_threads::Int=DEFAULT_BLAS_THREADS_COMPARATIVE)
    result, rows = _comparative_failure_payload(fx, arm, settings_hash, reg, state)
    return write_comparative_arm_fragments(out_dir, run_id, fx, result, rows;
        terminal_state=state, error_message=message,
        code_provenance=code_provenance, worker_project=worker_project,
        blas_threads=blas_threads, paths=paths)
end

function _comparative_expected(cell::SmokeCell, id::Int, arm::Symbol, fx, fixture_path::String,
                      reg::ComparatorRegistry.CanonicalRegistry, out_dir::String;
                      device::String="cpu")
    settings_hash = _comparative_settings_hash(cell, arm, reg; device=device)
    code = comparative_code_provenance(arm)
    return (cell=cell, cell_id=id, arm=arm, fx=fx, fixture_path=fixture_path,
            fixture_schema=fx.schema_version, fixture_sha256=fx.fixture_sha256,
            settings_hash=settings_hash, code_provenance=code,
            worker_project=_comparative_project_file(arm), device=device,
            paths=comparative_fragment_paths(out_dir, cell, id, arm, fx.fixture_sha256, settings_hash))
end

"""
    read_comparative_arm_fragment(expected; allow_failure=true, provenance=:current)

Read one arm-cell fragment and check its identity against `expected`.

`provenance=:current` (resume path) also requires the fragment's recorded code
provenance to equal the hash of the harness and arm sources as they are NOW,
so a worker never resumes from a fragment that different code wrote.
`provenance=:recorded` (report path) accepts the recorded hash as the
fragment's own provenance and returns it in the payload: a finished campaign
is reported as it was run, and the report records which code each arm ran.
The fixture hash and settings hash are always checked.
"""
function read_comparative_arm_fragment(expected; allow_failure::Bool=true,
                                       provenance::Symbol=:current)
    provenance in (:current, :recorded) ||
        throw(ArgumentError("provenance must be :current or :recorded, got $provenance"))
    paths = expected.paths
    isfile(paths.summary) || error("missing comparative summary fragment: $(paths.summary)")
    isfile(paths.points) || error("missing comparative points fragment: $(paths.points)")
    slines = readlines(paths.summary)
    length(slines) == 2 || error("comparative summary fragment must contain exactly one data row: $(paths.summary)")
    sd = _comparative_csv_dict(slines[1], slines[2])

    required = string.(_COMPARATIVE_SUMMARY_FIELDS)
    Set(split(slines[1], ',')) == Set(required) ||
        error("comparative summary schema mismatch in $(paths.summary)")
    sd[:fragment_version] == COMPARATIVE_FRAGMENT_VERSION || error("stale fragment version")
    parse(Int, sd[:cell_id]) == expected.cell_id || error("stale cell identity")
    Symbol(sd[:arm]) == expected.arm || error("stale arm identity")
    sd[:fixture_schema] == expected.fixture_schema || error("stale fixture schema")
    sd[:fixture_sha256] == expected.fixture_sha256 || error("stale fixture content hash")
    sd[:settings_hash] == expected.settings_hash || error("stale settings hash")
    provenance == :recorded || sd[:code_provenance] == expected.code_provenance ||
        error("stale code/runtime provenance")
    recorded_provenance = String(sd[:code_provenance])
    normpath(sd[:worker_project]) == normpath(expected.worker_project) ||
        error("wrong worker project: $(sd[:worker_project])")
    terminal_state = Symbol(sd[:terminal_state])
    (!allow_failure && terminal_state != :OK) && error("terminal failure fragments are retried, not resumed")

    dim = parse(Int, sd[:dim])
    point_records = read_point_fragment(paths.points)
    isempty(point_records) && error("empty comparative points fragment")
    rows = ArmQueryRow[]
    for pd in point_records
        _point_string(pd[:fragment_version]) == COMPARATIVE_FRAGMENT_VERSION || error("stale point fragment version")
        _point_int(pd[:cell_id]) == expected.cell_id || error("stale point cell identity")
        Symbol(_point_string(pd[:arm])) == expected.arm || error("stale point arm identity")
        _point_string(pd[:fixture_schema]) == expected.fixture_schema || error("stale point fixture schema")
        _point_string(pd[:fixture_sha256]) == expected.fixture_sha256 || error("stale point fixture hash")
        _point_string(pd[:settings_hash]) == expected.settings_hash || error("stale point settings hash")
        _point_string(pd[:code_provenance]) == recorded_provenance || error("stale point code provenance")
        qid = _point_int(pd[:query_id])
        x = [_parse_float(pd[Symbol("x_$d")]) for d in 1:dim]
        push!(rows, ArmQueryRow(expected.cell_id, qid, expected.arm, x,
            _parse_float(pd[:truth]), _parse_float(pd[:pred]),
            _parse_float(pd[:lower]), _parse_float(pd[:upper]),
            Symbol(_point_string(pd[:status])), _parse_float(pd[:elapsed_s]),
            _parse_float(pd[:smoothness_used])))
    end
    sort!(rows; by=row -> row.query_id)
    fx = expected.fx
    length(rows) == fx.n_query || error("comparative point count differs from fixture")
    for (i, row) in enumerate(rows)
        row.query_id == fx.query_ids[i] || error("comparative query ID/order mismatch")
        isequal(row.x, Vector{Float64}(fx.x_query[:, i])) || error("comparative query coordinate mismatch")
        isequal(row.truth, fx.y_query_truth[i]) || error("comparative query truth mismatch")
    end

    result = ArmCellResult(
        parse(Int, sd[:cell_id]), Symbol(sd[:name]),
        Symbol(sd[:layout]), sd[:tier_label], dim, Symbol(sd[:arm]), sd[:model_type],
        sd[:fixture_sha256], sd[:settings_hash], parse(Int, sd[:model_seed]),
        parse(Int, sd[:noise_seed]), parse(Int, sd[:n_train]), parse(Int, sd[:n_query]),
        _parse_float(sd[:fit_time_s]), _parse_float(sd[:inference_time_total_s]),
        _parse_float(sd[:inference_time_mean_s]), _parse_float(sd[:final_loss]),
        parse(Int, sd[:n_finite_pred]), parse(Int, sd[:n_finite_interval]),
        _parse_float(sd[:rel_l2]), _parse_float(sd[:mae]), _parse_float(sd[:mse]),
        _parse_float(sd[:coverage]), _parse_float(sd[:mean_width]),
        _comparative_status_counts_parse(sd[:status_counts]),
    )
    result.cell_id == fx.cell_id && result.arm == expected.arm ||
        error("comparative reconstructed result identity mismatch")
    result.n_query == length(rows) || error("comparative summary/point count mismatch")
    result.n_finite_pred == count(row -> isfinite(row.pred), rows) ||
        error("comparative finite prediction count mismatch")
    return (result=result, rows=rows, terminal_state=terminal_state,
            error=sd[:error], worker_project=sd[:worker_project],
            run_id=sd[:run_id], blas_threads=parse(Int, sd[:blas_threads]),
            code_provenance=recorded_provenance)
end

function _comparative_try_read(expected; allow_failure::Bool=true, provenance::Symbol=:current)
    try
        return (valid=true, error="",
                payload=read_comparative_arm_fragment(expected; allow_failure=allow_failure,
                                                      provenance=provenance))
    catch err
        return (valid=false, payload=nothing, error=sprint(showerror, err))
    end
end

function _comparative_fixture_matches(prep::PreparedSmokeCell, fx)
    cell = prep.cell
    scalars_ok = fx.schema_version == prep.schema_version && fx.cell_id == prep.cell_id &&
        Symbol(fx.name) == cell.name && fx.dim == cell.dim &&
        Symbol(fx.layout) == cell.layout && fx.tier_label == cell.tier_label &&
        fx.n_train_nominal == cell.n_train && fx.n_train_actual == prep.n_train_actual &&
        fx.n_query == prep.n_query && isequal(fx.noise_level, prep.noise_level) &&
        isequal(fx.noise_bound, prep.noise_bound) && fx.noise_seed == prep.noise_seed &&
        isequal(fx.output_range, prep.output_range)
    scalars_ok || return false
    return isequal(fx.x_train, prep.X_train) &&
           isequal(fx.y_train_clean, prep.y_train_clean) &&
           isequal(fx.y_train_noisy, prep.y_train_noisy) &&
           isequal(fx.x_query, prep.X_query) &&
           isequal(fx.y_query_truth, prep.y_query_truth) &&
           isequal(Vector{Int}(fx.query_ids), prep.query_ids)
end

function ensure_comparative_fixture(cell::SmokeCell, id::Int, out_dir::String)
    prep = prepare_smoke_cell(cell; cell_id=id)
    fpath, _ = fixture_paths(out_dir, cell, id)
    if isfile(fpath)
        try
            fx = read_fixture(fpath)
            _comparative_fixture_matches(prep, fx) && return (fx=fx, path=fpath, reused=true)
        catch err
            @info "comparative fixture is stale/corrupt and will be regenerated" cell_id=id exception=err
        end
    end
    materialize_fixture(prep, out_dir)
    return (fx=read_fixture(fpath), path=fpath, reused=false)
end

function run_comparative_arm_worker()
    out_dir = abspath(_smoke_required("out"))
    fixture_path = abspath(_smoke_required("fixture"))
    run_id = _smoke_opt("run-id", basename(out_dir))
    id = parse(Int, _smoke_required("cell"))
    arm = Symbol(lowercase(_smoke_required("arm")))
    settings_hash = _smoke_required("settings-hash")
    expected_code = _smoke_opt("code-provenance", comparative_code_provenance(arm))
    blas_threads = parse(Int, _smoke_opt("blas-threads", string(DEFAULT_BLAS_THREADS_COMPARATIVE)))
    blas_threads >= 1 || error("--blas-threads must be >= 1")
    LinearAlgebra.BLAS.set_num_threads(blas_threads)
    device = ComparatorRegistry.normalize_device(_smoke_opt("device", "cpu"))
    # The safety multipliers must be bound BEFORE the settings hash below, or
    # the worker recomputes the default and the --settings-hash check rejects
    # it. The orchestrator passes whatever the config resolved; the defaults
    # here keep a hand-run worker on the shipped values.
    set_lh_factor!(parse(Float64, _smoke_opt("lh-factor", string(_SMOKE_L_H_FACTOR_DEFAULT))))
    _NSM_AVAILABLE && NSMTrainer.set_gamma_factor!(
        parse(Float64, _smoke_opt("gamma-factor", string(NSMTrainer.NSM_GAMMA_FACTOR_DEFAULT))))

    cells = enumerate_cells()
    (1 <= id <= length(cells)) || error("--cell=$id out of range [1,$(length(cells))]")
    cell = cells[id]
    arm in applicable_arms(cell) || error("arm=$arm is not applicable to cell=$id")
    reg = ComparatorRegistry.canonical_registry()
    current_settings = _comparative_settings_hash(cell, arm, reg; device=device)
    current_settings == settings_hash ||
        error("--settings-hash mismatch: supplied=$settings_hash current=$current_settings")
    current_code = comparative_code_provenance(arm)
    current_code == expected_code ||
        error("--code-provenance mismatch: supplied=$expected_code current=$current_code")
    active_project = _comparative_active_project()
    normpath(active_project) == _comparative_project_file(arm) ||
        error("wrong Julia project for arm=$arm: active=$active_project expected=$(_comparative_project_file(arm))")

    local fx = nothing
    try
        fx = read_fixture(fixture_path)
        fx.cell_id == id || error("fixture cell_id=$(fx.cell_id) does not match --cell=$id")
        smoke_cell_from_fixture(fx) == cell || error("fixture descriptor does not match --cell=$id")
        println("comparative worker cell=$id arm=$arm project=$active_project " *
                "fixture=$(fx.fixture_sha256) settings=$settings_hash blas_threads=$blas_threads")
        flush(stdout)
        _smoke_flag("inject-failure") && error("deliberate orchestration-resilience-gate worker failure injection for ($id,$arm)")

        expected = _comparative_expected(cell, id, arm, fx, fixture_path, reg, out_dir; device=device)
        fragment_is_current = _comparative_try_read(expected; allow_failure=false).valid
        checkpoint_path = fragment_is_current ? nothing : expected.paths.checkpoint
        result, rows = run_arm_cell(cell, arm, out_dir; cell_id=id, reg=reg, fx=fx,
                                    verbose=false, checkpoint_path=checkpoint_path, device=device)
        result.fixture_sha256 == fx.fixture_sha256 || error("arm returned wrong fixture hash")
        result.settings_hash == settings_hash || error("arm returned wrong settings hash")
        paths = write_comparative_arm_fragments(out_dir, run_id, fx, result, rows;
            terminal_state=:OK, code_provenance=current_code,
            worker_project=active_project, blas_threads=blas_threads)
        println("DONE comparative cell=$id arm=$arm summary=$(paths.summary)")
        return nothing
    catch err
        message = sprint(showerror, err)
        if fx !== nothing
            write_comparative_failure_fragment(out_dir, run_id, fx, arm, settings_hash, reg,
                :ERROR, message; code_provenance=current_code,
                worker_project=active_project, blas_threads=blas_threads)
        end
        println(stderr, "FAILED comparative cell=$id arm=$arm: $message")
        Base.show_backtrace(stderr, catch_backtrace())
        rethrow()
    end
end

function comparative_worker_command(expected, out_dir::String, run_id::String;
                           blas_threads::Int=DEFAULT_BLAS_THREADS_COMPARATIVE,
                           inject_failure::Bool=false)
    julia = Base.julia_cmd()
    args = String[
        "--worker", "--cell=$(expected.cell_id)", "--arm=$(expected.arm)",
        "--fixture=$(expected.fixture_path)", "--out=$out_dir", "--run-id=$run_id",
        "--settings-hash=$(expected.settings_hash)",
        "--code-provenance=$(expected.code_provenance)",
        "--blas-threads=$blas_threads",
        "--device=$(expected.device)",
        "--lh-factor=$(_SMOKE_L_H_FACTOR[])",
        "--gamma-factor=$(_NSM_AVAILABLE ? NSMTrainer.NSM_GAMMA_FACTOR[] : _SMOKE_L_H_FACTOR[])",
    ]
    inject_failure && push!(args, "--inject-failure=true")
    cmd = `$julia --startup-file=no --threads=1 --project=$(_comparative_project_dir(expected.arm)) $(SCRIPT_PATH) $args`
    return addenv(cmd,
        "OPENBLAS_NUM_THREADS" => string(blas_threads),
        "OMP_NUM_THREADS" => string(blas_threads),
        "MKL_NUM_THREADS" => string(blas_threads),
        "BLIS_NUM_THREADS" => string(blas_threads))
end

function _comparative_failure_requested(spec::AbstractString, id::Int, arm::Symbol)
    isempty(strip(spec)) && return false
    target = "$(id):$(arm)"
    return target in strip.(split(lowercase(spec), ','))
end

function run_comparative_worker_stage(expectations::Vector, out_dir::String, run_id::String,
                             jobs::Int, timeout_s::Real;
                             blas_threads::Int=DEFAULT_BLAS_THREADS_COMPARATIVE,
                             inject_failure::AbstractString="")
    jobs >= 1 || error("comparative jobs must be >= 1")
    isempty(expectations) && return (outcomes=Dict{Tuple{Int,Symbol},Symbol}(), max_active=0)
    arm = first(expectations).arm
    all(exp.arm == arm for exp in expectations) || error("comparative stages must contain one arm only")
    println("\n=== comparative stage arm=$arm pairs=$(length(expectations)) jobs=$jobs timeout=$(timeout_s)s ===")
    tokens = Channel{Nothing}(jobs)
    for _ in 1:jobs
        put!(tokens, nothing)
    end
    outcomes = Dict{Tuple{Int,Symbol},Symbol}()
    lock_state = ReentrantLock()
    active = Ref(0)
    max_active = Ref(0)
    reg = ComparatorRegistry.canonical_registry()

    @sync for expected in expectations
        @async begin
            take!(tokens)
            key = (expected.cell_id, expected.arm)
            lock(lock_state) do
                active[] += 1
                max_active[] = max(max_active[], active[])
            end
            try
                inject = _comparative_failure_requested(inject_failure, expected.cell_id, expected.arm)
                cmd = comparative_worker_command(expected, out_dir, run_id;
                                        blas_threads=blas_threads, inject_failure=inject)
                mkpath(dirname(expected.paths.log))
                log_tmp = expected.paths.log * ".tmp-$(getpid())-$(expected.cell_id)-$(expected.arm)"
                outcome = open(log_tmp, "w") do io
                    proc = run(pipeline(ignorestatus(cmd); stdout=io, stderr=io); wait=false)
                    started = time()
                    while process_running(proc)
                        if time() - started > timeout_s
                            kill(proc)
                            sleep(0.25)
                            process_running(proc) && kill(proc, Base.SIGKILL)
                            return :TIMEOUT
                        end
                        sleep(0.25)
                    end
                    return proc.exitcode == 0 ? :OK : :FAILED
                end
                mv(log_tmp, expected.paths.log; force=true)

                fragment = _comparative_try_read(expected; allow_failure=true)
                if outcome == :TIMEOUT
                    write_comparative_failure_fragment(out_dir, run_id, expected.fx, expected.arm,
                        expected.settings_hash, reg, :TIMEOUT,
                        "worker exceeded $(timeout_s)s timeout and was killed; see $(expected.paths.log)";
                        code_provenance=expected.code_provenance,
                        worker_project=expected.worker_project, paths=expected.paths,
                        blas_threads=blas_threads)
                elseif !fragment.valid
                    write_comparative_failure_fragment(out_dir, run_id, expected.fx, expected.arm,
                        expected.settings_hash, reg, :ERROR,
                        "worker outcome=$outcome without a valid terminal fragment: $(fragment.error); see $(expected.paths.log)";
                        code_provenance=expected.code_provenance,
                        worker_project=expected.worker_project, paths=expected.paths,
                        blas_threads=blas_threads)
                    outcome = :FAILED
                end
                lock(lock_state) do
                    outcomes[key] = outcome
                end
                println("finish cell=$(expected.cell_id) arm=$(expected.arm): $outcome")
                flush(stdout)
            finally
                lock(lock_state) do
                    active[] -= 1
                end
                put!(tokens, nothing)
            end
        end
    end
    return (outcomes=outcomes, max_active=max_active[])
end

function _comparative_representative_ids(cells::Vector{SmokeCell})
    p1 = findfirst(c -> c.name == :ishigami &&
                        c.layout == :sobol && c.n_train == _SINGLE_CELL_ANCHOR_NTRAIN &&
                        c.noise_level == 0.0, cells)
    p2 = findfirst(c -> c.name == :ishigami &&
                        c.layout == :sobol && c.n_train == _SINGLE_CELL_ANCHOR_NTRAIN &&
                        c.noise_level == _SINGLE_CELL_ANCHOR_NOISE, cells)
    (p1 === nothing || p2 === nothing) && error("comparative representative cells not found")
    return Int[p1, p2]
end

function _comparative_parse_cell_ids(text::AbstractString, n_cells::Int)
    ids = sort(unique(parse.(Int, strip.(split(text, ',')))))
    all(id -> 1 <= id <= n_cells, ids) ||
        error("--cell-ids contains an id outside 1:$n_cells")
    return ids
end

function _comparative_selected_ids(mode::Symbol, cells::Vector{SmokeCell};
                          selected_ids_override=nothing)
    selected_ids_override !== nothing && return sort(unique(Int.(selected_ids_override)))
    ids = mode == :full ? collect(eachindex(cells)) :
          mode == :representative ? _comparative_representative_ids(cells) :
          error("comparative mode must be :representative or :full")
    ids_opt = _smoke_opt("cell-ids", nothing)
    ids_opt !== nothing && (ids = _comparative_parse_cell_ids(ids_opt, length(cells)))
    only_filter = _smoke_opt("only", nothing)
    only_filter !== nothing &&
        (ids = [id for id in ids if cell_matches_filter(cells[id], only_filter)])
    isempty(ids) && error("comparative comparative campaign selected no cells")
    return ids
end

"""
    _comparative_read_existing_fixture(cell, id, out_dir) -> (fx, path, reused=true)

Read a fixture that is ALREADY on disk, without calling `prepare_smoke_cell`.
That call is not free -- it evaluates the benchmark function at every training
point and runs `estimate_noise_bound` -- and it REWRITES the fixture, which a
read-only pass must not do while a shard's workers are keyed to it.

`read_fixture` verifies the file's own SHA-256 trailer, and the cell descriptor
is re-checked here, so a corrupt or misplaced fixture is an error rather than a
silently different dataset. A fixture that is intact but does not match what
`prepare_smoke_cell` would now produce still cannot pass unnoticed: the
fragments carry `fixture_sha256`, so they simply fail to resolve and are
reported as missing.
"""
function _comparative_read_existing_fixture(cell::SmokeCell, id::Int, out_dir::String)
    fpath, _ = fixture_paths(out_dir, cell, id)
    isfile(fpath) || error(
        "no fixture on disk for cell $(id) ($(cell.name)/$(cell.layout)/$(cell.tier_label)) " *
        "under $(out_dir): $(fpath). This pass does not generate fixtures. Run " *
        "`benchmark_data.jl --prepare-fixtures <config>` (or any shard covering this cell) " *
        "first.")
    fx = read_fixture(fpath)
    fx.cell_id == id || error("fixture at $(fpath) carries cell_id=$(fx.cell_id), expected $(id)")
    smoke_cell_from_fixture(fx) == cell ||
        error("fixture at $(fpath) describes a different cell than enumerate_cells()[$(id)]")
    return (fx=fx, path=fpath, reused=true)
end

function _comparative_prepare_expectations(ids::Vector{Int}, cells::Vector{SmokeCell},
                                  out_dir::String,
                                  reg::ComparatorRegistry.CanonicalRegistry;
                                  device::String="cpu",
                                  arms_override::Union{Nothing,AbstractVector{Symbol}}=nothing,
                                  ensure_fixtures::Bool=true)
    fixtures = Dict{Int,Any}()
    fixture_reused = Dict{Int,Bool}()
    for id in ids
        ensured = ensure_fixtures ?
            ensure_comparative_fixture(cells[id], id, out_dir) :
            _comparative_read_existing_fixture(cells[id], id, out_dir)
        fixtures[id] = ensured
        fixture_reused[id] = ensured.reused
    end
    arms_allowed = arms_override === nothing ? nothing : Set(Symbol.(arms_override))
    expectations = Any[]
    for id in ids, arm in applicable_arms(cells[id])
        arms_allowed === nothing || arm in arms_allowed || continue
        ensured = fixtures[id]
        arm_device = arm in (:nn,) ? device : "cpu"   # :nsm/:rbf/:idw are CPU-only
        push!(expectations, _comparative_expected(cells[id], id, arm, ensured.fx,
                                         ensured.path, reg, out_dir; device=arm_device))
    end
    return expectations, fixture_reused
end

"""
    _comparative_artifact_prefix(arms_allowed) -> String

Filename prefix for a run's own campaign-level artifacts. A whole-campaign run
owns the canonical `comparative_*` names; an ARM SHARD gets its own
`comparative_shard-<arms>_*` names instead, so that several shards writing into
one shared `run.out` cannot overwrite each other's summary and outcomes files
(the per-(cell, arm) fragments were always disjoint; these two were not).
The authoritative six-arm aggregate is then rebuilt by the `--report` pass,
which writes the canonical names.
"""
_comparative_artifact_prefix(::Nothing) = "comparative"
_comparative_artifact_prefix(arms_allowed) =
    "comparative_shard-" * join(sort(string.(collect(arms_allowed))), "-")

function write_comparative_campaign_outcomes(out_dir::String, expectations, payloads,
                                    resumed::Set{Tuple{Int,Symbol}}, stale,
                                    process_outcomes; prefix::AbstractString="comparative")
    path = joinpath(out_dir, "$(prefix)_outcomes.csv")
    stale_set = Set(stale)
    atomic_write(path) do io
        println(io, "cell_id,arm,terminal_state,resumed,stale_detected,process_outcome,fixture_schema,fixture_sha256,settings_hash,code_provenance,worker_project,error")
        for expected in expectations
            key = (expected.cell_id, expected.arm)
            payload = payloads[key]
            vals = (expected.cell_id, expected.arm, payload.terminal_state,
                    key in resumed, key in stale_set, get(process_outcomes, key, :RESUMED),
                    expected.fixture_schema, expected.fixture_sha256,
                    expected.settings_hash, expected.code_provenance,
                    payload.worker_project, payload.error)
            println(io, join(_safe_csv_field.(vals), ','))
        end
    end
    return path
end

function run_comparative_orchestrator(mode::Symbol=:representative;
        selected_ids_override=nothing, out_dir_override=nothing,
        inject_failure_override=nothing, jobs_override=nothing,
        timeout_override=nothing, device_override=nothing, arms_override=nothing)
    cells = enumerate_cells()
    ids = _comparative_selected_ids(mode, cells; selected_ids_override=selected_ids_override)
    out_dir = abspath(out_dir_override === nothing ?
        _smoke_opt("out", joinpath(REPO_ROOT, "results", "comparative_campaign",
            Dates.format(Dates.now(), "yyyy-mm-dd-HHMMSS") * "-$(mode)")) : out_dir_override)
    run_id = _smoke_opt("run-id", basename(out_dir))
    resume = lowercase(string(_smoke_opt("resume", "true"))) in ("1", "true", "yes")
    blas_threads = parse(Int, _smoke_opt("blas-threads", string(DEFAULT_BLAS_THREADS_COMPARATIVE)))
    inject_failure = inject_failure_override === nothing ?
        string(_smoke_opt("inject-worker-failure", "")) : String(inject_failure_override)
    # cpu or cuda, and it reaches the :nn arm only (_comparative_prepare_expectations
    # pins every other arm to "cpu"). It is part of the :nn settings hash, so a
    # device switch is a different fragment, never a silent resume.
    device = ComparatorRegistry.normalize_device(
        String(device_override === nothing ? _smoke_opt("device", "cpu") : device_override))
    reg = ComparatorRegistry.canonical_registry()
    mkpath(out_dir)

    jobs = Dict{Symbol,Int}()
    timeouts = Dict{Symbol,Float64}()
    for arm in _COMPARATIVE_ARM_ORDER
        jobs[arm] = jobs_override === nothing ?
            parse(Int, _smoke_opt("jobs-$(arm)", string(DEFAULT_JOBS_COMPARATIVE[arm]))) :
            get(jobs_override, arm, DEFAULT_JOBS_COMPARATIVE[arm])
        timeouts[arm] = timeout_override === nothing ?
            parse(Float64, _smoke_opt("timeout-$(arm)", string(DEFAULT_TIMEOUT_COMPARATIVE[arm]))) :
            get(timeout_override, arm, DEFAULT_TIMEOUT_COMPARATIVE[arm])
        jobs[arm] >= 1 || error("--jobs-$arm must be >= 1")
        timeouts[arm] > 0 || error("--timeout-$arm must be > 0")
    end

    arms_allowed = arms_override === nothing ? nothing : Set(Symbol.(arms_override))
    n_arm_cells = sum(count(arm -> arms_allowed === nothing || arm in arms_allowed,
                             applicable_arms(cells[id])) for id in ids)
    println("comparative comparative orchestrator mode=$mode cells=$(length(ids)) arm-cells=$n_arm_cells")
    println("out_dir = $out_dir")
    arms_override === nothing || println("arms restricted to: " * join(sort(collect(arms_allowed)), ", "))
    # Only the arms this run will actually schedule. Printing the orchestrator's
    # fallbacks for arms an arm shard never touches makes a shard log read as
    # though it were about to run them.
    scheduled_arms = [arm for arm in _COMPARATIVE_ARM_ORDER
                      if arms_allowed === nothing || arm in arms_allowed]
    println("jobs: " * join(("$arm=$(jobs[arm])" for arm in scheduled_arms), "  "))
    println("timeouts(s): " * join(("$arm=$(timeouts[arm])" for arm in scheduled_arms), "  "))
    println("worker BLAS threads = $blas_threads   resume = $resume   device = $device")

    expectations, fixture_reused = _comparative_prepare_expectations(ids, cells, out_dir, reg;
        device=device, arms_override=arms_override)
    resumed = Set{Tuple{Int,Symbol}}()
    stale = Tuple{Int,Symbol}[]
    pending = Any[]
    for expected in expectations
        key = (expected.cell_id, expected.arm)
        prior_exists = isfile(expected.paths.summary) || isfile(expected.paths.points)
        check = resume ? _comparative_try_read(expected; allow_failure=false) :
                         (valid=false, payload=nothing, error="resume disabled")
        if check.valid
            push!(resumed, key)
            println("resume cell=$(expected.cell_id) arm=$(expected.arm)")
        else
            prior_exists && push!(stale, key)
            prior_exists && println("stale cell=$(expected.cell_id) arm=$(expected.arm): $(check.error)")
            push!(pending, expected)
        end
    end

    process_outcomes = Dict{Tuple{Int,Symbol},Symbol}()
    max_active = Dict{Symbol,Int}(arm => 0 for arm in _COMPARATIVE_ARM_ORDER)
    for arm in _COMPARATIVE_ARM_ORDER
        stage = [expected for expected in pending if expected.arm == arm]
        isempty(stage) && continue
        stage_result = run_comparative_worker_stage(stage, out_dir, run_id, jobs[arm], timeouts[arm];
            blas_threads=blas_threads, inject_failure=inject_failure)
        merge!(process_outcomes, stage_result.outcomes)
        max_active[arm] = stage_result.max_active
    end

    payloads = Dict{Tuple{Int,Symbol},Any}()
    for expected in expectations
        key = (expected.cell_id, expected.arm)
        check = _comparative_try_read(expected; allow_failure=true)
        if !check.valid
            write_comparative_failure_fragment(out_dir, run_id, expected.fx, expected.arm,
                expected.settings_hash, reg, :ERROR,
                "orchestrator could not read a terminal fragment: $(check.error)";
                code_provenance=expected.code_provenance,
                worker_project=expected.worker_project, paths=expected.paths,
                blas_threads=blas_threads)
            check = _comparative_try_read(expected; allow_failure=true)
        end
        check.valid || error("comparative failed to create terminal accounting for key=$key: $(check.error)")
        payloads[key] = check.payload
    end

    results = ArmCellResult[payloads[(expected.cell_id, expected.arm)].result
                            for expected in expectations]
    row_groups = Vector{ArmQueryRow}[payloads[(expected.cell_id, expected.arm)].rows
                                     for expected in expectations]
    # Report against the arm set this run actually scheduled, per cell. With
    # arms_override that is a SHARD: n_common is intersected over the shard's
    # arms only and is not comparable with a full run's, which is why each row
    # carries its own `common_arms` column and why the shard's campaign-level
    # files are written under a shard-specific prefix rather than over the
    # canonical `comparative_*` names. The fragments themselves are keyed on
    # (cell, arm, fixture_sha, settings_hash) and written through atomic_write,
    # so disjoint shards into one out_dir stay safe; the --report pass
    # re-aggregates them into the canonical campaign-wide summary.
    expected_arms_by_cell = Dict{Int,Set{Symbol}}()
    for expected in expectations
        push!(get!(expected_arms_by_cell, expected.cell_id, Set{Symbol}()), expected.arm)
    end
    artifact_prefix = _comparative_artifact_prefix(arms_allowed)
    arms_override === nothing ||
        println("NOTE: this is an ARM SHARD. Its campaign artifacts are written under " *
                "$(artifact_prefix)_* so that concurrent shards in one out_dir cannot " *
                "overwrite each other; n_common in them is an intersection over " *
                "$(sort(collect(arms_allowed))) only (each row records that set in its " *
                "common_arms column). Run the --report pass over the full arm set to get " *
                "the canonical comparative_summary.csv.")
    artifacts = write_comparative_artifacts(out_dir, results, row_groups;
                                            prefix=artifact_prefix,
                                            expected_arms=expected_arms_by_cell)
    outcomes_path = write_comparative_campaign_outcomes(out_dir, expectations, payloads,
                                                resumed, stale, process_outcomes;
                                                prefix=artifact_prefix)
    n_failed = count(payload -> payload.terminal_state != :OK, values(payloads))
    n_ok = length(payloads) - n_failed
    success = n_failed == 0
    println("\nComparative campaign summary: terminal=$(length(payloads)) OK=$n_ok failed=$n_failed resumed=$(length(resumed)) stale_rerun=$(length(stale))")
    println("comparative summary: $(artifacts.summary_path)")
    println("comparative points:  $(artifacts.points_dir)/*.points.arrow")
    println("campaign outcomes:   $outcomes_path")
    println(success ? "comparative comparative campaign: SUCCESS" : "comparative comparative campaign: NON-SUCCESS")
    return (success=success, out_dir=out_dir, ids=ids, expectations=expectations,
            payloads=payloads, results=results, row_groups=row_groups,
            artifacts=artifacts, outcomes_path=outcomes_path,
            resumed=resumed, stale=stale, process_outcomes=process_outcomes,
            max_active=max_active, jobs=jobs, timeouts=timeouts,
            fixture_reused=fixture_reused)
end


const _FAST_SMOKE_TIER = "1000D"
const _FAST_SMOKE_NOISE_LEVEL = 0.1   # the 20 dB condition; see NOISE_LEVELS


_fast_smoke_sha256_present(value::AbstractString) = occursin(r"^[0-9a-f]{64}$", value)

function _fast_smoke_summary_row_checks(row)
    core = row.cell_id > 0 && row.dim > 0 && row.n_train > 0 && row.n_query > 0 &&
        !isempty(string(row.name)) &&
        !isempty(string(row.layout)) && !isempty(row.tier_label) &&
        !isempty(string(row.arm)) && !isempty(row.model_type) &&
        _fast_smoke_sha256_present(row.fixture_sha256) &&
        _fast_smoke_sha256_present(row.settings_hash) &&
        row.model_seed >= 0 && row.noise_seed >= 0 &&
        isfinite(row.fit_time_s) && row.fit_time_s >= 0.0 &&
        isfinite(row.inference_time_total_s) && row.inference_time_total_s >= 0.0 &&
        isfinite(row.inference_time_mean_s) && row.inference_time_mean_s >= 0.0 &&
        0 <= row.n_finite_pred <= row.n_query &&
        0 <= row.n_finite_interval <= row.n_query &&
        !isempty(row.status_counts) &&
        row.n_common == row.n_requested == row.n_query &&
        isfinite(row.rel_l2) && isfinite(row.mae) && isfinite(row.mse) &&
        isfinite(row.common_rel_l2) && isfinite(row.common_mae) &&
        isfinite(row.common_mse) &&
        !isempty(row.report_runtime_julia_version) &&
        !isempty(row.report_runtime_machine) &&
        !isempty(row.report_runtime_kernel) && row.report_runtime_threads >= 1

    loss_semantics = if row.arm == :diona
        isnan(row.final_loss)
    else
        isfinite(row.final_loss)
    end

    interval_semantics = if row.interval_kind == :none
        row.n_finite_interval == 0 && row.common_n_finite_interval == 0 &&
            isnan(row.coverage) && isnan(row.mean_width) &&
            isnan(row.common_coverage) && isnan(row.common_mean_width)
    else
        0 <= row.common_n_finite_interval <= row.n_common &&
            (row.n_finite_interval == 0 ||
                (isfinite(row.coverage) && isfinite(row.mean_width))) &&
            (row.common_n_finite_interval == 0 ||
                (isfinite(row.common_coverage) && isfinite(row.common_mean_width)))
    end

    return (core=core, loss_semantics=loss_semantics,
            interval_semantics=interval_semantics)
end

_fast_smoke_summary_row_populated(row) = all(values(_fast_smoke_summary_row_checks(row)))

function _fast_smoke_read_csv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("Fast comparative smoke gate comparison artifact is empty: $path")
    header = split(first(lines), ','; keepempty=true)
    return [Dict(Symbol(name) => value for (name, value) in
                 zip(header, split(line, ','; keepempty=true)))
            for line in lines[2:end] if !isempty(line)]
end

function _fast_smoke_read_points(out_dir::AbstractString, summaries)
    rows = read_point_fragments(joinpath(abspath(out_dir), "comparative_fragments", "runs"))
    selected = Set((parse(Int, summary[:cell_id]), Symbol(summary[:arm]),
                    summary[:fixture_sha256], summary[:settings_hash])
                   for summary in summaries)
    return filter(rows) do row
        (_point_int(row[:cell_id]), Symbol(_point_string(row[:arm])),
         _point_string(row[:fixture_sha256]), _point_string(row[:settings_hash])) in selected
    end
end



struct ProblemAggregateResult
    name::Symbol
    arm::Symbol
    model_type::String
    interval_kind::Symbol
    n_cells::Int
    n_requested::Int
    n_finite_pred::Int
    n_failed_pred::Int
    n_cells_with_pred_failures::Int
    fit_time_total_s::Float64
    inference_time_total_s::Float64
    n_finite_interval::Int
    rel_l2::Float64
    mae::Float64
    mse::Float64
    coverage::Float64
    mean_width::Float64
    n_common::Int
    common_n_finite_interval::Int
    common_rel_l2::Float64
    common_mae::Float64
    common_mse::Float64
    common_coverage::Float64
    common_mean_width::Float64
    status_counts::Dict{Symbol,Int}
end

function _comparative_audit_merge_status_counts(results)
    output = Dict{Symbol,Int}()
    for result in results, (status, count) in result.status_counts
        output[status] = get(output, status, 0) + count
    end
    return output
end

function problem_aggregate_results(
    results::AbstractVector{ArmCellResult},
    row_groups::AbstractVector{<:AbstractVector{ArmQueryRow}};
    expected_arms=nothing,
)::Vector{ProblemAggregateResult}
    result_by_key, rows_by_key, _ =
        _validated_comparative_index(results, row_groups; expected_arms=expected_arms)
    common_by_cell = common_success_query_ids(results, row_groups; expected_arms=expected_arms)
    problem_keys = sort!(unique((result.name, result.arm) for result in results);
        by=key -> (string(key[1]), _comparative_arm_rank(key[2])))

    output = ProblemAggregateResult[]
    for (name, arm) in problem_keys
        problem_results = sort!(
            [result for result in values(result_by_key)
             if result.name == name && result.arm == arm];
            by=result -> result.cell_id)
        problem_rows = ArmQueryRow[]
        common_rows = ArmQueryRow[]
        for result in problem_results
            rows = rows_by_key[(result.cell_id, arm)]
            append!(problem_rows, rows)
            common_ids = Set(common_by_cell[result.cell_id])
            append!(common_rows, (row for row in rows if row.query_id in common_ids))
        end

        mean_width, coverage, n_interval = _arm_interval_stats(problem_rows)
        common_mean_width, common_coverage, common_n_interval =
            _arm_interval_stats(common_rows)
        n_requested = length(problem_rows)
        n_finite = count(row -> isfinite(row.pred), problem_rows)
        push!(output, ProblemAggregateResult(
            name, arm, first(problem_results).model_type,
            comparative_interval_kind(arm), length(problem_results), n_requested,
            n_finite, n_requested - n_finite,
            count(result -> result.n_finite_pred < result.n_query, problem_results),
            sum(result.fit_time_s for result in problem_results),
            sum(result.inference_time_total_s for result in problem_results),
            n_interval, _arm_rel_l2(problem_rows), _arm_mae(problem_rows),
            _arm_mse(problem_rows), coverage, mean_width,
            length(common_rows), common_n_interval, _arm_rel_l2(common_rows),
            _arm_mae(common_rows), _arm_mse(common_rows), common_coverage,
            common_mean_width, _comparative_audit_merge_status_counts(problem_results),
        ))
    end
    return output
end

function _comparative_audit_problem_summary_rows(aggregates::Vector{ProblemAggregateResult})
    return [(
        name=row.name, arm=row.arm, model_type=row.model_type,
        interval_kind=row.interval_kind, n_cells=row.n_cells,
        n_requested=row.n_requested, n_finite_pred=row.n_finite_pred,
        n_failed_pred=row.n_failed_pred,
        n_cells_with_pred_failures=row.n_cells_with_pred_failures,
        fit_time_total_s=row.fit_time_total_s,
        inference_time_total_s=row.inference_time_total_s,
        n_finite_interval=row.n_finite_interval,
        rel_l2=row.rel_l2, mae=row.mae, mse=row.mse,
        coverage=row.coverage, mean_width=row.mean_width,
        n_common=row.n_common, common_n_finite_interval=row.common_n_finite_interval,
        common_rel_l2=row.common_rel_l2, common_mae=row.common_mae,
        common_mse=row.common_mse, common_coverage=row.common_coverage,
        common_mean_width=row.common_mean_width,
        status_counts=_status_counts_str(row.status_counts),
    ) for row in aggregates]
end

_comparative_audit_metric(value::Float64; digits::Int=5) =
    isfinite(value) ? string(round(value; digits=digits)) : "N/A"

"""
    _short_arm_note(rows, cells_by_key) -> Union{Nothing,String}

Explain an arm covering fewer cells than its neighbours in the same problem
section, or say plainly that the cause is not determinable here.

The :rbf memory pre-flight is named ONLY when the cells actually missing from
:rbf's row are EXACTLY the cells of this problem that `applicable_arms`
excludes it from. `cells_by_key` carries the real per-(problem, arm) cell sets
for that comparison; counting alone is not enough, because a narrowed or
incomplete report can produce the same deficit for an unrelated reason and a
confident wrong cause in a published table is worse than none.
"""
function _short_arm_note(rows, cells_by_key)
    cell_counts = unique(row.n_cells for row in rows)
    length(cell_counts) > 1 || return nothing
    name = first(rows).name
    max_cells = maximum(row.n_cells for row in rows)
    short = [row.arm for row in rows if row.n_cells < max_cells]

    covered = Dict(row.arm => get(cells_by_key, (name, row.arm), Set{Int}()) for row in rows)
    full = reduce(union!, (copy(s) for s in values(covered)); init=Set{Int}())
    excluded_rbf = Set(id for (id, c) in enumerate(enumerate_cells())
                       if c.name == name && !(:rbf in applicable_arms(c)))
    explained = short == [:rbf] && !isempty(excluded_rbf) &&
                setdiff(full, covered[:rbf]) == excluded_rbf

    cause = explained ?
        "`:rbf` is excluded on cell(s) $(join(sort(collect(excluded_rbf)), ", ")) of this " *
        "problem, where its dense n x n system exceeds its memory pre-flight (see " *
        "applicable_arms); those are exactly the cells missing from its row." :
        "Short arm(s): $(join(("`:$(a)`" for a in short), ", ")). The cause is not " *
        "determinable from this table -- it is an applicable_arms exclusion, an " *
        "incomplete shard, or a deliberately narrowed report. Check the campaign " *
        "outcomes and the config records before reading across rows."
    return "NOTE: the arms above do NOT cover the same number of cells " *
           "($(join(("$(row.arm)=$(row.n_cells)" for row in rows), ", "))). " *
           cause * " The rel-L2 columns are not row-comparable."
end

function _write_comparative_audit_problem_markdown(path::AbstractString,
                                     aggregates::Vector{ProblemAggregateResult};
                                     cells_by_key=Dict{Tuple{Symbol,Symbol},Set{Int}}())
    atomic_write(String(path)) do io
        println(io, "# Comparative smoke summaries by benchmark function")
        println(io)
        println(io, "Operational smoke evidence only: one seed, fixed method-specific settings, " *
                    "unmatched compute budgets, and different uncertainty semantics. These tables " *
                    "are not a scientific ranking and do not pool unrelated functions.")
        problem_keys = sort!(unique(row.name for row in aggregates); by=string)
        for name in problem_keys
            println(io)
            println(io, "## $(name)")
            println(io)
            println(io, "| Arm | Cells | Finite / requested | Failed predictions | Cells with failures | Common / requested | Finite rel-L2 | Common rel-L2 | Fit total (s) | Inference total (s) |")
            println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            rows = sort!([row for row in aggregates if row.name == name];
                         by=row -> _comparative_arm_rank(row.arm))
            for row in rows
                println(io, "| $(row.arm) | $(row.n_cells) | $(row.n_finite_pred) / $(row.n_requested) | " *
                    "$(row.n_failed_pred) | $(row.n_cells_with_pred_failures) | " *
                    "$(row.n_common) / $(row.n_requested) | $(_comparative_audit_metric(row.rel_l2)) | " *
                    "$(_comparative_audit_metric(row.common_rel_l2)) | $(_comparative_audit_metric(row.fit_time_total_s; digits=2)) | " *
                    "$(_comparative_audit_metric(row.inference_time_total_s; digits=2)) |")
            end
            println(io)
            println(io, "Interval semantics: every arm estimates its own constants from the " *
                        "training data, so no arm's interval is a guarantee. Diona = admissible-set " *
                        "bounds under ESTIMATED noise and smoothness constants; GP/SVGP = " *
                        "variational/approximate; NSM = set-membership cone band under an ESTIMATED " *
                        "eps/gamma; NN, RBF and IDW = unavailable (no interval is produced).")
            note = _short_arm_note(rows, cells_by_key)
            if note !== nothing
                println(io)
                println(io, note)
            end
        end
    end
    return String(path)
end

function write_problem_aggregate_artifacts(
    out_dir::AbstractString,
    results::AbstractVector{ArmCellResult},
    row_groups::AbstractVector{<:AbstractVector{ArmQueryRow}};
    expected_arms=nothing,
)
    aggregates = problem_aggregate_results(results, row_groups; expected_arms=expected_arms)
    # The real per-(problem, arm) cell sets, so the short-arm NOTE can prove its
    # cause instead of inferring one from a count.
    cells_by_key = Dict{Tuple{Symbol,Symbol},Set{Int}}()
    for result in results
        push!(get!(cells_by_key, (result.name, result.arm), Set{Int}()), result.cell_id)
    end
    csv_path = joinpath(out_dir, "comparative_problem_summary.csv")
    markdown_path = joinpath(out_dir, "comparative_problem_summary.md")
    _write_comparative_summary_csv(csv_path, _comparative_audit_problem_summary_rows(aggregates))
    _write_comparative_audit_problem_markdown(markdown_path, aggregates;
                                              cells_by_key=cells_by_key)
    return (csv_path=csv_path, markdown_path=markdown_path, rows=aggregates)
end
