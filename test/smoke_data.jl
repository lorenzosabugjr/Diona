#!/usr/bin/env julia
# Data-only Diona/NN/GP benchmark harness.
# Diona, NN and GP on synthetic benchmark functions.
# Run --gate=cell-enumeration or --gate=reporting-correctness for fast checks;
# --gate=comparative-smoke / --gate=comparative-full-audit run actual model fits.
# Fixtures, settings hashes and code hashes protect paired comparisons and resume.

using Printf
using Dates
using DelimitedFiles: readdlm, writedlm
using Statistics: median, mean, std
using LinearAlgebra
using Random
using SHA
include(joinpath(@__DIR__, "ml", "common", "driver.jl"))
using .BenchmarkDriver
using .BenchmarkDriver.Diona
include(joinpath(@__DIR__, "ml", "common", "fixture.jl"))
using .SmokeFixture
include(joinpath(@__DIR__, "ml", "common", "point_fragments.jl"))
using .PointFragments
include(joinpath(@__DIR__, "ml", "common", "model_checkpoints.jl"))
using .ModelCheckpoints
include(joinpath(@__DIR__, "ml", "common", "registry.jl"))
using .ComparatorRegistry

const _GP_TRAINER_AVAILABLE = try
    include(joinpath(@__DIR__, "ml", "gp", "trainer.jl"))
    using .GPTrainer
    true
catch e
    @debug "GPTrainer unavailable in this Julia project (expected under --project=test/ml/nn_env); " *
           "the GP arm will error with an actionable message if invoked here" exception=e
    false
end

const _NN_TRAINER_AVAILABLE = try
    include(joinpath(@__DIR__, "ml", "nn", "trainer.jl"))
    using .NNTrainer
    true
catch e
    @debug "NNTrainer unavailable in-process (expected under --project=test/ml, this file's " *
           "documented default); the NN arm will subprocess to --project=test/ml/nn_env instead" exception=e
    false
end

const _NSM_AVAILABLE = try
    include(joinpath(@__DIR__, "ml", "baselines", "nsm.jl"))
    using .NSMTrainer
    true
catch e
    @debug "NSMTrainer unavailable in this Julia project; the :nsm arm will error " *
           "with an actionable message if invoked here" exception=e
    false
end

# The Surrogates.jl adapter self-guards on the package (surrogates_adapter.jl:182),
# so including it is safe in every environment; only USING :rbf/:idw without
# Surrogates.jl is an error. The fits themselves happen under
# --project=test/ml/baselines/baseline_env.
const _SURROGATE_ADAPTER_AVAILABLE = try
    include(joinpath(@__DIR__, "ml", "baselines", "surrogates_adapter.jl"))
    using .SurrogateArms
    true
catch e
    @debug "SurrogateArms unavailable in this Julia project; the :rbf and :idw arms " *
           "will error with an actionable message if invoked here" exception=e
    false
end

# Mirror of SurrogateArms.rbf_dense_system_bytes / DEFAULT_MAX_FIT_BYTES. It is
# mirrored, not imported, so that applicable_arms() stays computable even where
# the adapter failed to load; the equality is asserted below, so the two cannot
# drift into excluding different cells than the arm itself refuses.
# Measured 2026-09-21; see SurrogateArms for the measurement and the budget.
_rbf_dense_system_bytes(n::Integer, precision::Symbol = :f32) =
    (precision === :f32 ? 6.04 : 12.1) * float(n)^2
const _RBF_MAX_FIT_BYTES = 96.0 * 1024^3

# :rbf runs at ONE precision (Float32) on every cell -- see SurrogateArms for
# why. This returns that precision when the dense system fits the budget and
# `nothing` when it does not, in which case the arm is inapplicable. Keyed on n,
# never on a cell id. Every cell of this benchmark fits: the largest, n = 125,000,
# is 87.9 GiB against 96 GiB.
function _rbf_precision_for(n::Integer)
    _rbf_dense_system_bytes(n, :f32) <= _RBF_MAX_FIT_BYTES && return :f32
    return nothing
end

if _SURROGATE_ADAPTER_AVAILABLE
    (all(_rbf_dense_system_bytes(n, p) == SurrogateArms.rbf_dense_system_bytes(n, p)
         for n in (1_000, 125_000), p in (:f64, :f32)) &&
     _RBF_MAX_FIT_BYTES == SurrogateArms.DEFAULT_MAX_FIT_BYTES &&
     all(_rbf_precision_for(n) === SurrogateArms.rbf_precision_for(n)
         for n in (1_000, 16_000, 125_000, 10_000_000))) ||
        error("smoke_data.jl: the :rbf memory pre-flight mirrored here has drifted from " *
              "SurrogateArms (rbf_dense_system_bytes / rbf_precision_for / " *
              "DEFAULT_MAX_FIT_BYTES); applicable_arms would disagree with the arm itself " *
              "about which cells run, and at what precision.")
end

const NN_ARM_WORKER_PATH = joinpath(@__DIR__, "ml", "nn", "nn_arm_worker.jl")
const NN_ENV_PROJECT     = joinpath(@__DIR__, "ml", "nn_env")
const BASELINE_ARM_WORKER_PATH =
    joinpath(@__DIR__, "ml", "baselines", "baseline_arm_worker.jl")
const BASELINE_ENV_PROJECT = joinpath(@__DIR__, "ml", "baselines", "baseline_env")
Core.atdoc!(Core.atdoc)

function _parse_smoke_options(args)
    opts = Dict{String,String}()
    for arg in args
        startswith(arg, "--") || continue
        item = arg[3:end]
        if occursin('=', item)
            k, v = split(item, '='; limit=2)
            opts[k] = v
        else
            opts[item] = "true"
        end
    end
    return opts
end

const _SMOKE_OPTS = _parse_smoke_options(ARGS)
_smoke_flag(k) = lowercase(string(get(_SMOKE_OPTS, k, "false"))) in ("1", "true", "yes")
_smoke_opt(k, default=nothing) = get(_SMOKE_OPTS, k, default)
_smoke_required(k) = haskey(_SMOKE_OPTS, k) ? _SMOKE_OPTS[k] : error("missing required --$k option")

include(joinpath(@__DIR__, "ml", "harness", "cells.jl"))


# `test/`. The harness fragments under ml/harness/ and ml/gates/ need it: their own
# @__DIR__ is ml/harness, not test/, so paths written before the split would break.
const TEST_DIR       = @__DIR__
const REPO_ROOT      = normpath(joinpath(@__DIR__, ".."))
const SCRIPT_PATH     = abspath(@__FILE__)
const WORKER_PROJECT  = joinpath(@__DIR__, "ml")
const DEFAULT_JOBS_FULL_SWEEP    = 16
const DEFAULT_TIMEOUT_FULL_SWEEP = 600.0   # seconds/cell, PRD §3.8 / plan Step 6.3
const _FULL_SWEEP_PHASES = ("t_neighbor", "t_lh", "t_assembly", "t_bounds")
const _FULL_SWEEP_STATS  = ("mean", "median", "max")
_full_sweep_phase_col_names() = [Symbol("$(p)_$(s)_s") for p in _FULL_SWEEP_PHASES for s in _FULL_SWEEP_STATS]
_full_sweep_phase_vals(m) = [getproperty(m, Symbol("$(p)_$(s)")) for p in _FULL_SWEEP_PHASES for s in _FULL_SWEEP_STATS]

const _FULL_SWEEP_SUMMARY_HEADER = join((
    "run_id", "cell_id", "name", "dim", "layout", "tier_label", "n_train_nominal",
    "n_train_actual", "noise_level", "noise_bound_used", "n_query", "rel_l2", "coverage",
    "mean_width", "width_over_2eps", "t_mean_s", "t_median_s", "t_max_s",
    string.(_full_sweep_phase_col_names())..., "status_counts", "state", "error"), ',')

const _FULL_SWEEP_POINTS_HEADER = join((
    "run_id", "cell_id", "name", "layout", "tier_label", "n_train", "noise_level",
    "query_id", "truth", "pred", "lower", "upper", "covered", "width", "status", "elapsed_s"), ',')

_safe_csv_field(v) = replace(replace(string(v), '\n' => ' '), ',' => ';')

function _status_counts_str(d::Dict{Symbol,Int})
    return join(["$(k):$(v)" for (k, v) in sort(collect(d); by=x -> string(x[1]))], '|')
end

_parse_float(s::AbstractString) = isempty(s) ? NaN : parse(Float64, s)
_parse_float(value::Real) = Float64(value)
_point_int(value::Integer) = Int(value)
_point_int(value::AbstractString) = parse(Int, value)
_point_string(value) = String(value)

function atomic_write(writer::Function, path::String)
    mkpath(dirname(path))
    tmp = path * ".tmp-$(getpid())"
    open(writer, tmp, "w")
    mv(tmp, path; force=true)
    return path
end

function cell_stem(cell::SmokeCell, id::Int)
    # Named by SNR, not by a percentage: the level is an amplitude ratio
    # 10^(-SNR/20) against sigma_f, so "5" would have meant nothing here.
    noise_tag = cell.noise_level == 0.0 ? "clean" :
                @sprintf("snr%ddb", round(Int, level_to_snr_db(cell.noise_level)))
    return @sprintf("%03d_%s_%s_n%d_%s",
                     id, cell.name, cell.layout, cell.n_train, noise_tag)
end

function cell_matches_filter(cell::SmokeCell, filter_str::AbstractString)
    haystack = lowercase(join((string(cell.name),
                                string(cell.layout), cell.tier_label), '_'))
    needles = split(lowercase(filter_str), ',')
    return any(n -> occursin(strip(n), haystack), needles)
end

function smoke_data_default_out_dir()
    stamp  = Dates.format(Dates.now(), "yyyy-mm-dd-HHMMSS")
    suffix = bytes2hex(rand(Random.RandomDevice(), UInt8, 4))
    return joinpath(REPO_ROOT, "results", "smoke_data", "$(stamp)-$(suffix)")
end

summary_fragment_path(out_dir, cell::SmokeCell, id::Int) =
    joinpath(out_dir, "fragments", "summary", cell_stem(cell, id) * ".csv")
const _SMOKE_MODEL_SEED = 0

function fixture_paths(out_dir, cell::SmokeCell, id::Int)
    fpath = joinpath(out_dir, "fixtures", cell_stem(cell, id) * ".fixture")
    return fpath, fixture_manifest_path(fpath)
end

function materialize_fixture(prep::PreparedSmokeCell, out_dir::AbstractString;
                              run_id::AbstractString=basename(out_dir))
    cell = prep.cell
    fpath, mpath = fixture_paths(out_dir, cell, prep.cell_id)
    return write_fixture(fpath;
        schema_version      = prep.schema_version,
        cell_id             = prep.cell_id,
        name                = string(cell.name),
        dim                 = cell.dim,
        layout              = string(cell.layout),
        tier_label          = cell.tier_label,
        n_train_nominal     = cell.n_train,
        n_train_actual      = prep.n_train_actual,
        n_query             = prep.n_query,
        noise_level         = prep.noise_level,
        noise_bound         = prep.noise_bound,
        noise_seed          = prep.noise_seed,
        model_seed          = _SMOKE_MODEL_SEED,
        output_range        = prep.output_range,
        x_train             = prep.X_train,
        y_train_clean       = prep.y_train_clean,
        y_train_noisy       = prep.y_train_noisy,
        x_query             = prep.X_query,
        y_query_truth       = prep.y_query_truth,
        query_ids           = prep.query_ids,
        generated_at        = prep.generated_at,
        manifest_path       = mpath,
    )
end

include(joinpath(@__DIR__, "ml", "harness", "arms.jl"))


include(joinpath(@__DIR__, "ml", "harness", "reporting.jl"))
include(joinpath(@__DIR__, "ml", "harness", "sweep.jl"))

include(joinpath(@__DIR__, "ml", "harness", "campaign.jl"))

# ---------------------------------------------------------------------------
# Gates. Plain includes into this same flat scope, appended last so that every
# definition they exercise already exists. See test/ml/gates/ for the split.
# ---------------------------------------------------------------------------
include(joinpath(@__DIR__, "ml", "gates", "cells_gates.jl"))
include(joinpath(@__DIR__, "ml", "gates", "arms_gates.jl"))
include(joinpath(@__DIR__, "ml", "gates", "reporting_gates.jl"))
include(joinpath(@__DIR__, "ml", "gates", "campaign_gates.jl"))

if abspath(PROGRAM_FILE) == (@__FILE__) && _smoke_flag("worker")
    run_worker()
elseif abspath(PROGRAM_FILE) == (@__FILE__)
    gate = _smoke_opt("gate", "all")
    gate in ("all", "cell-enumeration", "data-loaders",
             "single-cell-proof", "metrics-aggregation", "full-sweep", "arms",
             "reporting-correctness", "orchestration-resilience", "fast-comparative-smoke",
             "comparative-smoke", "comparative-full", "comparative-full-audit") ||
        error("unknown benchmark gate: $gate")

    if gate in ("cell-enumeration", "all")
        println()
        ok_cell_enum = check_gate_cell_enumeration(; verbose=true)
        ok_cell_enum || error("Cell enumeration gate FAILED — fix enumerate_cells() before proceeding")
    end

    if gate in ("data-loaders", "all")
        println()
        ok_loaders = check_gate_data_loaders(; verbose=true)
        ok_loaders || error("Data loaders gate FAILED — see mismatches above")
    end
    single_cell_cr = nothing
    if gate in ("single-cell-proof", "metrics-aggregation", "all")
        println()
        ok_single_cell, single_cell_cr =
            check_gate_single_cell_proof(; verbose=(gate != "metrics-aggregation"))
        gate == "single-cell-proof" &&
            (ok_single_cell || error("Single-cell proof gate FAILED — see report above"))
    end

    if gate in ("metrics-aggregation", "all")
        println()
        ok_metrics = check_gate_metrics_aggregation(; cr=single_cell_cr, verbose=true)
        ok_metrics || error("Metrics aggregation gate FAILED — see report above")
    end

    if gate == "full-sweep"
        run_orchestrator()
    end

    if gate == "arms"
        ok_arms = check_gate_arms(; verbose=true)
        ok_arms || error("Multi-arm anchor gate FAILED — see report above")
    end

    if gate == "reporting-correctness"
        ok_reporting = check_gate_reporting_correctness(; verbose=true)
        ok_reporting || error("Reporting correctness gate FAILED — see report above")
    end

    if gate == "orchestration-resilience"
        ok_resilience, _ = check_gate_orchestration_resilience(; verbose=true)
        ok_resilience || error("Orchestration resilience gate FAILED — see report above")
    end

    if gate == "fast-comparative-smoke"
        compare_to = _smoke_opt("compare-to", nothing)
        ok_fast_smoke, _ = check_gate_fast_comparative_smoke(; compare_to=compare_to, verbose=true)
        ok_fast_smoke || error("Fast comparative smoke gate FAILED — see report above")
    end

    if gate == "comparative-smoke"
        campaign = run_comparative_orchestrator(:representative)
        campaign.success || error("representative comparative campaign completed with failures; successful artifacts were retained")
    end

    if gate == "comparative-full"
        campaign = run_comparative_orchestrator(:full)
        if _smoke_opt("only", nothing) === nothing && _smoke_opt("cell-ids", nothing) === nothing
            length(campaign.expectations) == sum(length(applicable_arms(c)) for c in enumerate_cells()) ||
                error("full comparative campaign selected $(length(campaign.expectations)) arm-cells, expected $(sum(length(applicable_arms(c)) for c in enumerate_cells()))")
        end
        campaign.success || error("full comparative campaign completed with failures; successful artifacts were retained")
    end

    if gate == "comparative-full-audit"
        campaign = run_comparative_orchestrator(:full)
        length(campaign.expectations) == sum(length(applicable_arms(c)) for c in enumerate_cells()) ||
            error("full audited comparative campaign selected $(length(campaign.expectations)) arm-cells, expected $(sum(length(applicable_arms(c)) for c in enumerate_cells()))")
        problem_artifacts = write_problem_aggregate_artifacts(
            campaign.out_dir, campaign.results, campaign.row_groups)
        audit = check_gate_comparative_full_audit(campaign; problem_artifacts=problem_artifacts, verbose=true)
        audit.overall || error("Comparative complete-accounting audit FAILED — see $(audit.audit_path)")
        campaign.success || error("Audited campaign retained complete failure artifacts but had terminal worker failures")
    end

    println()
    println("smoke_data.jl: requested gates complete.")
end
