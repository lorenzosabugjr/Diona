#!/usr/bin/env julia

module FullBenchmarkDriver

using Dates
using Pkg
using Random

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))
const ENVIRONMENTS = (
    (name="test/ml", path=joinpath(@__DIR__, "ml"), automatic=true),
    (name="test/ml/nn_env", path=joinpath(@__DIR__, "ml", "nn_env"), automatic=true),
    (name="test/ml/baselines/baseline_env",
     path=joinpath(@__DIR__, "ml", "baselines", "baseline_env"), automatic=true),
)
const COMPARATIVE_AUDIT_PASS = "Comparative full-audit gate overall: PASS"

# ---------------------------------------------------------------------------
# Per-arm knob tables.
#
# Every per-arm command-line option, default and manifest line below is
# DERIVED from `BENCHMARK_ARMS`. Adding a seventh arm means adding one entry
# here plus one default timeout; no other edit to this file is needed. The
# previous hardcoded --jobs-diona / --jobs-nn / --jobs-gp / --timeout-gp
# quartet is gone.
#
# The harness (test/smoke_data.jl) decides which arms are applicable to which
# cell via `applicable_arms(cell)`; this list is only the set of arms whose
# knobs the driver is prepared to pass through. Passing a knob for an arm the
# harness never schedules is harmless: the orchestrator reads its own arm
# order out of the dictionaries and ignores surplus keys.
# ---------------------------------------------------------------------------
const BENCHMARK_ARMS = (:diona, :nn, :gp, :nsm, :rbf, :idw)

# Per-cell wall-clock ceilings, seconds.
#
# :diona is 6 h. The previous default was 3.00 h and the largest single cell of
# the last campaign took 2.98 h against it -- a 0.7% margin, i.e. effectively
# none. Six hours restores real headroom without making a genuinely hung cell
# cheap to sit through.
#
# :nsm, :rbf and :idw are UNPROBED allowances, not measurements: all three are
# direct fits with no iterative training, but the only timing anywhere is at
# n = 1000, and :nsm's gamma estimator is a blocked pairwise scan over the
# training set -- O(n^2) = 1.56e10 pairs on the 125,000-point grid cells.
#
# MUST EQUAL smoke_data.jl's DEFAULT_TIMEOUT_COMPARATIVE and benchmark_data.jl's
# _DATA_DEFAULT_TIMEOUT_S. This driver always passes its own table into the
# orchestrator, so when the three disagree the one that applies is not the one
# an operator reads; they disagreed by 2x on :nsm/:rbf/:idw. The equality is
# asserted in test/tests_benchmark_data_config.jl, which is the only place that
# can see all three at once.
const DEFAULT_ARM_TIMEOUTS_S = Dict{Symbol,Float64}(
    :diona => 21600.0,   # measured: worst cell of the last campaign 2.98 h
    :nn    => 1800.0,
    :gp    => 3600.0,
    :nsm   => 7200.0,    # UNPROBED
    :rbf   => 7200.0,    # UNPROBED
    :idw   => 7200.0,    # UNPROBED
)

for arm in BENCHMARK_ARMS
    haskey(DEFAULT_ARM_TIMEOUTS_S, arm) ||
        error("run_full_benchmark.jl: BENCHMARK_ARMS lists :$arm with no DEFAULT_ARM_TIMEOUTS_S entry")
end

# The devices `--device` accepts. :nn is the only arm that has a device at all;
# every other arm is CPU-only and the orchestrator pins it to "cpu".
#
# MUST EQUAL ComparatorRegistry.SUPPORTED_DEVICES. It is mirrored rather than
# imported so this driver keeps parsing its arguments before it has bootstrapped
# any environment or loaded the harness. The equality is asserted in
# test/tests_benchmark_data_config.jl, which can see both at once.
const SUPPORTED_DEVICES = ("cpu", "cuda")

jobs_option(arm::Symbol) = "jobs-" * String(arm)
timeout_option(arm::Symbol) = "timeout-" * String(arm)

struct DriverOptions
    out_dir::String
    jobs::Dict{Symbol,Int}
    timeouts::Dict{Symbol,Float64}
    blas_threads::Int
    device::String
end

"""
    DriverOptions(out_dir, jobs_diona, jobs_nn, jobs_gp, blas_threads, device, timeout_gp)

Legacy three-arm positional form, kept so callers written against the
pre-generalisation driver keep working. It populates the job table with
exactly `:diona`, `:nn` and `:gp`, which is what a three-arm caller means.
"""
function DriverOptions(out_dir::AbstractString, jobs_diona::Integer, jobs_nn::Integer,
                       jobs_gp::Integer, blas_threads::Integer, device::AbstractString,
                       timeout_gp::Union{Nothing,Real})
    jobs = Dict{Symbol,Int}(:diona => Int(jobs_diona), :nn => Int(jobs_nn),
                            :gp => Int(jobs_gp))
    timeouts = Dict{Symbol,Float64}(arm => DEFAULT_ARM_TIMEOUTS_S[arm] for arm in keys(jobs))
    timeout_gp === nothing || (timeouts[:gp] = Float64(timeout_gp))
    return DriverOptions(String(out_dir), jobs, timeouts, Int(blas_threads), String(device))
end

# `options.jobs_<arm>` / `options.timeout_<arm>` resolve against the tables, so
# per-arm reads do not need a struct field per arm.
function Base.getproperty(options::DriverOptions, name::Symbol)
    name in fieldnames(DriverOptions) && return getfield(options, name)
    text = String(name)
    if startswith(text, "jobs_")
        return getfield(options, :jobs)[Symbol(text[(length("jobs_") + 1):end])]
    elseif startswith(text, "timeout_")
        return get(getfield(options, :timeouts),
                   Symbol(text[(length("timeout_") + 1):end]), nothing)
    end
    return getfield(options, name)
end

# The UNDERSCORE spellings, which are what `getproperty` above resolves.
# `jobs_option`/`timeout_option` produce the CLI flag spellings (`jobs-diona`),
# and advertising those here made propertynames list 12 names that getproperty
# could not resolve -- so any generic consumer that iterates propertynames and
# reads them (a serializer, `dump`, a manifest writer) threw
# `type DriverOptions has no field jobs-diona`.
_jobs_property(arm::Symbol) = Symbol("jobs_", arm)
_timeout_property(arm::Symbol) = Symbol("timeout_", arm)

Base.propertynames(options::DriverOptions) = (
    fieldnames(DriverOptions)...,
    (_jobs_property(arm) for arm in sort!(collect(keys(getfield(options, :jobs)))))...,
    (_timeout_property(arm) for arm in sort!(collect(keys(getfield(options, :timeouts)))))...,
)

"""Arms this option set carries knobs for, in `BENCHMARK_ARMS` order."""
configured_arms(options::DriverOptions) =
    Tuple(arm for arm in BENCHMARK_ARMS if haskey(getfield(options, :jobs), arm))

function default_out_dir(; now=Dates.now(), random_bytes=rand(Random.RandomDevice(), UInt8, 4))
    stamp = Dates.format(now, "yyyy-mm-dd-HHMMSS")
    suffix = bytes2hex(random_bytes)
    return joinpath(REPOSITORY_ROOT, "results", "$(stamp)-$(suffix)")
end

function usage(io::IO=stdout)
    println(io, "Usage: julia --project=test/ml --startup-file=no test/run_full_benchmark.jl [options]")
    println(io)
    println(io, "Arms: " * join(String.(BENCHMARK_ARMS), ", "))
    println(io)
    println(io, "Options:")
    println(io, "  --out=PATH          Output directory (default: results/<timestamp>-<8hex>)")
    jobs_default = max(1, Sys.CPU_THREADS - 4)
    for arm in BENCHMARK_ARMS
        flag = "--" * jobs_option(arm) * "=N"
        println(io, "  " * rpad(flag, 20) *
                    "Concurrent $(arm) workers (default: $(jobs_default))")
    end
    for arm in BENCHMARK_ARMS
        flag = "--" * timeout_option(arm) * "=SEC"
        println(io, "  " * rpad(flag, 20) *
                    "Per-cell timeout for the $(arm) arm (default: $(DEFAULT_ARM_TIMEOUTS_S[arm]))")
    end
    println(io, "  --blas-threads=N    BLAS threads per comparative worker (default: 1)")
    println(io, "  --device=DEV        :nn device, cpu or cuda (default: cpu). Every other")
    println(io, "                      arm is CPU-only and ignores it.")
    println(io, "  --help              Show this message")
end

function _option_dict(args)
    values = Dict{String,String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        startswith(arg, "--") || error("unexpected positional argument: $arg")
        item = arg[3:end]
        if occursin('=', item)
            key, value = split(item, '='; limit=2)
            isempty(value) && error("--$key requires a value")
            values[key] = value
        elseif item == "help"
            values[item] = "true"
        else
            i == length(args) && error("--$item requires a value")
            startswith(args[i + 1], "--") && error("--$item requires a value")
            values[item] = args[i + 1]
            i += 1
        end
        i += 1
    end
    allowed = Set{String}(("out", "blas-threads", "device", "help"))
    for arm in BENCHMARK_ARMS
        push!(allowed, jobs_option(arm))
        push!(allowed, timeout_option(arm))
    end
    unknown = sort!(collect(setdiff(Set(keys(values)), allowed)))
    isempty(unknown) || error("unknown option(s): " * join(("--" * key for key in unknown), ", "))
    return values
end

function parse_options(args=ARGS)
    values = _option_dict(args)
    get(values, "help", "false") == "true" && return nothing
    jobs_default = max(1, Sys.CPU_THREADS - 4)
    positive_int(key, default) = begin
        value = parse(Int, get(values, key, string(default)))
        value >= 1 || error("--$key must be >= 1")
        value
    end
    positive_float(key, default) = begin
        value = parse(Float64, get(values, key, string(default)))
        value > 0 || error("--$key must be > 0")
        value
    end
    # cpu or cuda. It reaches the :nn arm only, and it is part of that arm's
    # settings hash, so a CPU fragment and a CUDA fragment never collide.
    device = lowercase(strip(get(values, "device", "cpu")))
    device in SUPPORTED_DEVICES ||
        error("--device must be one of " * join(SUPPORTED_DEVICES, ", ") * ", got '$device'")
    out_dir = abspath(get(values, "out", default_out_dir()))
    jobs = Dict{Symbol,Int}(
        arm => positive_int(jobs_option(arm), jobs_default) for arm in BENCHMARK_ARMS)
    timeouts = Dict{Symbol,Float64}(
        arm => positive_float(timeout_option(arm), DEFAULT_ARM_TIMEOUTS_S[arm])
        for arm in BENCHMARK_ARMS)
    return DriverOptions(out_dir, jobs, timeouts,
                         positive_int("blas-threads", 1), device)
end

function _with_environment(f::Function, path::AbstractString)
    previous = Base.active_project()
    try
        Pkg.activate(path; io=devnull)
        return f()
    finally
        previous === nothing || Pkg.activate(dirname(previous); io=devnull)
    end
end

function environment_snapshot(path::AbstractString)
    return _with_environment(path) do
        direct = Set(values(Pkg.project().dependencies))
        dependencies = Pkg.dependencies()
        packages = [(
            name=info.name,
            uuid=string(uuid),
            version=info.version === nothing ? "stdlib-or-path" : string(info.version),
            direct=uuid in direct,
            source=(hasproperty(info, :source) && info.source !== nothing ? String(info.source) : ""),
        ) for (uuid, info) in dependencies]
        sort!(packages; by=package -> (lowercase(package.name), package.uuid))
        return packages
    end
end

function environment_resolved(path::AbstractString)
    manifest = joinpath(path, "Manifest.toml")
    isfile(manifest) || return false
    try
        return _with_environment(path) do
            direct = Set(values(Pkg.project().dependencies))
            dependencies = Pkg.dependencies()
            all(haskey(dependencies, uuid) for uuid in direct) || return false
            all(begin
                !hasproperty(info, :source) ||
                    (info.source === nothing ? info.version === nothing : ispath(info.source))
            end for info in values(dependencies))
        end
    catch
        return false
    end
end

function _instantiate_environment(path::AbstractString)
    _with_environment(path) do
        Pkg.instantiate()
    end
    environment_resolved(path) || error("environment remains unresolved after Pkg.instantiate(): $path")
end


function bootstrap_environments!()
    snapshots = Dict{String,Any}()
    for environment in ENVIRONMENTS
        if !environment_resolved(environment.path)
            println("Instantiating $(environment.name).")
            _instantiate_environment(environment.path)
        end
        snapshots[environment.name] = environment_snapshot(environment.path)
    end
    return snapshots
end

"""
    expected_arm_cell_count(harness, cells) -> Int

Number of (cell, arm) fragments a complete campaign must produce, summed
CELL BY CELL from `applicable_arms(cell)`.

This must stay a per-cell sum rather than `length(cells) * length(arms)`:
`applicable_arms` is not uniform. `:rbf` is excluded on the two 50^3 Ishigami
grid cells (n = 125,000), where a dense n x n interpolation system would
project about 233 GiB, so those two cells expect one arm fewer than the rest.
Duplicates are collapsed so a repeated entry cannot inflate the expectation.
"""
function expected_arm_cell_count(harness, cells)
    applicable = getproperty(harness, :applicable_arms)
    total = 0
    for cell in cells
        arms = Base.invokelatest(applicable, cell)
        total += length(Set(Symbol(arm) for arm in arms))
    end
    return total
end

# `device` must be the device the campaign ran with: the :nn settings hash
# carries it, so expectations built at the wrong one look for fragments nothing
# wrote and the completed campaign is re-run instead of skipped.
function comparative_audit_valid(harness, out_dir::AbstractString; verbose::Bool=true,
                                 device::AbstractString="cpu")
    audit_path = joinpath(out_dir, "gate_comparative_full_audit.txt")
    if !isfile(audit_path) || !occursin(COMPARATIVE_AUDIT_PASS, read(audit_path, String))
        return false
    end
    required_artifacts = (
        "comparative_summary.csv",
        "comparative_problem_summary.csv",
        "comparative_problem_summary.md",
        "comparative_outcomes.csv",
    )
    all(name -> isfile(joinpath(out_dir, name)), required_artifacts) || return false
    try
        cells = Base.invokelatest(getproperty(harness, :enumerate_cells))
        ids = collect(eachindex(cells))
        registry = Base.invokelatest(
            getproperty(getproperty(harness, :ComparatorRegistry), :canonical_registry))
        expectations, _ = Base.invokelatest(
            getproperty(harness, :_comparative_prepare_expectations),
            ids, cells, String(out_dir), registry; device=String(device))
        expected_count = expected_arm_cell_count(harness, cells)
        if length(expectations) != expected_count
            verbose && println("Comparative audit validation: campaign holds $(length(expectations)) " *
                               "arm-cells but applicable_arms sums to $expected_count over " *
                               "$(length(cells)) cells.")
            return false
        end
        for expected in expectations
            check = Base.invokelatest(
                getproperty(harness, :_comparative_try_read), expected;
                allow_failure=false)
            check.valid || return false
        end
        verbose && println("Comparative audit and all current fixture/settings hashes are valid.")
        return true
    catch error_value
        verbose && println("Comparative audit validation failed: ", sprint(showerror, error_value))
        return false
    end
end

function _git_commit()
    command = `git -C $REPOSITORY_ROOT rev-parse HEAD`
    commit = strip(read(command, String))
    occursin(r"^[0-9a-f]{40}$", commit) || error("git rev-parse returned an invalid commit hash")
    return commit
end

function _atomic_write(path::AbstractString, text::AbstractString)
    mkpath(dirname(path))
    temporary = path * ".tmp-$(getpid())"
    open(temporary, "w") do io
        write(io, text)
    end
    mv(temporary, path; force=true)
    return String(path)
end

function write_run_manifest(path::AbstractString, options::DriverOptions,
                            timings, snapshots; generated_at=Dates.now(),
                            git_commit=_git_commit())
    io = IOBuffer()
    println(io, "schema=diona-full-benchmark-run-v1")
    timestamp = Dates.format(generated_at, "yyyy-mm-ddTHH:MM:SS")
    println(io, "generated_at=$timestamp")
    println(io, "git_commit=$git_commit")
    println(io, "julia_version=$(VERSION)")
    println(io, "out_dir=$(options.out_dir)")
    arms = configured_arms(options)
    println(io, "arms=" * join(String.(arms), ","))
    for arm in arms
        println(io, "jobs_$(arm)=$(getfield(options, :jobs)[arm])")
    end
    for arm in arms
        timeout = get(getfield(options, :timeouts), arm, nothing)
        println(io, "timeout_$(arm)_s=$(something(timeout, "default"))")
    end
    println(io, "blas_threads=$(options.blas_threads)")
    println(io, "device=$(options.device)")
    for stage in ("bootstrap", "comparative", "report")
        println(io, "wall_clock_$(stage)_s=$(get(timings, stage, NaN))")
    end
    for environment in ENVIRONMENTS
        println(io)
        println(io, "[environment $(environment.name)]")
        for package in get(snapshots, environment.name, Any[])
            println(io, "package=$(package.name)|$(package.uuid)|$(package.version)|direct=$(package.direct)")
        end
    end
    return _atomic_write(path, String(take!(io)))
end

function run_benchmark(harness, options::DriverOptions, snapshots;
                       bootstrap_time_s::Real=NaN,
                       comparative_validator=comparative_audit_valid)
    mkpath(options.out_dir)
    if isdefined(harness, :_SMOKE_OPTS)
        harness_options = getproperty(harness, :_SMOKE_OPTS)
        harness_options["out"] = options.out_dir
        harness_options["blas-threads"] = string(options.blas_threads)
        harness_options["device"] = options.device
    end
    timings = Dict{String,Float64}("bootstrap" => Float64(bootstrap_time_s))

    started = time()
    if comparative_validator(harness, options.out_dir; verbose=true, device=options.device)
        println("Skipping the completed comparative campaign.")
    else
        # Both tables are always passed, for every configured arm. The
        # orchestrator iterates its OWN arm order over them and falls back to
        # its own defaults for anything absent, so a driver that knows about
        # more arms than the harness schedules is not an error.
        jobs = copy(getfield(options, :jobs))
        timeout_override = copy(getfield(options, :timeouts))
        campaign = Base.invokelatest(
            getproperty(harness, :run_comparative_orchestrator), :full;
            out_dir_override=options.out_dir, jobs_override=jobs,
            timeout_override=timeout_override)
        artifacts = Base.invokelatest(
            getproperty(harness, :write_problem_aggregate_artifacts),
            campaign.out_dir, campaign.results, campaign.row_groups)
        audit = Base.invokelatest(
            getproperty(harness, :check_gate_comparative_full_audit), campaign;
            problem_artifacts=artifacts, verbose=true)
        audit.overall || error("comparative complete-accounting audit failed: $(audit.audit_path)")
        campaign.success || error("comparative campaign retained terminal worker failures")
    end
    timings["comparative"] = time() - started

    manifest_path = joinpath(options.out_dir, "run_manifest.txt")
    report_started = time()
    timings["report"] = NaN
    write_run_manifest(manifest_path, options, timings, snapshots)
    timings["report"] = time() - report_started
    write_run_manifest(manifest_path, options, timings, snapshots)
    println("Full benchmark completed successfully.")
    println("Output directory: $(options.out_dir)")
    println("Run manifest: $manifest_path")
    return (out_dir=options.out_dir, manifest_path=manifest_path,
            timings=timings)
end

end # module FullBenchmarkDriver

if abspath(PROGRAM_FILE) == @__FILE__
    options = FullBenchmarkDriver.parse_options(ARGS)
    if options === nothing
        FullBenchmarkDriver.usage()
        exit(0)
    end

    bootstrap_started = time()
    snapshots = FullBenchmarkDriver.bootstrap_environments!()
    include(joinpath(@__DIR__, "smoke_data.jl"))
    bootstrap_time_s = time() - bootstrap_started

    FullBenchmarkDriver.run_benchmark(
        Main, options, snapshots; bootstrap_time_s=bootstrap_time_s)
end
