"""
Schema tests for the data-only, config-driven benchmark runner.

These tests exercise `load_data_config` / `resolve_data_config` ONLY. They never
fit a model, never start a worker and never touch `run.out` -- the one test that
mentions the orchestrator checks the shape of the override dicts a config
resolves to, not a call.

Run standalone:

    julia --project=test/ml test/tests_benchmark_data_config.jl

They are NOT wired into test/runtests.jl: that runs under the root project,
which has no YAML dependency (YAML lives in test/ml/Project.toml), and the root
Project.toml is out of bounds for this change.
"""

using Test, YAML

# Guard prevents const redefinitions when run from a live session that already
# included benchmark_data.jl.
if !@isdefined(load_data_config)
    include(joinpath(@__DIR__, "benchmark_data.jl"))
end

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

function write_cfg(d::Dict)
    path, io = mktemp()
    YAML.write(io, d)
    close(io)
    return path
end

"""
    _merged(defaults, extra) -> Dict{String,Any}

Merge overrides onto a section's defaults. A `nothing` override DELETES the
default key, which is how a test asks for a section that does not carry it --
writing `nothing` into the YAML instead would round-trip as an explicit null
and `haskey` would still find it.
"""
function _merged(defaults::Dict{String,Any}, extra)
    d = merge(Dict{String,Any}(defaults), Dict{String,Any}(extra))
    for (k, v) in collect(d)
        v === nothing && delete!(d, k)
    end
    return d
end

function minimal_cfg(; extra_top=Dict{String,Any}(), extra_cells=Dict{String,Any}(),
                       extra_run=Dict{String,Any}(), extra_solver=nothing)
    d = Dict{String,Any}(
        "name"  => "unit-test",
        "arms"  => ["diona"],
        "cells" => _merged(Dict{String,Any}("ids" => [1]), extra_cells),
        "run"   => _merged(Dict{String,Any}("out" => "results/unit-test-out"), extra_run),
    )
    if extra_solver !== nothing
        d["solver"] = Dict{String,Any}(extra_solver)
    end
    for (k, v) in extra_top
        d[string(k)] = v
    end
    return d
end

# Delete a default key (rather than override it) so tests can build configs
# that are missing a required section.
function cfg_without(d::Dict, section::String, key::String)
    out = deepcopy(d)
    delete!(out[section], key)
    return out
end

load_err(d::Dict) = try
    load_data_config(write_cfg(d))
    nothing
catch exception
    exception
end

msg_of(err) = sprint(showerror, err)

const _CELLS = enumerate_cells()

# ─────────────────────────────────────────────────────────────────────────────
# Unknown keys are rejected BY NAME, with the accepted set printed
# ─────────────────────────────────────────────────────────────────────────────

@testset "unknown keys are rejected by name" begin
    for (section, cfg) in (
            ("(top-level)", minimal_cfg(; extra_top=Dict("bogus" => 1))),
            ("cells",       minimal_cfg(; extra_cells=Dict("bogus" => 1))),
            ("run",         minimal_cfg(; extra_run=Dict("bogus" => 1))),
            ("solver",      minimal_cfg(; extra_solver=Dict("bogus" => 1))))
        err = load_err(cfg)
        @test err isa ErrorException
        m = msg_of(err)
        @test occursin("Unknown key 'bogus'", m)
        @test occursin("in '$(section)'", m)
        @test occursin("Accepted:", m)
    end

    # The accepted list is the load-bearing part: a stale config must be able to
    # correct itself from the message alone.
    m = msg_of(load_err(minimal_cfg(; extra_run=Dict("bogus" => 1))))
    for k in ("blas_threads", "jobs", "out", "resume", "run_id", "timeout_s")
        @test occursin(k, m)
    end
end

@testset "removed sections and keys name their replacement" begin
    cases = [
        (minimal_cfg(; extra_top=Dict("test_data" => Dict("truthfile" => "t.csv"))),
         ["Removed section 'test_data'", "enumerate_cells", "'cells'"]),
        (minimal_cfg(; extra_top=Dict("pde" => Dict("plugin" => "x"))),
         ["Removed section 'pde'", "data-only"]),
        (minimal_cfg(; extra_top=Dict("diagnostics" => Dict("enabled" => true))),
         ["Removed section 'diagnostics'"]),
        (minimal_cfg(; extra_top=Dict("output" => Dict("console" => true))),
         ["Removed section 'output'", "run.out"]),
        (minimal_cfg(; extra_top=Dict("pipeline" => Dict("mode" => ["predict"]))),
         ["Renamed section 'pipeline'", "'cells'", "'run'"]),
        (minimal_cfg(; extra_cells=Dict("mode" => ["predict"])),
         ["Removed key 'cells.mode'", "nominal and interval together"]),
        (minimal_cfg(; extra_cells=Dict("uq" => "on")),
         ["Removed key 'cells.uq'"]),
        (minimal_cfg(; extra_solver=Dict("noise" => 0.1)),
         ["Removed key 'solver.noise'", "ESTIMATED", "oracle"]),
        (minimal_cfg(; extra_solver=Dict("neighbor_method" => "hybrid")),
         ["Removed key 'solver.neighbor_method'", "settings_hash"]),
        (minimal_cfg(; extra_solver=Dict("neighbor_param" => 20)),
         ["Removed key 'solver.neighbor_param'"]),
        (minimal_cfg(; extra_solver=Dict("neighbor_fit_layers" => 3)),
         ["Removed key 'solver.neighbor_fit_layers'"]),
        (minimal_cfg(; extra_solver=Dict("n_dim" => 3)),
         ["Removed key 'solver.n_dim'", "SmokeCell.dim"]),
        (minimal_cfg(; extra_solver=Dict("temporal_edges" => true)),
         ["Removed key 'solver.temporal_edges'", "Algorithm 2"]),
        (minimal_cfg(; extra_solver=Dict("time_weight" => 1.0)),
         ["Removed key 'solver.time_weight'"]),
        (minimal_cfg(; extra_solver=Dict("depth" => 2)),
         ["Removed key 'solver.depth'", "no graph depth"]),
        (minimal_cfg(; extra_solver=Dict("compute_uq" => false)),
         ["Removed key 'solver.compute_uq'", "no nominal-only mode"]),
        (minimal_cfg(; extra_solver=Dict("mu0" => 1e-3)),
         ["Removed key 'solver.mu0'", "_SMOKE_DIONA_MU0"]),
        (minimal_cfg(; extra_solver=Dict("layers" => 1)),
         ["Removed key 'solver.layers'", "settings_hash"]),
    ]
    for (cfg, needles) in cases
        err = load_err(cfg)
        @test err isa ErrorException
        m = msg_of(err)
        for needle in needles
            @test occursin(needle, m)
        end
        # A removed key must NOT collapse back into the generic rejection.
        @test !occursin("Unknown key", m)
    end
end

@testset "solver.L_H rejects anything but estimate, on protocol grounds" begin
    err = load_err(minimal_cfg(; extra_solver=Dict("L_H" => "global")))
    @test err isa ErrorException
    m = msg_of(err)
    @test occursin("solver.L_H must be \"estimate\"", m)
    @test occursin("a priori Lipschitz constant", m)
end

@testset "missing required sections are named" begin
    for section in ("name", "arms", "cells", "run")
        d = minimal_cfg()
        delete!(d, section)
        err = load_err(d)
        @test err isa ErrorException
        @test occursin("Config missing '$(section)' section", msg_of(err))
    end
    err = load_err(cfg_without(minimal_cfg(), "run", "out"))
    @test err isa ErrorException
    @test occursin("run: missing required key 'out'", msg_of(err))
end

# ─────────────────────────────────────────────────────────────────────────────
# Scalar -> list normalisation
# ─────────────────────────────────────────────────────────────────────────────

@testset "scalar values normalise to lists" begin
    _, scalar_arm = load_data_config(write_cfg(
        minimal_cfg(; extra_top=Dict("arms" => "diona"))))
    _, list_arm = load_data_config(write_cfg(minimal_cfg()))
    @test scalar_arm.arms == list_arm.arms == [:diona]

    _, scalar_id = load_data_config(write_cfg(
        minimal_cfg(; extra_cells=Dict("ids" => 7))))
    @test scalar_id.selected_ids == [7]

    _, scalar_fn = load_data_config(write_cfg(
        minimal_cfg(; extra_cells=Dict("ids" => nothing,
                                       "functions" => "rosenbrock"))))
    @test !isempty(scalar_fn.selected_ids)

    _, scalar_noise = load_data_config(write_cfg(
        minimal_cfg(; extra_cells=Dict("ids" => nothing, "noise" => 0.1))))
    @test all(_CELLS[id].noise_level == 0.1 for id in scalar_noise.selected_ids)

    _, scalar_tier = load_data_config(write_cfg(
        minimal_cfg(; extra_cells=Dict("ids" => nothing, "tiers" => "500D"))))
    @test all(_CELLS[id].tier_label == "500D" for id in scalar_tier.selected_ids)
end

@testset "arms dedupe, lowercase and sort into canonical order" begin
    _, r = load_data_config(write_cfg(minimal_cfg(;
        extra_top=Dict("arms" => ["IDW", "diona", "idw", "Gp"]))))
    @test r.arms == [:diona, :gp, :idw]
end

@testset "tier and noise aliases normalise" begin
    base(x) = minimal_cfg(; extra_cells=Dict("ids" => nothing, "tiers" => x))
    for (spec, want) in (("500D", "500D"), ("500d", "500D"), (500, "500D"),
                         ("2000D", "2000D"), ("10^3", "10³"), ("10cubed", "10³"),
                         (10, "10³"), ("50^3", "50³"), (50, "50³"))
        _, r = load_data_config(write_cfg(base(spec)))
        @test !isempty(r.selected_ids)
        @test all(_CELLS[id].tier_label == want for id in r.selected_ids)
    end
    # The level is an amplitude ratio 10^(-SNR/20); every spelling of the
    # 20 dB condition must land on 0.1, and every spelling of noiseless on 0.0.
    for (spec, want) in ((0, 0.0), ("0", 0.0), ("0%", 0.0),
                         ("clean", 0.0), ("noiseless", 0.0), ("none", 0.0),
                         (0.1, 0.1), ("0.1", 0.1),
                         (20, 0.1), ("20", 0.1), ("20db", 0.1), ("20dB", 0.1),
                         (20.0, 0.1))
        _, r = load_data_config(write_cfg(
            minimal_cfg(; extra_cells=Dict("ids" => nothing, "noise" => spec))))
        @test all(_CELLS[id].noise_level == want for id in r.selected_ids)
    end
end

@testset "bad enum values are rejected with the accepted set" begin
    err = load_err(minimal_cfg(; extra_top=Dict("arms" => ["dionaa"])))
    @test err isa ErrorException
    @test occursin("Unknown arm 'dionaa'", msg_of(err))
    @test occursin("diona", msg_of(err))

    err = load_err(minimal_cfg(; extra_cells=Dict("ids" => nothing,
                                                  "functions" => ["borehole"])))
    @test err isa ErrorException
    @test occursin("Unknown function 'borehole'", msg_of(err))
    @test occursin("rosenbrock", msg_of(err))

    err = load_err(minimal_cfg(; extra_cells=Dict("ids" => nothing,
                                                  "tiers" => ["750D"])))
    @test err isa ErrorException
    @test occursin("Unknown tier '750D'", msg_of(err))

    err = load_err(minimal_cfg(; extra_cells=Dict("ids" => nothing,
                                                  "noise" => [0.02])))
    @test err isa ErrorException
    @test occursin("cells.noise entry", msg_of(err))

    # One past the end, derived rather than literal: 37 was out of range until
    # :piston added cells 37-42, and the test silently stopped testing anything.
    err = load_err(minimal_cfg(; extra_cells=Dict("ids" => [length(_CELLS) + 1])))
    @test err isa ErrorException
    @test occursin("outside 1:$(length(_CELLS))", msg_of(err))
end

# ─────────────────────────────────────────────────────────────────────────────
# Cell selection resolves to the right id sets
# ─────────────────────────────────────────────────────────────────────────────

@testset "cell selection resolves to the right cell id sets" begin
    sel(cells_cfg) = resolve_cell_ids(Dict{String,Any}(cells_cfg), _CELLS)

    @test sel(Dict("all" => true)) == collect(1:42)
    @test length(_CELLS) == 42

    # functions
    ros = sel(Dict("functions" => ["rosenbrock"]))
    @test length(ros) == 6
    @test all(_CELLS[id].name == :rosenbrock for id in ros)

    # tiers: the sobol/grid tier labels are disjoint, which is what lets the
    # two ishigami shards split on `tiers` alone.
    ish_sobol = sel(Dict("functions" => ["ishigami"],
                         "tiers" => ["500D", "1000D", "2000D"]))
    ish_grid  = sel(Dict("functions" => ["ishigami"],
                         "tiers" => ["10^3", "20^3", "50^3"]))
    @test length(ish_sobol) == 6
    @test length(ish_grid) == 6
    @test isempty(intersect(ish_sobol, ish_grid))
    @test all(_CELLS[id].layout == :sobol for id in ish_sobol)
    @test all(_CELLS[id].layout == :grid  for id in ish_grid)
    @test ish_grid == [31, 32, 33, 34, 35, 36]

    # noise
    noisy = sel(Dict("noise" => [0.1]))
    @test length(noisy) == length(_CELLS) ÷ 2   # exactly one noisy cell per clean one
    @test all(_CELLS[id].noise_level == 0.1 for id in noisy)

    # conjunctive: functions AND tiers AND noise
    one = sel(Dict("functions" => ["hartmann6"], "tiers" => [1000],
                   "noise" => ["0%"]))
    @test length(one) == 1
    @test _CELLS[only(one)].name == :hartmann6
    @test _CELLS[only(one)].tier_label == "1000D"
    @test _CELLS[only(one)].noise_level == 0.0

    # ids
    @test sel(Dict("ids" => [30, 1, 18, 1])) == [1, 18, 30]

    # the diona shards partition every cell exactly once -- one shard per
    # Sobol' function plus one for the Ishigami grid block
    shards = [sel(Dict("functions" => ["ishigami"],
                       "tiers" => ["500D", "1000D", "2000D"])),
              sel(Dict("functions" => ["park91a"])),
              sel(Dict("functions" => ["friedman1"])),
              sel(Dict("functions" => ["hartmann6"])),
              sel(Dict("functions" => ["rosenbrock"])),
              sel(Dict("functions" => ["piston"])),
              sel(Dict("functions" => ["ishigami"],
                       "tiers" => ["10^3", "20^3", "50^3"]))]
    @test sort(vcat(shards...)) == collect(1:length(_CELLS))
    @test length(vcat(shards...)) == length(_CELLS)   # no overlap
end

@testset "exclusivity and empty-selection guards" begin
    err = load_err(minimal_cfg(; extra_cells=Dict("ids" => [1],
                                                  "functions" => ["ishigami"])))
    @test err isa ErrorException
    @test occursin("'ids' cannot be combined with", msg_of(err))

    err = load_err(minimal_cfg(; extra_cells=Dict("ids" => nothing,
                                                  "all" => true,
                                                  "noise" => [0.0])))
    @test err isa ErrorException
    @test occursin("'all' cannot be combined with", msg_of(err))

    err = load_err(minimal_cfg(; extra_cells=Dict("ids" => nothing, "all" => false)))
    @test err isa ErrorException
    @test occursin("cells.all must be true", msg_of(err))

    err = load_err(minimal_cfg(; extra_cells=Dict("ids" => nothing)))
    @test err isa ErrorException
    @test occursin("'cells' must name at least one of", msg_of(err))

    # hartmann6 has no grid tier -> conjunctive selection is empty
    err = load_err(minimal_cfg(; extra_cells=Dict("ids" => nothing,
                                                  "functions" => ["hartmann6"],
                                                  "tiers" => ["10^3"])))
    @test err isa ErrorException
    @test occursin("selection is empty", msg_of(err))
end

# ─────────────────────────────────────────────────────────────────────────────
# Arm applicability and :rbf's precision on the Ishigami 50^3 cells
# ─────────────────────────────────────────────────────────────────────────────

@testset ":rbf runs everywhere, uniformly at Float32" begin
    @test _CELLS[35].n_train == 125_000
    @test _CELLS[36].n_train == 125_000

    # No cell excludes any arm any more: :rbf is fitted at Float32 where Float64
    # does not fit (176.1 GiB vs 87.9 GiB, budget 96 GiB -- all measured).
    @test arm_applicable(:rbf, _CELLS[35], _CELLS)
    @test arm_applicable(:rbf, _CELLS[36], _CELLS)
    @test arm_applicable(:idw, _CELLS[35], _CELLS)
    @test arm_applicable(:rbf, _CELLS[1],  _CELLS)
    @test all(arm_applicable(:rbf, c, _CELLS) for c in _CELLS)

    # ONE precision, every cell. An earlier revision used Float64 where it fit
    # and Float32 only where it did not, which made the arm two configurations
    # under one name -- asymmetric WITHIN an arm. Pinned so that cannot return.
    @test all(_rbf_precision_for(c.n_train) === :f32 for c in _CELLS)
    # Past even the Float32 budget the arm becomes inapplicable again.
    @test _rbf_precision_for(10_000_000) === nothing

    # One shard covering all 36 with [rbf, idw] now drops nothing.
    _, r = load_data_config(write_cfg(minimal_cfg(;
        extra_top=Dict("arms" => ["rbf", "idw"]),
        extra_cells=Dict("ids" => nothing, "all" => true))))
    @test isempty(r.dropped_arm_cells)
    @test length(r.arm_cells) == 42 * 2
    @test (35, :rbf) in r.arm_cells
    @test (35, :idw) in r.arm_cells
end

# ─────────────────────────────────────────────────────────────────────────────
# run: defaults, coercions and the override dicts a config resolves to
# ─────────────────────────────────────────────────────────────────────────────

@testset "run defaults and coercions" begin
    _, r = load_data_config(write_cfg(minimal_cfg()))
    @test r.jobs == Dict(:diona => DEFAULT_JOBS_COMPARATIVE_AUTO)
    # Derived from the one harness table, not restated: the three per-arm
    # timeout tables drifted apart once and the drift test above now pins them.
    @test r.timeouts == Dict(:diona => DEFAULT_TIMEOUT_COMPARATIVE[:diona])
    @test r.timeouts[:diona] > 2.98 * 3600      # > the measured worst :diona cell
    @test r.blas_threads == 1
    @test r.resume === true
    @test r.run_id == "unit-test-out"
    @test r.out == abspath(joinpath(REPO_ROOT, "results", "unit-test-out"))
    @test r.L_H === :estimate
    @test r.L_H_factor == 2.5

    _, r2 = load_data_config(write_cfg(minimal_cfg(;
        extra_run=Dict("resume" => "off", "run_id" => "shard-a",
                       "blas_threads" => 2))))
    @test r2.resume === false
    @test r2.run_id == "shard-a"
    @test r2.blas_threads == 2

    for (key, value, needle) in (("jobs", 0, "run.jobs must be >= 1"),
                                 ("jobs", "8", "run.jobs must be a positive integer"),
                                 ("jobs", true, "run.jobs must be a positive integer"),
                                 ("timeout_s", 0, "run.timeout_s must be > 0"),
                                 ("timeout_s", -1.0, "run.timeout_s must be > 0"),
                                 ("blas_threads", 0, "run.blas_threads must be >= 1"),
                                 ("blas_threads", 1.5, "run.blas_threads must be a positive integer"),
                                 ("run_id", "", "run.run_id must be non-empty"))
        err = load_err(minimal_cfg(; extra_run=Dict(key => value)))
        @test err isa ErrorException
        @test occursin(needle, msg_of(err))
    end

    err = load_err(minimal_cfg(; extra_solver=Dict("L_H_factor" => 0)))
    @test err isa ErrorException
    @test occursin("solver.L_H_factor must be > 0", msg_of(err))
end

@testset "jobs/timeout_s accept a scalar or a per-arm map" begin
    _, scalar = load_data_config(write_cfg(minimal_cfg(;
        extra_top=Dict("arms" => ["diona", "nsm"]),
        extra_run=Dict("jobs" => 8, "timeout_s" => 900))))
    @test scalar.jobs == Dict(:diona => 8, :nsm => 8)
    @test scalar.timeouts == Dict(:diona => 900.0, :nsm => 900.0)

    _, mapped = load_data_config(write_cfg(minimal_cfg(;
        extra_top=Dict("arms" => ["diona", "nsm"]),
        extra_run=Dict("jobs" => Dict("diona" => 8, "nsm" => 2),
                       "timeout_s" => Dict("diona" => 14400)))))
    @test mapped.jobs == Dict(:diona => 8, :nsm => 2)
    # an arm the map omits falls back to its default, not to a missing key
    @test mapped.timeouts ==
          Dict(:diona => 14400.0, :nsm => DEFAULT_TIMEOUT_COMPARATIVE[:nsm])

    err = load_err(minimal_cfg(; extra_run=Dict("jobs" => Dict("bogus" => 4))))
    @test err isa ErrorException
    @test occursin("run.jobs map has unknown arm key 'bogus'", msg_of(err))

    # keys are restricted to the selected arms, matching jobs_override's shape
    @test Set(keys(mapped.jobs)) == Set(mapped.arms)
    @test Set(keys(mapped.timeouts)) == Set(mapped.arms)
end

@testset "a config round-trips to the orchestrator's override arguments" begin
    _, r = load_data_config(write_cfg(minimal_cfg(;
        extra_top=Dict("arms" => ["nn", "diona"]),
        extra_cells=Dict("ids" => [1, 18, 30]),
        extra_run=Dict("out" => "results/shared-out", "jobs" => 4,
                       "timeout_s" => 1800))))

    # These four are exactly the keyword arguments run_comparative_orchestrator
    # already accepts; the runner forwards them and adds nothing.
    overrides = (selected_ids_override = r.selected_ids,
                 out_dir_override      = r.out,
                 arms_override         = r.arms,
                 jobs_override         = r.jobs,
                 timeout_override      = r.timeouts)

    @test overrides.selected_ids_override == [1, 18, 30]
    @test overrides.out_dir_override == abspath(joinpath(REPO_ROOT, "results", "shared-out"))
    @test overrides.arms_override == [:diona, :nn]           # canonical order
    @test overrides.jobs_override isa Dict{Symbol,Int}
    @test overrides.timeout_override isa Dict{Symbol,Float64}
    @test overrides.jobs_override == Dict(:diona => 4, :nn => 4)
    @test overrides.timeout_override == Dict(:diona => 1800.0, :nn => 1800.0)

    # run_comparative_orchestrator's own signature must still take them.
    kwnames = reduce(vcat, Base.kwarg_decl.(collect(methods(run_comparative_orchestrator))))
    for kw in (:selected_ids_override, :out_dir_override, :arms_override,
               :jobs_override, :timeout_override, :device_override)
        @test kw in kwnames
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# The checked-in config corpus
# ─────────────────────────────────────────────────────────────────────────────

# Filename fragment => the single setting that config is allowed to vary.
# Empty today; kept so a deliberate ablation config has somewhere to declare
# itself instead of quietly breaking the uniformity test below.
const _VARIANT_EXEMPT = Dict{String,Vector{String}}()
_exempt(base::String) =
    reduce(vcat, [v for (k, v) in _VARIANT_EXEMPT if occursin(k, base)]; init=String[])

const _UNIFORM_SOLVER = ["L_H", "L_H_factor", "gamma_factor"]

# Every shard directory that must satisfy the coverage/solver invariants. There
# is one: test/configs is the manuscript's campaign (5000 x dim query points,
# 20 dB Gaussian noise, L_H_factor 2.5). A gap or overlap fails here rather
# than at the end of a 13-hour campaign.
const _CONFIG_DIRS = [joinpath(@__DIR__, "configs")]

@testset "every checked-in config loads and pins the canonical solver settings" begin
  @test !isempty(_CONFIG_DIRS)
  for configs_dir in _CONFIG_DIRS
    files = sort(filter(f -> endswith(f, ".yaml"), readdir(configs_dir; join=true)))
    @test !isempty(files)

    outs = Set{String}()
    for f in files
        cfg, resolved = load_data_config(f)
        @test resolved.L_H === :estimate
        # The factor is a per-CAMPAIGN knob, not a global constant. What must
        # hold is that it is positive and UNIFORM within a directory -- the
        # next testset checks uniformity, and that is the property the
        # comparison depends on.
        @test resolved.L_H_factor > 0
        @test resolved.gamma_factor > 0
        @test !isempty(resolved.arm_cells)
        push!(outs, resolved.out)
    end

    # Every shard writes into ONE shared out dir. That is the schema's whole
    # point: fragments are keyed by (cell_id, arm, fixture_sha256,
    # settings_hash) and written atomically, so disjoint shards compose.
    @test length(outs) == 1
  end
end

@testset "the checked-in shards share one solver configuration" begin
  for configs_dir in _CONFIG_DIRS
    files = sort(filter(f -> endswith(f, ".yaml"), readdir(configs_dir; join=true)))

    seen = Dict{String,Dict{Any,Vector{String}}}()
    for f in files
        base = basename(f)
        _, resolved = load_data_config(f)
        ex = _exempt(base)
        for k in _UNIFORM_SOLVER
            k in ex && continue
            value = k == "L_H" ? resolved.L_H :
                    k == "L_H_factor" ? resolved.L_H_factor : resolved.gamma_factor
            bucket = get!(seen, "solver." * k, Dict{Any,Vector{String}}())
            push!(get!(bucket, value, String[]), base)
        end
    end

    for (setting, buckets) in sort(collect(seen); by=first)
        if length(buckets) != 1
            offenders = join([string(v, " => ", join(sort(fs)[1:min(3, end)], ", "))
                              for (v, fs) in buckets], " | ")
            @info "NON-UNIFORM SETTING" setting offenders
        end
        @test length(buckets) == 1
    end
  end
end

@testset "the checked-in diona shards partition every cell exactly once" begin
  for configs_dir in _CONFIG_DIRS
    diona_files = sort(filter(f -> endswith(f, ".yaml") &&
                                   startswith(basename(f), "diona-"),
                              readdir(configs_dir; join=true)))
    # One shard per Sobol' function this campaign covers (six, :piston
    # included), plus one for the Ishigami grid block.
    @test !isempty(diona_files)

    all_pairs = Tuple{Int,Symbol}[]
    for f in diona_files
        _, r = load_data_config(f)
        @test r.arms == [:diona]
        append!(all_pairs, r.arm_cells)
    end
    # Against what THIS directory's report claims, not against every cell that
    # exists, so a deliberately partial campaign is not failed for being honest.
    claimed = sort(unique(first.(last(load_data_config(
        joinpath(configs_dir, "report-all.yaml"))).arm_cells)))
    @test length(all_pairs) == length(claimed)   # disjoint: no (cell, arm) twice
    @test length(unique(all_pairs)) == length(claimed)
    @test sort(first.(all_pairs)) == claimed
  end
end

@testset "the non-diona shards cover their arms without colliding" begin
  for configs_dir in _CONFIG_DIRS
    load(name) = last(load_data_config(joinpath(configs_dir, name)))
    # A campaign directory need not carry every arm: a partial re-run of one
    # arm reuses the other arms' fragments, so a missing shard here is a
    # deliberate partial re-run, not a gap. What must still hold is that
    # whatever IS present covers its arm.
    has(name) = isfile(joinpath(configs_dir, name))
    # What THIS campaign claims, not the whole cell table.
    n_claimed = length(unique(first.(last(load_data_config(
        joinpath(configs_dir, "report-all.yaml"))).arm_cells)))

    if has("nsm-all.yaml")
    nsm = load("nsm-all.yaml")
    @test nsm.arms == [:nsm]
    @test length(nsm.arm_cells) == n_claimed
    end

    # :rbf and :idw are SEPARATE shards. blas_threads is one scalar per shard and
    # the two arms need opposite settings: :rbf at jobs 1 wants the whole machine
    # for its dense 125k LU, :idw at jobs 8 must not have 8 x 32 threads.
    if has("rbf-all.yaml")
    rbf = load("rbf-all.yaml")
    @test rbf.arms == [:rbf]
    # Nothing is dropped: :rbf reaches the 50^3 cells at Float32.
    @test isempty(rbf.dropped_arm_cells)
    @test length(rbf.arm_cells) == n_claimed
    @test rbf.jobs == Dict(:rbf => 1)
    @test rbf.blas_threads == 32
    end

    if has("idw-all.yaml")
    idw = load("idw-all.yaml")
    @test idw.arms == [:idw]
    @test length(idw.arm_cells) == n_claimed
    @test idw.jobs == Dict(:idw => 8)
    end

    # THE THREADING RULE, asserted as a rule rather than as magic numbers:
    # jobs x blas_threads = 32 (the core count) for every concurrently-running
    # set, so each arm gets the same machine budget. The six diona shards run
    # THREE at a time, so their product is per wave.
    CORES = 32
    DIONA_CONCURRENT = 3
    for f in sort(readdir(configs_dir))
        (endswith(f, ".yaml") && !occursin("report-all", f)) || continue
        r = load(f)
        jobs = sum(values(r.jobs))
        product = jobs * r.blas_threads
        if startswith(f, "diona-")
            @test product * DIONA_CONCURRENT <= CORES
            @test product * DIONA_CONCURRENT >= CORES - DIONA_CONCURRENT
        else
            @test product == CORES
        end
    end

    # And the split exists for a reason: :rbf concentrates the budget in one
    # worker for its dense LU, :idw spreads it over eight. If these ever become
    # equal, the split has silently collapsed.
    if has("rbf-all.yaml") && has("idw-all.yaml")
    @test rbf.blas_threads != idw.blas_threads
    @test rbf.blas_threads * sum(values(rbf.jobs)) ==
          idw.blas_threads * sum(values(idw.jobs))
    end

    if has("gp-all.yaml")
    gp = load("gp-all.yaml")
    @test gp.arms == [:gp]
    @test length(gp.arm_cells) == n_claimed

    nn = load("nn-all.yaml")
    @test nn.arms == [:nn]
    @test length(nn.arm_cells) == n_claimed

    rep = load("report-all.yaml")
    @test rep.arms == [:diona, :nn, :gp, :nsm, :rbf, :idw]
    @test length(rep.selected_ids) == n_claimed

    # No two shards claim the same (cell, arm).
    every = Tuple{Int,Symbol}[]
    for f in sort(filter(f -> endswith(f, ".yaml") && basename(f) != "report-all.yaml",
                         readdir(configs_dir; join=true)))
        append!(every, last(load_data_config(f)).arm_cells)
    end
    @test length(every) == length(unique(every))
    end
  end
end

@testset "the shards cover exactly what report-all.yaml claims" begin
    # Disjointness alone is not enough, and its absence is not a cheap bug:
    # `report_from_config` hard-errors on any (cell, arm) it claims and cannot
    # read, so a claim no shard produces makes the campaign's final
    # aggregation step impossible to complete. Prior fragments cannot fill the
    # gap either -- COMPARATIVE_FRAGMENT_VERSION and comparative_code_provenance
    # both moved, which retires every fragment of the previous campaign.
  for configs_dir in _CONFIG_DIRS
    report  = last(load_data_config(joinpath(configs_dir, "report-all.yaml")))
    claimed = Set(report.arm_cells)

    produced = Tuple{Int,Symbol}[]
    for f in sort(filter(f -> endswith(f, ".yaml") && basename(f) != "report-all.yaml",
                         readdir(configs_dir; join=true)))
        append!(produced, last(load_data_config(f)).arm_cells)
    end

    missing_pairs = sort(collect(setdiff(claimed, Set(produced))))
    extra_pairs   = sort(collect(setdiff(Set(produced), claimed)))
    isempty(missing_pairs) ||
        @info "CLAIMED BY report-all.yaml BUT PRODUCED BY NO SHARD" missing_pairs
    isempty(extra_pairs) ||
        @info "PRODUCED BY A SHARD BUT NOT CLAIMED BY report-all.yaml" extra_pairs
    @test isempty(missing_pairs)
    @test isempty(extra_pairs)

    # Every cell x however many arms this campaign reports, with NO exclusion
    # within those arms -- :rbf fits the two Ishigami 50^3 cells at Float32
    # (87.9 GiB) where Float64 would need 176.1 GiB. The arm count is read from
    # the config rather than hardcoded, because a campaign may legitimately
    # re-run a subset of arms and reuse the others' fragments.
    @test length(claimed) == length(report.selected_ids) * length(report.arms)
    @test length(produced) == length(claimed)
  end
end

@testset "run.device selects the :nn execution device" begin
    # `run.device` used to be a REFUSED key ("data-only comparisons are
    # CPU-only"). It is now a real setting: the :nn arm runs on cpu or cuda,
    # every other arm is CPU-only and the orchestrator pins it to "cpu"
    # regardless, and the device is folded into the :nn settings hash so the
    # two devices' fragments can never be confused for one another.
    _, default_resolved = load_data_config(write_cfg(minimal_cfg()))
    @test default_resolved.device == "cpu"

    for (given, expected) in (("cpu", "cpu"), ("cuda", "cuda"),
                              ("CUDA", "cuda"), ("Cpu", "cpu"))
        _, r = load_data_config(write_cfg(minimal_cfg(; extra_run=Dict("device" => given))))
        @test r.device == expected
    end

    for bogus in ("gpu", "rocm", "")
        err = load_err(minimal_cfg(; extra_run=Dict("device" => bogus)))
        @test occursin("unsupported device", msg_of(err))
    end

    # The device changes the :nn settings hash and NOTHING else's: five arms
    # that have no device must not have their fragments invalidated by it.
    reg = ComparatorRegistry.canonical_registry()
    cell = enumerate_cells()[1]
    @test _comparative_settings_hash(cell, :nn, reg; device="cpu") !=
          _comparative_settings_hash(cell, :nn, reg; device="cuda")
    for arm in (:diona, :gp, :nsm, :rbf, :idw)
        @test _comparative_settings_hash(cell, arm, reg; device="cpu") ==
              _comparative_settings_hash(cell, arm, reg; device="cuda")
    end

    # Registry-driven precision, and it is inside the hash that the arm and the
    # campaign both recompute.
    nn = ComparatorRegistry.settings_for(reg, cell.dim, :nn)
    @test nn.precision == "Float32"
    @test ComparatorRegistry.float_type(nn) === Float32
    @test _comparative_settings_hash(cell, :nn, reg; device="cpu") ==
          ComparatorRegistry.nn_settings_hash(nn, "cpu")
end

@testset "the three timeout tables do not drift apart" begin
    # run_full_benchmark.jl's driver always passes its OWN table into the
    # orchestrator, so whichever value an operator reads in smoke_data.jl or
    # here is not necessarily the one that applies. The three tables disagreed
    # on :nsm/:rbf/:idw by 2x. Pin them together instead of documenting the
    # difference.
    if !@isdefined(FullBenchmarkDriver)
        include(joinpath(@__DIR__, "run_full_benchmark.jl"))
    end
    driver_timeouts = FullBenchmarkDriver.DEFAULT_ARM_TIMEOUTS_S
    for arm in FullBenchmarkDriver.BENCHMARK_ARMS
        @test driver_timeouts[arm] == DEFAULT_TIMEOUT_COMPARATIVE[arm]
        @test _DATA_DEFAULT_TIMEOUT_S[arm] == DEFAULT_TIMEOUT_COMPARATIVE[arm]
    end
    @test Set(FullBenchmarkDriver.BENCHMARK_ARMS) == Set(_DATA_ARM_ORDER)

    # The driver mirrors the device list because it parses its arguments before
    # it has bootstrapped any environment. Same reason as the timeout tables:
    # pin it rather than document the possible drift.
    @test Set(FullBenchmarkDriver.SUPPORTED_DEVICES) ==
          Set(ComparatorRegistry.SUPPORTED_DEVICES)
    @test FullBenchmarkDriver.parse_options(["--device", "CUDA"]).device == "cuda"
    @test FullBenchmarkDriver.parse_options(String[]).device == "cpu"
    @test_throws ErrorException FullBenchmarkDriver.parse_options(["--device", "gpu"])

    # Regression: propertynames advertised the CLI flag spellings
    # (`jobs-diona`), which getproperty -- matching `jobs_`/`timeout_` -- could
    # not resolve, so iterating propertynames and reading them threw.
    opts = FullBenchmarkDriver.DriverOptions(
        "/tmp/driver-options-probe",
        Dict(arm => 1 for arm in FullBenchmarkDriver.BENCHMARK_ARMS),
        copy(driver_timeouts), 1, "cpu")
    names = propertynames(opts)
    @test length(names) == length(fieldnames(FullBenchmarkDriver.DriverOptions)) +
                           2 * length(FullBenchmarkDriver.BENCHMARK_ARMS)
    @test isempty([n for n in names if (try (getproperty(opts, n); false) catch; true end)])
    @test opts.timeout_nsm == driver_timeouts[:nsm]
end
