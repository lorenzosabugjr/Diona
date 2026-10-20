#!/usr/bin/env julia
#
# benchmark_data.jl -- config-driven driver for the 36-cell data-only benchmark.
#
#   julia --project=test/ml test/benchmark_data.jl test/configs/<shard>.yaml
#   julia --project=test/ml test/benchmark_data.jl --report test/configs/report.yaml
#
# PROTOCOL (do not weaken):
#   Every arm sees the training dataset and nothing else. This config surface
#   carries no noise bound, no Lipschitz constant, no held-out truth and no
#   oracle of any kind -- each arm estimates its own constants from the
#   fixture's noisy training data. The only outside input is the domain box
#   bf.lb/bf.ub, applied identically by every arm for input normalisation.
#   `solver.noise` is therefore a NAMED, REFUSED key, not an omitted one.
#
#   Interval semantics differ per arm and none of them is a guarantee:
#   :diona = admissible-set bounds under ESTIMATED constants,
#   :nsm   = set-membership under an ESTIMATED gamma,
#   :gp    = variational / approximate,
#   :nn / :rbf / :idw = no interval at all.
#
# THIN BY CONSTRUCTION:
#   This file validates YAML and forwards the result into the override surface
#   that `run_comparative_orchestrator` already exposes (`selected_ids_override`,
#   `out_dir_override`, `arms_override`, `jobs_override`, `timeout_override`).
#   It contains no campaign logic, no fitting, no metric computation. The
#   `--report` pass likewise only re-reads fragments that the campaign already
#   wrote and hands them to the existing artifact writers.
#
# SHARDING:
#   Fragments are keyed by (cell_id, arm, fixture_sha256, settings_hash) and
#   written through `atomic_write`, so several processes may run different
#   configs into ONE `run.out` as long as their (cell, arm) sets are disjoint.
#   That is the whole point of the schema: `run.out` is explicit and shared,
#   and `cells`/`arms` carve the campaign into shards. The uniform-solver test
#   in tests_benchmark_data_config.jl is what stops two shards sharing an
#   out_dir under different `solver:` blocks.
#
#   Each shard's CAMPAIGN-level files are written under a shard-specific prefix
#   (`comparative_shard-<arms>_summary.csv` / `_outcomes.csv`) so that shards
#   cannot overwrite one another's; the canonical `comparative_summary.csv` is
#   written only by the `--report` pass.
#
# RUNBOOK -- warm the fixtures before launching shards concurrently:
#
#   julia --project=test/ml test/benchmark_data.jl --prepare-fixtures test/configs/report-all.yaml
#   # ... then launch the shards in parallel, all into the same run.out ...
#   julia --project=test/ml test/benchmark_data.jl --report test/configs/report-all.yaml
#
#   `fixture_sha256` folds in a wall-clock `generated_at`, so two processes that
#   both find a cell's fixture absent will each write one and key their
#   fragments to DIFFERENT stems. See `prepare_fixtures_from_config` for the
#   full account. On a warm out_dir the window does not exist.

using Dates
using YAML

if !@isdefined(run_comparative_orchestrator)
    include(joinpath(@__DIR__, "smoke_data.jl"))
end

# ============================================================================
# Config schema
# ============================================================================

const _VALID_TOP_KEYS    = Set(["name", "arms", "cells", "run", "solver"])
const _VALID_CELLS_KEYS  = Set(["functions", "tiers", "noise", "ids", "all"])
const _VALID_RUN_KEYS    = Set(["out", "jobs", "timeout_s", "blas_threads",
                                "resume", "run_id", "device", "query_multiplier"])
const _VALID_SOLVER_KEYS = Set(["L_H", "L_H_factor", "gamma_factor"])

# Canonical arm order for this benchmark. `_COMPARATIVE_ARM_ORDER` in
# smoke_data.jl is the orchestrator's own order; the two must agree on the
# arms they share, which `_check_arm_registry_agreement` asserts at load time
# rather than letting them drift silently.
const _DATA_ARM_ORDER = (:diona, :nn, :gp, :nsm, :rbf, :idw)

# Config-layer job/timeout fallbacks. DERIVED from the harness table rather
# than restated: three independent tables (here, DEFAULT_TIMEOUT_COMPARATIVE in
# smoke_data.jl, DEFAULT_ARM_TIMEOUTS_S in run_full_benchmark.jl) drifted apart
# once already, and since the driver always passes its own dict the one that
# wins at campaign time is not the one an operator reads here. Every shipped
# config sets `run.timeout_s` explicitly anyway; these are only the fallbacks.
# The :nsm/:rbf/:idw entries remain UNPROBED allowances -- see the comment on
# DEFAULT_TIMEOUT_COMPARATIVE.
const _DATA_DEFAULT_TIMEOUT_S =
    Dict{Symbol,Float64}(arm => Float64(DEFAULT_TIMEOUT_COMPARATIVE[arm])
                         for arm in keys(DEFAULT_TIMEOUT_COMPARATIVE))

# :rbf assembles a dense N x N kernel system and factorises it. The threshold is
# DERIVED from the same pre-flight the arm itself refuses with (mirrored in
# smoke_data.jl as `_rbf_precision_for`), not restated as a round number: a
# hand-written 100_000 once disagreed with the real rule for every cell with
# 32,768 < n_train <= 100,000 -- a band this benchmark happens not to populate,
# which is exactly how such a divergence survives.
#
# The rule is now TWO-TIER. :rbf fits at Float64 where that fits the budget and
# at Float32 where it does not; only when neither fits is the arm inapplicable.
# Measured 2026-09-21: 12.1 bytes/n^2 at Float64 and 6.04 at Float32, so at the
# Ishigami 50^3 tier (n = 125,000) the system is 176.1 GiB at Float64 but
# 87.9 GiB at Float32 -- inside the 96 GiB budget. Every cell in this benchmark
# is therefore applicable; the largest admissible n is now set by Float32.
#
# Applicability still matters even though nothing is currently excluded: an
# inapplicable arm must never be allowed to enter and fail, because
# `common_success_query_ids` intersects across the arms expected on a cell and
# one all-NaN arm would zero the common query set for every other arm there.
const _RBF_MAX_N_TRAIN =
    Int(floor(sqrt(_RBF_MAX_FIT_BYTES / _rbf_dense_system_bytes(1, :f32))))

(_rbf_precision_for(_RBF_MAX_N_TRAIN) !== nothing &&
 _rbf_precision_for(_RBF_MAX_N_TRAIN + 1) === nothing) ||
    error("benchmark_data.jl: _RBF_MAX_N_TRAIN=$(_RBF_MAX_N_TRAIN) is not the largest " *
          "n_train that smoke_data.jl's :rbf memory pre-flight admits; the two rules " *
          "would exclude different cells.")

const _SOBOL_TIER_LABELS = ("500D", "1000D", "2000D")
const _GRID_TIER_LABELS  = ("10³", "20³", "50³")

function _check_keys(dict, valid::Set{String}, section::String)
    for k in keys(dict)
        string(k) in valid ||
            error("Unknown key '$(k)' in '$(section)'. Accepted: " *
                  join(sort(collect(valid)), ", "))
    end
end

# Named errors for sections/keys that the PDE-era config carried and the
# data-only schema deliberately does not. Checked BEFORE `_check_keys`, so a
# stale config gets a message naming its replacement rather than a bare
# "unknown key".
function _check_removed_top_keys(cfg)
    haskey(cfg, "test_data") &&
        error("Removed section 'test_data'. The data-only benchmark takes its data " *
              "from enumerate_cells() and the generated fixtures, never from a " *
              "truthfile/inputfile. Select cells with the 'cells' section instead.")
    haskey(cfg, "pde") &&
        error("Removed section 'pde'. All PDE machinery was deleted at 365df87; " *
              "this benchmark is data-only.")
    haskey(cfg, "diagnostics") &&
        error("Removed section 'diagnostics'. Taylor-stage diagnostics were " *
              "PDE-path only and no longer exist.")
    haskey(cfg, "output") &&
        error("Removed section 'output'. The comparative orchestrator always writes " *
              "fragments under run.out and always echoes to the console.")
    haskey(cfg, "pipeline") &&
        error("Renamed section 'pipeline'. Cell selection moved to 'cells'; " *
              "execution settings moved to 'run'. There are no modes: every arm-cell " *
              "computes nominal and interval together.")
    return nothing
end

function _check_removed_cells_keys(cells_cfg)
    for k in ("mode", "uq")
        haskey(cells_cfg, k) &&
            error("Removed key 'cells.$(k)'. The data-only path has no modes; every " *
                  "arm-cell computes nominal and interval together.")
    end
    return nothing
end

# Deliberately empty, and kept: `run` has no refused keys today. `run.device`
# used to be one ("data-only comparisons are CPU-only"); it is now a real
# setting, validated in resolve_data_config. This hook is where the next
# removed key announces itself, next to the cells/solver ones.
function _check_removed_run_keys(run_cfg)
    return nothing
end

function _check_removed_solver_keys(solver_cfg)
    haskey(solver_cfg, "noise") &&
        error("Removed key 'solver.noise'. PROTOCOL: the noise bound is ESTIMATED " *
              "from the fixture's noisy training data. A config-supplied bound " *
              "would be an oracle.")
    for k in ("neighbor_method", "neighbor_param", "neighbor_fit_layers", "mu0", "layers")
        haskey(solver_cfg, k) &&
            error("Removed key 'solver.$(k)'. The :diona arm pins Algorithm 1's " *
                  "settings (mu0, three passes) in the harness (_SMOKE_DIONA_MU0, " *
                  "_SMOKE_DIONA_LAYERS); setting them per config would rotate the " *
                  ":diona settings_hash with no campaign behind it.")
    end
    haskey(solver_cfg, "n_dim") &&
        error("Removed key 'solver.n_dim'. Dimension comes from the cell (SmokeCell.dim).")
    for k in ("temporal_edges", "time_weight", "depth", "compute_uq")
        haskey(solver_cfg, k) &&
            error("Removed key 'solver.$(k)'. Diona has one algorithm (manuscript " *
                  "Algorithm 2): a star of Taylor relations between the query and " *
                  "its neighbours, two bound LPs, no graph depth and no nominal-only mode.")
    end
    return nothing
end

# ============================================================================
# Scalar -> list normalisation and per-element coercion
# ============================================================================

_as_list(v) = isa(v, AbstractVector) ? collect(v) : Any[v]

function _normalize_bool(raw, key::String)::Bool
    raw isa Bool && return raw
    s = lowercase(string(raw))
    s in ("on", "true", "yes", "1") && return true
    s in ("off", "false", "no", "0") && return false
    error("$key must be a boolean (true/false), got '$raw'")
end

function _positive_int(raw, key::String)::Int
    raw isa Integer && !(raw isa Bool) ||
        error("$key must be a positive integer, got $raw")
    v = Int(raw)
    v >= 1 || error("$key must be >= 1, got $v")
    return v
end

function _normalize_arm(x)::Symbol
    arm = Symbol(lowercase(string(x)))
    arm in _DATA_ARM_ORDER ||
        error("Unknown arm '$(x)'. Accepted: " *
              join(sort(string.(collect(_DATA_ARM_ORDER))), ", "))
    return arm
end

function _normalize_function(x)::Symbol
    name = Symbol(lowercase(string(x)))
    known = Tuple(bf.name for bf in BENCHMARK_FUNCTIONS)
    name in known ||
        error("Unknown function '$(x)' in cells.functions. Accepted: " *
              join(sort(string.(collect(known))), ", "))
    return name
end

"""
    _normalize_tier(x) -> String

Normalise a tier spec onto a `SmokeCell.tier_label`. `tier_label` embeds U+00B3
for the grid tiers, so ASCII aliases are accepted: nobody should have to type a
superscript into a shard config. The integer forms are unambiguous because the
Sobol multipliers `(500, 1000, 2000)` and the grid sides `(10, 20, 50)` are
disjoint.
"""
function _normalize_tier(x)::String
    s = lowercase(strip(string(x)))
    if x isa Integer && !(x isa Bool)
        Int(x) in N_TIERS_PER_DIM && return "$(Int(x))D"
        Int(x) in ISHIGAMI_GRID_SIDES && return "$(Int(x))³"
    end
    for mult in N_TIERS_PER_DIM
        s in (lowercase("$(mult)d"), string(mult)) && return "$(mult)D"
    end
    for k in ISHIGAMI_GRID_SIDES
        s in ("$(k)³", "$(k)^3", "$(k)cubed", "$(k)x$(k)x$(k)") && return "$(k)³"
    end
    error("Unknown tier '$(x)' in cells.tiers. Accepted: " *
          join(_SOBOL_TIER_LABELS, ", ") * ", " * join(_GRID_TIER_LABELS, ", ") *
          " (ASCII forms 500/1000/2000 and 10/20/50, 10^3, 10cubed also accepted)")
end

_noise_error(raw) =
    error("cells.noise entry '$(raw)' must be one of: " *
          join(NOISE_LEVELS, ", ") * " -- as an amplitude ratio (0, 0.1), " *
          "as an SNR (\"clean\", \"20db\"), or as 0 / 20.")

function _normalize_noise(x)::Float64
    v = if x isa Real && !(x isa Bool)
        f = Float64(x)
        # A bare number >= 1 is read as an SNR in dB (20 -> 0.1); below 1 it is
        # already an amplitude ratio. 0 is the noiseless condition either way.
        f >= 1.0 ? snr_db_to_level(f) : f
    else
        s = lowercase(strip(string(x)))
        s in ("clean", "noiseless", "none") && return 0.0
        if endswith(s, "db")
            parsed = tryparse(Float64, strip(s[1:prevind(s, lastindex(s), 2)]))
            parsed === nothing && _noise_error(x)
            snr_db_to_level(parsed)
        else
            pct = endswith(s, "%")
            parsed = tryparse(Float64, pct ? s[1:prevind(s, lastindex(s))] : s)
            parsed === nothing && _noise_error(x)
            pct ? parsed / 100 : (parsed >= 1.0 ? snr_db_to_level(parsed) : parsed)
        end
    end
    idx = findfirst(nl -> isapprox(v, nl; atol=1e-12), NOISE_LEVELS)
    idx === nothing && _noise_error(x)
    return Float64(NOISE_LEVELS[idx])
end

# ============================================================================
# Arm applicability
# ============================================================================

"""
    _smoke_knows_arm(arm, cells) -> Bool

True when `applicable_arms` in smoke_data.jl already enumerates `arm` on at
least one cell. When it does, smoke_data.jl is the authority on where that arm
may run and this file defers to it; when it does not (the arm has not been
registered there yet) the local rule below applies. This keeps the two from
drifting in either direction without this file having to edit smoke_data.jl.
"""
_smoke_knows_arm(arm::Symbol, cells::Vector{SmokeCell}) =
    any(arm in applicable_arms(c) for c in cells)

function arm_applicable(arm::Symbol, cell::SmokeCell, cells::Vector{SmokeCell})::Bool
    _smoke_knows_arm(arm, cells) && return arm in applicable_arms(cell)
    arm === :rbf && cell.n_train > _RBF_MAX_N_TRAIN && return false
    return true
end

function _check_arm_registry_agreement()
    isdefined(@__MODULE__, :_COMPARATIVE_ARM_ORDER) || return nothing
    for arm in _COMPARATIVE_ARM_ORDER
        arm in _DATA_ARM_ORDER ||
            error("smoke_data.jl's _COMPARATIVE_ARM_ORDER carries arm ':$(arm)' that " *
                  "benchmark_data.jl's _DATA_ARM_ORDER does not know. Add it to " *
                  "_DATA_ARM_ORDER (and to comparative_interval_kind) rather than " *
                  "letting the two registries drift.")
    end
    return nothing
end

# ============================================================================
# Cell selection
# ============================================================================

"""
    resolve_cell_ids(cells_cfg, cells) -> Vector{Int}

`functions` / `tiers` / `noise` combine CONJUNCTIVELY: an omitted axis places no
restriction on that axis. That is what lets per-function and per-noise shards
compose without a combinatorial key explosion. `ids` and `all` are each
exclusive with every other key and with each other.
"""
function resolve_cell_ids(cells_cfg, cells::Vector{SmokeCell})::Vector{Int}
    _check_removed_cells_keys(cells_cfg)
    _check_keys(cells_cfg, _VALID_CELLS_KEYS, "cells")

    present = [k for k in ("functions", "tiers", "noise", "ids", "all")
               if haskey(cells_cfg, k)]
    isempty(present) &&
        error("'cells' must name at least one of: " *
              join(sort(collect(_VALID_CELLS_KEYS)), ", "))

    if haskey(cells_cfg, "ids")
        length(present) == 1 ||
            error("cells: 'ids' cannot be combined with functions/tiers/noise/all")
        raw = _as_list(cells_cfg["ids"])
        isempty(raw) && error("cells.ids must be non-empty")
        ids = Int[]
        for v in raw
            n = if v isa Integer && !(v isa Bool)
                Int(v)
            else
                parsed = tryparse(Int, strip(string(v)))
                parsed === nothing &&
                    error("cells.ids entry '$(v)' is not an integer")
                parsed
            end
            1 <= n <= length(cells) ||
                error("cells.ids entry $(n) is outside 1:$(length(cells))")
            push!(ids, n)
        end
        return sort(unique(ids))
    end

    if haskey(cells_cfg, "all")
        length(present) == 1 ||
            error("cells: 'all' cannot be combined with functions/tiers/noise/ids")
        _normalize_bool(cells_cfg["all"], "cells.all") ||
            error("cells.all must be true (there is no empty selection); omit the " *
                  "section key instead")
        return collect(eachindex(cells))
    end

    fns   = haskey(cells_cfg, "functions") ?
            sort(unique(_normalize_function.(_as_list(cells_cfg["functions"])))) : nothing
    tiers = haskey(cells_cfg, "tiers") ?
            unique(_normalize_tier.(_as_list(cells_cfg["tiers"]))) : nothing
    nls   = haskey(cells_cfg, "noise") ?
            sort(unique(_normalize_noise.(_as_list(cells_cfg["noise"])))) : nothing
    fns   === nothing || isempty(fns)   && error("cells.functions must be non-empty")
    tiers === nothing || isempty(tiers) && error("cells.tiers must be non-empty")
    nls   === nothing || isempty(nls)   && error("cells.noise must be non-empty")

    ids = [id for id in eachindex(cells) if
           (fns   === nothing || cells[id].name in fns) &&
           (tiers === nothing || cells[id].tier_label in tiers) &&
           (nls   === nothing || any(isapprox(cells[id].noise_level, nl; atol=1e-12)
                                     for nl in nls))]
    isempty(ids) && error("cells: selection is empty (functions=" *
        (fns === nothing ? "*" : join(fns, ",")) * ", tiers=" *
        (tiers === nothing ? "*" : join(tiers, ",")) * ", noise=" *
        (nls === nothing ? "*" : join(nls, ",")) *
        ") -- no cell of the $(length(cells)) matches")
    return ids
end

# ============================================================================
# Load and resolve
# ============================================================================

"""
    load_data_config(path) -> (cfg, resolved)

`cfg` is the raw parsed dict; `resolved` is a NamedTuple in which every value is
defaulted, coerced and range-checked exactly once. Downstream code reads
`resolved` only. Validation only -- this never runs a campaign.
"""
function load_data_config(path::AbstractString)
    isfile(path) || error("Config file not found: $path")
    cfg = YAML.load_file(String(path))
    cfg isa AbstractDict || error("Config must be a YAML mapping: $path")
    return cfg, resolve_data_config(cfg)
end

function resolve_data_config(cfg)
    _check_arm_registry_agreement()
    _check_removed_top_keys(cfg)
    _check_keys(cfg, _VALID_TOP_KEYS, "(top-level)")

    for section in ("name", "arms", "cells", "run")
        haskey(cfg, section) || error("Config missing '$(section)' section")
    end

    name = String(strip(string(cfg["name"])))
    isempty(name) && error("'name' must be non-empty")

    cells = enumerate_cells()

    raw_arms = _as_list(cfg["arms"])
    isempty(raw_arms) && error("'arms' must be non-empty")
    arms_set = unique(_normalize_arm.(raw_arms))
    arms = [a for a in _DATA_ARM_ORDER if a in arms_set]

    selected_ids = resolve_cell_ids(cfg["cells"], cells)

    arm_cells = Tuple{Int,Symbol}[]
    dropped   = Tuple{Int,Symbol}[]
    for id in selected_ids, arm in arms
        push!(arm_applicable(arm, cells[id], cells) ? arm_cells : dropped, (id, arm))
    end
    isempty(arm_cells) &&
        error("selection is empty: none of arms [" * join(string.(arms), ", ") *
              "] is applicable to any of the $(length(selected_ids)) selected cells")

    run_cfg = cfg["run"]
    _check_removed_run_keys(run_cfg)
    _check_keys(run_cfg, _VALID_RUN_KEYS, "run")
    haskey(run_cfg, "out") || error("run: missing required key 'out'")
    out_raw = String(strip(string(run_cfg["out"])))
    isempty(out_raw) && error("run.out must be non-empty")
    out = abspath(isabspath(expanduser(out_raw)) ? expanduser(out_raw) :
                  joinpath(REPO_ROOT, expanduser(out_raw)))

    run_id = haskey(run_cfg, "run_id") ?
             String(strip(string(run_cfg["run_id"]))) : basename(out)
    isempty(run_id) && error("run.run_id must be non-empty")

    jobs     = _per_arm_map(get(run_cfg, "jobs", nothing), arms, "run.jobs",
                            arm -> DEFAULT_JOBS_COMPARATIVE_AUTO, _coerce_jobs)
    timeouts = _per_arm_map(get(run_cfg, "timeout_s", nothing), arms, "run.timeout_s",
                            arm -> _DATA_DEFAULT_TIMEOUT_S[arm], _coerce_timeout)

    query_multiplier = haskey(run_cfg, "query_multiplier") ?
        _positive_int(run_cfg["query_multiplier"], "run.query_multiplier") : 1000
    blas_threads = haskey(run_cfg, "blas_threads") ?
        _positive_int(run_cfg["blas_threads"], "run.blas_threads") :
        DEFAULT_BLAS_THREADS_COMPARATIVE
    resume = haskey(run_cfg, "resume") ?
             _normalize_bool(run_cfg["resume"], "run.resume") : true

    # The :nn arm's execution device. Every other arm is CPU-only and the
    # orchestrator pins it to "cpu" regardless. It is part of the :nn settings
    # hash, so flipping it here is a different fragment, never a silent resume
    # on top of the other device's fit.
    device = ComparatorRegistry.normalize_device(
        haskey(run_cfg, "device") ? string(run_cfg["device"]) : "cpu")

    solver_cfg = get(cfg, "solver", Dict{String,Any}())
    _check_removed_solver_keys(solver_cfg)
    _check_keys(solver_cfg, _VALID_SOLVER_KEYS, "solver")

    L_H_raw = String(lowercase(strip(string(get(solver_cfg, "L_H", "estimate")))))
    L_H_raw == "estimate" ||
        error("solver.L_H must be \"estimate\", got '$(L_H_raw)'. PROTOCOL: a fixed " *
              "L_H is an a priori Lipschitz constant; every arm must estimate " *
              "its own constants from the training data alone.")
    L_H = :estimate

    raw_factor = get(solver_cfg, "L_H_factor", _SMOKE_L_H_FACTOR_DEFAULT)
    (raw_factor isa Real && !(raw_factor isa Bool)) ||
        error("solver.L_H_factor must be a positive number, got $raw_factor")
    L_H_factor = Float64(raw_factor)
    L_H_factor > 0.0 || error("solver.L_H_factor must be > 0, got $L_H_factor")

    # :nsm carries the SAME safety multiplier as :diona by default, so that the
    # two bound-based arms are never compared at different settings by accident.
    raw_gamma = get(solver_cfg, "gamma_factor", L_H_factor)
    (raw_gamma isa Real && !(raw_gamma isa Bool)) ||
        error("solver.gamma_factor must be a positive number, got $raw_gamma")
    gamma_factor = Float64(raw_gamma)
    gamma_factor > 0.0 || error("solver.gamma_factor must be > 0, got $gamma_factor")

    return (name=name, arms=arms, selected_ids=selected_ids,
            arm_cells=arm_cells, dropped_arm_cells=dropped,
            out=out, run_id=run_id, jobs=jobs, timeouts=timeouts,
            blas_threads=blas_threads, resume=resume, device=device,
            query_multiplier=query_multiplier,
            L_H=L_H, L_H_factor=L_H_factor, gamma_factor=gamma_factor)
end

_coerce_jobs(v, key) = _positive_int(v, key)

function _coerce_timeout(v, key)
    v isa Real && !(v isa Bool) || error("$key must be > 0, got $v")
    t = Float64(v)
    t > 0 || error("$key must be > 0, got $t")
    return t
end

"""
    _per_arm_map(raw, arms, key, default_fn, coerce) -> Dict{Symbol,T}

`run.jobs` and `run.timeout_s` accept a scalar (broadcast to every selected arm)
or a per-arm map. The map form is a pass-through: `jobs_override` and
`timeout_override` on `run_comparative_orchestrator` are already
`Dict{Symbol,_}`-shaped.
"""
function _per_arm_map(raw, arms::Vector{Symbol}, key::String, default_fn, coerce)
    if raw === nothing
        return Dict(arm => coerce(default_fn(arm), key) for arm in arms)
    end
    if raw isa AbstractDict
        out = Dict(arm => coerce(default_fn(arm), key) for arm in arms)
        for (k, v) in raw
            arm = Symbol(lowercase(string(k)))
            arm in _DATA_ARM_ORDER ||
                error("$key map has unknown arm key '$(k)'")
            out[arm] = coerce(v, "$key.$(arm)")
        end
        return Dict(arm => out[arm] for arm in arms)
    end
    value = coerce(raw, key)
    return Dict(arm => value for arm in arms)
end

# ============================================================================
# Run
# ============================================================================

function describe_resolved(io::IO, resolved)
    println(io, "config      : $(resolved.name)")
    println(io, "out_dir     : $(resolved.out)   (run_id=$(resolved.run_id))")
    println(io, "cells       : $(length(resolved.selected_ids)) -> " *
                join(resolved.selected_ids, ","))
    println(io, "arms        : " * join(string.(resolved.arms), ", "))
    println(io, "arm-cells   : $(length(resolved.arm_cells))")
    println(io, "jobs        : " *
                join(("$arm=$(resolved.jobs[arm])" for arm in resolved.arms), "  "))
    println(io, "timeouts(s) : " *
                join(("$arm=$(resolved.timeouts[arm])" for arm in resolved.arms), "  "))
    println(io, "blas_threads: $(resolved.blas_threads)   resume = $(resolved.resume)")
    println(io, "query points: $(resolved.query_multiplier) x dim per function")
    println(io, "gamma_factor: $(resolved.gamma_factor)   (:nsm; :diona uses L_H_factor)")
    println(io, "device      : $(resolved.device)   (:nn only; every other arm is CPU-only)")
    println(io, "solver      : L_H=$(resolved.L_H) L_H_factor=$(resolved.L_H_factor) " *
                "mu0=$(_SMOKE_DIONA_MU0) layers=$(_SMOKE_DIONA_LAYERS)   (:diona only; ignored by other arms)")
    if !isempty(resolved.dropped_arm_cells)
        println(io, "NOT APPLICABLE (skipped, not failed): " *
            join(("cell $(id)/:$(arm)" for (id, arm) in resolved.dropped_arm_cells), ", "))
    end
    return nothing
end

"""
    write_resolved_config_record(resolved) -> path

Write this config's resolution into `run.out` as `config_<name>.txt`.

Without it the only durable statement of an `applicable_arms` exclusion is the
per-problem markdown, which the campaign path does not write -- so a reader of
the output directory would find cells 35 and 36 carrying five arms instead of
six with no reason recorded anywhere, and `comparative_outcomes.csv` would
simply have no row for the skipped pair. The filename is per config, so
concurrent shards sharing one `run.out` each write their own record.
"""
function write_resolved_config_record(resolved)
    mkpath(resolved.out)
    path = joinpath(resolved.out, "config_$(resolved.name).txt")
    open(path, "w") do io
        println(io, "# Resolved benchmark config, written by test/benchmark_data.jl.")
        println(io, "# One record per config; shards sharing this out dir each write their own.")
        println(io, "written_at  : $(Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"))")
        describe_resolved(io, resolved)
        println(io)
        println(io, "arm-cells claimed by this config:")
        for (id, arm) in resolved.arm_cells
            println(io, "  cell $(id)/:$(arm)")
        end
        if isempty(resolved.dropped_arm_cells)
            println(io, "\nno arm-cell was dropped as NOT APPLICABLE.")
        else
            println(io, "\nNOT APPLICABLE (skipped by applicable_arms, never attempted, " *
                        "never failed):")
            for (id, arm) in resolved.dropped_arm_cells
                println(io, "  cell $(id)/:$(arm)")
            end
        end
    end
    return path
end

"""
    run_from_config(path) -> campaign

Forward a resolved config into `run_comparative_orchestrator`'s existing
override surface. `resume`, `blas_threads` and `run_id` have no keyword
override there -- they are read from the process option table -- so they are
injected into `_SMOKE_OPTS` rather than by editing smoke_data.jl.
"""
function run_from_config(path::AbstractString)
    _, resolved = load_data_config(path)
    set_query_multiplier!(resolved.query_multiplier)   # before any fixture is built or read
    set_lh_factor!(resolved.L_H_factor)                # before any settings hash is computed
    NSMTrainer.set_gamma_factor!(resolved.gamma_factor)
    println("benchmark_data.jl: campaign")
    describe_resolved(stdout, resolved)
    println("config record: $(write_resolved_config_record(resolved))")
    println()
    flush(stdout)

    _SMOKE_OPTS["resume"]       = resolved.resume ? "true" : "false"
    _SMOKE_OPTS["blas-threads"] = string(resolved.blas_threads)
    _SMOKE_OPTS["run-id"]       = resolved.run_id
    _SMOKE_OPTS["query-multiplier"] = string(resolved.query_multiplier)

    return run_comparative_orchestrator(:full;
        selected_ids_override = resolved.selected_ids,
        out_dir_override      = resolved.out,
        arms_override         = resolved.arms,
        jobs_override         = resolved.jobs,
        timeout_override      = resolved.timeouts,
        device_override       = resolved.device)
end

"""
    _write_provenance_table(out, provenance_by_arm)

Write `<out>/comparative_provenance.csv`: one row per arm with the code
provenance its fragments recorded and the provenance the current harness would
compute. The two differ for any arm whose fragments were written by an older
harness; the report says so rather than refusing.
"""
function _write_provenance_table(out::AbstractString, provenance_by_arm::Dict{Symbol,Set{String}})
    path = joinpath(out, "comparative_provenance.csv")
    open(path, "w") do io
        println(io, "arm,recorded_code_provenance,current_code_provenance,same")
        for arm in sort(collect(keys(provenance_by_arm)); by=string)
            recorded = only(provenance_by_arm[arm])
            current = comparative_code_provenance(arm)
            println(io, join((arm, recorded, current, recorded == current), ","))
            recorded == current ||
                println("provenance: :$arm fragments were written by harness/arm code " *
                        "$(recorded[1:12])…, current is $(current[1:12])…")
        end
    end
    println("provenance table    : $path")
    return path
end

"""
    report_from_config(path) -> NamedTuple

Reporting/audit pass over a SHARED out_dir: re-read the terminal fragments the
shards already wrote and hand them to the existing artifact writers.

READ-ONLY with respect to the fixtures. It passes `ensure_fixtures=false`, so
it neither re-runs `prepare_smoke_cell` (a benchmark-function evaluation at
every training point plus `estimate_noise_bound`, on both 125,000-point cells
among others) nor rewrites the fixture files that a concurrently running
shard's workers are keyed to. It fits nothing and starts no worker; missing
(cell, arm) pairs are listed, not filled in. A cell with no fixture on disk is
an error naming `--prepare-fixtures`.
"""
function report_from_config(path::AbstractString)
    _, resolved = load_data_config(path)
    set_query_multiplier!(resolved.query_multiplier)   # before any fixture is built or read
    set_lh_factor!(resolved.L_H_factor)                # before any settings hash is computed
    NSMTrainer.set_gamma_factor!(resolved.gamma_factor)
    println("benchmark_data.jl: report (no fitting, no workers, no fixture writes)")
    describe_resolved(stdout, resolved)
    println()
    flush(stdout)

    cells = enumerate_cells()
    reg = ComparatorRegistry.canonical_registry()
    # The device must match the campaign's, or every :nn fragment is looked up
    # under the other device's settings hash and reported as missing.
    expectations, _ = _comparative_prepare_expectations(
        resolved.selected_ids, cells, resolved.out, reg;
        device=resolved.device, arms_override=resolved.arms, ensure_fixtures=false)

    results    = ArmCellResult[]
    row_groups = Vector{ArmQueryRow}[]
    missing_keys = Tuple{Int,Symbol}[]
    provenance_by_arm = Dict{Symbol,Set{String}}()
    # The arm set this report CLAIMS, per cell, taken from the config rather
    # than from whatever happened to be on disk. Handing the reporting layer
    # "whatever turned up" would let a silently missing shard pass as a
    # complete report; narrow `arms:`/`cells:` in the config to report a
    # subset deliberately instead.
    expected_arms = Dict{Int,Set{Symbol}}()
    for expected in expectations
        push!(get!(expected_arms, expected.cell_id, Set{Symbol}()), expected.arm)
        # The report reads a finished campaign AS IT WAS RUN: each fragment's
        # recorded code provenance is accepted and tabulated below, instead of
        # being required to equal the hash of the harness as it is today. The
        # fixture hash (identical data) and settings hash (identical method
        # settings) are still enforced per fragment.
        check = _comparative_try_read(expected; allow_failure=true, provenance=:recorded)
        if check.valid
            push!(results, check.payload.result)
            push!(row_groups, check.payload.rows)
            push!(get!(provenance_by_arm, expected.arm, Set{String}()),
                  check.payload.code_provenance)
        else
            push!(missing_keys, (expected.cell_id, expected.arm))
        end
    end

    println("readable arm-cell fragments: $(length(results)) / $(length(expectations))")
    # One code provenance per arm, or the arm's fragments came from different
    # code and the report would silently mix them.
    mixed = [arm for (arm, hs) in provenance_by_arm if length(hs) > 1]
    isempty(mixed) ||
        error("report: fragments of arm(s) $(join(string.(mixed), ", ")) carry more than one " *
              "code provenance under $(resolved.out); a report must not mix code versions " *
              "within an arm. Re-run that arm's shards on one code version.")
    _write_provenance_table(resolved.out, provenance_by_arm)
    isempty(results) &&
        error("report: no readable fragments under $(resolved.out) for this selection")
    # A :nn fragment is keyed by a settings hash that INCLUDES the device, so a
    # report whose `run.device` differs from the campaign's looks for the other
    # device's fragments and finds none. That is the intended failure -- it is
    # loud, and it never mixes the two devices into one table -- but it reads as
    # "the shard never ran", so name the likelier cause.
    device_hint = any(arm === :nn for (_, arm) in missing_keys) ?
        " Every missing pair here is :nn, and this report resolved " *
        "run.device=$(resolved.device): a :nn fragment fitted on the other device " *
        "carries a different settings hash and will not be found. Check that this " *
        "config's run.device matches the one the :nn shard ran with." : ""
    isempty(missing_keys) ||
        error("report: $(length(missing_keys)) arm-cell fragment(s) claimed by this " *
              "config are missing or unreadable under $(resolved.out): " *
              join(("cell $(id)/:$(arm)" for (id, arm) in missing_keys), ", ") *
              ". Run the shard that owns them, or narrow 'arms'/'cells' to report " *
              "the subset that has finished." * device_hint)

    # The canonical `comparative_*` names. Arm shards write their campaign files
    # under `comparative_shard-<arms>_*` instead, so this pass is the only
    # writer of the authoritative aggregate and cannot be clobbered by one.
    artifacts = write_comparative_artifacts(resolved.out, results, row_groups;
                                            prefix="comparative",
                                            expected_arms=expected_arms)
    problem = write_problem_aggregate_artifacts(resolved.out, results, row_groups;
                                                expected_arms=expected_arms)
    println("comparative summary : $(artifacts.summary_path)")
    println("problem summary     : $(problem.csv_path)")
    println("problem markdown    : $(problem.markdown_path)")
    return (out_dir=resolved.out, results=results, row_groups=row_groups,
            artifacts=artifacts, problem_artifacts=problem,
            missing_keys=missing_keys, resolved=resolved)
end

"""
    prepare_fixtures_from_config(path) -> NamedTuple

Materialise (or verify) the fixture of every cell this config selects, and stop.

WHY THIS EXISTS. `fixture_sha256` is not content-addressed: `generated_at`, a
wall-clock string, is folded into the hashed fixture metadata
(test/ml/common/fixture.jl). `ensure_comparative_fixture` does
prepare -> isfile? -> materialize, so two processes that BOTH find a fixture
absent each write their own and key their fragments to different `fx...` stems.
The consequences are asymmetric: `--report` then fails loudly naming the pair,
but re-running the orphaned shard with `resume: true` silently REFITS it and
leaves the old fragment behind -- on a :diona cell that is hours of wasted wall
time that looks like a normal first run.

The window is one fixture write per cell, so it only opens on a cold start.
Running this once over the union of cells before launching shards concurrently
closes it: `_comparative_fixture_matches` compares data and identity but NOT
`generated_at`, so every later process reuses what is on disk. Making the hash
content-addressed would remove the race outright, but it rotates every fixture
hash and therefore every fragment stem, which invalidates a campaign.
"""
function prepare_fixtures_from_config(path::AbstractString)
    _, resolved = load_data_config(path)
    set_query_multiplier!(resolved.query_multiplier)   # before any fixture is built or read
    set_lh_factor!(resolved.L_H_factor)                # before any settings hash is computed
    NSMTrainer.set_gamma_factor!(resolved.gamma_factor)
    println("benchmark_data.jl: prepare fixtures (no arms, no fitting, no workers)")
    describe_resolved(stdout, resolved)
    println()
    flush(stdout)

    cells = enumerate_cells()
    mkpath(resolved.out)
    reused = Int[]
    written = Int[]
    for id in resolved.selected_ids
        ensured = ensure_comparative_fixture(cells[id], id, resolved.out)
        push!(ensured.reused ? reused : written, id)
        println("  cell $(id): $(ensured.reused ? "reused" : "written") " *
                "sha=$(ensured.fx.fixture_sha256[1:12]) $(ensured.path)")
        flush(stdout)
    end
    println("\nfixtures reused: $(length(reused))   written: $(length(written))   " *
            "total: $(length(resolved.selected_ids))")
    println("Shards launched against $(resolved.out) after this point cannot race on " *
            "fixture generation.")
    return (out_dir=resolved.out, reused=reused, written=written, resolved=resolved)
end

function benchmark_data_main(args::Vector{String})
    report  = any(a -> a in ("--report", "--report=true"), args)
    prepare = any(a -> a in ("--prepare-fixtures", "--prepare-fixtures=true"), args)
    report && prepare &&
        error("--report and --prepare-fixtures are mutually exclusive: --report is " *
              "read-only and deliberately does not generate fixtures.")
    positionals = [a for a in args if !startswith(a, "--")]
    length(positionals) == 1 ||
        error("usage: julia --project=test/ml test/benchmark_data.jl " *
              "[--report | --prepare-fixtures] <config.yaml>")
    prepare && return prepare_fixtures_from_config(positionals[1])
    return report ? report_from_config(positionals[1]) : run_from_config(positionals[1])
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    outcome = benchmark_data_main(copy(ARGS))
    if outcome isa NamedTuple && haskey(outcome, :success)
        outcome.success ||
            error("campaign completed with failures; successful artifacts were retained")
    end
end
