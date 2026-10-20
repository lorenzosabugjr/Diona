# Reporting layer: interval semantics, common-success intersection, artifacts.
#
# Extracted verbatim from test/smoke_data.jl. This is a plain `include`d FRAGMENT,
# not a module: every name must stay top-level in the including scope, because
# test/run_full_benchmark.jl reaches into the harness via getproperty(harness, :name)
# and test/benchmark_data.jl mutates _SMOKE_OPTS directly. The include sits at the
# exact position the code occupied, so evaluation order is unchanged.
#
# NOTE: SCRIPT_PATH stays in smoke_data.jl -- it is abspath(@__FILE__) and is the
# target workers re-invoke, so it must resolve to the entry point, not to this file.

# REPORTING_LAYER_BEGIN

"""
    comparative_interval_kind(arm) -> Symbol

What an arm's `lower`/`upper` ARE. Every arm estimates its own constants from
the training data, so no arm's interval certifies anything and none of these
symbols says it does.

  `:diona` `:admissible_set`            bounds over the Taylor-admissible set at
                                        ESTIMATED noise and smoothness constants
  `:gp`    `:variational_approximate`   variational posterior credible interval
  `:nsm`   `:set_membership_estimated`  set-membership cone band at an ESTIMATED
                                        eps/gamma (NSMTrainer.nsm_interval_kind)
  `:nn`/`:rbf`/`:idw` `:none`           no interval is produced -- an absence,
                                        not a failure (SurrogateArms.surrogate_interval_kind)
"""
function comparative_interval_kind(arm::Symbol)::Symbol
    arm == :diona && return :admissible_set
    arm == :gp    && return :variational_approximate
    arm == :nsm   && return :set_membership_estimated
    arm in (:nn, :rbf, :idw) && return :none
    error("comparative_interval_kind: unrecognized arm=$arm")
end

struct CommonSuccessResult
    cell_id::Int
    name::Symbol
    layout::Symbol
    tier_label::String
    dim::Int
    arm::Symbol
    model_type::String
    interval_kind::Symbol
    n_common::Int
    n_requested::Int
    n_finite_interval::Int
    rel_l2::Float64
    mae::Float64
    mse::Float64
    coverage::Float64
    mean_width::Float64
end

_comparative_arm_rank(arm::Symbol) =
    arm == :diona ? 1 : arm == :nn ? 2 : arm == :gp ? 3 :
    arm == :nsm   ? 4 : arm == :rbf ? 5 : arm == :idw ? 6 : 99

const _COMPARATIVE_CELL_ARMS_CACHE = Ref{Union{Nothing,Dict{Int,Set{Symbol}}}}(nothing)

function _comparative_default_expected_arms()
    cached = _COMPARATIVE_CELL_ARMS_CACHE[]
    cached === nothing || return cached
    built = Dict{Int,Set{Symbol}}(id => Set(applicable_arms(cell))
                                  for (id, cell) in enumerate(enumerate_cells()))
    _COMPARATIVE_CELL_ARMS_CACHE[] = built
    return built
end

"""
    _comparative_expected_arms(cell_id, spec) -> Set{Symbol}

The arm set the reporting layer REQUIRES `cell_id` to carry, so that a silently
missing arm cannot pass as a complete report.

`spec === nothing` (the default) means `applicable_arms` of the enumerated cell
with that id -- the honest expectation for a whole-benchmark run, and per-cell,
so a cell where an arm is legitimately excluded is not held to the others'
arm set. Anything else is an explicit expectation, either one collection of arms
for every cell or a `Dict` of `cell_id => arms`; that is what lets a PARTIAL
campaign (an arm shard, or synthetic gate cells with ids outside the benchmark)
still be reported without weakening the check into "whatever turned up".

A partial report's `n_common` is an intersection over the arms IN THAT REPORT
only, and is not comparable with a full run's.
"""
_comparative_expected_arms(cell_id::Int, ::Nothing) = begin
    defaults = _comparative_default_expected_arms()
    haskey(defaults, cell_id) || error(
        "comparative reporting: cell_id=$cell_id is not an enumerated benchmark cell, so " *
        "there is no default expected-arm set for it. Pass expected_arms (a collection of " *
        "arms, or a Dict of cell_id => arms) to report a synthetic or partial campaign.")
    defaults[cell_id]
end

_comparative_expected_arms(cell_id::Int, spec::AbstractDict) = begin
    haskey(spec, cell_id) || error(
        "comparative reporting: expected_arms carries no entry for cell_id=$cell_id")
    Set(Symbol.(collect(spec[cell_id])))
end

_comparative_expected_arms(::Int, spec) = Set(Symbol.(collect(spec)))

function _validated_comparative_index(
    results::AbstractVector{ArmCellResult},
    row_groups::AbstractVector{<:AbstractVector{ArmQueryRow}};
    expected_arms=nothing,
)
    result_by_key = Dict{Tuple{Int,Symbol},ArmCellResult}()
    for result in results
        key = (result.cell_id, result.arm)
        haskey(result_by_key, key) &&
            error("comparative reporting: duplicate summary key $key")
        result_by_key[key] = result
    end

    rows_by_key = Dict{Tuple{Int,Symbol},Vector{ArmQueryRow}}()
    point_keys = Set{Tuple{Int,Int,Symbol}}()
    for group_rows in row_groups
        isempty(group_rows) && error("comparative reporting: an arm row group is empty")
        key = (first(group_rows).cell_id, first(group_rows).arm)
        haskey(rows_by_key, key) &&
            error("comparative reporting: duplicate row group for key $key")
        copied_rows = Vector{ArmQueryRow}(group_rows)
        for row in copied_rows
            (row.cell_id, row.arm) == key ||
                error("comparative reporting: mixed cell/arm values in row group $key")
            point_key = (row.cell_id, row.query_id, row.arm)
            point_key in point_keys &&
                error("comparative reporting: duplicate point key $point_key")
            push!(point_keys, point_key)
        end
        rows_by_key[key] = copied_rows
    end

    Set(keys(result_by_key)) == Set(keys(rows_by_key)) ||
        error("comparative reporting: summary keys and point-row-group keys differ")

    cell_ids = sort(unique(first(key) for key in keys(result_by_key)))
    query_ids_by_cell = Dict{Int,Vector{Int}}()
    for cell_id in cell_ids
        cell_results = [result for result in values(result_by_key) if result.cell_id == cell_id]
        ref_result = first(cell_results)
        ref_meta = (ref_result.name, ref_result.layout,
                    ref_result.tier_label, ref_result.dim)
        all((result.name, result.layout, result.tier_label, result.dim) == ref_meta
            for result in cell_results) ||
            error("comparative reporting: inconsistent cell metadata for cell_id=$cell_id")

        arms = Set(result.arm for result in cell_results)
        expected = _comparative_expected_arms(cell_id, expected_arms)
        arms == expected ||
            error("comparative reporting: cell_id=$cell_id has arms=$(sort!(collect(arms))) " *
                  "but expected $(sort!(collect(expected)))")

        ref_arm = sort!(collect(arms); by=_comparative_arm_rank)[1]
        ref_rows = rows_by_key[(cell_id, ref_arm)]
        ref_by_id = Dict(row.query_id => row for row in ref_rows)
        length(ref_by_id) == length(ref_rows) ||
            error("comparative reporting: duplicate query IDs for cell_id=$cell_id arm=$ref_arm")
        ref_ids = sort!(collect(keys(ref_by_id)))
        query_ids_by_cell[cell_id] = ref_ids

        for result in cell_results
            rows = rows_by_key[(cell_id, result.arm)]
            length(rows) == result.n_query ||
                error("comparative reporting: row count for $((cell_id, result.arm)) does not match n_query")
            count(row -> isfinite(row.pred), rows) == result.n_finite_pred ||
                error("comparative reporting: finite-prediction count for $((cell_id, result.arm)) is inconsistent")
            all(length(row.x) == result.dim for row in rows) ||
                error("comparative reporting: coordinate dimension mismatch for $((cell_id, result.arm))")

            rows_for_arm = Dict(row.query_id => row for row in rows)
            length(rows_for_arm) == length(rows) ||
                error("comparative reporting: duplicate query IDs for $((cell_id, result.arm))")
            sort!(collect(keys(rows_for_arm))) == ref_ids ||
                error("comparative reporting: query-ID coverage differs for $((cell_id, result.arm))")
            for query_id in ref_ids
                ref_row = ref_by_id[query_id]
                row = rows_for_arm[query_id]
                isequal(row.x, ref_row.x) ||
                    error("comparative reporting: query coordinates differ at cell=$cell_id query=$query_id")
                isequal(row.truth, ref_row.truth) ||
                    error("comparative reporting: query truth differs at cell=$cell_id query=$query_id")
            end
        end
    end
    return result_by_key, rows_by_key, query_ids_by_cell
end

function common_success_query_ids(
    results::AbstractVector{ArmCellResult},
    row_groups::AbstractVector{<:AbstractVector{ArmQueryRow}};
    expected_arms=nothing,
)
    result_by_key, rows_by_key, query_ids_by_cell =
        _validated_comparative_index(results, row_groups; expected_arms=expected_arms)
    common_by_cell = Dict{Int,Vector{Int}}()
    for (cell_id, requested_ids) in query_ids_by_cell
        common = Set(requested_ids)
        arms = [arm for (cid, arm) in keys(result_by_key) if cid == cell_id]
        for arm in arms
            finite_ids = Set(row.query_id for row in rows_by_key[(cell_id, arm)] if isfinite(row.pred))
            intersect!(common, finite_ids)
        end
        common_by_cell[cell_id] = sort!(collect(common))
    end
    return common_by_cell
end

function common_success_results(
    results::AbstractVector{ArmCellResult},
    row_groups::AbstractVector{<:AbstractVector{ArmQueryRow}};
    expected_arms=nothing,
)::Vector{CommonSuccessResult}
    result_by_key, rows_by_key, query_ids_by_cell =
        _validated_comparative_index(results, row_groups; expected_arms=expected_arms)
    common_by_cell = common_success_query_ids(results, row_groups; expected_arms=expected_arms)
    output = CommonSuccessResult[]
    for key in sort!(collect(keys(result_by_key));
                     by=key -> (first(key), _comparative_arm_rank(last(key))))
        result = result_by_key[key]
        common_ids = Set(common_by_cell[result.cell_id])
        common_rows = [row for row in rows_by_key[key] if row.query_id in common_ids]
        mean_width, coverage, n_interval = _arm_interval_stats(common_rows)
        push!(output, CommonSuccessResult(
            result.cell_id, result.name, result.layout,
            result.tier_label, result.dim, result.arm, result.model_type,
            comparative_interval_kind(result.arm), length(common_rows),
            length(query_ids_by_cell[result.cell_id]), n_interval,
            _arm_rel_l2(common_rows), _arm_mae(common_rows), _arm_mse(common_rows),
            coverage, mean_width,
        ))
    end
    return output
end

function comparative_point_rows(
    results::AbstractVector{ArmCellResult},
    row_groups::AbstractVector{<:AbstractVector{ArmQueryRow}};
    expected_arms=nothing,
)
    result_by_key, rows_by_key, _ =
        _validated_comparative_index(results, row_groups; expected_arms=expected_arms)
    output = NamedTuple[]
    for key in sort!(collect(keys(result_by_key));
                     by=key -> (first(key), _comparative_arm_rank(last(key))))
        result = result_by_key[key]
        kind = comparative_interval_kind(result.arm)
        for row in sort(rows_by_key[key]; by=row -> row.query_id)
            has_interval = isfinite(row.lower) && isfinite(row.upper)
            push!(output, (
                cell_id=result.cell_id, query_id=row.query_id, arm=result.arm,
                name=result.name, layout=result.layout,
                tier_label=result.tier_label, dim=result.dim,
                model_type=result.model_type, interval_kind=kind,
                fixture_sha256=result.fixture_sha256, settings_hash=result.settings_hash,
                x=copy(row.x), truth=row.truth, pred=row.pred,
                lower=row.lower, upper=row.upper, interval_available=has_interval,
                status=row.status, elapsed_s=row.elapsed_s,
                smoothness_used=row.smoothness_used,
            ))
        end
    end
    return output
end

function comparative_summary_rows(
    results::AbstractVector{ArmCellResult},
    row_groups::AbstractVector{<:AbstractVector{ArmQueryRow}};
    expected_arms=nothing,
)
    _validated_comparative_index(results, row_groups; expected_arms=expected_arms)
    common = common_success_results(results, row_groups; expected_arms=expected_arms)
    common_by_key = Dict((row.cell_id, row.arm) => row for row in common)
    output = NamedTuple[]
    for result in sort(collect(results);
                       by=result -> (result.cell_id, _comparative_arm_rank(result.arm)))
        common_row = common_by_key[(result.cell_id, result.arm)]
        # The arm set `n_common` was intersected over, RECORDED IN THE ROW.
        # Without it `n_common` silently changes meaning between an arm-shard
        # summary and a whole-campaign one, and between cells 35/36 (five arms,
        # :rbf excluded) and every other cell (six) -- and a reader opening the
        # CSV by name has no way to tell which they are holding.
        cell_arms = sort!(collect(_comparative_expected_arms(result.cell_id, expected_arms)))
        push!(output, (
            cell_id=result.cell_id, name=result.name,
            layout=result.layout, tier_label=result.tier_label, dim=result.dim,
            arm=result.arm, model_type=result.model_type,
            interval_kind=comparative_interval_kind(result.arm),
            fixture_sha256=result.fixture_sha256, settings_hash=result.settings_hash,
            model_seed=result.model_seed, noise_seed=result.noise_seed,
            n_train=result.n_train, n_query=result.n_query,
            fit_time_s=result.fit_time_s,
            inference_time_total_s=result.inference_time_total_s,
            inference_time_mean_s=result.inference_time_mean_s,
            final_loss=result.final_loss,
            n_finite_pred=result.n_finite_pred,
            n_finite_interval=result.n_finite_interval,
            rel_l2=result.rel_l2, mae=result.mae, mse=result.mse,
            coverage=result.coverage, mean_width=result.mean_width,
            status_counts=_status_counts_str(result.status_counts),
            n_common=common_row.n_common, n_requested=common_row.n_requested,
            common_arms=join(string.(cell_arms), '|'),
            n_common_arms=length(cell_arms),
            common_n_finite_interval=common_row.n_finite_interval,
            common_rel_l2=common_row.rel_l2, common_mae=common_row.mae,
            common_mse=common_row.mse, common_coverage=common_row.coverage,
            common_mean_width=common_row.mean_width,
            report_runtime_julia_version=string(VERSION),
            report_runtime_machine=string(Sys.MACHINE),
            report_runtime_kernel=string(Sys.KERNEL),
            report_runtime_threads=Threads.nthreads(),
        ))
    end
    return output
end

function _comparative_csv_field(value)
    text = string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text) || occursin('\r', text)
        return "\"" * replace(text, '"' => "\"\"") * "\""
    end
    return text
end

function _write_comparative_summary_csv(path::AbstractString, rows)
    header = isempty(rows) ? Symbol[] : collect(propertynames(first(rows)))
    atomic_write(String(path)) do io
        println(io, join(string.(header), ','))
        for row in rows
            println(io, join((_comparative_csv_field(getproperty(row, field)) for field in header), ','))
        end
    end
    return String(path)
end

function write_comparative_artifacts(
    out_dir::AbstractString,
    results::AbstractVector{ArmCellResult},
    row_groups::AbstractVector{<:AbstractVector{ArmQueryRow}};
    prefix::AbstractString="comparative",
    expected_arms=nothing,
)
    point_rows = comparative_point_rows(results, row_groups; expected_arms=expected_arms)
    summary_rows = comparative_summary_rows(results, row_groups; expected_arms=expected_arms)
    common_rows = common_success_results(results, row_groups; expected_arms=expected_arms)
    summary_path = joinpath(out_dir, "$(prefix)_summary.csv")
    _write_comparative_summary_csv(summary_path, summary_rows)
    return (points_dir=joinpath(out_dir, "comparative_fragments", "runs"),
            summary_path=summary_path,
            point_rows=point_rows, summary_rows=summary_rows,
            common_results=common_rows)
end

# REPORTING_LAYER_END
